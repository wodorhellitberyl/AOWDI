#!/usr/bin/env python3
"""Validate A-OWDI against mapped washover polygons.

The validation follows the manuscript logic:
- positive pixels are rasterised from independent or development reference polygons;
- candidate background is drawn from the valid A-OWDI analysis domain outside the
  positive mask and a configurable exclusion zone (default: 2 pixels);
- matching uses PRE-event surface state only: NDVI_pre and BSI_pre;
- NDVI_pre and BSI_pre are standardised with the candidate-background mean and
  population standard deviation;
- nearest-neighbour matching is performed with replacement;
- ROC-AUC and PR-AUC are reported for A-OWDI, z(dBSI), and z(dNDVI);
- metrics are calculated for matched background and for the full eligible
  background domain after applying the fixed exclusion zone;
- feature-level A-OWDI summaries and their scene-percentile positions are exported.

All input rasters must be co-registered and should come from aowdi_calculate.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, auc
from sklearn.neighbors import NearestNeighbors


FLOAT_NODATA = -9999.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate A-OWDI against mapped washover polygons.")
    p.add_argument("--aowdi", required=True, type=Path)
    p.add_argument("--z-dbsi", required=True, type=Path)
    p.add_argument("--z-dndvi", required=True, type=Path)
    p.add_argument("--ndvi-pre", required=True, type=Path)
    p.add_argument("--bsi-pre", required=True, type=Path)
    p.add_argument("--analysis-mask", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path,
                   help="Reference polygons (GeoPackage, GeoJSON, Shapefile, etc.).")
    p.add_argument("--layer", default=None,
                   help="Layer name for multi-layer vector files such as GeoPackage.")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--exclusion-pixels", type=int, default=2,
                   help="Number of raster pixels excluded around mapped polygon margins (default: 2).")
    p.add_argument("--all-touched", action="store_true",
                   help="Rasterise every pixel touched by a polygon. Default uses pixel-centre inclusion.")
    p.add_argument("--id-field", default=None,
                   help="Optional polygon attribute used as feature identifier in feature-level output.")
    return p.parse_args()


def read_float(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float64")
        data = np.asarray(arr.filled(np.nan), dtype="float64")
        return data, src.profile.copy(), src.transform, src.crs, src.shape


def read_mask(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        mask = (~np.ma.getmaskarray(arr)) & (np.asarray(arr.filled(0)) > 0)
        return mask, src.transform, src.crs, src.shape


def assert_same_grid(reference, others):
    ref_transform, ref_crs, ref_shape = reference
    for name, transform, crs, shape in others:
        if shape != ref_shape or crs != ref_crs or tuple(transform) != tuple(ref_transform):
            raise RuntimeError(f"Raster grid mismatch for {name}.")


def standardise_background(x: np.ndarray, bg_mask: np.ndarray):
    vals = x[bg_mask & np.isfinite(x)]
    if vals.size < 2:
        raise RuntimeError("Insufficient candidate background pixels for matching.")
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise RuntimeError("Background standard deviation is zero or non-finite.")
    z = (x - mean) / std
    return z, mean, std


def metric_pair(scores_pos: np.ndarray, scores_neg: np.ndarray):
    y = np.concatenate([np.ones(scores_pos.size, dtype=np.uint8), np.zeros(scores_neg.size, dtype=np.uint8)])
    s = np.concatenate([scores_pos, scores_neg]).astype("float64")
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if np.unique(y).size != 2:
        raise RuntimeError("Both positive and negative classes are required for ROC/PR metrics.")
    roc = float(roc_auc_score(y, s))
    precision, recall, _ = precision_recall_curve(y, s)
    # PR-AUC by trapezoidal integration, matching the usual area-under-curve definition.
    pr = float(auc(recall, precision))
    ap = float(average_precision_score(y, s))
    baseline = float(y.mean())
    return roc, pr, ap, baseline, int((y == 1).sum()), int((y == 0).sum())


def scene_percentile(scene_values: np.ndarray, value: float) -> float:
    vals = scene_values[np.isfinite(scene_values)]
    if vals.size == 0 or not np.isfinite(value):
        return math.nan
    return 100.0 * float(np.mean(vals <= value))


def write_rows(path: Path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    aowdi, profile, transform, crs, shape = read_float(args.aowdi)
    z_dbsi, _, t2, c2, s2 = read_float(args.z_dbsi)
    z_dndvi, _, t3, c3, s3 = read_float(args.z_dndvi)
    ndvi_pre, _, t4, c4, s4 = read_float(args.ndvi_pre)
    bsi_pre, _, t5, c5, s5 = read_float(args.bsi_pre)
    analysis_mask, t6, c6, s6 = read_mask(args.analysis_mask)

    assert_same_grid((transform, crs, shape), [
        ("z_dBSI", t2, c2, s2),
        ("z_dNDVI", t3, c3, s3),
        ("NDVI_pre", t4, c4, s4),
        ("BSI_pre", t5, c5, s5),
        ("analysis_mask", t6, c6, s6),
    ])

    gdf = gpd.read_file(args.reference, layer=args.layer) if args.layer else gpd.read_file(args.reference)
    if gdf.empty:
        raise RuntimeError("Reference vector contains no features.")
    if gdf.crs is None:
        raise RuntimeError("Reference vector has no CRS.")
    if crs is None:
        raise RuntimeError("Raster has no CRS.")
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    if args.id_field and args.id_field not in gdf.columns:
        raise RuntimeError(f"ID field '{args.id_field}' not found in reference vector.")

    geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        raise RuntimeError("Reference vector contains no valid geometries.")

    # Rasterise the complete mapped reference geometry first.  The exclusion zone
    # must be based on the full polygon margins, not only on the subset of positive
    # pixels that happens to fall inside the analysis domain.  This reproduces the
    # manuscript validation workflow and avoids admitting background pixels next to
    # polygon segments crossing masked/non-analysis cells.
    reference_mask = rasterize(
        [(geom, 1) for geom in geoms],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=args.all_touched,
        dtype="uint8",
    ).astype(bool)

    positive = reference_mask & analysis_mask

    # Construct the fixed exclusion zone in raster space. The reference polygons
    # are first rasterised using the same pixel-centre rule as the positive mask,
    # then dilated by the requested number of pixels using an 8-neighbour 3x3
    # structuring element. The frozen manuscript workflow uses two pixels.
    if args.exclusion_pixels > 0:
        exclusion = binary_dilation(
            reference_mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=args.exclusion_pixels,
        )
    else:
        exclusion = reference_mask.copy()

    common_finite = (
        np.isfinite(aowdi) & np.isfinite(z_dbsi) & np.isfinite(z_dndvi)
        & np.isfinite(ndvi_pre) & np.isfinite(bsi_pre)
    )
    positive &= common_finite
    candidate_background = analysis_mask & common_finite & ~exclusion
    # "Full background" in the manuscript denotes the complete eligible background
    # domain after applying the fixed exclusion zone, before nearest-neighbour matching.
    full_background = candidate_background.copy()

    n_pos = int(positive.sum())
    n_candidate = int(candidate_background.sum())
    if n_pos == 0:
        raise RuntimeError("No reference-positive raster pixels overlap the valid analysis domain.")
    if n_candidate == 0:
        raise RuntimeError("No candidate background pixels remain after applying the exclusion zone.")

    z_ndvi_pre, ndvi_bg_mean, ndvi_bg_std = standardise_background(ndvi_pre, candidate_background)
    z_bsi_pre, bsi_bg_mean, bsi_bg_std = standardise_background(bsi_pre, candidate_background)

    bg_rc = np.argwhere(candidate_background)
    pos_rc = np.argwhere(positive)
    bg_features = np.column_stack([
        z_ndvi_pre[candidate_background],
        z_bsi_pre[candidate_background],
    ])
    pos_features = np.column_stack([
        z_ndvi_pre[positive],
        z_bsi_pre[positive],
    ])

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
    nn.fit(bg_features)
    distances, indices = nn.kneighbors(pos_features, return_distance=True)
    indices = indices[:, 0]
    distances = distances[:, 0]
    matched_rc = bg_rc[indices]
    matched_flat = np.ravel_multi_index((matched_rc[:, 0], matched_rc[:, 1]), shape)
    unique_matched = int(np.unique(matched_flat).size)

    # Extract matched negative samples with replacement.
    def take_matched(arr):
        return arr[matched_rc[:, 0], matched_rc[:, 1]]

    variables = {
        "AOWDI": aowdi,
        "z_dBSI": z_dbsi,
        "z_dNDVI": z_dndvi,
    }

    matched_rows = []
    full_rows = []
    for name, arr in variables.items():
        roc, pr, ap, baseline, np_, nn_ = metric_pair(arr[positive], take_matched(arr))
        matched_rows.append({
            "variable": name,
            "ROC_AUC": roc,
            "PR_AUC": pr,
            "average_precision": ap,
            "PR_baseline_prevalence": baseline,
            "n_positive_samples": np_,
            "n_negative_samples": nn_,
            "n_unique_matched_negative_pixels": unique_matched,
        })

        roc, pr, ap, baseline, np_, nn_ = metric_pair(arr[positive], arr[full_background])
        full_rows.append({
            "variable": name,
            "ROC_AUC": roc,
            "PR_AUC": pr,
            "average_precision": ap,
            "PR_baseline_prevalence": baseline,
            "n_positive_samples": np_,
            "n_negative_samples": nn_,
        })

    write_rows(
        args.outdir / "validation_metrics_matched.csv",
        ["variable", "ROC_AUC", "PR_AUC", "average_precision", "PR_baseline_prevalence",
         "n_positive_samples", "n_negative_samples", "n_unique_matched_negative_pixels"],
        matched_rows,
    )
    write_rows(
        args.outdir / "validation_metrics_full_background.csv",
        ["variable", "ROC_AUC", "PR_AUC", "average_precision", "PR_baseline_prevalence",
         "n_positive_samples", "n_negative_samples"],
        full_rows,
    )

    # Export the exact matched samples for auditability.
    match_rows = []
    for i, (prc, mrc, dist) in enumerate(zip(pos_rc, matched_rc, distances), start=1):
        match_rows.append({
            "pair_id": i,
            "positive_row": int(prc[0]),
            "positive_col": int(prc[1]),
            "matched_row": int(mrc[0]),
            "matched_col": int(mrc[1]),
            "matching_distance": float(dist),
            "positive_NDVI_pre": float(ndvi_pre[prc[0], prc[1]]),
            "positive_BSI_pre": float(bsi_pre[prc[0], prc[1]]),
            "matched_NDVI_pre": float(ndvi_pre[mrc[0], mrc[1]]),
            "matched_BSI_pre": float(bsi_pre[mrc[0], mrc[1]]),
        })
    write_rows(
        args.outdir / "matched_background_pairs.csv",
        ["pair_id", "positive_row", "positive_col", "matched_row", "matched_col", "matching_distance",
         "positive_NDVI_pre", "positive_BSI_pre", "matched_NDVI_pre", "matched_BSI_pre"],
        match_rows,
    )

    # Feature-level A-OWDI summaries.
    scene_vals = aowdi[analysis_mask & np.isfinite(aowdi)]
    feature_rows = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        fm = rasterize(
            [(geom, 1)], out_shape=shape, transform=transform, fill=0,
            all_touched=args.all_touched, dtype="uint8"
        ).astype(bool)
        fm &= analysis_mask & np.isfinite(aowdi)
        vals = aowdi[fm]
        if vals.size == 0:
            continue
        q50, q75, q90 = [float(x) for x in np.percentile(vals, [50, 75, 90])]
        fid = row[args.id_field] if args.id_field else idx
        feature_rows.append({
            "feature_id": fid,
            "n_pixels": int(vals.size),
            "AOWDI_mean": float(np.mean(vals)),
            "AOWDI_P50": q50,
            "AOWDI_P75": q75,
            "AOWDI_P90": q90,
            "scene_percentile_of_P50": scene_percentile(scene_vals, q50),
            "scene_percentile_of_P75": scene_percentile(scene_vals, q75),
            "scene_percentile_of_P90": scene_percentile(scene_vals, q90),
        })
    write_rows(
        args.outdir / "feature_level_AOWDI_summary.csv",
        ["feature_id", "n_pixels", "AOWDI_mean", "AOWDI_P50", "AOWDI_P75", "AOWDI_P90",
         "scene_percentile_of_P50", "scene_percentile_of_P75", "scene_percentile_of_P90"],
        feature_rows,
    )

    # Save masks so the validation domain can be inspected in GIS.
    mask_profile = profile.copy()
    mask_profile.update(dtype="uint8", count=1, nodata=255, compress="deflate", predictor=2)
    for fname, mask in (
        ("validation_positive_mask.tif", positive),
        ("validation_candidate_background_mask.tif", candidate_background),
        ("validation_full_background_mask.tif", full_background),
    ):
        with rasterio.open(args.outdir / fname, "w", **mask_profile) as dst:
            dst.write(mask.astype("uint8"), 1)

    summary = {
        "positive_pixels": n_pos,
        "candidate_background_pixels_after_exclusion": n_candidate,
        "matched_negative_samples_with_replacement": n_pos,
        "unique_matched_negative_pixels": unique_matched,
        "exclusion_pixels": args.exclusion_pixels,
        "exclusion_method": "raster dilation; 8-neighbour 3x3 structure",
        "matching_variables": ["NDVI_pre", "BSI_pre"],
        "matching_standardisation": {
            "NDVI_pre_background_mean": ndvi_bg_mean,
            "NDVI_pre_background_std_population": ndvi_bg_std,
            "BSI_pre_background_mean": bsi_bg_mean,
            "BSI_pre_background_std_population": bsi_bg_std,
        },
        "rasterization_all_touched": bool(args.all_touched),
        "reference_file": str(args.reference),
        "reference_layer": args.layer,
    }
    with open(args.outdir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Validation complete. Positive pixels: {n_pos:,}")
    print(f"Candidate background pixels: {n_candidate:,}")
    print(f"Matched negative samples: {n_pos:,} ({unique_matched:,} unique pixels)")
    for row in matched_rows:
        print(f"MATCHED {row['variable']}: ROC-AUC={row['ROC_AUC']:.4f}, PR-AUC={row['PR_AUC']:.4f}")
    for row in full_rows:
        print(f"FULL {row['variable']}: ROC-AUC={row['ROC_AUC']:.4f}, PR-AUC={row['PR_AUC']:.4f}, baseline={row['PR_baseline_prevalence']:.4f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
