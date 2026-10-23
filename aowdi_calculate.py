#!/usr/bin/env python3
"""Calculate the Adaptive Overwash Disturbance Index (A-OWDI).

Implements the final frozen formulation described in the manuscript:

    NDVI = (B08 - B04) / (B08 + B04)
    BSI = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))
    MNDWI = (B03 - B11) / (B03 + B11)

    dBSI = BSI_post - BSI_pre
    dNDVI = NDVI_pre - NDVI_post

Persistent open water is excluded before scene standardisation:
    MNDWI_pre > 0 AND MNDWI_post > 0 AND NDVI_pre < 0 AND NDVI_post < 0

Scene-standardised disturbance components:
    S = max(0, z(dBSI))
    Vpre = clip(NDVI_pre, 0, 1)
    G = Vpre * max(0, z(dNDVI))
    A-OWDI = sqrt(S**2 + G**2)

The z-scores use the population standard deviation (ddof=0) calculated only
from valid, non-persistent-water pixels in the analysis domain.

Input assumption
----------------
PRE and POST directories each contain one single-band GeoTIFF for each of
B02, B03, B04, B08 and B11. Filenames must contain the band token, e.g.
"PRE_B02.tif" or "20200420_B11_surface_reflectance.tif".

The rasters must already be co-registered to the same shape, CRS, transform
and extent. The script does not resample or reproject input data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio

BANDS = ("B02", "B03", "B04", "B08", "B11")
FLOAT_NODATA = -9999.0
MASK_NODATA = 255


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calculate the frozen Sentinel-2 Adaptive Overwash Disturbance Index (A-OWDI)."
    )
    p.add_argument("--pre-dir", required=True, type=Path,
                   help="Directory containing PRE-event B02, B03, B04, B08 and B11 GeoTIFFs.")
    p.add_argument("--post-dir", required=True, type=Path,
                   help="Directory containing POST-event B02, B03, B04, B08 and B11 GeoTIFFs.")
    p.add_argument("--outdir", required=True, type=Path,
                   help="Output directory.")
    p.add_argument("--prefix", default="",
                   help="Optional filename prefix, e.g. 'Matagorda_'.")
    p.add_argument("--write-intermediates", action="store_true",
                   help="Also write all pre/post indices and change rasters.")
    return p.parse_args()


def find_band(directory: Path, band: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    candidates = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        candidates.extend(directory.glob(ext))
    matches = sorted({p for p in candidates if band.lower() in p.name.lower()})
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {band} GeoTIFF in {directory}, found {len(matches)}: "
            + ", ".join(str(x.name) for x in matches)
        )
    return matches[0]


def read_band(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float64")
        data = np.asarray(arr.filled(np.nan), dtype="float64")
        valid = ~np.ma.getmaskarray(arr) & np.isfinite(data)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        shape = src.shape
    return data, valid, profile, transform, crs, shape


def same_transform(a, b, tol=1e-9) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(tuple(a), tuple(b)))


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype="float64")
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    out[ok] = num[ok] / den[ok]
    return out


def ndvi(b08, b04):
    return safe_ratio(b08 - b04, b08 + b04)


def bsi(b02, b04, b08, b11):
    return safe_ratio((b11 + b04) - (b08 + b02), (b11 + b04) + (b08 + b02))


def mndwi(b03, b11):
    return safe_ratio(b03 - b11, b03 + b11)


def zscore_scene(x: np.ndarray, mask: np.ndarray):
    vals = x[mask & np.isfinite(x)]
    if vals.size < 2:
        raise RuntimeError("Not enough valid pixels for scene standardisation.")
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise RuntimeError("Scene standard deviation is zero or non-finite.")
    z = np.full(x.shape, np.nan, dtype="float64")
    z[mask] = (x[mask] - mean) / std
    return z, mean, std


def write_float(path: Path, arr: np.ndarray, base_profile: dict):
    profile = base_profile.copy()
    profile.update(dtype="float32", count=1, nodata=FLOAT_NODATA,
                   compress="deflate", predictor=3)
    out = np.where(np.isfinite(arr), arr, FLOAT_NODATA).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def write_mask(path: Path, mask: np.ndarray, base_profile: dict):
    profile = base_profile.copy()
    profile.update(dtype="uint8", count=1, nodata=MASK_NODATA,
                   compress="deflate", predictor=2)
    out = mask.astype("uint8")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def pct(values, ps):
    if values.size == 0:
        return [math.nan] * len(ps)
    return [float(v) for v in np.percentile(values, ps)]


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    files = {"pre": {}, "post": {}}
    for phase, directory in (("pre", args.pre_dir), ("post", args.post_dir)):
        for band in BANDS:
            files[phase][band] = find_band(directory, band)

    data = {"pre": {}, "post": {}}
    masks = {"pre": {}, "post": {}}
    ref_profile = ref_transform = ref_crs = ref_shape = None

    for phase in ("pre", "post"):
        for band in BANDS:
            arr, valid, profile, transform, crs, shape = read_band(files[phase][band])
            if ref_shape is None:
                ref_profile, ref_transform, ref_crs, ref_shape = profile, transform, crs, shape
            else:
                if shape != ref_shape:
                    raise RuntimeError(f"Grid shape mismatch: {files[phase][band]} has {shape}, expected {ref_shape}.")
                if crs != ref_crs:
                    raise RuntimeError(f"CRS mismatch: {files[phase][band]} has {crs}, expected {ref_crs}.")
                if not same_transform(transform, ref_transform):
                    raise RuntimeError(f"Transform mismatch: {files[phase][band]} is not co-registered with reference grid.")
            data[phase][band] = arr
            masks[phase][band] = valid

    # Initial validity: finite/unmasked in all ten inputs. Pixels where every required
    # band equals zero on both dates are treated as outside the exported AOI.
    valid = np.ones(ref_shape, dtype=bool)
    for phase in ("pre", "post"):
        for band in BANDS:
            valid &= masks[phase][band]
            valid &= np.isfinite(data[phase][band])

    all_zero = np.ones(ref_shape, dtype=bool)
    for phase in ("pre", "post"):
        for band in BANDS:
            all_zero &= (data[phase][band] == 0)
    valid &= ~all_zero

    ndvi_pre = ndvi(data["pre"]["B08"], data["pre"]["B04"])
    ndvi_post = ndvi(data["post"]["B08"], data["post"]["B04"])
    bsi_pre = bsi(data["pre"]["B02"], data["pre"]["B04"], data["pre"]["B08"], data["pre"]["B11"])
    bsi_post = bsi(data["post"]["B02"], data["post"]["B04"], data["post"]["B08"], data["post"]["B11"])
    mndwi_pre = mndwi(data["pre"]["B03"], data["pre"]["B11"])
    mndwi_post = mndwi(data["post"]["B03"], data["post"]["B11"])

    index_finite = np.isfinite(ndvi_pre) & np.isfinite(ndvi_post) & np.isfinite(bsi_pre) & np.isfinite(bsi_post) & np.isfinite(mndwi_pre) & np.isfinite(mndwi_post)
    valid &= index_finite

    d_bsi = bsi_post - bsi_pre
    d_ndvi = ndvi_pre - ndvi_post
    d_mndwi = mndwi_post - mndwi_pre

    persistent_water = (
        valid
        & (mndwi_pre > 0)
        & (mndwi_post > 0)
        & (ndvi_pre < 0)
        & (ndvi_post < 0)
    )
    analysis_mask = valid & ~persistent_water

    z_dbsi, dbsi_mean, dbsi_std = zscore_scene(d_bsi, analysis_mask)
    z_dndvi, dndvi_mean, dndvi_std = zscore_scene(d_ndvi, analysis_mask)

    vpre = np.full(ref_shape, np.nan, dtype="float64")
    vpre[analysis_mask] = np.clip(ndvi_pre[analysis_mask], 0.0, 1.0)

    s = np.full(ref_shape, np.nan, dtype="float64")
    g = np.full(ref_shape, np.nan, dtype="float64")
    aowdi = np.full(ref_shape, np.nan, dtype="float64")

    s[analysis_mask] = np.maximum(0.0, z_dbsi[analysis_mask])
    g[analysis_mask] = vpre[analysis_mask] * np.maximum(0.0, z_dndvi[analysis_mask])
    aowdi[analysis_mask] = np.sqrt(s[analysis_mask] ** 2 + g[analysis_mask] ** 2)

    p = args.prefix
    # Essential outputs used by the validation script.
    write_float(args.outdir / f"{p}AOWDI.tif", aowdi, ref_profile)
    write_float(args.outdir / f"{p}z_dBSI.tif", z_dbsi, ref_profile)
    write_float(args.outdir / f"{p}z_dNDVI.tif", z_dndvi, ref_profile)
    write_float(args.outdir / f"{p}NDVI_pre.tif", ndvi_pre, ref_profile)
    write_float(args.outdir / f"{p}BSI_pre.tif", bsi_pre, ref_profile)
    write_float(args.outdir / f"{p}S_sediment_component.tif", s, ref_profile)
    write_float(args.outdir / f"{p}G_vegetation_component.tif", g, ref_profile)
    write_mask(args.outdir / f"{p}analysis_mask.tif", analysis_mask, ref_profile)
    write_mask(args.outdir / f"{p}persistent_water_mask.tif", persistent_water, ref_profile)

    if args.write_intermediates:
        intermediates = {
            "NDVI_post": ndvi_post,
            "BSI_post": bsi_post,
            "MNDWI_pre": mndwi_pre,
            "MNDWI_post": mndwi_post,
            "dBSI": d_bsi,
            "dNDVI": d_ndvi,
            "dMNDWI": d_mndwi,
            "Vpre": vpre,
        }
        for name, arr in intermediates.items():
            write_float(args.outdir / f"{p}{name}.tif", arr, ref_profile)

    analysis_vals = aowdi[analysis_mask & np.isfinite(aowdi)]
    ndvipre_vals = ndvi_pre[analysis_mask & np.isfinite(ndvi_pre)]
    dmndwi_vals = d_mndwi[analysis_mask & np.isfinite(d_mndwi)]
    ps = [1, 5, 25, 50, 75, 95, 99]
    pvals = pct(analysis_vals, ps)

    stats = {
        "valid_pixels": int(valid.sum()),
        "persistent_water_pixels": int(persistent_water.sum()),
        "analysis_pixels": int(analysis_mask.sum()),
        "analysis_fraction_of_valid": float(analysis_mask.sum() / valid.sum()) if valid.sum() else math.nan,
        "dBSI_mean": dbsi_mean,
        "dBSI_std_population": dbsi_std,
        "dNDVI_mean": dndvi_mean,
        "dNDVI_std_population": dndvi_std,
        "NDVI_pre_mean_analysis": float(np.mean(ndvipre_vals)) if ndvipre_vals.size else math.nan,
        "NDVI_pre_median_analysis": float(np.median(ndvipre_vals)) if ndvipre_vals.size else math.nan,
        "dMNDWI_mean_analysis": float(np.mean(dmndwi_vals)) if dmndwi_vals.size else math.nan,
        **{f"AOWDI_P{q}": v for q, v in zip(ps, pvals)},
    }

    with open(args.outdir / f"{p}scene_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with open(args.outdir / f"{p}scene_statistics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key, value in stats.items():
            w.writerow([key, value])

    with open(args.outdir / f"{p}input_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "pre": {k: str(v) for k, v in files["pre"].items()},
            "post": {k: str(v) for k, v in files["post"].items()},
            "crs": str(ref_crs),
            "transform": list(ref_transform)[:6],
            "shape": list(ref_shape),
            "formulation": {
                "dBSI": "BSI_post - BSI_pre",
                "dNDVI": "NDVI_pre - NDVI_post",
                "persistent_water": "MNDWI_pre>0 & MNDWI_post>0 & NDVI_pre<0 & NDVI_post<0",
                "S": "max(0, z(dBSI))",
                "Vpre": "clip(NDVI_pre, 0, 1)",
                "G": "Vpre * max(0, z(dNDVI))",
                "AOWDI": "sqrt(S^2 + G^2)",
                "zscore_std": "population standard deviation (ddof=0)",
            },
        }, f, indent=2)

    print(f"A-OWDI calculation complete: {args.outdir / f'{p}AOWDI.tif'}")
    print(f"Valid pixels: {stats['valid_pixels']:,}")
    print(f"Persistent water: {stats['persistent_water_pixels']:,}")
    print(f"Analysis pixels: {stats['analysis_pixels']:,} ({100*stats['analysis_fraction_of_valid']:.2f}% of valid)")
    print(f"dBSI mean/std: {dbsi_mean:.6f} / {dbsi_std:.6f}")
    print(f"dNDVI mean/std: {dndvi_mean:.6f} / {dndvi_std:.6f}")
    print("A-OWDI percentiles:", ", ".join(f"P{q}={v:.4f}" for q, v in zip(ps, pvals)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
