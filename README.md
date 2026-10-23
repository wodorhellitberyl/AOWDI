# Adaptive Overwash Disturbance Index (A-OWDI)

Reference implementation of the **Adaptive Overwash Disturbance Index (A-OWDI)** for detecting storm- and inundation-driven coastal surface disturbance from Sentinel-2 imagery.

The repository accompanies the manuscript:

> Moskalewicz, D. *A transferable Sentinel-2 Adaptive Overwash Disturbance Index (A-OWDI) for detecting inundation-driven change across contrasting coastal barriers.*

## Recommended archive workflow

Use **GitHub + Zenodo together**:

1. Keep the readable, version-controlled code on GitHub.
2. Connect the GitHub repository to Zenodo.
3. Create a GitHub release corresponding exactly to the manuscript/submission version (for example `v1.0.0`).
4. Let Zenodo archive that release and mint a DOI.
5. Cite the Zenodo DOI in the manuscript under **Data and code availability**.

GitHub is best for reading, issue tracking, and future updates. Zenodo is best for permanent archiving, reproducibility, and citation of the exact version used in the paper.

---

## Repository contents

### `aowdi_calculate.py`
Calculates the final, frozen A-OWDI formulation from pre- and post-event Sentinel-2 Level-2A surface-reflectance bands.

### `aowdi_validate.py`
Validates A-OWDI against mapped washover polygons using the pre-event-state-matched background procedure described in the manuscript. It also reports full-background metrics and feature-level A-OWDI summaries.

### `requirements.txt`
Minimal Python dependencies.

---

# 1. Input data required to calculate A-OWDI

For **each event**, prepare two directories: one containing the PRE-event scene and one containing the POST-event scene.

Each directory must contain one single-band GeoTIFF for each of the following Sentinel-2 bands:

| Band | Spectral region | Native Sentinel-2 resolution | Use in A-OWDI |
|---|---|---:|---|
| B02 | Blue | 10 m | BSI |
| B03 | Green | 10 m | MNDWI |
| B04 | Red | 10 m | NDVI, BSI |
| B08 | NIR | 10 m | NDVI, BSI |
| B11 | SWIR | 20 m | BSI, MNDWI |

Filenames may be arbitrary but must contain the corresponding band token (`B02`, `B03`, `B04`, `B08`, `B11`). Exactly one GeoTIFF per band must be present in each PRE/POST directory.

Example:

```text
PRE/
  Sentinel2_PRE_B02.tif
  Sentinel2_PRE_B03.tif
  Sentinel2_PRE_B04.tif
  Sentinel2_PRE_B08.tif
  Sentinel2_PRE_B11.tif

POST/
  Sentinel2_POST_B02.tif
  Sentinel2_POST_B03.tif
  Sentinel2_POST_B04.tif
  Sentinel2_POST_B08.tif
  Sentinel2_POST_B11.tif
```

## Important preprocessing requirement

All ten rasters must already have the **same CRS, raster dimensions, pixel grid, transform and extent**. The script intentionally does **not** resample or reproject input data because spatial harmonisation should be controlled before the index is calculated.

The workflow used in the study exported all bands for each event pair to an identical grid and extent. Although a common output grid may have pixel spacing close to 10 m, A-OWDI contains B11 information and therefore its effective SWIR information content remains constrained by the native 20 m resolution of Sentinel-2 B11.

The script assumes Level-2A surface reflectance or consistently scaled reflectance values. Normalized indices are unaffected by a purely multiplicative scaling factor. If raw Sentinel-2 SAFE digital numbers are used, any product-specific additive offsets must be handled according to the product metadata before running this implementation.

---

# 2. A-OWDI formulation

For PRE and POST imagery, the following spectral indices are calculated:

```text
NDVI = (B08 - B04) / (B08 + B04)

BSI = ((B11 + B04) - (B08 + B02)) /
      ((B11 + B04) + (B08 + B02))

MNDWI = (B03 - B11) / (B03 + B11)
```

Event-related changes are defined as:

```text
DeltaBSI  = BSI_post - BSI_pre
DeltaNDVI = NDVI_pre - NDVI_post
```

Thus, a positive `DeltaBSI` represents increasing bare-soil/sediment response, while a positive `DeltaNDVI` represents vegetation loss.

## Persistent-water exclusion

Persistent open water is excluded **before scene standardisation** using the fixed rule:

```text
MNDWI_pre  > 0
AND MNDWI_post > 0
AND NDVI_pre   < 0
AND NDVI_post  < 0
```

Newly inundated land, wetlands, and surfaces whose hydrological state changed during the event remain in the analysis domain.

## Scene standardisation

`DeltaBSI` and `DeltaNDVI` are standardized independently over the valid non-persistent-water analysis domain:

```text
z(x) = (x - scene_mean) / scene_standard_deviation
```

The implementation uses the **population standard deviation (`ddof=0`)**.

## Final frozen formulation

```text
S = max(0, z(DeltaBSI))

Vpre = clip(NDVI_pre, 0, 1)

G = Vpre * max(0, z(DeltaNDVI))

A-OWDI = sqrt(S^2 + G^2)
```

Only positive evidence is retained. A negative response in one spectral component therefore cannot cancel positive disturbance evidence in the other component.

A-OWDI is a **continuous, scene-relative disturbance indicator**. It is not a universal binary washover classifier and no universal threshold is fitted by these scripts.

---

# 3. Installation

Python 3.10 or newer is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies are NumPy, Rasterio, GeoPandas, SciPy and scikit-learn.

---

# 4. Calculate A-OWDI

Example:

```bash
python aowdi_calculate.py \
  --pre-dir /path/to/PRE \
  --post-dir /path/to/POST \
  --outdir /path/to/AOWDI_output \
  --prefix Matagorda_ \
  --write-intermediates
```

On Windows PowerShell, the same command can be entered on one line:

```powershell
python aowdi_calculate.py --pre-dir "D:\site\PRE" --post-dir "D:\site\POST" --outdir "D:\site\AOWDI_output" --prefix "Site_" --write-intermediates
```

## Essential outputs

The script always writes:

```text
AOWDI.tif
z_dBSI.tif
z_dNDVI.tif
NDVI_pre.tif
BSI_pre.tif
S_sediment_component.tif
G_vegetation_component.tif
analysis_mask.tif
persistent_water_mask.tif
scene_statistics.csv
scene_statistics.json
input_manifest.json
```

With `--write-intermediates`, it additionally writes POST indices, MNDWI rasters, change rasters and `Vpre`.

`scene_statistics.csv/json` reports pixel counts, standardisation parameters, pre-event NDVI statistics and A-OWDI percentiles (`P1`, `P5`, `P25`, `P50`, `P75`, `P95`, `P99`).

`input_manifest.json` records exactly which files were used and stores the final formula in machine-readable form.

---

# 5. Validation against mapped washovers

Validation requires the essential outputs from `aowdi_calculate.py` plus a polygon reference dataset.

Supported vector formats include GeoPackage, GeoJSON and Shapefile. The vector is automatically reprojected to the raster CRS when required.

Example:

```bash
python aowdi_validate.py \
  --aowdi /path/to/AOWDI.tif \
  --z-dbsi /path/to/z_dBSI.tif \
  --z-dndvi /path/to/z_dNDVI.tif \
  --ndvi-pre /path/to/NDVI_pre.tif \
  --bsi-pre /path/to/BSI_pre.tif \
  --analysis-mask /path/to/analysis_mask.tif \
  --reference /path/to/reference.gpkg \
  --layer NCB_Florence_Washover \
  --outdir /path/to/validation_output
```

For Quissamã-style reference polygons stored in a single-layer file, `--layer` may be omitted.

## Matched-background procedure

1. Reference polygons are rasterised to the A-OWDI grid.
2. Positive pixels outside the valid analysis domain are discarded.
3. Candidate background pixels are taken from the valid analysis domain outside the positive polygons and a **2-pixel raster exclusion zone** around them. The exclusion zone is created by dilating the rasterised polygon mask for two iterations with an 8-neighbour 3 x 3 structuring element.
4. Matching uses only PRE-event conditions:
   - `NDVI_pre`
   - `BSI_pre`
5. Both matching variables are standardized using the candidate-background distribution.
6. Each positive pixel is assigned its nearest background neighbour in this two-dimensional standardized feature space.
7. Matching is performed **with replacement**, so the same background pixel may be used for more than one positive pixel when it is the closest spectral analogue.

The frozen manuscript workflow uses a two-pixel raster exclusion zone. The option is exposed only for sensitivity testing; published/transfer validation should retain the frozen value:

```bash
--exclusion-pixels 2
```

The default rasterisation follows pixel-centre inclusion. Use `--all-touched` only if that convention is explicitly required for a separate experiment; do not change rasterisation rules during external validation of a frozen workflow.

## Validation outputs

```text
validation_metrics_matched.csv
validation_metrics_full_background.csv
matched_background_pairs.csv
feature_level_AOWDI_summary.csv
validation_summary.json
validation_positive_mask.tif
validation_candidate_background_mask.tif
validation_full_background_mask.tif
```

For A-OWDI, `z(DeltaBSI)` and `z(DeltaNDVI)`, the script reports:

- ROC-AUC
- PR-AUC
- average precision (additional diagnostic)
- prevalence-derived PR baseline
- positive and negative sample counts

For matched validation, it also reports how many **unique** background pixels were used because matching is performed with replacement.

At feature level, each reference polygon receives:

- number of raster pixels
- mean A-OWDI
- P50, P75 and P90 A-OWDI
- the corresponding percentile positions in the scene-wide A-OWDI distribution

These outputs reproduce the type of feature-level analysis used for the independent Quissamã validation in the manuscript.

---

# 6. Minimal workflow for applying A-OWDI to a new site

For a new storm event:

1. Select the closest suitable cloud-free PRE-event Sentinel-2 Level-2A image that represents undisturbed conditions before storm-related wave forcing begins.
2. Select the earliest sufficiently cloud-free POST-event image.
3. Export B02, B03, B04, B08 and B11 for both dates to exactly the same grid and extent.
4. Run `aowdi_calculate.py`.
5. Inspect `analysis_mask.tif` and `persistent_water_mask.tif` in GIS.
6. Inspect A-OWDI as a continuous disturbance surface; do not apply a universal threshold.
7. If independent mapped washover polygons are available, run `aowdi_validate.py`.
8. Interpret high A-OWDI values together with geomorphology, post-event imagery and, where possible, higher-resolution or field data.

---

# 7. Interpretation and known limitations

A-OWDI was designed to remain informative when the dominant disturbance signal changes between environments, but the following limitations remain important:

- phenology can change NDVI independently of storm disturbance;
- antecedent sediment moisture and tidal stage can modify BSI/SWIR response;
- cloud, haze and residual atmospheric effects can influence pre/post differences;
- wet sand, tidal flats, beach accretion, vegetation damage, nourishment, agricultural activity or other anthropogenic changes may create spectrally similar disturbance signals;
- B11 has a native 20 m resolution, so small or narrow washover forms may be poorly resolved even if data are exported to a finer common grid;
- scene standardisation makes A-OWDI a relative indicator within each analysis domain; absolute A-OWDI values should not be interpreted as direct measures of storm energy or sediment volume between events;
- independent high-resolution validation is strongly recommended when A-OWDI is used quantitatively.

---

# 8. Reproducibility notes

For a published analysis, archive at least:

- the exact PRE and POST acquisition dates;
- filenames/product identifiers for all Sentinel-2 bands;
- the analysis area;
- the generated `input_manifest.json`;
- the generated `scene_statistics.json`;
- any reference polygons used for validation, subject to their licence/data-sharing conditions;
- the GitHub release and corresponding Zenodo DOI for the code version.

Do not modify the A-OWDI equation, vegetation weight, water-mask rule or standardisation procedure after examining external validation/application results if the objective is to test transferability of the frozen formulation.

Before tagging the archival `v1.0.0` release, rerun at least one development site and the independent Quissamã validation and confirm that the manuscript values correspond to this exact code version. The release candidate was regression-tested on Matagorda (A-OWDI calculation) and Quissamã (polygon validation). For Quissamã, the frozen two-pixel raster-exclusion implementation returned 772 positive pixels and 12,491 eligible background pixels; matched A-OWDI ROC-AUC/PR-AUC = 0.7040/0.6886 and full-background ROC-AUC/PR-AUC = 0.7750/0.2186. These values should be used when checking the final manuscript against code v1.0.0.

---

# 9. Suggested manuscript statement

> **Data and code availability.** Sentinel-2 imagery used in this study is publicly available through the Copernicus Data Space Ecosystem. Reference datasets obtained from collaborators are available from the respective authors subject to their data-sharing conditions. The Python implementation of A-OWDI calculation, scene standardisation and validation is available at [GitHub URL] and archived at Zenodo under DOI: [Zenodo DOI].

---

# 10. Citation

When a DOI is minted, cite the archived Zenodo release rather than only the mutable GitHub URL.

Suggested repository citation format:

> Moskalewicz, D. (2026). Adaptive Overwash Disturbance Index (A-OWDI): reference Python implementation (Version 1.0.0). Zenodo. https://doi.org/[DOI]


---

## Software licence

Before the public release, choose an explicit software licence in GitHub/Zenodo. A permissive licence such as **MIT** is a common choice for research code when unrestricted reuse with attribution is desired, but the licence should be selected by the author/institution.
