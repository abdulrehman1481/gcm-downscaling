# ML-based GCM Downscaling Pipeline for Pakistan

End-to-end machine learning pipeline for downscaling monthly temperature and precipitation from 9 Global Climate Models (GCMs) to 0.25° resolution over Pakistan.

## Overview

**Objective:** Downscale GCM climate projections (historical + SSP126/585 scenarios) from coarse resolution (~1-2°) to fine resolution (0.25°) using machine learning models trained on ERA5 reanalysis as the target and CRU as spatial reference.

**Training Period:** 1980-2014  
**Target Region:** Pakistan (approximately 23-38°N, 60-78°E)  
**Target Resolution:** 0.25° (~27.5 km)

## Data

### Input Datasets

- **CRU (Climate Research Unit):**  
  - Temperature: `cru_tmp.1901.2024.0.25deg.pakistan.nc` (°C)
  - Precipitation: `cru_pre.1901.2024.0.25deg.pakistan.nc` (mm/month)
  - Resolution: 0.25°, 1901-2024

- **ERA5 (Reanalysis):**  
  - Temperature: `data_stream-moda_stepType-avgua.nc` (t2m, K)
  - Precipitation: `data_stream-moda_stepType-avgad.nc` (tp, m)
  - Coarser resolution, regridded to 0.25°

- **GCMs (9 models):**
  - BCC-CSM2-MR, CAMS-CSM1-0, CanESM5, CESM2, CESM2-WACCM, EC-Earth3, IPSL-CM6A-LR, MIROC6, MRI-ESM2-0
  - Variables: `tas` (K), `pr` (kg m⁻² s⁻¹)
  - Scenarios: `hist` (historical), `ssp126` (low emissions), `ssp585` (high emissions)

## Methodology

### Preprocessing Pipeline

1. **Load CRU data** → subset 1980-2014 → extract 0.25° Pakistan grid as reference
2. **Load ERA5 data** → convert units (K→°C, m→mm/month) → regrid to CRU grid
3. **Load GCM data** → convert units → regrid to CRU grid → align temporal indices
4. **Harmonize** all datasets to common monthly time index (1980-2014)

### Feature Engineering

**Predictors:**
- GCM temperature (°C)
- GCM precipitation (mm/month)
- Latitude, Longitude
- Cyclic month encoding: `sin(2π·month/12)`, `cos(2π·month/12)`
- Log-transformed precipitation: `log1p(pr_mm)` for precipitation model

**Targets:**
- ERA5 temperature (°C)
- ERA5 precipitation (log1p transformed mm/month)

### Machine Learning Models

**Temperature Downscaling:**
- Algorithm: `RandomForestRegressor`
- Features: `[gcm_tas, gcm_pr_mm, lat, lon, month_sin, month_cos]`
- Target: `era_t2m (°C)`
- Hyperparameters: `n_estimators=200, max_depth=20, n_jobs=-1`

**Precipitation Downscaling:**
- Algorithm: `GradientBoostingRegressor`
- Features: `[gcm_pr_log1p, gcm_tas, lat, lon, month_sin, month_cos]`
- Target: `era_tp_log1p`
- Hyperparameters: `n_estimators=300, learning_rate=0.05, max_depth=3`

### Train/Validation/Test Split

- **Train:** 1980-2005 (26 years)
- **Validation:** 2006-2010 (5 years)
- **Test:** 2011-2014 (4 years)

## Project Structure

```
cep ml/
├── AI_GCMs/                     # Raw climate data
│   ├── CRU/                     # CRU reference data
│   ├── ERA5/                    # ERA5 reanalysis (target)
│   └── GCMs/                    # GCM model outputs
├── src/
│   ├── data/
│   │   ├── preprocessors.py     # Data loading, regridding, unit conversion
│   │   └── loaders.py           # Feature engineering, flattening, splitting
│   ├── models/
│   │   └── train.py             # Model training and evaluation
│   ├── inference/
│   │   └── downscale_future.py  # Apply models to future scenarios
│   └── evaluation/
│       └── metrics.py           # Evaluation metrics and visualization
├── notebooks/
│   ├── 01_data_inspection.ipynb       # Explore NetCDF files
│   ├── 02_preprocessing.ipynb         # Run preprocessing
│   ├── 03_model_training.ipynb        # Train models
│   └── 04_evaluation.ipynb            # Evaluate and visualize
├── data/
│   └── processed/
│       ├── train/               # Processed NetCDF files
│       ├── train_data.parquet   # Training DataFrame
│       ├── val_data.parquet     # Validation DataFrame
│       └── test_data.parquet    # Test DataFrame
├── outputs/
│   ├── models/
│   │   ├── xgb_tas.pkl          # Trained temperature model
│   │   ├── xgb_pr.pkl           # Trained precipitation model
│   │   ├── rf_tas.json          # Training history
│   │   └── gb_pr.json           # Training history
│   ├── downscaled/              # Downscaled future scenarios
│   │   ├── {MODEL}_{SCENARIO}_tas_downscaled_0.25deg.nc
│   │   └── {MODEL}_{SCENARIO}_pr_downscaled_0.25deg.nc
│   └── figures/                 # Diagnostic plots
└── requirements.txt
```

## Installation

### 1. Create Python Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

**Key dependencies:**
- `xarray`, `netCDF4` - NetCDF file handling
- `xesmf` - High-quality regridding (requires ESMPy)
- `scikit-learn` - Machine learning models
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn`, `cartopy` - Visualization

**Note:** `xesmf` installation may require additional setup. If it fails, the pipeline falls back to basic bilinear interpolation.

## Usage

### Step 1: Inspect Data

```powershell
# Open and run the inspection notebook
jupyter notebook notebooks/01_data_inspection.ipynb
```

This will verify:
- Variable names and coordinate conventions
- Temporal ranges and overlap
- Spatial grids and units
- Data quality

### Step 2: Preprocess Data

```powershell
python src/data/preprocessors.py `
  --base-path "d:\appdev\cep ml\AI_GCMs" `
  --output-dir "d:\appdev\cep ml\data\processed\train" `
  --gcm-model "BCC-CSM2-MR" `
  --start-year 1980 `
  --end-year 2014
```

**Output:**
- Regridded, temporally aligned NetCDF files for CRU, ERA5, and GCM (1980-2014)

### Step 3: Create Training DataFrames

```powershell
python src/data/loaders.py `
  --processed-dir "d:\appdev\cep ml\data\processed\train" `
  --output-dir "d:\appdev\cep ml\data\processed" `
  --gcm-model "BCC-CSM2-MR"
```

**Output:**
- `train_data.parquet`, `val_data.parquet`, `test_data.parquet`
- Flattened 3D fields with engineered features

### Step 4: Train Models

```powershell
python src/models/train.py `
  --data-dir "d:\appdev\cep ml\data\processed" `
  --output-dir "d:\appdev\cep ml\outputs\models"
```

**Output:**
-- `xgb_tas.pkl` - Trained temperature model (algorithm-prefixed)
-- `xgb_pr.pkl` - Trained precipitation model (algorithm-prefixed)
- JSON files with training metrics and feature importance

### Step 5: Apply to Future Scenarios

**Single scenario:**
```powershell
python src/inference/downscale_future.py `
  --gcm-model "BCC-CSM2-MR" `
  --scenario "ssp126" `
  --output-dir "d:\appdev\cep ml\outputs\downscaled"
```

**All scenarios (9 GCMs × 2 SSPs = 18 outputs):**
```powershell
python src/inference/downscale_future.py --all `
  --output-dir "d:\appdev\cep ml\outputs\downscaled"
```

**Output:**
- Downscaled NetCDF files: `{MODEL}_{SCENARIO}_{VARIABLE}_downscaled_0.25deg.nc`
- 0.25° resolution, Pakistan domain, 2015-2100

### Step 6: Evaluate and Visualize

```powershell
# Open evaluation notebook
jupyter notebook notebooks/04_evaluation.ipynb
```

**Evaluation includes:**
- Test-set RMSE, MAE, R²
- Seasonal climatology maps (DJF, JJA)
- Spatial pattern correlation
- Scatter plots (predicted vs observed)
- Time series at selected grid points

## Evaluation Metrics

### Temperature (Test Period: 2011-2014)
- RMSE (°C)
- MAE (°C)
- Bias (°C)
- Spatial correlation

### Precipitation (Test Period: 2011-2014)
- RMSE (mm/month) - in both log and original space
- MAE (mm/month)
- Bias (mm/month)
- Spatial correlation

### Seasonal Pattern Skill
- Pattern correlation for DJF, MAM, JJA, SON
- Bias maps for each season
- Climatological monthly cycle comparison

## Key Considerations

### 1. ERA5 Variable Verification
The ERA5 filenames (`avgad`, `avgua`) are non-standard. **Before running preprocessing**, verify in the inspection notebook that these files contain:
- `t2m` (2-meter temperature) or equivalent
- `tp` (total precipitation) or equivalent

If not, you may need to download standard ERA5 monthly data.

### 2. Regridding Method
- Default: **Bilinear interpolation** for both variables
- Recommended for precipitation: **Conservative regridding** (mass-preserving)
- To use conservative: modify `method='conservative'` in `preprocessors.py`

### 3. MVP Approach
This pipeline trains on **one GCM** (BCC-CSM2-MR by default) to establish the workflow. To scale up:

**Option A:** Train separate models for each GCM (9 models)
**Option B:** Pool all GCMs with GCM-ID as categorical feature (1 model)
**Option C:** Multi-model ensemble approach

### 4. Hyperparameter Tuning
Current hyperparameters are fixed for MVP. For production:
- Use validation set (2006-2010) with `GridSearchCV`
- Optimize `max_depth`, `n_estimators`, `learning_rate`
- Consider early stopping for GradientBoosting

### 5. Precipitation Zero-Inflation
Log-transform helps but doesn't fully address dry months. Consider:
- Two-stage model: (1) classify occurrence, (2) predict amount
- Alternative: Quantile regression or zero-inflated models

## Output Format

Downscaled NetCDF files follow CF conventions:

```python
<xarray.DataArray 'tas' (time: 1032, lat: 61, lon: 73)>
Dimensions:
  time: 1032  # Monthly timesteps (2015-2100)
  lat: 61     # 0.25° grid
  lon: 73     # 0.25° grid
Attributes:
  long_name: Near-Surface Air Temperature (downscaled)
  units: degC
  downscaling_method: RandomForest regression
  target_resolution: 0.25 degrees
  source_gcm: BCC-CSM2-MR
  scenario: ssp126
```

### Produced filenames (exact)

- Models (saved to `outputs/models/`):
  - `<algorithm>_tas.pkl` (e.g., `xgboost_tas.pkl`)
  - `<algorithm>_pr.pkl` (single-stage) or `<algorithm>_pr_occ.pkl` + `<algorithm>_pr_amt.pkl` (two-stage)
  - Corresponding training history JSON files (same basename, `.json`)

- Downscaled future outputs (saved to `outputs/downscaled/`):
  - `{GCM}_{SCENARIO}_tas_downscaled_0.25deg.nc` (e.g., `BCC-CSM2-MR_ssp126_tas_downscaled_0.25deg.nc`)
  - `{GCM}_{SCENARIO}_pr_downscaled_0.25deg.nc`
  - Diagnostics: `{GCM}_{SCENARIO}_diag_tas_timeseries.png`, `{GCM}_{SCENARIO}_diag_pr_timeseries.png`

- Ensemble statistics (saved to `outputs/downscaled/`):
  - `ensemble_mean_{SCENARIO}.nc`
  - `ensemble_std_{SCENARIO}.nc`

Each downscaled NetCDF includes provenance attributes pointing to the model files used (e.g., `model_file`, `precip_occurrence_model`, `precip_amount_model`).

## Citation

If using this pipeline, please cite:
- **CRU TS:** Harris et al. (2020)
- **ERA5:** Hersbach et al. (2020)
- **CMIP6 GCMs:** Individual model citations
- Downscaling methodology: [Your publication]

## Troubleshooting

**xesmf installation fails:**
- Fallback: Pipeline uses basic `xarray.interp()` instead
- For production: Install ESMPy via conda: `conda install -c conda-forge esmpy`

**Memory errors:**
- Process GCMs sequentially instead of in batch
- Reduce spatial domain or temporal chunks
- Use Dask for lazy loading: `xr.open_dataset(..., chunks={'time': 120})`

**Time coordinate misalignment:**
- Different calendars (360-day, noleap, gregorian)
- Solution: Convert to common calendar in preprocessing

## Contact

For questions or issues, please open an issue in the repository.

## License

[Specify license]
