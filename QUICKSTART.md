# Quick Start Guide

Get started with the GCM downscaling pipeline in 5 steps.

## Prerequisites

- Python 3.8+
- 16GB+ RAM recommended
- ~50GB disk space for processed data

## Installation

```powershell
# Clone or navigate to project directory
cd "d:\appdev\cep ml"

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Workflow

### 1. Inspect Data (5 minutes)

Verify your NetCDF files contain the expected variables:

```powershell
jupyter notebook notebooks/01_data_inspection.ipynb
```

**Important:** Check that ERA5 files contain `t2m` (temperature) and `tp` (precipitation). The filenames are non-standard, so verify variable names before proceeding.

### 2. Preprocess Data (10-20 minutes)

Regrid all data to common 0.25° grid and align temporally:

```powershell
python src/data/preprocessors.py --gcm-model "BCC-CSM2-MR"
```

**Output:** Processed NetCDF files in `data/processed/train/`

### 3. Create Training Data (5 minutes)

Flatten 3D fields and create train/val/test splits:

```powershell
python src/data/loaders.py --gcm-model "BCC-CSM2-MR"
```

**Output:** Parquet files in `data/processed/`
- `train_data.parquet` (~300MB)
- `val_data.parquet`
- `test_data.parquet`

### 4. Train Models (30-60 minutes)

Train RandomForest (temperature) and GradientBoosting (precipitation):

```powershell
python src/models/train.py
```

**Output:** 
-- `outputs/models/xgb_tas.pkl` - Temperature model
-- `outputs/models/xgb_pr.pkl` - Precipitation model
- JSON files with metrics

**Expected Performance (Test Set 2011-2014):**
- Temperature RMSE: ~1-2°C
- Precipitation RMSE: ~20-40 mm/month
- R² > 0.85 for both variables

### 5. Apply to Future Scenarios (2-4 hours for all 18 scenarios)

**Single scenario:**
```powershell
python src/inference/downscale_future.py `
  --gcm-model "BCC-CSM2-MR" `
  --scenario "ssp126"
```

**All scenarios (9 GCMs × 2 SSPs):**
```powershell
python src/inference/downscale_future.py --all
```

**Output:** Downscaled NetCDF files in `outputs/downscaled/`
- `{MODEL}_{SCENARIO}_tas_downscaled_0.25deg.nc`
- `{MODEL}_{SCENARIO}_pr_downscaled_0.25deg.nc`

## Complete Workflow Notebook

For an interactive step-by-step guide:

```powershell
jupyter notebook notebooks/02_complete_workflow.ipynb
```

This notebook runs all steps in sequence with visualizations.

## Validation

Evaluate model performance:

```powershell
jupyter notebook notebooks/04_evaluation.ipynb
```

Generates:
- Scatter plots (predicted vs observed)
- Spatial bias maps
- Seasonal climatology comparisons
- Time series at selected locations

## Troubleshooting

### xESMF not installed
- Pipeline will use basic `xarray.interp()` instead
- For better regridding: `conda install -c conda-forge esmpy xesmf`

### Memory errors during training
- Reduce data: Process fewer years or smaller spatial domain
- Use Dask for lazy loading: `xr.open_dataset(..., chunks={'time': 120})`

### Time coordinate issues
- Different GCMs may use different calendars (360-day, noleap, gregorian)
- Preprocessing handles this automatically via `xr.decode_cf()`

### ERA5 variables not found
- Check `01_data_inspection.ipynb` output
- ERA5 filenames (`avgad`, `avgua`) may not contain standard variable names
- You may need to download standard ERA5 monthly data

## File Size Expectations

| File Type | Size (approx) |
|-----------|---------------|
| Raw GCM NetCDF | 50-200 MB each |
| Processed NetCDF (1980-2014) | 20-50 MB each |
| Training parquet | 200-400 MB |
| Trained models | 100-500 MB each |
| Downscaled future NetCDF | 100-300 MB each |

## Performance Tips

1. **Use SSD for data directory** - Much faster I/O for NetCDF files
2. **Parallel processing** - Models use `n_jobs=-1` by default
3. **Batch inference** - Process multiple scenarios overnight
4. **Monitor memory** - Close unused datasets with `.close()`

## Next Steps

After completing the workflow:

1. **Validate downscaling quality**
   - Compare spatial patterns
   - Check seasonal cycles
   - Verify extreme values

2. **Multi-model ensemble**
   - Train on all 9 GCMs (option B from README)
   - Generate ensemble mean and spread
   - Quantify inter-model uncertainty

3. **Production deployment**
   - Hyperparameter tuning with validation set
   - Cross-validation across GCMs
   - Implement conservative regridding for precipitation

## Questions?

- Check `README.md` for detailed documentation
- Review notebooks for examples
- Open an issue if you encounter problems

## Timeline

| Task | Duration |
|------|----------|
| Setup + inspection | 15 min |
| Preprocessing (1 GCM) | 15 min |
| Feature engineering | 5 min |
| Model training | 45 min |
| **Total for MVP** | **~1.5 hours** |
| Apply to all futures | 3 hours |
| **Complete pipeline** | **~5 hours** |

*Times are approximate and depend on system performance.*
