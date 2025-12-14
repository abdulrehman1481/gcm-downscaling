# Pre-Flight Checklist

Use this checklist before running the downscaling pipeline to ensure everything is set up correctly.

## ☑️ Environment Setup

- [ ] Python 3.8+ installed
  ```powershell
  python --version
  ```

- [ ] Virtual environment created and activated
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

- [ ] Dependencies installed
  ```powershell
  pip install -r requirements.txt
  ```

- [ ] Installation successful (check for errors)
  ```powershell
  python -c "import xarray, pandas, sklearn, numpy; print('✓ Core packages OK')"
  ```

## ☑️ Data Verification

- [ ] All 58 NetCDF files present in `AI_GCMs/`
  - [ ] 2 CRU files in `AI_GCMs/CRU/`
  - [ ] 2 ERA5 files in `AI_GCMs/ERA5/`
  - [ ] 54 GCM files in `AI_GCMs/GCMs/` (9 models × 3 scenarios × 2 vars)

- [ ] Data inspection notebook runs without errors
  ```powershell
  jupyter notebook notebooks/01_data_inspection.ipynb
  ```

- [ ] **CRITICAL:** ERA5 variable names verified
  - [ ] Temperature variable identified (t2m or similar)
  - [ ] Precipitation variable identified (tp or similar)
  - [ ] Note: If variables are NOT found, you need to download standard ERA5 data

- [ ] Coordinate conventions checked
  - [ ] Time coordinate name (time/TIME/Time)
  - [ ] Latitude coordinate name (lat/latitude/y)
  - [ ] Longitude coordinate name (lon/longitude/x)

- [ ] Temporal ranges verified
  - [ ] CRU: Contains 1980-2014
  - [ ] ERA5: Contains 1980-2014
  - [ ] GCM hist: Contains 1980-2014
  - [ ] GCM future: Contains data beyond 2014

- [ ] Units confirmed
  - [ ] GCM temperature in Kelvin (values > 250)
  - [ ] ERA5 temperature in Kelvin or Celsius
  - [ ] GCM precipitation in kg m⁻² s⁻¹ (small values)
  - [ ] ERA5 precipitation in meters or mm

## ☑️ Directory Structure

- [ ] Project directories exist
  - [ ] `src/data/`
  - [ ] `src/models/`
  - [ ] `src/inference/`
  - [ ] `src/evaluation/`
  - [ ] `notebooks/`
  - [ ] `data/processed/train/` (will be created)
  - [ ] `outputs/models/` (will be created)
  - [ ] `outputs/downscaled/` (will be created)
  - [ ] `outputs/figures/` (will be created)

## ☑️ Configuration

- [ ] `config.yaml` reviewed and paths are correct
- [ ] Base path set correctly: `d:\appdev\cep ml`
- [ ] GCM model selected (default: BCC-CSM2-MR for MVP)
- [ ] Training period confirmed: 1980-2014
- [ ] Hyperparameters reviewed (or use defaults)

## ☑️ System Resources

- [ ] Available RAM: **16GB+ recommended** (check Task Manager)
- [ ] Available disk space: **50GB+ free** (check drive properties)
  ```powershell
  Get-PSDrive D | Select-Object Used,Free
  ```

- [ ] CPU cores available for parallel processing
  ```powershell
  (Get-CimInstance Win32_Processor).NumberOfCores
  ```

## ☑️ Optional: xESMF Installation

xESMF provides high-quality regridding but is optional (pipeline has fallback).

- [ ] Try installing xESMF (if conda available):
  ```powershell
  conda install -c conda-forge esmpy xesmf
  ```

- [ ] If xESMF fails: **Pipeline will use basic interpolation** ✓

## ☑️ Test Run (Recommended)

Before full pipeline, test each module:

- [ ] Test preprocessing (quick check):
  ```powershell
  python -c "from src.data.preprocessors import ClimateDataPreprocessor; print('✓ Preprocessor OK')"
  ```

- [ ] Test data loader:
  ```powershell
  python -c "from src.data.loaders import DownscalingDataLoader; print('✓ Loader OK')"
  ```

- [ ] Test model module:
  ```powershell
  python -c "from src.models.train import DownscalingModel; print('✓ Models OK')"
  ```

- [ ] Test evaluation:
  ```powershell
  python -c "from src.evaluation.metrics import DownscalingEvaluator; print('✓ Evaluator OK')"
  ```

## ☑️ Workflow Decision

Choose your workflow path:

### Option A: Interactive (Recommended for First Time)
- [ ] Use Jupyter notebooks for step-by-step exploration
- [ ] Start with: `notebooks/01_data_inspection.ipynb`
- [ ] Then run: `notebooks/02_complete_workflow.ipynb`

### Option B: Automated Script
- [ ] Run full pipeline: `.\run_pipeline.ps1 -GcmModel "BCC-CSM2-MR"`
- [ ] Monitor console output for progress
- [ ] Check logs for any warnings

### Option C: Manual Step-by-Step
- [ ] Run preprocessing: `python src/data/preprocessors.py`
- [ ] Run feature engineering: `python src/data/loaders.py`
- [ ] Run training: `python src/models/train.py`
- [ ] Run inference: `python src/inference/downscale_future.py`

## ☑️ Time Budget

Allocate sufficient time for your chosen workflow:

- [ ] **Inspection:** 10-15 minutes
- [ ] **Preprocessing:** 15-20 minutes
- [ ] **Feature engineering:** 5 minutes
- [ ] **Model training:** 30-60 minutes
- [ ] **Single scenario inference:** 5-10 minutes per scenario
- [ ] **Total for MVP:** ~1.5-2 hours (uninterrupted)

## ☑️ Expected Errors (Normal)

These warnings/errors are **acceptable** and handled by the pipeline:

✓ `xesmf not found` → Pipeline uses fallback interpolation  
✓ `FutureWarning` from pandas/xarray → Informational only  
✓ Different calendar warnings → Handled automatically  
✓ Deprecation warnings → Can be ignored for now

## ⛔ Stop If You See

These errors indicate problems that must be fixed:

❌ `FileNotFoundError` for NetCDF files → Check data paths  
❌ `MemoryError` → Reduce chunk size or close other programs  
❌ `KeyError` for variables → ERA5 variable names incorrect  
❌ `ImportError` for core packages → Reinstall dependencies  
❌ `ValueError` in coordinate alignment → Check time ranges

## 📋 Pre-Run Summary

**Before proceeding, confirm:**

1. ✅ All dependencies installed without errors
2. ✅ Data files verified in inspection notebook
3. ✅ ERA5 variables identified correctly
4. ✅ Sufficient disk space and RAM available
5. ✅ Workflow path chosen (interactive/automated/manual)
6. ✅ Time allocated for uninterrupted run

## 🚦 Ready to Start?

If all checks above are ✅, you're ready to run the pipeline!

### Quick Start Command

```powershell
# Interactive workflow (recommended)
jupyter notebook notebooks/01_data_inspection.ipynb

# OR automated workflow
.\run_pipeline.ps1 -GcmModel "BCC-CSM2-MR"
```

## 📞 If Something Goes Wrong

1. **Re-run data inspection notebook** to verify inputs
2. **Check console output** for specific error messages
3. **Review troubleshooting section** in README.md
4. **Verify file paths** in config.yaml
5. **Check available resources** (RAM, disk space)

## 📝 Notes Section

Use this space to note any issues or observations:

```
Date: ___________
Issue: 
Solution:

Date: ___________
Issue:
Solution:
```

---

**Good luck! 🚀**

Once all items are checked, proceed to `PROJECT_SUMMARY.md` for detailed instructions.
