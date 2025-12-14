# GCM Downscaling Pipeline - Complete Workflow
# This script runs the entire pipeline from preprocessing to inference

param(
    [string]$GcmModel = "BCC-CSM2-MR",
    [string]$BasePath = "d:\appdev\cep ml",
    [switch]$SkipPreprocessing,
    [switch]$SkipTraining,
    [switch]$ProcessAllScenarios
)

$ErrorActionPreference = "Stop"

Write-Host "================================================================================================" -ForegroundColor Cyan
Write-Host "GCM DOWNSCALING PIPELINE FOR PAKISTAN" -ForegroundColor Cyan
Write-Host "================================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  GCM Model: $GcmModel" -ForegroundColor White
Write-Host "  Base Path: $BasePath" -ForegroundColor White
Write-Host ""

# Change to project directory
Set-Location $BasePath

# Activate virtual environment if it exists
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & .\venv\Scripts\Activate.ps1
} else {
    Write-Host "WARNING: Virtual environment not found. Using system Python." -ForegroundColor Yellow
}

Write-Host ""

# Step 1: Preprocessing
if (-not $SkipPreprocessing) {
    Write-Host "================================================================================================" -ForegroundColor Cyan
    Write-Host "STEP 1: PREPROCESSING DATA" -ForegroundColor Cyan
    Write-Host "================================================================================================" -ForegroundColor Cyan
    Write-Host "Regridding to 0.25° grid and aligning temporally (1980-2014)..." -ForegroundColor White
    Write-Host ""
    
    python src/data/preprocessors.py `
        --base-path "$BasePath\AI_GCMs" `
        --output-dir "$BasePath\data\processed\train" `
        --gcm-model $GcmModel `
        --start-year 1980 `
        --end-year 2014
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Preprocessing failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "✓ Preprocessing complete" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "STEP 1: PREPROCESSING - SKIPPED" -ForegroundColor Yellow
    Write-Host ""
}

# Step 2: Create Training DataFrames
Write-Host "================================================================================================" -ForegroundColor Cyan
Write-Host "STEP 2: CREATING TRAINING DATAFRAMES" -ForegroundColor Cyan
Write-Host "================================================================================================" -ForegroundColor Cyan
Write-Host "Flattening 3D fields and adding features..." -ForegroundColor White
Write-Host ""

python src/data/loaders.py `
    --processed-dir "$BasePath\data\processed\train" `
    --output-dir "$BasePath\data\processed" `
    --gcm-model $GcmModel

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Data loading failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ Training data prepared" -ForegroundColor Green
Write-Host ""

# Step 3: Train Models
if (-not $SkipTraining) {
    Write-Host "================================================================================================" -ForegroundColor Cyan
    Write-Host "STEP 3: TRAINING ML MODELS" -ForegroundColor Cyan
    Write-Host "================================================================================================" -ForegroundColor Cyan
    Write-Host "Training RandomForest (temperature) and GradientBoosting (precipitation)..." -ForegroundColor White
    Write-Host "This may take 30-60 minutes..." -ForegroundColor Yellow
    Write-Host ""
    
    python src/models/train.py `
        --data-dir "$BasePath\data\processed" `
        --output-dir "$BasePath\outputs\models"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Model training failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "✓ Models trained and saved" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "STEP 3: TRAINING - SKIPPED" -ForegroundColor Yellow
    Write-Host ""
}

# Step 4: Apply to Future Scenarios
Write-Host "================================================================================================" -ForegroundColor Cyan
Write-Host "STEP 4: APPLYING TO FUTURE SCENARIOS" -ForegroundColor Cyan
Write-Host "================================================================================================" -ForegroundColor Cyan

if ($ProcessAllScenarios) {
    Write-Host "Processing ALL scenarios (9 GCMs × 2 SSPs = 18 jobs)..." -ForegroundColor White
    Write-Host "This may take 2-4 hours..." -ForegroundColor Yellow
    Write-Host ""
    
    python src/inference/downscale_future.py --all `
        --models-path "$BasePath\outputs\models" `
        --base-path "$BasePath\AI_GCMs" `
        --output-dir "$BasePath\outputs\downscaled"
} else {
    Write-Host "Processing single GCM: $GcmModel (SSP126 and SSP585)..." -ForegroundColor White
    Write-Host ""
    
    # Process SSP126
    python src/inference/downscale_future.py `
        --models-path "$BasePath\outputs\models" `
        --base-path "$BasePath\AI_GCMs" `
        --output-dir "$BasePath\outputs\downscaled" `
        --gcm-model $GcmModel `
        --scenario "ssp126"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Downscaling SSP126 failed!" -ForegroundColor Red
        exit 1
    }
    
    # Process SSP585
    python src/inference/downscale_future.py `
        --models-path "$BasePath\outputs\models" `
        --base-path "$BasePath\AI_GCMs" `
        --output-dir "$BasePath\outputs\downscaled" `
        --gcm-model $GcmModel `
        --scenario "ssp585"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Downscaling SSP585 failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "✓ Future scenarios processed" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "================================================================================================" -ForegroundColor Cyan
Write-Host "PIPELINE COMPLETE!" -ForegroundColor Cyan
Write-Host "================================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output Locations:" -ForegroundColor Yellow
Write-Host "  Processed Data:    $BasePath\data\processed\" -ForegroundColor White
Write-Host "  Trained Models:    $BasePath\outputs\models\" -ForegroundColor White
Write-Host "  Downscaled Data:   $BasePath\outputs\downscaled\" -ForegroundColor White
Write-Host "  Figures:           $BasePath\outputs\figures\" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review model performance metrics in outputs/models/*.json" -ForegroundColor White
Write-Host "  2. Run evaluation notebook: jupyter notebook notebooks/04_evaluation.ipynb" -ForegroundColor White
Write-Host "  3. Analyze downscaled future projections in outputs/downscaled/" -ForegroundColor White
Write-Host ""
Write-Host "For detailed documentation, see README.md and QUICKSTART.md" -ForegroundColor Cyan
Write-Host "================================================================================================" -ForegroundColor Cyan
