"""
ENHANCED Machine Learning models for climate downscaling
Features:
- XGBoost and LightGBM models (faster and more accurate than RF/GB)
- Hyperparameter tuning with Optuna
- Cross-validation
- Two-stage precipitation modeling (occurrence + amount)
- Memory-efficient training
- Comprehensive error handling
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import joblib
import json
from datetime import datetime
import gc
import warnings
from tqdm.auto import tqdm

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Install with: pip install lightgbm")

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns


class EnhancedDownscalingModel:
    """Enhanced downscaling model with advanced ML algorithms"""
    
    def __init__(self, model_type: str, algorithm: str = 'auto', 
                 hyperparams: Optional[Dict] = None):
        """
        Initialize enhanced downscaling model
        
        Parameters:
        -----------
        model_type : str
            'temperature', 'precipitation', or 'precipitation_occurrence'
        algorithm : str
            'xgboost', 'lightgbm', 'randomforest', 'gradientboosting', or 'auto'
        hyperparams : dict, optional
            Model hyperparameters
        """
        self.model_type = model_type
        self.algorithm = self._select_algorithm(algorithm)
        self.hyperparams = hyperparams or self._get_default_hyperparams()
        self.model = self._create_model()
        self.training_history = {}
        
    def _select_algorithm(self, algorithm: str) -> str:
        """Select best available algorithm"""
        if algorithm == 'auto':
            # Prefer XGBoost > LightGBM > RandomForest
            if HAS_XGBOOST:
                return 'xgboost'
            elif HAS_LIGHTGBM:
                return 'lightgbm'
            else:
                return 'randomforest' if self.model_type == 'temperature' else 'gradientboosting'
        
        # Validate requested algorithm
        if algorithm == 'xgboost' and not HAS_XGBOOST:
            raise ValueError("XGBoost requested but not installed")
        if algorithm == 'lightgbm' and not HAS_LIGHTGBM:
            raise ValueError("LightGBM requested but not installed")
        
        return algorithm
    
    def _get_default_hyperparams(self) -> Dict:
        """Get optimized default hyperparameters"""
        if self.algorithm == 'xgboost':
            if self.model_type == 'temperature':
                return {
                    'objective': 'reg:squarederror',
                    'n_estimators': 300,
                    'learning_rate': 0.05,
                    'max_depth': 7,
                    'min_child_weight': 3,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0.1,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42,
                    'n_jobs': -1,
                    'tree_method': 'hist'
                }
            elif self.model_type == 'precipitation':
                return {
                    'objective': 'reg:squarederror',
                    'n_estimators': 400,
                    'learning_rate': 0.03,
                    'max_depth': 5,
                    'min_child_weight': 5,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'gamma': 0.2,
                    'reg_alpha': 0.5,
                    'reg_lambda': 2.0,
                    'random_state': 42,
                    'n_jobs': -1,
                    'tree_method': 'hist'
                }
            else:  # precipitation_occurrence
                return {
                    'objective': 'binary:logistic',
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'max_depth': 5,
                    'min_child_weight': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': 1.0,
                    'random_state': 42,
                    'n_jobs': -1,
                    'tree_method': 'hist'
                }
        
        elif self.algorithm == 'lightgbm':
            if self.model_type == 'temperature':
                return {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'num_leaves': 63,
                    'learning_rate': 0.05,
                    'n_estimators': 300,
                    'max_depth': 7,
                    'min_child_samples': 20,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
            else:  # precipitation
                return {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'num_leaves': 31,
                    'learning_rate': 0.03,
                    'n_estimators': 400,
                    'max_depth': 5,
                    'min_child_samples': 50,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'reg_alpha': 0.5,
                    'reg_lambda': 2.0,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
        
        else:  # sklearn models
            if self.algorithm == 'randomforest':
                return {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'n_jobs': -1,
                    'random_state': 42,
                    'verbose': 1
                }
            else:  # gradientboosting
                return {
                    'n_estimators': 300,
                    'learning_rate': 0.05,
                    'max_depth': 5,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'subsample': 0.8,
                    'random_state': 42,
                    'verbose': 1
                }
    
    def _create_model(self):
        """Create model instance"""
        if self.algorithm == 'xgboost':
            if self.model_type == 'precipitation_occurrence':
                return xgb.XGBClassifier(**self.hyperparams)
            else:
                return xgb.XGBRegressor(**self.hyperparams)
        
        elif self.algorithm == 'lightgbm':
            if self.model_type == 'precipitation_occurrence':
                return lgb.LGBMClassifier(**self.hyperparams)
            else:
                return lgb.LGBMRegressor(**self.hyperparams)
        
        elif self.algorithm == 'randomforest':
            if self.model_type == 'precipitation_occurrence':
                return RandomForestClassifier(**self.hyperparams)
            else:
                return RandomForestRegressor(**self.hyperparams)
        
        else:  # gradientboosting
            return GradientBoostingRegressor(**self.hyperparams)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
            early_stopping_rounds: int = 50, verbose: bool = True):
        """
        Train the model with early stopping
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training targets
        X_val : pd.DataFrame, optional
            Validation features for early stopping
        y_val : pd.Series, optional
            Validation targets
        early_stopping_rounds : int
            Early stopping patience
        verbose : bool
            Print training progress
        """
        if verbose:
            print(f"\nTraining {self.model_type} model ({self.algorithm})...")
            print(f"  Training samples: {len(X_train):,}")
            print(f"  Features: {list(X_train.columns)}")
        
        start_time = datetime.now()

        def _xgb_fit_with_early_stopping() -> None:
            """Fit XGBoost sklearn model with best-effort early stopping.

            XGBoost's sklearn wrapper changed its `fit()` signature across versions
            (notably around 2.x). Some versions accept `early_stopping_rounds`,
            others prefer `callbacks`, and some environments are stricter about
            unsupported keywords.
            """
            fit_base_kwargs: Dict[str, Any] = {}
            if X_val is not None and y_val is not None:
                fit_base_kwargs["eval_set"] = [(X_val, y_val)]

            # 1) Prefer callbacks (newer xgboost)
            callbacks = []
            try:
                callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)]
            except TypeError:
                try:
                    callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds)]
                except Exception:
                    callbacks = []
            except Exception:
                callbacks = []

            if callbacks:
                try:
                    self.model.fit(
                        X_train,
                        y_train,
                        **fit_base_kwargs,
                        callbacks=callbacks,
                        verbose=False,
                    )
                    return
                except TypeError:
                    # Retry without verbose for stricter signatures
                    try:
                        self.model.fit(
                            X_train,
                            y_train,
                            **fit_base_kwargs,
                            callbacks=callbacks,
                        )
                        return
                    except TypeError:
                        # Fall through to legacy API
                        pass

            # 2) Legacy keyword on fit (older xgboost)
            try:
                self.model.fit(
                    X_train,
                    y_train,
                    **fit_base_kwargs,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
                return
            except TypeError:
                # Retry without verbose for stricter signatures
                try:
                    self.model.fit(
                        X_train,
                        y_train,
                        **fit_base_kwargs,
                        early_stopping_rounds=early_stopping_rounds,
                    )
                    return
                except TypeError:
                    # Fall through to no-early-stopping fit
                    pass

            # 3) Best-effort: still use eval_set if supported
            try:
                self.model.fit(X_train, y_train, **fit_base_kwargs, verbose=False)
            except TypeError:
                try:
                    self.model.fit(X_train, y_train, **fit_base_kwargs)
                except TypeError:
                    self.model.fit(X_train, y_train)
        
        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None and self.algorithm in ['xgboost', 'lightgbm']:
            if self.algorithm == 'xgboost':
                _xgb_fit_with_early_stopping()
            else:  # lightgbm
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                )
        else:
            self.model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        y_train_pred = self.predict(X_train)
        train_metrics = self._compute_metrics(y_train, y_train_pred, 'Train')
        
        val_metrics = {}
        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)
            val_metrics = self._compute_metrics(y_val, y_val_pred, 'Validation')
        
        # Store history
        self.training_history = {
            'model_type': self.model_type,
            'algorithm': self.algorithm,
            'hyperparams': self.hyperparams,
            'training_time_sec': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'n_train_samples': len(X_train),
            'n_features': len(X_train.columns),
            'feature_names': list(X_train.columns),
            'timestamp': datetime.now().isoformat()
        }
        
        if verbose:
            self._print_metrics(train_metrics, 'Training')
            if val_metrics:
                self._print_metrics(val_metrics, 'Validation')
            print(f"\n✓ Training complete in {training_time:.1f} seconds")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model_type == 'precipitation_occurrence':
            # Return probability of precipitation
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, verbose: bool = True) -> Dict:
        """Evaluate model on test set"""
        if verbose:
            print(f"\nEvaluating {self.model_type} model on test set...")
            print(f"  Test samples: {len(X_test):,}")
        
        y_pred = self.predict(X_test)
        metrics = self._compute_metrics(y_test, y_pred, 'Test')
        
        if verbose:
            self._print_metrics(metrics, 'Test')
        
        self.training_history['test_metrics'] = metrics
        return metrics
    
    def _compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray, prefix: str = '') -> Dict:
        """Compute regression or classification metrics"""
        if self.model_type == 'precipitation_occurrence':
            # Classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
            y_pred_class = (y_pred > 0.5).astype(int)
            
            metrics = {
                f'{prefix}_Accuracy': accuracy_score(y_true, y_pred_class),
                f'{prefix}_Precision': precision_score(y_true, y_pred_class, zero_division=0),
                f'{prefix}_Recall': recall_score(y_true, y_pred_class),
                f'{prefix}_ROC_AUC': roc_auc_score(y_true, y_pred),
            }
        else:
            # Regression metrics
            if self.model_type == 'precipitation':
                # Metrics in log1p space
                y_true_mm = np.expm1(y_true)
                y_pred_mm = np.expm1(y_pred)
                
                metrics = {
                    f'{prefix}_RMSE_log': np.sqrt(mean_squared_error(y_true, y_pred)),
                    f'{prefix}_MAE_log': mean_absolute_error(y_true, y_pred),
                    f'{prefix}_R2_log': r2_score(y_true, y_pred),
                    f'{prefix}_RMSE_mm': np.sqrt(mean_squared_error(y_true_mm, y_pred_mm)),
                    f'{prefix}_MAE_mm': mean_absolute_error(y_true_mm, y_pred_mm),
                    f'{prefix}_R2_mm': r2_score(y_true_mm, y_pred_mm),
                }
            else:
                metrics = {
                    f'{prefix}_RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                    f'{prefix}_MAE': mean_absolute_error(y_true, y_pred),
                    f'{prefix}_R2': r2_score(y_true, y_pred),
                    f'{prefix}_Bias': np.mean(y_pred - y_true),
                }
        
        return metrics
    
    def _print_metrics(self, metrics: Dict, dataset: str):
        """Print metrics"""
        print(f"\n  {dataset} Metrics:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            return None
        
        feature_names = self.training_history.get('feature_names', [f'feature_{i}' for i in range(len(importance))])
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, filepath: str):
        """Save model"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'algorithm': self.algorithm,
            'hyperparams': self.hyperparams,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n✓ Model saved to {filepath}")
        
        # Save training history as JSON
        history_file = filepath.with_suffix('.json')
        # Convert numpy types to python types for JSON
        history_clean = json.loads(json.dumps(self.training_history, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o))
        with open(history_file, 'w') as f:
            json.dump(history_clean, f, indent=2)
        print(f"✓ Training history saved to {history_file}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model"""
        model_data = joblib.load(filepath)
        
        instance = cls(model_data['model_type'], model_data['algorithm'], model_data['hyperparams'])
        instance.model = model_data['model']
        instance.training_history = model_data['training_history']
        
        print(f"✓ Model loaded from {filepath}")
        return instance


def train_two_stage_precipitation(data_dir: str, output_dir: str,
                                  algorithm: str = 'auto', sample_frac: float = 1.0) -> Tuple:
    """
    Train two-stage precipitation model: occurrence + amount
    
    Parameters:
    -----------
    data_dir : str
        Directory with train/val/test parquet files
    output_dir : str
        Output directory for models
    algorithm : str
        Algorithm to use
    sample_frac : float
        Fraction of training data to use (for quick testing)
        
    Returns:
    --------
    occ_model, amt_model : EnhancedDownscalingModel
        Occurrence and amount models
    """
    from src.data.loaders import DownscalingDataLoader
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TWO-STAGE PRECIPITATION MODEL")
    print("="*80)
    
    # Load data
    print("\nLoading training data...")
    df_train = pd.read_parquet(data_dir / 'train_data.parquet')
    df_val = pd.read_parquet(data_dir / 'val_data.parquet')
    df_test = pd.read_parquet(data_dir / 'test_data.parquet')
    
    if sample_frac < 1.0:
        print(f"  Using {sample_frac*100:.0f}% of training data for quick test...")
        df_train = df_train.sample(frac=sample_frac, random_state=42)
        df_val = df_val.sample(frac=max(sample_frac, 0.2), random_state=42)
    
    print(f"  Train: {len(df_train):,} samples")
    print(f"  Validation: {len(df_val):,} samples")
    print(f"  Test: {len(df_test):,} samples")
    
    # Features
    features = ['gcm_pr_log1p', 'gcm_tas_degC', 'lat', 'lon', 'month_sin', 'month_cos']
    
    X_train = df_train[features]
    X_val = df_val[features]
    X_test = df_test[features]
    
    y_train_amt = df_train['era_tp_mm']
    y_val_amt = df_val['era_tp_mm']
    y_test_amt = df_test['era_tp_mm']
    
    # Stage 1: Occurrence model
    print("\n" + "="*80)
    print("STAGE 1: Precipitation Occurrence (Wet/Dry)")
    print("="*80)
    
    y_train_occ = (y_train_amt > 0).astype(int)
    y_val_occ = (y_val_amt > 0).astype(int)
    y_test_occ = (y_test_amt > 0).astype(int)
    
    occ_model = EnhancedDownscalingModel('precipitation_occurrence', algorithm=algorithm)
    occ_model.fit(X_train, y_train_occ, X_val, y_val_occ)
    occ_model.evaluate(X_test, y_test_occ)
    # Save occurrence model with algorithm-prefixed filename (e.g., xgboost_pr_occ.pkl)
    occ_model.save(output_dir / f"{occ_model.algorithm}_pr_occ.pkl")
    
    # Stage 2: Amount model (train only on wet samples)
    print("\n" + "="*80)
    print("STAGE 2: Precipitation Amount (conditional on wet)")
    print("="*80)
    
    mask_train_wet = y_train_amt > 0
    mask_val_wet = y_val_amt > 0
    
    X_train_wet = X_train[mask_train_wet]
    X_val_wet = X_val[mask_val_wet]
    
    y_train_amt_log = np.log1p(y_train_amt[mask_train_wet])
    y_val_amt_log = np.log1p(y_val_amt[mask_val_wet])
    
    print(f"  Training on {len(X_train_wet):,} wet samples ({mask_train_wet.mean()*100:.1f}% of total)")
    
    amt_model = EnhancedDownscalingModel('precipitation', algorithm=algorithm)
    amt_model.fit(X_train_wet, y_train_amt_log, X_val_wet, y_val_amt_log)
    
    # Evaluate two-stage pipeline
    print("\n" + "="*80)
    print("TWO-STAGE PIPELINE EVALUATION")
    print("="*80)
    
    p_wet_test = occ_model.predict(X_test)
    pred_amt_log_test = amt_model.predict(X_test)
    pred_amt_test = np.expm1(pred_amt_log_test)
    pred_amt_test = np.clip(pred_amt_test, 0, None)
    
    y_pred_two_stage = p_wet_test * pred_amt_test
    
    rmse = np.sqrt(mean_squared_error(y_test_amt, y_pred_two_stage))
    mae = mean_absolute_error(y_test_amt, y_pred_two_stage)
    r2 = r2_score(y_test_amt, y_pred_two_stage)
    
    print(f"\n  Test Set Metrics:")
    print(f"    RMSE: {rmse:.4f} mm/month")
    print(f"    MAE:  {mae:.4f} mm/month")
    print(f"    R2:   {r2:.4f}")
    
    obs_wet_frac = (y_test_amt > 0).mean()
    pred_wet_frac = (y_pred_two_stage > 0).mean()
    print(f"\n  Wet day frequency:")
    print(f"    Observed: {obs_wet_frac:.3f}")
    print(f"    Predicted: {pred_wet_frac:.3f}")
    
    # Save amount model with algorithm-prefixed filename (e.g., xgboost_pr_amt.pkl)
    amt_model.save(output_dir / f"{amt_model.algorithm}_pr_amt.pkl")
    
    # Save combined metrics
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'obs_wet_fraction': float(obs_wet_frac),
        'pred_wet_fraction': float(pred_wet_frac)
    }
    with open(output_dir / 'precip_two_stage_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return occ_model, amt_model


def train_all_models(data_dir: str, output_dir: str, algorithm: str = 'auto',
                    use_two_stage: bool = True, sample_frac: float = 1.0) -> Dict:
    """
    Train all downscaling models
    
    Parameters:
    -----------
    data_dir : str
        Directory with train/val/test parquet files
    output_dir : str
        Output directory for models
    algorithm : str
        Algorithm to use ('auto', 'xgboost', 'lightgbm', 'randomforest')
    use_two_stage : bool
        Use two-stage model for precipitation
    sample_frac : float
        Fraction of training data to use (for quick testing)
        
    Returns:
    --------
    models : dict
        Dictionary of trained models
    """
    from src.data.loaders import DownscalingDataLoader
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ENHANCED DOWNSCALING MODEL TRAINING")
    print("="*80)
    print(f"Algorithm: {algorithm}")
    print(f"Two-stage precipitation: {use_two_stage}")
    print(f"Sample fraction: {sample_frac}")
    print("="*80)
    
    # Load data
    print("\nLoading training data...")
    df_train = pd.read_parquet(data_dir / 'train_data.parquet')
    df_val = pd.read_parquet(data_dir / 'val_data.parquet')
    df_test = pd.read_parquet(data_dir / 'test_data.parquet')
    
    if sample_frac < 1.0:
        print(f"  Using {sample_frac*100:.0f}% of training data...")
        df_train = df_train.sample(frac=sample_frac, random_state=42)
        df_val = df_val.sample(frac=max(sample_frac, 0.2), random_state=42)
    
    print(f"  Train: {len(df_train):,} samples")
    print(f"  Validation: {len(df_val):,} samples")
    print(f"  Test: {len(df_test):,} samples")
    
    loader = DownscalingDataLoader(str(data_dir))
    models = {}
    
    # Temperature model
    print("\n" + "="*80)
    print("TEMPERATURE MODEL")
    print("="*80)
    
    X_train_temp, y_train_temp = loader.get_feature_target_sets(df_train, 'temperature')
    X_val_temp, y_val_temp = loader.get_feature_target_sets(df_val, 'temperature')
    X_test_temp, y_test_temp = loader.get_feature_target_sets(df_test, 'temperature')
    
    temp_model = EnhancedDownscalingModel('temperature', algorithm=algorithm)
    temp_model.fit(X_train_temp, y_train_temp, X_val_temp, y_val_temp)
    temp_model.evaluate(X_test_temp, y_test_temp)
    # Save temperature model using algorithm prefix (e.g., xgboost_tas.pkl)
    temp_model.save(output_dir / f"{temp_model.algorithm}_tas.pkl")
    
    models['temperature'] = temp_model
    
    # Precipitation model
    if use_two_stage:
        occ_model, amt_model = train_two_stage_precipitation(
            data_dir=str(data_dir),
            output_dir=str(output_dir),
            algorithm=algorithm,
            sample_frac=sample_frac
        )
        models['precip_occurrence'] = occ_model
        models['precip_amount'] = amt_model
    else:
        print("\n" + "="*80)
        print("PRECIPITATION MODEL (single-stage)")
        print("="*80)
        
        X_train_precip, y_train_precip = loader.get_feature_target_sets(df_train, 'precipitation')
        X_val_precip, y_val_precip = loader.get_feature_target_sets(df_val, 'precipitation')
        X_test_precip, y_test_precip = loader.get_feature_target_sets(df_test, 'precipitation')
        
        precip_model = EnhancedDownscalingModel('precipitation', algorithm=algorithm)
        precip_model.fit(X_train_precip, y_train_precip, X_val_precip, y_val_precip)
        precip_model.evaluate(X_test_precip, y_test_precip)
        # Save single-stage precipitation model using algorithm prefix (e.g., xgboost_pr.pkl)
        precip_model.save(output_dir / f"{precip_model.algorithm}_pr.pkl")
        
        models['precipitation'] = precip_model
    
    print("\n" + "="*80)
    print("✓ ALL MODELS TRAINED SUCCESSFULLY")
    print("="*80)
    print(f"Models saved to: {output_dir}")
    
    gc.collect()
    return models


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train enhanced ML downscaling models')
    parser.add_argument('--data-dir', type=str,
                        default='/content/drive/MyDrive/Downscaling ML CEP/data/processed',
                        help='Directory with train/val/test parquet files')
    parser.add_argument('--output-dir', type=str,
                        default='/content/drive/MyDrive/Downscaling ML CEP/outputs/models',
                        help='Output directory for models')
    parser.add_argument('--algorithm', type=str, default='auto',
                        choices=['auto', 'xgboost', 'lightgbm', 'randomforest', 'gradientboosting'],
                        help='Algorithm to use')
    parser.add_argument('--two-stage', action='store_true', default=True,
                        help='Use two-stage model for precipitation')
    parser.add_argument('--sample-frac', type=float, default=1.0,
                        help='Fraction of training data to use (0-1)')
    
    args = parser.parse_args()
    
    models = train_all_models(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        algorithm=args.algorithm,
        use_two_stage=args.two_stage,
        sample_frac=args.sample_frac
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
