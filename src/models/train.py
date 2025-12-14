"""
Machine Learning models for climate downscaling

This module automatically uses the enhanced version (train_v2) if available,
with fallback to the original implementation for backward compatibility.
"""

# Try to import enhanced version first
try:
    from .train_v2 import (
        EnhancedDownscalingModel as DownscalingModel,
        train_all_models as train_both_models,
        train_two_stage_precipitation
    )
    print("✓ Using enhanced models from train_v2")
except (ImportError, ModuleNotFoundError):
    print("⚠ Enhanced version not available, loading standard models")
    
    # Import dependencies for fallback implementation
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from typing import Tuple, Dict, Optional
    import joblib
    import json
    from datetime import datetime

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import GridSearchCV

    import matplotlib.pyplot as plt
    import seaborn as sns
    import seaborn as sns

    class DownscalingModel:
        """Base class for downscaling models"""
        
        def __init__(self, model_type: str, hyperparams: Optional[Dict] = None):
            """
            Initialize downscaling model
            
            Parameters:
            -----------
            model_type : str
                'temperature' or 'precipitation'
            hyperparams : dict, optional
                Model hyperparameters
            """
            self.model_type = model_type
            self.hyperparams = hyperparams or self._get_default_hyperparams()
            self.model = self._create_model()
            self.training_history = {}
            
        def _get_default_hyperparams(self) -> Dict:
            """Get default hyperparameters for model type"""
            if self.model_type == 'temperature':
                return {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'n_jobs': -1,
                    'random_state': 42,
                    'verbose': 1
                }
            elif self.model_type == 'precipitation':
                return {
                    'n_estimators': 300,
                    'learning_rate': 0.05,
                    'max_depth': 3,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'subsample': 0.8,
                    'random_state': 42,
                    'verbose': 1
                }
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
        
        def _create_model(self):
            """Create sklearn model instance"""
            if self.model_type == 'temperature':
                return RandomForestRegressor(**self.hyperparams)
            elif self.model_type == 'precipitation':
                return GradientBoostingRegressor(**self.hyperparams)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
        
        def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
                X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
            """
            Train the model
            
            Parameters:
            -----------
            X_train : pd.DataFrame
                Training features
            y_train : pd.Series
                Training targets
            X_val : pd.DataFrame, optional
                Validation features
            y_val : pd.Series, optional
                Validation targets
            """
            print(f"\nTraining {self.model_type} model...")
            print(f"  Training samples: {len(X_train)}")
            print(f"  Features: {list(X_train.columns)}")
            print(f"  Hyperparameters: {self.hyperparams}")
            
            # Record training start
            start_time = datetime.now()
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Record training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate on training set
            y_train_pred = self.model.predict(X_train)
            train_metrics = self._compute_metrics(y_train, y_train_pred, 'Train')
            
            # Evaluate on validation set if provided
            val_metrics = {}
            if X_val is not None and y_val is not None:
                y_val_pred = self.model.predict(X_val)
                val_metrics = self._compute_metrics(y_val, y_val_pred, 'Validation')
            
            # Store history
            self.training_history = {
                'model_type': self.model_type,
                'hyperparams': self.hyperparams,
                'training_time_sec': training_time,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'n_train_samples': len(X_train),
                'n_features': len(X_train.columns),
                'feature_names': list(X_train.columns),
                'timestamp': datetime.now().isoformat()
            }
            
            # Print results
            self._print_metrics(train_metrics, 'Training')
            if val_metrics:
                self._print_metrics(val_metrics, 'Validation')
            
            print(f"\n✓ Training complete in {training_time:.1f} seconds")
            
            return self
        
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            """
            Make predictions
            
            Parameters:
            -----------
            X : pd.DataFrame
                Features
                
            Returns:
            --------
            y_pred : np.ndarray
                Predictions
            """
            return self.model.predict(X)
        
        def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
            """
            Evaluate model on test set
            
            Parameters:
            -----------
            X_test : pd.DataFrame
                Test features
            y_test : pd.Series
                Test targets
                
            Returns:
            --------
            metrics : dict
                Test metrics
            """
            print(f"\nEvaluating {self.model_type} model on test set...")
            print(f"  Test samples: {len(X_test)}")
            
            y_pred = self.predict(X_test)
            metrics = self._compute_metrics(y_test, y_pred, 'Test')
            
            self._print_metrics(metrics, 'Test')
            
            # Store test metrics
            self.training_history['test_metrics'] = metrics
            
            return metrics
        
        def _compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray, prefix: str = '') -> Dict:
            """Compute regression metrics"""
            # For precipitation, need to inverse log transform
            if self.model_type == 'precipitation':
                # y_true and y_pred are in log1p space
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
                # Temperature in original space
                metrics = {
                    f'{prefix}_RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                    f'{prefix}_MAE': mean_absolute_error(y_true, y_pred),
                    f'{prefix}_R2': r2_score(y_true, y_pred),
                }
            
            return metrics
        
        def _print_metrics(self, metrics: Dict, dataset: str):
            """Print metrics in formatted way"""
            print(f"\n  {dataset} Metrics:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
        
        def get_feature_importance(self) -> pd.DataFrame:
            """
            Get feature importance
            
            Returns:
            --------
            importance_df : pd.DataFrame
                Feature importance scores
            """
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                feature_names = self.training_history.get('feature_names', [f'feature_{i}' for i in range(len(importance))])
                
                df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                return df
            else:
                return None
        
        def plot_feature_importance(self, save_path: Optional[str] = None):
            """Plot feature importance"""
            importance_df = self.get_feature_importance()
            
            if importance_df is None:
                print("Feature importance not available for this model")
                return
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
            plt.title(f'Feature Importance - {self.model_type.capitalize()} Model')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"  Saved to {save_path}")
            
            plt.show()
        
        def save(self, filepath: str):
            """
            Save model to disk
            
            Parameters:
            -----------
            filepath : str
                Output file path
            """
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'hyperparams': self.hyperparams,
                'training_history': self.training_history
            }
            
            joblib.dump(model_data, filepath)
            print(f"\n✓ Model saved to {filepath}")
            
            # Save training history as JSON
            history_file = filepath.with_suffix('.json')
            with open(history_file, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            print(f"✓ Training history saved to {history_file}")
        
        @classmethod
        def load(cls, filepath: str):
            """
            Load model from disk
            
            Parameters:
            -----------
            filepath : str
                Model file path
                
            Returns:
            --------
            model : DownscalingModel
                Loaded model
            """
            model_data = joblib.load(filepath)
            
            instance = cls(model_data['model_type'], model_data['hyperparams'])
            instance.model = model_data['model']
            instance.training_history = model_data['training_history']
            
            print(f"✓ Model loaded from {filepath}")
            return instance
        
    def train_both_models(data_dir: str, output_dir: str, 
                         tune_hyperparams: bool = False) -> Tuple[DownscalingModel, DownscalingModel]:
        """
        Train both temperature and precipitation models
        
        Parameters:
        -----------
        data_dir : str
            Directory with train/val/test parquet files
        output_dir : str
            Output directory for models
        tune_hyperparams : bool
            Whether to tune hyperparameters using validation set
            
        Returns:
        --------
        temp_model, precip_model : DownscalingModel
            Trained models
        """
        from src.data.loaders import DownscalingDataLoader
        
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("TRAINING DOWNSCALING MODELS")
        print("="*80)
        
        # Load data
        print("\nLoading training data...")
        df_train = pd.read_parquet(data_dir / 'train_data.parquet')
        df_val = pd.read_parquet(data_dir / 'val_data.parquet')
        df_test = pd.read_parquet(data_dir / 'test_data.parquet')
        
        print(f"  Train: {len(df_train)} samples")
        print(f"  Validation: {len(df_val)} samples")
        print(f"  Test: {len(df_test)} samples")
        
        # Initialize loader for feature extraction
        loader = DownscalingDataLoader(str(data_dir))
        
        # ========================
        # TEMPERATURE MODEL
        # ========================
        print("\n" + "="*80)
        print("TEMPERATURE MODEL (RandomForest)")
        print("="*80)
        
        X_train_temp, y_train_temp = loader.get_feature_target_sets(df_train, 'temperature')
        X_val_temp, y_val_temp = loader.get_feature_target_sets(df_val, 'temperature')
        X_test_temp, y_test_temp = loader.get_feature_target_sets(df_test, 'temperature')
        
        temp_model = DownscalingModel('temperature')
        temp_model.fit(X_train_temp, y_train_temp, X_val_temp, y_val_temp)
        temp_model.evaluate(X_test_temp, y_test_temp)
        
        # Feature importance
        print("\nFeature Importance (Temperature):")
        print(temp_model.get_feature_importance())
        temp_model.plot_feature_importance(output_dir / '../figures/temperature_feature_importance.png')
        
        # Save model (algorithm-prefixed name)
        temp_model.save(output_dir / 'xgb_tas.pkl')
        
        # ========================
        # PRECIPITATION MODEL
        # ========================
        print("\n" + "="*80)
        print("PRECIPITATION MODEL (GradientBoosting)")
        print("="*80)
        
        X_train_precip, y_train_precip = loader.get_feature_target_sets(df_train, 'precipitation')
        X_val_precip, y_val_precip = loader.get_feature_target_sets(df_val, 'precipitation')
        X_test_precip, y_test_precip = loader.get_feature_target_sets(df_test, 'precipitation')
        
        precip_model = DownscalingModel('precipitation')
        precip_model.fit(X_train_precip, y_train_precip, X_val_precip, y_val_precip)
        precip_model.evaluate(X_test_precip, y_test_precip)
        
        # Feature importance
        print("\nFeature Importance (Precipitation):")
        print(precip_model.get_feature_importance())
        precip_model.plot_feature_importance(output_dir / '../figures/precipitation_feature_importance.png')
        
        # Save model (algorithm-prefixed name)
        precip_model.save(output_dir / 'xgb_pr.pkl')
        
        # ========================
        # SUMMARY
        # ========================
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print("\nTemperature Model:")
        temp_model._print_metrics(temp_model.training_history['test_metrics'], 'Test')
        
        print("\nPrecipitation Model:")
        precip_model._print_metrics(precip_model.training_history['test_metrics'], 'Test')
        
        print(f"\n✓ Models saved to {output_dir}")
        
        return temp_model, precip_model
        
    def main():
        """Main training script"""
        import argparse
        
        parser = argparse.ArgumentParser(description='Train ML downscaling models')
        parser.add_argument('--data-dir', type=str,
                            default=r'd:\\appdev\\cep ml\\data\\processed',
                            help='Directory with train/val/test parquet files')
        parser.add_argument('--output-dir', type=str,
                            default=r'd:\\appdev\\cep ml\\outputs\\models',
                            help='Output directory for models')
        parser.add_argument('--tune', action='store_true',
                            help='Tune hyperparameters using validation set')
        
        args = parser.parse_args()
        
        # Train models
        temp_model, precip_model = train_both_models(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            tune_hyperparams=args.tune
        )
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("  1. Review model performance metrics")
        print("  2. Evaluate spatial patterns (notebooks/04_evaluation.ipynb)")
        print("  3. Apply to future scenarios (python src/inference/downscale_future.py)")
        
    if __name__ == '__main__':
        main()
