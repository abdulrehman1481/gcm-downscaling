"""
Data loaders and feature engineering for ML downscaling
- Flatten 3D fields to tabular format
- Merge datasets
- Add temporal features
- Train/validation/test split
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import warnings


class DownscalingDataLoader:
    """Load and prepare training data for ML downscaling models"""
    
    def __init__(self, processed_data_dir: str):
        """
        Initialize data loader
        
        Parameters:
        -----------
        processed_data_dir : str
            Path to processed NetCDF files
        """
        self.data_dir = Path(processed_data_dir)
        
    def load_processed_data(self, gcm_model: str = 'BCC-CSM2-MR') -> dict:
        """
        Load all processed NetCDF files
        
        Parameters:
        -----------
        gcm_model : str
            GCM model name
            
        Returns:
        --------
        data_dict : dict
            Dictionary with all loaded data arrays
        """
        print(f"Loading processed data from {self.data_dir}")

        data = {}

        # Helper to find the first existing candidate file
        def _find_file(candidates):
            for c in candidates:
                p = self.data_dir / c
                if p.exists():
                    return p
            return None

        # Candidates for CRU/ERA5 filenames (support older and newer names)
        cru_candidates = [
            'cru_1980_2014.nc',
            'cru_tmp_1980_2014_processed.nc',
            'cru_tmp_1980_2014.nc'
        ]
        era5_candidates = [
            'era5_1980_2014.nc',
            'era5_t2m_1980_2014_processed.nc',
            'era5_t2m_1980_2014.nc'
        ]

        cru_file = _find_file(cru_candidates)
        era5_file = _find_file(era5_candidates)

        if cru_file is None:
            raise FileNotFoundError(f"No CRU processed file found in {self.data_dir}. Checked: {cru_candidates}")
        if era5_file is None:
            raise FileNotFoundError(f"No ERA5 processed file found in {self.data_dir}. Checked: {era5_candidates}")

        # GCM filename candidates - support a few naming variants
        gcm_candidates = [
            f'{gcm_model}_hist_1980_2014.nc',
            f'{gcm_model}_hist_tas_1980_2014_processed.nc',
            f'{gcm_model}_hist_tas_1980_2014.nc',
            f'{gcm_model}_hist.nc'
        ]
        gcm_file = _find_file(gcm_candidates)
        if gcm_file is None:
            # List available GCM files to help user debug
            available = sorted([p.name for p in self.data_dir.glob('*_hist*.nc')])
            raise FileNotFoundError(f"No processed GCM file found for model '{gcm_model}'.\nSearched: {gcm_candidates}\nAvailable: {available}")

        # Open datasets
        cru_ds = xr.open_dataset(cru_file)
        era5_ds = xr.open_dataset(era5_file)
        gcm_ds = xr.open_dataset(gcm_file)

        # Extract variables (robust: try common names)
        def _get_var(ds, names):
            for n in names:
                if n in ds:
                    return ds[n]
            raise KeyError(f"None of {names} found in {ds}")

        data['cru_tmp'] = _get_var(cru_ds, ['tmp', 'tas', 'temperature'])
        data['cru_pre'] = _get_var(cru_ds, ['pre', 'pr', 'precipitation'])
        data['era5_t2m'] = _get_var(era5_ds, ['t2m', 'temperature', 'tas'])
        data['era5_tp'] = _get_var(era5_ds, ['tp', 'pr', 'precipitation'])
        data['gcm_tas'] = _get_var(gcm_ds, ['tas', 'TMP', 'tas_monthly', 'temperature'])
        data['gcm_pr'] = _get_var(gcm_ds, ['pr', 'PRE', 'precipitation'])
        
        print(f"  Loaded {len(data)} variables")
        for name, arr in data.items():
            print(f"    {name}: {arr.shape}")
        
        return data
    
    def create_training_dataframe(self, gcm_model: str = 'BCC-CSM2-MR',
                                   include_cru: bool = True) -> pd.DataFrame:
        """
        Create flattened training DataFrame
        
        Parameters:
        -----------
        gcm_model : str
            GCM model name
        include_cru : bool
            Whether to include CRU as additional predictors
            
        Returns:
        --------
        df : pd.DataFrame
            Flattened training data with all features and targets
        """
        print("\nCreating training DataFrame...")
        
        # Load data
        data = self.load_processed_data(gcm_model)
        
        # Flatten each 3D array to DataFrame
        dfs = []
        
        for name, arr in data.items():
            print(f"  Flattening {name}...")
            df_flat = self._flatten_to_dataframe(arr, name)
            dfs.append(df_flat)
        
        # Merge all on (time, lat, lon)
        print("  Merging datasets on (time, lat, lon) ...")
        df = dfs[0]
        for df_i in dfs[1:]:
            df = df.merge(df_i, on=['time', 'lat', 'lon'], how='inner')

        print(f"  Merged shape: {df.shape}")

        # If merge produced 0 rows, try a more permissive merge on (year, month, lat, lon)
        if len(df) == 0:
            warnings.warn(
                "Merged DataFrame is empty after merging on exact 'time'. "
                "This often happens when datasets use different time encodings (e.g. period start vs end, cftime). "
                "Attempting fallback merge on (year, month, lat, lon)."
            )

            # Ensure each flattened df has year/month columns
            dfs_by_ym = []
            for df_i in dfs:
                df_copy = df_i.copy()
                # Convert time to datetime where possible
                try:
                    df_copy['time'] = pd.to_datetime(df_copy['time'], errors='coerce')
                except Exception:
                    df_copy['time'] = pd.to_datetime(df_copy['time'].astype(str), errors='coerce')

                # Create year/month
                df_copy['year'] = df_copy['time'].dt.year
                df_copy['month'] = df_copy['time'].dt.month
                # Drop original 'time' to avoid duplicate/time-suffix merge errors
                df_copy = df_copy.drop(columns=['time'], errors='ignore')
                dfs_by_ym.append(df_copy)

            # Merge on year, month, lat, lon
            df_ym = dfs_by_ym[0]
            for df_i in dfs_by_ym[1:]:
                df_ym = df_ym.merge(df_i, on=['year', 'month', 'lat', 'lon'], how='inner')

            print(f"  Fallback merged shape (year,month,lat,lon): {df_ym.shape}")

            if len(df_ym) > 0:
                # If successful, use this DataFrame and derive a canonical 'time' column
                df = df_ym
                # Create a representative time (first day of month)
                df['time'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
                print("  Successfully merged on year/month; created canonical 'time' as first-of-month")
            else:
                warnings.warn(
                    "Fallback merge on (year, month, lat, lon) also produced 0 rows. "
                    "Please inspect processed NetCDF time coordinates for inconsistencies."
                )
        
        # Add temporal features
        print("  Adding temporal features...")
        df = self._add_temporal_features(df)
        
        # Add log-transformed precipitation
        print("  Adding log-transformed precipitation...")
        df['era_tp_log1p'] = np.log1p(df['era5_tp'])
        df['gcm_pr_log1p'] = np.log1p(df['gcm_pr'])
        
        # Rename columns for clarity
        df = df.rename(columns={
            'cru_tmp': 'cru_tmp_degC',
            'cru_pre': 'cru_pre_mm',
            'era5_t2m': 'era_t2m_degC',
            'era5_tp': 'era_tp_mm',
            'gcm_tas': 'gcm_tas_degC',
            'gcm_pr': 'gcm_pr_mm'
        })
        
        # Drop rows with NaNs in core predictors/targets
        print("  Checking for missing values...")
        core_cols = ['gcm_tas_degC', 'gcm_pr_mm', 'era_t2m_degC', 'era_tp_mm']
        n_before = len(df)
        df = df.dropna(subset=core_cols)
        n_after = len(df)
        
        if n_before > n_after:
            print(f"  Dropped {n_before - n_after} rows with NaNs ({100*(n_before-n_after)/n_before:.2f}%)")
        else:
            print(f"  No missing values in core columns")
        
        print(f"\n✓ Final DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        return df
    
    def _flatten_to_dataframe(self, arr: xr.DataArray, var_name: str) -> pd.DataFrame:
        """
        Flatten 3D xarray DataArray to DataFrame
        
        Parameters:
        -----------
        arr : xr.DataArray
            3D array with dimensions (time, lat, lon)
        var_name : str
            Variable name
            
        Returns:
        --------
        df : pd.DataFrame
            Flattened DataFrame with columns: time, lat, lon, var_name
        """
        # Stack spatial dimensions
        arr_stacked = arr.stack(grid_point=['lat', 'lon'])
        
        # Convert to DataFrame
        df = arr_stacked.to_dataframe(name=var_name).reset_index()
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features for seasonality
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with 'time' column
            
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with added temporal features
        """
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        # Extract year and month
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        
        # Cyclic encoding of month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def train_val_test_split(self, df: pd.DataFrame,
                             train_years: Tuple[int, int] = (1980, 2005),
                             val_years: Tuple[int, int] = (2006, 2010),
                             test_years: Tuple[int, int] = (2011, 2014)) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by year into train/validation/test sets
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset
        train_years : tuple
            (start, end) years for training
        val_years : tuple
            (start, end) years for validation
        test_years : tuple
            (start, end) years for testing
            
        Returns:
        --------
        df_train, df_val, df_test : pd.DataFrame
            Split datasets
        """
        print("\nSplitting data by year...")
        
        # Ensure year column exists
        if 'year' not in df.columns:
            df = self._add_temporal_features(df)
        
        # Split by year
        df_train = df[(df['year'] >= train_years[0]) & (df['year'] <= train_years[1])].copy()
        df_val = df[(df['year'] >= val_years[0]) & (df['year'] <= val_years[1])].copy()
        df_test = df[(df['year'] >= test_years[0]) & (df['year'] <= test_years[1])].copy()
        
        print(f"  Train: {len(df_train)} samples ({train_years[0]}-{train_years[1]})")
        print(f"  Validation: {len(df_val)} samples ({val_years[0]}-{val_years[1]})")
        print(f"  Test: {len(df_test)} samples ({test_years[0]}-{test_years[1]})")
        
        return df_train, df_val, df_test
    
    def get_feature_target_sets(self, df: pd.DataFrame,
                                model_type: str = 'temperature') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract feature and target sets for specific model
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        model_type : str
            'temperature' or 'precipitation'
            
        Returns:
        --------
        X, y : pd.DataFrame, pd.Series
            Features and target
        """
        if model_type == 'temperature':
            # Temperature model features
            feature_cols = [
                'gcm_tas_degC',
                'gcm_pr_mm',
                'lat',
                'lon',
                'month_sin',
                'month_cos'
            ]
            target_col = 'era_t2m_degC'
            
        elif model_type == 'precipitation':
            # Precipitation model features
            feature_cols = [
                'gcm_pr_log1p',
                'gcm_tas_degC',
                'lat',
                'lon',
                'month_sin',
                'month_cos'
            ]
            target_col = 'era_tp_log1p'
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        return X, y
    
    def save_to_parquet(self, df: pd.DataFrame, output_path: str, name: str = 'training_data'):
        """
        Save DataFrame to parquet format
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to save
        output_path : str
            Output directory
        name : str
            File name prefix
        """
        output_file = Path(output_path) / f'{name}.parquet'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_file, index=False, compression='snappy')
        print(f"\n✓ Saved to {output_file}")
        print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    def prepare_future_data(self, gcm_model: str, scenario: str,
                           processed_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare future scenario data for inference
        
        Parameters:
        -----------
        gcm_model : str
            GCM model name
        scenario : str
            Scenario name ('ssp126', 'ssp585')
        processed_dir : str, optional
            Directory with processed future data
            
        Returns:
        --------
        df : pd.DataFrame
            DataFrame ready for inference
        """
        print(f"\nPreparing future data: {gcm_model} {scenario}")
        
        if processed_dir is None:
            processed_dir = self.data_dir.parent / 'future'
        else:
            processed_dir = Path(processed_dir)
        
        # Load future GCM data (assumed to be pre-processed)
        tas_file = processed_dir / f'{gcm_model}_{scenario}_tas_processed.nc'
        pr_file = processed_dir / f'{gcm_model}_{scenario}_pr_processed.nc'
        
        if not tas_file.exists() or not pr_file.exists():
            raise FileNotFoundError(
                f"Processed future data not found. Please run preprocessing first.\n"
                f"Expected: {tas_file} and {pr_file}"
            )
        
        tas = xr.open_dataarray(tas_file)
        pr = xr.open_dataarray(pr_file)
        
        # Flatten
        df_tas = self._flatten_to_dataframe(tas, 'gcm_tas_degC')
        df_pr = self._flatten_to_dataframe(pr, 'gcm_pr_mm')
        
        # Merge
        df = df_tas.merge(df_pr, on=['time', 'lat', 'lon'])
        
        # Add features
        df = self._add_temporal_features(df)
        df['gcm_pr_log1p'] = np.log1p(df['gcm_pr_mm'])
        
        print(f"  Future data shape: {df.shape}")
        print(f"  Time range: {df['time'].min()} to {df['time'].max()}")
        
        return df


def main():
    """Main data loading and preparation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare training data for ML downscaling')
    parser.add_argument('--processed-dir', type=str, 
                        default=r'd:\appdev\cep ml\data\processed\train',
                        help='Directory with processed NetCDF files')
    parser.add_argument('--output-dir', type=str,
                        default=r'd:\appdev\cep ml\data\processed',
                        help='Output directory for parquet files')
    parser.add_argument('--gcm-model', type=str, default='BCC-CSM2-MR',
                        help='GCM model name')
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = DownscalingDataLoader(args.processed_dir)
    
    # Create full DataFrame
    df_full = loader.create_training_dataframe(gcm_model=args.gcm_model)
    
    # Split into train/val/test
    df_train, df_val, df_test = loader.train_val_test_split(df_full)
    
    # Save to parquet
    output_path = Path(args.output_dir)
    loader.save_to_parquet(df_full, output_path, 'full_data')
    loader.save_to_parquet(df_train, output_path, 'train_data')
    loader.save_to_parquet(df_val, output_path, 'val_data')
    loader.save_to_parquet(df_test, output_path, 'test_data')
    
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"Files saved to: {output_path}")
    print("\nYou can now train models using:")
    print("  python src/models/train.py")


if __name__ == '__main__':
    main()
