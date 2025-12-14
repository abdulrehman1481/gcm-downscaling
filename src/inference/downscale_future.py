"""
Apply trained models to future SSP scenarios
- Process future GCM data
- Generate predictions
- Save downscaled outputs as NetCDF
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.train import DownscalingModel
from src.data.preprocessors import ClimateDataPreprocessor
from src.data.loaders import DownscalingDataLoader


class FutureDownscaler:
    """Apply downscaling models to future scenarios

    This class can accept a `models_path` directory and will discover the
    temperature and precipitation model files automatically. This makes the
    notebook/user workflow simpler: provide `models_path` and the downscaler
    will pick the best-matching files.
    """

    def __init__(self, models_path: str = None, temp_model_path: str = None,
                 precip_model_path: str = None, base_data_path: str = None):
        """
        Initialize future downscaler.

        Parameters
        ----------
        models_path : str, optional
            Directory containing model files (preferred). If provided, the
            downscaler will search for files matching '*_tas.pkl' and
            '*_pr*.pkl' (including two-stage names) and load them.
        temp_model_path : str, optional
            Explicit path to temperature model (used if `models_path` not set).
        precip_model_path : str, optional
            Explicit path to precipitation model (used if `models_path` not set).
        base_data_path : str, optional
            Path to AI_GCMs directory (used to locate GCM inputs). If not
            provided, calling code must set `base_data_path` when invoking
            processing methods.
        """
        # Resolve base paths
        self.models_path = Path(models_path) if models_path is not None else None
        self.base_path = Path(base_data_path) if base_data_path is not None else None

        # Discover model files if models_path provided
        temp_model_file = None
        precip_model_file = None
        self.precip_two_stage = False

        if self.models_path is not None and self.models_path.exists():
            tas_candidates = sorted(self.models_path.glob('*_tas.pkl'))
            pr_candidates = sorted(self.models_path.glob('*_pr*.pkl'))

            if tas_candidates:
                temp_model_file = str(tas_candidates[0])

            # Prefer two-stage amount model for precipitation if present
            if pr_candidates:
                amt_candidates = [p for p in pr_candidates if '_pr_amt' in p.name or '_pr_amount' in p.name or '_pr_amt' in p.name]
                occ_candidates = [p for p in pr_candidates if '_pr_occ' in p.name or '_pr_occurrence' in p.name]
                if len(amt_candidates) > 0 and len(occ_candidates) > 0:
                    # Two-stage: occurrence + amount
                    precip_model_file = (str(occ_candidates[0]), str(amt_candidates[0]))
                    self.precip_two_stage = True
                elif len(pr_candidates) > 0:
                    precip_model_file = str(pr_candidates[0])

        # Fall back to explicit paths if discovery failed
        if temp_model_file is None and temp_model_path is not None:
            temp_model_file = temp_model_path
        if precip_model_file is None and precip_model_path is not None:
            precip_model_file = precip_model_path

        if temp_model_file is None or precip_model_file is None:
            raise ValueError('Temperature or precipitation model not found. Provide `models_path` or explicit model paths.')

        # Load models
        print('Loading trained models...')
        # store discovered file paths for provenance
        self.temp_model_file = temp_model_file
        self.precip_model_file = precip_model_file

        self.temp_model = DownscalingModel.load(temp_model_file)
        if self.precip_two_stage:
            # precip_model_file is a tuple (occurrence, amount)
            occ_path, amt_path = precip_model_file
            self.precip_occ_model = DownscalingModel.load(occ_path)
            self.precip_amt_model = DownscalingModel.load(amt_path)
            self.precip_model = None
        else:
            self.precip_model = DownscalingModel.load(precip_model_file)
            self.precip_occ_model = None
            self.precip_amt_model = None

        # Preprocessor / base data path
        if self.base_path is None:
            raise ValueError('base_data_path must be provided (path to AI_GCMs)')
        self.preprocessor = ClimateDataPreprocessor(str(self.base_path))

        # Load reference grid from CRU
        print('Loading reference grid...')
        cru_tmp, _ = self.preprocessor.load_cru()
        self.reference_grid = {
            'lat': cru_tmp['lat'].values,
            'lon': cru_tmp['lon'].values,
            'time_template': cru_tmp['time']  # For attributes
        }
        cru_tmp.close()

        # Infer algorithm names where available (train_v2 stores `.algorithm`)
        self.temp_algorithm = getattr(self.temp_model, 'algorithm', None) or (self.temp_model.training_history.get('algorithm') if hasattr(self.temp_model, 'training_history') else None)
        if self.precip_two_stage:
            self.precip_occ_file = occ_path
            self.precip_amt_file = amt_path
            self.precip_algorithm = getattr(self.precip_amt_model, 'algorithm', None) or (self.precip_amt_model.training_history.get('algorithm') if hasattr(self.precip_amt_model, 'training_history') else None)
        else:
            self.precip_algorithm = getattr(self.precip_model, 'algorithm', None) or (self.precip_model.training_history.get('algorithm') if hasattr(self.precip_model, 'training_history') else None)

        print('✓ Initialization complete')
    
    def process_future_scenario(self, gcm_model: str, scenario: str,
                                output_dir: str) -> Tuple[xr.DataArray, xr.DataArray, list]:
        """
        Process one GCM future scenario
        
        Parameters:
        -----------
        gcm_model : str
            GCM model name (e.g., 'BCC-CSM2-MR')
        scenario : str
            Scenario name ('ssp126', 'ssp585')
        output_dir : str
            Output directory for downscaled data
            
        Returns:
        --------
        tas_downscaled, pr_downscaled : xr.DataArray
            Downscaled temperature and precipitation
        """
        print(f"\n{'='*80}")
        print(f"Processing: {gcm_model} {scenario}")
        print(f"{'='*80}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess future GCM data
        print("\n1. Loading and preprocessing GCM data...")
        gcm_tas, gcm_pr = self.preprocessor.load_gcm(
            model_name=gcm_model,
            scenario=scenario,
            regrid=True
        )
        
        # Convert to DataFrame
        print("\n2. Preparing features...")
        df = self._prepare_features(gcm_tas, gcm_pr)
        
        # Make predictions
        print("\n3. Generating predictions...")
        tas_pred, pr_pred = self._predict(df)
        
        # Reshape to spatial arrays
        print("\n4. Reshaping to spatial grids...")
        tas_downscaled = self._reshape_to_grid(df, tas_pred, gcm_tas.time, 'tas')
        pr_downscaled = self._reshape_to_grid(df, pr_pred, gcm_pr.time, 'pr')
        
        # Save outputs
        print("\n5. Saving downscaled data...")
        self._save_netcdf(
            tas_downscaled,
            output_path / f'{gcm_model}_{scenario}_tas_downscaled_0.25deg.nc',
            'tas',
            gcm_model,
            scenario
        )
        self._save_netcdf(
            pr_downscaled,
            output_path / f'{gcm_model}_{scenario}_pr_downscaled_0.25deg.nc',
            'pr',
            gcm_model,
            scenario
        )

        # Produce quick diagnostics comparing raw GCM vs downscaled output
        try:
            self._visualize_comparison(gcm_tas, gcm_pr, tas_downscaled, pr_downscaled, output_path, gcm_model, scenario)
        except Exception as e:
            print(f"  Warning: visualization failed: {e}")
        
        print(f"\n✓ Completed: {gcm_model} {scenario}")
        
        saved = [
            output_path / f'{gcm_model}_{scenario}_tas_downscaled_0.25deg.nc',
            output_path / f'{gcm_model}_{scenario}_pr_downscaled_0.25deg.nc'
        ]

        return tas_downscaled, pr_downscaled, saved
    
    def _prepare_features(self, gcm_tas: xr.DataArray, gcm_pr: xr.DataArray) -> pd.DataFrame:
        """Prepare features from GCM data"""
        # Flatten to DataFrame
        loader = DownscalingDataLoader(str(self.base_path / 'data' / 'processed' / 'train'))
        
        df_tas = loader._flatten_to_dataframe(gcm_tas, 'gcm_tas_degC')
        df_pr = loader._flatten_to_dataframe(gcm_pr, 'gcm_pr_mm')
        
        # Merge
        df = df_tas.merge(df_pr, on=['time', 'lat', 'lon'])
        
        # Add temporal features
        df = loader._add_temporal_features(df)
        
        # Add log-transformed precipitation
        df['gcm_pr_log1p'] = np.log1p(df['gcm_pr_mm'])
        
        print(f"  Features prepared: {df.shape}")
        print(f"  Time range: {df['time'].min()} to {df['time'].max()}")
        
        return df
    
    def _predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for both variables"""
        # Temperature features
        temp_features = ['gcm_tas_degC', 'gcm_pr_mm', 'lat', 'lon', 'month_sin', 'month_cos']
        X_temp = df[temp_features]
        
        # Precipitation features
        precip_features = ['gcm_pr_log1p', 'gcm_tas_degC', 'lat', 'lon', 'month_sin', 'month_cos']
        X_precip = df[precip_features]
        
        print("  Predicting temperature...")
        tas_pred = self.temp_model.predict(X_temp)

        print("  Predicting precipitation...")
        # Support two-stage precipitation models (occurrence + amount)
        if getattr(self, 'precip_two_stage', False) and self.precip_occ_model is not None and self.precip_amt_model is not None:
            p_wet = self.precip_occ_model.predict(X_precip)
            pred_log_amt = self.precip_amt_model.predict(X_precip)
            pred_amt = np.expm1(pred_log_amt)
            pr_pred = p_wet * pred_amt
        else:
            pr_pred_log = self.precip_model.predict(X_precip)
            # Inverse log transform for precipitation
            pr_pred = np.expm1(pr_pred_log)

        # Ensure non-negative precipitation
        pr_pred = np.maximum(pr_pred, 0)
        
        print(f"  Temperature predictions: min={tas_pred.min():.2f}, max={tas_pred.max():.2f}, mean={tas_pred.mean():.2f} °C")
        print(f"  Precipitation predictions: min={pr_pred.min():.2f}, max={pr_pred.max():.2f}, mean={pr_pred.mean():.2f} mm/month")
        
        return tas_pred, pr_pred
    
    def _reshape_to_grid(self, df: pd.DataFrame, values: np.ndarray,
                        time_coord: xr.DataArray, var_name: str) -> xr.DataArray:
        """Reshape predictions back to 3D grid"""
        # Add predictions to DataFrame
        df = df.copy()
        df['prediction'] = values
        
        # Get unique times
        unique_times = sorted(df['time'].unique())
        n_time = len(unique_times)
        n_lat = len(self.reference_grid['lat'])
        n_lon = len(self.reference_grid['lon'])
        
        # Initialize 3D array
        data_3d = np.zeros((n_time, n_lat, n_lon))
        data_3d[:] = np.nan
        
        # Create mapping dictionaries
        lat_idx = {lat: i for i, lat in enumerate(self.reference_grid['lat'])}
        lon_idx = {lon: i for i, lon in enumerate(self.reference_grid['lon'])}
        time_idx = {pd.Timestamp(t): i for i, t in enumerate(unique_times)}
        
        # Fill array
        for _, row in df.iterrows():
            t_i = time_idx[pd.Timestamp(row['time'])]
            lat_i = lat_idx[row['lat']]
            lon_i = lon_idx[row['lon']]
            data_3d[t_i, lat_i, lon_i] = row['prediction']
        
        # Create DataArray
        da = xr.DataArray(
            data_3d,
            coords={
                'time': unique_times,
                'lat': self.reference_grid['lat'],
                'lon': self.reference_grid['lon']
            },
            dims=['time', 'lat', 'lon'],
            name=var_name
        )
        
        return da
    
    def _save_netcdf(self, da: xr.DataArray, filepath: Path,
                    var_name: str, gcm_model: str, scenario: str):
        """Save DataArray to NetCDF with metadata"""
        # Add attributes
        # Use a generic description; algorithm details are saved inside model metadata
        common_attrs = {
            'target_resolution': '0.25 degrees',
            'source_gcm': gcm_model,
            'scenario': scenario,
            'creation_date': pd.Timestamp.now().isoformat()
        }

        if var_name == 'tas':
            da.attrs = {
                'long_name': 'Near-Surface Air Temperature (downscaled)',
                'units': 'degC',
                'standard_name': 'air_temperature',
                'description': f'ML-downscaled temperature from {gcm_model} {scenario}'
            }
        elif var_name == 'pr':
            da.attrs = {
                'long_name': 'Precipitation (downscaled)',
                'units': 'mm/month',
                'standard_name': 'precipitation_amount',
                'description': f'ML-downscaled precipitation from {gcm_model} {scenario}'
            }

        # Merge common attrs
        da.attrs.update(common_attrs)

        # Add model provenance
        try:
            if var_name == 'tas':
                da.attrs['model_file'] = str(self.temp_model_file)
                if getattr(self, 'temp_algorithm', None):
                    da.attrs['model_algorithm'] = str(self.temp_algorithm)
            elif var_name == 'pr':
                if getattr(self, 'precip_two_stage', False):
                    da.attrs['precip_occurrence_model'] = str(self.precip_occ_file)
                    da.attrs['precip_amount_model'] = str(self.precip_amt_file)
                    if getattr(self, 'precip_algorithm', None):
                        da.attrs['precip_algorithm'] = str(self.precip_algorithm)
                else:
                    da.attrs['model_file'] = str(self.precip_model_file)
                    if getattr(self, 'precip_algorithm', None):
                        da.attrs['model_algorithm'] = str(self.precip_algorithm)
        except Exception:
            pass
        
        # Save with compression
        encoding = {
            var_name: {
                'zlib': True,
                'complevel': 5,
                'dtype': 'float32'
            }
        }
        
        da.to_netcdf(filepath, encoding=encoding)
        print(f"  ✓ Saved: {filepath.name}")
        print(f"    Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")

    def _visualize_comparison(self, gcm_tas: xr.DataArray, gcm_pr: xr.DataArray,
                              tas_down: xr.DataArray, pr_down: xr.DataArray, output_path: Path,
                              gcm_model: str, scenario: str):
        """Create simple diagnostics comparing GCM raw fields and downscaled outputs.

        Saves time-series of spatial means and a couple of sample maps.
        """
        output_path.mkdir(parents=True, exist_ok=True)

        # Time series of spatial mean
        try:
            gcm_tas_mean = gcm_tas.mean(dim=['lat', 'lon']).values
            down_tas_mean = tas_down.mean(dim=['lat', 'lon']).values
            times = pd.to_datetime(tas_down['time'].values)

            plt.figure(figsize=(10, 4))
            plt.plot(times, gcm_tas_mean, label='GCM (raw)')
            plt.plot(times, down_tas_mean, label='Downscaled', alpha=0.8)
            plt.legend()
            plt.title(f'{gcm_tas.name or "tas"} - Spatial mean time series')
            plt.xlabel('Time')
            plt.ylabel('Temperature (°C)')
            ts_file = output_path / f'{gcm_model}_{scenario}_diag_tas_timeseries.png'
            plt.tight_layout()
            plt.savefig(ts_file, dpi=150)
            plt.close()
            print(f"  ✓ Saved diagnostic: {ts_file.name}")
        except Exception:
            pass

        try:
            gcm_pr_mean = gcm_pr.mean(dim=['lat', 'lon']).values
            down_pr_mean = pr_down.mean(dim=['lat', 'lon']).values
            times = pd.to_datetime(pr_down['time'].values)

            plt.figure(figsize=(10, 4))
            plt.plot(times, gcm_pr_mean, label='GCM (raw)')
            plt.plot(times, down_pr_mean, label='Downscaled', alpha=0.8)
            plt.legend()
            plt.title(f'{gcm_pr.name or "pr"} - Spatial mean time series')
            plt.xlabel('Time')
            plt.ylabel('Precipitation (mm/month)')
            ts_file = output_path / f'{gcm_model}_{scenario}_diag_pr_timeseries.png'
            plt.tight_layout()
            plt.savefig(ts_file, dpi=150)
            plt.close()
            print(f"  ✓ Saved diagnostic: {ts_file.name}")
        except Exception:
            pass
    
    def process_all_scenarios(self, gcm_models: Optional[List[str]] = None,
                             scenarios: Optional[List[str]] = None,
                             output_dir: str = 'outputs/downscaled'):
        """
        Process all GCM models and scenarios
        
        Parameters:
        -----------
        gcm_models : list, optional
            List of GCM models to process. If None, process all available.
        scenarios : list, optional
            List of scenarios to process. Default: ['ssp126', 'ssp585']
        output_dir : str
            Output directory
        """
        # Default scenarios
        if scenarios is None:
            scenarios = ['ssp126', 'ssp585']
        
        # Default GCM models
        if gcm_models is None:
            gcm_models = [
                'BCC-CSM2-MR',
                'CAMS-CSM1-0',
                'CanESM5',
                'CESM2',
                'CESM2-WACCM',
                'EC-Earth3',
                'IPSL-CM6A-LR',
                'MIROC6',
                'MRI-ESM2-0'
            ]
        
        print("\n" + "="*80)
        print("BATCH PROCESSING ALL SCENARIOS")
        print("="*80)
        print(f"GCM models: {len(gcm_models)}")
        print(f"Scenarios: {scenarios}")
        print(f"Total jobs: {len(gcm_models) * len(scenarios)}")
        print("="*80)
        
        # Process each combination
        results = []
        failed = []
        saved_files = []
        
        for gcm in gcm_models:
            for scenario in scenarios:
                try:
                    tas_ds, pr_ds, saved = self.process_future_scenario(
                        gcm_model=gcm,
                        scenario=scenario,
                        output_dir=output_dir
                    )
                    results.append((gcm, scenario, 'success'))
                    saved_files.extend(saved)
                except Exception as e:
                    print(f"\n✗ FAILED: {gcm} {scenario}")
                    print(f"  Error: {str(e)}")
                    failed.append((gcm, scenario, str(e)))
                    results.append((gcm, scenario, 'failed'))
        
        # Summary
        print("\n" + "="*80)
        print("BATCH PROCESSING SUMMARY")
        print("="*80)
        print(f"Successful: {sum(1 for r in results if r[2] == 'success')}")
        print(f"Failed: {sum(1 for r in results if r[2] == 'failed')}")
        
        if failed:
            print("\nFailed jobs:")
            for gcm, scenario, error in failed:
                print(f"  {gcm} {scenario}: {error}")
        
        print(f"\n✓ Downscaled data saved to: {output_dir}")

        # Create ensemble statistics (mean + std) per scenario when possible
        try:
            print('\nCreating ensemble statistics for each scenario...')
            outp = Path(output_dir)
            for scenario in scenarios:
                ds_list = []
                models_present = []
                for gcm in gcm_models:
                    tas_fp = outp / f"{gcm}_{scenario}_tas_downscaled_0.25deg.nc"
                    pr_fp = outp / f"{gcm}_{scenario}_pr_downscaled_0.25deg.nc"
                    if tas_fp.exists() and pr_fp.exists():
                        try:
                            ds_tas = xr.open_dataset(tas_fp)
                            ds_pr = xr.open_dataset(pr_fp)
                            # Merge tas and pr into single dataset for this model
                            ds_model = xr.merge([ds_tas, ds_pr])
                            # add model coordinate and expand dims
                            ds_model = ds_model.expand_dims({'model': [gcm]})
                            ds_list.append(ds_model)
                            models_present.append(gcm)
                        except Exception as e:
                            print(f"  Warning: could not load {gcm} {scenario}: {e}")

                if len(ds_list) == 0:
                    print(f"  No complete model outputs found for {scenario}; skipping ensemble")
                    continue

                # Concatenate along model dimension and compute stats
                ds_concat = xr.concat(ds_list, dim='model')
                ensemble_mean = ds_concat.mean(dim='model')
                ensemble_std = ds_concat.std(dim='model')

                mean_file = outp / f"ensemble_mean_{scenario}.nc"
                std_file = outp / f"ensemble_std_{scenario}.nc"

                ensemble_mean.to_netcdf(mean_file)
                ensemble_std.to_netcdf(std_file)

                print(f"  ✓ Ensemble mean saved: {mean_file.name}")
                print(f"  ✓ Ensemble std saved: {std_file.name}")
        except Exception as e:
            print(f"  Warning: ensemble creation failed: {e}")

        return saved_files


def main():
    """Main inference script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply downscaling models to future scenarios')
    parser.add_argument('--models-path', type=str,
                        default=r'd:\appdev\cep ml\outputs\models',
                        help='Directory containing trained model files (preferred)')
    parser.add_argument('--base-path', type=str,
                        default=r'd:\appdev\cep ml\AI_GCMs',
                        help='Path to AI_GCMs directory')
    parser.add_argument('--temp-model', type=str,
                        help='Fallback: explicit path to temperature model')
    parser.add_argument('--precip-model', type=str,
                        help='Fallback: explicit path to precipitation model or tuple for two-stage')
    parser.add_argument('--output-dir', type=str,
                        default=r'd:\appdev\cep ml\outputs\downscaled',
                        help='Output directory')
    parser.add_argument('--gcm-model', type=str,
                        help='Single GCM model to process (if not processing all)')
    parser.add_argument('--scenario', type=str,
                        help='Single scenario to process (if not processing all)')
    parser.add_argument('--all', action='store_true',
                        help='Process all GCMs and scenarios')
    
    args = parser.parse_args()
    
    # Initialize downscaler
    downscaler = FutureDownscaler(
        models_path=args.models_path,
        temp_model_path=args.temp_model,
        precip_model_path=args.precip_model,
        base_data_path=args.base_path
    )
    
    if args.all:
        # Process all scenarios
        downscaler.process_all_scenarios(output_dir=args.output_dir)
    elif args.gcm_model and args.scenario:
        # Process single scenario
        downscaler.process_future_scenario(
            gcm_model=args.gcm_model,
            scenario=args.scenario,
            output_dir=args.output_dir
        )
    else:
        print("Error: Either specify --all, or both --gcm-model and --scenario")
        parser.print_help()
        return
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Validate downscaled outputs")
    print("  2. Generate diagnostic plots")
    print("  3. Analyze future climate projections")


if __name__ == '__main__':
    main()
