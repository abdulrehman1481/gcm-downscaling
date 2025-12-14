"""
Preprocessing pipeline for GCM downscaling - UPDATED FOR ACTUAL DATA

This module automatically uses the enhanced version (preprocessors_v2) if available,
with fallback to the original implementation for backward compatibility.
"""

# Try to import enhanced version first
try:
    from .preprocessors_v2 import ClimateDataPreprocessor
    print("✓ Using enhanced ClimateDataPreprocessor from preprocessors_v2")
except (ImportError, ModuleNotFoundError):
    print("⚠ Enhanced version not available, loading standard ClimateDataPreprocessor")
    
    # Import dependencies for fallback implementation
    import xarray as xr
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from typing import Tuple, Optional, List
    import warnings
    import cftime

    try:
        import xesmf as xe
    except ImportError:
        warnings.warn("xesmf not installed. Regridding will use basic interpolation.")
        xe = None
    
    # Original class implementation
    class ClimateDataPreprocessor:
        """Preprocess climate data for ML downscaling"""
        
        def __init__(self, base_path: str, start_year: int = 1980, end_year: int = 2014,
                     checkpoint_file: Optional[str] = None):
            """
            Initialize preprocessor
            
            Parameters:
            -----------
            base_path : str
                Path to AI_GCMs directory
            start_year : int
                Start year for training period
            end_year : int
                End year for training period  
            checkpoint_file : str, optional
                Ignored in standard version (only used in enhanced version)
            """
            self.base_path = Path(base_path)
            self.cru_path = self.base_path / 'CRU'
            self.era5_path = self.base_path / 'ERA5'
            self.gcm_path = self.base_path / 'GCMs'
            
            self.start_year = start_year
            self.end_year = end_year
            self.start_date = f'{start_year}-01-01'
            self.end_date = f'{end_year}-12-31'
            
            self.reference_grid = None
            
            # GCM models available
            self.gcm_models = [
                'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CanESM5', 'CESM2', 'CESM2-WACCM',
                'EC-Earth3', 'IPSL-CM6A-LR', 'MIROC6', 'MRI-ESM2-0'
            ]
            
        def load_cru(self) -> Tuple[xr.DataArray, xr.DataArray]:
            """Load CRU temperature and precipitation data"""
            print(f"Loading CRU data from {self.cru_path}")
            
            # Load temperature
            tmp_file = self.cru_path / 'cru_tmp.1901.2024.0.25deg.pakistan.nc'
            tmp_ds = xr.open_dataset(tmp_file)
            tmp = tmp_ds['tmp']
            
            # Load precipitation
            pre_file = self.cru_path / 'cru_pre.1901.2024.0.25deg.pakistan.nc'
            pre_ds = xr.open_dataset(pre_file)
            pre = pre_ds['pre']
            
            # Subset to training period
            tmp = tmp.sel(time=slice(self.start_date, self.end_date))
            pre = pre.sel(time=slice(self.start_date, self.end_date))
            
            # Store reference grid
            self.reference_grid = {
                'lat': tmp['lat'].values,
                'lon': tmp['lon'].values
            }
            
            print(f"  ✓ Temperature: {tmp.shape}")
            print(f"  ✓ Precipitation: {pre.shape}")
            
            tmp_ds.close()
            pre_ds.close()
            
            return tmp, pre
        
        def load_era5(self, regrid: bool = True) -> Tuple[xr.DataArray, xr.DataArray]:
            """Load ERA5 data (temperature and precipitation)"""
            print(f"\nLoading ERA5 data from {self.era5_path}")
            
            # Load both ERA5 files
            ua_file = self.era5_path / 'data_stream-moda_stepType-avgua.nc'
            ad_file = self.era5_path / 'data_stream-moda_stepType-avgad.nc'
            
            ds_ua = xr.open_dataset(ua_file)
            ds_ad = xr.open_dataset(ad_file)
            
            # Extract variables
            t2m = ds_ua['t2m']
            tp = ds_ad['tp']
            
            # Drop problematic coordinates
            coords_to_drop = [c for c in ['expver', 'number'] if c in t2m.coords]
            if coords_to_drop:
                t2m = t2m.drop_vars(coords_to_drop)
            
            coords_to_drop = [c for c in ['expver', 'number'] if c in tp.coords]
            if coords_to_drop:
                tp = tp.drop_vars(coords_to_drop)
            
            # Standardize coordinate names
            t2m = t2m.rename({'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'})
            tp = tp.rename({'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'})
            
            # Convert units
            t2m = t2m - 273.15  # K to °C
            t2m.attrs['units'] = 'degC'
            
            tp = tp * 1000  # m to mm
            tp.attrs['units'] = 'mm/month'
            
            # Convert time
            try:
                t2m['time'] = pd.to_datetime(t2m['time'].values)
                tp['time'] = pd.to_datetime(tp['time'].values)
            except Exception:
                t2m['time'] = pd.to_datetime(t2m['time'].astype(str).values, errors='coerce')
                tp['time'] = pd.to_datetime(tp['time'].astype(str).values, errors='coerce')
            
            # Resample if needed
            if len(t2m['time']) > 0:
                unique_months = len(t2m['time'].astype('datetime64[M]').unique())
                if unique_months < len(t2m['time']):
                    t2m = t2m.resample(time='MS').mean()
                    tp = tp.resample(time='MS').sum()
            
            # Subset to training period
            t2m = t2m.sel(time=slice(self.start_date, self.end_date))
            tp = tp.sel(time=slice(self.start_date, self.end_date))
            
            # Regrid
            if regrid and self.reference_grid is not None:
                print("  → Regridding ERA5 to CRU 0.25° grid...")
                t2m = self._regrid_to_reference(t2m)
                tp = self._regrid_to_reference(tp)
            
            print(f"  ✓ Final shapes: t2m={t2m.shape}, tp={tp.shape}")
            
            ds_ua.close()
            ds_ad.close()
            
            return t2m, tp
        
        def load_gcm(self, model_name: str, scenario: str = 'hist', 
                     regrid: bool = True) -> Tuple[xr.DataArray, xr.DataArray]:
            """Load GCM temperature and precipitation data"""
            print(f"\nLoading GCM data: {model_name} {scenario}")
            
            # Load temperature and precipitation
            tas_file = self.gcm_path / f'{model_name}_{scenario}_tas.nc'
            pr_file = self.gcm_path / f'{model_name}_{scenario}_pr.nc'
            
            # Try to decode times automatically first
            try:
                ds_tas = xr.open_dataset(tas_file)
                ds_pr = xr.open_dataset(pr_file)
            except:
                ds_tas = xr.open_dataset(tas_file, decode_times=False)
                ds_pr = xr.open_dataset(pr_file, decode_times=False)
            
            # Get variables
            tas = ds_tas['TMP']
            pr = ds_pr['PRE']
            
            # Standardize coordinate names
            tas = tas.rename({'TIME': 'time', 'YAX': 'lat', 'XAX': 'lon'})
            pr = pr.rename({'TIME': 'time', 'YAX': 'lat', 'XAX': 'lon'})
            
            # Convert time from cftime to datetime
            tas = self._convert_cftime_to_datetime(tas)
            pr = self._convert_cftime_to_datetime(pr)
            
            # Convert precipitation: mm/day → mm/month
            pr = pr * 30
            pr.attrs['units'] = 'mm/month'
            
            # For historical data, subset to training period
            if scenario == 'hist':
                tas = tas.sel(time=slice(self.start_date, self.end_date))
                pr = pr.sel(time=slice(self.start_date, self.end_date))
            
            # Regrid
            if regrid and self.reference_grid is not None:
                print(f"  → Regridding GCM from {len(tas.lat)} x {len(tas.lon)} to CRU 0.25° grid...")
                tas = self._regrid_to_reference(tas)
                pr = self._regrid_to_reference(pr)
            
            print(f"  ✓ Final shapes: tas={tas.shape}, pr={pr.shape}")
            
            ds_tas.close()
            ds_pr.close()
            
            return tas, pr
        
        def _convert_cftime_to_datetime(self, da: xr.DataArray) -> xr.DataArray:
            """Convert cftime objects to pandas datetime"""
            if len(da.time) > 0 and hasattr(da.time.values[0], 'year'):
                if not isinstance(da.time.values[0], (pd.Timestamp, np.datetime64)):
                    print("  → Converting cftime to pandas datetime...")
                    times = []
                    for t in da.time.values:
                        try:
                            times.append(pd.Timestamp(year=t.year, month=t.month, day=t.day, 
                                                    hour=getattr(t, 'hour', 0)))
                        except Exception as e:
                            print(f"    Warning: Could not convert time {t}: {e}")
                            continue
                    
                    if len(times) > 0:
                        da = da.isel(time=slice(0, len(times)))
                        da = da.assign_coords(time=pd.DatetimeIndex(times))
            
            return da
        
        def _regrid_to_reference(self, da: xr.DataArray) -> xr.DataArray:
            """Regrid DataArray to reference CRU grid"""
            if self.reference_grid is None:
                raise ValueError("Reference grid not set. Load CRU data first.")
            
            target_lat = self.reference_grid['lat']
            target_lon = self.reference_grid['lon']
            
            # Check if regridding needed
            if (len(da.lat) == len(target_lat) and len(da.lon) == len(target_lon) and
                np.allclose(da.lat.values, target_lat) and np.allclose(da.lon.values, target_lon)):
                return da
            
            # Try xESMF first
            if xe is not None:
                try:
                    ds_in = da.to_dataset(name='var')
                    ds_out = xr.Dataset({
                        'lat': (['lat'], target_lat),
                        'lon': (['lon'], target_lon),
                    })
                    
                    regridder = xe.Regridder(ds_in, ds_out, 'bilinear', periodic=False)
                    da_regrid = regridder(da)
                    
                    print("    ✓ Regridded using xESMF (bilinear)")
                    return da_regrid
                except Exception as e:
                    print(f"    xESMF failed ({e}), falling back to xarray interpolation")
            
            # Fallback to xarray interp
            da_regrid = da.interp(lat=target_lat, lon=target_lon, method='linear')
            print("    ✓ Regridded using xarray interpolation (linear)")
            
            return da_regrid
        
        def process_and_save(self, output_dir: str, gcm_models: Optional[List[str]] = None,
                            skip_existing: bool = False) -> str:
            """Complete preprocessing pipeline for ALL GCMs"""
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            print("="*80)
            print("CLIMATE DATA PREPROCESSING PIPELINE")
            print("="*80)
            
            # 1. Load CRU reference data
            print("\n[1/4] Loading CRU reference data...")
            cru_tmp, cru_pre = self.load_cru()
            
            # Save CRU
            cru_ds = xr.Dataset({'tmp': cru_tmp, 'pre': cru_pre})
            cru_out = output_path / 'cru_1980_2014.nc'
            cru_ds.to_netcdf(cru_out)
            print(f"  ✓ Saved: {cru_out}")
            
            # 2. Load ERA5 target data
            print("\n[2/4] Loading ERA5 target data...")
            era5_t2m, era5_tp = self.load_era5(regrid=True)
            
            # Save ERA5
            era5_ds = xr.Dataset({'t2m': era5_t2m, 'tp': era5_tp})
            era5_out = output_path / 'era5_1980_2014.nc'
            era5_ds.to_netcdf(era5_out)
            print(f"  ✓ Saved: {era5_out}")
            
            # 3. Process all GCMs
            if gcm_models is None:
                gcm_models = self.gcm_models
            
            print(f"\n[3/4] Processing {len(gcm_models)} GCM models (historical)...")
            
            for i, model in enumerate(gcm_models, 1):
                print(f"\n  [{i}/{len(gcm_models)}] {model}")
                try:
                    gcm_tas, gcm_pr = self.load_gcm(model, scenario='hist', regrid=True)
                    
                    # Save GCM
                    gcm_ds = xr.Dataset({'tas': gcm_tas, 'pr': gcm_pr})
                    gcm_out = output_path / f'{model}_hist_1980_2014.nc'
                    gcm_ds.to_netcdf(gcm_out)
                    print(f"    ✓ Saved: {gcm_out}")
                    
                except Exception as e:
                    print(f"    ✗ ERROR processing {model}: {e}")
                    continue
            
            print("\n[4/4] Preprocessing complete!")
            print("="*80)
            
            return str(output_path)


if __name__ == '__main__':
    # Example usage for Google Colab
    BASE_PATH = '/content/drive/MyDrive/Downscaling ML CEP/AI_GCMs'
    OUTPUT_DIR = '/content/drive/MyDrive/Downscaling ML CEP/data/processed/train'
    
    preprocessor = ClimateDataPreprocessor(base_path=BASE_PATH)
    preprocessor.process_and_save(output_dir=OUTPUT_DIR)
