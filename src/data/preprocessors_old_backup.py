"""
Preprocessing pipeline for GCM downscaling
- Load CRU, ERA5, and GCM data
- Regrid to common 0.25° grid
- Convert units (K→°C, kg m⁻²s⁻¹→mm/month)
- Temporal alignment (1980-2014)
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import warnings

try:
    import xesmf as xe
except ImportError:
    warnings.warn("xesmf not installed. Regridding will use basic interpolation.")
    xe = None


class ClimateDataPreprocessor:
    """Preprocess climate data for ML downscaling"""
    
    def __init__(self, base_path: str, start_year: int = 1980, end_year: int = 2014):
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
        
    def load_cru(self) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Load CRU temperature and precipitation data
        
        Returns:
        --------
        tmp_ds, pre_ds : xr.Dataset
            CRU temperature and precipitation datasets
        """
        print(f"Loading CRU data from {self.cru_path}")
        
        # Load temperature
        tmp_file = self.cru_path / 'cru_tmp.1901.2024.0.25deg.pakistan.nc'
        tmp_ds = xr.open_dataset(tmp_file)
        
        # Load precipitation
        pre_file = self.cru_path / 'cru_pre.1901.2024.0.25deg.pakistan.nc'
        pre_ds = xr.open_dataset(pre_file)
        
        # Standardize coordinate names
        tmp_ds = self._standardize_coords(tmp_ds)
        pre_ds = self._standardize_coords(pre_ds)
        
        # Subset to training period
        tmp_ds = tmp_ds.sel(time=slice(self.start_date, self.end_date))
        pre_ds = pre_ds.sel(time=slice(self.start_date, self.end_date))
        
        # Store reference grid
        self.reference_grid = {
            'lat': tmp_ds['lat'].values,
            'lon': tmp_ds['lon'].values
        }
        
        print(f"  Temperature: {list(tmp_ds.data_vars)}, shape: {tmp_ds[list(tmp_ds.data_vars)[0]].shape}")
        print(f"  Precipitation: {list(pre_ds.data_vars)}, shape: {pre_ds[list(pre_ds.data_vars)[0]].shape}")
        print(f"  Time range: {tmp_ds.time.values[0]} to {tmp_ds.time.values[-1]}")
        print(f"  Grid: {len(tmp_ds.lat)} x {len(tmp_ds.lon)} (0.25° resolution)")
        
        return tmp_ds, pre_ds
    
    def load_era5(self, regrid: bool = True) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Load ERA5 data (temperature and precipitation)
        
        Parameters:
        -----------
        regrid : bool
            Whether to regrid to CRU grid
            
        Returns:
        --------
        t2m, tp : xr.DataArray
            ERA5 temperature (°C) and precipitation (mm/month)
        """
        print(f"\nLoading ERA5 data from {self.era5_path}")
        
        # Load both ERA5 files
        ua_file = self.era5_path / 'data_stream-moda_stepType-avgua.nc'
        ad_file = self.era5_path / 'data_stream-moda_stepType-avgad.nc'
        
        ds_ua = xr.open_dataset(ua_file)
        ds_ad = xr.open_dataset(ad_file)
        
        # Standardize coordinates
        ds_ua = self._standardize_coords(ds_ua)
        ds_ad = self._standardize_coords(ds_ad)
        
        # Identify variables (assuming t2m in ua, tp in ad based on typical ERA5 structure)
        # This should be verified from inspection notebook
        var_ua = list(ds_ua.data_vars)[0]  # Temperature variable
        var_ad = list(ds_ad.data_vars)[0]  # Precipitation variable
        
        print(f"  Variables found: {var_ua} (temp), {var_ad} (precip)")
        
        # Extract variables
        t2m = ds_ua[var_ua]
        tp = ds_ad[var_ad]
        
        # Convert units
        # Temperature: K → °C
        if t2m.max() > 100:  # Likely in Kelvin
            print("  Converting temperature: K → °C")
            t2m = t2m - 273.15
            t2m.attrs['units'] = 'degC'
        
        # Precipitation: m → mm/month
        # ERA5 monthly total precipitation is in meters
        if tp.max() < 10:  # Likely in meters
            print("  Converting precipitation: m → mm/month")
            tp = tp * 1000  # m to mm
            tp.attrs['units'] = 'mm/month'
        
        # Subset to training period
        t2m = t2m.sel(time=slice(self.start_date, self.end_date))
        tp = tp.sel(time=slice(self.start_date, self.end_date))
        
        # Regrid to CRU grid
        if regrid and self.reference_grid is not None:
            print("  Regridding ERA5 to CRU 0.25° grid...")
            t2m = self._regrid_to_reference(t2m)
            tp = self._regrid_to_reference(tp)
        
        print(f"  Temperature shape after processing: {t2m.shape}")
        print(f"  Precipitation shape after processing: {tp.shape}")
        
        ds_ua.close()
        ds_ad.close()
        
        return t2m, tp
    
    def load_gcm(self, model_name: str, scenario: str = 'hist', 
                 regrid: bool = True) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Load GCM temperature and precipitation data
        
        Parameters:
        -----------
        model_name : str
            GCM model name (e.g., 'BCC-CSM2-MR')
        scenario : str
            Scenario name ('hist', 'ssp126', 'ssp585')
        regrid : bool
            Whether to regrid to CRU grid
            
        Returns:
        --------
        tas, pr : xr.DataArray
            GCM temperature (°C) and precipitation (mm/month)
        """
        print(f"\nLoading GCM data: {model_name} {scenario}")
        
        # Load temperature
        tas_file = self.gcm_path / f'{model_name}_{scenario}_tas.nc'
        pr_file = self.gcm_path / f'{model_name}_{scenario}_pr.nc'
        
        ds_tas = xr.open_dataset(tas_file)
        ds_pr = xr.open_dataset(pr_file)
        
        # Standardize coordinates
        ds_tas = self._standardize_coords(ds_tas)
        ds_pr = self._standardize_coords(ds_pr)
        
        # Get variable names
        tas_var = list(ds_tas.data_vars)[0]
        pr_var = list(ds_pr.data_vars)[0]
        
        tas = ds_tas[tas_var]
        pr = ds_pr[pr_var]
        
        # Convert temperature: K → °C
        if tas.max() > 100:  # Likely in Kelvin
            print("  Converting temperature: K → °C")
            tas = tas - 273.15
            tas.attrs['units'] = 'degC'
        
        # Convert precipitation: kg m⁻² s⁻¹ → mm/month
        # pr in kg m⁻² s⁻¹ is equivalent to mm/s
        # Monthly mean: multiply by seconds per month (~30 days)
        if pr.max() < 1:  # Likely in kg m⁻² s⁻¹
            print("  Converting precipitation: kg m⁻² s⁻¹ → mm/month")
            # Approximate: 30 days * 24 hours * 3600 seconds
            seconds_per_month = 30 * 24 * 3600
            pr = pr * seconds_per_month
            pr.attrs['units'] = 'mm/month'
        
        # For historical data, subset to training period
        if scenario == 'hist':
            tas = tas.sel(time=slice(self.start_date, self.end_date))
            pr = pr.sel(time=slice(self.start_date, self.end_date))
        
        # Regrid to CRU grid
        if regrid and self.reference_grid is not None:
            print(f"  Regridding GCM from {len(tas.lat)} x {len(tas.lon)} to CRU 0.25° grid...")
            tas = self._regrid_to_reference(tas)
            pr = self._regrid_to_reference(pr)
        
        print(f"  Temperature shape: {tas.shape}")
        print(f"  Precipitation shape: {pr.shape}")
        print(f"  Time range: {tas.time.values[0]} to {tas.time.values[-1]}")
        
        ds_tas.close()
        ds_pr.close()
        
        return tas, pr
    
    def _standardize_coords(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize coordinate names to time, lat, lon"""
        rename_dict = {}
        
        for coord in ds.coords:
            coord_lower = coord.lower()
            if 'time' in coord_lower and coord != 'time':
                rename_dict[coord] = 'time'
            elif 'lat' in coord_lower and coord != 'lat':
                rename_dict[coord] = 'lat'
            elif 'lon' in coord_lower and coord != 'lon':
                rename_dict[coord] = 'lon'
        
        if rename_dict:
            ds = ds.rename(rename_dict)
        
        # Ensure time is decoded
        if 'time' in ds.coords:
            ds = xr.decode_cf(ds)
        
        return ds
    
    def _regrid_to_reference(self, da: xr.DataArray, method: str = 'bilinear') -> xr.DataArray:
        """
        Regrid DataArray to reference grid
        
        Parameters:
        -----------
        da : xr.DataArray
            Input data array
        method : str
            Regridding method ('bilinear', 'conservative', 'nearest')
            
        Returns:
        --------
        da_regrid : xr.DataArray
            Regridded data array
        """
        if self.reference_grid is None:
            raise ValueError("Reference grid not set. Call load_cru() first.")
        
        # Create target grid
        target_grid = xr.Dataset({
            'lat': (['lat'], self.reference_grid['lat']),
            'lon': (['lon'], self.reference_grid['lon'])
        })
        
        if xe is not None:
            # Use xESMF for high-quality regridding
            regridder = xe.Regridder(da, target_grid, method=method, periodic=False)
            da_regrid = regridder(da)
            regridder.clean_weight_file()
        else:
            # Fallback to basic interpolation
            da_regrid = da.interp(
                lat=self.reference_grid['lat'],
                lon=self.reference_grid['lon'],
                method='linear'
            )
        
        return da_regrid
    
    def process_and_save(self, output_dir: str, gcm_model: str = 'BCC-CSM2-MR'):
        """
        Process all datasets and save to NetCDF
        
        Parameters:
        -----------
        output_dir : str
            Output directory for processed files
        gcm_model : str
            GCM model to process for MVP
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"PROCESSING CLIMATE DATA FOR {gcm_model}")
        print(f"{'='*80}")
        
        # Load CRU (reference)
        cru_tmp, cru_pre = self.load_cru()
        
        # Load ERA5 (target)
        era5_t2m, era5_tp = self.load_era5(regrid=True)
        
        # Load GCM (predictor)
        gcm_tas, gcm_pr = self.load_gcm(gcm_model, scenario='hist', regrid=True)
        
        # Ensure temporal alignment
        print("\nAligning time coordinates...")
        common_times = self._get_common_times([
            cru_tmp.time,
            era5_t2m.time,
            gcm_tas.time
        ])
        
        print(f"  Common time steps: {len(common_times)}")
        print(f"  Range: {common_times[0].values} to {common_times[-1].values}")
        
        # Subset all to common times
        cru_tmp = cru_tmp.sel(time=common_times)
        cru_pre = cru_pre.sel(time=common_times)
        era5_t2m = era5_t2m.sel(time=common_times)
        era5_tp = era5_tp.sel(time=common_times)
        gcm_tas = gcm_tas.sel(time=common_times)
        gcm_pr = gcm_pr.sel(time=common_times)
        
        # Save processed data
        print(f"\nSaving processed data to {output_path}")
        
        # CRU
        cru_tmp_var = list(cru_tmp.data_vars)[0]
        cru_pre_var = list(cru_pre.data_vars)[0]
        
        cru_tmp[cru_tmp_var].to_netcdf(
            output_path / 'cru_tmp_1980_2014_processed.nc',
            encoding={cru_tmp_var: {'zlib': True, 'complevel': 5}}
        )
        cru_pre[cru_pre_var].to_netcdf(
            output_path / 'cru_pre_1980_2014_processed.nc',
            encoding={cru_pre_var: {'zlib': True, 'complevel': 5}}
        )
        
        # ERA5
        era5_t2m.to_netcdf(
            output_path / 'era5_t2m_1980_2014_processed.nc',
            encoding={'__xarray_dataarray_variable__': {'zlib': True, 'complevel': 5}}
        )
        era5_tp.to_netcdf(
            output_path / 'era5_tp_1980_2014_processed.nc',
            encoding={'__xarray_dataarray_variable__': {'zlib': True, 'complevel': 5}}
        )
        
        # GCM
        gcm_tas.to_netcdf(
            output_path / f'{gcm_model}_hist_tas_1980_2014_processed.nc',
            encoding={'__xarray_dataarray_variable__': {'zlib': True, 'complevel': 5}}
        )
        gcm_pr.to_netcdf(
            output_path / f'{gcm_model}_hist_pr_1980_2014_processed.nc',
            encoding={'__xarray_dataarray_variable__': {'zlib': True, 'complevel': 5}}
        )
        
        print("\n✓ Processing complete!")
        
        # Close datasets
        cru_tmp.close()
        cru_pre.close()
        
        return output_path
    
    def _get_common_times(self, time_coords: list) -> pd.DatetimeIndex:
        """Find common time steps across multiple coordinates"""
        # Convert all to pandas datetime
        time_series = [pd.to_datetime(tc.values) for tc in time_coords]
        
        # Find intersection
        common = time_series[0]
        for ts in time_series[1:]:
            common = common.intersection(ts)
        
        return xr.DataArray(common, dims=['time'], coords={'time': common})


def main():
    """Main preprocessing script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess climate data for ML downscaling')
    parser.add_argument('--base-path', type=str, default=r'd:\appdev\cep ml\AI_GCMs',
                        help='Path to AI_GCMs directory')
    parser.add_argument('--output-dir', type=str, default=r'd:\appdev\cep ml\data\processed\train',
                        help='Output directory')
    parser.add_argument('--gcm-model', type=str, default='BCC-CSM2-MR',
                        help='GCM model name for MVP')
    parser.add_argument('--start-year', type=int, default=1980,
                        help='Start year')
    parser.add_argument('--end-year', type=int, default=2014,
                        help='End year')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = ClimateDataPreprocessor(
        base_path=args.base_path,
        start_year=args.start_year,
        end_year=args.end_year
    )
    
    # Process and save
    preprocessor.process_and_save(
        output_dir=args.output_dir,
        gcm_model=args.gcm_model
    )


if __name__ == '__main__':
    main()
