"""
Evaluation metrics and diagnostics for downscaling validation
- Spatial pattern correlation
- Seasonal climatologies
- Bias maps
- Time series comparisons
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class DownscalingEvaluator:
    """Evaluate downscaling model performance"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.results = {}
    
    def spatial_correlation(self, pred: xr.DataArray, obs: xr.DataArray,
                           dim: str = 'time') -> xr.DataArray:
        """
        Compute spatial pattern correlation
        
        Parameters:
        -----------
        pred : xr.DataArray
            Predicted values
        obs : xr.DataArray
            Observed values
        dim : str
            Dimension to correlate over (typically 'time')
            
        Returns:
        --------
        corr : xr.DataArray
            Correlation at each grid point
        """
        # Compute correlation along specified dimension
        corr = xr.corr(pred, obs, dim=dim)
        return corr
    
    def compute_bias(self, pred: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
        """Compute bias (pred - obs)"""
        return pred - obs
    
    def compute_rmse(self, pred: xr.DataArray, obs: xr.DataArray,
                    dim: str = 'time') -> xr.DataArray:
        """Compute RMSE"""
        return np.sqrt(((pred - obs) ** 2).mean(dim=dim))
    
    def seasonal_climatology(self, da: xr.DataArray, season: str) -> xr.DataArray:
        """
        Compute seasonal mean climatology
        
        Parameters:
        -----------
        da : xr.DataArray
            Input data with time dimension
        season : str
            Season: 'DJF', 'MAM', 'JJA', 'SON'
            
        Returns:
        --------
        clim : xr.DataArray
            Seasonal climatology
        """
        # Select season
        da_season = da.sel(time=da.time.dt.season == season)
        
        # Compute mean
        clim = da_season.mean(dim='time')
        
        return clim
    
    def monthly_climatology(self, da: xr.DataArray) -> xr.DataArray:
        """Compute monthly climatology"""
        return da.groupby('time.month').mean(dim='time')
    
    def compute_metrics_at_points(self, pred: xr.DataArray, obs: xr.DataArray,
                                  points: list) -> pd.DataFrame:
        """
        Compute metrics at specific grid points
        
        Parameters:
        -----------
        pred : xr.DataArray
            Predictions
        obs : xr.DataArray
            Observations
        points : list
            List of (lat, lon) tuples
            
        Returns:
        --------
        metrics_df : pd.DataFrame
            Metrics at each point
        """
        results = []
        
        for lat, lon in points:
            # Extract time series
            pred_ts = pred.sel(lat=lat, lon=lon, method='nearest')
            obs_ts = obs.sel(lat=lat, lon=lon, method='nearest')
            
            # Compute metrics
            rmse = float(np.sqrt(((pred_ts - obs_ts) ** 2).mean()))
            mae = float(np.abs(pred_ts - obs_ts).mean())
            bias = float((pred_ts - obs_ts).mean())
            corr = float(xr.corr(pred_ts, obs_ts))
            
            results.append({
                'lat': lat,
                'lon': lon,
                'RMSE': rmse,
                'MAE': mae,
                'Bias': bias,
                'Correlation': corr
            })
        
        return pd.DataFrame(results)
    
    def evaluate_test_period(self, pred_temp: xr.DataArray, obs_temp: xr.DataArray,
                            pred_precip: xr.DataArray, obs_precip: xr.DataArray,
                            test_years: Tuple[int, int] = (2011, 2014)) -> Dict:
        """
        Comprehensive evaluation on test period
        
        Parameters:
        -----------
        pred_temp : xr.DataArray
            Predicted temperature
        obs_temp : xr.DataArray
            Observed temperature (ERA5)
        pred_precip : xr.DataArray
            Predicted precipitation
        obs_precip : xr.DataArray
            Observed precipitation (ERA5)
        test_years : tuple
            (start, end) test years
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        print(f"Evaluating test period: {test_years[0]}-{test_years[1]}")
        
        # Subset to test period
        pred_temp = pred_temp.sel(time=slice(str(test_years[0]), str(test_years[1])))
        obs_temp = obs_temp.sel(time=slice(str(test_years[0]), str(test_years[1])))
        pred_precip = pred_precip.sel(time=slice(str(test_years[0]), str(test_years[1])))
        obs_precip = obs_precip.sel(time=slice(str(test_years[0]), str(test_years[1])))
        
        metrics = {}
        
        # Temperature metrics
        metrics['temp_rmse'] = float(self.compute_rmse(pred_temp, obs_temp).mean())
        metrics['temp_mae'] = float(np.abs(pred_temp - obs_temp).mean())
        metrics['temp_bias'] = float((pred_temp - obs_temp).mean())
        metrics['temp_spatial_corr'] = float(self.spatial_correlation(pred_temp, obs_temp).mean())
        
        # Precipitation metrics
        metrics['precip_rmse'] = float(self.compute_rmse(pred_precip, obs_precip).mean())
        metrics['precip_mae'] = float(np.abs(pred_precip - obs_precip).mean())
        metrics['precip_bias'] = float((pred_precip - obs_precip).mean())
        metrics['precip_spatial_corr'] = float(self.spatial_correlation(pred_precip, obs_precip).mean())
        
        # Seasonal metrics
        for season in ['DJF', 'MAM', 'JJA', 'SON']:
            # Temperature
            pred_temp_season = self.seasonal_climatology(pred_temp, season)
            obs_temp_season = self.seasonal_climatology(obs_temp, season)
            metrics[f'temp_{season}_pattern_corr'] = float(
                stats.pearsonr(
                    pred_temp_season.values.flatten(),
                    obs_temp_season.values.flatten()
                )[0]
            )
            
            # Precipitation
            pred_precip_season = self.seasonal_climatology(pred_precip, season)
            obs_precip_season = self.seasonal_climatology(obs_precip, season)
            metrics[f'precip_{season}_pattern_corr'] = float(
                stats.pearsonr(
                    pred_precip_season.values.flatten(),
                    obs_precip_season.values.flatten()
                )[0]
            )
        
        self.results['test_period_metrics'] = metrics
        
        return metrics
    
    def plot_spatial_comparison(self, pred: xr.DataArray, obs: xr.DataArray,
                               title: str, cmap: str = 'RdBu_r',
                               save_path: Optional[str] = None):
        """
        Plot spatial comparison: predicted, observed, and bias
        
        Parameters:
        -----------
        pred : xr.DataArray
            Predicted field (2D)
        obs : xr.DataArray
            Observed field (2D)
        title : str
            Plot title
        cmap : str
            Colormap
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Predicted
        pred.plot(ax=axes[0], cmap=cmap, add_colorbar=True)
        axes[0].set_title(f'{title} - Predicted')
        
        # Observed
        obs.plot(ax=axes[1], cmap=cmap, add_colorbar=True)
        axes[1].set_title(f'{title} - Observed (ERA5)')
        
        # Bias
        bias = pred - obs
        vmax = np.abs(bias).max()
        bias.plot(ax=axes[2], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=True)
        axes[2].set_title(f'{title} - Bias (Pred - Obs)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to {save_path}")
        
        plt.show()
    
    def plot_seasonal_comparison(self, pred: xr.DataArray, obs: xr.DataArray,
                                variable: str, season: str,
                                save_path: Optional[str] = None):
        """Plot seasonal climatology comparison"""
        pred_season = self.seasonal_climatology(pred, season)
        obs_season = self.seasonal_climatology(obs, season)
        
        cmap = 'RdYlBu_r' if variable == 'temperature' else 'YlGnBu'
        
        self.plot_spatial_comparison(
            pred_season,
            obs_season,
            f'{variable.capitalize()} {season} Climatology',
            cmap=cmap,
            save_path=save_path
        )
    
    def plot_scatter(self, pred: xr.DataArray, obs: xr.DataArray,
                    title: str, xlabel: str, ylabel: str,
                    save_path: Optional[str] = None):
        """
        Plot scatter plot of predicted vs observed
        
        Parameters:
        -----------
        pred : xr.DataArray
            Predictions
        obs : xr.DataArray
            Observations
        title : str
            Plot title
        xlabel, ylabel : str
            Axis labels
        save_path : str, optional
            Path to save figure
        """
        # Flatten and remove NaNs
        pred_flat = pred.values.flatten()
        obs_flat = obs.values.flatten()
        
        mask = ~(np.isnan(pred_flat) | np.isnan(obs_flat))
        pred_flat = pred_flat[mask]
        obs_flat = obs_flat[mask]
        
        # Compute statistics
        r2 = stats.pearsonr(pred_flat, obs_flat)[0] ** 2
        rmse = np.sqrt(np.mean((pred_flat - obs_flat) ** 2))
        bias = np.mean(pred_flat - obs_flat)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Hexbin for large datasets
        if len(pred_flat) > 10000:
            hb = ax.hexbin(obs_flat, pred_flat, gridsize=50, cmap='Blues', mincnt=1)
            plt.colorbar(hb, ax=ax, label='Count')
        else:
            ax.scatter(obs_flat, pred_flat, alpha=0.5, s=10)
        
        # 1:1 line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='1:1 line')
        
        # Statistics text
        stats_text = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nBias = {bias:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to {save_path}")
        
        plt.show()
    
    def plot_time_series_at_point(self, pred: xr.DataArray, obs: xr.DataArray,
                                  lat: float, lon: float, title: str,
                                  ylabel: str, save_path: Optional[str] = None):
        """Plot time series comparison at a specific point"""
        # Extract time series
        pred_ts = pred.sel(lat=lat, lon=lon, method='nearest')
        obs_ts = obs.sel(lat=lat, lon=lon, method='nearest')
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        pred_ts.plot(ax=ax, label='Predicted', color='blue', linewidth=1.5)
        obs_ts.plot(ax=ax, label='Observed (ERA5)', color='red', linewidth=1.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} at ({lat:.2f}°N, {lon:.2f}°E)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to {save_path}")
        
        plt.show()


def print_metrics(metrics: Dict):
    """Print metrics in formatted way"""
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    
    print("\nTemperature:")
    print(f"  RMSE: {metrics['temp_rmse']:.4f} °C")
    print(f"  MAE: {metrics['temp_mae']:.4f} °C")
    print(f"  Bias: {metrics['temp_bias']:.4f} °C")
    print(f"  Spatial Correlation: {metrics['temp_spatial_corr']:.4f}")
    
    print("\nPrecipitation:")
    print(f"  RMSE: {metrics['precip_rmse']:.4f} mm/month")
    print(f"  MAE: {metrics['precip_mae']:.4f} mm/month")
    print(f"  Bias: {metrics['precip_bias']:.4f} mm/month")
    print(f"  Spatial Correlation: {metrics['precip_spatial_corr']:.4f}")
    
    print("\nSeasonal Pattern Correlations:")
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        print(f"  {season}:")
        print(f"    Temperature: {metrics[f'temp_{season}_pattern_corr']:.4f}")
        print(f"    Precipitation: {metrics[f'precip_{season}_pattern_corr']:.4f}")
    
    print("="*80)
