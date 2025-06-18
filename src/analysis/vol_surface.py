# src/analysis/vol_surface.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import griddata, RBFInterpolator
import warnings
from datetime import datetime

class VolatilitySurface:
    """
    Class for constructing and analyzing options volatility surfaces.
    
    Creates smooth interpolated surfaces from discrete options data and
    provides analysis tools for volatility patterns.
    """
    
    def __init__(self, options_data: pd.DataFrame, spot_price: float, 
                 risk_free_rate: float = 0.05):
        """
        Initialize volatility surface with options data.
        
        Args:
            options_data: DataFrame with options data including IV
            spot_price: Current underlying stock price
            risk_free_rate: Risk-free interest rate
        """
        self.options_data = options_data.copy()
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        
        # Process the data
        self._prepare_data()
        
        # Surface data will be stored here
        self.surface_data = {}
        
    def _prepare_data(self):
        """Prepare and clean options data for surface construction."""
        
        # Ensure required columns exist
        required_cols = ['strike', 'expiration', 'impliedVolatility', 'type', 'daysToExpiration']
        missing_cols = [col for col in required_cols if col not in self.options_data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert expiration to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.options_data['expiration']):
            self.options_data['expiration'] = pd.to_datetime(self.options_data['expiration'])
        
        # Calculate time to expiration in years
        self.options_data['timeToExpiration'] = self.options_data['daysToExpiration'] / 365.0
        
        # Calculate moneyness (strike / spot)
        self.options_data['moneyness'] = self.options_data['strike'] / self.spot_price
        
        # Calculate log-moneyness (more common in volatility surface analysis)
        self.options_data['logMoneyness'] = np.log(self.options_data['moneyness'])
        
        # Filter out bad data
        self._clean_data()
        
        print(f"Prepared {len(self.options_data)} options for surface construction")
        
    def _clean_data(self):
        """Clean options data by removing outliers and bad points."""
        
        initial_count = len(self.options_data)
        
        # Remove options with unrealistic implied volatilities
        self.options_data = self.options_data[
            (self.options_data['impliedVolatility'] > 0.05) &  # Min 5% vol
            (self.options_data['impliedVolatility'] < 2.0) &   # Max 200% vol
            (self.options_data['timeToExpiration'] > 0.01) &   # Min 4 days
            (self.options_data['timeToExpiration'] < 2.0)      # Max 2 years
        ]
        
        # Remove extreme moneyness values
        self.options_data = self.options_data[
            (self.options_data['moneyness'] > 0.5) &   # Not too far OTM
            (self.options_data['moneyness'] < 2.0)     # Not too far ITM
        ]
        
        # Remove statistical outliers in IV for each expiration
        cleaned_data = []
        for exp_date in self.options_data['expiration'].unique():
            exp_data = self.options_data[self.options_data['expiration'] == exp_date].copy()
            
            if len(exp_data) < 3:  # Need at least 3 points
                continue
                
            # Remove IV outliers using IQR method
            Q1 = exp_data['impliedVolatility'].quantile(0.25)
            Q3 = exp_data['impliedVolatility'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            exp_data_clean = exp_data[
                (exp_data['impliedVolatility'] >= lower_bound) &
                (exp_data['impliedVolatility'] <= upper_bound)
            ]
            
            cleaned_data.append(exp_data_clean)
        
        if cleaned_data:
            self.options_data = pd.concat(cleaned_data, ignore_index=True)
        
        removed_count = initial_count - len(self.options_data)
        if removed_count > 0:
            print(f"Removed {removed_count} outlier options during cleaning")
    
    def construct_surface(self, method: str = 'linear', separate_calls_puts: bool = False) -> Dict:
        """
        Construct interpolated volatility surface.
        
        Args:
            method: Interpolation method ('linear', 'cubic', 'rbf')
            separate_calls_puts: Whether to build separate surfaces for calls/puts
            
        Returns:
            Dictionary containing surface data and metadata
        """
        
        if separate_calls_puts:
            # Build separate surfaces for calls and puts
            calls_data = self.options_data[self.options_data['type'] == 'call'].copy()
            puts_data = self.options_data[self.options_data['type'] == 'put'].copy()
            
            surfaces = {}
            
            if len(calls_data) > 10:  # Need sufficient data
                surfaces['calls'] = self._build_single_surface(calls_data, method, 'calls')
            
            if len(puts_data) > 10:
                surfaces['puts'] = self._build_single_surface(puts_data, method, 'puts')
            
            # Also build combined surface
            surfaces['combined'] = self._build_single_surface(self.options_data, method, 'combined')
            
            return surfaces
        
        else:
            # Build single combined surface
            return {'combined': self._build_single_surface(self.options_data, method, 'combined')}
    
    def _build_single_surface(self, data: pd.DataFrame, method: str, surface_type: str) -> Dict:
        """Build a single volatility surface from the given data."""
        
        if len(data) < 5:
            raise ValueError(f"Insufficient data for {surface_type} surface: {len(data)} points")
        
        # Extract coordinates and values
        strikes = data['strike'].values
        times = data['timeToExpiration'].values
        ivs = data['impliedVolatility'].values
        
        # Create regular grid for interpolation
        strike_range = np.linspace(strikes.min(), strikes.max(), 50)
        time_range = np.linspace(times.min(), times.max(), 30)
        
        # Create meshgrid
        strike_grid, time_grid = np.meshgrid(strike_range, time_range)
        
        # Prepare points for interpolation
        points = np.column_stack((strikes, times))
        grid_points = np.column_stack((strike_grid.ravel(), time_grid.ravel()))
        
        try:
            if method == 'linear':
                # Linear interpolation
                iv_grid = griddata(points, ivs, grid_points, method='linear')
                
            elif method == 'cubic':
                # Cubic interpolation
                iv_grid = griddata(points, ivs, grid_points, method='cubic')
                
            elif method == 'rbf':
                # Radial basis function interpolation
                rbf = RBFInterpolator(points, ivs, kernel='thin_plate_spline', smoothing=0.1)
                iv_grid = rbf(grid_points)
                
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
            
            # Reshape back to grid
            iv_grid = iv_grid.reshape(strike_grid.shape)
            
            # Handle NaN values by filling with nearest neighbor
            if np.isnan(iv_grid).any():
                iv_grid_filled = griddata(points, ivs, grid_points, method='nearest')
                iv_grid_filled = iv_grid_filled.reshape(strike_grid.shape)
                
                # Replace NaNs in original grid
                nan_mask = np.isnan(iv_grid)
                iv_grid[nan_mask] = iv_grid_filled[nan_mask]
            
            # Calculate moneyness grid
            moneyness_grid = strike_grid / self.spot_price
            
            # Store surface data
            surface = {
                'strikes': strike_grid,
                'times': time_grid,
                'moneyness': moneyness_grid,
                'implied_vols': iv_grid,
                'method': method,
                'type': surface_type,
                'raw_data': data,
                'spot_price': self.spot_price,
                'strike_range': strike_range,
                'time_range': time_range
            }
            
            # Calculate surface statistics
            surface['stats'] = self._calculate_surface_stats(surface)
            
            return surface
            
        except Exception as e:
            print(f"Error building {surface_type} surface with {method} method: {e}")
            # Fallback to linear method
            if method != 'linear':
                return self._build_single_surface(data, 'linear', surface_type)
            else:
                raise
    
    def _calculate_surface_stats(self, surface: Dict) -> Dict:
        """Calculate statistics for the volatility surface."""
        
        iv_grid = surface['implied_vols']
        moneyness_grid = surface['moneyness']
        
        stats = {
            'min_vol': np.nanmin(iv_grid),
            'max_vol': np.nanmax(iv_grid),
            'mean_vol': np.nanmean(iv_grid),
            'vol_range': np.nanmax(iv_grid) - np.nanmin(iv_grid),
            'data_points': len(surface['raw_data'])
        }
        
        # Find ATM volatility (closest to moneyness = 1.0)
        try:
            atm_idx = np.unravel_index(
                np.nanargmin(np.abs(moneyness_grid - 1.0)), 
                moneyness_grid.shape
            )
            stats['atm_vol'] = iv_grid[atm_idx]
        except:
            stats['atm_vol'] = np.nanmean(iv_grid)
        
        # Calculate surface curvature/convexity
        try:
            # Approximate second derivative along strike dimension
            d2_vol_dk2 = np.gradient(np.gradient(iv_grid, axis=1), axis=1)
            stats['mean_convexity'] = np.nanmean(d2_vol_dk2)
        except:
            stats['mean_convexity'] = 0.0
        
        return stats
    
    def get_atm_term_structure(self, surface_data: Dict) -> pd.DataFrame:
        """Extract at-the-money volatility term structure."""
        
        surface = surface_data.get('combined', surface_data.get('calls', list(surface_data.values())[0]))
        
        strikes = surface['strikes']
        times = surface['times']
        ivs = surface['implied_vols']
        moneyness = surface['moneyness']
        
        # Find ATM points for each time slice
        atm_vols = []
        time_points = []
        
        for i, time_slice in enumerate(surface['time_range']):
            # Get moneyness and IV for this time slice
            time_moneyness = moneyness[i, :]
            time_iv = ivs[i, :]
            
            # Find closest to ATM (moneyness = 1.0)
            atm_idx = np.nanargmin(np.abs(time_moneyness - 1.0))
            
            if not np.isnan(time_iv[atm_idx]):
                atm_vols.append(time_iv[atm_idx])
                time_points.append(time_slice)
        
        return pd.DataFrame({
            'timeToExpiration': time_points,
            'atmVolatility': atm_vols,
            'daysToExpiration': [t * 365 for t in time_points]
        })
    
    def get_vol_smile(self, surface_data: Dict, target_expiration: float) -> pd.DataFrame:
        """Extract volatility smile for a specific expiration."""
        
        surface = surface_data.get('combined', surface_data.get('calls', list(surface_data.values())[0]))
        
        # Find closest time slice to target expiration
        time_range = surface['time_range']
        time_idx = np.argmin(np.abs(time_range - target_expiration))
        
        # Extract smile data
        strikes = surface['strikes'][time_idx, :]
        moneyness = surface['moneyness'][time_idx, :]
        ivs = surface['implied_vols'][time_idx, :]
        
        # Remove NaN values
        valid_mask = ~np.isnan(ivs)
        
        return pd.DataFrame({
            'strike': strikes[valid_mask],
            'moneyness': moneyness[valid_mask],
            'impliedVolatility': ivs[valid_mask],
            'expiration': target_expiration
        }).sort_values('moneyness')
    
    def analyze_smile_skew(self, smile_data: pd.DataFrame) -> Dict:
        """Analyze volatility smile/skew characteristics."""
        
        if len(smile_data) < 3:
            return {'error': 'Insufficient data for skew analysis'}
        
        moneyness = smile_data['moneyness'].values
        ivs = smile_data['impliedVolatility'].values
        
        # Calculate skew (slope)
        try:
            # Linear fit to get overall skew
            coeffs = np.polyfit(moneyness, ivs, 1)
            skew_slope = coeffs[0]
            
            # Calculate 25-delta risk reversal (if we have enough range)
            if moneyness.min() < 0.9 and moneyness.max() > 1.1:
                # Interpolate at 90% and 110% moneyness
                iv_90 = np.interp(0.9, moneyness, ivs)
                iv_110 = np.interp(1.1, moneyness, ivs)
                risk_reversal = iv_90 - iv_110
            else:
                risk_reversal = None
            
            # Calculate convexity (curvature)
            if len(smile_data) >= 5:
                coeffs_quad = np.polyfit(moneyness, ivs, 2)
                convexity = 2 * coeffs_quad[0]  # Second derivative
            else:
                convexity = 0
            
            # Find ATM vol
            atm_idx = np.argmin(np.abs(moneyness - 1.0))
            atm_vol = ivs[atm_idx]
            
            return {
                'atm_vol': atm_vol,
                'skew_slope': skew_slope,
                'risk_reversal': risk_reversal,
                'convexity': convexity,
                'vol_range': ivs.max() - ivs.min(),
                'min_vol': ivs.min(),
                'max_vol': ivs.max()
            }
            
        except Exception as e:
            return {'error': f'Error in skew analysis: {e}'}
    
    def get_surface_summary(self, surfaces: Dict) -> Dict:
        """Get comprehensive summary of all surfaces."""
        
        summary = {
            'spot_price': self.spot_price,
            'risk_free_rate': self.risk_free_rate,
            'total_options': len(self.options_data),
            'surface_types': list(surfaces.keys()),
            'construction_time': datetime.now().isoformat()
        }
        
        # Add statistics for each surface
        for surface_name, surface_data in surfaces.items():
            if isinstance(surface_data, dict) and 'stats' in surface_data:
                summary[f'{surface_name}_stats'] = surface_data['stats']
        
        # Get ATM term structure if available
        if 'combined' in surfaces:
            try:
                atm_ts = self.get_atm_term_structure(surfaces)
                summary['atm_term_structure'] = {
                    'min_vol': atm_ts['atmVolatility'].min(),
                    'max_vol': atm_ts['atmVolatility'].max(),
                    'short_term_vol': atm_ts['atmVolatility'].iloc[0] if len(atm_ts) > 0 else None,
                    'long_term_vol': atm_ts['atmVolatility'].iloc[-1] if len(atm_ts) > 0 else None
                }
            except:
                pass
        
        return summary