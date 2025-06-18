# src/pricing/implied_vol.py

import numpy as np
from typing import Optional, Tuple
import warnings
from .black_scholes import BlackScholesModel, OptionGreeks

class ImpliedVolatilityCalculator:
    """
    Calculate implied volatility using Newton-Raphson and other numerical methods.
    
    Implied volatility is the volatility that makes the Black-Scholes theoretical
    price equal to the market price of the option.
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Initialize the implied volatility calculator.
        
        Args:
            max_iterations: Maximum number of iterations for numerical methods
            tolerance: Convergence tolerance for implied volatility
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def newton_raphson(self, market_price: float, S: float, K: float, T: float, 
                      r: float, option_type: str, q: float = 0.0,
                      initial_guess: float = 0.2) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        This is the most efficient method for implied volatility calculation.
        Uses vega (derivative of price w.r.t. volatility) to iterate quickly.
        
        Args:
            market_price: Observed market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            q: Dividend yield
            initial_guess: Starting volatility guess
            
        Returns:
            Implied volatility or None if no convergence
        """
        # Validate inputs
        if not self._validate_inputs(market_price, S, K, T, r, option_type):
            return None
        
        # Check for intrinsic value violations
        if not self._check_arbitrage_bounds(market_price, S, K, T, r, option_type, q):
            return None
        
        sigma = initial_guess
        
        for iteration in range(self.max_iterations):
            # Calculate theoretical price and vega
            try:
                theoretical_price = BlackScholesModel.option_price(S, K, T, r, sigma, option_type, q)
                vega = OptionGreeks.vega(S, K, T, r, sigma, q)
                
                # Newton-Raphson update: sigma_new = sigma - f(sigma)/f'(sigma)
                price_diff = theoretical_price - market_price
                
                # Check for convergence
                if abs(price_diff) < self.tolerance:
                    return sigma
                
                # Check if vega is too small (causes instability)
                if abs(vega) < 1e-10:
                    # Try bisection method as fallback
                    return self.bisection(market_price, S, K, T, r, option_type, q)
                
                # Newton-Raphson step
                sigma_new = sigma - price_diff / (vega * 100)  # vega is per 1% vol change
                
                # Ensure volatility stays positive and reasonable
                sigma_new = max(0.001, min(sigma_new, 5.0))  # Between 0.1% and 500%
                
                # Check for convergence in sigma
                if abs(sigma_new - sigma) < self.tolerance:
                    return sigma_new
                
                sigma = sigma_new
                
            except (ZeroDivisionError, OverflowError, ValueError):
                # Numerical issues - try different starting point
                return self.bisection(market_price, S, K, T, r, option_type, q)
        
        # If we get here, Newton-Raphson didn't converge
        warnings.warn(f"Newton-Raphson did not converge after {self.max_iterations} iterations")
        
        # Try bisection as fallback
        return self.bisection(market_price, S, K, T, r, option_type, q)
    
    def bisection(self, market_price: float, S: float, K: float, T: float,
                 r: float, option_type: str, q: float = 0.0,
                 vol_low: float = 0.001, vol_high: float = 5.0) -> Optional[float]:
        """
        Calculate implied volatility using bisection method.
        
        More robust but slower than Newton-Raphson. Used as fallback.
        
        Args:
            market_price: Observed market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            q: Dividend yield
            vol_low: Lower bound for volatility search
            vol_high: Upper bound for volatility search
            
        Returns:
            Implied volatility or None if no solution found
        """
        # Validate inputs
        if not self._validate_inputs(market_price, S, K, T, r, option_type):
            return None
        
        # Check bounds
        price_low = BlackScholesModel.option_price(S, K, T, r, vol_low, option_type, q)
        price_high = BlackScholesModel.option_price(S, K, T, r, vol_high, option_type, q)
        
        # Market price must be between low and high theoretical prices
        if market_price < price_low or market_price > price_high:
            # Try to expand the search range
            if market_price > price_high:
                vol_high = min(10.0, vol_high * 2)  # Increase upper bound
                price_high = BlackScholesModel.option_price(S, K, T, r, vol_high, option_type, q)
                
                if market_price > price_high:
                    warnings.warn(f"Market price {market_price:.3f} exceeds theoretical maximum {price_high:.3f}")
                    return None
        
        # Bisection algorithm
        for iteration in range(self.max_iterations):
            vol_mid = (vol_low + vol_high) / 2
            price_mid = BlackScholesModel.option_price(S, K, T, r, vol_mid, option_type, q)
            
            # Check for convergence
            if abs(price_mid - market_price) < self.tolerance:
                return vol_mid
            
            # Update bounds
            if price_mid > market_price:
                vol_high = vol_mid
            else:
                vol_low = vol_mid
            
            # Check if search range is too narrow
            if vol_high - vol_low < self.tolerance:
                return (vol_high + vol_low) / 2
        
        warnings.warn(f"Bisection method did not converge after {self.max_iterations} iterations")
        return None
    
    def brent_method(self, market_price: float, S: float, K: float, T: float,
                    r: float, option_type: str, q: float = 0.0) -> Optional[float]:
        """
        Calculate implied volatility using Brent's method.
        
        Combines bisection with inverse quadratic interpolation for faster convergence.
        
        Args:
            market_price: Observed market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            q: Dividend yield
            
        Returns:
            Implied volatility or None if no solution found
        """
        # For simplicity, use scipy's brentq if available, otherwise fall back to bisection
        try:
            from scipy.optimize import brentq
            
            def price_diff(sigma):
                return BlackScholesModel.option_price(S, K, T, r, sigma, option_type, q) - market_price
            
            # Find bounds where function changes sign
            vol_low, vol_high = 0.001, 5.0
            
            # Check if solution exists in range
            if price_diff(vol_low) * price_diff(vol_high) > 0:
                # Try to find better bounds
                vol_high = 10.0
                if price_diff(vol_low) * price_diff(vol_high) > 0:
                    return None
            
            result = brentq(price_diff, vol_low, vol_high, xtol=self.tolerance, maxiter=self.max_iterations)
            return result
            
        except ImportError:
            # Fall back to bisection if scipy not available
            return self.bisection(market_price, S, K, T, r, option_type, q)
        except Exception:
            # Fall back to bisection if brentq fails
            return self.bisection(market_price, S, K, T, r, option_type, q)
    
    def calculate_implied_vol(self, market_price: float, S: float, K: float, T: float,
                            r: float, option_type: str, q: float = 0.0,
                            method: str = 'newton') -> Tuple[Optional[float], str]:
        """
        Calculate implied volatility using the specified method.
        
        Args:
            market_price: Observed market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            q: Dividend yield
            method: 'newton', 'bisection', or 'brent'
            
        Returns:
            Tuple of (implied_volatility, method_used)
        """
        method = method.lower()
        
        if method == 'newton':
            iv = self.newton_raphson(market_price, S, K, T, r, option_type, q)
            if iv is not None:
                return iv, 'newton'
            # Fall back to bisection
            iv = self.bisection(market_price, S, K, T, r, option_type, q)
            return iv, 'bisection' if iv is not None else 'failed'
            
        elif method == 'bisection':
            iv = self.bisection(market_price, S, K, T, r, option_type, q)
            return iv, 'bisection' if iv is not None else 'failed'
            
        elif method == 'brent':
            iv = self.brent_method(market_price, S, K, T, r, option_type, q)
            if iv is not None:
                return iv, 'brent'
            # Fall back to bisection
            iv = self.bisection(market_price, S, K, T, r, option_type, q)
            return iv, 'bisection' if iv is not None else 'failed'
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'newton', 'bisection', or 'brent'")
    
    def _validate_inputs(self, market_price: float, S: float, K: float, T: float,
                        r: float, option_type: str) -> bool:
        """Validate inputs for implied volatility calculation."""
        try:
            # Use Black-Scholes validation
            BlackScholesModel.validate_inputs(S, K, T, r, 0.2)  # Use dummy vol for validation
            
            if market_price <= 0:
                warnings.warn(f"Market price must be positive, got {market_price}")
                return False
            
            if option_type.lower() not in ['call', 'put']:
                warnings.warn(f"option_type must be 'call' or 'put', got {option_type}")
                return False
            
            return True
            
        except ValueError as e:
            warnings.warn(str(e))
            return False
    
    def _check_arbitrage_bounds(self, market_price: float, S: float, K: float, T: float,
                               r: float, option_type: str, q: float = 0.0) -> bool:
        """
        Check if market price violates arbitrage bounds.
        
        Options cannot trade below intrinsic value or above certain upper bounds.
        """
        option_type = option_type.lower()
        
        # Calculate intrinsic value
        if option_type == 'call':
            intrinsic = max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
            upper_bound = S * np.exp(-q * T)  # Call cannot be worth more than stock
        else:  # put
            intrinsic = max(0, K * np.exp(-r * T) - S * np.exp(-q * T))
            upper_bound = K * np.exp(-r * T)  # Put cannot be worth more than strike
        
        # Check bounds
        if market_price < intrinsic * 0.95:  # Allow small tolerance for bid-ask spread
            warnings.warn(f"Market price {market_price:.3f} below intrinsic value {intrinsic:.3f}")
            return False
        
        if market_price > upper_bound * 1.05:  # Allow small tolerance
            warnings.warn(f"Market price {market_price:.3f} above upper bound {upper_bound:.3f}")
            return False
        
        return True


def calculate_smile_metrics(strikes: np.ndarray, implied_vols: np.ndarray, 
                          spot_price: float) -> dict:
    """
    Calculate volatility smile/skew metrics.
    
    Args:
        strikes: Array of strike prices
        implied_vols: Array of corresponding implied volatilities
        spot_price: Current stock price
        
    Returns:
        Dictionary with smile metrics
    """
    # Convert to moneyness (strike / spot)
    moneyness = strikes / spot_price
    
    # Find ATM volatility (closest to moneyness = 1.0)
    atm_idx = np.argmin(np.abs(moneyness - 1.0))
    atm_vol = implied_vols[atm_idx]
    
    # Calculate skew (slope of vol surface)
    # Use 90% and 110% moneyness points if available
    try:
        # Find closest points to 90% and 110% moneyness
        idx_90 = np.argmin(np.abs(moneyness - 0.9))
        idx_110 = np.argmin(np.abs(moneyness - 1.1))
        
        if abs(moneyness[idx_90] - 0.9) < 0.05 and abs(moneyness[idx_110] - 1.1) < 0.05:
            skew = (implied_vols[idx_90] - implied_vols[idx_110]) / 0.2  # 20% moneyness difference
        else:
            # Use linear regression for skew calculation
            coeffs = np.polyfit(moneyness, implied_vols, 1)
            skew = coeffs[0]  # Slope of the line
            
    except Exception:
        skew = 0.0
    
    # Calculate convexity (curvature of smile)
    try:
        # Fit quadratic polynomial and get second derivative
        coeffs = np.polyfit(moneyness, implied_vols, 2)
        convexity = 2 * coeffs[0]  # Second derivative coefficient
    except Exception:
        convexity = 0.0
    
    # Calculate volatility range
    vol_range = np.max(implied_vols) - np.min(implied_vols)
    
    # Term structure slope (if multiple expirations available)
    # This would need to be calculated at a higher level with multiple expiration data
    
    return {
        'atm_vol': atm_vol,
        'skew': skew,
        'convexity': convexity,
        'vol_range': vol_range,
        'min_vol': np.min(implied_vols),
        'max_vol': np.max(implied_vols),
        'vol_std': np.std(implied_vols)
    }