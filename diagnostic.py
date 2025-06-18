#!/usr/bin/env python3
"""
Debug script to identify issues in your Black-Scholes and ImpliedVol implementations
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Import your models
try:
    from src.pricing.black_scholes import BlackScholesModel, OptionGreeks
    from src.pricing.implied_vol import ImpliedVolatilityCalculator
    from src.analysis.vol_surface import VolatilitySurface
    MODELS_AVAILABLE = True
    print("✅ Successfully imported your models")
except ImportError as e:
    print(f"❌ Failed to import models: {e}")
    MODELS_AVAILABLE = False
    sys.exit(1)

def debug_black_scholes():
    """Debug Black-Scholes implementation"""
    print("\n🔍 DEBUGGING BLACK-SCHOLES MODEL")
    print("=" * 50)
    
    bs = BlackScholesModel()
    
    # Test with PLTR-like parameters
    S = 62.0    # PLTR current price
    K = 62.0    # ATM strike
    T = 30/365  # 30 days
    r = 0.05    # 5% risk-free rate
    
    print("Testing with PLTR parameters:")
    print(f"Stock: ${S}, Strike: ${K}, Time: {T:.3f}Y, Rate: {r:.1%}")
    
    # Test different volatilities
    test_vols = [0.20, 0.50, 0.75, 1.00, 1.50]
    
    for sigma in test_vols:
        try:
            call_price = bs.call_price(S, K, T, r, sigma)
            put_price = bs.put_price(S, K, T, r, sigma)
            
            print(f"Vol {sigma:5.1%}: Call=${call_price:6.3f}, Put=${put_price:6.3f}")
            
            # Check for issues
            if call_price <= 0:
                print(f"  ❌ ERROR: Call price is {call_price}")
            if put_price <= 0:
                print(f"  ❌ ERROR: Put price is {put_price}")
            if not np.isfinite(call_price):
                print(f"  ❌ ERROR: Call price is not finite")
            if not np.isfinite(put_price):
                print(f"  ❌ ERROR: Put price is not finite")
                
        except Exception as e:
            print(f"  ❌ ERROR at vol {sigma:.1%}: {e}")
    
    # Test edge cases
    print(f"\n🔍 Testing edge cases:")
    
    # Very short time
    try:
        T_short = 1/365  # 1 day
        call_short = bs.call_price(S, K, T_short, r, 0.5)
        print(f"1-day ATM call at 50% vol: ${call_short:.3f}")
    except Exception as e:
        print(f"❌ Short time error: {e}")
    
    # Very long time
    try:
        T_long = 2.0  # 2 years
        call_long = bs.call_price(S, K, T_long, r, 0.5)
        print(f"2-year ATM call at 50% vol: ${call_long:.3f}")
    except Exception as e:
        print(f"❌ Long time error: {e}")
    
    # Deep OTM/ITM
    try:
        call_otm = bs.call_price(S, S*1.5, T, r, 0.5)  # Deep OTM call
        put_otm = bs.put_price(S, S*0.5, T, r, 0.5)   # Deep OTM put
        print(f"Deep OTM call: ${call_otm:.3f}, Deep OTM put: ${put_otm:.3f}")
    except Exception as e:
        print(f"❌ Deep OTM error: {e}")

def debug_greeks():
    """Debug Greeks calculations"""
    print("\n🔍 DEBUGGING GREEKS CALCULATIONS")
    print("=" * 50)
    
    # PLTR parameters
    S = 62.0
    K = 62.0
    T = 30/365
    r = 0.05
    sigma = 0.75  # High vol for PLTR
    
    print(f"PLTR ATM option with {sigma:.1%} volatility:")
    
    try:
        delta = OptionGreeks.delta(S, K, T, r, sigma, 'call')
        gamma = OptionGreeks.gamma(S, K, T, r, sigma)
        theta = OptionGreeks.theta(S, K, T, r, sigma, 'call')
        vega = OptionGreeks.vega(S, K, T, r, sigma)
        
        print(f"Delta: {delta:8.4f}")
        print(f"Gamma: {gamma:8.4f}")
        print(f"Theta: {theta:8.4f}")
        print(f"Vega:  {vega:8.4f}")
        
        # Check for realistic ranges
        issues = []
        
        if not (0.0 <= delta <= 1.0):
            issues.append(f"Delta {delta:.3f} outside [0,1] for call")
        if gamma < 0:
            issues.append(f"Gamma {gamma:.4f} is negative")
        if theta > 0:
            issues.append(f"Theta {theta:.4f} is positive (should be negative for long)")
        if vega < 0:
            issues.append(f"Vega {vega:.4f} is negative")
        
        # Check magnitude
        if vega < 0.1:
            issues.append(f"Vega {vega:.4f} seems too low for 75% vol")
        if gamma < 0.001:
            issues.append(f"Gamma {gamma:.4f} seems too low")
        
        if issues:
            print("⚠️ POTENTIAL ISSUES FOUND:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("✅ Greeks look reasonable")
            
    except Exception as e:
        print(f"❌ Greeks calculation error: {e}")

def debug_implied_vol():
    """Debug implied volatility calculation"""
    print("\n🔍 DEBUGGING IMPLIED VOLATILITY CALCULATION")
    print("=" * 50)
    
    iv_calc = ImpliedVolatilityCalculator()
    bs = BlackScholesModel()
    
    # Test parameters
    S = 62.0
    K = 62.0
    T = 30/365
    r = 0.05
    true_vol = 0.75  # What we expect for PLTR
    
    print(f"Test: Calculate market price at {true_vol:.1%} vol, then back out IV")
    
    try:
        # Step 1: Calculate market price using known volatility
        market_price = bs.call_price(S, K, T, r, true_vol)
        print(f"Market price at {true_vol:.1%} vol: ${market_price:.3f}")
        
        if market_price <= 0:
            print(f"❌ ERROR: Market price is {market_price}")
            return
        
        # Step 2: Calculate implied volatility from market price
        implied_vol, method = iv_calc.calculate_implied_vol(
            market_price, S, K, T, r, 'call'
        )
        
        if implied_vol is None:
            print(f"❌ ERROR: Could not calculate implied volatility")
            return
        
        print(f"Calculated IV: {implied_vol:.1%} (method: {method})")
        print(f"Error: {abs(implied_vol - true_vol)/true_vol:.1%}")
        
        if abs(implied_vol - true_vol) < 0.01:  # Within 1%
            print("✅ IV calculation is accurate")
        else:
            print(f"⚠️ IV calculation has significant error")
            
        # Test different volatility levels
        print(f"\nTesting multiple volatility levels:")
        test_vols = [0.20, 0.40, 0.60, 0.80, 1.00]
        
        for test_vol in test_vols:
            try:
                market_price = bs.call_price(S, K, T, r, test_vol)
                calculated_iv, method = iv_calc.calculate_implied_vol(
                    market_price, S, K, T, r, 'call'
                )
                
                if calculated_iv:
                    error = abs(calculated_iv - test_vol)
                    print(f"  {test_vol:.1%} → {calculated_iv:.1%} (error: {error:.3f})")
                else:
                    print(f"  {test_vol:.1%} → FAILED")
                    
            except Exception as e:
                print(f"  {test_vol:.1%} → ERROR: {e}")
        
    except Exception as e:
        print(f"❌ IV calculation error: {e}")

def debug_surface_construction():
    """Debug volatility surface construction"""
    print("\n🔍 DEBUGGING VOLATILITY SURFACE CONSTRUCTION")
    print("=" * 50)
    
    # Create sample options data for PLTR
    spot_price = 62.0
    options_data = []
    
    # Create a small set of realistic options
    strikes = [50, 55, 60, 62, 65, 70, 75]  # Around PLTR price
    days_to_exp = [7, 30, 60]
    
    bs = BlackScholesModel()
    
    for days in days_to_exp:
        time_to_exp = days / 365
        
        for strike in strikes:
            # Create realistic implied volatility with skew
            moneyness = strike / spot_price
            
            # Base vol with skew (higher vol for lower strikes)
            base_vol = 0.75
            skew_adj = -0.2 * (moneyness - 1.0)  # Negative skew
            smile_adj = 0.1 * (moneyness - 1.0)**2  # Smile
            
            iv = base_vol + skew_adj + smile_adj
            iv = max(0.1, min(iv, 2.0))  # Bounds
            
            try:
                # Calculate option prices
                call_price = bs.call_price(spot_price, strike, time_to_exp, 0.05, iv)
                put_price = bs.put_price(spot_price, strike, time_to_exp, 0.05, iv)
                
                # Add call
                options_data.append({
                    'symbol': 'PLTR',
                    'strike': strike,
                    'expiration': datetime.now(),
                    'daysToExpiration': days,
                    'type': 'call',
                    'last_price': call_price,
                    'volume': 100,
                    'openInterest': 500,
                    'impliedVolatility': iv
                })
                
                # Add put
                options_data.append({
                    'symbol': 'PLTR',
                    'strike': strike,
                    'expiration': datetime.now(),
                    'daysToExpiration': days,
                    'type': 'put',
                    'last_price': put_price,
                    'volume': 80,
                    'openInterest': 400,
                    'impliedVolatility': iv
                })
                
            except Exception as e:
                print(f"❌ Error creating option data for strike {strike}, days {days}: {e}")
    
    if not options_data:
        print("❌ No options data created")
        return
    
    df = pd.DataFrame(options_data)
    print(f"Created {len(df)} option contracts")
    print(f"IV range: {df['impliedVolatility'].min():.1%} to {df['impliedVolatility'].max():.1%}")
    
    # Test surface construction
    try:
        vol_surface = VolatilitySurface(df, spot_price, 0.05)
        surface_dict = vol_surface.construct_surface(method='linear')
        
        if 'combined' in surface_dict:
            surface_data = surface_dict['combined']
            print(f"✅ Surface constructed successfully")
            print(f"Surface shape: {surface_data['implied_vols'].shape}")
            
            # Check for issues
            vols = surface_data['implied_vols']
            if np.any(np.isnan(vols)):
                print(f"⚠️ Surface contains NaN values")
            if np.any(vols <= 0):
                print(f"⚠️ Surface contains non-positive volatilities")
            if np.any(vols > 3.0):
                print(f"⚠️ Surface contains extremely high volatilities (>300%)")
                
            print(f"Surface vol range: {np.nanmin(vols):.1%} to {np.nanmax(vols):.1%}")
            
            # Check if skew is correct direction
            # For equity options, we expect higher vol for lower strikes
            mid_time_idx = vols.shape[0] // 2
            vol_slice = vols[mid_time_idx, :]
            strikes_slice = surface_data['strikes'][mid_time_idx, :]
            
            if len(vol_slice) > 2:
                left_vol = vol_slice[0]   # Lowest strike
                right_vol = vol_slice[-1] # Highest strike
                
                if left_vol > right_vol:
                    print(f"✅ Correct skew: Lower strikes have higher vol ({left_vol:.1%} vs {right_vol:.1%})")
                else:
                    print(f"❌ WRONG skew: Higher strikes have higher vol ({left_vol:.1%} vs {right_vol:.1%})")
        else:
            print(f"❌ Surface construction failed - no 'combined' surface")
            
    except Exception as e:
        print(f"❌ Surface construction error: {e}")

def run_comprehensive_debug():
    """Run all debugging tests"""
    print("🔬 COMPREHENSIVE DEBUG OF YOUR PRICING MODELS")
    print("=" * 60)
    
    if not MODELS_AVAILABLE:
        print("❌ Cannot debug - models not available")
        return
    
    debug_black_scholes()
    debug_greeks()
    debug_implied_vol()
    debug_surface_construction()
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY AND RECOMMENDATIONS")
    print("=" * 60)
    
    print("Based on the debug results above:")
    print("1. Check if Black-Scholes returns reasonable option prices")
    print("2. Verify Greeks are in expected ranges")
    print("3. Ensure IV calculation can round-trip accurately")
    print("4. Confirm surface construction preserves skew direction")
    print("\nCommon issues:")
    print("• Wrong skew direction (higher vol for higher strikes)")
    print("• Volatility levels too low for high-vol stocks like PLTR")
    print("• Greeks calculated with wrong parameters")
    print("• Surface interpolation issues")

if __name__ == "__main__":
    run_comprehensive_debug()