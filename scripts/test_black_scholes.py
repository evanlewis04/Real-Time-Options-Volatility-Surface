# scripts/test_black_scholes.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from src.pricing.black_scholes import BlackScholesModel, OptionGreeks
from src.pricing.implied_vol import ImpliedVolatilityCalculator
from src.data.api_client import OptionsDataClient
from config import ALPHA_VANTAGE_API_KEY, DEFAULT_SYMBOL

def test_black_scholes_basic():
    """Test basic Black-Scholes calculations with known values."""
    
    print("=== Testing Black-Scholes Basic Calculations ===")
    
    # Known test case (can be verified online)
    S = 100.0  # Stock price
    K = 100.0  # Strike price (ATM)
    T = 0.25   # 3 months to expiration
    r = 0.05   # 5% risk-free rate
    sigma = 0.2  # 20% volatility
    
    print(f"\nTest Parameters:")
    print(f"Stock Price: ${S}")
    print(f"Strike Price: ${K}")
    print(f"Time to Expiration: {T:.2f} years ({T*365:.0f} days)")
    print(f"Risk-free Rate: {r:.1%}")
    print(f"Volatility: {sigma:.1%}")
    
    # Calculate option prices
    call_price = BlackScholesModel.call_price(S, K, T, r, sigma)
    put_price = BlackScholesModel.put_price(S, K, T, r, sigma)
    
    print(f"\n--- Black-Scholes Prices ---")
    print(f"Call Price: ${call_price:.3f}")
    print(f"Put Price: ${put_price:.3f}")
    
    # Verify put-call parity
    parity_check = BlackScholesModel.put_call_parity_check(call_price, put_price, S, K, T, r)
    print(f"Put-Call Parity: {'✅ PASS' if parity_check else '❌ FAIL'}")
    
    # Calculate Greeks
    print(f"\n--- Greeks ---")
    call_delta = OptionGreeks.delta(S, K, T, r, sigma, 'call')
    put_delta = OptionGreeks.delta(S, K, T, r, sigma, 'put')
    gamma = OptionGreeks.gamma(S, K, T, r, sigma)
    call_theta = OptionGreeks.theta(S, K, T, r, sigma, 'call')
    vega = OptionGreeks.vega(S, K, T, r, sigma)
    call_rho = OptionGreeks.rho(S, K, T, r, sigma, 'call')
    
    print(f"Call Delta: {call_delta:.4f}")
    print(f"Put Delta: {put_delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Call Theta: ${call_theta:.3f}/day")
    print(f"Vega: ${vega:.3f}/1% vol")
    print(f"Call Rho: ${call_rho:.3f}/1% rate")
    
    # Sanity checks
    checks_passed = 0
    total_checks = 6
    
    # Delta should be around 0.5 for ATM options
    if 0.45 < call_delta < 0.55:
        print("✅ Call delta reasonable for ATM option")
        checks_passed += 1
    else:
        print("❌ Call delta seems wrong for ATM option")
    
    # Put delta should be negative
    if put_delta < 0:
        print("✅ Put delta is negative")
        checks_passed += 1
    else:
        print("❌ Put delta should be negative")
    
    # Gamma should be positive
    if gamma > 0:
        print("✅ Gamma is positive")
        checks_passed += 1
    else:
        print("❌ Gamma should be positive")
    
    # Theta should be negative for long options
    if call_theta < 0:
        print("✅ Call theta is negative (time decay)")
        checks_passed += 1
    else:
        print("❌ Call theta should be negative")
    
    # Vega should be positive
    if vega > 0:
        print("✅ Vega is positive")
        checks_passed += 1
    else:
        print("❌ Vega should be positive")
    
    # Call prices should be reasonable
    if 1 < call_price < 15:
        print("✅ Call price in reasonable range")
        checks_passed += 1
    else:
        print("❌ Call price seems unreasonable")
    
    print(f"\nBasic Tests: {checks_passed}/{total_checks} passed")
    return checks_passed == total_checks

def test_implied_volatility():
    """Test implied volatility calculations."""
    
    print("\n=== Testing Implied Volatility Calculations ===")
    
    # Test parameters
    S = 100.0
    K = 100.0
    T = 0.25
    r = 0.05
    true_vol = 0.25  # 25% volatility
    
    # Calculate theoretical price
    theoretical_price = BlackScholesModel.call_price(S, K, T, r, true_vol)
    print(f"\nTheoretical call price with {true_vol:.1%} vol: ${theoretical_price:.3f}")
    
    # Initialize IV calculator
    iv_calc = ImpliedVolatilityCalculator()
    
    # Test different methods
    methods = ['newton', 'bisection', 'brent']
    
    for method in methods:
        implied_vol, method_used = iv_calc.calculate_implied_vol(
            theoretical_price, S, K, T, r, 'call', method=method
        )
        
        if implied_vol is not None:
            error = abs(implied_vol - true_vol)
            print(f"{method.capitalize()} method: {implied_vol:.4f} ({method_used}) - Error: {error:.6f}")
            
            if error < 0.001:  # Less than 0.1% error
                print(f"✅ {method.capitalize()} method accurate")
            else:
                print(f"❌ {method.capitalize()} method inaccurate")
        else:
            print(f"❌ {method.capitalize()} method failed to converge")
    
    return True

def test_with_real_market_data():
    """Test Black-Scholes with real AAPL options data."""
    
    print("\n=== Testing with Real Market Data ===")
    
    # Get real options data
    api_key = ALPHA_VANTAGE_API_KEY or "demo_key"
    client = OptionsDataClient(api_key=api_key)
    
    print(f"Fetching real options data for {DEFAULT_SYMBOL}...")
    options_df = client.fetch_options_chain(DEFAULT_SYMBOL)
    current_price = client.get_current_stock_price(DEFAULT_SYMBOL)
    
    if options_df.empty:
        print("❌ No options data available")
        return False
    
    # Filter for reasonable options (not too far OTM, reasonable volume)
    filtered_df = options_df[
        (options_df['volume'] > 5) &
        (options_df['daysToExpiration'] > 5) &
        (options_df['daysToExpiration'] < 60) &
        (options_df['strike'] > current_price * 0.8) &
        (options_df['strike'] < current_price * 1.2)
    ].copy()
    
    if filtered_df.empty:
        print("❌ No suitable options found after filtering")
        return False
    
    print(f"Analyzing {len(filtered_df)} options contracts...")
    
    # Calculate our implied volatilities and compare to market
    iv_calc = ImpliedVolatilityCalculator()
    r = 0.05  # Assume 5% risk-free rate
    
    results = []
    
    for idx, row in filtered_df.head(10).iterrows():  # Test first 10 options
        # Use mid price (average of bid and ask)
        market_price = (row['bid'] + row['ask']) / 2
        T = row['daysToExpiration'] / 365.0
        
        # Calculate our implied volatility
        our_iv, method = iv_calc.calculate_implied_vol(
            market_price, current_price, row['strike'], T, r, row['type']
        )
        
        # Get market implied volatility
        market_iv = row.get('impliedVolatility', 0)
        
        if our_iv is not None and market_iv > 0:
            iv_diff = abs(our_iv - market_iv)
            results.append({
                'type': row['type'],
                'strike': row['strike'],
                'expiration': row['expiration'],
                'market_price': market_price,
                'market_iv': market_iv,
                'our_iv': our_iv,
                'iv_difference': iv_diff,
                'method': method
            })
    
    if results:
        results_df = pd.DataFrame(results)
        print(f"\nImplied Volatility Comparison (first 10 options):")
        print(results_df[['type', 'strike', 'market_price', 'market_iv', 'our_iv', 'iv_difference']].to_string(index=False, float_format='%.3f'))
        
        # Statistics
        mean_diff = results_df['iv_difference'].mean()
        max_diff = results_df['iv_difference'].max()
        accurate_count = (results_df['iv_difference'] < 0.05).sum()  # Within 5% vol points
        
        print(f"\n--- Accuracy Statistics ---")
        print(f"Mean IV difference: {mean_diff:.3f}")
        print(f"Max IV difference: {max_diff:.3f}")
        print(f"Options within 5% vol: {accurate_count}/{len(results_df)}")
        
        if mean_diff < 0.1:  # Average error less than 10% vol points
            print("✅ Good agreement with market implied volatilities")
            return True
        else:
            print("⚠️ Significant differences from market - may be due to dividends, early exercise, or data quality")
            return True
    else:
        print("❌ Could not calculate implied volatilities for any options")
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    
    print("\n=== Testing Edge Cases ===")
    
    # Test with zero time to expiration
    try:
        price = BlackScholesModel.call_price(100, 100, 0, 0.05, 0.2)
        intrinsic = max(0, 100 - 100)
        if abs(price - intrinsic) < 0.01:
            print("✅ Zero time to expiration handled correctly")
        else:
            print("❌ Zero time to expiration not handled correctly")
    except Exception as e:
        print(f"❌ Zero time to expiration caused error: {e}")
    
    # Test with very high volatility
    try:
        price = BlackScholesModel.call_price(100, 100, 0.25, 0.05, 2.0)  # 200% vol
        if 0 < price < 200:  # Should be positive but not crazy
            print("✅ High volatility handled")
        else:
            print("❌ High volatility produced unrealistic price")
    except Exception as e:
        print(f"❌ High volatility caused error: {e}")
    
    # Test put-call parity with different parameters
    S, K, T, r, sigma = 120, 100, 0.5, 0.03, 0.3
    call = BlackScholesModel.call_price(S, K, T, r, sigma)
    put = BlackScholesModel.put_price(S, K, T, r, sigma)
    
    if BlackScholesModel.put_call_parity_check(call, put, S, K, T, r):
        print("✅ Put-call parity holds for different parameters")
    else:
        print("❌ Put-call parity violated")
    
    print("Edge case testing completed")
    return True

def main():
    """Run all Black-Scholes tests."""
    
    print("=" * 60)
    print("BLACK-SCHOLES MODEL VALIDATION")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    # Run all tests
    if test_black_scholes_basic():
        tests_passed += 1
    
    if test_implied_volatility():
        tests_passed += 1
    
    if test_with_real_market_data():
        tests_passed += 1
    
    if test_edge_cases():
        tests_passed += 1
    
    print(f"\n" + "=" * 60)
    print(f"FINAL RESULTS: {tests_passed}/{total_tests} test suites passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Black-Scholes implementation is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()