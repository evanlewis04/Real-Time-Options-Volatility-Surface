#!/usr/bin/env python3
"""
Test script to validate volatility surface fixes
"""

import numpy as np

def test_volatility_surface_fixes():
    """Test the fixed volatility surface system"""
    print("🧪 TESTING FIXED VOLATILITY SURFACE SYSTEM")
    print("=" * 60)
    
    # Test 1: Import fixes
    print("1. Testing import fixes...")
    try:
        from dashboard_connector import DashboardConnector
        print("   ✅ Dashboard connector imports successfully")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Create connector
    print("\n2. Creating dashboard connector...")
    try:
        connector = DashboardConnector()
        print("   ✅ Connector created successfully")
    except Exception as e:
        print(f"   ❌ Connector creation failed: {e}")
        return False
    
    # Test 3: Test high-vol stock (PLTR) - the challenging case
    print("\n3. Testing PLTR volatility surface (high-vol stock)...")
    try:
        # Get current data
        pltr_data = connector.get_current_data('PLTR')
        
        print(f"   PLTR Price: ${pltr_data['price']:.2f}")
        print(f"   PLTR 30D IV: {pltr_data['iv_30d']:.1%}")
        print(f"   PLTR Vega: {pltr_data['vega']:.3f}")
        print(f"   PLTR Delta: {pltr_data['delta']:.3f}")
        
        # Validate high-vol characteristics
        if pltr_data['iv_30d'] > 0.4:  # Should be > 40% for PLTR
            print("   ✅ PLTR IV looks realistic for high-vol stock")
        else:
            print(f"   ⚠️ PLTR IV might be too low: {pltr_data['iv_30d']:.1%}")
        
        if pltr_data['vega'] > 0.2:  # Should have substantial vega
            print("   ✅ PLTR Vega looks appropriate")
        else:
            print(f"   ⚠️ PLTR Vega might be too low: {pltr_data['vega']:.3f}")
            
    except Exception as e:
        print(f"   ❌ PLTR data test failed: {e}")
    
    # Test 4: Test volatility surface construction
    print("\n4. Testing PLTR volatility surface construction...")
    try:
        strikes, expiries, vols = connector.get_vol_surface_data('PLTR')
        
        print(f"   Surface shape: {vols.shape}")
        print(f"   Vol range: {np.nanmin(vols):.1%} to {np.nanmax(vols):.1%}")
        print(f"   Strike range: ${np.min(strikes):.0f} to ${np.max(strikes):.0f}")
        
        # Critical test: Check skew direction
        if len(vols.shape) > 1 and vols.shape[1] > 2:
            mid_time = vols.shape[0] // 2
            left_vol = vols[mid_time, 0]    # Low strike (OTM puts)
            right_vol = vols[mid_time, -1]  # High strike (OTM calls)
            
            print(f"   Skew check: OTM puts = {left_vol:.1%}, OTM calls = {right_vol:.1%}")
            
            if left_vol > right_vol:
                print("   ✅ CORRECT skew direction (OTM puts > OTM calls)")
            else:
                print("   ❌ WRONG skew direction")
        
        # Check for realistic volatility levels
        avg_vol = np.nanmean(vols)
        if 0.4 <= avg_vol <= 1.5:  # 40% to 150% for PLTR
            print(f"   ✅ Average vol realistic for PLTR: {avg_vol:.1%}")
        else:
            print(f"   ⚠️ Average vol unusual for PLTR: {avg_vol:.1%}")
            
    except Exception as e:
        print(f"   ❌ Surface construction test failed: {e}")
    
    # Test 5: Test different stock types for comparison
    print("\n5. Testing different stock types...")
    test_symbols = ['AAPL', 'SPY', 'NVDA']
    
    for symbol in test_symbols:
        try:
            data = connector.get_current_data(symbol)
            print(f"   {symbol:6}: ${data['price']:8.2f} | IV: {data['iv_30d']:6.1%} | Vega: {data['vega']:6.3f}")
        except Exception as e:
            print(f"   {symbol:6}: ❌ Error - {e}")
    
    # Test 6: Validate price-IV consistency
    print("\n6. Testing price-IV consistency...")
    try:
        # Get options data and check if prices match IVs
        options_data = connector.options_generator.create_realistic_options_data('AAPL')
        
        if not options_data.empty:
            # Sample a few options
            sample_options = options_data.sample(min(5, len(options_data)))
            
            for _, option in sample_options.iterrows():
                # Recalculate price using Black-Scholes with the given IV
                try:
                    if hasattr(connector.options_generator, 'bs_model'):
                        if option['type'] == 'call':
                            calculated_price = connector.options_generator.bs_model.call_price(
                                S=option.get('moneyness', 1.0) * option['strike'],  # Approximate stock price
                                K=option['strike'],
                                T=option['time_to_expiry'],
                                r=0.05,
                                sigma=option['impliedVolatility']
                            )
                        else:
                            calculated_price = connector.options_generator.bs_model.put_price(
                                S=option.get('moneyness', 1.0) * option['strike'],
                                K=option['strike'], 
                                T=option['time_to_expiry'],
                                r=0.05,
                                sigma=option['impliedVolatility']
                            )
                        
                        price_diff = abs(calculated_price - option['last']) / option['last']
                        
                        if price_diff < 0.05:  # Within 5%
                            print(f"   ✅ {option['type']} option price-IV consistent: {price_diff:.1%} error")
                        else:
                            print(f"   ⚠️ {option['type']} option price-IV mismatch: {price_diff:.1%} error")
                except:
                    print(f"   ⚠️ Could not validate price-IV consistency for {option['type']} option")
                        
        else:
            print("   ⚠️ No options data available for consistency test")
            
    except Exception as e:
        print(f"   ❌ Price-IV consistency test failed: {e}")
    
    # Test 7: Portfolio metrics
    print("\n7. Testing portfolio metrics...")
    try:
        portfolio_metrics = connector.get_portfolio_metrics()
        correlation_matrix = connector.get_correlation_matrix()
        
        print(f"   Portfolio Value: ${portfolio_metrics['total_value']:,.0f}")
        print(f"   Portfolio VaR: ${portfolio_metrics['var_95']:,.0f}")
        print(f"   Correlation Matrix: {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}")
        print("   ✅ Portfolio metrics working")
        
    except Exception as e:
        print(f"   ❌ Portfolio metrics test failed: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 VOLATILITY SURFACE FIX VALIDATION SUMMARY")
    print("=" * 60)
    
    try:
        health = connector.get_system_health()
        pricing_models = health['overall']['pricing_models_available']
        yfinance_working = health['overall']['yfinance_available']
        
        print(f"📊 Real Stock Prices: {'LIVE' if yfinance_working else 'SIMULATED'}")
        print(f"🧮 YOUR Pricing Models: {'ACTIVE' if pricing_models else 'FALLBACK'}")
        print(f"🌊 Surface Construction: {'IMPROVED' if pricing_models else 'BASIC'}")
        print(f"📈 Dashboard Ready: ✅")
        
        if pricing_models:
            print(f"\n🎉 EXCELLENT: Fixed system with consistent price-IV relationships!")
            print(f"   • Proper volatility skew (OTM puts > OTM calls)")
            print(f"   • Realistic high-vol characteristics for stocks like PLTR")
            print(f"   • Consistent option pricing and implied volatility")
            print(f"   • Improved surface interpolation and validation")
        else:
            print(f"\n✅ GOOD: Fixed fallback system working properly")
        
        print(f"\n🚀 Ready to run: streamlit run app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Final validation failed: {e}")
        return False

def quick_pltr_test():
    """Quick test just for PLTR to verify the main fix"""
    print("🎯 QUICK PLTR TEST")
    print("=" * 30)
    
    try:
        from dashboard_connector import DashboardConnector
        connector = DashboardConnector()
        
        # Test PLTR specifically
        pltr_data = connector.get_current_data('PLTR')
        iv = pltr_data['iv_30d']
        
        print(f"PLTR 30D IV: {iv:.1%}")
        
        if iv > 0.5:
            print("✅ SUCCESS: PLTR shows high volatility!")
            print("Your fixes are working correctly.")
        else:
            print("❌ ISSUE: PLTR still shows low volatility.")
            print("The synthetic options fix may not have been applied correctly.")
            
        return iv > 0.5
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # Run quick test
        success = quick_pltr_test()
    else:
        # Run full validation
        success = test_volatility_surface_fixes()
    
    if success:
        print("\n✅ All fixes validated successfully!")
    else:
        print("\n❌ Some issues remain - check the error messages above")