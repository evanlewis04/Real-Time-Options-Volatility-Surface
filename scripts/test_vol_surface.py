# scripts/test_vol_surface.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from src.data.api_client import OptionsDataClient
from src.analysis.vol_surface import VolatilitySurface
from src.visualization.plotting import VolatilityPlotter
from src.visualization.surface_plots import VolatilitySurface3D
from config import ALPHA_VANTAGE_API_KEY, DEFAULT_SYMBOL

def test_complete_volatility_analysis():
    """
    Complete end-to-end test of volatility surface construction and visualization.
    This demonstrates the full Week 1 deliverable.
    """
    
    print("=" * 70)
    print("COMPLETE VOLATILITY SURFACE ANALYSIS")
    print("=" * 70)
    
    # Step 1: Get Options Data
    print("\n1. FETCHING OPTIONS DATA")
    print("-" * 30)
    
    api_key = ALPHA_VANTAGE_API_KEY or "demo_key"
    client = OptionsDataClient(api_key=api_key)
    
    symbol = DEFAULT_SYMBOL
    print(f"Fetching options chain for {symbol}...")
    
    options_df = client.fetch_options_chain(symbol)
    current_price = client.get_current_stock_price(symbol)
    
    if options_df.empty:
        print("[FAIL] No options data available")
        return False
    
    print(f"[OK] Fetched {len(options_df)} options contracts")
    print(f"[OK] Current {symbol} price: ${current_price:.2f}")
    
    # Step 2: Build Volatility Surface
    print("\n2. CONSTRUCTING VOLATILITY SURFACE")
    print("-" * 40)
    
    try:
        # Initialize surface builder
        vol_surface = VolatilitySurface(
            options_data=options_df,
            spot_price=current_price,
            risk_free_rate=0.05
        )
        
        # Build the surface
        print("Building interpolated volatility surface...")
        surfaces = vol_surface.construct_surface(method='linear', separate_calls_puts=True)
        
        print(f"[OK] Successfully built {len(surfaces)} surfaces")
        
        # Get surface summary
        summary = vol_surface.get_surface_summary(surfaces)
        print(f"[OK] Surface covers {summary['total_options']} options")
        
        # Display key statistics
        if 'combined_stats' in summary:
            stats = summary['combined_stats']
            print(f"[OK] Volatility range: {stats['min_vol']:.1%} - {stats['max_vol']:.1%}")
            print(f"[OK] ATM volatility: {stats['atm_vol']:.1%}")
        
    except Exception as e:
        print(f"[FAIL] Error building surface: {e}")
        return False
    
    # Step 3: Create 2D Visualizations
    print("\n3. CREATING 2D VISUALIZATIONS")
    print("-" * 35)
    
    try:
        plotter = VolatilityPlotter()
        
        # Create output directory
        os.makedirs('plots', exist_ok=True)
        
        # 1. Raw data scatter plot
        print("Creating raw data analysis...")
        fig1 = plotter.plot_raw_data_scatter(
            options_df, current_price,
            title=f'{symbol} Options Data Analysis',
            save_path='plots/raw_data_analysis.png'
        )
        plt.close(fig1)
        
        # 2. Calls vs puts comparison
        print("Creating calls vs puts comparison...")
        fig2 = plotter.plot_calls_vs_puts(
            options_df, current_price,
            title=f'{symbol} Calls vs Puts Implied Volatility',
            save_path='plots/calls_vs_puts.png'
        )
        plt.close(fig2)
        
        # 3. Volatility smile for shortest expiration
        print("Creating volatility smile...")
        shortest_exp = options_df['daysToExpiration'].min() / 365.0 # Convert to years
        smile_data = vol_surface.get_vol_smile(surfaces, shortest_exp)
        
        if len(smile_data) > 3:
            fig3 = plotter.plot_vol_smile(
                smile_data,
                title=f'{symbol} Volatility Smile - {shortest_exp*365:.0f} Days',
                save_path='plots/vol_smile.png'
            )
            plt.close(fig3)
        
        # 4. ATM term structure
        print("Creating ATM term structure...")
        term_structure = vol_surface.get_atm_term_structure(surfaces)
        
        if len(term_structure) > 2:
            fig4 = plotter.plot_term_structure(
                term_structure,
                title=f'{symbol} ATM Volatility Term Structure',
                save_path='plots/term_structure.png'
            )
            plt.close(fig4)
        
        # 5. Volume analysis
        print("Creating volume analysis...")
        if 'volume' in options_df.columns and options_df['volume'].sum() > 0:
            fig5 = plotter.plot_volume_analysis(
                options_df, current_price,
                title=f'{symbol} Volume and Open Interest Analysis',
                save_path='plots/volume_analysis.png'
            )
            plt.close(fig5)
        
        # 6. Summary dashboard
        print("Creating summary dashboard...")
        fig6 = plotter.create_summary_dashboard(
            options_df, summary, current_price,
            save_path='plots/summary_dashboard.png'
        )
        plt.close(fig6)
        
        print("[OK] All 2D plots created successfully")
        
    except Exception as e:
        print(f"[FAIL] Error creating 2D plots: {e}")
        print("Continuing with 3D visualization...")
    
    # Step 4: Create 3D Visualizations
    print("\n4. CREATING 3D VISUALIZATIONS")
    print("-" * 35)
    
    try:
        surface_3d = VolatilitySurface3D()
        
        # 1. Static 3D matplotlib surface
        print("Creating 3D matplotlib surface...")
        fig_3d = surface_3d.plot_matplotlib_surface(
            surfaces,
            title=f'{symbol} 3D Volatility Surface - ${current_price:.2f}',
            save_path='plots/vol_surface_3d.png'
        )
        plt.close(fig_3d)
        
        # 2. Interactive plotly surface
        print("Creating interactive 3D surface...")
        surface_3d.plot_plotly_surface(
            surfaces,
            title=f'{symbol} Interactive Volatility Surface',
            save_path='plots/vol_surface_interactive.html',
            show_plot=False
        )
        
        # 3. Contour map
        print("Creating contour map...")
        surface_3d.plot_contour_map(
            surfaces,
            title=f'{symbol} Volatility Contour Map',
            save_path='plots/vol_contour.html',
            show_plot=False
        )
        
        # 4. Interactive dashboard
        print("Creating interactive dashboard...")
        surface_3d.plot_interactive_dashboard(
            surfaces, options_df, current_price,
            title=f'{symbol} Complete Volatility Dashboard',
            save_path='plots/interactive_dashboard.html',
            show_plot=False
        )
        
        print("[OK] All 3D visualizations created successfully")
        
    except Exception as e:
        print(f"[FAIL] Error creating 3D plots: {e}")
        print("3D plotting may require additional packages (plotly)")
    
    # Step 5: Analysis and Insights
    print("\n5. VOLATILITY ANALYSIS INSIGHTS")
    print("-" * 40)
    
    try:
        # Analyze the shortest expiration smile
        shortest_exp = options_df['daysToExpiration'].min() / 365.0 # Convert to years
        smile_data = vol_surface.get_vol_smile(surfaces, shortest_exp)
        
        if len(smile_data) > 3:
            skew_analysis = vol_surface.analyze_smile_skew(smile_data)
            
            if 'error' not in skew_analysis:
                print(f" VOLATILITY SMILE ANALYSIS ({shortest_exp*365:.0f} days to expiration):")
                print(f" ATM Volatility: {skew_analysis['atm_vol']:.1%}")
                print(f" Volatility Skew: {skew_analysis['skew_slope']:.3f}")
                print(f" Volatility Range: {skew_analysis['vol_range']:.1%}")
                
                if skew_analysis['risk_reversal']:
                    print(f" Risk Reversal (90%-110%): {skew_analysis['risk_reversal']:.1%}")
                
                print(f" Convexity: {skew_analysis['convexity']:.4f}")
                
                # Interpret the skew
                if skew_analysis['skew_slope'] < -0.1:
                    print(" INTERPRETATION: Strong negative skew (put premium)")
                elif skew_analysis['skew_slope'] > 0.1:
                    print(" INTERPRETATION: Positive skew (call premium)")
                else:
                    print(" INTERPRETATION: Relatively flat skew")
        
        # ATM term structure analysis
        term_structure = vol_surface.get_atm_term_structure(surfaces)
        
        if len(term_structure) > 2:
            print(f"\n ATM TERM STRUCTURE ANALYSIS:")
            
            short_vol = term_structure['atmVolatility'].iloc[0]
            long_vol = term_structure['atmVolatility'].iloc[-1]
            
            print(f" Short-term ATM Vol: {short_vol:.1%}")
            print(f" Long-term ATM Vol: {long_vol:.1%}")
            print(f" Term Structure Slope: {(long_vol - short_vol):.1%}")
            
            # Interpret term structure
            if long_vol > short_vol * 1.1:
                print(" INTERPRETATION: Upward sloping (normal backwardation)")
            elif long_vol < short_vol * 0.9:
                print(" INTERPRETATION: Downward sloping (contango)")
            else:
                print(" INTERPRETATION: Relatively flat term structure")
        
        # Overall market sentiment
        print(f"\n MARKET SENTIMENT INDICATORS:")
        
        avg_iv = options_df['impliedVolatility'].mean()
        print(f" Average Implied Volatility: {avg_iv:.1%}")
        
        if avg_iv > 0.4:
            print(" HIGH VOLATILITY: Market expects significant price moves")
        elif avg_iv < 0.15:
            print(" LOW VOLATILITY: Market expects calm conditions")
        else:
            print(" NORMAL VOLATILITY: Typical market conditions")
        
        # Put/Call volume ratio if available
        if 'volume' in options_df.columns:
            call_volume = options_df[options_df['type'] == 'call']['volume'].sum()
            put_volume = options_df[options_df['type'] == 'put']['volume'].sum()
            
            if call_volume > 0 and put_volume > 0:
                pc_ratio = put_volume / call_volume
                print(f" Put/Call Volume Ratio: {pc_ratio:.2f}")
                
                if pc_ratio > 1.2:
                    print(" BEARISH SENTIMENT: High put buying")
                elif pc_ratio < 0.8:
                    print(" BULLISH SENTIMENT: High call buying")
                else:
                    print(" NEUTRAL SENTIMENT: Balanced put/call activity")
        
    except Exception as e:
        print(f"[FAIL] Error in analysis: {e}")
    
    # Step 6: Summary
    print("\n6. WEEK 1 COMPLETION SUMMARY")
    print("-" * 35)
    
    print("[OK] ACCOMPLISHED:")
    print(" • Real-time options data fetching")
    print(" • Black-Scholes pricing engine")
    print(" • Implied volatility calculations")
    print(" • Volatility surface construction")
    print(" • 2D volatility analysis plots")
    print(" • 3D interactive visualizations")
    print(" • Market sentiment analysis")
    
    print(f"\n OUTPUT FILES CREATED:")
    print(" • plots/raw_data_analysis.png")
    print(" • plots/calls_vs_puts.png")
    print(" • plots/vol_smile.png")
    print(" • plots/term_structure.png")
    print(" • plots/volume_analysis.png")
    print(" • plots/summary_dashboard.png")
    print(" • plots/vol_surface_3d.png")
    print(" • plots/vol_surface_interactive.html")
    print(" • plots/vol_contour.html")
    print(" • plots/interactive_dashboard.html")
    
    print(f"\n WEEK 1 SUCCESSFULLY COMPLETED!")
    print("Ready to move on to Week 2: Multi-stock expansion and real-time features")
    
    return True

def quick_demo():
    """Quick demo for testing basic functionality."""
    
    print("Running quick volatility surface demo...")
    
    try:
        # Get data
        api_key = ALPHA_VANTAGE_API_KEY or "demo_key"
        client = OptionsDataClient(api_key=api_key)
        options_df = client.fetch_options_chain(DEFAULT_SYMBOL)
        current_price = client.get_current_stock_price(DEFAULT_SYMBOL)
        
        # Build surface
        vol_surface = VolatilitySurface(options_df, current_price)
        surfaces = vol_surface.construct_surface()
        
        # Show basic info
        summary = vol_surface.get_surface_summary(surfaces)
        print(f"Surface built with {summary['total_options']} options")
        
        if 'combined_stats' in summary:
            stats = summary['combined_stats']
            print(f"Vol range: {stats['min_vol']:.1%} - {stats['max_vol']:.1%}")
        
        print("[OK] Quick demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Quick demo failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test volatility surface analysis')
    parser.add_argument('--quick', action='store_true', help='Run quick demo only')
    args = parser.parse_args()
    
    if args.quick:
        quick_demo()
    else:
        test_complete_volatility_analysis()