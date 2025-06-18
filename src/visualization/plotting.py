# src/visualization/plotting.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from datetime import datetime

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class VolatilityPlotter:
    """
    Class for creating 2D volatility plots and analysis charts.
    
    Provides various plotting functions for volatility surfaces,
    smiles, term structures, and market data analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the plotter.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = plt.cm.viridis
        
    def plot_vol_smile(self, smile_data: pd.DataFrame, title: str = None,
                      expiration_days: float = None, save_path: str = None) -> plt.Figure:
        """
        Plot volatility smile for a single expiration.
        
        Args:
            smile_data: DataFrame with moneyness and impliedVolatility
            title: Plot title
            expiration_days: Days to expiration for title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort by moneyness for smooth plotting
        smile_data = smile_data.sort_values('moneyness')
        
        # Plot the smile
        ax.plot(smile_data['moneyness'], smile_data['impliedVolatility'], 
               'bo-', linewidth=2, markersize=6, label='Implied Volatility')
        
        # Add ATM line
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='ATM (Moneyness = 1.0)')
        
        # Formatting
        ax.set_xlabel('Moneyness (Strike / Spot)', fontsize=12)
        ax.set_ylabel('Implied Volatility', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        elif expiration_days:
            ax.set_title(f'Volatility Smile - {expiration_days:.0f} Days to Expiration', 
                        fontsize=14, fontweight='bold')
        else:
            ax.set_title('Volatility Smile', fontsize=14, fontweight='bold')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add annotations for min/max volatility
        if len(smile_data) > 0:
            min_vol_idx = smile_data['impliedVolatility'].idxmin()
            max_vol_idx = smile_data['impliedVolatility'].idxmax()
            
            min_point = smile_data.loc[min_vol_idx]
            max_point = smile_data.loc[max_vol_idx]
            
            ax.annotate(f'Min: {min_point["impliedVolatility"]:.1%}',
                       xy=(min_point['moneyness'], min_point['impliedVolatility']),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax.annotate(f'Max: {max_point["impliedVolatility"]:.1%}',
                       xy=(max_point['moneyness'], max_point['impliedVolatility']),
                       xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Volatility smile plot saved to {save_path}")
        
        return fig
    
    def plot_term_structure(self, term_structure: pd.DataFrame, title: str = None,
                           save_path: str = None) -> plt.Figure:
        """
        Plot ATM volatility term structure.
        
        Args:
            term_structure: DataFrame with timeToExpiration and atmVolatility
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort by time to expiration
        term_structure = term_structure.sort_values('timeToExpiration')
        
        # Plot term structure
        ax.plot(term_structure['daysToExpiration'], term_structure['atmVolatility'],
               'ro-', linewidth=2, markersize=6, label='ATM Implied Volatility')
        
        # Formatting
        ax.set_xlabel('Days to Expiration', fontsize=12)
        ax.set_ylabel('ATM Implied Volatility', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('ATM Volatility Term Structure', fontsize=14, fontweight='bold')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add trend line
        if len(term_structure) > 2:
            z = np.polyfit(term_structure['daysToExpiration'], term_structure['atmVolatility'], 1)
            p = np.poly1d(z)
            ax.plot(term_structure['daysToExpiration'], p(term_structure['daysToExpiration']),
                   'g--', alpha=0.7, label=f'Trend (slope: {z[0]*365:.3f}/year)')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Term structure plot saved to {save_path}")
        
        return fig
    
    def plot_multiple_smiles(self, smiles_data: Dict[float, pd.DataFrame], 
                           title: str = None, save_path: str = None) -> plt.Figure:
        """
        Plot multiple volatility smiles for different expirations.
        
        Args:
            smiles_data: Dictionary with expiration (in years) as key and smile data as value
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(smiles_data)))
        
        for i, (expiration, smile_data) in enumerate(sorted(smiles_data.items())):
            smile_data = smile_data.sort_values('moneyness')
            
            days_to_exp = expiration * 365
            ax.plot(smile_data['moneyness'], smile_data['impliedVolatility'],
                   'o-', color=colors[i], linewidth=2, markersize=4,
                   label=f'{days_to_exp:.0f} days')
        
        # Add ATM line
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='ATM')
        
        # Formatting
        ax.set_xlabel('Moneyness (Strike / Spot)', fontsize=12)
        ax.set_ylabel('Implied Volatility', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Volatility Smiles - Multiple Expirations', fontsize=14, fontweight='bold')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multiple smiles plot saved to {save_path}")
        
        return fig
    
    def plot_raw_data_scatter(self, options_data: pd.DataFrame, spot_price: float,
                            title: str = None, save_path: str = None) -> plt.Figure:
        """
        Create scatter plot of raw options data.
        
        Args:
            options_data: Raw options DataFrame
            spot_price: Current stock price
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calculate moneyness
        options_data = options_data.copy()
        options_data['moneyness'] = options_data['strike'] / spot_price
        
        # Plot 1: Strike vs IV colored by days to expiration
        scatter1 = ax1.scatter(options_data['moneyness'], options_data['impliedVolatility'],
                              c=options_data['daysToExpiration'], cmap='viridis',
                              alpha=0.6, s=30)
        
        ax1.set_xlabel('Moneyness (Strike / Spot)', fontsize=12)
        ax1.set_ylabel('Implied Volatility', fontsize=12)
        ax1.set_title('IV vs Moneyness (colored by Days to Expiration)', fontsize=12)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Days to Expiration', fontsize=10)
        
        # Plot 2: Days to expiration vs IV colored by moneyness
        scatter2 = ax2.scatter(options_data['daysToExpiration'], options_data['impliedVolatility'],
                              c=options_data['moneyness'], cmap='RdYlBu',
                              alpha=0.6, s=30)
        
        ax2.set_xlabel('Days to Expiration', fontsize=12)
        ax2.set_ylabel('Implied Volatility', fontsize=12)
        ax2.set_title('IV vs Time (colored by Moneyness)', fontsize=12)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Moneyness', fontsize=10)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        else:
            fig.suptitle('Raw Options Data Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Raw data scatter plot saved to {save_path}")
        
        return fig
    
    def plot_calls_vs_puts(self, options_data: pd.DataFrame, spot_price: float,
                          title: str = None, save_path: str = None) -> plt.Figure:
        """
        Compare call and put implied volatilities.
        
        Args:
            options_data: Options DataFrame with 'type' column
            spot_price: Current stock price
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Separate calls and puts
        calls = options_data[options_data['type'] == 'call'].copy()
        puts = options_data[options_data['type'] == 'put'].copy()
        
        if len(calls) == 0 or len(puts) == 0:
            print("Warning: Missing calls or puts data for comparison")
            return fig
        
        # Calculate moneyness
        calls['moneyness'] = calls['strike'] / spot_price
        puts['moneyness'] = puts['strike'] / spot_price
        
        # Plot calls and puts
        ax.scatter(calls['moneyness'], calls['impliedVolatility'], 
                  alpha=0.6, s=30, color='blue', label='Calls')
        ax.scatter(puts['moneyness'], puts['impliedVolatility'], 
                  alpha=0.6, s=30, color='red', label='Puts')
        
        # Add ATM line
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.7, label='ATM')
        
        # Formatting
        ax.set_xlabel('Moneyness (Strike / Spot)', fontsize=12)
        ax.set_ylabel('Implied Volatility', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Calls vs Puts Implied Volatility', fontsize=14, fontweight='bold')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Calculate and display put-call parity violations
        # Find matching strikes and expirations
        call_put_pairs = []
        for _, call in calls.iterrows():
            matching_puts = puts[
                (abs(puts['strike'] - call['strike']) < 0.01) &
                (puts['expiration'] == call['expiration'])
            ]
            
            if len(matching_puts) > 0:
                put = matching_puts.iloc[0]
                iv_diff = abs(call['impliedVolatility'] - put['impliedVolatility'])
                call_put_pairs.append(iv_diff)
        
        if call_put_pairs:
            mean_iv_diff = np.mean(call_put_pairs)
            ax.text(0.02, 0.98, f'Mean Call-Put IV Diff: {mean_iv_diff:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calls vs puts plot saved to {save_path}")
        
        return fig
    
    def plot_volume_analysis(self, options_data: pd.DataFrame, spot_price: float,
                           title: str = None, save_path: str = None) -> plt.Figure:
        """
        Analyze option volume and open interest patterns.
        
        Args:
            options_data: Options DataFrame with volume/openInterest columns
            spot_price: Current stock price
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate moneyness
        options_data = options_data.copy()
        options_data['moneyness'] = options_data['strike'] / spot_price
        
        # Filter for reasonable data
        data_filtered = options_data[
            (options_data['volume'] > 0) & 
            (options_data['openInterest'] > 0)
        ].copy()
        
        if len(data_filtered) == 0:
            print("Warning: No volume/open interest data available")
            return fig
        
        # Plot 1: Volume vs Moneyness
        ax1.scatter(data_filtered['moneyness'], data_filtered['volume'], 
                   alpha=0.6, s=30, color='green')
        ax1.set_xlabel('Moneyness')
        ax1.set_ylabel('Volume')
        ax1.set_title('Volume vs Moneyness')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
        
        # Plot 2: Open Interest vs Moneyness
        ax2.scatter(data_filtered['moneyness'], data_filtered['openInterest'], 
                   alpha=0.6, s=30, color='purple')
        ax2.set_xlabel('Moneyness')
        ax2.set_ylabel('Open Interest')
        ax2.set_title('Open Interest vs Moneyness')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
        
        # Plot 3: Volume by expiration
        volume_by_exp = data_filtered.groupby('daysToExpiration')['volume'].sum().sort_index()
        ax3.bar(volume_by_exp.index, volume_by_exp.values, alpha=0.7, color='orange')
        ax3.set_xlabel('Days to Expiration')
        ax3.set_ylabel('Total Volume')
        ax3.set_title('Volume by Expiration')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Volume vs IV (bubble chart with OI as size)
        scatter = ax4.scatter(data_filtered['impliedVolatility'], data_filtered['volume'],
                             s=data_filtered['openInterest']/100, alpha=0.6, 
                             c=data_filtered['moneyness'], cmap='viridis')
        ax4.set_xlabel('Implied Volatility')
        ax4.set_ylabel('Volume')
        ax4.set_title('Volume vs IV (bubble size = Open Interest)')
        ax4.set_yscale('log')
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Moneyness')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        else:
            fig.suptitle('Options Volume and Open Interest Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Volume analysis plot saved to {save_path}")
        
        return fig
    
    def create_summary_dashboard(self, options_data: pd.DataFrame, surface_summary: Dict,
                               spot_price: float, save_path: str = None) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple analysis plots.
        
        Args:
            options_data: Raw options DataFrame
            surface_summary: Summary statistics from volatility surface
            spot_price: Current stock price
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Calculate moneyness
        options_data = options_data.copy()
        options_data['moneyness'] = options_data['strike'] / spot_price
        
        # Plot 1: Raw data scatter (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        scatter1 = ax1.scatter(options_data['moneyness'], options_data['impliedVolatility'],
                              c=options_data['daysToExpiration'], cmap='viridis', alpha=0.6, s=20)
        ax1.set_xlabel('Moneyness')
        ax1.set_ylabel('Implied Volatility')
        ax1.set_title('IV vs Moneyness (colored by Days to Expiration)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
        plt.colorbar(scatter1, ax=ax1, label='Days to Expiration')
        
        # Plot 2: Calls vs Puts (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        calls = options_data[options_data['type'] == 'call']
        puts = options_data[options_data['type'] == 'put']
        
        if len(calls) > 0:
            ax2.scatter(calls['moneyness'], calls['impliedVolatility'], 
                       alpha=0.6, s=20, color='blue', label='Calls')
        if len(puts) > 0:
            ax2.scatter(puts['moneyness'], puts['impliedVolatility'], 
                       alpha=0.6, s=20, color='red', label='Puts')
        
        ax2.set_xlabel('Moneyness')
        ax2.set_ylabel('Implied Volatility')
        ax2.set_title('Calls vs Puts')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=1.0, color='black', linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Plot 3: Volume distribution (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        volume_data = options_data[options_data['volume'] > 0]
        if len(volume_data) > 0:
            ax3.hist(volume_data['volume'], bins=30, alpha=0.7, color='green', edgecolor='black')
            ax3.set_xlabel('Volume')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Volume Distribution')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: IV distribution (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        ax4.hist(options_data['impliedVolatility'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_xlabel('Implied Volatility')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Implied Volatility Distribution')
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Summary statistics (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create summary text
        summary_text = f"""
VOLATILITY SURFACE SUMMARY
Current Stock Price: ${spot_price:.2f}
Total Options Analyzed: {len(options_data)}
Calls: {len(options_data[options_data['type'] == 'call'])} | Puts: {len(options_data[options_data['type'] == 'put'])}

VOLATILITY STATISTICS:
Min IV: {options_data['impliedVolatility'].min():.1%}
Max IV: {options_data['impliedVolatility'].max():.1%}
Mean IV: {options_data['impliedVolatility'].mean():.1%}
Median IV: {options_data['impliedVolatility'].median():.1%}

EXPIRATION RANGE:
Shortest: {options_data['daysToExpiration'].min():.0f} days
Longest: {options_data['daysToExpiration'].max():.0f} days
Unique Expirations: {options_data['expiration'].nunique()}

STRIKE RANGE:
Lowest: ${options_data['strike'].min():.0f}
Highest: ${options_data['strike'].max():.0f}
ATM (Current Price): ${spot_price:.2f}
"""
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.suptitle(f'Options Volatility Analysis Dashboard - Generated {timestamp}', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary dashboard saved to {save_path}")
        
        return fig