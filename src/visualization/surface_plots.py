# src/visualization/surface_plots.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class VolatilitySurface3D:
    """
    Class for creating interactive 3D volatility surface visualizations.
    
    Uses both matplotlib and plotly for different types of 3D visualizations.
    """
    
    def __init__(self):
        """Initialize the 3D surface plotter."""
        self.default_colorscale = 'Viridis'
        
    def plot_matplotlib_surface(self, surface_data: Dict, title: str = None,
                               save_path: str = None, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Create 3D surface plot using matplotlib.
        
        Args:
            surface_data: Surface data dictionary from VolatilitySurface
            title: Plot title
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        
        # Get surface data
        surface = surface_data.get('combined', list(surface_data.values())[0])
        
        strikes = surface['strikes']
        times = surface['times']
        ivs = surface['implied_vols']
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot
        surf = ax.plot_surface(strikes, times * 365,  # Convert to days
                              ivs, cmap='viridis',
                              alpha=0.8, linewidth=0, antialiased=True)
        
        # Add contour lines
        contours = ax.contour(strikes, times * 365, ivs, 
                             zdir='z', offset=np.nanmin(ivs), cmap='viridis', alpha=0.5)
        
        # Formatting
        ax.set_xlabel('Strike Price ($)', fontsize=12)
        ax.set_ylabel('Days to Expiration', fontsize=12)
        ax.set_zlabel('Implied Volatility', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('3D Volatility Surface', fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Implied Volatility', fontsize=10)
        
        # Format z-axis as percentage
        ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda z, _: f'{z:.1%}'))
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Add spot price line if available
        spot_price = surface.get('spot_price')
        if spot_price:
            time_range = times.flatten()
            spot_line_times = np.linspace(time_range.min(), time_range.max(), 50) * 365
            spot_line_strikes = np.full_like(spot_line_times, spot_price)
            
            # Interpolate IV at spot price
            spot_line_ivs = []
            for t in spot_line_times / 365:
                # Find closest time slice
                time_idx = np.argmin(np.abs(times[:, 0] - t))
                strike_idx = np.argmin(np.abs(strikes[time_idx, :] - spot_price))
                spot_line_ivs.append(ivs[time_idx, strike_idx])
            
            ax.plot(spot_line_strikes, spot_line_times, spot_line_ivs,
                   color='red', linewidth=3, label='ATM Line')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D surface plot saved to {save_path}")
        
        return fig
    
    def plot_plotly_surface(self, surface_data: Dict, title: str = None,
                           save_path: str = None, show_plot: bool = True) -> go.Figure:
        """
        Create interactive 3D surface plot using plotly.
        
        Args:
            surface_data: Surface data dictionary from VolatilitySurface
            title: Plot title
            save_path: Path to save HTML file
            show_plot: Whether to display the plot
            
        Returns:
            Plotly figure object
        """
        
        # Get surface data
        surface = surface_data.get('combined', list(surface_data.values())[0])
        
        strikes = surface['strikes']
        times = surface['times'] * 365  # Convert to days
        ivs = surface['implied_vols']
        
        # Create surface plot
        fig = go.Figure(data=[go.Surface(
            x=strikes,
            y=times,
            z=ivs,
            colorscale=self.default_colorscale,
            name='Volatility Surface',
            hovertemplate='Strike: $%{x:.0f}<br>Days: %{y:.0f}<br>IV: %{z:.2%}<extra></extra>'
        )])
        
        # Add spot price line if available
        spot_price = surface.get('spot_price')
        if spot_price:
            time_range = np.linspace(times.min(), times.max(), 50)
            spot_line_strikes = np.full_like(time_range, spot_price)
            
            # Interpolate IV at spot price
            spot_line_ivs = []
            for t in time_range:
                # Find closest indices
                time_idx = np.argmin(np.abs(times[:, 0] - t))
                strike_idx = np.argmin(np.abs(strikes[time_idx, :] - spot_price))
                spot_line_ivs.append(ivs[time_idx, strike_idx])
            
            fig.add_trace(go.Scatter3d(
                x=spot_line_strikes,
                y=time_range,
                z=spot_line_ivs,
                mode='lines',
                line=dict(color='red', width=6),
                name='ATM Line',
                hovertemplate='ATM Strike: $%{x:.0f}<br>Days: %{y:.0f}<br>IV: %{z:.2%}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=title or '3D Volatility Surface',
            scene=dict(
                xaxis_title='Strike Price ($)',
                yaxis_title='Days to Expiration',
                zaxis_title='Implied Volatility',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            width=900,
            height=700,
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive 3D plot saved to {save_path}")
        
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_contour_map(self, surface_data: Dict, title: str = None,
                        save_path: str = None, show_plot: bool = True) -> go.Figure:
        """
        Create 2D contour map of volatility surface.
        
        Args:
            surface_data: Surface data dictionary
            title: Plot title
            save_path: Path to save HTML file
            show_plot: Whether to display the plot
            
        Returns:
            Plotly figure object
        """
        
        # Get surface data
        surface = surface_data.get('combined', list(surface_data.values())[0])
        
        strikes = surface['strikes']
        times = surface['times'] * 365  # Convert to days
        ivs = surface['implied_vols']
        
        # Create contour plot
        fig = go.Figure(data=go.Contour(
            x=strikes[0, :],  # Strike range
            y=times[:, 0],    # Time range
            z=ivs,
            colorscale=self.default_colorscale,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            hovertemplate='Strike: $%{x:.0f}<br>Days: %{y:.0f}<br>IV: %{z:.2%}<extra></extra>'
        ))
        
        # Add spot price line
        spot_price = surface.get('spot_price')
        if spot_price:
            fig.add_vline(x=spot_price, line_dash="dash", line_color="red",
                         annotation_text="ATM", annotation_position="top")
        
        # Update layout
        fig.update_layout(
            title=title or 'Volatility Surface Contour Map',
            xaxis_title='Strike Price ($)',
            yaxis_title='Days to Expiration',
            width=800,
            height=600,
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Contour map saved to {save_path}")
        
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_interactive_dashboard(self, surface_data: Dict, options_data: pd.DataFrame,
                                 spot_price: float, title: str = None,
                                 save_path: str = None, show_plot: bool = True) -> go.Figure:
        """
        Create comprehensive interactive dashboard with multiple views.
        
        Args:
            surface_data: Surface data dictionary
            options_data: Raw options data
            spot_price: Current stock price
            title: Dashboard title
            save_path: Path to save HTML file
            show_plot: Whether to display the plot
            
        Returns:
            Plotly figure object with subplots
        """
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "surface"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]],
            subplot_titles=['3D Volatility Surface', 'Volatility Smile (Next Expiration)',
                           'ATM Term Structure', 'Raw Data Scatter'],
            vertical_spacing=0.1
        )
        
        # Get surface data
        surface = surface_data.get('combined', list(surface_data.values())[0])
        strikes = surface['strikes']
        times = surface['times'] * 365
        ivs = surface['implied_vols']
        
        # 1. 3D Surface (top left)
        fig.add_trace(
            go.Surface(
                x=strikes, y=times, z=ivs,
                colorscale=self.default_colorscale,
                showscale=False,
                name='Vol Surface'
            ),
            row=1, col=1
        )
        
        # 2. Volatility Smile (top right) - shortest expiration
        options_data = options_data.copy()
        options_data['moneyness'] = options_data['strike'] / spot_price
        
        shortest_exp = options_data['daysToExpiration'].min()
        smile_data = options_data[
            abs(options_data['daysToExpiration'] - shortest_exp) < 2
        ].sort_values('moneyness')
        
        if len(smile_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=smile_data['moneyness'],
                    y=smile_data['impliedVolatility'],
                    mode='markers+lines',
                    name=f'{shortest_exp:.0f}d Smile',
                    line=dict(color='blue')
                ),
                row=1, col=2
            )
            
            # Add ATM line
            fig.add_vline(x=1.0, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. ATM Term Structure (bottom left)
        # Group by expiration and find ATM vol for each
        atm_vols = []
        expirations = []
        
        for exp_days in sorted(options_data['daysToExpiration'].unique()):
            exp_data = options_data[options_data['daysToExpiration'] == exp_days]
            
            # Find closest to ATM
            atm_option = exp_data.loc[exp_data['moneyness'].sub(1.0).abs().idxmin()]
            atm_vols.append(atm_option['impliedVolatility'])
            expirations.append(exp_days)
        
        if atm_vols:
            fig.add_trace(
                go.Scatter(
                    x=expirations,
                    y=atm_vols,
                    mode='markers+lines',
                    name='ATM Term Structure',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        # 4. Raw Data Scatter (bottom right)
        calls = options_data[options_data['type'] == 'call']
        puts = options_data[options_data['type'] == 'put']
        
        if len(calls) > 0:
            fig.add_trace(
                go.Scatter(
                    x=calls['moneyness'],
                    y=calls['impliedVolatility'],
                    mode='markers',
                    name='Calls',
                    marker=dict(color='blue', size=4, opacity=0.6)
                ),
                row=2, col=2
            )
        
        if len(puts) > 0:
            fig.add_trace(
                go.Scatter(
                    x=puts['moneyness'],
                    y=puts['impliedVolatility'],
                    mode='markers',
                    name='Puts',
                    marker=dict(color='red', size=4, opacity=0.6)
                ),
                row=2, col=2
            )
        
        # Add ATM line to raw data plot
        fig.add_vline(x=1.0, line_dash="dash", line_color="black", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=title or f'Volatility Analysis Dashboard - Spot: ${spot_price:.2f}',
            height=800,
            showlegend=True,
            font=dict(size=10)
        )
        
        # Update 3D scene for the surface plot
        fig.update_scenes(
            xaxis_title="Strike ($)",
            yaxis_title="Days",
            zaxis_title="IV"
        )
        
        # Update individual 2D subplot axes
        fig.update_xaxes(title_text="Moneyness", row=1, col=2)
        fig.update_yaxes(title_text="Implied Vol", row=1, col=2)
        
        fig.update_xaxes(title_text="Days to Expiration", row=2, col=1)
        fig.update_yaxes(title_text="ATM Implied Vol", row=2, col=1)
        
        fig.update_xaxes(title_text="Moneyness", row=2, col=2)
        fig.update_yaxes(title_text="Implied Vol", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive dashboard saved to {save_path}")
        
        if show_plot:
            fig.show()
        
        return fig
    
    def save_multiple_views(self, surface_data: Dict, options_data: pd.DataFrame,
                          spot_price: float, output_dir: str = "plots"):
        """
        Save multiple visualization views to files.
        
        Args:
            surface_data: Surface data dictionary
            options_data: Raw options data
            spot_price: Current stock price
            output_dir: Directory to save plots
        """
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 3D Matplotlib surface
        fig_3d = self.plot_matplotlib_surface(
            surface_data, 
            title=f'3D Volatility Surface - ${spot_price:.2f}',
            save_path=f'{output_dir}/vol_surface_3d_{timestamp}.png'
        )
        plt.close(fig_3d)
        
        # 2. Interactive Plotly surface
        self.plot_plotly_surface(
            surface_data,
            title=f'Interactive Volatility Surface - ${spot_price:.2f}',
            save_path=f'{output_dir}/vol_surface_interactive_{timestamp}.html',
            show_plot=False
        )
        
        # 3. Contour map
        self.plot_contour_map(
            surface_data,
            title=f'Volatility Contour Map - ${spot_price:.2f}',
            save_path=f'{output_dir}/vol_contour_{timestamp}.html',
            show_plot=False
        )
        
        # 4. Interactive dashboard
        self.plot_interactive_dashboard(
            surface_data, options_data, spot_price,
            title=f'Volatility Dashboard - ${spot_price:.2f}',
            save_path=f'{output_dir}/vol_dashboard_{timestamp}.html',
            show_plot=False
        )
        
        print(f"All visualization views saved to {output_dir}/")
        print(f"Files created:")
        print(f"  - vol_surface_3d_{timestamp}.png (static 3D plot)")
        print(f"  - vol_surface_interactive_{timestamp}.html (interactive 3D)")
        print(f"  - vol_contour_{timestamp}.html (contour map)")
        print(f"  - vol_dashboard_{timestamp}.html (full dashboard)")


def create_animation(surface_data_list: list, output_path: str = "vol_surface_animation.gif"):
    """
    Create animated GIF showing volatility surface evolution over time.
    
    Args:
        surface_data_list: List of surface data dictionaries over time
        output_path: Path to save the animation
    """
    
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        
        # Setup figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def animate(frame):
            ax.clear()
            
            surface_data = surface_data_list[frame]
            surface = surface_data.get('combined', list(surface_data.values())[0])
            
            strikes = surface['strikes']
            times = surface['times'] * 365
            ivs = surface['implied_vols']
            
            # Plot surface
            surf = ax.plot_surface(strikes, times, ivs, cmap='viridis', alpha=0.8)
            
            # Formatting
            ax.set_xlabel('Strike Price ($)')
            ax.set_ylabel('Days to Expiration')
            ax.set_zlabel('Implied Volatility')
            ax.set_title(f'Volatility Surface Evolution - Frame {frame + 1}')
            
            # Set consistent z-limits
            all_ivs = [s.get('combined', list(s.values())[0])['implied_vols'] 
                      for s in surface_data_list]
            z_min = min(np.nanmin(iv) for iv in all_ivs)
            z_max = max(np.nanmax(iv) for iv in all_ivs)
            ax.set_zlim(z_min, z_max)
            
            return surf,
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(surface_data_list), 
                           interval=500, blit=False, repeat=True)
        
        # Save as GIF
        writer = PillowWriter(fps=2)
        anim.save(output_path, writer=writer)
        
        print(f"Animation saved to {output_path}")
        plt.close(fig)
        
    except ImportError:
        print("Animation requires additional packages: pip install pillow")
    except Exception as e:
        print(f"Error creating animation: {e}")


def compare_surfaces(surface_data1: Dict, surface_data2: Dict, 
                    labels: tuple = ("Surface 1", "Surface 2"),
                    title: str = "Surface Comparison") -> go.Figure:
    """
    Compare two volatility surfaces side by side.
    
    Args:
        surface_data1: First surface data dictionary
        surface_data2: Second surface data dictionary
        labels: Labels for the two surfaces
        title: Overall plot title
        
    Returns:
        Plotly figure with side-by-side comparison
    """
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=labels
    )
    
    # First surface
    surface1 = surface_data1.get('combined', list(surface_data1.values())[0])
    fig.add_trace(
        go.Surface(
            x=surface1['strikes'],
            y=surface1['times'] * 365,
            z=surface1['implied_vols'],
            colorscale='Viridis',
            name=labels[0]
        ),
        row=1, col=1
    )
    
    # Second surface
    surface2 = surface_data2.get('combined', list(surface_data2.values())[0])
    fig.add_trace(
        go.Surface(
            x=surface2['strikes'],
            y=surface2['times'] * 365,
            z=surface2['implied_vols'],
            colorscale='Plasma',
            name=labels[1]
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Strike ($)',
            yaxis_title='Days',
            zaxis_title='IV'
        ),
        scene2=dict(
            xaxis_title='Strike ($)',
            yaxis_title='Days',
            zaxis_title='IV'
        ),
        height=600
    )
    
    return fig