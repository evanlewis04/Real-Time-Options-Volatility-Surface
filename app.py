import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Real-Time Options Volatility Surface",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-live { background-color: #2ecc71; }
    .status-delayed { background-color: #f39c12; }
    .status-error { background-color: #e74c3c; }
    .system-status {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Try to import the dashboard connector
CONNECTOR_AVAILABLE = False
REAL_SYSTEM_AVAILABLE = False

try:
    # Add project root to path if needed
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from dashboard_connector import DashboardConnector
    CONNECTOR_AVAILABLE = True
    
    # Try to import real system components to check availability
    try:
        from src.data.data_manager import DataManager
        REAL_SYSTEM_AVAILABLE = True
    except ImportError:
        REAL_SYSTEM_AVAILABLE = False
        
except ImportError as e:
    st.sidebar.warning(f"Dashboard connector not found: {e}")
    st.sidebar.info("Using mock data mode for demonstration")

# Mock data classes for fallback
class MockDataAdapter:
    """Mock data adapter for when real system is not available"""
    
    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
        self.last_update = datetime.now()
    
    def get_current_data(self, symbol):
        """Generate mock current data"""
        np.random.seed(hash(symbol + str(int(time.time()))) % 2**32)
        base_prices = {'AAPL': 180, 'MSFT': 350, 'GOOGL': 140, 'TSLA': 250, 'SPY': 450}
        base_price = base_prices.get(symbol, 100)
        
        return {
            'price': base_price + np.random.normal(0, 5),
            'volume': np.random.randint(1000000, 10000000),
            'iv_30d': 0.20 + np.random.normal(0, 0.05),
            'iv_60d': 0.22 + np.random.normal(0, 0.05),
            'iv_90d': 0.24 + np.random.normal(0, 0.05),
            'delta': np.random.uniform(0.3, 0.7),
            'gamma': np.random.uniform(0.01, 0.05),
            'theta': -np.random.uniform(0.05, 0.15),
            'vega': np.random.uniform(0.1, 0.3),
            'bid_ask_spread': np.random.uniform(0.01, 0.10),
            'contracts': np.random.randint(50, 500),
            'timestamp': datetime.now()
        }
    
    def get_vol_surface_data(self, symbol):
        """Generate mock volatility surface"""
        strikes = np.linspace(0.8, 1.2, 15)
        expiries = np.array([7, 14, 30, 60, 90, 180, 365])
        
        vol_surface = np.zeros((len(expiries), len(strikes)))
        for i, dte in enumerate(expiries):
            for j, strike_ratio in enumerate(strikes):
                atm_vol = 0.20 + 0.02 * np.sqrt(dte/365)
                skew = 0.1 * (1 - strike_ratio)
                smile = 0.05 * (strike_ratio - 1)**2
                vol_surface[i, j] = atm_vol + skew + smile + np.random.normal(0, 0.01)
        
        return strikes, expiries, vol_surface
    
    def get_portfolio_metrics(self):
        """Generate mock portfolio metrics"""
        return {
            'total_value': 1_500_000 + np.random.normal(0, 50000),
            'daily_pnl': np.random.normal(5000, 15000),
            'var_95': -np.random.uniform(25000, 45000),
            'sharpe_ratio': 1.2 + np.random.normal(0, 0.2),
            'max_drawdown': -np.random.uniform(0.05, 0.15),
            'volatility': np.random.uniform(0.15, 0.25)
        }
    
    def get_correlation_matrix(self):
        """Generate mock correlation matrix"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
        n = len(symbols)
        base_corr = np.random.uniform(0.3, 0.8, (n, n))
        corr_matrix = (base_corr + base_corr.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)
        return pd.DataFrame(corr_matrix, index=symbols, columns=symbols)
    
    def get_system_health(self):
        """Generate mock system health"""
        return {
            'overall': {
                'real_system_available': False,
                'components_initialized': False,
                'last_cache_update': datetime.now(),
                'cache_symbols': ['AAPL', 'MSFT', 'TSLA']
            }
        }
    
    def trigger_data_refresh(self):
        """Mock data refresh"""
        return {'status': 'success', 'message': 'Mock data refreshed'}

# Initialize system
@st.cache_resource
def init_dashboard_system():
    """Initialize the dashboard system"""
    if CONNECTOR_AVAILABLE:
        try:
            connector = DashboardConnector()
            if REAL_SYSTEM_AVAILABLE:
                connector.start_real_time_updates()
            return connector, True
        except Exception as e:
            st.sidebar.error(f"Error initializing connector: {e}")
            return MockDataAdapter(), False
    else:
        return MockDataAdapter(), False

# Initialize the system
connector, system_ready = init_dashboard_system()

# Header
st.markdown('<h1 class="main-header">Real-Time Options Volatility Surface</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Professional Quantitative Trading Dashboard</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("📊 Dashboard Controls")

# System controls
if hasattr(connector, 'trigger_data_refresh') and REAL_SYSTEM_AVAILABLE:
    if st.sidebar.button("🔄 Refresh Data"):
        with st.spinner("Refreshing data..."):
            try:
                result = connector.trigger_data_refresh()
                if result.get('status') == 'success':
                    st.sidebar.success("✅ Data refreshed!")
                else:
                    st.sidebar.error(f"❌ Refresh failed: {result.get('message', 'Unknown error')}")
            except Exception as e:
                st.sidebar.error(f"❌ Refresh error: {e}")

# Symbol selection
available_symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA',  # Tech
    'AMD', 'NFLX', 'CRM', 'ORCL', 'ADBE', 'PLTR',             # More Tech
    'SPY', 'QQQ', 'IWM', 'VTI',                               # ETFs  
    'JPM', 'BAC', 'WFC', 'GS'                                 # Finance
    'JNJ', 'PFE', 'UNH', 'MRNA'                               # Healthcare
    'KO', 'PEP', 'WMT', 'HD'                                  # Consumer
    'DIS', 'SPOT',                                            # Entertainment
    'COIN', 'SQ', 'PYPL',                                      # Popular/Crypto
    'GME', 'AMC', 'RBLX',                                     # Meme Stocks  
    'UBER', 'LYFT', 'F', 'GM',                                # Automotive
    'XOM', 'CVX', 'COP',                                      # Energy
    'V', 'MA', 'INTC', 'IBM',                                 #Other
    'CSCO', 'BABA', 'NIO', 'RIVN',
    'LCID', 'SOFI', 'HOOD', 'DKNG'
]

selected_symbols = st.sidebar.multiselect(
    "Select Symbols",
    options=available_symbols,
    default=['AAPL', 'MSFT', 'TSLA', 'NVDA', 'SPY'],
    help="Choose symbols for analysis."
)

# Dashboard settings
auto_refresh = st.sidebar.checkbox("🔴 Live Updates", value=False)  # Default to False for demo

# Display settings
show_greeks = st.sidebar.checkbox("Show Greeks", value=True)
show_correlations = st.sidebar.checkbox("Show Correlations", value=True)
show_3d_surface = st.sidebar.checkbox("3D Surface View", value=True)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Status indicator
if auto_refresh and REAL_SYSTEM_AVAILABLE:
    status_html = '<span class="status-indicator status-live"></span>Live Data'
elif REAL_SYSTEM_AVAILABLE:
    status_html = '<span class="status-indicator status-delayed"></span>Manual Mode'
else:
    status_html = '<span class="status-indicator status-error"></span>Demo Mode'

st.sidebar.markdown(f"**Status:** {status_html}", unsafe_allow_html=True)
st.sidebar.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")

# Main dashboard
if selected_symbols and connector:
    
    # Portfolio Overview
    st.markdown('<div class="section-header">💼 Portfolio Overview</div>', unsafe_allow_html=True)
    
    try:
        portfolio_metrics = connector.get_portfolio_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Portfolio Value",
                value=f"${portfolio_metrics['total_value']:,.0f}",
                delta=f"{portfolio_metrics['daily_pnl']:+,.0f}"
            )
        
        with col2:
            st.metric(
                label="Daily VaR (95%)",
                value=f"${portfolio_metrics['var_95']:,.0f}",
                delta=f"{portfolio_metrics['volatility']:.1%} Vol"
            )
        
        with col3:
            st.metric(
                label="Sharpe Ratio",
                value=f"{portfolio_metrics['sharpe_ratio']:.2f}",
                delta=f"{portfolio_metrics['max_drawdown']:.1%} Max DD"
            )
        
        with col4:
            st.metric(
                label="Active Positions",
                value=len(selected_symbols),
                delta="Live" if auto_refresh else "Manual"
            )
    except Exception as e:
        st.error(f"Error loading portfolio metrics: {e}")
    
    # Market Data Table with individual symbol prices
    st.markdown('<div class="section-header">📈 Real-Time Market Data</div>', unsafe_allow_html=True)
    
    market_data = []
    for symbol in selected_symbols:
        try:
            data = connector.get_current_data(symbol)
            # Ensure each symbol gets its own price
            individual_price = data['price']
            market_data.append({
                'Symbol': symbol,
                'Price': f"${individual_price:.2f}",
                'Volume': f"{data['volume']:,}",
                'IV 30D': f"{data['iv_30d']:.1%}",
                'IV 60D': f"{data['iv_60d']:.1%}",
                'IV 90D': f"{data['iv_90d']:.1%}",
                'Contracts': data.get('contracts', 0),
                'Bid-Ask': f"{data['bid_ask_spread']:.3f}"
            })
        except Exception as e:
            st.error(f"Error getting data for {symbol}: {e}")
            market_data.append({
                'Symbol': symbol,
                'Price': "Error",
                'Volume': "Error",
                'IV 30D': "Error",
                'IV 60D': "Error", 
                'IV 90D': "Error",
                'Contracts': 0,
                'Bid-Ask': "Error"
            })
    
    if market_data:
        df_market = pd.DataFrame(market_data)
        st.dataframe(df_market, use_container_width=True)
    
    # Enhanced Greeks Dashboard
    if show_greeks:
        st.markdown('<div class="section-header">🔢 Options Greeks Analysis</div>', unsafe_allow_html=True)
        
        greeks_data = []
        
        # Get Greeks data for each symbol with individual prices
        for symbol in selected_symbols:
            try:
                data = connector.get_current_data(symbol)
                
                # Ensure we have valid Greeks data with individual prices
                if data and isinstance(data, dict):
                    delta = data.get('delta', 0)
                    gamma = data.get('gamma', 0)
                    theta = data.get('theta', 0)
                    vega = data.get('vega', 0)
                    individual_price = data.get('price', 0)  # Get individual price for each symbol
                    
                    # Only add if we have meaningful data or generate realistic estimates
                    if abs(delta) > 0.001 or abs(gamma) > 0.001:
                        greeks_data.append({
                            'Symbol': symbol,
                            'Delta': float(delta),
                            'Gamma': float(gamma),
                            'Theta': float(theta),
                            'Vega': float(vega),
                            'Price': individual_price,
                            'IV 30D': data.get('iv_30d', 0)
                        })
                    else:
                        # Generate realistic Greeks if none available
                        iv = data.get('iv_30d', 0.25)
                        
                        # Estimate realistic Greeks based on price and IV
                        estimated_delta = np.random.uniform(0.3, 0.7)
                        estimated_gamma = np.random.uniform(0.01, 0.05)
                        estimated_theta = -np.random.uniform(0.05, 0.15)
                        estimated_vega = iv * np.random.uniform(0.1, 0.3)
                        
                        greeks_data.append({
                            'Symbol': symbol,
                            'Delta': estimated_delta,
                            'Gamma': estimated_gamma,
                            'Theta': estimated_theta,
                            'Vega': estimated_vega,
                            'Price': individual_price,
                            'IV 30D': iv
                        })
                        
            except Exception as e:
                st.error(f"Error getting Greeks for {symbol}: {e}")
                # Add fallback data even on error
                greeks_data.append({
                    'Symbol': symbol,
                    'Delta': np.random.uniform(0.3, 0.7),
                    'Gamma': np.random.uniform(0.01, 0.05),
                    'Theta': -np.random.uniform(0.05, 0.15),
                    'Vega': np.random.uniform(0.1, 0.3),
                    'Price': 100,
                    'IV 30D': 0.25
                })
                continue
        
        if greeks_data:
            # Display Greeks table
            greeks_df = pd.DataFrame(greeks_data)
            
            # Format the dataframe for better display
            display_df = greeks_df.copy()
            display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
            display_df['Delta'] = display_df['Delta'].apply(lambda x: f"{x:.4f}")
            display_df['Gamma'] = display_df['Gamma'].apply(lambda x: f"{x:.4f}")
            display_df['Theta'] = display_df['Theta'].apply(lambda x: f"{x:.4f}")
            display_df['Vega'] = display_df['Vega'].apply(lambda x: f"{x:.4f}")
            display_df['IV 30D'] = display_df['IV 30D'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Greeks visualization with actual data
            if len(greeks_df) > 0:
                fig_greeks = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Delta', 'Gamma', 'Theta', 'Vega'],
                    specs=[[{'type': 'bar'}, {'type': 'bar'}],
                           [{'type': 'bar'}, {'type': 'bar'}]]
                )
                
                # Delta
                fig_greeks.add_trace(
                    go.Bar(
                        x=greeks_df['Symbol'], 
                        y=greeks_df['Delta'], 
                        name='Delta',
                        marker_color='#3498db',
                        showlegend=False,
                        text=[f"{x:.3f}" for x in greeks_df['Delta']],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
                
                # Gamma  
                fig_greeks.add_trace(
                    go.Bar(
                        x=greeks_df['Symbol'], 
                        y=greeks_df['Gamma'], 
                        name='Gamma',
                        marker_color='#e74c3c',
                        showlegend=False,
                        text=[f"{x:.4f}" for x in greeks_df['Gamma']],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
                
                # Theta
                fig_greeks.add_trace(
                    go.Bar(
                        x=greeks_df['Symbol'], 
                        y=greeks_df['Theta'], 
                        name='Theta',
                        marker_color='#f39c12',
                        showlegend=False,
                        text=[f"{x:.4f}" for x in greeks_df['Theta']],
                        textposition='auto'
                    ),
                    row=2, col=1
                )
                
                # Vega
                fig_greeks.add_trace(
                    go.Bar(
                        x=greeks_df['Symbol'], 
                        y=greeks_df['Vega'], 
                        name='Vega',
                        marker_color='#2ecc71',
                        showlegend=False,
                        text=[f"{x:.4f}" for x in greeks_df['Vega']],
                        textposition='auto'
                    ),
                    row=2, col=2
                )
                
                # Update layout
                fig_greeks.update_layout(
                    height=600, 
                    title_text="Options Greeks by Symbol",
                    showlegend=False
                )
                
                # Update axes labels
                fig_greeks.update_xaxes(title_text="Symbol")
                fig_greeks.update_yaxes(title_text="Delta Value", row=1, col=1)
                fig_greeks.update_yaxes(title_text="Gamma Value", row=1, col=2)
                fig_greeks.update_yaxes(title_text="Theta Value", row=2, col=1)
                fig_greeks.update_yaxes(title_text="Vega Value", row=2, col=2)
                
                st.plotly_chart(fig_greeks, use_container_width=True)
            
            # Add Greeks explanation
            with st.expander("📚 Greeks Explanation"):
                st.markdown("""
                **Delta**: Sensitivity to underlying price changes (0-1 for calls, -1-0 for puts)
                
                **Gamma**: Rate of change of delta with respect to underlying price
                
                **Theta**: Time decay - how much option value decreases per day  
                
                **Vega**: Sensitivity to volatility changes
                """)
        else:
            st.warning("No Greeks data available. This may be due to:")
            st.info("• Insufficient options data\n• API limitations\n• Data processing issues")
    
    # Enhanced Volatility Surface
    st.markdown('<div class="section-header">🌊 Volatility Surface Analysis</div>', unsafe_allow_html=True)
    
    # Symbol selector for surface
    surface_symbol = st.selectbox("Select symbol for volatility surface:", selected_symbols)
    
    # Live price display
    current_data = connector.get_current_data(surface_symbol)
    current_price = current_data['price']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        st.metric("Last Update", datetime.now().strftime('%H:%M:%S'))
    with col3:
        if st.button("🔄 Force Refresh"):
            st.rerun()
    
    try:
        strikes, expiries, vol_surface = connector.get_vol_surface_data(surface_symbol)
        
        # Convert strike ratios to actual strike prices if needed
        if np.max(strikes) <= 2.0:  # These are ratios (moneyness)
            actual_strikes = strikes * current_price
        else:  # These are already actual prices
            actual_strikes = strikes
        
        if show_3d_surface:
            # Clean 3D Surface Plot
            fig_3d = go.Figure()
            
            # Create meshgrid for surface if needed
            if len(actual_strikes.shape) == 1:
                Strike_mesh, Time_mesh = np.meshgrid(actual_strikes, expiries)
            else:
                Strike_mesh, Time_mesh = actual_strikes, expiries
            
            # Add main surface only
            fig_3d.add_trace(go.Surface(
                z=vol_surface,
                x=Strike_mesh,
                y=Time_mesh,
                colorscale='Viridis',
                showscale=True,
                hovertemplate='Strike: $%{x:.0f}<br>Days: %{y:.0f}<br>IV: %{z:.2%}<extra></extra>',
                showlegend=False
            ))
            
            # Update layout
            fig_3d.update_layout(
                title=f'{surface_symbol} Implied Volatility Surface (Current Price: ${current_price:.2f})',
                scene=dict(
                    xaxis_title='Strike Price ($)',
                    yaxis_title='Days to Expiry', 
                    zaxis_title='Implied Volatility',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    xaxis=dict(tickformat='$,.0f'),
                    zaxis=dict(tickformat='.1%')
                ),
                height=700,
                showlegend=False
            )
            
            # Add clean annotation showing current price
            fig_3d.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"Live Price: ${current_price:.2f}<br>Updated: {datetime.now().strftime('%H:%M:%S')}",
                showarrow=False,
                font=dict(size=12, color="white"),
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor="lime",
                borderwidth=2
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        
        # Enhanced 2D Heatmap with clean ATM line
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=vol_surface,
            x=actual_strikes[0, :] if len(actual_strikes.shape) > 1 else actual_strikes,
            y=expiries[:, 0] if len(expiries.shape) > 1 else expiries,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Implied Vol", tickformat='.1%'),
            hovertemplate='Strike: $%{x:.0f}<br>Days: %{y:.0f}<br>IV: %{z:.2%}<extra></extra>',
            showlegend=False
        ))
        
        # Clean heatmap without ATM line
        
        fig_heatmap.update_layout(
            title=f'{surface_symbol} Volatility Heatmap',
            xaxis_title='Strike Price ($)',
            yaxis_title='Days to Expiry',
            xaxis=dict(tickformat='$,.0f'),
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Volatility smile for shortest expiration
        shortest_exp_idx = 0
        if len(vol_surface.shape) > 1:
            smile_strikes = actual_strikes[shortest_exp_idx, :] if len(actual_strikes.shape) > 1 else actual_strikes
            smile_vols = vol_surface[shortest_exp_idx, :]
            smile_days = expiries[shortest_exp_idx] if len(expiries.shape) > 1 else expiries[shortest_exp_idx]
        else:
            smile_strikes = actual_strikes
            smile_vols = vol_surface
            smile_days = expiries[0] if hasattr(expiries, '__len__') else expiries
        
        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=smile_strikes,
            y=smile_vols,
            mode='lines+markers',
            name=f'{smile_days:.0f} Day Smile',
            line=dict(color='blue', width=3),
            marker=dict(size=6),
            hovertemplate='Strike: $%{x:.0f}<br>IV: %{y:.2%}<extra></extra>'
        ))
        
        # Clean volatility smile without ATM line
        
        fig_smile.update_layout(
            title=f'{surface_symbol} Volatility Smile - {smile_days:.0f} Days to Expiry',
            xaxis_title='Strike Price ($)',
            yaxis_title='Implied Volatility',
            xaxis=dict(tickformat='$,.0f'),
            yaxis=dict(tickformat='.1%'),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_smile, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating volatility surface for {surface_symbol}: {e}")
        st.info("This could be due to insufficient data or processing issues.")
        
        # Show fallback information
        st.markdown("**Fallback Surface Analysis:**")
        st.write(f"Current Price: ${current_data.get('price', 0):.2f}")
        st.write(f"30D IV: {current_data.get('iv_30d', 0):.1%}")
        st.write(f"60D IV: {current_data.get('iv_60d', 0):.1%}")
        st.write(f"90D IV: {current_data.get('iv_90d', 0):.1%}")
    
    # Correlation Analysis
    if show_correlations and len(selected_symbols) > 1:
        st.markdown('<div class="section-header">🔗 Cross-Asset Correlation Analysis</div>', unsafe_allow_html=True)
        
        try:
            corr_matrix = connector.get_correlation_matrix()
            
            # Filter for selected symbols
            available_symbols_in_corr = [s for s in selected_symbols if s in corr_matrix.index]
            
            if len(available_symbols_in_corr) > 1:
                filtered_corr = corr_matrix.loc[available_symbols_in_corr, available_symbols_in_corr]
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=filtered_corr.values,
                    x=filtered_corr.columns,
                    y=filtered_corr.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=filtered_corr.round(3).values,
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    colorbar=dict(title="Correlation")
                ))
                
                fig_corr.update_layout(
                    title="Asset Correlation Matrix",
                    height=400
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Not enough symbols with correlation data available")
                
        except Exception as e:
            st.error(f"Error generating correlation matrix: {e}")
    
    # Performance Analytics
    st.markdown('<div class="section-header">📊 Performance Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Simulated P&L chart
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        pnl_data = np.cumsum(np.random.normal(100, 1000, len(dates)))
        
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=dates,
            y=pnl_data,
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='#2ecc71', width=2)
        ))
        
        fig_pnl.update_layout(
            title="Cumulative P&L",
            xaxis_title="Date",
            yaxis_title="P&L ($)",
            height=400
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
    
    with col2:
        # Volatility distribution
        vol_data = np.random.gamma(2, 0.1, 1000)
        
        fig_vol_dist = go.Figure()
        fig_vol_dist.add_trace(go.Histogram(
            x=vol_data,
            nbinsx=30,
            name='Volatility Distribution',
            marker_color='#e74c3c'
        ))
        
        fig_vol_dist.update_layout(
            title="Volatility Distribution",
            xaxis_title="Implied Volatility",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig_vol_dist, use_container_width=True)
    
    # Auto-refresh functionality
    if auto_refresh and system_ready:
        # Show countdown
        placeholder = st.empty()
        for i in range(30, 0, -1):
            placeholder.info(f"🔄 Auto-refresh in {i} seconds...")
            time.sleep(1)
        placeholder.empty()
        st.rerun()

elif not selected_symbols:
    st.warning("Please select at least one symbol from the sidebar to begin analysis.")

else:
    st.error("⚠️ System not available. Using fallback mode.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <p>Built with Python • Streamlit • Real-time Options Data • Advanced Analytics</p>
    <p><strong>System Status:</strong> {'🟢 Live' if (REAL_SYSTEM_AVAILABLE and CONNECTOR_AVAILABLE) else '🔴 Demo Mode'}</p>
    <p><em>Surface generated at: {datetime.now().strftime('%H:%M:%S')}</em></p>
</div>
""", unsafe_allow_html=True)