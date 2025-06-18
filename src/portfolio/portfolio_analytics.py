"""
Portfolio-Level Volatility Analytics
Cross-asset analysis and portfolio risk management for volatility surfaces
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from src.analysis.vol_surface import VolatilitySurface



@dataclass
class CorrelationMetrics:
    """Container for correlation analysis results"""
    symbol_pair: Tuple[str, str]
    iv_correlation: float
    price_correlation: float
    volume_correlation: float
    correlation_stability: float  # How stable correlation has been
    last_updated: datetime


@dataclass
class PortfolioRisk:
    """Container for portfolio risk metrics"""
    timestamp: datetime
    portfolio_iv: float
    portfolio_var: float  # Value at Risk
    portfolio_diversification_ratio: float
    concentrated_risk_symbols: List[str]
    risk_contribution: Dict[str, float]  # Risk contribution by asset
    correlation_matrix: pd.DataFrame


@dataclass
class CrossAssetSignal:
    """Container for cross-asset trading signals"""
    signal_type: str  # 'vol_spread', 'correlation_break', 'sector_rotation'
    symbols: List[str]
    strength: float  # Signal strength 0-1
    description: str
    timestamp: datetime
    supporting_data: Dict[str, Any]


class PortfolioAnalytics:
    """
    Portfolio-level volatility analytics and cross-asset analysis
    Provides risk management and trading signal generation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.surface_data = {}  # Dict[str, SurfaceUpdate]
        self.market_data = {}   # Dict[str, MarketUpdate]
        self.correlation_history = defaultdict(list)  # Dict[Tuple[str, str], List[CorrelationMetrics]]
        self.portfolio_risk_history = []  # List[PortfolioRisk]

        # Cross-asset signals
        self.active_signals: List[CrossAssetSignal] = []
        self.signal_history: List[CrossAssetSignal] = []
        
        # Asset classifications
        self.asset_sectors = {
            'AAPL': 'tech',
            'MSFT': 'tech', 
            'GOOGL': 'tech',
            'NVDA': 'tech',
            'TSLA': 'automotive_tech',
            'JPM': 'financial',
            'BAC': 'financial',
            'SPY': 'broad_market',
            'QQQ': 'tech_etf',
            'GME': 'meme_stock'
        }
        
        # Risk parameters
        self.risk_params = {
            'var_confidence': 0.95,
            'correlation_window': 20,  # periods for correlation calculation
            'signal_threshold': 0.7,   # minimum signal strength
            'risk_concentration_limit': 0.4  # max weight in single asset
        }
        
        self.logger.info("Portfolio analytics initialized")
    
    def update_surface_data(self, surface_update):
        """Update with new surface data"""
        if surface_update.success:
            self.surface_data[surface_update.symbol] = surface_update
            self._update_correlations()
            self._update_portfolio_risk()
            self._detect_cross_asset_signals()
    
    def update_market_data(self, market_update):
        """Update with new market data"""
        if market_update.success:
            self.market_data[market_update.symbol] = market_update
    
    def calculate_correlation_matrix(self, lookback_periods: int = 20) -> pd.DataFrame:
        """Calculate correlation matrix for implied volatilities"""
        try:
            # Get symbols with sufficient data
            symbols = [
                symbol for symbol, update in self.surface_data.items()
                if update.success and 'mean_iv' in update.statistics
            ]
            
            if len(symbols) < 2:
                return pd.DataFrame()
            
            # Create correlation matrix
            correlation_data = {}
            
            for symbol in symbols:
                # Get recent IV data
                iv_data = []
                update = self.surface_data[symbol]
                
                # For now, use current mean IV (in production, would use time series)
                if 'mean_iv' in update.statistics:
                    iv_data.append(update.statistics['mean_iv'])
                
                correlation_data[symbol] = iv_data
            
            # Convert to DataFrame
            df = pd.DataFrame(correlation_data)
            
            # Calculate correlation matrix (simplified for current implementation)
            if len(df) > 1:
                correlation_matrix = df.corr()
            else:
                # Create identity matrix if insufficient data
                correlation_matrix = pd.DataFrame(
                    np.eye(len(symbols)), 
                    index=symbols, 
                    columns=symbols
                )
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def _update_correlations(self):
        """Update pairwise correlations between assets"""
        try:
            symbols = list(self.surface_data.keys())
            
            # Calculate pairwise correlations
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    correlation = self._calculate_pairwise_correlation(symbol1, symbol2)
                    if correlation:
                        pair = (symbol1, symbol2)
                        self.correlation_history[pair].append(correlation)
                        
                        # Keep only recent correlations
                        max_history = 100
                        if len(self.correlation_history[pair]) > max_history:
                            self.correlation_history[pair] = self.correlation_history[pair][-max_history:]
        
        except Exception as e:
            self.logger.error(f"Error updating correlations: {e}")
    
    def _calculate_pairwise_correlation(self, symbol1: str, symbol2: str) -> Optional[CorrelationMetrics]:
        """Calculate correlation between two assets"""
        try:
            update1 = self.surface_data.get(symbol1)
            update2 = self.surface_data.get(symbol2)
            market1 = self.market_data.get(symbol1)
            market2 = self.market_data.get(symbol2)
            
            if not all([update1, update2, market1, market2]):
                return None
            
            # IV correlation (simplified - using current values)
            iv1 = update1.statistics.get('mean_iv', 0)
            iv2 = update2.statistics.get('mean_iv', 0)
            
            # For real correlation, we'd need time series data
            # This is a simplified implementation
            iv_correlation = 0.5  # Placeholder
            
            # Price correlation (using underlying prices)
            price_correlation = 0.3  # Placeholder
            
            # Volume correlation
            vol1 = update1.statistics.get('total_volume', 0)
            vol2 = update2.statistics.get('total_volume', 0)
            volume_correlation = 0.2  # Placeholder
            
            return CorrelationMetrics(
                symbol_pair=(symbol1, symbol2),
                iv_correlation=iv_correlation,
                price_correlation=price_correlation,
                volume_correlation=volume_correlation,
                correlation_stability=0.8,  # Placeholder
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation for {symbol1}-{symbol2}: {e}")
            return None
    
    def _update_portfolio_risk(self):
        """Update portfolio-level risk metrics"""
        try:
            if not self.surface_data:
                return
            
            # Calculate portfolio IV (weighted average)
            total_volume = 0
            weighted_iv = 0
            
            symbol_weights = {}
            symbol_ivs = {}
            
            for symbol, update in self.surface_data.items():
                if not update.success or 'mean_iv' not in update.statistics:
                    continue
                
                volume = update.statistics.get('total_volume', 0)
                iv = update.statistics['mean_iv']
                
                total_volume += volume
                weighted_iv += volume * iv
                
                symbol_weights[symbol] = volume
                symbol_ivs[symbol] = iv
            
            if total_volume == 0:
                return
            
            # Normalize weights
            for symbol in symbol_weights:
                symbol_weights[symbol] /= total_volume
            
            portfolio_iv = weighted_iv / total_volume
            
            # Calculate diversification ratio
            individual_risk = sum(weight * iv for weight, iv in zip(symbol_weights.values(), symbol_ivs.values()))
            portfolio_risk = portfolio_iv
            diversification_ratio = individual_risk / portfolio_risk if portfolio_risk > 0 else 1.0
            
            # Identify concentrated positions
            concentration_limit = self.risk_params['risk_concentration_limit']
            concentrated_symbols = [
                symbol for symbol, weight in symbol_weights.items()
                if weight > concentration_limit
            ]
            
            # Calculate risk contribution
            risk_contribution = {}
            for symbol, weight in symbol_weights.items():
                if symbol in symbol_ivs:
                    contribution = weight * symbol_ivs[symbol] / portfolio_iv if portfolio_iv > 0 else 0
                    risk_contribution[symbol] = contribution
            
            # Calculate VaR (simplified)
            var_multiplier = stats.norm.ppf(self.risk_params['var_confidence'])
            portfolio_var = portfolio_iv * var_multiplier
            
            # Get correlation matrix
            correlation_matrix = self.calculate_correlation_matrix()
            
            # Create portfolio risk object
            portfolio_risk = PortfolioRisk(
                timestamp=datetime.now(),
                portfolio_iv=portfolio_iv,
                portfolio_var=portfolio_var,
                portfolio_diversification_ratio=diversification_ratio,
                concentrated_risk_symbols=concentrated_symbols,
                risk_contribution=risk_contribution,
                correlation_matrix=correlation_matrix
            )
            
            self.portfolio_risk_history.append(portfolio_risk)
            
            # Keep only recent history
            max_history = 1000
            if len(self.portfolio_risk_history) > max_history:
                self.portfolio_risk_history = self.portfolio_risk_history[-max_history:]
            
            self.logger.debug(
                f"Updated portfolio risk: IV={portfolio_iv:.4f}, "
                f"VaR={portfolio_var:.4f}, Diversification={diversification_ratio:.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio risk: {e}")
    
    def _detect_cross_asset_signals(self):
        """Detect cross-asset trading signals"""
        try:
            current_time = datetime.now()
            new_signals = []
            
            # Volatility spread signals
            vol_spread_signals = self._detect_volatility_spreads()
            new_signals.extend(vol_spread_signals)
            
            # Correlation break signals
            correlation_signals = self._detect_correlation_breaks()
            new_signals.extend(correlation_signals)
            
            # Sector rotation signals
            sector_signals = self._detect_sector_rotation()
            new_signals.extend(sector_signals)
            
            # Update active signals
            self.active_signals = [
                signal for signal in new_signals
                if signal.strength >= self.risk_params['signal_threshold']
            ]
            
            # Add to history
            self.signal_history.extend(new_signals)
            
            # Keep only recent history
            max_history = 500
            if len(self.signal_history) > max_history:
                self.signal_history = self.signal_history[-max_history:]
            
            if new_signals:
                self.logger.info(f"Detected {len(new_signals)} cross-asset signals")
            
        except Exception as e:
            self.logger.error(f"Error detecting cross-asset signals: {e}")
    
    def _detect_volatility_spreads(self) -> List[CrossAssetSignal]:
        """Detect unusual volatility spreads between assets"""
        signals = []
        
        try:
            symbols = list(self.surface_data.keys())
            
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    update1 = self.surface_data[symbol1]
                    update2 = self.surface_data[symbol2]
                    
                    if not (update1.success and update2.success):
                        continue
                    
                    iv1 = update1.statistics.get('mean_iv', 0)
                    iv2 = update2.statistics.get('mean_iv', 0)
                    
                    if iv1 == 0 or iv2 == 0:
                        continue
                    
                    # Calculate volatility spread
                    vol_spread = abs(iv1 - iv2) / max(iv1, iv2)
                    
                    # Check if spread is unusually wide
                    # In production, would compare to historical spreads
                    if vol_spread > 0.3:  # 30% difference
                        strength = min(vol_spread / 0.5, 1.0)  # Normalize to 0-1
                        
                        signals.append(CrossAssetSignal(
                            signal_type='vol_spread',
                            symbols=[symbol1, symbol2],
                            strength=strength,
                            description=f"Wide volatility spread between {symbol1} ({iv1:.2%}) and {symbol2} ({iv2:.2%})",
                            timestamp=datetime.now(),
                            supporting_data={
                                'iv1': iv1,
                                'iv2': iv2,
                                'spread': vol_spread,
                                'higher_vol_symbol': symbol1 if iv1 > iv2 else symbol2
                            }
                        ))
        
        except Exception as e:
            self.logger.error(f"Error detecting volatility spreads: {e}")
        
        return signals
    
    def _detect_correlation_breaks(self) -> List[CrossAssetSignal]:
        """Detect correlation breakdown between assets"""
        signals = []
        
        try:
            # Check for pairs with historically high correlation that are now diverging
            for pair, correlation_history in self.correlation_history.items():
                if len(correlation_history) < 5:  # Need some history
                    continue
                
                recent_correlations = correlation_history[-5:]
                historical_correlations = correlation_history[-20:-5] if len(correlation_history) > 20 else []
                
                if not historical_correlations:
                    continue
                
                recent_avg = np.mean([c.iv_correlation for c in recent_correlations])
                historical_avg = np.mean([c.iv_correlation for c in historical_correlations])
                
                # Check for correlation breakdown
                correlation_drop = historical_avg - recent_avg
                
                if correlation_drop > 0.3 and historical_avg > 0.5:  # Significant drop from high correlation
                    strength = min(correlation_drop / 0.5, 1.0)
                    
                    signals.append(CrossAssetSignal(
                        signal_type='correlation_break',
                        symbols=list(pair),
                        strength=strength,
                        description=f"Correlation breakdown between {pair[0]} and {pair[1]} "
                                  f"(from {historical_avg:.2f} to {recent_avg:.2f})",
                        timestamp=datetime.now(),
                        supporting_data={
                            'historical_correlation': historical_avg,
                            'recent_correlation': recent_avg,
                            'correlation_drop': correlation_drop
                        }
                    ))
        
        except Exception as e:
            self.logger.error(f"Error detecting correlation breaks: {e}")
        
        return signals
    
    def _detect_sector_rotation(self) -> List[CrossAssetSignal]:
        """Detect sector rotation signals"""
        signals = []
        
        try:
            # Group assets by sector
            sector_performance = defaultdict(list)
            
            for symbol, update in self.surface_data.items():
                if not update.success or 'mean_iv' not in update.statistics:
                    continue
                
                sector = self.asset_sectors.get(symbol, 'other')
                iv = update.statistics['mean_iv']
                volume = update.statistics.get('total_volume', 0)
                
                sector_performance[sector].append({
                    'symbol': symbol,
                    'iv': iv,
                    'volume': volume
                })
            
            # Calculate sector-level metrics
            sector_metrics = {}
            for sector, assets in sector_performance.items():
                if len(assets) < 2:  # Need at least 2 assets per sector
                    continue
                
                total_volume = sum(asset['volume'] for asset in assets)
                if total_volume == 0:
                    continue
                
                # Weighted average IV by volume
                weighted_iv = sum(asset['iv'] * asset['volume'] for asset in assets) / total_volume
                avg_volume = total_volume / len(assets)
                
                sector_metrics[sector] = {
                    'weighted_iv': weighted_iv,
                    'avg_volume': avg_volume,
                    'asset_count': len(assets)
                }
            
            # Detect rotation signals
            if len(sector_metrics) >= 2:
                sectors = list(sector_metrics.keys())
                
                # Find sectors with diverging volatility
                for i, sector1 in enumerate(sectors):
                    for sector2 in sectors[i+1:]:
                        iv1 = sector_metrics[sector1]['weighted_iv']
                        iv2 = sector_metrics[sector2]['weighted_iv']
                        
                        vol1 = sector_metrics[sector1]['avg_volume']
                        vol2 = sector_metrics[sector2]['avg_volume']
                        
                        # Check for significant divergence
                        iv_ratio = max(iv1, iv2) / min(iv1, iv2) if min(iv1, iv2) > 0 else 1
                        vol_ratio = max(vol1, vol2) / min(vol1, vol2) if min(vol1, vol2) > 0 else 1
                        
                        if iv_ratio > 1.5 or vol_ratio > 2.0:  # Significant divergence
                            strength = min((iv_ratio - 1) / 1.5 + (vol_ratio - 1) / 3.0, 1.0)
                            
                            hot_sector = sector1 if iv1 > iv2 else sector2
                            cold_sector = sector2 if iv1 > iv2 else sector1
                            
                            signals.append(CrossAssetSignal(
                                signal_type='sector_rotation',
                                symbols=[symbol['symbol'] for symbol in sector_performance[hot_sector] + sector_performance[cold_sector]],
                                strength=strength,
                                description=f"Sector rotation: {hot_sector} showing higher volatility/volume vs {cold_sector}",
                                timestamp=datetime.now(),
                                supporting_data={
                                    'hot_sector': hot_sector,
                                    'cold_sector': cold_sector,
                                    'iv_ratio': iv_ratio,
                                    'volume_ratio': vol_ratio,
                                    'sector_metrics': {hot_sector: sector_metrics[hot_sector], cold_sector: sector_metrics[cold_sector]}
                                }
                            ))
        
        except Exception as e:
            self.logger.error(f"Error detecting sector rotation: {e}")
        
        return signals
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            if not self.portfolio_risk_history:
                return {'status': 'no_data'}
            
            latest_risk = self.portfolio_risk_history[-1]
            
            # Active assets
            active_assets = list(self.surface_data.keys())
            
            # Performance summary
            successful_updates = sum(1 for update in self.surface_data.values() if update.success)
            
            # Risk summary
            risk_summary = {
                'portfolio_iv': latest_risk.portfolio_iv,
                'portfolio_var': latest_risk.portfolio_var,
                'diversification_ratio': latest_risk.portfolio_diversification_ratio,
                'concentrated_positions': latest_risk.concentrated_risk_symbols,
                'top_risk_contributors': sorted(
                    latest_risk.risk_contribution.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
            
            # Signal summary
            signal_summary = {
                'active_signals': len(self.active_signals),
                'signals_by_type': defaultdict(int),
                'high_strength_signals': []
            }
            
            for signal in self.active_signals:
                signal_summary['signals_by_type'][signal.signal_type] += 1
                if signal.strength > 0.8:
                    signal_summary['high_strength_signals'].append({
                        'type': signal.signal_type,
                        'symbols': signal.symbols,
                        'strength': signal.strength,
                        'description': signal.description
                    })
            
            # Correlation insights
            correlation_insights = self._get_correlation_insights()
            
            return {
                'timestamp': latest_risk.timestamp,
                'status': 'active',
                'active_assets': active_assets,
                'successful_updates': successful_updates,
                'total_assets': len(active_assets),
                'risk_summary': risk_summary,
                'signal_summary': dict(signal_summary),
                'correlation_insights': correlation_insights,
                'system_health': {
                    'portfolio_risk_updates': len(self.portfolio_risk_history),
                    'correlation_pairs': len(self.correlation_history),
                    'signal_history_length': len(self.signal_history)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio summary: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _get_correlation_insights(self) -> Dict[str, Any]:
        """Generate insights from correlation analysis"""
        try:
            insights = {
                'highest_correlations': [],
                'lowest_correlations': [],
                'unstable_correlations': []
            }
            
            # Get recent correlations
            recent_correlations = []
            for pair, history in self.correlation_history.items():
                if history:
                    latest = history[-1]
                    recent_correlations.append(latest)
            
            if not recent_correlations:
                return insights
            
            # Sort by correlation strength
            sorted_by_correlation = sorted(recent_correlations, key=lambda x: abs(x.iv_correlation), reverse=True)
            
            # Highest correlations
            insights['highest_correlations'] = [
                {
                    'pair': f"{corr.symbol_pair[0]}-{corr.symbol_pair[1]}",
                    'correlation': corr.iv_correlation,
                    'stability': corr.correlation_stability
                }
                for corr in sorted_by_correlation[:5]
            ]
            
            # Lowest correlations
            insights['lowest_correlations'] = [
                {
                    'pair': f"{corr.symbol_pair[0]}-{corr.symbol_pair[1]}",
                    'correlation': corr.iv_correlation,
                    'stability': corr.correlation_stability
                }
                for corr in sorted_by_correlation[-5:]
            ]
            
            # Unstable correlations
            unstable = [corr for corr in recent_correlations if corr.correlation_stability < 0.5]
            insights['unstable_correlations'] = [
                {
                    'pair': f"{corr.symbol_pair[0]}-{corr.symbol_pair[1]}",
                    'correlation': corr.iv_correlation,
                    'stability': corr.correlation_stability
                }
                for corr in sorted(unstable, key=lambda x: x.correlation_stability)[:5]
            ]
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating correlation insights: {e}")
            return {}
    
    def generate_risk_report(self) -> str:
        """Generate a comprehensive risk report"""
        try:
            if not self.portfolio_risk_history:
                return "No risk data available"
            
            latest_risk = self.portfolio_risk_history[-1]
            
            report = []
            report.append("PORTFOLIO VOLATILITY RISK REPORT")
            report.append("=" * 50)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Data as of: {latest_risk.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Risk metrics
            report.append("RISK METRICS")
            report.append("-" * 20)
            report.append(f"Portfolio Implied Volatility: {latest_risk.portfolio_iv:.2%}")
            report.append(f"Portfolio VaR (95%): {latest_risk.portfolio_var:.2%}")
            report.append(f"Diversification Ratio: {latest_risk.portfolio_diversification_ratio:.3f}")
            report.append("")
            
            # Risk concentration
            if latest_risk.concentrated_risk_symbols:
                report.append("RISK CONCENTRATION ALERTS")
                report.append("-" * 30)
                for symbol in latest_risk.concentrated_risk_symbols:
                    contribution = latest_risk.risk_contribution.get(symbol, 0)
                    report.append(f"  {symbol}: {contribution:.1%} risk contribution")
                report.append("")
            
            # Top risk contributors
            report.append("TOP RISK CONTRIBUTORS")
            report.append("-" * 25)
            sorted_contributors = sorted(
                latest_risk.risk_contribution.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for symbol, contribution in sorted_contributors[:10]:
                report.append(f"  {symbol}: {contribution:.1%}")
            report.append("")
            
            # Active signals
            if self.active_signals:
                report.append("ACTIVE CROSS-ASSET SIGNALS")
                report.append("-" * 30)
                for signal in sorted(self.active_signals, key=lambda x: x.strength, reverse=True):
                    report.append(f"  [{signal.signal_type.upper()}] {signal.description}")
                    report.append(f"    Strength: {signal.strength:.2f} | Symbols: {', '.join(signal.symbols)}")
                    report.append("")
            
            # Correlation matrix summary
            if not latest_risk.correlation_matrix.empty:
                report.append("CORRELATION MATRIX SUMMARY")
                report.append("-" * 30)
                
                # Find highest and lowest correlations
                corr_matrix = latest_risk.correlation_matrix
                upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                if not upper_triangle.isna().all().all():
                    max_corr = upper_triangle.max().max()
                    min_corr = upper_triangle.min().min()
                    avg_corr = upper_triangle.mean().mean()
                    
                    report.append(f"  Average Correlation: {avg_corr:.3f}")
                    report.append(f"  Highest Correlation: {max_corr:.3f}")
                    report.append(f"  Lowest Correlation: {min_corr:.3f}")
                report.append("")
            
            # Historical trend
            if len(self.portfolio_risk_history) > 1:
                report.append("RISK TREND ANALYSIS")
                report.append("-" * 25)
                
                # Compare to previous period
                prev_risk = self.portfolio_risk_history[-2]
                iv_change = latest_risk.portfolio_iv - prev_risk.portfolio_iv
                var_change = latest_risk.portfolio_var - prev_risk.portfolio_var
                
                report.append(f"  IV Change: {iv_change:+.2%}")
                report.append(f"  VaR Change: {var_change:+.2%}")
                
                # Trend over longer period
                if len(self.portfolio_risk_history) >= 10:
                    recent_ivs = [r.portfolio_iv for r in self.portfolio_risk_history[-10:]]
                    iv_trend = np.polyfit(range(len(recent_ivs)), recent_ivs, 1)[0]
                    trend_direction = "increasing" if iv_trend > 0 else "decreasing"
                    report.append(f"  10-Period Trend: {trend_direction} ({iv_trend:+.4f}/period)")
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return f"Error generating risk report: {str(e)}"
    
    def get_cross_asset_opportunities(self) -> List[Dict[str, Any]]:
        """Identify cross-asset trading opportunities"""
        opportunities = []
        
        try:
            # Volatility arbitrage opportunities
            for signal in self.active_signals:
                if signal.signal_type == 'vol_spread' and signal.strength > 0.7:
                    opportunities.append({
                        'type': 'volatility_arbitrage',
                        'description': f"Trade volatility spread between {signal.symbols[0]} and {signal.symbols[1]}",
                        'symbols': signal.symbols,
                        'strength': signal.strength,
                        'strategy': 'Long lower vol, short higher vol',
                        'supporting_data': signal.supporting_data
                    })
                
                elif signal.signal_type == 'correlation_break' and signal.strength > 0.8:
                    opportunities.append({
                        'type': 'correlation_trade',
                        'description': f"Correlation breakdown trade between {signal.symbols[0]} and {signal.symbols[1]}",
                        'symbols': signal.symbols,
                        'strength': signal.strength,
                        'strategy': 'Pairs trade on correlation reversion',
                        'supporting_data': signal.supporting_data
                    })
                
                elif signal.signal_type == 'sector_rotation' and signal.strength > 0.6:
                    opportunities.append({
                        'type': 'sector_rotation',
                        'description': signal.description,
                        'symbols': signal.symbols,
                        'strength': signal.strength,
                        'strategy': f"Long {signal.supporting_data['hot_sector']}, short {signal.supporting_data['cold_sector']}",
                        'supporting_data': signal.supporting_data
                    })
            
            # Sort by strength
            opportunities.sort(key=lambda x: x['strength'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error identifying opportunities: {e}")
        
        return opportunities
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time data for dashboard display"""
        try:
            portfolio_summary = self.get_portfolio_summary()
            opportunities = self.get_cross_asset_opportunities()
            
            # Current surface data
            surface_summary = {}
            for symbol, update in self.surface_data.items():
                if update.success:
                    surface_summary[symbol] = {
                        'timestamp': update.timestamp,
                        'mean_iv': update.statistics.get('mean_iv', 0),
                        'contracts': update.statistics.get('total_contracts', 0),
                        'volume': update.statistics.get('total_volume', 0),
                        'processing_time': update.processing_time
                    }
            
            return {
                'portfolio_summary': portfolio_summary,
                'surface_summary': surface_summary,
                'opportunities': opportunities,
                'active_signals': [
                    {
                        'type': signal.signal_type,
                        'symbols': signal.symbols,
                        'strength': signal.strength,
                        'description': signal.description,
                        'timestamp': signal.timestamp
                    }
                    for signal in self.active_signals
                ],
                'correlation_matrix': self.calculate_correlation_matrix().to_dict() if not self.calculate_correlation_matrix().empty else {},
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {
                'error': str(e),
                'last_updated': datetime.now()
            }