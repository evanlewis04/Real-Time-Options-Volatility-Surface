"""
Real-Time Stream Processor for Volatility Surface System
Handles continuous data streams and incremental surface updates
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty

from src.data.data_manager import DataManager
from src.analysis.vol_surface import VolatilitySurface
from src.pricing.implied_vol import ImpliedVolatilityCalculator


@dataclass
class SurfaceUpdate:
    """Container for volatility surface updates"""
    symbol: str
    timestamp: datetime
    surface: Optional[VolatilitySurface]
    statistics: Dict[str, float]
    success: bool = True
    error_message: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class AlertEvent:
    """Container for market alerts"""
    symbol: str
    alert_type: str
    message: str
    severity: str  # low, medium, high, critical
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)


class StreamProcessor:
    """
    Real-time stream processor for multi-asset volatility surfaces
    Manages continuous data updates and surface reconstruction
    """
    
    def __init__(self, data_manager: DataManager):
        self.logger = logging.getLogger(__name__)
        self.data_manager = data_manager
        self.iv_calculator = ImpliedVolatilityCalculator()
        
        # Surface management
        self.surfaces: Dict[str, VolatilitySurface] = {}
        self.surface_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)  # Keep last 100 surface updates
        )
        
        # Processing queues
        self.update_queue = Queue(maxsize=1000)
        self.surface_queue = Queue(maxsize=500)
        self.alert_queue = Queue(maxsize=200)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.processing_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()
        
        # Callbacks
        self.surface_callbacks: List[Callable[[SurfaceUpdate], None]] = []
        self.alert_callbacks: List[Callable[[AlertEvent], None]] = []
        
        # Performance tracking
        self.processing_stats = defaultdict(list)
        self.last_surface_times: Dict[str, datetime] = {}
        
        # Alert thresholds
        self.alert_thresholds = {
            'iv_spike': 0.20,  # 20% increase in IV
            'volume_spike': 3.0,  # 3x average volume
            'surface_error': 0.05,  # 5% surface fit error
            'data_gap': 300,  # 5 minutes without data
        }
        
        # Register with data manager
        self.data_manager.register_update_callback(self.on_market_update)
        
        self.logger.info("Stream processor initialized")
    
    def start_processing(self):
        """Start all processing threads"""
        self.logger.info("Starting stream processing threads")
        
        # Start processing threads
        threads = [
            ('update_processor', self._update_processing_loop),
            ('surface_processor', self._surface_processing_loop),
            ('alert_processor', self._alert_processing_loop),
            ('health_monitor', self._health_monitoring_loop)
        ]
        
        for name, target in threads:
            thread = threading.Thread(target=target, daemon=True, name=name)
            thread.start()
            self.processing_threads.append(thread)
            self.logger.info(f"Started {name} thread")
        
        self.logger.info(f"All {len(threads)} processing threads started")
    
    def stop_processing(self):
        """Stop all processing threads"""
        self.logger.info("Stopping stream processing")
        self.stop_event.set()
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=5)
        
        self.processing_threads.clear()
        self.stop_event.clear()
        self.executor.shutdown(wait=True)
        self.logger.info("Stream processing stopped")
    
    def on_market_update(self, update):
        """Handle incoming market data updates"""
        if not update.success:
            self.logger.warning(f"Received failed update for {update.symbol}: {update.error_message}")
            return
        
        try:
            # Add to processing queue
            self.update_queue.put(update, timeout=1)
            self.logger.debug(f"Queued market update for {update.symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to queue update for {update.symbol}: {e}")
    
    def _update_processing_loop(self):
        """Main loop for processing market updates"""
        while not self.stop_event.is_set():
            try:
                # Get next update
                update = self.update_queue.get(timeout=1)
                
                # Process the update
                self._process_market_update(update)
                
                # Mark task as done
                self.update_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in update processing loop: {e}", exc_info=True)
    
    def _process_market_update(self, update):
        """Process a single market update"""
        start_time = time.time()
        symbol = update.symbol
        
        try:
            # Check for data quality issues
            data_alerts = self._check_data_quality(update)
            for alert in data_alerts:
                self.alert_queue.put(alert)
            
            # Calculate implied volatilities
            options_with_iv = self._calculate_implied_volatilities(update)
            
            if options_with_iv.empty:
                self.logger.warning(f"No valid options with IV for {symbol}")
                return
            
            # Check for volatility alerts
            vol_alerts = self._check_volatility_alerts(symbol, options_with_iv)
            for alert in vol_alerts:
                self.alert_queue.put(alert)
            
            # Queue for surface construction
            surface_data = {
                'symbol': symbol,
                'timestamp': update.timestamp,
                'options_data': options_with_iv,
                'underlying_price': update.underlying_price,
                'risk_free_rate': update.risk_free_rate
            }
            
            self.surface_queue.put(surface_data, timeout=1)
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_stats[symbol].append({
                'timestamp': update.timestamp,
                'processing_time': processing_time,
                'contract_count': len(options_with_iv),
                'stage': 'market_update'
            })
            
            self.logger.debug(
                f"Processed market update for {symbol} in {processing_time:.3f}s "
                f"({len(options_with_iv)} contracts)"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing market update for {symbol}: {e}", exc_info=True)
    
    def _calculate_implied_volatilities(self, update) -> pd.DataFrame:
        """Calculate implied volatilities for options data"""
        options_data = update.options_data.copy()
        
        if options_data.empty:
            return options_data
        
        # Add implied volatility column
        options_data['implied_volatility'] = np.nan
        
        # Calculate IV for each option
        for idx, row in options_data.iterrows():
            try:
                iv = self.iv_calculator.calculate_iv(
                    option_price=row['last_price'],
                    underlying_price=update.underlying_price,
                    strike=row['strike'],
                    time_to_expiry=row['time_to_expiry'],
                    risk_free_rate=update.risk_free_rate,
                    option_type=row['type']
                )
                
                # Only accept reasonable IV values
                if 0.01 <= iv <= 5.0:  # 1% to 500%
                    options_data.at[idx, 'implied_volatility'] = iv
                    
            except Exception as e:
                self.logger.debug(f"Failed to calculate IV for {update.symbol} option: {e}")
                continue
        
        # Filter out options without valid IV
        valid_options = options_data.dropna(subset=['implied_volatility'])
        
        self.logger.debug(
            f"Calculated IV for {len(valid_options)}/{len(options_data)} options "
            f"for {update.symbol}"
        )
        
        return valid_options
    
    def _surface_processing_loop(self):
        """Main loop for constructing volatility surfaces"""
        while not self.stop_event.is_set():
            try:
                # Get next surface data
                surface_data = self.surface_queue.get(timeout=1)
                
                # Construct surface
                surface_update = self._construct_surface(surface_data)
                
                # Notify callbacks
                for callback in self.surface_callbacks:
                    try:
                        callback(surface_update)
                    except Exception as e:
                        self.logger.error(f"Error in surface callback: {e}")
                
                # Mark task as done
                self.surface_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in surface processing loop: {e}", exc_info=True)
    
    def _construct_surface(self, surface_data: Dict) -> SurfaceUpdate:
        """Construct volatility surface from options data"""
        start_time = time.time()
        symbol = surface_data['symbol']
        
        try:
            # Create surface
            surface = VolatilitySurface(
                symbol=symbol,
                underlying_price=surface_data['underlying_price'],
                risk_free_rate=surface_data['risk_free_rate']
            )
            
            # Fit surface to data
            surface.fit_surface(surface_data['options_data'])
            
            # Calculate statistics
            stats = self._calculate_surface_statistics(surface, surface_data['options_data'])
            
            # Store surface
            self.surfaces[symbol] = surface
            self.last_surface_times[symbol] = surface_data['timestamp']
            
            # Add to history
            self.surface_history[symbol].append({
                'timestamp': surface_data['timestamp'],
                'surface': surface,
                'statistics': stats
            })
            
            processing_time = time.time() - start_time
            
            # Track processing time
            self.processing_stats[symbol].append({
                'timestamp': surface_data['timestamp'],
                'processing_time': processing_time,
                'stage': 'surface_construction'
            })
            
            surface_update = SurfaceUpdate(
                symbol=symbol,
                timestamp=surface_data['timestamp'],
                surface=surface,
                statistics=stats,
                success=True,
                processing_time=processing_time
            )
            
            self.logger.info(
                f"Constructed volatility surface for {symbol} in {processing_time:.3f}s "
                f"(fit error: {stats.get('fit_error', 'N/A'):.4f})"
            )
            
            return surface_update
            
        except Exception as e:
            error_msg = f"Failed to construct surface for {symbol}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            processing_time = time.time() - start_time
            
            return SurfaceUpdate(
                symbol=symbol,
                timestamp=surface_data['timestamp'],
                surface=None,
                statistics={},
                success=False,
                error_message=error_msg,
                processing_time=processing_time
            )
    
    def _calculate_surface_statistics(self, surface: VolatilitySurface, options_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistics for the volatility surface"""
        try:
            stats = {}
            
            # Fit error (RMSE of surface vs market IVs)
            if hasattr(surface, 'interpolator') and surface.interpolator is not None:
                predicted_ivs = []
                actual_ivs = []
                
                for _, row in options_data.iterrows():
                    try:
                        predicted_iv = surface.interpolator(
                            row['time_to_expiry'], 
                            row['moneyness']
                        )
                        predicted_ivs.append(predicted_iv)
                        actual_ivs.append(row['implied_volatility'])
                    except:
                        continue
                
                if predicted_ivs and actual_ivs:
                    rmse = np.sqrt(np.mean((np.array(predicted_ivs) - np.array(actual_ivs))**2))
                    stats['fit_error'] = rmse
                    stats['mean_absolute_error'] = np.mean(np.abs(np.array(predicted_ivs) - np.array(actual_ivs)))
            
            # IV statistics
            ivs = options_data['implied_volatility'].dropna()
            if not ivs.empty:
                stats['mean_iv'] = ivs.mean()
                stats['median_iv'] = ivs.median()
                stats['iv_std'] = ivs.std()
                stats['min_iv'] = ivs.min()
                stats['max_iv'] = ivs.max()
                stats['iv_range'] = stats['max_iv'] - stats['min_iv']
            
            # Moneyness statistics
            moneyness = options_data['moneyness'].dropna()
            if not moneyness.empty:
                stats['min_moneyness'] = moneyness.min()
                stats['max_moneyness'] = moneyness.max()
                stats['moneyness_range'] = stats['max_moneyness'] - stats['min_moneyness']
            
            # Time to expiry statistics
            tte = options_data['time_to_expiry'].dropna()
            if not tte.empty:
                stats['min_tte'] = tte.min()
                stats['max_tte'] = tte.max()
                stats['tte_range'] = stats['max_tte'] - stats['min_tte']
            
            # Contract counts
            stats['total_contracts'] = len(options_data)
            stats['call_contracts'] = len(options_data[options_data['type'] == 'call'])
            stats['put_contracts'] = len(options_data[options_data['type'] == 'put'])
            
            # Volume and open interest
            if 'volume' in options_data.columns:
                stats['total_volume'] = options_data['volume'].sum()
                stats['avg_volume'] = options_data['volume'].mean()
            
            if 'open_interest' in options_data.columns:
                stats['total_open_interest'] = options_data['open_interest'].sum()
                stats['avg_open_interest'] = options_data['open_interest'].mean()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating surface statistics: {e}")
            return {}
    
    def _check_data_quality(self, update) -> List[AlertEvent]:
        """Check for data quality issues and generate alerts"""
        alerts = []
        symbol = update.symbol
        
        try:
            # Check for empty data
            if update.options_data.empty:
                alerts.append(AlertEvent(
                    symbol=symbol,
                    alert_type='data_quality',
                    message=f"No options data received for {symbol}",
                    severity='medium',
                    timestamp=update.timestamp
                ))
            
            # Check for data freshness
            if symbol in self.last_surface_times:
                time_since_last = update.timestamp - self.last_surface_times[symbol]
                if time_since_last.total_seconds() > self.alert_thresholds['data_gap']:
                    alerts.append(AlertEvent(
                        symbol=symbol,
                        alert_type='data_gap',
                        message=f"Data gap of {time_since_last.total_seconds():.0f}s for {symbol}",
                        severity='medium',
                        timestamp=update.timestamp,
                        data={'gap_seconds': time_since_last.total_seconds()}
                    ))
            
            # Check for unusual data patterns
            if not update.options_data.empty:
                # Check for extreme spreads
                if 'bid' in update.options_data.columns and 'ask' in update.options_data.columns:
                    spreads = (update.options_data['ask'] - update.options_data['bid']) / update.options_data['last_price']
                    extreme_spreads = spreads[spreads > 0.5].count()  # More than 50% spread
                    
                    if extreme_spreads > len(update.options_data) * 0.2:  # More than 20% of contracts
                        alerts.append(AlertEvent(
                            symbol=symbol,
                            alert_type='data_quality',
                            message=f"High proportion of extreme spreads in {symbol} options data",
                            severity='low',
                            timestamp=update.timestamp,
                            data={'extreme_spread_ratio': extreme_spreads / len(update.options_data)}
                        ))
                
                # Check for zero volumes
                if 'volume' in update.options_data.columns:
                    zero_volume_ratio = (update.options_data['volume'] == 0).sum() / len(update.options_data)
                    if zero_volume_ratio > 0.8:  # More than 80% zero volume
                        alerts.append(AlertEvent(
                            symbol=symbol,
                            alert_type='data_quality',
                            message=f"High proportion of zero-volume options for {symbol}",
                            severity='low',
                            timestamp=update.timestamp,
                            data={'zero_volume_ratio': zero_volume_ratio}
                        ))
            
        except Exception as e:
            self.logger.error(f"Error checking data quality for {symbol}: {e}")
        
        return alerts
    
    def _check_volatility_alerts(self, symbol: str, options_data: pd.DataFrame) -> List[AlertEvent]:
        """Check for volatility-related alerts"""
        alerts = []
        
        try:
            if options_data.empty or 'implied_volatility' not in options_data.columns:
                return alerts
            
            ivs = options_data['implied_volatility'].dropna()
            if ivs.empty:
                return alerts
            
            current_mean_iv = ivs.mean()
            
            # Check against historical IV if available
            if symbol in self.surface_history and len(self.surface_history[symbol]) > 0:
                # Get recent historical IV
                recent_history = list(self.surface_history[symbol])[-10:]  # Last 10 updates
                historical_ivs = []
                
                for hist_entry in recent_history:
                    if 'mean_iv' in hist_entry['statistics']:
                        historical_ivs.append(hist_entry['statistics']['mean_iv'])
                
                if historical_ivs:
                    historical_mean = np.mean(historical_ivs)
                    iv_change = (current_mean_iv - historical_mean) / historical_mean
                    
                    # IV spike alert
                    if abs(iv_change) > self.alert_thresholds['iv_spike']:
                        severity = 'high' if abs(iv_change) > 0.5 else 'medium'
                        direction = 'increased' if iv_change > 0 else 'decreased'
                        
                        alerts.append(AlertEvent(
                            symbol=symbol,
                            alert_type='iv_spike',
                            message=f"Implied volatility {direction} by {abs(iv_change)*100:.1f}% for {symbol}",
                            severity=severity,
                            timestamp=datetime.now(),
                            data={
                                'current_iv': current_mean_iv,
                                'historical_iv': historical_mean,
                                'change_percent': iv_change * 100
                            }
                        ))
            
            # Check for extreme IV values
            if current_mean_iv > 2.0:  # More than 200% IV
                alerts.append(AlertEvent(
                    symbol=symbol,
                    alert_type='extreme_iv',
                    message=f"Extremely high implied volatility ({current_mean_iv*100:.1f}%) for {symbol}",
                    severity='high',
                    timestamp=datetime.now(),
                    data={'mean_iv': current_mean_iv}
                ))
            
            # Check volume spikes
            if 'volume' in options_data.columns:
                total_volume = options_data['volume'].sum()
                
                # Compare to historical volume if available
                if symbol in self.surface_history and len(self.surface_history[symbol]) > 0:
                    recent_volumes = []
                    for hist_entry in list(self.surface_history[symbol])[-5:]:
                        if 'total_volume' in hist_entry['statistics']:
                            recent_volumes.append(hist_entry['statistics']['total_volume'])
                    
                    if recent_volumes:
                        avg_volume = np.mean(recent_volumes)
                        if avg_volume > 0 and total_volume > avg_volume * self.alert_thresholds['volume_spike']:
                            alerts.append(AlertEvent(
                                symbol=symbol,
                                alert_type='volume_spike',
                                message=f"Options volume spike for {symbol}: {total_volume:,.0f} vs avg {avg_volume:,.0f}",
                                severity='medium',
                                timestamp=datetime.now(),
                                data={
                                    'current_volume': total_volume,
                                    'average_volume': avg_volume,
                                    'spike_ratio': total_volume / avg_volume
                                }
                            ))
        
        except Exception as e:
            self.logger.error(f"Error checking volatility alerts for {symbol}: {e}")
        
        return alerts
    
    def _alert_processing_loop(self):
        """Main loop for processing alerts"""
        while not self.stop_event.is_set():
            try:
                # Get next alert
                alert = self.alert_queue.get(timeout=1)
                
                # Process the alert
                self._process_alert(alert)
                
                # Notify callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        self.logger.error(f"Error in alert callback: {e}")
                
                # Mark task as done
                self.alert_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in alert processing loop: {e}", exc_info=True)
    
    def _process_alert(self, alert: AlertEvent):
        """Process a single alert"""
        # Log the alert
        log_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(alert.severity, logging.INFO)
        
        self.logger.log(log_level, f"[{alert.alert_type.upper()}] {alert.message}")
        
        # Additional processing based on alert type
        if alert.alert_type == 'iv_spike' and alert.severity in ['high', 'critical']:
            # Could trigger additional analysis or notifications
            pass
        elif alert.alert_type == 'data_gap' and alert.severity in ['high', 'critical']:
            # Could trigger data source failover
            pass
    
    def _health_monitoring_loop(self):
        """Monitor system health and performance"""
        while not self.stop_event.is_set():
            try:
                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds
                
                # Check queue sizes
                update_queue_size = self.update_queue.qsize()
                surface_queue_size = self.surface_queue.qsize()
                alert_queue_size = self.alert_queue.qsize()
                
                # Log queue sizes if they're getting large
                if update_queue_size > 100:
                    self.logger.warning(f"Large update queue: {update_queue_size} items")
                if surface_queue_size > 50:
                    self.logger.warning(f"Large surface queue: {surface_queue_size} items")
                if alert_queue_size > 20:
                    self.logger.warning(f"Large alert queue: {alert_queue_size} items")
                
                # Check for stale surfaces
                now = datetime.now()
                for symbol, last_time in self.last_surface_times.items():
                    if (now - last_time).total_seconds() > 600:  # 10 minutes
                        self.logger.warning(f"Stale surface for {symbol}: last updated {last_time}")
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
    
    def register_surface_callback(self, callback: Callable[[SurfaceUpdate], None]):
        """Register callback for surface updates"""
        self.surface_callbacks.append(callback)
    
    def register_alert_callback(self, callback: Callable[[AlertEvent], None]):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_current_surface(self, symbol: str) -> Optional[VolatilitySurface]:
        """Get current volatility surface for a symbol"""
        return self.surfaces.get(symbol)
    
    def get_all_surfaces(self) -> Dict[str, VolatilitySurface]:
        """Get all current volatility surfaces"""
        return self.surfaces.copy()
    
    def get_surface_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get surface history for a symbol"""
        if symbol not in self.surface_history:
            return []
        
        history = list(self.surface_history[symbol])
        return history[-limit:] if limit else history
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics"""
        stats = {}
        
        for symbol in self.processing_stats:
            symbol_stats = self.processing_stats[symbol]
            if not symbol_stats:
                continue
            
            recent_stats = symbol_stats[-20:]  # Last 20 operations
            
            # Separate by processing stage
            update_stats = [s for s in recent_stats if s['stage'] == 'market_update']
            surface_stats = [s for s in recent_stats if s['stage'] == 'surface_construction']
            
            stats[symbol] = {
                'update_processing': {
                    'avg_time': np.mean([s['processing_time'] for s in update_stats]) if update_stats else 0,
                    'count': len(update_stats)
                },
                'surface_construction': {
                    'avg_time': np.mean([s['processing_time'] for s in surface_stats]) if surface_stats else 0,
                    'count': len(surface_stats)
                },
                'last_update': max([s['timestamp'] for s in recent_stats]) if recent_stats else None
            }
        
        # Overall system stats
        stats['system'] = {
            'update_queue_size': self.update_queue.qsize(),
            'surface_queue_size': self.surface_queue.qsize(),
            'alert_queue_size': self.alert_queue.qsize(),
            'active_surfaces': len(self.surfaces),
            'processing_threads': len([t for t in self.processing_threads if t.is_alive()])
        }
        
        return stats
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_processing()