"""
Multi-Asset Data Manager for Real-Time Volatility Surface System
Coordinates data fetching, processing, and caching across multiple assets
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
from collections import defaultdict

# Remove the problematic self-import - these should come from other files
# from src.data.api_client import OptionsAPIClient
# from src.data.data_processor import OptionsDataProcessor
# from src.data.market_data import MarketDataProvider
# from src.utils.helpers import get_risk_free_rate

# For now, let's create placeholder classes to avoid import errors


class OptionsAPIClient:
    """Placeholder API client"""
    def get_options_data(self, symbol):
        # Return sample data for testing
        return pd.DataFrame({
            'strike': [100, 105, 110],
            'last_price': [5.0, 3.0, 1.0],
            'volume': [100, 50, 25],
            'open_interest': [500, 300, 100],
            'type': ['call', 'call', 'call']
        })
    
    def get_underlying_price(self, symbol):
        return 102.50


class OptionsDataProcessor:
    """Placeholder data processor"""
    def process_options_data(self, raw_data, **kwargs):
        # Add required columns for testing
        processed = raw_data.copy()
        processed['time_to_expiry'] = 0.25  # 3 months
        processed['moneyness'] = processed['strike'] / 102.50
        return processed


class MarketDataProvider:
    """Placeholder market data provider"""
    pass


def get_risk_free_rate():
    """Placeholder function"""
    return 0.05  # 5% risk-free rate


@dataclass
class AssetConfig:
    """Configuration for individual assets"""
    symbol: str
    priority: str = "medium"  # low, medium, high, critical
    update_frequency: int = 60  # seconds
    min_volume: int = 10
    max_spread: float = 0.10
    min_open_interest: int = 5
    enabled: bool = True
    last_update: Optional[datetime] = None


@dataclass
class MarketUpdate:
    """Container for market data updates"""
    symbol: str
    timestamp: datetime
    options_data: pd.DataFrame
    underlying_price: float
    risk_free_rate: float
    success: bool = True
    error_message: Optional[str] = None


class DataManager:
    """
    Central coordinator for multi-asset options data management
    Handles real-time updates, caching, and coordination across assets
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.api_client = OptionsAPIClient()
        self.data_processor = OptionsDataProcessor()
        self.market_data = MarketDataProvider()
        
        # Asset configuration
        self.assets: Dict[str, AssetConfig] = {}
        self.data_cache: Dict[str, MarketUpdate] = {}
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.update_threads: Dict[str, threading.Thread] = {}
        self.stop_event = threading.Event()
        
        # Callbacks for real-time updates
        self.update_callbacks: List[Callable[[MarketUpdate], None]] = []
        
        # Performance tracking
        self.performance_stats = defaultdict(list)
        
        self._initialize_assets()
    
    def _initialize_assets(self):
        """Initialize default asset configurations"""
        default_assets = {
            'AAPL': AssetConfig('AAPL', 'high', 30, 50, 0.05, 20),
            'MSFT': AssetConfig('MSFT', 'high', 45, 30, 0.06, 15),
            'GOOGL': AssetConfig('GOOGL', 'high', 45, 20, 0.08, 10),
            'NVDA': AssetConfig('NVDA', 'high', 30, 40, 0.07, 15),
            'TSLA': AssetConfig('TSLA', 'high', 30, 60, 0.10, 25),
            'SPY': AssetConfig('SPY', 'critical', 15, 100, 0.02, 50),
            'QQQ': AssetConfig('QQQ', 'high', 30, 80, 0.03, 40),
            'JPM': AssetConfig('JPM', 'medium', 60, 20, 0.08, 10),
            'BAC': AssetConfig('BAC', 'medium', 60, 15, 0.09, 8),
            'GME': AssetConfig('GME', 'medium', 45, 30, 0.15, 10)
        }
        
        for symbol, config in default_assets.items():
            self.assets[symbol] = config
            self.logger.info(f"Initialized asset configuration for {symbol}")
    
    def add_asset(self, symbol: str, config: AssetConfig):
        """Add a new asset to monitor"""
        self.assets[symbol] = config
        self.logger.info(f"Added asset {symbol} with priority {config.priority}")
    
    def remove_asset(self, symbol: str):
        """Remove an asset from monitoring"""
        if symbol in self.assets:
            self.assets[symbol].enabled = False
            if symbol in self.update_threads:
                # Thread will stop on next iteration
                pass
        self.logger.info(f"Disabled monitoring for {symbol}")
    
    def register_update_callback(self, callback: Callable[[MarketUpdate], None]):
        """Register callback for real-time updates"""
        self.update_callbacks.append(callback)
    
    def fetch_single_asset(self, symbol: str) -> MarketUpdate:
        """Fetch data for a single asset"""
        start_time = time.time()
        
        try:
            config = self.assets[symbol]
            if not config.enabled:
                return MarketUpdate(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    options_data=pd.DataFrame(),
                    underlying_price=0.0,
                    risk_free_rate=0.0,
                    success=False,
                    error_message="Asset disabled"
                )
            
            # Fetch options data
            self.logger.debug(f"Fetching options data for {symbol}")
            raw_options = self.api_client.get_options_data(symbol)
            
            if raw_options.empty:
                raise ValueError(f"No options data received for {symbol}")
            
            # Process and filter data
            processed_options = self.data_processor.process_options_data(
                raw_options, 
                min_volume=config.min_volume,
                max_spread=config.max_spread,
                min_open_interest=config.min_open_interest
            )
            
            # Get underlying price
            underlying_price = self.api_client.get_underlying_price(symbol)
            
            # Get risk-free rate
            risk_free_rate = get_risk_free_rate()
            
            # Create update object
            update = MarketUpdate(
                symbol=symbol,
                timestamp=datetime.now(),
                options_data=processed_options,
                underlying_price=underlying_price,
                risk_free_rate=risk_free_rate,
                success=True
            )
            
            # Cache the update
            self.data_cache[symbol] = update
            config.last_update = update.timestamp
            
            # Track performance
            fetch_time = time.time() - start_time
            self.performance_stats[symbol].append({
                'timestamp': update.timestamp,
                'fetch_time': fetch_time,
                'contract_count': len(processed_options),
                'success': True
            })
            
            self.logger.info(
                f"Successfully fetched {len(processed_options)} contracts for {symbol} "
                f"(took {fetch_time:.2f}s)"
            )
            
            return update
            
        except Exception as e:
            error_msg = f"Error fetching data for {symbol}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Track failed performance
            fetch_time = time.time() - start_time
            self.performance_stats[symbol].append({
                'timestamp': datetime.now(),
                'fetch_time': fetch_time,
                'contract_count': 0,
                'success': False,
                'error': str(e)
            })
            
            return MarketUpdate(
                symbol=symbol,
                timestamp=datetime.now(),
                options_data=pd.DataFrame(),
                underlying_price=0.0,
                risk_free_rate=0.0,
                success=False,
                error_message=error_msg
            )
    
    def fetch_all_assets(self) -> Dict[str, MarketUpdate]:
        """Fetch data for all enabled assets in parallel"""
        enabled_symbols = [
            symbol for symbol, config in self.assets.items() 
            if config.enabled
        ]
        
        self.logger.info(f"Fetching data for {len(enabled_symbols)} assets in parallel")
        
        # Submit all fetch tasks
        future_to_symbol = {
            self.executor.submit(self.fetch_single_asset, symbol): symbol
            for symbol in enabled_symbols
        }
        
        results = {}
        
        # Collect results as they complete
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                update = future.result(timeout=30)  # 30 second timeout per asset
                results[symbol] = update
                
                # Notify callbacks
                for callback in self.update_callbacks:
                    try:
                        callback(update)
                    except Exception as e:
                        self.logger.error(f"Error in update callback: {e}")
                        
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {e}")
                results[symbol] = MarketUpdate(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    options_data=pd.DataFrame(),
                    underlying_price=0.0,
                    risk_free_rate=0.0,
                    success=False,
                    error_message=str(e)
                )
        
        self.logger.info(f"Completed parallel fetch for {len(results)} assets")
        return results
    
    def start_real_time_updates(self):
        """Start real-time updating for all assets"""
        self.logger.info("Starting real-time updates for all assets")
        
        for symbol, config in self.assets.items():
            if config.enabled:
                thread = threading.Thread(
                    target=self._asset_update_loop,
                    args=(symbol,),
                    daemon=True
                )
                thread.start()
                self.update_threads[symbol] = thread
                self.logger.info(f"Started update thread for {symbol}")
    
    def stop_real_time_updates(self):
        """Stop all real-time updates"""
        self.logger.info("Stopping real-time updates")
        self.stop_event.set()
        
        # Wait for threads to finish
        for symbol, thread in self.update_threads.items():
            thread.join(timeout=5)
            self.logger.info(f"Stopped update thread for {symbol}")
        
        self.update_threads.clear()
        self.stop_event.clear()
    
    def _asset_update_loop(self, symbol: str):
        """Continuous update loop for a single asset"""
        config = self.assets[symbol]
        
        while not self.stop_event.is_set() and config.enabled:
            try:
                # Check if it's time to update
                if (config.last_update is None or 
                    datetime.now() - config.last_update >= timedelta(seconds=config.update_frequency)):
                    
                    # Fetch new data
                    update = self.fetch_single_asset(symbol)
                    
                    # Notify callbacks
                    for callback in self.update_callbacks:
                        try:
                            callback(update)
                        except Exception as e:
                            self.logger.error(f"Error in update callback for {symbol}: {e}")
                
                # Sleep for a short period before checking again
                time.sleep(min(5, config.update_frequency // 6))
                
            except Exception as e:
                self.logger.error(f"Error in update loop for {symbol}: {e}")
                time.sleep(10)  # Wait longer after errors
    
    def get_cached_data(self, symbol: str) -> Optional[MarketUpdate]:
        """Get cached data for an asset"""
        return self.data_cache.get(symbol)
    
    def get_all_cached_data(self) -> Dict[str, MarketUpdate]:
        """Get all cached data"""
        return self.data_cache.copy()
    
    def get_performance_stats(self, symbol: Optional[str] = None) -> Dict:
        """Get performance statistics"""
        if symbol:
            stats = self.performance_stats.get(symbol, [])
            if not stats:
                return {}
                
            recent_stats = stats[-10:]  # Last 10 updates
            successful_stats = [s for s in recent_stats if s['success']]
            
            if not successful_stats:
                return {'symbol': symbol, 'status': 'no_successful_updates'}
            
            return {
                'symbol': symbol,
                'avg_fetch_time': np.mean([s['fetch_time'] for s in successful_stats]),
                'avg_contract_count': np.mean([s['contract_count'] for s in successful_stats]),
                'success_rate': len(successful_stats) / len(recent_stats),
                'last_update': max([s['timestamp'] for s in successful_stats]),
                'total_updates': len(stats)
            }
        else:
            # Return stats for all symbols
            return {
                symbol: self.get_performance_stats(symbol)
                for symbol in self.assets.keys()
            }
    
    def get_system_health(self) -> Dict:
        """Get overall system health metrics"""
        enabled_count = sum(1 for config in self.assets.values() if config.enabled)
        cached_count = len(self.data_cache)
        active_threads = len([t for t in self.update_threads.values() if t.is_alive()])
        
        # Calculate overall success rate
        all_stats = []
        for symbol_stats in self.performance_stats.values():
            all_stats.extend(symbol_stats[-5:])  # Last 5 updates per symbol
        
        if all_stats:
            success_rate = sum(1 for s in all_stats if s['success']) / len(all_stats)
            avg_fetch_time = np.mean([s['fetch_time'] for s in all_stats if s['success']])
        else:
            success_rate = 0
            avg_fetch_time = 0
        
        return {
            'enabled_assets': enabled_count,
            'cached_assets': cached_count,
            'active_threads': active_threads,
            'overall_success_rate': success_rate,
            'avg_fetch_time': avg_fetch_time,
            'uptime': datetime.now(),
            'status': 'healthy' if success_rate > 0.8 else 'degraded' if success_rate > 0.5 else 'unhealthy'
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_real_time_updates()
        self.executor.shutdown(wait=True)