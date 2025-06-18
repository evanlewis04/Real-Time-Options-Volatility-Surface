# src/data/api_client.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import os
import logging

logger = logging.getLogger(__name__)

class OptionsDataClient:
    """
    Enhanced client using yfinance for REAL stock prices + realistic options data
    No API keys needed - just works!
    """
    
    def __init__(self, api_key: str = None, provider: str = 'yfinance'):
        self.provider = provider
        self.api_key = api_key  # Keep for backwards compatibility, but not used
        self.rate_limit_delay = 0.1  # Much more lenient than API limits
        
        # Test yfinance availability
        try:
            # Quick test
            test_ticker = yf.Ticker("AAPL")
            test_data = test_ticker.history(period="1d")
            if not test_data.empty:
                self.yfinance_available = True
                logger.info("✅ yfinance working - using REAL stock prices!")
            else:
                self.yfinance_available = False
                logger.warning("⚠️ yfinance installed but not responding")
        except ImportError:
            self.yfinance_available = False
            logger.error("❌ yfinance not installed. Run: pip install yfinance")
        except Exception as e:
            self.yfinance_available = False
            logger.warning(f"⚠️ yfinance test failed: {e}")
        
        # Fallback prices (updated December 2024)
        self.fallback_prices = {
            'AAPL': 196.50,   'MSFT': 416.50,   'GOOGL': 166.80,
            'NVDA': 138.50,   'TSLA': 325.00,   'SPY': 578.00,
            'QQQ': 478.00,    'META': 540.00,   'AMZN': 177.50,
            'JPM': 241.00,    'BAC': 42.50,     'GME': 20.50,
            'NFLX': 720.00,   'AMD': 132.00,    'UBER': 68.50,
            'DIS': 95.00,     'COIN': 245.00,   'PLTR': 62.00
        }
        
        # Price cache to avoid excessive API calls
        self.price_cache = {}
        self.cache_timestamps = {}
        self.cache_duration = 300  # 5 minutes cache
        
        logger.info(f"📊 OptionsDataClient initialized with yfinance")
    
    def get_current_stock_price(self, symbol: str) -> float:
        """Get REAL current stock price using yfinance"""
        
        # Check cache first
        now = datetime.now()
        if (symbol in self.price_cache and 
            symbol in self.cache_timestamps and
            (now - self.cache_timestamps[symbol]).total_seconds() < self.cache_duration):
            
            cached_price = self.price_cache[symbol]
            logger.debug(f"📋 Using cached price for {symbol}: ${cached_price:.2f}")
            return cached_price
        
        if not self.yfinance_available:
            fallback = self.fallback_prices.get(symbol, 100.0)
            logger.info(f"📊 yfinance unavailable, using fallback for {symbol}: ${fallback:.2f}")
            return fallback
        
        try:
            logger.debug(f"🌐 Fetching REAL price for {symbol} via yfinance...")
            ticker = yf.Ticker(symbol)
            
            # Method 1: Get from recent history (most reliable)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                real_price = float(hist['Close'].iloc[-1])
                if real_price > 0:
                    self._cache_price(symbol, real_price)
                    logger.info(f"✅ REAL price for {symbol}: ${real_price:.2f} (from history)")
                    return real_price
            
            # Method 2: Try daily history if minute data fails
            hist_daily = ticker.history(period="2d")
            if not hist_daily.empty:
                real_price = float(hist_daily['Close'].iloc[-1])
                if real_price > 0:
                    self._cache_price(symbol, real_price)
                    logger.info(f"✅ REAL price for {symbol}: ${real_price:.2f} (from daily)")
                    return real_price
            
            # Method 3: Try ticker info
            info = ticker.info
            for price_field in ['currentPrice', 'regularMarketPrice', 'previousClose']:
                if price_field in info and info[price_field] and info[price_field] > 0:
                    real_price = float(info[price_field])
                    self._cache_price(symbol, real_price)
                    logger.info(f"✅ REAL price for {symbol}: ${real_price:.2f} (from {price_field})")
                    return real_price
            
            logger.warning(f"⚠️ yfinance returned no valid price for {symbol}")
            
        except Exception as e:
            logger.warning(f"❌ yfinance failed for {symbol}: {e}")
        
        # Fallback to updated static price
        fallback = self.fallback_prices.get(symbol, 100.0)
        logger.info(f"📊 Using updated fallback for {symbol}: ${fallback:.2f}")
        return fallback
    
    def _cache_price(self, symbol: str, price: float):
        """Cache price to reduce API calls"""
        self.price_cache[symbol] = price
        self.cache_timestamps[symbol] = datetime.now()
    
    def fetch_options_chain(self, symbol: str) -> pd.DataFrame:
        """
        Fetch options chain - tries yfinance first, then creates realistic synthetic data
        """
        
        # Try to get real options data from yfinance
        if self.yfinance_available:
            try:
                logger.info(f"🔍 Trying to fetch REAL options data for {symbol}...")
                ticker = yf.Ticker(symbol)
                
                # Get available expiration dates
                exp_dates = ticker.options
                
                if exp_dates and len(exp_dates) > 0:
                    logger.info(f"✅ Found {len(exp_dates)} expiration dates for {symbol}")
                    
                    all_options = []
                    
                    # Get options for first few expirations (to avoid too much data)
                    for exp_date in exp_dates[:6]:  # Limit to first 6 expirations
                        try:
                            option_chain = ticker.option_chain(exp_date)
                            
                            # Process calls
                            calls = option_chain.calls.copy()
                            calls['type'] = 'call'
                            calls['expiration'] = exp_date
                            
                            # Process puts  
                            puts = option_chain.puts.copy()
                            puts['type'] = 'put'
                            puts['expiration'] = exp_date
                            
                            # Combine
                            exp_options = pd.concat([calls, puts], ignore_index=True)
                            all_options.append(exp_options)
                            
                            time.sleep(0.1)  # Small delay to be nice to yfinance
                            
                        except Exception as e:
                            logger.warning(f"Failed to get options for {symbol} exp {exp_date}: {e}")
                            continue
                    
                    if all_options:
                        df = pd.concat(all_options, ignore_index=True)
                        df = self._clean_yfinance_options(df, symbol)
                        
                        if not df.empty:
                            logger.info(f"✅ Got {len(df)} REAL options contracts for {symbol}")
                            return df
                
            except Exception as e:
                logger.warning(f"⚠️ Real options data failed for {symbol}: {e}")
        
        # Fallback to enhanced synthetic options data
        logger.info(f"📊 Creating enhanced synthetic options data for {symbol}")
        return self._create_enhanced_synthetic_options(symbol)
    
    def _clean_yfinance_options(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and standardize yfinance options data with proper IV calculation"""
        try:
            # Get current stock price
            stock_price = self.get_current_stock_price(symbol)
            
            # Rename columns to match our expected format
            column_mapping = {
                'contractSymbol': 'contractSymbol',
                'lastTradeDate': 'timestamp',
                'strike': 'strike',
                'lastPrice': 'last',
                'bid': 'bid',
                'ask': 'ask',
                'change': 'change',
                'percentChange': 'percentChange',
                'volume': 'volume',
                'openInterest': 'openInterest',
                'impliedVolatility': 'impliedVolatility'
            }
            
            # Rename columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Convert expiration to datetime
            df['expiration'] = pd.to_datetime(df['expiration'])
            
            # Calculate days to expiration
            today = datetime.now().date()
            df['daysToExpiration'] = df['expiration'].apply(lambda x: (x.date() - today).days)
            df['time_to_expiry'] = df['daysToExpiration'] / 365.0
            
            # Calculate moneyness
            df['moneyness'] = df['strike'] / stock_price
            
            # Filter out expired or very short-term options
            df = df[df['daysToExpiration'] > 0]
            
            # Ensure numeric columns
            numeric_cols = ['strike', 'last', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter out bad data
            df = df.dropna(subset=['strike', 'last'])
            df = df[df['strike'] > 0]
            df = df[df['last'] > 0]
            
            # FIXED: Validate and recalculate IV if necessary
            if 'impliedVolatility' in df.columns:
                # Check for unrealistic IV values from yfinance
                df['iv_valid'] = (df['impliedVolatility'] > 0.01) & (df['impliedVolatility'] < 5.0)
                
                # For invalid IVs, recalculate using our IV calculator
                invalid_iv_mask = ~df['iv_valid']
                
                if invalid_iv_mask.sum() > 0:
                    logger.info(f"Recalculating IV for {invalid_iv_mask.sum()} options with invalid yfinance IV")
                    
                    # Initialize IV calculator
                    try:
                        from src.pricing.implied_vol import ImpliedVolatilityCalculator
                        iv_calc = ImpliedVolatilityCalculator()
                        risk_free_rate = 0.05  # Use current risk-free rate
                        
                        for idx in df[invalid_iv_mask].index:
                            try:
                                row = df.loc[idx]
                                
                                # Calculate IV from market price
                                iv, method = iv_calc.calculate_implied_vol(
                                    market_price=row['last'],
                                    S=stock_price,
                                    K=row['strike'],
                                    T=row['time_to_expiry'],
                                    r=risk_free_rate,
                                    option_type=row['type'],
                                    method='newton'
                                )
                                
                                if iv is not None:
                                    df.loc[idx, 'impliedVolatility'] = iv
                                    df.loc[idx, 'iv_valid'] = True
                                    logger.debug(f"Recalculated IV for {row['contractSymbol']}: {iv:.2%}")
                                else:
                                    # Remove options where IV calculation failed
                                    df.loc[idx, 'iv_valid'] = False
                                    
                            except Exception as e:
                                logger.debug(f"Failed to calculate IV for option {idx}: {e}")
                                df.loc[idx, 'iv_valid'] = False
                    except ImportError:
                        # If IV calculator not available, just filter
                        logger.warning("IV calculator not available, using yfinance IVs as-is")
                
                # Filter to only valid IV options
                df = df[df['iv_valid']].drop('iv_valid', axis=1)
            
            else:
                # No IV column from yfinance - this is unusual, but handle it
                logger.warning(f"No IV data from yfinance for {symbol}")
                
                # Add a basic IV estimate based on option type and moneyness
                df['impliedVolatility'] = 0.25  # Default 25% IV
                
                # Adjust IV based on moneyness for realism
                for idx, row in df.iterrows():
                    base_iv = 0.25
                    moneyness = row['moneyness']
                    
                    # Simple skew: OTM puts have higher IV
                    if row['type'] == 'put' and moneyness > 1.0:  # OTM put
                        base_iv *= 1.2
                    elif row['type'] == 'call' and moneyness < 1.0:  # OTM call
                        base_iv *= 1.1
                    
                    df.loc[idx, 'impliedVolatility'] = base_iv
            
            # Calculate bid-ask spread
            if 'bid' in df.columns and 'ask' in df.columns:
                df['bidAskSpread'] = df['ask'] - df['bid']
                df['bidAskSpreadPct'] = df['bidAskSpread'] / ((df['bid'] + df['ask']) / 2)
                
                # Filter out options with unrealistic spreads
                df = df[
                    (df['bid'] > 0) & 
                    (df['ask'] > df['bid']) & 
                    (df['bidAskSpreadPct'] < 1.0)  # Less than 100% spread
                ]
            
            # Final validation
            df = df[
                (df['impliedVolatility'] > 0.01) & 
                (df['impliedVolatility'] < 5.0) &
                (df['time_to_expiry'] > 0.003)  # At least 1 day
            ]
            
            # Sort by expiration and strike
            df = df.sort_values(['expiration', 'strike'])
            
            logger.info(f"✅ Cleaned {len(df)} options for {symbol} with valid IVs")
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning yfinance options data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _create_enhanced_synthetic_options(self, symbol: str) -> pd.DataFrame:
        """
        Create enhanced synthetic options data based on REAL current stock price
        """
        
        # Get REAL current stock price
        stock_price = self.get_current_stock_price(symbol)
        
        logger.info(f"📊 Creating synthetic options for {symbol} at REAL price ${stock_price:.2f}")
        
        # Generate strikes around REAL current price
        strikes = []
        
        # Calculate appropriate strike spacing based on stock price
        if stock_price < 20:
            strike_spacing = 1  # $1 spacing for cheap stocks
        elif stock_price < 100:
            strike_spacing = 2.5  # $2.50 spacing
        elif stock_price < 200:
            strike_spacing = 5  # $5 spacing
        else:
            strike_spacing = 10  # $10 spacing for expensive stocks
        
        # Generate strikes from 70% to 130% of current price
        num_strikes = 25
        for i in range(-num_strikes//2, num_strikes//2 + 1):
            strike = stock_price + (i * strike_spacing)
            if strike > 0:
                strikes.append(strike)
        
        # Generate realistic expiration dates
        today = datetime.now()
        expirations = []
        
        # Weekly options (next 4 weeks)
        for i in range(1, 5):
            exp_date = today + timedelta(weeks=i)
            # Adjust to Friday
            days_ahead = 4 - exp_date.weekday()
            if days_ahead < 0:
                days_ahead += 7
            exp_date += timedelta(days=days_ahead)
            expirations.append(exp_date)
        
        # Monthly options (next 6 months)
        for i in range(1, 7):
            exp_date = today + timedelta(days=30*i)
            # Third Friday of the month
            exp_date = exp_date.replace(day=15)
            days_ahead = 4 - exp_date.weekday()
            if days_ahead < 0:
                days_ahead += 7
            exp_date += timedelta(days=days_ahead)
            expirations.append(exp_date)
        
        # Symbol-specific parameters
        symbol_params = {
            'AAPL': {'base_vol': 0.25, 'volume_mult': 800},
            'MSFT': {'base_vol': 0.22, 'volume_mult': 600},
            'GOOGL': {'base_vol': 0.28, 'volume_mult': 400},
            'NVDA': {'base_vol': 0.40, 'volume_mult': 1000},
            'TSLA': {'base_vol': 0.50, 'volume_mult': 1200},
            'SPY': {'base_vol': 0.15, 'volume_mult': 2000},
            'QQQ': {'base_vol': 0.20, 'volume_mult': 1500},
            'META': {'base_vol': 0.35, 'volume_mult': 700},
            'AMZN': {'base_vol': 0.30, 'volume_mult': 600}
        }
        
        params = symbol_params.get(symbol, {'base_vol': 0.25, 'volume_mult': 200})
        base_vol = params['base_vol']
        volume_mult = params['volume_mult']
        
        options_data = []
        
        for exp_date in expirations:
            days_to_exp = (exp_date - today).days
            time_to_exp = days_to_exp / 365.0
            
            for strike in strikes:
                moneyness = stock_price / strike
                
                # Calculate realistic implied volatility with smile/skew
                vol = base_vol + 0.05 * np.sqrt(time_to_exp)  # Term structure
                vol += 0.1 * (1 - moneyness)  # Volatility skew
                vol += 0.05 * (moneyness - 1)**2  # Volatility smile
                vol = max(0.05, min(vol, 3.0))  # Reasonable bounds
                
                # Simplified Black-Scholes for option pricing
                d1 = (np.log(stock_price/strike) + (0.05 + 0.5*vol**2)*time_to_exp) / (vol*np.sqrt(time_to_exp))
                d2 = d1 - vol*np.sqrt(time_to_exp)
                
                from scipy.stats import norm
                
                # Call price
                call_price = stock_price*norm.cdf(d1) - strike*np.exp(-0.05*time_to_exp)*norm.cdf(d2)
                call_price = max(0.01, call_price)
                
                # Put price (put-call parity)
                put_price = call_price - stock_price + strike*np.exp(-0.05*time_to_exp)
                put_price = max(0.01, put_price)
                
                # Volume based on moneyness and popularity
                base_volume = int(volume_mult * (1.2 - abs(1 - moneyness)) * np.exp(-time_to_exp))
                base_volume = max(1, base_volume)
                
                # Create call option
                call_bid = call_price * 0.995
                call_ask = call_price * 1.005
                
                options_data.append({
                    'symbol': symbol,
                    'contractSymbol': f"{symbol}{exp_date.strftime('%y%m%d')}C{int(strike*1000):08d}",
                    'strike': strike,
                    'expiration': exp_date,
                    'type': 'call',
                    'bid': round(call_bid, 2),
                    'ask': round(call_ask, 2),
                    'last': round(call_price, 2),
                    'volume': int(base_volume),
                    'openInterest': int(base_volume * 5),
                    'impliedVolatility': round(vol, 4),
                    'timestamp': today.strftime('%Y-%m-%d')
                })
                
                # Create put option
                put_bid = put_price * 0.995
                put_ask = put_price * 1.005
                
                options_data.append({
                    'symbol': symbol,
                    'contractSymbol': f"{symbol}{exp_date.strftime('%y%m%d')}P{int(strike*1000):08d}",
                    'strike': strike,
                    'expiration': exp_date,
                    'type': 'put',
                    'bid': round(put_bid, 2),
                    'ask': round(put_ask, 2),
                    'last': round(put_price, 2),
                    'volume': int(base_volume * 0.8),
                    'openInterest': int(base_volume * 4),
                    'impliedVolatility': round(vol, 4),
                    'timestamp': today.strftime('%Y-%m-%d')
                })
        
        df = pd.DataFrame(options_data)
        
        # Add calculated fields
        df['expiration'] = pd.to_datetime(df['expiration'])
        today_date = datetime.now().date()
        df['daysToExpiration'] = df['expiration'].apply(lambda x: (x.date() - today_date).days)
        df['bidAskSpread'] = df['ask'] - df['bid']
        df['bidAskSpreadPct'] = df['bidAskSpread'] / ((df['bid'] + df['ask']) / 2)
        
        logger.info(f"✅ Created {len(df)} synthetic options contracts for {symbol}")
        return df
    
    def save_data(self, df: pd.DataFrame, symbol: str, data_dir: str = 'data/raw'):
        """Save fetched data to file for caching"""
        os.makedirs(data_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_options_{timestamp}.csv"
        filepath = os.path.join(data_dir, filename)
        
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Data saved to {filepath}")
        
        return filepath


def test_yfinance_client():
    """Test the yfinance-based client"""
    print("🧪 Testing yfinance-based OptionsDataClient")
    print("=" * 50)
    
    client = OptionsDataClient()
    
    test_symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA']
    
    print("📊 Testing stock prices:")
    for symbol in test_symbols:
        price = client.get_current_stock_price(symbol)
        print(f"  {symbol}: ${price:8.2f}")
    
    print(f"\n📈 Testing options data for AAPL:")
    options = client.fetch_options_chain('AAPL')
    print(f"  Got {len(options)} options contracts")
    
    if not options.empty:
        print(f"  Strike range: ${options['strike'].min():.0f} - ${options['strike'].max():.0f}")
        print(f"  Expiration range: {options['daysToExpiration'].min()} - {options['daysToExpiration'].max()} days")


if __name__ == "__main__":
    test_yfinance_client()