import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.api_client import OptionsDataClient
from config import ALPHA_VANTAGE_API_KEY, DEFAULT_SYMBOL

def test_api_client():
    """Test the options data client with AAPL."""
    
    print("=== Testing Options Data API Client ===")
    
    # Initialize client
    # If you don't have an API key yet, it will use sample data
    api_key = ALPHA_VANTAGE_API_KEY or "demo_key"
    client = OptionsDataClient(api_key=api_key, provider='alpha_vantage')
    
    # Fetch options chain
    symbol = DEFAULT_SYMBOL
    print(f"\nFetching options chain for {symbol}...")
    
    options_df = client.fetch_options_chain(symbol)
    
    if not options_df.empty:
        print(f"\n✅ Successfully fetched {len(options_df)} options contracts")
        
        # Display basic statistics
        print(f"\nData Overview:")
        print(f"Unique expirations: {options_df['expiration'].nunique()}")
        print(f"Strike range: ${options_df['strike'].min():.0f} - ${options_df['strike'].max():.0f}")
        print(f"Calls: {len(options_df[options_df['type'] == 'call'])}")
        print(f"Puts: {len(options_df[options_df['type'] == 'put'])}")
        
        # Show sample data
        print(f"\nSample Options (first 10 contracts):")
        sample_cols = ['type', 'strike', 'expiration', 'bid', 'ask', 'impliedVolatility', 'daysToExpiration']
        print(options_df[sample_cols].head(10).to_string(index=False))
        
        # Get current stock price
        current_price = client.get_current_stock_price(symbol)
        print(f"\nCurrent {symbol} stock price: ${current_price:.2f}")
        
        # Save the data
        filepath = client.save_data(options_df, symbol)
        print(f"\nData saved to: {filepath}")
        
        print(f"\n✅ API Client test completed successfully!")
        return True
        
    else:
        print("❌ No options data received")
        return False

if __name__ == "__main__":
    test_api_client()