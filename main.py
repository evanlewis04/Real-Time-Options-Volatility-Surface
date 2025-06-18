#!/usr/bin/env python3
"""
Real-Time Options Volatility Surface System - Simple Starter
Works with your current system components, builds up gradually
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup logging with Windows-friendly encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('volatility_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import your config
try:
    import app_config as config
    logger.info("+ Config imported successfully")
    
    # Check for API key using your config structure
    api_key = config.ALPHA_VANTAGE_API_KEY  # Direct access to module variable
    logger.info(f"+ Alpha Vantage API key configured: {bool(api_key)}")
    
except ImportError as e:
    logger.error(f"- Could not import config: {e}")
    sys.exit(1)
except AttributeError as e:
    logger.error(f"- Config structure issue: {e}")
    sys.exit(1)

class SimpleVolatilitySystem:
    """
    Simple starter system that works with available components
    Builds up functionality as you add more components
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_running = False
        self.available_components = {}
        self.component_status = {}
        
        # Test what components are available
        self._discover_components()
    
    def _discover_components(self):
        """Discover what components are actually available"""
        self.logger.info("+ Discovering available components...")
        
        # Test core components
        components_to_test = [
            ('api_client', 'src.data.api_client', 'OptionsDataClient'),
            ('data_manager', 'src.data.data_manager', 'DataManager'),
            ('black_scholes', 'src.pricing.black_scholes', 'BlackScholesModel'),
            ('implied_vol', 'src.pricing.implied_vol', 'ImpliedVolatilityCalculator'),
            ('vol_surface', 'src.analysis.vol_surface', 'VolatilitySurface'),
            ('portfolio_analytics', 'src.portfolio.portfolio_analytics', 'PortfolioAnalytics'),
            ('stream_processor', 'src.realtime.stream_processor', 'StreamProcessor')
        ]
        
        for comp_name, module_path, class_name in components_to_test:
            try:
                module = __import__(module_path, fromlist=[class_name])
                component_class = getattr(module, class_name)
                self.available_components[comp_name] = component_class
                self.logger.info(f"  + {comp_name} ({class_name}) available")
            except ImportError as e:
                self.logger.info(f"  - {comp_name}: Module not found")
            except AttributeError as e:
                self.logger.info(f"  - {comp_name}: Class not found")
            except Exception as e:
                self.logger.info(f"  - {comp_name}: Other error")
        
        self.logger.info(f"Available components: {len(self.available_components)}/{len(components_to_test)}")
    
    def initialize_api_client(self):
        """Initialize API client if available"""
        if 'api_client' in self.available_components:
            try:
                OptionsDataClient = self.available_components['api_client']
                # Use your config structure - direct module variable access
                api_key = config.ALPHA_VANTAGE_API_KEY or 'demo'
                self.api_client = OptionsDataClient(api_key)
                self.component_status['api_client'] = 'initialized'
                self.logger.info("+ API Client initialized")
                return True
            except Exception as e:
                self.logger.error(f"- Failed to initialize API client: {e}")
                return False
        else:
            self.logger.warning("- API Client not available")
            return False
    
    def test_data_fetch(self, symbol='AAPL'):
        """Test fetching data for a single symbol"""
        if hasattr(self, 'api_client'):
            try:
                self.logger.info(f"Testing data fetch for {symbol}...")
                
                # Try to get current stock price
                price = self.api_client.get_current_stock_price(symbol)
                self.logger.info(f"+ Current {symbol} price: ${price:.2f}")
                
                # Try to fetch options data
                options_data = self.api_client.fetch_options_chain(symbol)
                self.logger.info(f"+ Fetched {len(options_data)} options contracts for {symbol}")
                
                return True, options_data
                
            except Exception as e:
                self.logger.error(f"- Data fetch failed: {e}")
                return False, None
        else:
            self.logger.warning("- No API client available for data fetch")
            return False, None
    
    def test_pricing_models(self):
        """Test pricing model components"""
        if 'black_scholes' in self.available_components:
            try:
                BlackScholesModel = self.available_components['black_scholes']
                
                # Test a simple call option price
                price = BlackScholesModel.call_price(
                    S=100,      # Stock price
                    K=105,      # Strike
                    T=0.25,     # 3 months
                    r=0.05,     # 5% risk-free rate
                    sigma=0.2   # 20% volatility
                )
                
                self.logger.info(f"+ Black-Scholes test: Call option price = ${price:.3f}")
                return True
                
            except Exception as e:
                self.logger.error(f"- Black-Scholes test failed: {e}")
                return False
        else:
            self.logger.warning("- Black-Scholes model not available")
            return False
    
    def create_simple_surface(self, options_data):
        """Create a simple volatility surface if components available"""
        if 'vol_surface' in self.available_components and options_data is not None:
            try:
                VolatilitySurface = self.available_components['vol_surface']
                
                # Create a volatility surface using your config
                surface = VolatilitySurface(
                    options_data=options_data,
                    spot_price=180.0,  # Approximate AAPL price
                    risk_free_rate=config.RISK_FREE_RATE  # Direct access
                )
                
                # Try to construct the surface
                surface_data = surface.construct_surface()
                self.logger.info("+ Volatility surface created successfully")
                
                return True, surface_data
                
            except Exception as e:
                self.logger.error(f"- Volatility surface creation failed: {e}")
                return False, None
        else:
            self.logger.warning("- Volatility surface components not available")
            return False, None
    
    def run_system_test(self):
        """Run a comprehensive system test"""
        self.logger.info("Running comprehensive system test...")
        self.logger.info("=" * 50)
        
        # Test 1: Initialize API client
        self.logger.info("Test 1: API Client Initialization")
        api_success = self.initialize_api_client()
        
        # Test 2: Test pricing models
        self.logger.info("\nTest 2: Pricing Models")
        pricing_success = self.test_pricing_models()
        
        # Test 3: Test data fetching
        self.logger.info("\nTest 3: Data Fetching")
        data_success, options_data = self.test_data_fetch()
        
        # Test 4: Test volatility surface
        self.logger.info("\nTest 4: Volatility Surface")
        surface_success, surface_data = self.create_simple_surface(options_data)
        
        # Summary
        self.logger.info("\n" + "=" * 50)
        self.logger.info("SYSTEM TEST SUMMARY")
        self.logger.info("=" * 50)
        
        tests = [
            ("API Client", api_success),
            ("Pricing Models", pricing_success), 
            ("Data Fetching", data_success),
            ("Volatility Surface", surface_success)
        ]
        
        passed_tests = sum(1 for _, success in tests if success)
        
        for test_name, success in tests:
            status = "PASS" if success else "FAIL"
            self.logger.info(f"  {test_name:20}: {status}")
        
        self.logger.info(f"\nOverall: {passed_tests}/{len(tests)} tests passed")
        
        if passed_tests >= 2:
            self.logger.info("System has core functionality working!")
            self.logger.info("Dashboard should work with real data")
        else:
            self.logger.info("System has limited functionality")
            self.logger.info("Dashboard will use mock data (still looks great!)")
        
        return passed_tests, len(tests)
    
    def start_basic_monitoring(self, symbols=['AAPL']):
        """Start basic monitoring loop"""
        # Initialize API client first
        if not hasattr(self, 'api_client'):
            self.logger.info("Initializing API client for monitoring...")
            api_success = self.initialize_api_client()
            if not api_success:
                self.logger.error("- Cannot start monitoring - API client initialization failed")
                return
        
        self.logger.info(f"+ Starting basic monitoring for {symbols}...")
        self.is_running = True
        
        try:
            while self.is_running:
                for symbol in symbols:
                    try:
                        success, data = self.test_data_fetch(symbol)
                        if success:
                            self.logger.info(f"+ {symbol}: Updated successfully ({len(data)} contracts)")
                        else:
                            self.logger.warning(f"- {symbol}: Update failed")
                    except Exception as e:
                        self.logger.error(f"- {symbol}: Error - {e}")
                
                self.logger.info("Waiting 30 seconds...")
                time.sleep(30)
                
        except KeyboardInterrupt:
            self.logger.info("+ Monitoring stopped by user")
        finally:
            self.is_running = False
    
    def interactive_mode(self):
        """Simple interactive mode"""
        print("\n" + "="*50)
        print("🚀 Simple Volatility System")
        print("="*50)
        print("Commands:")
        print("  test    - Run system test")
        print("  start   - Start basic monitoring")
        print("  status  - Show component status") 
        print("  quit    - Exit")
        print()
        
        while True:
            try:
                command = input("volatility> ").strip().lower()
                
                if command == 'test':
                    self.run_system_test()
                    
                elif command == 'start':
                    print("Starting monitoring... Press Ctrl+C to stop")
                    self.start_basic_monitoring()
                    
                elif command == 'status':
                    print(f"\n📊 Component Status:")
                    for comp, status in self.component_status.items():
                        print(f"  {comp}: {status}")
                    print(f"\n🔧 Available: {list(self.available_components.keys())}")
                    print()
                    
                elif command in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                    
                elif command == '':
                    continue
                    
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except EOFError:
                break

def main():
    """Main entry point"""
    system = SimpleVolatilitySystem()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            system.run_system_test()
        elif command == 'start':
            print("Starting basic monitoring... Press Ctrl+C to stop")
            system.start_basic_monitoring()
        else:
            print(f"Unknown command: {command}")
            print("Available: test, start")
    else:
        # Default to interactive mode
        system.interactive_mode()

if __name__ == "__main__":
    main()