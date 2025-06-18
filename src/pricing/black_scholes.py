import numpy as np
from scipy.stats import norm
from typing import Union, Tuple
import warnings

class BlackScholesModel:
    """
    Black-Scholes-Merton option pricing model implementation.
    
    Supports both European calls and puts with dividend yield.
    All calculations use continuous compounding.
    """
    
    def __init__(self):
        """Initialize the Black-Scholes model."""
        pass
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate d1 parameter for Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            q: Dividend yield (default 0)
            
        Returns:
            d1 value
        """
        if T <= 0:
            # Handle edge case of expiration
            return float('inf') if S > K else float('-inf')
        
        if sigma <= 0:
            # Handle zero volatility case
            return float('inf') if S > K else float('-inf')
        
        numerator = np.log(S / K) + (r - q + 0.5 * sigma**2) * T
        denominator = sigma * np.sqrt(T)
        
        return numerator / denominator
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate d2 parameter for Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price  
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            q: Dividend yield (default 0)
            
        Returns:
            d2 value
        """
        return BlackScholesModel.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate European call option price using Black-Scholes formula.
        
        Formula: C = S*e^(-q*T)*N(d1) - K*e^(-r*T)*N(d2)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            q: Dividend yield (default 0)
            
        Returns:
            Call option price
        """
        # Handle edge cases
        if T <= 0:
            return max(0, S - K)  # Intrinsic value at expiration
        
        if sigma <= 0:
            # Zero volatility - option worth intrinsic value discounted
            return max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
        
        # Calculate d1 and d2
        d1_val = BlackScholesModel.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholesModel.d2(S, K, T, r, sigma, q)
        
        # Black-Scholes call formula
        call_value = (S * np.exp(-q * T) * norm.cdf(d1_val) - 
                     K * np.exp(-r * T) * norm.cdf(d2_val))
        
        return max(0, call_value)  # Options can't have negative value
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate European put option price using Black-Scholes formula.
        
        Formula: P = K*e^(-r*T)*N(-d2) - S*e^(-q*T)*N(-d1)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            q: Dividend yield (default 0)
            
        Returns:
            Put option price
        """
        # Handle edge cases
        if T <= 0:
            return max(0, K - S)  # Intrinsic value at expiration
        
        if sigma <= 0:
            # Zero volatility - option worth intrinsic value discounted
            return max(0, K * np.exp(-r * T) - S * np.exp(-q * T))
        
        # Calculate d1 and d2
        d1_val = BlackScholesModel.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholesModel.d2(S, K, T, r, sigma, q)
        
        # Black-Scholes put formula
        put_value = (K * np.exp(-r * T) * norm.cdf(-d2_val) - 
                    S * np.exp(-q * T) * norm.cdf(-d1_val))
        
        return max(0, put_value)  # Options can't have negative value
    
    @staticmethod
    def option_price(S: float, K: float, T: float, r: float, sigma: float, 
                    option_type: str, q: float = 0.0) -> float:
        """
        Calculate option price for either call or put.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
            q: Dividend yield (default 0)
            
        Returns:
            Option price
        """
        option_type = option_type.lower()
        
        if option_type == 'call':
            return BlackScholesModel.call_price(S, K, T, r, sigma, q)
        elif option_type == 'put':
            return BlackScholesModel.put_price(S, K, T, r, sigma, q)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
    
    @staticmethod
    def put_call_parity_check(call_price: float, put_price: float, S: float, K: float, 
                             T: float, r: float, q: float = 0.0, tolerance: float = 0.01) -> bool:
        """
        Verify put-call parity: C - P = S*e^(-q*T) - K*e^(-r*T)
        
        Args:
            call_price: Call option price
            put_price: Put option price
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            q: Dividend yield
            tolerance: Acceptable difference (default 1 cent)
            
        Returns:
            True if parity holds within tolerance
        """
        left_side = call_price - put_price
        right_side = S * np.exp(-q * T) - K * np.exp(-r * T)
        
        difference = abs(left_side - right_side)
        return difference <= tolerance
    
    @staticmethod
    def validate_inputs(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> bool:
        """
        Validate input parameters for Black-Scholes calculation.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            q: Dividend yield
            
        Returns:
            True if inputs are valid
            
        Raises:
            ValueError: If inputs are invalid
        """
        if S <= 0:
            raise ValueError(f"Stock price must be positive, got {S}")
        
        if K <= 0:
            raise ValueError(f"Strike price must be positive, got {K}")
        
        if T < 0:
            raise ValueError(f"Time to expiration cannot be negative, got {T}")
        
        if sigma < 0:
            raise ValueError(f"Volatility cannot be negative, got {sigma}")
        
        if abs(r) > 1:
            warnings.warn(f"Risk-free rate seems unusual: {r:.2%}")
        
        if abs(q) > 1:
            warnings.warn(f"Dividend yield seems unusual: {q:.2%}")
        
        if sigma > 5:
            warnings.warn(f"Volatility seems very high: {sigma:.2%}")
        
        return True


class OptionGreeks:
    """
    Calculate option Greeks using Black-Scholes model.
    
    Greeks measure sensitivity of option prices to various parameters.
    """
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, 
             option_type: str, q: float = 0.0) -> float:
        """
        Calculate Delta: sensitivity to underlying price changes.
        
        Delta measures how much the option price changes for a $1 change in stock price.
        Call delta: N(d1) * e^(-q*T)
        Put delta: -N(-d1) * e^(-q*T)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
            q: Dividend yield
            
        Returns:
            Delta value (0 to 1 for calls, -1 to 0 for puts)
        """
        if T <= 0:
            # At expiration, delta is 1 for ITM calls, 0 for OTM calls
            if option_type.lower() == 'call':
                return 1.0 if S > K else 0.0
            else:  # put
                return -1.0 if S < K else 0.0
        
        d1_val = BlackScholesModel.d1(S, K, T, r, sigma, q)
        
        if option_type.lower() == 'call':
            return norm.cdf(d1_val) * np.exp(-q * T)
        elif option_type.lower() == 'put':
            return -norm.cdf(-d1_val) * np.exp(-q * T)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate Gamma: sensitivity of delta to underlying price changes.
        
        Gamma measures how much delta changes for a $1 change in stock price.
        Gamma is the same for calls and puts.
        
        Formula: Gamma = phi(d1) * e^(-q*T) / (S * sigma * sqrt(T))
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            q: Dividend yield
            
        Returns:
            Gamma value
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1_val = BlackScholesModel.d1(S, K, T, r, sigma, q)
        
        gamma = (norm.pdf(d1_val) * np.exp(-q * T)) / (S * sigma * np.sqrt(T))
        return gamma
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, 
             option_type: str, q: float = 0.0) -> float:
        """
        Calculate Theta: sensitivity to time decay.
        
        Theta measures how much the option price decreases as time passes.
        Usually expressed as dollars lost per day, so divide by 365.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
            q: Dividend yield
            
        Returns:
            Theta value (negative for long options)
        """
        if T <= 0:
            return 0.0
        
        d1_val = BlackScholesModel.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholesModel.d2(S, K, T, r, sigma, q)
        
        # Common terms
        term1 = -(S * norm.pdf(d1_val) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
        term2 = q * S * norm.cdf(d1_val) * np.exp(-q * T)
        term3 = r * K * np.exp(-r * T)
        
        if option_type.lower() == 'call':
            theta = term1 - term2 - term3 * norm.cdf(d2_val)
        elif option_type.lower() == 'put':
            theta = term1 + term2 + term3 * norm.cdf(-d2_val)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
        
        return theta / 365  # Convert to daily theta
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate Vega: sensitivity to volatility changes.
        
        Vega measures how much the option price changes for a 1% change in volatility.
        Vega is the same for calls and puts.
        
        Formula: Vega = S * phi(d1) * sqrt(T) * e^(-q*T)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            q: Dividend yield
            
        Returns:
            Vega value (divide by 100 for 1% vol change)
        """
        if T <= 0:
            return 0.0
        
        d1_val = BlackScholesModel.d1(S, K, T, r, sigma, q)
        
        vega = S * norm.pdf(d1_val) * np.sqrt(T) * np.exp(-q * T)
        return vega / 100  # Convert to 1% vol change
    
    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, 
           option_type: str, q: float = 0.0) -> float:
        """
        Calculate Rho: sensitivity to interest rate changes.
        
        Rho measures how much the option price changes for a 1% change in interest rates.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
            q: Dividend yield
            
        Returns:
            Rho value (divide by 100 for 1% rate change)
        """
        if T <= 0:
            return 0.0
        
        d2_val = BlackScholesModel.d2(S, K, T, r, sigma, q)
        
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2_val)
        elif option_type.lower() == 'put':
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2_val)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
        
        return rho / 100  # Convert to 1% rate change