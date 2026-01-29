import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds technical indicators to the DataFrame.
        Returns a new DataFrame with features.
        """
        df = df.copy()
        
        # 1. Log Returns
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Volatility (ATR-like approx using std dev of returns)
        df['volatility'] = df['log_ret'].rolling(window=14).std()
        
        # 3. EMA Distance & Slopes
        ema_50 = df['Close'].ewm(span=50, adjust=False).mean()
        df['ema_dist'] = (df['Close'] - ema_50) / ema_50
        df['ema_slope'] = ema_50.diff() / ema_50.shift(1) # Normalized slope
        
        # 4. Kaufman Efficiency Ratio (ER)
        # Net change over a period / Sum of absolute changes
        period = 10
        net_change = abs(df['Close'] - df['Close'].shift(period))
        sum_abs_changes = df['Close'].diff().abs().rolling(window=period).sum()
        df['efficiency_ratio'] = net_change / sum_abs_changes
        
        # 5. Volatility Regime (Rolling percentile of volatility)
        df['vol_percentile'] = df['volatility'].rolling(window=100).rank(pct=True)
        
        # 6. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Normalize/Fill
        df = df.dropna()
        return df

class FeatureStore:
    """
    Manages versioning of feature sets.
    """
    VERSION = "v2.0" # Research Grade
    
    def validate_features(self, df: pd.DataFrame):
        required = [
            'log_ret', 'volatility', 'ema_dist', 'rsi', 
            'ema_slope', 'efficiency_ratio', 'vol_percentile'
        ]
        if not all(col in df.columns for col in required):
            raise ValueError(f"FeatureStore {self.VERSION} missing columns")
