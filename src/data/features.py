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
        
        # 3. EMA Distance
        ema_50 = df['Close'].ewm(span=50, adjust=False).mean()
        df['ema_dist'] = (df['Close'] - ema_50) / ema_50
        
        # 4. RSI
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
    VERSION = "v1.0"
    
    def validate_features(self, df: pd.DataFrame):
        required = ['log_ret', 'volatility', 'ema_dist', 'rsi']
        if not all(col in df.columns for col in required):
            raise ValueError(f"FeatureStore {self.VERSION} missing columns")
