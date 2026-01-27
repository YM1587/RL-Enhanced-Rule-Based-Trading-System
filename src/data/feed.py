import pandas as pd
import numpy as np
import logging

class PointInTimeValidator:
    """
    Ensures that data accessed at time t does not contain information from t+k.
    """
    @staticmethod
    def validate_row(full_df: pd.DataFrame, current_timestamp:  pd.Timestamp, lookback_window: pd.DataFrame) -> bool:
        # Check if the max timestamp in lookback window <= current_timestamp
        if not lookback_window.empty:
            max_ts = lookback_window.index.max()
            if max_ts > current_timestamp:
                raise ValueError(f"FUTURE LEAKAGE: Lookback max {max_ts} > Current {current_timestamp}")
        return True

class DataLoader:
    def __init__(self, csv_path: str):
        self.logger = logging.getLogger("DataLoader")
        self.df = self._load_data(csv_path)
        
    def _load_data(self, path: str) -> pd.DataFrame:
        # TODO: Real CSV loading
        # For now, generate synthetic data for testing
        dates = pd.date_range("2023-01-01", periods=1000, freq="1h")
        df = pd.DataFrame(index=dates)
        df['Close'] = 100 * (1 + np.random.randn(1000).cumsum() * 0.01)
        df['Open'] = df['Close'].shift(1).fillna(100)
        df['High'] = df[['Open', 'Close']].max(axis=1) * 1.01
        df['Low'] = df[['Open', 'Close']].min(axis=1) * 0.99
        df['Volume'] = np.random.randint(100, 1000, 1000)
        return df

    def get_window(self, current_time: pd.Timestamp, window_size: int) -> pd.DataFrame:
        """
        Returns the specific window of data ending at current_time.
        Strictly enforces point-in-time.
        """
        # Slice data strictly up to current_time (inclusive)
        # Assuming current_time corresponds to the OPEN of the next bar, 
        # we can only see data corresponding to times BEFORE current_time.
        # But if current_time is the 'time of decision', we usually have Closed candles up to T-1.
        
        # Scenario: We are at 10:00 AM. We want to decide for the 10:00-11:00 candle.
        # We have High,Low,Close for 09:00-10:00 (which is indexed at 09:00 or 10:00 depending on convention).
        # Let's assume index is Open Time. So at 10:00 AM, we have the row for 09:00 fully complete.
        
        mask = self.df.index < current_time
        available = self.df[mask]
        return available.iloc[-window_size:]
