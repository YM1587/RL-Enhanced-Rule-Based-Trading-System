import pandas as pd
from .base import BaseStrategy

class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, short_window: int = 10, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, market_data: pd.DataFrame) -> int:
        """
        Determines the signal for the LAST row in market_data.
        Assumes market_data has 'Close' column.
        """
        if len(market_data) < self.long_window:
            return 0
        
        # Calculate Indicators (can be optimized to not recalc entire series)
        short_ema = market_data['Close'].ewm(span=self.short_window, adjust=False).mean()
        long_ema = market_data['Close'].ewm(span=self.long_window, adjust=False).mean()
        
        last_short = short_ema.iloc[-1]
        last_long = long_ema.iloc[-1]
        
        if last_short > last_long:
            return 1
        elif last_short < last_long:
            return -1
        else:
            return 0
