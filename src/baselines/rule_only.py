from typing import Dict
import pandas as pd
from src.strategy.base import BaseStrategy

class RuleBasedBaseline:
    """
    Runs the strategy without RL intervention.
    Used to establish the performance floor.
    """
    def __init__(self, strategy: BaseStrategy, risk_constraints: dict = None):
        self.strategy = strategy
        # TODO: Integrate RiskManager here too for fair comparison
        
    def run(self, df: pd.DataFrame) -> Dict:
        """
        Backtests the raw strategy on the dataframe.
        """
        signals = []
        equity_curve = [10000.0] # Init balance
        position = 0 # 0, 1, -1
        
        for i in range(50, len(df)):
            window = df.iloc[i-50:i]
            signal = self.strategy.generate_signal(window)
            
            # Simple simulation: Always 100% size on signal
            current_price = df.iloc[i]['Close']
            prev_price = df.iloc[i-1]['Close']
            
            # Calculate PnL from previous step
            if position != 0:
                ret = (current_price - prev_price) / prev_price
                if position == -1: ret = -ret
                equity_curve.append(equity_curve[-1] * (1 + ret))
            else:
                equity_curve.append(equity_curve[-1])
            
            # Update Position
            position = signal 
            
        final_equity = equity_curve[-1]
        return {
            "final_equity": final_equity,
            "total_return": (final_equity - 10000) / 10000,
            "equity_curve": equity_curve
        }
