from dataclasses import dataclass
import numpy as np

@dataclass
class Trade:
    timestamp: any
    symbol: str
    side: str # "BUY" or "SELL"
    size: float
    price: float
    commission: float

class Simulator:
    """
    Stateful simulator for order execution.
    Models slippage and transaction costs.
    """
    def __init__(self, trading_fee: float = 0.001, slippage_std: float = 0.0005):
        self.trading_fee = trading_fee
        self.slippage_std = slippage_std 
        self.trades = []

    def reset(self):
        self.trades = []

    def execute_order(self, price: float, size: float, volatility: float = 0.0) -> Trade:
        """
        Simulates filling an order. 
        Applies slippage based on volatility.
        """
        # Slippage Model: Price +/- (Vol * Noise) + Fixed_Impact
        # We assume we cross the spread, so price gets worse.
        
        slippage = max(0, np.random.normal(0, self.slippage_std)) # Slippage is usually against us
        if volatility > 0:
             slippage += volatility * 0.1 # Impact scales with vol
        
        executed_price = price * (1 + slippage) if size > 0 else price * (1 - slippage) # Buy High, Sell Low
        
        commission = abs(size * executed_price) * self.trading_fee
        
        trade = Trade(
            timestamp=None, # Injected by context
            symbol="BTC-USD",
            side="BUY" if size > 0 else "SELL",
            size=size,
            price=executed_price,
            commission=commission
        )
        self.trades.append(trade)
        return trade
