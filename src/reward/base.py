from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseReward(ABC):
    @abstractmethod
    def calculate(self, 
                  account_pnl_pct: float, 
                  drawdown_pct: float, 
                  is_in_position: bool, 
                  trade_cost: float,
                  **kwargs) -> float:
        """
        Calculates the scalar reward for the current step.
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Resets internal state if any."""
        pass
