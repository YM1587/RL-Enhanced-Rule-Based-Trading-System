import numpy as np
from .base import BaseReward

class RiskAdjustedReward(BaseReward):
    def __init__(self, lambda_risk: float = 0.5, gamma_cost: float = 0.5, drawdown_penalty: float = 2.0, hold_penalty: float = 0.0, benchmark_relative: bool = True):
        self.lambda_risk = lambda_risk
        self.gamma_cost = gamma_cost
        self.drawdown_penalty = drawdown_penalty
        self.hold_penalty = hold_penalty
        
    def calculate(self, account_pnl_pct: float, drawdown_pct: float, is_in_position: bool, trade_cost: float, benchmark_return: float = 0.0) -> float:
        # 1. Base Return (Relative to Benchmark if needed, currently absolute for simplicity in this step)
        r_t = account_pnl_pct 
        
        # 2. Risk Penalty (Volatility Proxy: roughly squared return if we don't have rolling window here)
        # In a full impl, we'd pass rolling std. Here we punish large instantaneous variance.
        risk_penalty = self.lambda_risk * (r_t ** 2)
        
        # 3. Cost Penalty
        cost_penalty = self.gamma_cost * trade_cost
        
        # 4. Drawdown Penalty (Soft Constraint)
        dd_penalty = 0.0
        if drawdown_pct > 0.10: # 10% soft limit
             dd_penalty = self.drawdown_penalty * drawdown_pct
             
        # 5. Holding cost (optional, tiny)
        hold_cost = self.hold_penalty if is_in_position else 0.0
        
        total_reward = r_t - risk_penalty - cost_penalty - dd_penalty - hold_cost
        return total_reward

    def reset(self):
        pass
