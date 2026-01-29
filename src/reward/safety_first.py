import numpy as np
from src.reward.base import BaseReward

class SafetyFirstReward(BaseReward):
    def __init__(self, 
                 lambda_risk: float = 2.0, 
                 abstain_bonus: float = 0.1,
                 miss_penalty: float = 0.05,
                 hold_penalty: float = 0.001):
        self.lambda_risk = lambda_risk
        self.abstain_bonus = abstain_bonus
        self.miss_penalty = miss_penalty
        self.hold_penalty = hold_penalty
        self.last_equity = None
        self.in_trade = False
        self.trade_pnl = 0.0
        
    def calculate(self, 
                  account_pnl_pct: float, 
                  drawdown_pct: float, 
                  is_in_position: bool, 
                  trade_cost: float,
                  benchmark_return: float = 0.0) -> float:
        """
        Implements discrete realized trade rewards + penalties.
        """
        reward = 0.0
        
        # 1. Realized PnL Reward (only when closing/holding)
        if is_in_position:
            # We are in a trade. Accumulate PnL but don't reward yet?
            # Or reward incrementally but weight by risk?
            # The design says "Applies rewards only at trade resolution".
            # To do that perfectly, we need to know when a trade closes.
            # In the current Env, we can detect transition from in_pos to not_in_pos.
            self.in_trade = True
            self.trade_pnl += account_pnl_pct
            reward -= self.hold_penalty # Penalty for staying in market
        else:
            if self.in_trade:
                # Trade just closed!
                # Volatility-normalized realized reward
                # (Simple version: use log return capped by tanh)
                reward += np.tanh(self.trade_pnl * 10.0) 
                self.in_trade = False
                self.trade_pnl = 0.0
        
        # 2. Drawdown Penalty (Continuous)
        reward -= self.lambda_risk * max(0, drawdown_pct - 0.05) # Only if DD > 5%
        
        # 3. Trade Cost Penalty
        reward -= trade_cost * 10.0 # High sensitivity to fees
        
        return float(reward)

    def reset(self):
        self.in_trade = False
        self.trade_pnl = 0.0
        self.last_equity = None
