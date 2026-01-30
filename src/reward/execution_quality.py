import numpy as np
from src.reward.base import BaseReward

class ExecutionQualityReward(BaseReward):
    def __init__(self, 
                 lambda_risk: float = 2.0, 
                 abstain_bonus: float = 0.1,
                 miss_penalty: float = 0.1,    # Higher penalty to encourage entry when rule is right
                 turnover_lambda: float = 0.05, # Penalty for changing actions
                 holding_bonus: float = 0.01,  # Reward for holding winners in high ER
                 compression_penalty: float = 0.05 # Penalty for large size in high vol
                 ):
        self.lambda_risk = lambda_risk
        self.abstain_bonus = abstain_bonus
        self.miss_penalty = miss_penalty
        self.turnover_lambda = turnover_lambda
        self.holding_bonus = holding_bonus
        self.compression_penalty = compression_penalty
        
        self.in_trade = False
        self.trade_pnl = 0.0
        self.last_action = 0
        
    def calculate(self, 
                  account_pnl_pct: float, 
                  drawdown_pct: float, 
                  is_in_position: bool, 
                  trade_cost: float,
                  **kwargs) -> float:
        """
        Optimizes execution quality (Frequency, Holding, Compression).
        """
        reward = 0.0
        
        # Extract metadata
        strategy_signal = kwargs.get('strategy_signal', 0)
        current_action = kwargs.get('current_action', 0)
        efficiency_ratio = kwargs.get('efficiency_ratio', 0.5)
        vol_percentile = kwargs.get('vol_percentile', 0.5)
        
        # 1. Realized Trade Result (Discrete)
        if is_in_position:
            self.in_trade = True
            self.trade_pnl += account_pnl_pct
        else:
            if self.in_trade:
                # Trade Closed - Reward pure outcome
                reward += np.tanh(self.trade_pnl * 15.0) 
                self.in_trade = False
                self.trade_pnl = 0.0

        # 2. Strategy Alignment (Direction is Frozen)
        is_participating = current_action in [1, 2, 3] or (current_action == 4 and is_in_position)
        
        if strategy_signal > 0:
            if not is_participating:
                reward -= self.miss_penalty
        elif strategy_signal == 0:
            if is_participating:
                reward -= 0.1 # General "Over-trading" penalty
            else:
                reward += self.abstain_bonus

        # 3. Turnover Control (SUPPRESS CHURN)
        if current_action != self.last_action:
            reward -= self.turnover_lambda
        self.last_action = current_action

        # 4. Asymmetric Holding (POSITION MANAGEMENT)
        if is_in_position:
            # Reward holding winners in clean trends (High ER)
            if self.trade_pnl > 0.02 and efficiency_ratio > 0.6:
                reward += self.holding_bonus
            # Penalize holding loses in deteriorating trends
            elif self.trade_pnl < -0.01 and efficiency_ratio < 0.4:
                reward -= self.holding_bonus

        # 5. Exposure Compression (RISK FILTER)
        # Penalize Full Size (Action 3) when Volatility is in top quartile
        if current_action == 3 and vol_percentile > 0.75:
            reward -= self.compression_penalty

        # 6. Safety Anchors
        reward -= self.lambda_risk * max(0, drawdown_pct - 0.05)
        reward -= trade_cost * 5.0 # Cost is already a real penalty in balance, but here we bias against it
        
        return float(reward)

    def reset(self):
        self.in_trade = False
        self.trade_pnl = 0.0
        self.last_action = 0
