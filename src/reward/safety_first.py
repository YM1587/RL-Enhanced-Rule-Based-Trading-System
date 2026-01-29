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
                  strategy_signal: int = 0, # -1, 0, 1
                  current_action: int = 0    # 0=Flat, 1-3=Long, 4=NoOp
                  ) -> float:
        """
        Implements discrete realized trade rewards + bonuses for alignment.
        """
        reward = 0.0
        
        # 1. Realized PnL Reward (Volatility-normalized)
        if is_in_position:
            self.in_trade = True
            self.trade_pnl += account_pnl_pct
            reward -= self.hold_penalty 
        else:
            if self.in_trade:
                # Trade just closed!
                reward += np.tanh(self.trade_pnl * 10.0) 
                self.in_trade = False
                self.trade_pnl = 0.0
        
        # 2. Strategy Alignment Rewards
        # Binary: 1 = Participation (Action 1-3), 0 = Abstention (Action 0 or 4-with-no-pos)
        is_participating = current_action in [1, 2, 3] or (current_action == 4 and is_in_position)
        
        # CASE: Strategy wants to BUY
        if strategy_signal > 0:
            if not is_participating:
                # Agent ignored a BUY signal (Abstention)
                # If subsequent PnL is positive, penalize. If negative, reward.
                # However, since we don't have future PnL here, we penalize the "Missed Opportunity" 
                # as a baseline, and the Agent must prove it was right by avoiding a drawndown.
                reward -= self.miss_penalty
            else:
                # Agent followed signal. No extra bonus (PnL will provide it).
                pass
        
        # CASE: Strategy wants to be FLAT
        elif strategy_signal == 0:
            if is_participating:
                # Agent stayed in market when strategy said stay out (Over-trading risk)
                reward -= self.hold_penalty * 2.0 # Double penalty
            else:
                # Agent stayed out correctly
                reward += self.abstain_bonus

        # 3. Risk & Cost Penalties
        reward -= self.lambda_risk * max(0, drawdown_pct - 0.05)
        reward -= trade_cost * 10.0 
        
        return float(reward)

    def reset(self):
        self.in_trade = False
        self.trade_pnl = 0.0
        self.last_equity = None
