import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple

from src.execution.risk_manager import RiskManager, PortfolioState
from src.execution.simulator import Simulator
from src.strategy.base import BaseStrategy
from src.reward.base import BaseReward

class TradingEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 df: pd.DataFrame, 
                 strategy: BaseStrategy, 
                 reward_func: BaseReward,
                 risk_manager: RiskManager,
                 simulator: Simulator,
                 config: dict):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.strategy = strategy
        self.reward_func = reward_func
        self.risk_manager = risk_manager
        self.simulator = simulator
        self.window_size = config.get('window_size', 50)
        self.initial_balance = config.get('initial_balance', 10000.0)
        
        # Define Action Space: 0=FLAT, 1=LONG_SMALL, 2=LONG_MED, 3=LONG_FULL, 4=NO_OP
        self.action_space = spaces.Discrete(5)
        
        # Define Observation Space: 10 dimensions for regime context
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.max_equity = self.initial_balance
        self.reward_func.reset()
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Construct vector
        # 1. Market Data
        window = self.df.iloc[self.current_step - self.window_size : self.current_step]
        current_data = self.df.iloc[self.current_step]
        
        # Base Strategy Signal
        signal = self.strategy.generate_signal(window)
        
        # State Construction (10 features)
        obs = np.array([
            current_data.get('log_ret', 0),
            current_data.get('volatility', 0),
            current_data.get('ema_dist', 0),
            current_data.get('ema_slope', 0),        # NEW
            current_data.get('efficiency_ratio', 0), # NEW
            current_data.get('vol_percentile', 0),   # NEW
            current_data.get('rsi', 50) / 100.0,     # NEW (Normalized)
            signal,
            self.shares_held / (self.max_equity / current_data['Close']), # Normalized Pos
            (self._get_equity() - self.initial_balance) / self.initial_balance # Norm PnL
        ], dtype=np.float32)
        return np.nan_to_num(obs)
        
    def _get_equity(self):
        current_price = self.df.iloc[self.current_step]['Close']
        return self.balance + (self.shares_held * current_price)
        
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        current_vol = self.df.iloc[self.current_step].get('volatility', 0.0)
        prev_equity = self._get_equity()
        
        # 1. Map Action to Target Size
        # 0=Flat(0), 1=0.25, 2=0.5, 3=1.0, 4=NoOp
        target_pct = 0.0
        is_noop = False
        
        if action == 4:
            is_noop = True
        elif action == 0:
            target_pct = 0.0
        elif action == 1:
            target_pct = 0.25
        elif action == 2:
            target_pct = 0.50
        elif action == 3:
            target_pct = 1.0
            
        trade_cost = 0.0
        
        if not is_noop:
            # Calculate desired change
            target_equity_alloc = self._get_equity() * target_pct
            current_equity_alloc = self.shares_held * current_price
            delta_value = target_equity_alloc - current_equity_alloc
            
            # Risk Gatekeeper
            state = PortfolioState(
                equity=self._get_equity(),
                initial_equity=self.initial_balance,
                current_drawdown=(self.max_equity - self._get_equity()) / self.max_equity,
                daily_pnl=self._get_equity() - self.initial_balance, # Simple approx
                position_size=target_pct
            )
            
            is_valid, reason = self.risk_manager.validate_action(state, target_pct)
            
            if is_valid:
                # Execute Trade
                shares_delta = delta_value / current_price
                trade = self.simulator.execute_order(current_price, shares_delta, current_vol)
                
                self.balance -= (trade.size * trade.price) + trade.commission
                self.shares_held += trade.size
                trade_cost = trade.commission
            else:
                # Action Rejected - treat as NoOp or force close?
                # For now, treat as NoOp (keep current pos) OR force close if breached?
                # The RiskManager allows closing. If opening is rejected, we do nothing.
                pass
                
        # Time Step
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Calculate Reward
        current_equity = self._get_equity()
        self.max_equity = max(self.max_equity, current_equity)
        drawdown_pct = (self.max_equity - current_equity) / self.max_equity
        equity_return = np.log(current_equity / prev_equity) if prev_equity > 0 else 0.0
        
        reward = self.reward_func.calculate(
            account_pnl_pct=equity_return,
            drawdown_pct=drawdown_pct,
            is_in_position=(self.shares_held > 0),
            trade_cost=trade_cost
        )
        
        info = {
            'equity': current_equity,
            'drawdown': drawdown_pct,
            'action': action
        }
        
        return self._get_obs(), reward, terminated, truncated, info
