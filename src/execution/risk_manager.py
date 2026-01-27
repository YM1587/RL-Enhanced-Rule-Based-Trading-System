from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

@dataclass
class RiskConfig:
    max_position_size: float
    max_drawdown_limit: float
    max_daily_loss: float
    allowed_sectors: list

@dataclass
class PortfolioState:
    equity: float
    initial_equity: float
    current_drawdown: float
    daily_pnl: float
    position_size: float  # -1.0 to 1.0

class RiskManager:
    """
    Deterministic Gatekeeper.
    Decides if an action is safe to execute based on PortfolioState and RiskConfig.
    """
    def __init__(self, config: dict):
        self.config = RiskConfig(**config)
        self.logger = logging.getLogger("RiskManager")
        
    def validate_action(self, state: PortfolioState, projected_size: float) -> Tuple[bool, str]:
        """
        Checks if the projected position size violates any risk constraints.
        Returns: (is_valid, reason)
        """
        # 1. Check Hard Drawdown Limit
        if state.current_drawdown > self.config.max_drawdown_limit:
            if abs(projected_size) > 0: # If we are in drawdown, only allow closing positions (size=0)
                 return False, f"MAX_DRAWDOWN_BREACH: {state.current_drawdown:.2%} > {self.config.max_drawdown_limit:.2%}"

        # 2. Check Daily Loss Limit
        if state.daily_pnl < -self.config.max_daily_loss * state.initial_equity:
             if abs(projected_size) > 0:
                 return False, f"DAILY_LOSS_BREACH: {state.daily_pnl:.2f}"

        # 3. Check Position Sizing Limit
        if abs(projected_size) > self.config.max_position_size:
            return False, f"SIZE_LIMIT_BREACH: {projected_size:.2f} > {self.config.max_position_size:.2f}"

        return True, "APPROVED"

    def check_kill_switch(self, feature_drift_score: float, threshold: float) -> bool:
        """
        Checks if the RL agent should be disabled due to feature drift.
        """
        if feature_drift_score > threshold:
            self.logger.critical(f"KILL_SWITCH: Drift {feature_drift_score:.4f} > {threshold}")
            return True
        return False
