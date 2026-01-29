import os
import yaml
import pandas as pd
import numpy as np
import logging
from src.data.feed import DataLoader
from src.data.features import FeatureEngineer
from src.strategy.rules import EMACrossoverStrategy
from src.reward.safety_first import SafetyFirstReward
from src.execution.risk_manager import RiskManager
from src.execution.simulator import Simulator
from src.env.trading_env import TradingEnv

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_sanity():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Sanity")
    
    config_dir = "config"
    env_cfg = load_config(os.path.join(config_dir, "env.yaml"))
    risk_cfg = load_config(os.path.join(config_dir, "risk.yaml"))
    reward_cfg = load_config(os.path.join(config_dir, "reward.yaml"))
    
    loader = DataLoader("Dataset/NASDAQ_100.csv")
    engineer = FeatureEngineer()
    df = engineer.compute_all_features(loader.df)
    
    strategy = EMACrossoverStrategy()
    risk_manager = RiskManager(risk_cfg)
    simulator = Simulator()
    
    if 'type' in reward_cfg: reward_cfg.pop('type')
    reward_func = SafetyFirstReward(**reward_cfg)
    
    # 1. FLAT-ONLY TEST
    print("\n[TEST 1] FLAT-ONLY AGENT")
    env = TradingEnv(df, strategy, reward_func, risk_manager, simulator, env_cfg)
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(0) # Action 0 = FLAT
        done = terminated or truncated
    print(f"Final Equity: {env._get_equity():.2f} (Expected: 10000.00 minus maybe tiny float drift)")
    
    # 2. HOLD-ONLY TEST (Action 3 = FULL LONG)
    print("\n[TEST 2] HOLD-ONLY AGENT")
    simulator.reset()
    env = TradingEnv(df, strategy, reward_func, risk_manager, simulator, env_cfg)
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(3) # Action 3 = FULL LONG
        done = terminated or truncated
    
    buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[250] - 1) # Approx math
    print(f"Final Equity: {env._get_equity():.2f}")
    print(f"Approx Market Return: {buy_hold_return:.2%}")

if __name__ == "__main__":
    run_sanity()
