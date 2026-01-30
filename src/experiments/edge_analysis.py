import os
import yaml
import pandas as pd
import numpy as np
import logging
from typing import Dict, List

from src.data.feed import DataLoader
from src.data.features import FeatureEngineer
from src.strategy.rules import EMACrossoverStrategy
from src.execution.risk_manager import RiskManager
from src.execution.simulator import Simulator
from src.env.trading_env import TradingEnv

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_edge_analysis():
    logging.basicConfig(level=logging.ERROR)
    
    config_dir = "config"
    env_cfg = load_config(os.path.join(config_dir, "env.yaml"))
    risk_cfg = load_config(os.path.join(config_dir, "risk.yaml"))
    env_cfg['window_size'] = 250 
    risk_cfg['max_drawdown_limit'] = 1.0 # Disable kill-switch for edge discovery
    
    loader = DataLoader("Dataset/NASDAQ_100.csv")
    engineer = FeatureEngineer()
    df = engineer.compute_all_features(loader.df)
    
    strategy = EMACrossoverStrategy()
    risk_manager = RiskManager(risk_cfg)
    simulator = Simulator()
    
    # We use a dummy reward since we only care about equity curve for Sharpe calculation
    from src.reward.safety_first import SafetyFirstReward
    reward_func = SafetyFirstReward()
    
    # Execution Logic
    env = TradingEnv(df, strategy, reward_func, risk_manager, simulator, env_cfg)
    
    obs, _ = env.reset()
    done = False
    
    results_log = []
    
    print("Simulating Base Strategy...")
    # Get Market Returns for comparison
    df['mkt_log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    while not done:
        # Mock Agent: Perfect Follower (Action 3 = Always Buy on Signal)
        signal = obs[7]
        action = 3 if signal > 0 else 0
        
        # Capture current metadata before step
        current_data = env.df.iloc[env.current_step]
        er = current_data.get('efficiency_ratio', 0)
        vol = current_data.get('vol_percentile', 0)
        mkt_ret = current_data.get('mkt_log_ret', 0)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        results_log.append({
            'date': current_data.name,
            'efficiency_ratio': er,
            'vol_percentile': vol,
            'log_ret': np.log(info['equity'] / (results_log[-1]['equity'] if results_log else env.initial_balance)),
            'mkt_ret': mkt_ret,
            'equity': info['equity'],
            'signal': signal,
            'in_pos': env.shares_held > 0
        })

    results_df = pd.DataFrame(results_log)
    results_df.set_index('date', inplace=True)

    # 1. SEGMENTATION BY REGIME
    print("\n--- Conditional Edge Analysis Results ---")
    
    segments = {
        "Strong Trend (ER > 0.6)": results_df[results_df['efficiency_ratio'] > 0.6],
        "High Chop (ER < 0.3)": results_df[results_df['efficiency_ratio'] < 0.3],
        "High Vol (ATR% > 0.75)": results_df[results_df['vol_percentile'] > 0.75],
        "Low Vol (ATR% < 0.25)": results_df[results_df['vol_percentile'] < 0.25],
        "Standard (0.3 < ER < 0.6)": results_df[(results_df['efficiency_ratio'] >= 0.3) & (results_df['efficiency_ratio'] <= 0.6)]
    }

    print(f"{'Regime':<25} | {'Trades':<6} | {'Avg Ret':<8} | {'Mkt Ret':<8} | {'Sharpe':<6}")
    print("-" * 75)

    for name, data in segments.items():
        if len(data) < 10:
            continue
        
        # Count trades: defined as signal changes or position entries
        trades = (data['signal'].diff() != 0).sum()
        
        avg_ret = data['log_ret'].mean() * 252 
        mkt_ret = data['mkt_ret'].mean() * 252
        std_ret = data['log_ret'].std() * np.sqrt(252)
        sharpe = avg_ret / std_ret if std_ret > 0 else 0
        
        print(f"{name:<25} | {trades:<6} | {avg_ret:>7.2%} | {mkt_ret:>7.2%} | {sharpe:>6.2f}")

    # 2. ANALYSIS BY FIXED TIME REGIMES
    time_segments = [
        ("Dotcom Crash", "2000-01-01", "2002-12-31"),
        ("GFC Crash", "2008-01-01", "2009-12-31"),
        ("Bull Run", "2013-01-01", "2014-12-31"),
        ("COVID Recovery", "2020-03-01", "2021-12-31")
    ]

    print("\n--- Time-Segmented Baseline Performance ---")
    for name, start, end in time_segments:
        seg_data = results_df.loc[pd.to_datetime(start):pd.to_datetime(end)]
        if len(seg_data) < 10: continue
        
        total_ret = (seg_data['equity'].iloc[-1] / seg_data['equity'].iloc[0]) - 1
        avg_ret = seg_data['log_ret'].mean() * 252
        std_ret = seg_data['log_ret'].std() * np.sqrt(252)
        sharpe = avg_ret / std_ret if std_ret > 0 else 0
        
        print(f"{name:<15} | Tot Ret: {total_ret:>8.2%} | Sharpe: {sharpe:>8.2f}")

if __name__ == "__main__":
    run_edge_analysis()
