import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from src.data.feed import DataLoader
from src.data.features import FeatureEngineer
from src.strategy.rules import EMACrossoverStrategy as EMACrossover
from src.reward.execution_quality import ExecutionQualityReward
from src.execution.risk_manager import RiskManager
from src.execution.simulator import Simulator
from src.env.trading_env import TradingEnv
from src.agent.ppo_wrapper import AgentFactory

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Audit")

def run_simulation(df, model, env_cfg, reward_func, risk_manager, simulator):
    """Runs a full simulation and returns step-by-step history."""
    strategy = EMACrossover() # Re-init strategy for each run
    env = TradingEnv(df, strategy, reward_func, risk_manager, simulator, env_cfg)
    
    obs, _ = env.reset()
    done = False
    
    history = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Capture state
        history.append({
            'timestamp': env.df.index[env.current_step-1],
            'equity': info['equity'],
            'drawdown': info['drawdown'],
            'shares': env.shares_held,
            'exposure': 1.0 if env.shares_held > 0 else 0.0,
            'pnl': info.get('daily_pnl', 0), # Need to ensure info has this or compute it
            'reward': reward,
            'action': action,
            'signal': env.last_signal
        })
        
    return pd.DataFrame(history)

def perform_audit():
    # 1. Load Configurations
    root_dir = os.getcwd()
    config_dir = os.path.join(root_dir, "config")
    
    with open(os.path.join(config_dir, "env.yaml"), 'r') as f: env_cfg = yaml.safe_load(f)
    with open(os.path.join(config_dir, "reward.yaml"), 'r') as f: reward_cfg = yaml.safe_load(f)
    with open(os.path.join(config_dir, "risk.yaml"), 'r') as f: risk_cfg = yaml.safe_load(f)
    
    # 2. Components
    reward_type = reward_cfg.pop('type', 'execution_quality')
    reward_func = ExecutionQualityReward(**reward_cfg)
    risk_manager = RiskManager(risk_cfg)
    simulator = Simulator() # Default costs
    
    # 3. Data
    data_loader = DataLoader(os.path.join(root_dir, "Dataset", "NASDAQ_100.csv"))
    df_raw = data_loader.df
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.compute_all_features(df_raw)
    
    # 4. Model
    model_path = "ppo_trading_agent_p3_v2"
    if not os.path.exists(model_path + ".zip"):
        logger.error(f"Model {model_path} not found!")
        return
    
    logger.info(f"Loading model {model_path}...")
    model = AgentFactory.load_ppo(model_path)
    
    # --- AUDIT PASS 1: Baseline & Normal Performance ---
    logger.info("Running Audit Pass 1: Normal Simulation...")
    history_normal = run_simulation(df_features, model, env_cfg, reward_func, risk_manager, simulator)
    
    # --- AUDIT PASS 2: Stress Test (3x Costs) ---
    logger.info("Running Audit Pass 2: Stress Test (3x Costs)...")
    stressed_simulator = Simulator(trading_fee=0.003, slippage_std=0.0015) 
    history_stressed = run_simulation(df_features, model, env_cfg, reward_func, risk_manager, stressed_simulator)
    
    # --- AUDIT PASS 3: Look-Ahead / Leakage Check ---
    # We shift features FORWARD by 1, meaning we use "future" features to predict current price? 
    # No, that's not right. We want to see if the model BROKE. 
    # A better test: Shift target signals BACKWARD. If the model was leaking, it would fail to capture the shift.
    # Simpler leakage test: Re-run with features shifted by -1. If returns stay identical or high, it's a flag.
    logger.info("Running Audit Pass 3: Leakage Shift Test...")
    df_shifted = df_features.copy()
    # Shift technical indicators forward (introducing 1-period lag). If profit collapses, it's healthy.
    # If profit STAYS high, it means the model was somehow ignoring the lag or using future data.
    feature_cols = [c for c in df_features.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Price']]
    df_shifted[feature_cols] = df_shifted[feature_cols].shift(1)
    df_shifted = df_shifted.dropna()
    history_shifted = run_simulation(df_shifted, model, env_cfg, reward_func, risk_manager, simulator)
    
    # --- ANALYTICS ---
    
    # A. Equity & Exposure Visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(history_normal['timestamp'], history_normal['equity'], label='RL Normal')
    plt.plot(history_stressed['timestamp'], history_stressed['equity'], label='RL Stressed (3x Cost)', alpha=0.7)
    plt.title("Equity Curve Plausibility")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.fill_between(history_normal['timestamp'], 0, history_normal['exposure'], alpha=0.3, label='Exposure')
    plt.title("Exposure Over Time")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    pnl_returns = history_normal['equity'].pct_change().dropna()
    plt.hist(pnl_returns, bins=100, color='blue', alpha=0.7)
    plt.title("Trade PnL Distribution (Returns)")
    
    plt.tight_layout()
    plt.savefig("audit_plots.png")
    logger.info("Plots saved to audit_plots.png")
    
    # B. Regime Exclusion Check
    segments = [
        ("Full", "1999-01-01", "2024-01-01"),
        ("Dotcom", "2000-01-01", "2002-12-31"),
        ("GFC", "2007-10-01", "2009-03-31"),
        ("Bull", "2013-01-01", "2019-12-31"),
        ("Chop", "2010-01-01", "2012-12-31"),
        ("COVID", "2020-03-01", "2021-12-31")
    ]
    
    initial_balance = env_cfg.get('initial_balance', 10000.0)
    total_ret = (history_normal.iloc[-1]['equity'] - initial_balance) / initial_balance
    logger.info(f"Total Return: {total_ret:.2%}")
    
    regime_results = {}
    for name, start, end in segments:
        mask = (history_normal['timestamp'] >= pd.to_datetime(start)) & (history_normal['timestamp'] <= pd.to_datetime(end))
        seg_data = history_normal[mask]
        if len(seg_data) < 2: continue
        
        start_eq = seg_data.iloc[0]['equity']
        end_eq = seg_data.iloc[-1]['equity']
        ret = (end_eq - start_eq) / start_eq
        regime_results[name] = ret

    # Exclude best
    best_regime = max([k for k in regime_results.keys() if k != 'Full'], key=lambda k: regime_results[k])
    ret_ex_best = total_ret - regime_results[best_regime]
    logger.info(f"Return excluding {best_regime}: {ret_ex_best:.2%}")
    
    # C. Plausibility Metrics
    max_dd = history_normal['drawdown'].max()
    sharpe = (pnl_returns.mean() / pnl_returns.std() * np.sqrt(252)) if pnl_returns.std() != 0 else 0
    exposure_avg = history_normal['exposure'].mean()
    
    stressed_ret = (history_stressed.iloc[-1]['equity'] - initial_balance) / initial_balance
    shifted_ret = (history_shifted.iloc[-1]['equity'] - initial_balance) / initial_balance
    
    report = f"""
# Purity Audit Verdict: {'PASS' if max_dd < 0.2 and total_ret > 0.3 and shifted_ret < 0.1 else 'FLAGGED'}

## Core Metrics
- Total Net Return: {total_ret:.2%}
- Max Drawdown: {max_dd:.2%}
- Annualized Sharpe: {sharpe:.2f}
- Avg Exposure: {exposure_avg:.1%}

## Robustness Tests
- Best Regime ({best_regime}) Impact: {regime_results[best_regime]:.2%}
- Return Excluding Best: {ret_ex_best:.2%}
- Stressed (3x Cost) Return: {stressed_ret:.2%}
- Leakage Shift Test Return: {shifted_ret:.2%} (Lower is healthier)

## Distribution Check
- Return Volatility: {pnl_returns.std():.4f}
- Max Single-Step Loss: {pnl_returns.min():.4f}
- Max Single-Step Gain: {pnl_returns.max():.4f}

    """
    with open("audit_results.md", "w") as f:
        f.write(report)
    logger.info("Audit Report saved to audit_results.md")

if __name__ == "__main__":
    perform_audit()
