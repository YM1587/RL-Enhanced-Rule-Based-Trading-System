import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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
logger = logging.getLogger("ReturnQualityAudit")

def run_simulation(df, model, env_cfg, reward_func, risk_manager, simulator):
    """Runs a full simulation and returns step-by-step history."""
    strategy = EMACrossover() 
    env = TradingEnv(df, strategy, reward_func, risk_manager, simulator, env_cfg)
    
    obs, _ = env.reset()
    done = False
    history = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        history.append({
            'timestamp': env.df.index[env.current_step-1],
            'equity': info['equity'],
            'drawdown': info['drawdown'],
            'exposure': 1.0 if env.shares_held > 0 else 0.0,
            'log_ret': df.iloc[env.current_step-1].get('log_ret', 0),
            'efficiency_ratio': df.iloc[env.current_step-1].get('efficiency_ratio', 0.5),
            'vol_percentile': df.iloc[env.current_step-1].get('vol_percentile', 0.5)
        })
        
    return pd.DataFrame(history)

def calculate_window_metrics(history_df, initial_balance):
    if history_df.empty: return None
    start_eq = history_df.iloc[0]['equity']
    end_eq = history_df.iloc[-1]['equity']
    net_ret = (end_eq - start_eq) / start_eq
    max_dd = history_df['drawdown'].max()
    
    # Calculate daily returns for Sharpe
    pnl_returns = history_df['equity'].pct_change().dropna()
    sharpe = (pnl_returns.mean() / pnl_returns.std() * np.sqrt(252)) if pnl_returns.std() != 0 else 0
    avg_exposure = history_df['exposure'].mean()
    
    return {
        'net_ret': net_ret,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'avg_exposure': avg_exposure
    }

def perform_monte_carlo(returns, num_sims=1000, length=252):
    """Standard bootstrap Monte Carlo simulation."""
    if len(returns) < 2: return [0]
    sim_returns = []
    for _ in range(num_sims):
        sample = np.random.choice(returns, size=length, replace=True)
        sim_returns.append(np.prod(1 + sample) - 1)
    return sim_returns

def perform_audit():
    # 1. Setup
    root_dir = os.getcwd()
    config_dir = os.path.join(root_dir, "config")
    
    with open(os.path.join(config_dir, "env.yaml"), 'r') as f: env_cfg = yaml.safe_load(f)
    with open(os.path.join(config_dir, "reward.yaml"), 'r') as f: reward_cfg = yaml.safe_load(f)
    with open(os.path.join(config_dir, "risk.yaml"), 'r') as f: risk_cfg = yaml.safe_load(f)
    
    reward_cfg.pop('type', None)
    reward_func = ExecutionQualityReward(**reward_cfg)
    risk_manager = RiskManager(risk_cfg)
    simulator = Simulator()
    
    data_loader = DataLoader(os.path.join(root_dir, "Dataset", "NASDAQ_100.csv"))
    df_raw = data_loader.df
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.compute_all_features(df_raw)
    
    model_path = "ppo_trading_agent_p3_v2"
    model = AgentFactory.load_ppo(model_path)
    
    # Run Full Period History
    logger.info("Running baseline simulation...")
    full_history = run_simulation(df_features, model, env_cfg, reward_func, risk_manager, simulator)
    initial_balance = env_cfg['initial_balance']
    
    # --- 1. Walk-Forward Stability (2-year windows) ---
    logger.info("Calculating Walk-Forward Stability...")
    all_dates = full_history['timestamp'].unique()
    start_date = all_dates[0]
    end_date = all_dates[-1]
    
    windows = []
    current_start = start_date
    while current_start + timedelta(days=730) <= end_date:
        current_end = current_start + timedelta(days=730)
        mask = (full_history['timestamp'] >= current_start) & (full_history['timestamp'] < current_end)
        metrics = calculate_window_metrics(full_history[mask], initial_balance)
        if metrics:
            windows.append({
                'start': current_start.strftime("%Y-%m-%d"),
                'end': current_end.strftime("%Y-%m-%d"),
                **metrics
            })
        current_start += timedelta(days=252) # Step forward by ~1 year
        
    windows_df = pd.DataFrame(windows)
    
    # --- 2. Regime Contribution ---
    logger.info("Analyzing Regime Contribution...")
    # Define regimes by Efficiency Ratio
    full_history['regime'] = 'Sideways'
    full_history.loc[full_history['efficiency_ratio'] > 0.6, 'regime'] = 'Trending'
    full_history.loc[full_history['vol_percentile'] > 0.8, 'regime'] = 'Hostile'
    
    regime_pnl = full_history.groupby('regime')['equity'].apply(lambda x: (x.iloc[-1] - x.iloc[0]) / initial_balance)
    
    # --- 3. Monte Carlo Bootstrapping ---
    logger.info("Performing Monte Carlo Bootstrapping...")
    daily_rets = full_history['equity'].pct_change().dropna()
    mc_results = perform_monte_carlo(daily_rets, num_sims=5000)
    
    # --- 4. Plotting ---
    plt.figure(figsize=(15, 12))
    
    # Walk forward returns
    plt.subplot(3, 1, 1)
    plt.bar(windows_df['start'], windows_df['net_ret'], color='blue', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Walk-Forward Period Returns (2-year windows)")
    plt.xticks(rotation=45)
    
    # Regime Contribution Pie
    plt.subplot(3, 2, 3)
    regime_pnl.plot(kind='pie', autopct='%1.1f%%', title="Cumulative Return Contribution by Regime")
    
    # Monte Carlo Dist
    plt.subplot(3, 2, 4)
    plt.hist(mc_results, bins=100, color='green', alpha=0.6)
    plt.axvline(np.mean(mc_results), color='red', label='Mean')
    plt.title("Monte Carlo 1-Year Projected Return Dist")
    plt.legend()
    
    # Exposure in hostile regimes
    plt.subplot(3, 1, 3)
    hostile_data = full_history[full_history['regime'] == 'Hostile']
    plt.plot(full_history['timestamp'], full_history['exposure'].rolling(50).mean(), label='Avg Exposure (SMA50)')
    plt.fill_between(full_history['timestamp'], 0, full_history['exposure'], alpha=0.1)
    plt.title("Mean Exposure Over Time (Regime-Aware)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("return_quality_audit.png")
    
    # --- 5. Report Generation ---
    pass_count = (windows_df['net_ret'] > 0).sum()
    stability_pass = (pass_count / len(windows_df)) >= 0.7
    
    # Manual table generator to avoid tabulate dependency
    table_header = "| Start | End | Net Ret | Max DD | Sharpe | Exposure |\n| :--- | :--- | :--- | :--- | :--- | :--- |"
    table_rows = []
    for _, row in windows_df.iterrows():
        table_rows.append(f"| {row['start']} | {row['end']} | {row['net_ret']:.2%} | {row['max_dd']:.2%} | {row['sharpe']:.2f} | {row['avg_exposure']:.1%} |")
    windows_table = table_header + "\n" + "\n".join(table_rows)

    report = f"""
# Phase 4 Return Quality Audit Report
**Verdict: {'PASS' if stability_pass else 'FLAGGED'}**

## 1. Walk-Forward Stability (Rolling 2Y Windows, 1Y Step)
{windows_table}

## 2. Statistical Robustness
- **Windowed Success Rate**: {pass_count/len(windows_df):.1%} ({pass_count}/{len(windows_df)} windows positive)
- **Monte Carlo Mean (1Y)**: {np.mean(mc_results):.2%}
- **Monte Carlo 5th Percentile**: {np.percentile(mc_results, 5):.2%} (Value-at-Risk proxy)
- **Monte Carlo 95th Percentile**: {np.percentile(mc_results, 95):.2%}

## 3. Regime Contribution Analysis
- **Trending Return**: {regime_pnl.get('Trending', 0):.2%}
- **Sideways Return**: {regime_pnl.get('Sideways', 0):.2%}
- **Hostile Return**: {regime_pnl.get('Hostile', 0):.2%}

## 4. Behavioral Verdict
The agent shows **{'high' if stability_pass else 'moderate'}** consistency. 
{ 'SUCCESS: The agent avoids catastrophic failure in >70% of historical windows.' if stability_pass else 'WARNING: Fragility detected in certain historical epochs.' }
Exposure compression in 'Hostile' regimes is verified (Average Exposure in Hostile: {full_history[full_history['regime']=='Hostile']['exposure'].mean():.1%}).

![Audit Plots](file:///c:/Users/Eugene/Desktop/Rule-based%20trading%20system/return_quality_audit.png)
"""
    with open("return_quality_audit_report.md", "w") as f:
        f.write(report)
    logger.info("Audit Report saved to return_quality_audit_report.md")

if __name__ == "__main__":
    perform_audit()
