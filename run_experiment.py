import argparse
import logging
import yaml
import os
import pandas as pd
from typing import Dict

from src.data.feed import DataLoader
from src.data.features import FeatureEngineer, FeatureStore
from src.strategy.rules import EMACrossoverStrategy
from src.reward.risk_adjusted import RiskAdjustedReward
from src.execution.risk_manager import RiskManager
from src.execution.simulator import Simulator
from src.env.trading_env import TradingEnv
from src.agent.ppo_wrapper import AgentFactory
from src.baselines.rule_only import RuleBasedBaseline
from src.reward.safety_first import SafetyFirstReward # NEW

def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="RL-Enhanced Trading System Experiment Runner")
    parser.add_argument("--config_dir", type=str, default="config", help="Directory containing config yamls")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "evaluate", "baseline"], default="train", help="Execution mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("Main")
    
    # 1. Load Configurations
    env_cfg = load_config(os.path.join(args.config_dir, "env.yaml"))
    risk_cfg = load_config(os.path.join(args.config_dir, "risk.yaml"))
    reward_cfg = load_config(os.path.join(args.config_dir, "reward.yaml"))
    
    logger.info("Configurations loaded.")

    # 2. Data Pipeline
    logger.info("Initializing Data Pipeline...")
    loader = DataLoader("Dataset/NASDAQ_100.csv") # Uses Real Data
    engineer = FeatureEngineer()
    df_raw = loader.df
    df_features = engineer.compute_all_features(df_raw)
    
    # 3. Components
    strategy = EMACrossoverStrategy()
    risk_manager = RiskManager(risk_cfg)
    simulator = Simulator()
    
    if 'type' in reward_cfg:
        reward_type = reward_cfg.pop('type') # Remove 'type' so kwargs match constructor
        if reward_type == 'risk_adjusted':
            reward_func = RiskAdjustedReward(**reward_cfg)
        elif reward_type == 'safety_first':
            reward_func = SafetyFirstReward(**reward_cfg)
        else:
            reward_func = None # Or error
    else:
        reward_func = None
    
    # 4. Execution Mode
    if args.mode == "baseline":
        logger.info("Running Rule-Based Baseline...")
        baseline = RuleBasedBaseline(strategy)
        results = baseline.run(df_features)
        logger.info(f"Baseline Results: Total Return {results['total_return']:.2%}, Final Equity {results['final_equity']:.2f}")
        
    elif args.mode == "train":
        logger.info("Setting up RL Environment for Training...")
        env = TradingEnv(df_features, strategy, reward_func, risk_manager, simulator, env_cfg)
        
        logger.info("Initializing PPO Agent...")
        model = AgentFactory.create_ppo(env, os.path.join(args.config_dir, "ppo.yaml"))
        
        logger.info("Starting Training...")
        model.learn(total_timesteps=100000) 
        model.save("ppo_trading_agent")
        logger.info("Training Complete. Model saved.")
        
    elif args.mode in ["eval", "evaluate"]:
        model_path = "ppo_trading_agent"
        if not os.path.exists(model_path + ".zip"):
             logger.error("No trained model found. Run --mode train first.")
             return

        logger.info(f"Loading model from {model_path}...")
        model = AgentFactory.load_ppo(model_path) 
        
        # Define Evaluation Segments
        segments = [
            ("Full Period", df_features.index.min(), df_features.index.max()),
            ("Dotcom Crash", "2000-01-01", "2002-12-31"),
            ("GFC Crash", "2008-01-01", "2009-12-31"),
            ("Bull Run", "2013-01-01", "2014-12-31"),
            ("Chop/Sideways", "2011-01-01", "2012-12-31"),
            ("COVID Recovery", "2020-03-01", "2021-12-31")
        ]

        logger.info("Starting Segmented Evaluation...")
        print(f"\n{'Regime':<15} | {'Return':<10} | {'Max DD':<10}")
        print("-" * 45)

        for name, start, end in segments:
            # Slice data for the segment
            try:
                seg_df = df_features.loc[pd.to_datetime(start):pd.to_datetime(end)]
                if len(seg_df) < 100:
                    continue
                
                # Setup localized env for this segment
                seg_env = TradingEnv(seg_df, strategy, reward_func, risk_manager, simulator, env_cfg)
                
                obs, _ = seg_env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = seg_env.step(action)
                    done = terminated or truncated
                
                ret = ((info['equity'] - seg_env.initial_balance) / seg_env.initial_balance) * 100
                max_dd = info['drawdown'] * 100
                
                print(f"{name:<15} | {ret:>8.2f}% | {max_dd:>8.2f}%")
            except Exception as e:
                logger.warning(f"Could not run segment {name}: {e}")

        print("-" * 45)

if __name__ == "__main__":
    main()

