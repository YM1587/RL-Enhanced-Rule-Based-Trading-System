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

def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="RL-Enhanced Trading System Experiment Runner")
    parser.add_argument("--config_dir", type=str, default="config", help="Directory containing config yamls")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "baseline"], default="train", help="Execution mode")
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
    loader = DataLoader("dummy_path.csv") # Uses synthetic data for now
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
        model.learn(total_timesteps=10000) # Short run for verification
        model.save("ppo_trading_agent")
        logger.info("Training Complete. Model saved.")
        
    elif args.mode == "eval":
        logger.info("Evaluation Mode not yet fully implemented.")

if __name__ == "__main__":
    main()

