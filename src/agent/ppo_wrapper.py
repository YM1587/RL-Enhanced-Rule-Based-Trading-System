from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import yaml

class AgentFactory:
    @staticmethod
    def create_ppo(env: gym.Env, config_path: str = "config/ppo.yaml") -> PPO:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        model = PPO(
            policy=cfg['policy'],
            env=env,
            learning_rate=cfg['learning_rate'],
            n_steps=cfg['n_steps'],
            batch_size=cfg['batch_size'],
            n_epochs=cfg['n_epochs'],
            gamma=cfg['gamma'],
            gae_lambda=cfg['gae_lambda'],
            clip_range=cfg['clip_range'],
            ent_coef=cfg['ent_coef'],
            verbose=cfg['verbose'],
            seed=cfg['seed']
        )
        return model
