"""
Generate a training dataset by driving the crane with a randomised operator in a domain
randomised MuJoCo MJX simulation.

  python scripts/generate_data.py num_episodes=10
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl") # headless rendering, set before importing mujoco

import hydra
from omegaconf import DictConfig

from aware.data_generation import generate_dataset

@hydra.main(config_path="../configs", config_name="generate", version_base=None)
def main(cfg: DictConfig):

  env = hydra.utils.instantiate(cfg.env)
  agent = hydra.utils.instantiate(cfg.model)

  # save the configs alongside the data
  agent.storage.save_configs(cfg)

  generate_dataset(env, agent, num_episodes=cfg.num_episodes,
                   rerandomise_within_episode=cfg.rerandomise_within_episode,
                   rerandomise_operator=cfg.rerandomise_operator,
                   render_first_episode=cfg.render_first_episode)

if __name__ == "__main__":
  main()
