import time
import logging; pylogger = logging.getLogger(__name__)

import torch
import jax.random as jrand
import mujoco

from aware.env.mjx import _torch_to_jax

def overview_camera():
  """
  Fixed camera which keeps the whole crane and payload in view as the crane moves.
  """
  camera = mujoco.MjvCamera()
  camera.lookat[:] = [0, 0, 0.2]
  camera.distance = 7.0
  camera.azimuth = 120
  camera.elevation = -12
  return camera

def generate_dataset(env, agent, num_episodes, rerandomise_within_episode=None,
                     rerandomise_operator=False, render_first_episode=False):
  """
  Roll out the operator in the domain randomised crane simulation. Each episode is saved
  by the agent as one data file in agent.save_data_folder (unstable episodes are discarded).
  If rerandomise_operator, each episode draws a new operator type (e.g. random, sinusoid)
  and new operator parameters for every environment and actuator. Otherwise these are drawn
  once at the start, as for the paper dataset.
  If render_first_episode, the first environment of the first episode is also saved there
  as a video, to check the simulation and operator.
  """

  obs, infos = env.reset()

  pylogger.info(f"Generating {num_episodes} episodes of {agent.num_env_steps_per_update} "
                f"steps with {env.num_envs} environments, saving to {agent.save_data_folder}")

  for i_episode in range(num_episodes):

    t0 = time.time()

    if rerandomise_operator and i_episode > 0:
      agent.mixed_agent.reset()

    # record the states of the first environment for rendering
    if render_first_episode and i_episode == 0:
      env.render_start()

    for i in range(agent.num_env_steps_per_update):

      # get the next action from the operator, the agent also stores the observation
      actions = agent.get_action(obs, info=infos)
      obs, rewards, dones, truncs, infos = env.step(actions)

      # apply new domain randomisation parameters part-way through the episode
      if rerandomise_within_episode is not None:
        if i % rerandomise_within_episode == 0 and i != 0:
          # reset to the current state of the crane, but re-randomise
          qpos = infos["qpos_crane"]
          qvel = infos["qvel_crane"]
          env.key, new_key = jrand.split(env.key)
          key_batch = jrand.split(new_key, qpos.shape[0])
          obs, infos = env.reset(qpos=qpos, qvel=qvel, key_batch=key_batch,
                                 force_rerandomise=True)

      # reset any environments which reached a terminal state
      reset_mask = _torch_to_jax(torch.logical_or(dones, truncs))
      obs, infos = env.reset(reset_mask)

    # render the recorded states into a video, this also stops recording
    if env.enable_rendering:
      env.save_render(savedir=agent.save_data_folder, savename="episode_1_env_0",
                      add_timestamp=False, camera=overview_camera())

    # save the episode into the dataset
    agent.update()

    pylogger.info(f"Episode {i_episode + 1}/{num_episodes} finished in {time.time() - t0:.1f}s")
