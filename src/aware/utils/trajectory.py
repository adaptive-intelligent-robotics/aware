import numpy as np
from copy import deepcopy
from datetime import datetime
import pickle


import logging; logging.basicConfig(
    level=logging.INFO); pylogging = logging.getLogger(__name__)





default_save_folder = "trajectories"

# ----- core utilities ----- #

def get_empty_trajectory(length, num_envs, name=None, num_joints=7, num_actions=3, motion_platform=False,
                         measurement_field_num=None, valid_data_field_num=None):
    """
    This function defines a trajectory. Call it to initialise an empty trajectory.
    """

    if name == None:
      name = f"trajectory_{length}ep_{num_envs}envs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    trajectory = {
        "name" : name,
        "time_elapsed" : np.zeros((num_envs, length), dtype=np.float32),
        # for the crane
        "joint_angles" : np.zeros((num_envs, length, num_joints), dtype=np.float32),
        "joint_velocities" : np.zeros((num_envs, length, num_joints), dtype=np.float32),
        "last_action" : np.zeros((num_envs, length, num_actions), dtype=np.float32),
        "last_executed_action": np.zeros((num_envs, length, num_actions), dtype=np.float32),
        "boom_tip_position" : np.zeros((num_envs, length, 3), dtype=np.float32),
        "payload_position" : np.zeros((num_envs, length, 3), dtype=np.float32),
        "payload_velocity" : np.zeros((num_envs, length, 3), dtype=np.float32),
        "payload_distance" : np.zeros((num_envs, length, 3), dtype=np.float32),
        # manipulation info
        "manipulation_bool":  np.zeros((num_envs, length), dtype=bool),
        "action_scale_factors": np.zeros((num_envs, length, 3), dtype=np.float32),
    }

    if measurement_field_num is not None:
      trajectory["measurement"] = np.zeros((num_envs, length, measurement_field_num), dtype=np.float32)

    if valid_data_field_num is not None:
      trajectory["valid_data"] = np.ones((num_envs, length, valid_data_field_num), dtype=np.int32)

    if motion_platform:
      num_joints_mp = 6 # hardcoded for now
      num_actions_mp = 6 # hardcoded for now
      trajectory = trajectory | {
        # for the motion platform
        "joint_angles_mp" : np.zeros((num_envs, length, num_joints_mp), dtype=np.float32),
        "joint_velocities_mp" : np.zeros((num_envs, length, num_joints_mp), dtype=np.float32),
        "last_action_mp" : np.zeros((num_envs, length, num_actions_mp), dtype=np.float32),
      }

    return trajectory

def load_trajectory(loadpath):
    """
    Load a trajectory from the given path.
    """

    pylogging.info(f"Preparing to load a trajectory at path: {loadpath}")

    with open(loadpath, 'rb') as f:
      trajectory = pickle.load(f)

    pylogging.info("Successfully loaded trajectory")

    return trajectory

def index_trajectory_batch(trajectory, indexes, axis=0):
  """
  Return only certain batches in a trajectory
  """
  traj = deepcopy(trajectory)
  for key in traj:
    if key == "name": continue
    if axis == 0:
      traj[key] = traj[key][indexes]
    elif axis == 1:
      traj[key] = traj[key][:, indexes]
    elif axis == 2:
      raise RuntimeError(f"axis=2 not supported by index_trajectory_batch")

  return traj

def data_to_trajectory(data, name="data_to_trajectory", i_qpos=None,
                       i_qvel=None, i_action=None, i_priv_info=None,
                       use_priv_info=True, use_action=True):
  """
  Convert data loaded for training into a trajectory
  """

  B, T, N = data.shape
  traj = get_empty_trajectory(num_envs=B, length=T)
  traj["name"] = name

  print(f"WARNING: data_to_trajectory() contains HARDCODED observation info, "
        f"requires obs = [qposx7, qvelx7, actionsx3, privx7]")

  if i_qpos is None:
    i_qpos = np.arange(0, 7)
    pylogging.warning(f"data_to_trajectory: using hardcoded i_qpos = {i_qpos}")
  if i_qvel is None:
    i_qvel = np.arange(7, 14)
    pylogging.warning(f"data_to_trajectory: using hardcoded i_qvel = {i_qvel}")
  if i_action is None and use_action:
    i_action = np.arange(14, 17)
    pylogging.warning(f"data_to_trajectory: using hardcoded i_action = {i_action}")
  if i_priv_info is None and use_priv_info:
    i_priv_info = np.arange(17, 24)
    pylogging.warning(f"data_to_trajectory: using hardcoded i_priv_info = {i_priv_info}")

  traj["joint_angles"] = data[:, :, i_qpos]
  traj["joint_velocities"] = data[:, :, i_qvel]
  traj["last_action"] = np.zeros((B,T,3)) if not use_action else data[:, :, i_action] 

  if use_priv_info:
    traj["privileged_info_normalised"] = data[:, :, i_priv_info]

  # action here is the action which causes the next state
  # we want the action which caused this state
  # so move all actions forward one timestep
  # state_1   state_2    state_3   =>  state_1    state_2    state_3
  # a->2      a->3       a->4      =>  ...        a->2       a->3
  # state_1 will be missing an action (it gets the final action which never resolved)
  traj["last_action"] = np.roll(traj["last_action"], shift=1, axis=1)

  return traj

def deep_dict_update(dictionary, overrides, strict=False):
  """
  Recursively update a dictionary. Modifies the original object.
  """
  for key, val in overrides.items():
    if isinstance(val, dict) and isinstance(dictionary.get(key), dict):
      deep_dict_update(dictionary[key], val)
    else:
      if strict and key not in dictionary:
        raise RuntimeError(f"deep_dict_update() error: strict=True and "
                            f"key={key} not present in dictionary")
      dictionary[key] = val
  return dictionary

# ----- functions for fixing real world trajectory data ----- #
