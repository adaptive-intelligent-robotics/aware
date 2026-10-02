from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from typing import NamedTuple
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jrand
from jax.lax import stop_gradient as jSG
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
import tqdm
import xml.etree.ElementTree as ET
import logging; logger = logging.getLogger(__name__)
from copy import deepcopy

from mujoco_playground._src import mjx_env
from flax.core.frozen_dict import freeze, unfreeze

# get the path to the current directory (that this file is in)
import pathlib
pathhere = pathlib.Path(__file__).parent.resolve()

USE_FREEZING = False

if not USE_FREEZING:
  def freeze(any): return any
  def unfreeze(any): return any

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(

      # define the model files to use
      xml_name = "crane.xml",
      xml_motion_platform = "crane_with_base.xml",

      # key simulation parameters
      ctrl_dt = 0.02,                     # timestep for control (i.e. policy Hz)
      sim_dt = 0.005,                     # timestep for underlying physics
      action_repeat = 1,                  # has no known effect currently
      vision = False,                     # add Madrona rendering and visual observations (NOT available)
      seed = 0,                           # random seed

      # RL specific settings
      episode_length = 1000,              # maximum number of control steps per episode
      distance_normalisation = 3.0,       # distance at which reward is capped and normalised to -1
      acceleration_normalisation = 0.5,   # acceleration at which reward is capped and normalised to -1
      relative_actions = True,            # actions refer to a change in velocity (True) or absolute command (False)
      clip_actions = True,                # network action values will be clipped to a given range
      action_clip_range = 1,              # if clip_actions=True, clip to [-x, +x] => clip(action, -x, +x)
      scale_actions = True,               # scale actions before inputting into mujoco => (action * ctrlmax)
      action_scale_ctrlrange_frac = 0.25, # scale by fraction of possible ctrlrange => (action * ctrlmax * frac)
      # new RL related ideas
      penalise_action_mag = False,        # add a penalty on the magnitude of actions
      action_penalty_scale = 0.2,         # multiplies with the action penalty (reduce it to avoid learning to stay still)
      action_penalty_threshold = 0.5,     # stop penalising actions if at this factor of the normalised value
      use_action_change_mag = True,       # action penalty uses instead magnitude in change of action (if enabled)
      action_mag_normalisation = 2.0,     # normalise action mag with this (with clip=1, largest change=2.0, -1 to +1)

      # task definition
      target_threshold = 0.1,             # distance to target to consider episode a success
      target_move_chance = 0.0,           # chance for the target to move to a new position
      threshold_termination = True,       # terminate after reaching desired steps within threshold
      num_required_within_threshold = 20, # number of steps required to be within the threshold for done=True
      reset_noise_scale = 1e-1,           # noise on joint position reset
      joystick_target = False,            # target moves based on joystick motions
      joystick_target_max_step = 0.1,     # maximum change of a joint angle if using joystick target

      # motion platform parameters
      use_motion_platform = False,        # use the motion plaform
      mp_all_off_chance = 0.0,            # chance every motion platform motor is off
      mp_each_off_chance = 0.0,           # chance each individual motion platform motor is off
      mp_max_pos_amplitude = 0.2,         # maximum amplitude in metres for motion plat position actuators
      mp_max_rot_amplitude = 0.2,         # maximum amplitude in radians for motion plat rotation actautors
      mp_min_cycle_time = 6,              # minimum cycle time for each motion plat actuator
      mp_max_cycle_time = 12,             # maximum cycle time for each motion plat actuator
      mp_max_pos_noise = 0.0,             # maximum noise added to motion plat position targets
      mp_max_rot_noise = 0.0,             # maximum noise added to motion plat rotation targets

      # environment related domain randomisation
      mass_max_scaling = 0.3,             # max random scaling (+-) applied to mass in bound, range [1-this, 1+this]
      com_max_move = 50e-3,               # max random distance (+-) to move centre of mass, each dim independent
      actuator_max_scaling = 0.3,         # max random scaling (+-) applied to actuator gain, range [1-this, 1+this]
      payload_min_mass = 0.1,             # minimum mass of the payload
      payload_max_mass = 1.0,             # maximum mass of the payload
      payload_min_inertial_length = 0.1,  # minimum side length when calculating payload inertia based on a cuboid
      payload_max_inertial_length = 0.4,  # maximum side length when calculating payload inertia based on a cuboid
      min_armature_scale = 1e-2,          # minimum 'rotor inertia' for motors
      max_armature_scale = 1e2,           # maximum 'rotor inertia' for motors
      min_gear_ratio = 0.5,
      max_gear_ratio = 2.0,
      min_ctrlrange_scale = 0.5,
      max_ctrlrange_scale = 2.0,

      # configuraitons for ROA
      use_roa = False,                    # turn on ROA, observation includes below items
      roa_n_proprioceptive = 17,          # number of proprioceptive values
      roa_n_privileged_info = 12,         # number of actual privileged datapoints in the environment vector
      roa_n_timesteps_history = 10,       # number of timesteps of history given to the adaptation module
      roa_action_in_obs = False,          # include last action in observation

      # configuraitons for Dreamer
      use_dreamer = False
  )

# normalise values to [-1, +1]
def norm_min_max(x, min, max):
  x = jnp.clip(x, min=min, max=max)
  return 2 * (x - min) / (max - min + 1e-6) - 1
def norm_scale(x, scale):
  return norm_min_max(x, 1 - scale, 1 + scale)

@jax.jit
def mjx_init(mjx_model, qpos, qvel):
  return mjx_env.init(mjx_model, qpos, qvel)

# motion platform motor parameter tuples
class MotionParams(NamedTuple):
  amp: jnp.ndarray
  freq: jnp.ndarray
  enabled: jnp.ndarray

class Motors(NamedTuple):
  pos_z: MotionParams
  pos_y: MotionParams
  pos_x: MotionParams
  rot_x: MotionParams
  rot_y: MotionParams
  rot_z: MotionParams

class MotionParamsNorm(NamedTuple):
  amp_norm: jnp.ndarray
  freq_norm: jnp.ndarray

class MotorsNorm(NamedTuple):
  pos_z: MotionParamsNorm
  pos_y: MotionParamsNorm
  pos_x: MotionParamsNorm
  rot_x: MotionParamsNorm
  rot_y: MotionParamsNorm
  rot_z: MotionParamsNorm

# @jax.jit
# def mjx_step(mjx_model, data, action, n_substeps):
#   return mjx_env.step(mjx_model, data, action, n_substeps)

class Crane(mjx_env.MjxEnv):

  def __init__(
      self,
      xml_folder_path: str = f"{pathhere}/mjcf/",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      debug_jit: bool = False,
    ):
    """
    TRUSTLINE crane environment.
    """
    # updates config, saves at self._config
    super().__init__(config, config_overrides)
    self.debug_jit = debug_jit

    if self._config.vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    if xml_folder_path[-1] != "/": xml_folder_path += "/"

    if self._config.use_motion_platform:
      xml_name = self._config.xml_motion_platform
    else:
      xml_name = self._config.xml_name

    # create the mujoco model on the GPU
    self._xml_path = xml_folder_path + xml_name
    logger.info(f"Getting model from: {self._xml_path}")
    logger.debug(f"Current path is: ", os.getcwd())
    self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
    self._mj_dummy_data = mujoco.MjData(self._mj_model) # only for forward kinematics
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model)
    self._post_init()

    logger.info(f"Crane environment successfully initialised.")

  def _post_init(self) -> None:
    """
    Configure environment specific options, and extract important information.
    """

    # --- HARDCODE KEY INFORMATION WHICH MUST MATCH THE XML FILE --- #
    crane_actuated_joint_names = ["slew-joint", "luff-joint", "hoist-slider"]
    crane_passive_joint_names = ["boom-tip-beta-hinge", "boom-tip-phi-hinge",
                                 "p-tip-eta-hinge", "p-tip-sigma-hinge"]
    crane_actuator_names = ["slew-velocity", "luff-velocity", "hoist-velocity"]
    crane_body_names = ["cab", "boom", "payload"]
    mp_actuated_joint_names = ["base_x_pos", "base_y_pos", "base_z_pos",
                               "base_x_rot", "base_y_rot", "base_z_rot"]
    mp_actuator_names = ["z_slide_joint", "z_slide_joint", "z_slide_joint",
                         "base_x_rot_joint", "base_y_rot_joint", "base_z_rot_joint"]

    # configure solver options
    self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    self._mj_model.opt.iterations = 6
    self._mj_model.opt.ls_iterations = 6

    # discard any motion platform joints/actuators if it isn't in use
    if not self._config.use_motion_platform:
      mp_actuated_joint_names = []
      mp_actuator_names = []

    # set the 'get observation' function
    if self._config["use_roa"]:
      self._get_obs = self._get_obs_roa_main
    elif self._config["use_dreamer"]:
      self._get_obs = self._get_obs_dreamer
    else:
      self._get_obs = self._get_obs_default

    # parse the xml file to extract information
    tree = ET.parse(self._xml_path)
    root = tree.getroot()

    # find specific elements
    actuator_elements = root.find("actuator")
    sensor_elements = root.find("sensor")

    if actuator_elements == None:
      logger.warning(
          f"Crane._post_init() warning: no <actuator> section found in XML file at path: {self._xml_path}")
    if sensor_elements == None:
      logger.warning(
          f"Crane._post_init() warning: no <sensor> section found in XML file at path: {self._xml_path}")

    self.actions_names = [actuator.attrib['name'] for actuator in actuator_elements]
    self.sensor_names = [sensor.attrib['name'] for sensor in sensor_elements]
    self.n_actions = len(self.actions_names)
    self.sensor_dim = self._mj_model.sensor_dim

    # extract important info about bodies
    self._key_bodies = ["cab", "boom", "payload"]
    self._body_dict = {}

    for name in self._key_bodies:

      id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
      if id == -1:
        raise ValueError(f"Body name '{name}' not found in model.")

      self._body_dict[name] = {
          "id": id,
          "mass": self.mj_model.body_mass[id].copy(),
          "pos": self.mj_model.body_pos[id].copy(),
          "ipos": self.mj_model.body_ipos[id].copy(),
          "inertia": self.mj_model.body_inertia[id].copy(),
      }

    # extract important info about actuators
    self._key_actuators = ["slew-velocity", "luff-velocity", "hoist-velocity"]
    self._actuator_dict = {}
    self._actuator_dict["act_ids"] = jnp.zeros((len(self._key_actuators)), dtype=int)
    for i, name in enumerate(self._key_actuators):

      id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
      if id == -1:
        raise ValueError(f"Actuator name '{name}' not found in model.")

      self._actuator_dict[name] = {
          "id": id,
          "gear": self.mj_model.actuator_gear[id][0].copy(),
          "bias": self.mj_model.actuator_biasprm[id][:3].copy(),
          "gain": self.mj_model.actuator_gainprm[id][0].copy(),
          "dynprm": self.mj_model.actuator_dynprm[id][0].copy(),
          "forcerange" : self.mj_model.actuator_forcerange[id].copy(),
          "ctrlrange" : self.mj_model.actuator_ctrlrange[id].copy(),
      }

      self._actuator_dict["act_ids"] = self._actuator_dict["act_ids"].at[i].set(id)

    # sort so indexes are not dependent on the ordering of joint names
    self._actuator_dict["act_ids"] = jnp.sort(self._actuator_dict["act_ids"])

    # extract important info about joints
    self._key_joints = ["slew-joint", "luff-joint", "hoist-slider"]
    self._joint_dict = {}
    self._joint_dict["minvals"] = jnp.zeros((len(self._key_joints)))
    self._joint_dict["maxvals"] = jnp.zeros((len(self._key_joints)))
    self._joint_dict["qpos_adr"] = jnp.zeros((len(self._key_joints)), dtype=int)
    self._joint_dict["dof_adr"] = jnp.zeros((len(self._key_joints)), dtype=int)

    for i, name in enumerate(self._key_joints):

      id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
      if id == -1:
        raise ValueError(f"Joint name '{name}' not found in model.")

      self._joint_dict[name] = {
          "id": id,
          "range": self.mj_model.jnt_range[id].copy(),
          "limited": self.mj_model.jnt_limited[id].copy(),
          "qpos_adr": self.mj_model.jnt_qposadr[id].copy().astype(int),
          "dofadr": self.mj_model.jnt_dofadr[id].copy().astype(int),
          "pos": self.mj_model.jnt_pos[id].copy(),
          "armature" : self.mj_model.dof_armature[id].copy(),
          "damping" : self.mj_model.dof_damping[id].copy(),
      }

      # save the joint min/max and corresponding qpos address into jax tensors
      self._joint_dict["minvals"] = self._joint_dict["minvals"].at[i].set(
          self.mj_model.jnt_range[id, 0])
      self._joint_dict["maxvals"] = self._joint_dict["maxvals"].at[i].set(
          self.mj_model.jnt_range[id, 1])
      self._joint_dict["qpos_adr"] = self._joint_dict["qpos_adr"].at[i].set(
          self.mj_model.jnt_qposadr[id])
      self._joint_dict["dof_adr"] = self._joint_dict["dof_adr"].at[i].set(
          self.mj_model.jnt_dofadr[id])

    # collect qpos addresses of all crane joints (not just actuated ones)
    self._crane_joints = self._key_joints + ["boom-tip-beta-hinge",
                                             "boom-tip-phi-hinge",
                                             "p-tip-eta-hinge",
                                             "p-tip-sigma-hinge"]
    self._joint_dict["qpos_adr_crane"] = jnp.zeros((len(self._crane_joints)), dtype=int)
    self._joint_dict["dof_adr_crane"] = jnp.zeros((len(self._crane_joints)), dtype=int)
    for i, name in enumerate(self._crane_joints):
      id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
      if id == -1:
        raise ValueError(f"Joint name '{name}' not found in model.")
      self._joint_dict["qpos_adr_crane"] = self._joint_dict["qpos_adr_crane"].at[i].set(
          self.mj_model.jnt_qposadr[id])
      self._joint_dict["dof_adr_crane"] = self._joint_dict["dof_adr_crane"].at[i].set(
          self.mj_model.jnt_dofadr[id])

    # sort so indexes are not dependent on the ordering of joint names
    self._joint_dict["qpos_adr_crane"] = jnp.sort(self._joint_dict["qpos_adr_crane"])
    self._joint_dict["dof_adr_crane"] = jnp.sort(self._joint_dict["dof_adr_crane"])

    # extract important info about mocap markers
    self._key_mocap = ["target"]
    self._mocap_dict = {}
    for name in self._key_mocap:

      id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
      if id == -1:
        raise ValueError(f"Mocap name '{name}' not found in model.")

      self._mocap_dict[name] = {
          "id": id,
          "mocap_id": self.mj_model.body_mocapid[id],
      }

    # extract important info about motion platform
    if self._config.use_motion_platform:

      self._key_actuators_mp = ["base_x_pos", "base_y_pos", "base_z_pos",
                                "base_x_rot", "base_y_rot", "base_z_rot"]
      self._actuator_dict_mp = {}
      self._actuator_dict_mp["act_ids"] = jnp.zeros(
          (len(self._key_actuators_mp)), dtype=int)
      for i, name in enumerate(self._key_actuators_mp):
        id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if id == -1: raise ValueError(f"Actuator name '{name}' not found in model.")
        self._actuator_dict_mp[name] = {
            "id": id,
            "gear": self.mj_model.actuator_gear[id][0].copy(),
            "bias": self.mj_model.actuator_biasprm[id][:3].copy(),
            "gain": self.mj_model.actuator_gainprm[id][0].copy(),
            "dynprm": self.mj_model.actuator_dynprm[id][0].copy(),
            "forcerange" : self.mj_model.actuator_forcerange[id].copy(),
            "ctrlrange" : self.mj_model.actuator_ctrlrange[id].copy(),
        }

        self._actuator_dict_mp["act_ids"] = self._actuator_dict_mp["act_ids"].at[i].set(id)

      # sort so indexes are not dependent on the ordering of actuator names
      self._actuator_dict_mp["act_ids"] = jnp.sort(self._actuator_dict_mp["act_ids"])

      # extract important info about joints
      self._key_joints_mp = ["z_slide_joint", "y_slide_joint", "x_slide_joint",
                             "base_x_rot_joint", "base_y_rot_joint", "base_z_rot_joint"]
      self._joint_dict_mp = {}
      self._joint_dict_mp["minvals"] = jnp.zeros((len(self._key_joints_mp)))
      self._joint_dict_mp["maxvals"] = jnp.zeros((len(self._key_joints_mp)))
      self._joint_dict_mp["qpos_adr"] = jnp.zeros((len(self._key_joints_mp)), dtype=int)
      self._joint_dict_mp["dof_adr"] = jnp.zeros((len(self._key_joints_mp)), dtype=int)

      for i, name in enumerate(self._key_joints_mp):

        id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if id == -1:
          raise ValueError(f"Joint name '{name}' not found in model.")

        self._joint_dict_mp[name] = {
            "id": id,
            "range": self.mj_model.jnt_range[id].copy(),
            "limited": self.mj_model.jnt_limited[id].copy(),
            "qpos_adr": self.mj_model.jnt_qposadr[id].copy().astype(int),
            "dofadr": self.mj_model.jnt_dofadr[id].copy().astype(int),
            "pos": self.mj_model.jnt_pos[id].copy(),
        }

        # save the joint min/max and corresponding qpos address into jax tensors
        self._joint_dict_mp["minvals"] = self._joint_dict_mp["minvals"].at[i].set(
            self.mj_model.jnt_range[id, 0])
        self._joint_dict_mp["maxvals"] = self._joint_dict_mp["maxvals"].at[i].set(
            self.mj_model.jnt_range[id, 1])
        self._joint_dict_mp["qpos_adr"] = self._joint_dict_mp["qpos_adr"].at[i].set(
            self.mj_model.jnt_qposadr[id])
        self._joint_dict_mp["dof_adr"] = self._joint_dict_mp["dof_adr"].at[i].set(
            self.mj_model.jnt_dofadr[id])

      # sort so indexes are not dependent on the ordering of joint names
      self._joint_dict_mp["qpos_adr"] = jnp.sort(self._joint_dict_mp["qpos_adr"])
      self._joint_dict_mp["dof_adr"] = jnp.sort(self._joint_dict_mp["dof_adr"])

      # print("MP Actuator information:\n", self._actuator_dict_mp)
      # print("--- joint qpos ---")
      # print("Crane actuated joints qposadr:", self._joint_dict["qpos_adr"])
      # print("Crane all joints qposadr:", self._joint_dict["qpos_adr_crane"])
      # print("Motion platform joints qposadr:", self._joint_dict_mp["qpos_adr"])
      # print("--- joint dof ---")
      # print("Crane actuated joints dofadr:", self._joint_dict["dof_adr"])
      # print("Crane all joints dofadr:", self._joint_dict["dof_adr_crane"])
      # print("Motion platform joints dofadr:", self._joint_dict_mp["dof_adr"])
      # print("--- actuator ids ---")
      # print("Crane actuator ids:", self._actuator_dict["act_ids"])
      # print("Motion platform actuator ids:", self._actuator_dict_mp["act_ids"])

    # for debugging, print additional information
    # print("Body information:\n", self._body_dict)
    # print("Body mass information: Cab -> ", self._body_dict["cab"])
    # print("Body mass information: Boom -> ", self._body_dict["boom"])
    # print("Body mass information: Payload -> ", self._body_dict["payload"])
    # print("Body information:\n", self._body_dict)
    # print("Actuator information:\n", self._actuator_dict)
    # print("Joint information:\n", self._joint_dict)
    # print("Mocap information:\n", self._mocap_dict)

    # create empty dictionary for randomisation info
    self._extra_randomisation_info = {
        "payload_mass" : jnp.zeros(()),
        "payload_inertial_length" : jnp.zeros(()),
        "cab_mass_scale" : jnp.zeros(()),
        "cab_com_move" : jnp.zeros(()),
        "boom_mass_scale" : jnp.zeros(()),
        "boom_com_move" : jnp.zeros(()),
        "slew_act_scale" : jnp.zeros(()),
        "luff_act_scale" : jnp.zeros(()),
        "hoist_act_scale" : jnp.zeros(()),
        "slew_gear_ratio" : jnp.zeros(()),
        "luff_gear_ratio" : jnp.zeros(()),
        "hoist_gear_ratio" : jnp.zeros(()),
        "slew_armature_scale" : jnp.zeros(()),
        "luff_armature_scale" : jnp.zeros(()),
        "hoist_armature_scale" : jnp.zeros(()),
        "slew_ctrl_scale" : jnp.zeros(()),
        "luff_ctrl_scale" : jnp.zeros(()),
        "hoist_ctrl_scale" : jnp.zeros(()),
    }

    # self._lowers = self._mj_model.jnt_range[3:, 0]
    # self._uppers = self._mj_model.jnt_range[3:, 1]

    # create the spec object for dynamic runtime adjustments
    self.mj_spec = mujoco.MjSpec.from_file(self._xml_path)

  # ----- main public functions ----- #

  def domain_randomise(self, rng=None, num_envs=None, structured=False,
                       return_extra=True, structured_repeats=None,
                       cfg_override_groups=None):
    """
    Returns a vector of environments with parameters varying along pre-determined
    axes. For randomisation of these parameters, pass in rng, a batch of keys, one
    per vectorised env. To linearaly interpolate these parameters between their
    minimum and maximum, set structured=True and pass in the the integer num_envs,
    which is the number of environments desired in the resultant vectorisation.
    """

    def lin(ratio, min, max, shape=()):
      return ((ratio * (max - min)) + min) * jnp.ones(shape), ratio

    def rand(rng, min, max, shape=()):
      new_rng, rng = jrand.split(rng)
      return jrand.uniform(rng, minval=min, maxval=max, shape=shape), new_rng

    # will we randomise with a structured or fully random approach
    if structured:
      fcn = lin
    else:
      fcn = rand
      param = rng # full batch of rng keys
      num_envs = rng.shape[0]

    num = num_envs

    # handle splitting env batch into different config groups
    if cfg_override_groups == None:
      cfg_override_groups = [{}]
    groups = len(cfg_override_groups)
    group_inds = []
    for g in range(groups):
      group_inds.append(g * (num_envs // groups))
    group_inds = np.array(group_inds)

    if groups > 1:
      num = num // groups
    if structured_repeats != None and structured_repeats > 1:
      num = num // structured_repeats
    else:
      num = num

    if structured:
      if num_envs == 1:
        param = jnp.array([0.5])
      elif num == 1:
        param = jnp.array([0])
      elif num == 2:
        param = jnp.array([0, 1])
      else:
        # normal linear interpolation
        param = jnp.arange(num)
        param = jnp.divide(param, num - 1)

      if structured_repeats:
        param = jnp.repeat(param, structured_repeats)
        if groups <= 1:
          to_pad = num_envs - len(param)
        else:
          to_pad = (num_envs // groups) - len(param)
        param = jnp.concat((param, jnp.ones(to_pad)))
      if groups > 1:
        param = jnp.tile(param, groups)
        to_pad = num_envs - len(param)
        param = jnp.concat((param, jnp.ones(to_pad)))
    
    original_configs = deepcopy(self._config)

    # manually vmap, to avoid JAX tracer issues with mj_spec needing numpy arrays
    # all done with numpy, to avoid copies from immutable JAX arrays when not compiled
    vmapped_dims = ([], [], [])
    for i in range(len(param)):
      if len(cfg_override_groups):
        group_swap = np.where(group_inds == i)[0]
        if len(group_swap) == 1:
          # apply the corresponding configs
          self._config = deepcopy(original_configs)
          for cfg, val in cfg_override_groups[int(group_swap[0])].items():
            if cfg not in self._config:
              print(f"Crane.domain_randomise(): cfg override = {cfg} "
                    f"not recognised. Doing nothing.")
            else:
              print(f"Crane.domain_randomise(): New group at i={i}, "
                    f"overriding cfg={cfg} from {self._config[cfg]} to {val}")
              self._config[cfg] = val
      out = self._randomise_parameters(param[i], fcn)
      # loop over direct/indirect/details
      for j, vectors in enumerate(out):
        # loop over every vector within
        for k in range(len(vectors)):
          if i == 0:
            # expand dimension, and pre-assign zeros across vmap
            vmap_vector = np.expand_dims(vectors[k], axis=0)
            num_to = len(param) - 1
            zeros_pad = np.zeros((num_to, *vmap_vector.shape[1:]))
            vmap_vector_padded = np.concatenate((vmap_vector, zeros_pad), axis=0)
            vmapped_dims[j].append(vmap_vector_padded)
          else:
            # assign this vmap dimension into the overall vector
            vmapped_dims[j][k][i] = vectors[k]

    self._config = original_configs

    # extract the final result, as would have been output from vmap
    (directly_changed,
     indirectly_changed,
     details) = vmapped_dims

    # extract the individual vectors - CAREFUL must match self._randomise_parameters() output
    (body_mass, 
     body_inertia, 
     body_pos, 
     body_ipos,
     actuator_gainprm,
     actuator_biasprm,
     actuator_gear,
     actuator_ctrlrange,
     dof_armature,
     ) = directly_changed

    (body_subtreemass,
     body_invweight0,
     dof_invweight0,
     dof_M0,
     eq_data,
     light_poscom0,
     actuator_acc0,
     meanmass,
     meansize,
     extent,
     center,
     body_sameframe,
     ) = indirectly_changed


    (payload_mass,
     payload_inertial_length,
     cab_mass_scale,
     cab_com_move,
     boom_mass_scale,
     boom_com_move,
     slew_act_scale,
     luff_act_scale,
     hoist_act_scale,
     slew_gear_ratio,
     luff_gear_ratio,
     hoist_gear_ratio,
     slew_armature_scale,
     luff_armature_scale,
     hoist_armature_scale,
     slew_ctrl_scale,
     luff_ctrl_scale,
     hoist_ctrl_scale,
     ) = details

    # assign the axes along which we will vmap
    in_axes = jax.tree.map(lambda x: None, self.mjx_model)
    in_axes = in_axes.tree_replace({
        # directly changed
        "body_mass" : 0,
        "body_inertia" : 0,
        "body_pos" : 0,
        "body_ipos" : 0,
        "actuator_gainprm": 0,
        "actuator_biasprm": 0,
        "actuator_gear": 0,
        "actuator_ctrlrange": 0,
        "dof_armature": 0,
        # indirectly changed
        "body_subtreemass": 0,
        "body_invweight0": 0,
        "dof_invweight0": 0,
        "dof_M0": 0,
        "eq_data": 0,
        # indirectly changed, but ignored for incompatibility, not involved in physics
        # "light_poscom0": 0, # np.array
        # "actuator_acc0": 0, # np.array
        # "stat.meanmass": 0, # member of stat field
        # "stat.meansize": 0, # member of stat field
        # "stat.extent" : 0, # member of stat field
        # "stat.center": 0, # member of stat field
        # "body_sameframe": 0, # np.array
    })

    # replace the same axes as above with extra dimensional tensors of options
    # conversion from numpy to JAX array done here
    mjx_model = self.mjx_model.tree_replace({
        # directly changed
        "body_mass" : body_mass,
        "body_inertia" : body_inertia,
        "body_pos" : body_pos,
        "body_ipos" : body_ipos,
        "actuator_gainprm" : actuator_gainprm,
        "actuator_biasprm" : actuator_biasprm,
        "actuator_gear": actuator_gear,
        "actuator_ctrlrange" : actuator_ctrlrange,
        "dof_armature": dof_armature,
        # indirectly changed
        "body_subtreemass" : body_subtreemass,
        "body_invweight0" : body_invweight0,
        "dof_invweight0" : dof_invweight0,
        "dof_M0" : dof_M0,
        "eq_data" : eq_data,
        # indirectly changed, but ignored for incompatibility, not involved in physics
        # "light_poscom0" : light_poscom0,  # np.array
        # "actuator_acc0" : actuator_acc0,  # np.array
        # "stat.meanmass" : meanmass, # member of stat field
        # "stat.meansize" : meansize, # member of stat field
        # "stat.extent" : extent, # member of stat field
        # "stat.center" : center, # member of stat field
        # "body_sameframe" : body_sameframe, # np.array
    })

    # mp, mp_norm = jax.vmap(self._init_motion_platform_test)(rng)

    # save extra information which describes how the environment is randomised
    extra_dict = {
      "payload_mass" : payload_mass,
      "payload_inertial_length" : payload_inertial_length,
      "cab_mass_scale" : cab_mass_scale,
      "cab_com_move" : cab_com_move,
      "boom_mass_scale" : boom_mass_scale,
      "boom_com_move" : boom_com_move,
      "slew_act_scale" : slew_act_scale,
      "luff_act_scale" : luff_act_scale,
      "hoist_act_scale" : hoist_act_scale,
      "slew_gear_ratio" : slew_gear_ratio,
      "luff_gear_ratio" : luff_gear_ratio,
      "hoist_gear_ratio" : hoist_gear_ratio,
      "slew_armature_scale" : slew_armature_scale,
      "luff_armature_scale" : luff_armature_scale,
      "hoist_armature_scale" : hoist_armature_scale,
      "slew_ctrl_scale" : slew_ctrl_scale,
      "luff_ctrl_scale" : luff_ctrl_scale,
      "hoist_ctrl_scale" : hoist_ctrl_scale,
    }

    # maintain ability to return brax compatible (no extras)
    if return_extra:
      return mjx_model, in_axes, extra_dict
    else:
      return mjx_model, in_axes

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """
    Reset one state to its initial conditions, including applying random noise.
    """

    if self.debug_jit:
      t0 = time.process_time()
      print("Reset called for JIT")
      print(f"rng -> shape = {rng.shape}, dtype = {rng.dtype}")

    # important! Do NOT change the order of keys. Add new keys by splitting more
    # eg rngs[0] will always be identical given rng, regardless of split num
    # maintaining order ensures random behaviour is always repeatable
    rngs = jrand.split(rng, 6)
    rng = rngs[0] # running key, put into info
    initial_pose_key = rngs[1]
    target_position_key = rngs[2]
    qpos_noise_key = rngs[3]
    qvel_noise_key = rngs[4]
    motion_plat_key = rngs[5]

    if self.debug_jit: 
      t1 = time.process_time()
      print(f"Time to trace the rng split: {t1-t0:.3f}")

    # randomise the initial position of the actuated crane joints
    qpos_init = self._get_random_initial_position(initial_pose_key)
    qvel_init = jnp.zeros((self._mj_model.nv))

    if self.debug_jit: 
      t2 = time.process_time()
      print(f"Time to trace the random initial pos: {t2-t1:.3f}")

    # add small random noise to initial qpos and qvel values (crane only, not motion plat)
    low, hi = -self._config.reset_noise_scale, self._config.reset_noise_scale
    crane_qpos = self._joint_dict["qpos_adr_crane"] # addresses of crane joints
    crane_qvel = self._joint_dict["dof_adr_crane"] # addresses of qvel of crane joints
    qpos_noise = jrand.uniform(qpos_noise_key, (len(crane_qpos)), minval=low, maxval=hi)
    qpos = qpos_init.at[crane_qpos].set(jnp.add(qpos_init[crane_qpos], qpos_noise))
    qvel = qvel_init.at[crane_qvel].set(
      jrand.uniform(qvel_noise_key, (len(crane_qvel)), minval=low, maxval=hi))
    
    if self.debug_jit: 
      t3 = time.process_time()
      print(f"Time to trace the random noise: {t3-t2:.3f}")

    # initialise a fresh environment, with a given qpos/qvel
    data = mjx_init(self.mjx_model, qpos=qpos, qvel=qvel)

    if self.debug_jit: 
      t4 = time.process_time()
      print(f"Time to trace the MJX init: {t4-t3:.3f}")

    # define initial conditions as zero
    metrics = jSG({
        "reward": jnp.zeros(()),
        "dist_reward": jnp.zeros(()),
        "threshold_reward": jnp.zeros(()),
        "acc_penalty": jnp.zeros(()),
        "unstable_reward": jnp.zeros(()),
        "action_penalty": jnp.zeros(()),
    })

    info = { 
        "rng" : rng,
        "unstable" : jnp.zeros((), dtype=int),
        "action" : jnp.zeros((3)),
        "ctrl_input" : jnp.zeros((3)),

        # used for action penalties
        "action_mean_mag" : jnp.zeros(()),
        "action_mean_change_mag": jnp.zeros(()),

        # used for reward
        "num_steps_within_threshold" : jnp.zeros((), dtype=int),
        "max_steps_within_threshold" : jnp.zeros((), dtype=int),

        # used for roa only
        "roa_state_history" : jnp.zeros(
            (self._config["roa_n_timesteps_history"],
             self._config["roa_n_proprioceptive"])
        ),
    }

    if self.debug_jit: 
      t5 = time.process_time()
      print(f"Time to trace the info/metrics creation: {t5-t4:.3f}")

    if self._config.use_motion_platform:
      info["mp"], info["mp_norm"] = self._init_motion_platform(motion_plat_key)
      info["action_mp"] = jnp.zeros((6))

    if self.debug_jit: 
      t6 = time.process_time()
      print(f"Time to trace the init motion platform: {t6-t5:.3f}")

    # initially, reward is zero and done is false
    reward = jnp.zeros(())
    done = jnp.zeros(())

    # update the sensors (required for _get_random_target and _get_obs)
    info = self._update_sensors(data, info)

    if self.debug_jit: 
      t7 = time.process_time()
      print(f"Time to trace the update sensors: {t7-t6:.3f}")

    # initialise with a random target position (must call AFTER _update_sensors())
    info = self._init_random_target(info, target_position_key, 
                                    qpos=qpos_init.at[self._joint_dict["qpos_adr"]].get())

    if self.debug_jit: 
      t8 = time.process_time()
      print(f"Time to trace the init random target: {t8-t7:.3f}")

    # finally, get the next observation (must call AFTER _update_sensors())
    obs, info = self._get_obs(data, info)

    if self.debug_jit: 
      t9 = time.process_time()
      print(f"Time to trace the get observation: {t9-t8:.3f}")

    # add useful information from mjdata into info
    info = self.add_q_values_to_info(data, info)
    info = self.add_domain_randomisation_to_info(info)

    if self.debug_jit: 
      t10 = time.process_time()
      print(f"Time to trace the add q values: {t10-t9:.3f}")

    info = freeze(info)

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """
    Apply an action to the environment, and resolve to a resultant state.
    """

    debug_prints = False
    if debug_prints:
      # DEBUGGING: print out domain randomised parameters
      id = self._body_dict["payload"]["id"]
      cid = self._body_dict["cab"]["id"]
      bid = self._body_dict["boom"]["id"]
      jax.debug.print(
          "Payload mass = {mass:.3f} kg, inertia = [{ixx:.3f}, {iyy:.3f}, {izz:.3f}] * 1e-3"
          ". Cab: mass = {cmass:.3f}, scale factor = {cscale:.2f}, com move = [{cmove1:.3f}, {cmove2:.3f}, {cmove3:.3f}]"
          ". Boom: mass = {bmass:.3f}, scale factor = {bscale:.2f}, com move = [{bmove1:.3f}, {bmove2:.3f}, {bmove3:.3f}]"
          ,
          mass=self.mjx_model.body_mass.at[id].get(),
          ixx=self.mjx_model.body_inertia.at[id, 0].get() * 1e3,
          iyy=self.mjx_model.body_inertia.at[id, 1].get() * 1e3,
          izz=self.mjx_model.body_inertia.at[id, 2].get() * 1e3,
          cmass=self.mjx_model.body_mass.at[cid].get(),
          cscale=self._extra_randomisation_info["cab_mass_scale"],
          cmove1=self._extra_randomisation_info["cab_com_move"][0],
          cmove2=self._extra_randomisation_info["cab_com_move"][1],
          cmove3=self._extra_randomisation_info["cab_com_move"][2],
          bmass=self.mjx_model.body_mass.at[bid].get(),
          bscale=self._extra_randomisation_info["boom_mass_scale"],
          bmove1=self._extra_randomisation_info["boom_com_move"][0],
          bmove2=self._extra_randomisation_info["boom_com_move"][1],
          bmove3=self._extra_randomisation_info["boom_com_move"][2],
      )

    info = unfreeze(state.info)

    if self.debug_jit:
      t0 = time.process_time()
      print("Step called for JIT")
      print(f"Action -> shape = {action.shape}, dtype = {action.dtype}")
      print(f"Obs -> shape = {state.obs.shape}, dtype = {state.obs.dtype}")
      print(f"Reward -> shape = {state.reward.shape}, dtype = {state.reward.dtype}")
      print(f"Done -> shape = {state.done.shape}, dtype = {state.done.dtype}")
      for key, val in state.metrics.items():
        try:
          print(f"Metrics.{key} -> shape = {val.shape}, dtype = {val.dtype}")
        except:
          print(f"Metrics.{val} not a jax tensor")
      for key, val in state.info.items():
        try:
          print(f"Info.{key} -> shape = {val.shape}, dtype = {val.dtype}")
        except:
          print(f"Info.{val} not a jax tensor")
      print("First action: ", action[0])
      # print("Info dict STEP:\n", state.info)

    # get the actions
    action, info = self._get_actions(action, info)
    if self._config.use_motion_platform:
      action_mp, info = self._get_motion_platform_action(state.data, info)
      action = jnp.concat((action_mp, action))
      # action = jnp.concat((jnp.zeros((6)), action))

    if self.debug_jit: 
      t1 = time.process_time()
      print(f"Time to trace the actions: {t1-t0:.3f}")

    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)

    if self.debug_jit: 
      t2 = time.process_time()
      print(f"Time to trace the MJX step: {t2-t1:.3f}")

    # update the sensors (required for _get_random_target and _get_obs)
    info = self._update_sensors(data, info)

    if self.debug_jit: 
      t3 = time.process_time()
      print(f"Time to trace the update sensors: {t3-t2:.3f}")

    # calculate the reward, update info and metrics, then lastly check for termination
    reward, info, metrics = self._get_reward(data, action, info, state.metrics)
    done = self._is_done(data, action, info, metrics) # must call AFTER _get_reward()

    if self.debug_jit: 
      t4 = time.process_time()
      print(f"Time to trace reward/done: {t4-t3:.3f}")

    # check if we randomly move the target position
    if self._config.target_move_chance > 1e-5:
      info["rng"], rng1 = jrand.split(info["rng"], 2)
      new_target_mask = (jrand.uniform(rng1) < self._config.target_move_chance)
      # must call AFTER _update_sensors()
      info = self._update_targets(info, new_target_mask)

    if self.debug_jit: 
      t5 = time.process_time()
      print(f"Time to trace the target update: {t5-t4:.3f}")

    # finally, get the next observation (must call AFTER _update_sensors())
    obs, info = self._get_obs(data, info)

    if self.debug_jit: 
      t6 = time.process_time()
      print(f"Time to trace the update sensors: {t6-t5:.3f}")

    # add useful information from mjdata into info
    info = self.add_q_values_to_info(data, info)
    info = self.add_domain_randomisation_to_info(info)

    if self.debug_jit: 
      t7 = time.process_time()
      print(f"Time to trace the add q values to info: {t7-t6:.3f}")

    info = freeze(info)

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def reset_to_state(self, rng: jax.Array, qpos: jax.Array, qvel: jax.Array) -> mjx_env.State:
    """
    Reset one state to its initial conditions, including applying random noise.
    """

    if self.debug_jit:
      t0 = time.process_time()
      print("Reset called for JIT")
      print(f"rng -> shape = {rng.shape}, dtype = {rng.dtype}")

    # important! Do NOT change the order of keys. Add new keys by splitting more
    # eg rngs[0] will always be identical given rng, regardless of split num
    # maintaining order ensures random behaviour is always repeatable
    rngs = jrand.split(rng, 6)
    rng = rngs[0] # running key, put into info
    initial_pose_key = rngs[1]
    target_position_key = rngs[2]
    qpos_noise_key = rngs[3]
    qvel_noise_key = rngs[4]
    motion_plat_key = rngs[5]

    if self.debug_jit: 
      t1 = time.process_time()
      print(f"Time to trace the rng split: {t1-t0:.3f}")

    # set the initial state to be that given to this function
    qpos_init = jnp.zeros((self._mj_model.nq))
    qvel_init = jnp.zeros((self._mj_model.nv))
    crane_qpos = self._joint_dict["qpos_adr_crane"] # addresses of crane joints
    crane_qvel = self._joint_dict["dof_adr_crane"] # addresses of qvel of crane joints
    qpos = qpos_init.at[crane_qpos].set(qpos)
    qvel = qvel_init.at[crane_qvel].set(qvel)
    
    if self.debug_jit: 
      t2 = time.process_time()
      print(f"Time to trace the given initial pos: {t2-t1:.3f}")

    # initialise a fresh environment, with a given qpos/qvel
    data = mjx_init(self.mjx_model, qpos=qpos, qvel=qvel)

    if self.debug_jit: 
      t4 = time.process_time()
      print(f"Time to trace the MJX init: {t4-t2:.3f}")

    # define initial conditions as zero
    metrics = jSG({
        "reward": jnp.zeros(()),
        "dist_reward": jnp.zeros(()),
        "threshold_reward": jnp.zeros(()),
        "acc_penalty": jnp.zeros(()),
        "unstable_reward": jnp.zeros(()),
        "action_penalty": jnp.zeros(()),
    })

    info = { 
        "rng" : rng,
        "unstable" : jnp.zeros((), dtype=int),
        "action" : jnp.zeros((3)),
        "ctrl_input" : jnp.zeros((3)),

        # used for action penalties
        "action_mean_mag" : jnp.zeros(()),
        "action_mean_change_mag": jnp.zeros(()),

        # used for reward
        "num_steps_within_threshold" : jnp.zeros((), dtype=int),
        "max_steps_within_threshold" : jnp.zeros((), dtype=int),

        # used for roa only
        "roa_state_history" : jnp.zeros(
            (self._config["roa_n_timesteps_history"],
             self._config["roa_n_proprioceptive"])
        ),
    }

    if self.debug_jit: 
      t5 = time.process_time()
      print(f"Time to trace the info/metrics creation: {t5-t4:.3f}")

    if self._config.use_motion_platform:
      info["mp"], info["mp_norm"] = self._init_motion_platform(motion_plat_key)
      info["action_mp"] = jnp.zeros((6))

    if self.debug_jit: 
      t6 = time.process_time()
      print(f"Time to trace the init motion platform: {t6-t5:.3f}")

    # initially, reward is zero and done is false
    reward = jnp.zeros(())
    done = jnp.zeros(())

    # update the sensors (required for _get_random_target and _get_obs)
    info = self._update_sensors(data, info)

    if self.debug_jit: 
      t7 = time.process_time()
      print(f"Time to trace the update sensors: {t7-t6:.3f}")

    # initialise with a random target position (must call AFTER _update_sensors())
    info = self._init_random_target(info, target_position_key, 
                                    qpos=qpos_init.at[self._joint_dict["qpos_adr"]].get())

    if self.debug_jit: 
      t8 = time.process_time()
      print(f"Time to trace the init random target: {t8-t7:.3f}")

    # finally, get the next observation (must call AFTER _update_sensors())
    obs, info = self._get_obs(data, info)

    if self.debug_jit: 
      t9 = time.process_time()
      print(f"Time to trace the get observation: {t9-t8:.3f}")

    # add useful information from mjdata into info
    info = self.add_q_values_to_info(data, info)
    info = self.add_domain_randomisation_to_info(info)

    if self.debug_jit: 
      t10 = time.process_time()
      print(f"Time to trace the add q values: {t10-t9:.3f}")

    info = freeze(info)

    return mjx_env.State(data, obs, reward, done, metrics, info)

  # ----- internals for public functions ----- #

  def _init_random_target(self, info, rng, qpos=None):
    """
    Initialise with a random target position
    """

    if self._config.joystick_target:

      # convert our current qpos to a payload position
      info["target_position"] = jSG(self.payload_pos_from_joint_angles(qpos))
      info["target_joints_final"] = jSG(qpos)
      info["target_joints_current"] = jSG(qpos.copy())
      info["target_stepsize"] = jSG(jrand.uniform(rng,
                                    self._joint_dict["minvals"].shape,
                                    minval=0,
                                    maxval=self._config.joystick_target_max_step))

      # info["target_joints_current"] = jSG(info["target_joints_final"].copy())
      # info["target_stepsize"] = (self._config.joystick_target_max_step * 
      #                            jnp.ones(self._joint_dict["minvals"].shape))

    else:

      # get a random target
      (info["target_position"],
       info["target_joints_final"]) = self._get_random_target(rng)

    # do any conversions to a global frame for the target position
    info["target_position_global"] = self.get_global_target(info)

    # old - joystick target would be placed randomly
    # if self._config.joystick_target:
    #   # info["target_joints_current"] = jSG(info["target_joints_final"].copy())
    #   # info["target_stepsize"] = jSG(jrand.uniform(joystick_vel_key,
    #   #                                                 self._joint_dict["minvals"].shape,
    #   #                                                 minval=0.2 * self._config.joystick_target_max_step,
    #   #                                                 maxval=self._config.joystick_target_max_step))
      
    #   # info["target_joints_current"] = jnp.zeros(target_joints_final.shape)
    #   # info["target_stepsize"] = jnp.zeros(self._joint_dict["minvals"].shape)

    #   info["target_joints_current"] = jSG(target_joints_final.copy())
    #   info["target_stepsize"] = (self._config.joystick_target_max_step * 
    #                              jnp.ones(self._joint_dict["minvals"].shape))

    # info["target_position"] = target_position
    # info["target_joints_final"] = target_joints_final
    
    return info

  def _update_targets(self, info, new_target_mask):
    """
    Update any targets positions using the new_target_mask
    """

    def mask_update(new, name, info):
      info[name] = jax.lax.select(new_target_mask, new, info[name])
      return info

    info["rng"], new_target_key = jrand.split(info["rng"], 2)

    # generate the new target positions
    (new_target_position,
     new_target_joints_final) = self._get_random_target(new_target_key)
    
    # if we are using a joystick target, increment our target to the final position
    if self._config.joystick_target:

      # update the final targets, based on the mask
      info = mask_update(new_target_joints_final, "target_joints_final", info)

      # update all target positions based on approaching the final target
      diff = jSG(info["target_joints_final"] - info["target_joints_current"])
      step = jSG(jnp.clip(diff, -info["target_stepsize"], info["target_stepsize"]))
      info["target_joints_current"] = jSG(step + info["target_joints_current"])
      info["target_position"] = jSG(self.payload_pos_from_joint_angles(info["target_joints_current"]))

    else:
      # simply update targets with the new random positions based on the mask
      info = mask_update(new_target_joints_final, "target_joints_final", info)
      info = mask_update(new_target_position, "target_position", info)

    # update target co-ordinates in global frame
    info["target_position_global"] = self.get_global_target(info)

    return info

  def _get_random_target(self, rng):
    """
    Returns a random target position for the crane, based on a random set of joint
    angles. Returns both the target, and the corresponding joint angles.
    """

    # randomised set of joint angles
    target_joints = jrand.uniform(rng, 
                                  self._joint_dict["minvals"].shape,
                                  minval=self._joint_dict["minvals"],
                                  maxval=self._joint_dict["maxvals"])

    # corresponding randomised cartesian position (in LOCAL frame)
    target_pos = self.payload_pos_from_joint_angles(target_joints)

    # remove any gradient information from the random target
    target_pos = jSG(target_pos)
    target_joints = jSG(target_joints)

    return target_pos, target_joints

  def _get_random_initial_position(self, rng):
    """
    Get a random initial joint configuration for the crane.
    """

    # for our actuated joints, set them randomly anywhere within their limits
    qpos = jnp.array(self._mj_model.qpos0)
    qpos = qpos.at[self._joint_dict["qpos_adr"]].set(
        jrand.uniform(rng, 
                      self._joint_dict["minvals"].shape,
                      minval=self._joint_dict["minvals"],
                      maxval=self._joint_dict["maxvals"])
    )

    # set the phi string hinge to the negative luff angle, so the 'string' starts straight
    qpos = qpos.at[self._joint_dict["luff-joint"]["qpos_adr"] + 2].set(
        qpos[self._joint_dict["luff-joint"]["qpos_adr"]]
    )

    # remove any gradient information from the randomised initial configuration
    qpos = jSG(qpos)

    return qpos

  def _init_motion_platform(self, rng):
    """
    Initialise a random motion plan for the motion platform
    """
    rngs = jrand.uniform(rng, (7, 3))

    # each actuator has a cycle time and amplitude defined
    min_amp_pos = -self._config.mp_max_pos_amplitude
    max_amp_pos = self._config.mp_max_pos_amplitude
    min_amp_rot = -self._config.mp_max_rot_amplitude
    max_amp_rot = self._config.mp_max_rot_amplitude
    min_cycle_s = self._config.mp_min_cycle_time
    max_cycle_s = self._config.mp_max_cycle_time
    min_freq_hz = 1 / max_cycle_s
    max_freq_hz = 1 / min_cycle_s

    # # for testing ONLY, set all to maximum
    # min_amp_pos = self._config.mp_max_pos_amplitude
    # min_amp_rot = self._config.mp_max_rot_amplitude
    # max_cycle_s = self._config.mp_min_cycle_time
    # min_noise_pos = self._config.mp_max_pos_noise
    # min_noise_rot = self._config.mp_max_rot_noise

    all_disabled = (rngs[6][0] < self._config.mp_all_off_chance)

    # make immutable motion platform object to characterise motions
    mp = Motors(
        pos_z=MotionParams(
            amp=min_amp_pos + rngs[0][0] * (max_amp_pos - min_amp_pos),
            freq=min_freq_hz + rngs[0][1] * (max_freq_hz - min_freq_hz),
            enabled=(1 - all_disabled) * (rngs[0][2] > self._config.mp_each_off_chance),
        ),
        pos_y=MotionParams(
            amp=min_amp_pos + rngs[1][0] * (max_amp_pos - min_amp_pos),
            freq=min_freq_hz + rngs[1][1] * (max_freq_hz - min_freq_hz),
            enabled=(1 - all_disabled) * (rngs[1][2] > self._config.mp_each_off_chance),
        ),
        pos_x=MotionParams(
            amp=min_amp_pos + rngs[2][0] * (max_amp_pos - min_amp_pos),
            freq=min_freq_hz + rngs[2][1] * (max_freq_hz - min_freq_hz),
            enabled=(1 - all_disabled) * (rngs[2][2] > self._config.mp_each_off_chance),
        ),
        rot_x=MotionParams(
            amp=min_amp_rot + rngs[3][0] * (max_amp_rot - min_amp_rot),
            freq=min_freq_hz + rngs[3][1] * (max_freq_hz - min_freq_hz),
            enabled=(1 - all_disabled) * (rngs[3][2] > self._config.mp_each_off_chance),
        ),
        rot_y=MotionParams(
            amp=min_amp_rot + rngs[4][0] * (max_amp_rot - min_amp_rot),
            freq=min_freq_hz + rngs[4][1] * (max_freq_hz - min_freq_hz),
            enabled=(1 - all_disabled) * (rngs[4][2] > self._config.mp_each_off_chance),
        ),
        rot_z=MotionParams(
            amp=min_amp_rot + rngs[5][0] * (max_amp_rot - min_amp_rot),
            freq=min_freq_hz + rngs[5][1] * (max_freq_hz - min_freq_hz),
            enabled=0, # permanently disabled, as these oscillations are unrealistic for waves
        ),
    )

    # calculate normalised parameters for amplitude and frequency
    mp_norm = MotorsNorm(
        pos_z=MotionParamsNorm(
            amp_norm=norm_min_max(mp.pos_z.amp * mp.pos_z.enabled, 
                                  min=min_amp_pos, max=max_amp_pos),
            freq_norm=norm_min_max(mp.pos_z.freq * mp.pos_z.enabled, 
                                   min=min_freq_hz, max=max_freq_hz),
        ),
        pos_y=MotionParamsNorm(
            amp_norm=norm_min_max(mp.pos_y.amp * mp.pos_y.enabled, 
                                  min=min_amp_pos, max=max_amp_pos),
            freq_norm=norm_min_max(mp.pos_y.freq * mp.pos_y.enabled, 
                                   min=min_freq_hz, max=max_freq_hz),
        ),
        pos_x=MotionParamsNorm(
            amp_norm=norm_min_max(mp.pos_x.amp * mp.pos_x.enabled, 
                                  min=min_amp_pos, max=max_amp_pos),
            freq_norm=norm_min_max(mp.pos_x.freq * mp.pos_x.enabled, 
                                   min=min_freq_hz, max=max_freq_hz),
        ),
        rot_x=MotionParamsNorm(
            amp_norm=norm_min_max(mp.rot_x.amp * mp.rot_x.enabled, 
                                  min=min_amp_rot, max=max_amp_rot),
            freq_norm=norm_min_max(mp.rot_x.freq * mp.rot_x.enabled, 
                                   min=min_freq_hz, max=max_freq_hz),
        ),
        rot_y=MotionParamsNorm(
            amp_norm=norm_min_max(mp.rot_y.amp * mp.rot_y.enabled, 
                                  min=min_amp_rot, max=max_amp_rot),
            freq_norm=norm_min_max(mp.rot_y.freq * mp.rot_y.enabled, 
                                   min=min_freq_hz, max=max_freq_hz),
        ),
        rot_z=MotionParamsNorm(
            amp_norm=norm_min_max(mp.rot_z.amp * mp.rot_z.enabled, 
                                  min=min_amp_rot, max=max_amp_rot),
            freq_norm=norm_min_max(mp.rot_z.freq * mp.rot_z.enabled, 
                                   min=min_freq_hz, max=max_freq_hz),
        ),
    )

    return mp, mp_norm

  def _get_motion_platform_action(self, data, info):
    """
    Get the next action for the motion platform
    """

    # mp = self._extra_randomisation_info["mp"]
    mp = info["mp"]

    # create periodic motion of each of the 6DoF
    c = data.time * 2 * jnp.pi
    action = jnp.array([
        mp.pos_z.enabled * mp.pos_z.amp * jnp.sin(c * mp.pos_z.freq),
        mp.pos_y.enabled * mp.pos_y.amp * jnp.sin(c * mp.pos_y.freq),
        mp.pos_x.enabled * mp.pos_x.amp * jnp.sin(c * mp.pos_x.freq),
        mp.rot_x.enabled * mp.rot_x.amp * jnp.sin(c * mp.rot_x.freq),
        mp.rot_y.enabled * mp.rot_y.amp * jnp.sin(c * mp.rot_y.freq),
        mp.rot_z.enabled * mp.rot_z.amp * jnp.sin(c * mp.rot_z.freq),
    ])

    # generate noise to add onto the position commands
    info["rng"], rng = jrand.split(info["rng"])
    rand = jrand.uniform(rng, 6)
    action_noise = jnp.array([
        mp.pos_z.enabled * (2 * rand[0] - 1) * self._config.mp_max_pos_noise,
        mp.pos_y.enabled * (2 * rand[1] - 1) * self._config.mp_max_pos_noise,
        mp.pos_x.enabled * (2 * rand[2] - 1) * self._config.mp_max_pos_noise,
        mp.rot_x.enabled * (2 * rand[3] - 1) * self._config.mp_max_rot_noise,
        mp.rot_y.enabled * (2 * rand[4] - 1) * self._config.mp_max_rot_noise,
        mp.rot_z.enabled * (2 * rand[5] - 1) * self._config.mp_max_rot_noise,
    ])

    final_action = jnp.add(action, action_noise)
    info["action_mp"] = action

    # for debugging
    # amp_norm = jnp.concatenate([
    #   jnp.expand_dims(info["mp_norm"].pos_x.amp_norm, 0), # n=1
    #   jnp.expand_dims(info["mp_norm"].pos_y.amp_norm, 0), # n=1
    #   jnp.expand_dims(info["mp_norm"].pos_z.amp_norm, 0), # n=1
    #   jnp.expand_dims(info["mp_norm"].rot_x.amp_norm, 0), # n=1
    #   jnp.expand_dims(info["mp_norm"].rot_y.amp_norm, 0), # n=1
    # ])
    # freq_norm = jnp.concatenate([
    #   jnp.expand_dims(info["mp_norm"].pos_x.freq_norm, 0), # n=1
    #   jnp.expand_dims(info["mp_norm"].pos_y.freq_norm, 0), # n=1
    #   jnp.expand_dims(info["mp_norm"].pos_z.freq_norm, 0), # n=1
    #   jnp.expand_dims(info["mp_norm"].rot_x.freq_norm, 0), # n=1
    #   jnp.expand_dims(info["mp_norm"].rot_y.freq_norm, 0), # n=1
    # ])
    # jax.debug.print("Final MP action={a}, action_noise={b}, action={c}"
    #                 "\n -> amp_norm={d}, freq_norm={e}", 
    #                 a=final_action, b=action_noise, c=action,
    #                 d=amp_norm, e=freq_norm)

    return final_action, info

  def _update_sensors(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, Any]:
    """
    Add in sensor data readings from the simulation into the given info dict. NaNs
    can arise in readings, which triggers done=True. However, the NaNs will corrupt
    potentially both the observation and the reward for this final state in the
    trajectory. Hence, enforce safe NaN handling.
    """
    # check for unstable physics
    info["unstable"] = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()

    for name in self.sensor_names:
      sensor_data = mjx_env.get_sensor_data(self.mj_model, data, name)
      sensor_data = jnp.nan_to_num(sensor_data, nan=0.0) # avoid unstable physics
      info[name] = sensor_data

    # convert into local co-ordinate system (only makes a difference with motion plat)
    if self._config.use_motion_platform:
      info["payload_position_local"] = self.vector_global_to_frame(info["payload_position"],
                                                                   frame_pos=info["base_position"],
                                                                   frame_quat=info["base_orientation"])
    else:
      info["payload_position_local"] = jSG(
        info["payload_position"].copy()
      )

    return info

  def _get_actions(self, action: jax.Array, info: dict[str, Any]) -> tuple[jax.Array, dict[str, Any]]:
    """
    Normalise actions from the range [-1, +1] into the mujoco simulation, based on
    the ctrlrange parameter for each motor.
    """

    debug_actions = False
    
    if debug_actions: original_action = action

    # extract average abs action size, before any clips
    info["action_mean_mag"] = jSG(jnp.mean(jnp.abs(action)))
    info["action_mean_change_mag"] = jSG(jnp.mean(jnp.abs(action - info["action"])))

    # clip network output actions into a defined range
    if self._config["clip_actions"]:
      x = self._config["action_clip_range"]
      action = jnp.clip(action, min=-x, max=x)

    # save the clipped action
    info["action"] = action
    
    if debug_actions: clip_action = action

    # after action clipped to [-1, +1], scale it up to the maximum ctrlrange
    # WARNING: minimum ctrlrange is assumed to have identical magnitude to maximum
    if self._config["scale_actions"]:
      # 0=min (need *-1), 1=max
      ids = self._actuator_dict["act_ids"]
      action = jnp.multiply(action, self.mjx_model.actuator_ctrlrange[ids, 1])
      action = jnp.multiply(self._config["action_scale_ctrlrange_frac"], action)

    if debug_actions: scaled_action = action

    if self._config["relative_actions"]:
      # the action refers to a change relative to the target
      action = jnp.add(action, info["ctrl_input"])
      # clip the resultant relative action to remain in the ctrlrange
      xmin = self.mjx_model.actuator_ctrlrange[ids, 0]
      xmax = self.mjx_model.actuator_ctrlrange[ids, 1]
      action = jnp.clip(action, min=xmin, max=xmax)

    if debug_actions: 
      final_action = action
      jax.debug.print("Action pre-clip: {a1}, after clip: {a2}, after ctrlrange scale: {a3}"
                      ", final_action: {a4}"
                      ", ctrlrange: {ctrlrange}"
                      ", ctrlrange[ids]: {ctrlrangeids}"
                      ", ctrlrange_frac: {frac}"
                      ,
                      a1=original_action, 
                      a2=clip_action, 
                      a3=scaled_action, 
                      a4=final_action, 
                      ctrlrange=self.mjx_model.actuator_ctrlrange,
                      ctrlrangeids=self.mjx_model.actuator_ctrlrange[ids, 1],
                      frac=self._config["action_scale_ctrlrange_frac"],
                      )

    info["ctrl_input"] = action

    return action, info

  def _get_obs_default(self, data: mjx.Data, info: dict[str, Any]) -> tuple[jax.Array, dict[str, Any]]:
    """
    State observation to pass to the agent
    """

    # exclude motion platform (if in use) from observation
    obs = jnp.concatenate([
        data.qpos[self._joint_dict["qpos_adr_crane"]],
        data.qvel[self._joint_dict["dof_adr_crane"]],
    ])

    # important! Handle possibility of NaNs in qpos and qvel
    obs = jnp.nan_to_num(obs, nan=0.0)

    return obs, info
  
  def _get_obs_dreamer(self, data: mjx.Data, info: dict[str, Any]) -> tuple[jax.Array, dict[str, Any]]:
    """
    State observation to pass to the agent, includes the target position for training dreamer policies
    """

    # exclude motion platform (if in use) from observation
    obs = jnp.concatenate([
        data.qpos[self._joint_dict["qpos_adr_crane"]],
        data.qvel[self._joint_dict["dof_adr_crane"]],
        info["target_position"]
    ])

    # important! Handle possibility of NaNs in qpos and qvel
    obs = jnp.nan_to_num(obs, nan=0.0)

    return obs, info

  def _get_obs_roa_main(self, data: mjx.Data, info: dict[str, Any]) -> tuple[jax.Array, dict[str, Any]]:
    """
    Return a state observation, but include also the privileged environment vector.

    The main paper for extreme-parkour (implementing ROA), uses an observation vector
    with the following components:
      - proprioceptive data, joint angles etc
      - scan points, depth data over the local terrain
      - private state data, base linear velocity which is predicted by the Estimator
      - private environment data, privileged information about the environment, predicted
        by the adaptation module

    To meet the requirments of their code with minimal changes, this function should
    output 'data' for all of these fields. However, we are only interested in the
    proprioceptive information and the private environmnet vector.
    """

    # exclude motion platform (if in use) from observation
    obs_prop = jnp.concatenate([
        data.qpos[self._joint_dict["qpos_adr_crane"]],
        data.qvel[self._joint_dict["dof_adr_crane"]],
        info["target_position"]
    ])

    # are we including the action in the observation
    if self._config.roa_action_in_obs:
      obs_prop = jnp.concatenate([
        obs_prop,
        info["action"], # raw action from network, clipped [-1, +1]
      ])

    # important! Handle possibility of NaNs in qpos and qvel
    obs_prop = jnp.nan_to_num(obs_prop, nan=0.0)

    # also normalise the privileged information given the limits
    norm_payload_mass = norm_min_max(self._extra_randomisation_info["payload_mass"],
                                     self._config.payload_min_mass,
                                     self._config.payload_max_mass)
    norm_inertial_len = norm_min_max(self._extra_randomisation_info["payload_inertial_length"],
                                     self._config.payload_min_inertial_length,
                                     self._config.payload_max_inertial_length)
    norm_cab_scale = norm_scale(self._extra_randomisation_info["cab_mass_scale"],
                                self._config.mass_max_scaling)
    norm_boom_scale = norm_scale(self._extra_randomisation_info["boom_mass_scale"],
                                 self._config.mass_max_scaling)
    norm_slew_scale = norm_scale(self._extra_randomisation_info["slew_act_scale"],
                                 self._config.actuator_max_scaling)
    norm_luff_scale = norm_scale(self._extra_randomisation_info["luff_act_scale"],
                                 self._config.actuator_max_scaling)
    norm_hoist_scale = norm_scale(self._extra_randomisation_info["hoist_act_scale"],
                                  self._config.actuator_max_scaling)
    norm_slew_gear_ratio = norm_min_max(self._extra_randomisation_info["slew_gear_ratio"],
                                        self._config.min_gear_ratio,
                                        self._config.max_gear_ratio)
    norm_luff_gear_ratio = norm_min_max(self._extra_randomisation_info["luff_gear_ratio"],
                                        self._config.min_gear_ratio,
                                        self._config.max_gear_ratio)
    norm_hoist_gear_ratio = norm_min_max(self._extra_randomisation_info["hoist_gear_ratio"],
                                         self._config.min_gear_ratio,
                                         self._config.max_gear_ratio)
    norm_slew_armature_scale = norm_min_max(self._extra_randomisation_info["slew_armature_scale"],
                                            self._config.min_armature_scale,
                                            self._config.max_armature_scale)
    norm_luff_armature_scale = norm_min_max(self._extra_randomisation_info["luff_armature_scale"],
                                            self._config.min_armature_scale,
                                            self._config.max_armature_scale)
    norm_hoist_armature_scale = norm_min_max(self._extra_randomisation_info["hoist_armature_scale"],
                                             self._config.min_armature_scale,
                                             self._config.max_armature_scale)
    norm_slew_ctrl_scale = norm_min_max(self._extra_randomisation_info["slew_ctrl_scale"],
                                        self._config.min_ctrlrange_scale,
                                        self._config.max_ctrlrange_scale)
    norm_luff_ctrl_scale = norm_min_max(self._extra_randomisation_info["luff_ctrl_scale"],
                                        self._config.min_ctrlrange_scale,
                                        self._config.max_ctrlrange_scale)
    norm_hoist_ctrl_scale = norm_min_max(self._extra_randomisation_info["hoist_ctrl_scale"],
                                         self._config.min_ctrlrange_scale,
                                         self._config.max_ctrlrange_scale)

    # build the privileged information vector
    obs_priv = jnp.concatenate([
        jnp.expand_dims(norm_payload_mass, 0), # n=1
        jnp.expand_dims(norm_slew_armature_scale, 0), # n=1
        jnp.expand_dims(norm_luff_armature_scale, 0), # n=1
        jnp.expand_dims(norm_hoist_armature_scale, 0), # n=1
        jnp.expand_dims(norm_slew_ctrl_scale, 0), # n=1
        jnp.expand_dims(norm_luff_ctrl_scale, 0), # n=1
        jnp.expand_dims(norm_hoist_ctrl_scale, 0), # n=1
    ])

    if self._config.use_motion_platform:
      # mp_norm = self._extra_randomisation_info["mp_norm"]
      mp_norm = info["mp_norm"]
      obs_priv = jnp.concatenate([
          obs_priv,
          # normalised amplitude and frequency of motion platform motors
          jnp.expand_dims(mp_norm.pos_x.amp_norm, 0), # n=1
          jnp.expand_dims(mp_norm.pos_y.amp_norm, 0), # n=1
          jnp.expand_dims(mp_norm.pos_z.amp_norm, 0), # n=1
          jnp.expand_dims(mp_norm.rot_x.amp_norm, 0), # n=1
          jnp.expand_dims(mp_norm.rot_y.amp_norm, 0), # n=1
          # jnp.expand_dims(mp_norm.rot_z.amp_norm, 0), # n=1 # this DoF is not actuated
          jnp.expand_dims(mp_norm.pos_x.freq_norm, 0), # n=1
          jnp.expand_dims(mp_norm.pos_y.freq_norm, 0), # n=1
          jnp.expand_dims(mp_norm.pos_z.freq_norm, 0), # n=1
          jnp.expand_dims(mp_norm.rot_x.freq_norm, 0), # n=1
          jnp.expand_dims(mp_norm.rot_y.freq_norm, 0), # n=1
          # jnp.expand_dims(mp_norm.rot_z.freq_norm, 0), # n=1 # this DoF is not actuated
      ])

    # get observation at the current timestep
    obs = jnp.concatenate([
        # propriceptive data
        obs_prop,
        # privileged environment vector
        obs_priv,
        # full state history for N timesteps (but NOT including the current observation)
        jnp.ravel(info["roa_state_history"]),
    ])

    # TESTING: swap target vector for last action in prop obs
    obs_prop_hist = jnp.concatenate([
        data.qpos[self._joint_dict["qpos_adr_crane"]],
        data.qvel[self._joint_dict["dof_adr_crane"]],
        # jnp.zeros((3)), # mask out target
        jSG(info["action"]), # raw action from network, clipped [-1, +1]
    ])
    obs_prop_hist = jnp.nan_to_num(obs_prop_hist, nan=0.0)

    # now update the state history buffer to include the new proprioceptive observation
    info["roa_state_history"] = jnp.roll(info["roa_state_history"], shift=-1, axis=0)
    info["roa_state_history"] = info["roa_state_history"].at[-1].set(obs_prop_hist)

    return obs, info

  def _is_done(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    """
    Check whether the environment has reached a terminal state.
    """
    del action, data, metrics # unused

    # check for any nans in the physics (i.e., unstable simulation)
    done = info["unstable"]

    # check if the payload has reached the target position for enough steps
    if self._config.threshold_termination:
      within_threshold = jnp.greater_equal(info["num_steps_within_threshold"], 
                                           self._config.num_required_within_threshold)
      done = jnp.logical_or(done, within_threshold)

    # convert bool to float
    done = done.astype(float)

    return done

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> tuple[jax.Array, dict[str, Any], dict[str, Any]]:

    del action, data # unused.

    payload_pos = info["payload_position_local"] # local position
    payload_acc = jnp.linalg.norm(info["payload_linear_acceleration"])
    target_pos = info["target_position"] # local position (must match frame of payload!)

    # apply a penalty based on our distance, normalised to give -1 past distance_normalisation
    dist = jnp.linalg.norm(payload_pos - target_pos)
    dist_penalty = -(jnp.minimum(dist, self._config["distance_normalisation"])
                     / (self._config["distance_normalisation"])) 

    # further normalise to get -1 at truncation
    dist_penalty = dist_penalty / self._config["episode_length"]

    # track the number of steps in a row we remain inside the desired threshold
    within_threshold = jnp.less(dist, self._config.target_threshold).astype(int)
    info["num_steps_within_threshold"] = jnp.multiply(info["num_steps_within_threshold"], 
                                                      within_threshold) # wipe if false
    info["num_steps_within_threshold"] = jnp.add(info["num_steps_within_threshold"], 
                                                 within_threshold) # increment if true

    if self._config.threshold_termination:
      # reward remaining at the target position for longer than ever before
      new_record = jnp.greater(info["num_steps_within_threshold"], 
                               info["max_steps_within_threshold"]).astype(int)
      info["max_steps_within_threshold"] = jnp.add(info["max_steps_within_threshold"], 
                                                   new_record) # increment if true
      threshold_bonus = (1 / self._config["num_required_within_threshold"])
      threshold_bonus *= new_record.astype(float) # add only if we have a new record
    else:
      # simply provide the reward if we are within the threshold
      threshold_bonus = within_threshold.astype(
          float) * (1 / self._config["episode_length"])

    # apply a penalty based on the acceleration of the payload
    acc_penalty = -(jnp.minimum(payload_acc, self._config["acceleration_normalisation"])
                    / (self._config["acceleration_normalisation"]))
    acc_penalty = acc_penalty / self._config["episode_length"]

    if self._config.penalise_action_mag:
      if self._config.use_action_change_mag:
        value = info["action_mean_change_mag"] # penalise change in action size
      else:
        value = info["action_mean_mag"] # penalise raw action size
      action_penalty = -(jnp.minimum(value, self._config["action_mag_normalisation"])
                         / (self._config["action_mag_normalisation"]))
      # if the below the threshold, no penalty
      action_penalty = jax.lax.select(jnp.abs(action_penalty) > self._config.action_penalty_threshold,
                                      action_penalty,
                                      jnp.zeros(()))
      # now scale down to the desired magnitude
      action_penalty = action_penalty / self._config["episode_length"]
      action_penalty *= self._config.action_penalty_scale
    else:
      action_penalty = jnp.zeros(())

    # penalise if physics is unstable (bad data)
    unstable_penalty = -1.0 * info["unstable"]

    # set the final reward
    reward = (dist_penalty + threshold_bonus + acc_penalty + unstable_penalty
              + action_penalty)

    metrics["reward"] = jSG(reward)
    metrics["dist_reward"] = jSG(dist_penalty)
    metrics["threshold_reward"] = jSG(threshold_bonus)
    metrics["acc_penalty"] = jSG(acc_penalty)
    metrics["unstable_reward"] = jSG(unstable_penalty)
    metrics["action_penalty"] = jSG(action_penalty)

    return reward, info, metrics

  def _randomise_parameters(self, param, fcn):
    """
    Pass in a parameter (vectorised) and a function (not vectorised). The parameter,
    for example an rng key, will be given as input to the function in order to
    determine the final randomisation. The fcn should return the randomised parameter,
    and a new value for 'param' (e.g. a new rng key).
    """

    for name in self._key_bodies:

      if name == "payload":

        # calculate payload mass and size (then infer inertia)
        payload_mass, param = fcn(param, self._config.payload_min_mass,
                                  self._config.payload_max_mass)
        payload_inertial_length, param = fcn(param, self._config.payload_min_inertial_length,
                                             self._config.payload_max_inertial_length)
        payload_inertia = (2.0/12.0) * payload_mass * payload_inertial_length ** 2

        # apply these values
        body_spec = self.mj_spec.find_body(name)
        body_spec.mass = np.float32(payload_mass)
        body_spec.inertia = np.array(
            [payload_inertia for i in range(3)], dtype=np.float32)

      else:

        # get a random mass scaling and centre of mass peturbation
        mass_scale, param = fcn(param, 1 - self._config.mass_max_scaling,
                                1 + self._config.mass_max_scaling)
        com_move, param = fcn(param, -self._config.com_max_move, 
                              self._config.com_max_move, shape=3)

        # apply these peturbations
        body_spec = self.mj_spec.find_body(name)
        body_spec.mass = self._body_dict[name]["mass"] * np.float32(mass_scale)
        body_spec.inertia = self._body_dict[name]["inertia"] * np.float32(mass_scale)
        body_spec.pos = self._body_dict[name]["pos"] + np.float32(com_move)
        body_spec.ipos = self._body_dict[name]["ipos"] + np.float32(com_move)

        if name == "cab":
          cab_mass_scale = mass_scale
          cab_com_move = com_move
        elif name == "boom":
          boom_mass_scale = mass_scale
          boom_com_move = com_move

    # apply randomisation over the actuators
    for name in self._key_actuators:

      act_scale, param = fcn(param, 1 - self._config.actuator_max_scaling,
                             1 + self._config.actuator_max_scaling)
      gear_ratio, param = fcn(param, self._config.min_gear_ratio,
                              self._config.max_gear_ratio)
      ctrl_scale, param = fcn(param, self._config.min_ctrlrange_scale,
                              self._config.max_ctrlrange_scale)

      # apply the actuator scaling
      actuator = self.mj_spec.find_actuator(name)
      # velocity servo
      # gain = [kv, 0, 0]
      # bias = [0, 0, -kv]
      if np.abs(actuator.biasprm[1]) < 1e-5:
        actuator.gainprm[0] = self._actuator_dict[name]["gain"] * np.float32(act_scale)
        actuator.biasprm[2] = self._actuator_dict[name]["bias"][2] * np.float32(act_scale)
      # int-velocity servo
      # gain = [kp, 0, 0]
      # bias = [0, -kp, -kv]
      else:
        actuator.gainprm[0] = self._actuator_dict[name]["gain"] * np.float32(act_scale)
        actuator.biasprm[1] = self._actuator_dict[name]["bias"][1] * np.float32(act_scale)

      # ---!! Important !!--- #
      # actuator.ctrlrange *= dict["ctrlrange"] * np.float32(ctrl) # FAILED! Weird values, not correct
      # actuator.gear *= dict["gear"] * np.float32(ratio) # FAILED! Weird values, not correct

      # adjust the ctrlrange minimum [0] and maximum [1]
      actuator.ctrlrange[0] = self._actuator_dict[name]["ctrlrange"][0] * np.float32(ctrl_scale)
      actuator.ctrlrange[1] = self._actuator_dict[name]["ctrlrange"][1] * np.float32(ctrl_scale)

      # gear is a 6DoF vector, first element is ratio for scalar actuator
      actuator.gear[0] = self._actuator_dict[name]["gear"] * np.float32(gear_ratio)

      if name == "slew-velocity":
        slew_act_scale = act_scale
        slew_gear_ratio = gear_ratio
        slew_ctrl_scale = ctrl_scale
      elif name == "luff-velocity":
        luff_act_scale = act_scale
        luff_gear_ratio = gear_ratio
        luff_ctrl_scale = ctrl_scale
      elif name == "hoist-velocity":
        hoist_act_scale = act_scale
        hoist_gear_ratio = gear_ratio
        hoist_ctrl_scale = ctrl_scale

    # apply randomisation over the joints
    for name in self._key_joints:

      # chose a random value in log space, then convert back after
      armature, param = fcn(param, self._config.min_armature_scale,
                            self._config.max_armature_scale)
      
      # apply these values
      joint_spec = next(j for j in self.mj_spec.joints if j.name == name)
      joint_spec.armature = self._joint_dict[name]["armature"] * np.float32(armature)

      if name == "slew-joint":
        slew_armature_scale = armature
      elif name == "luff-joint":
        luff_armature_scale = armature
      elif name == "hoist-slider":
        hoist_armature_scale = armature

    details = (
        payload_mass,
        payload_inertial_length,
        cab_mass_scale, 
        cab_com_move,
        boom_mass_scale, 
        boom_com_move,
        slew_act_scale,
        luff_act_scale,
        hoist_act_scale,
        slew_gear_ratio,
        luff_gear_ratio,
        hoist_gear_ratio,
        slew_armature_scale,
        luff_armature_scale,
        hoist_armature_scale,
        slew_ctrl_scale,
        luff_ctrl_scale,
        hoist_ctrl_scale,
    )

    # recompile the mj_model to get a full suite of new parameters
    mj_model = self.mj_spec.compile()

    # for debugging and checking only
    self.last_domain_randomised_model = mj_model

    # debug_xml = self.mj_spec.to_xml()
    # with open(f"debug_{int(param)}.xml", "w") as f:
    #   f.write(debug_xml)

    directly_changed = (
        mj_model.body_mass.copy(), 
        mj_model.body_inertia.copy(), 
        mj_model.body_pos.copy(), 
        mj_model.body_ipos.copy(),
        mj_model.actuator_gainprm.copy(),
        mj_model.actuator_biasprm.copy(),
        mj_model.actuator_gear.copy(),
        mj_model.actuator_ctrlrange.copy(),
        mj_model.dof_armature.copy(),
    )

    indirectly_changed = (
        # used
        mj_model.body_subtreemass.copy(),
        mj_model.body_invweight0.copy(),
        mj_model.dof_invweight0.copy(),
        mj_model.dof_M0.copy(),
        mj_model.eq_data.copy(),
        # not used
        mj_model.light_poscom0.copy(),
        mj_model.actuator_acc0.copy(),
        mj_model.stat.meanmass,
        mj_model.stat.meansize,
        mj_model.stat.extent,
        mj_model.stat.center,
        mj_model.body_sameframe.copy(),
    )

    return directly_changed, indirectly_changed, details

  # ----- helper functions ----- #

  def add_q_values_to_info(self, data, info):
    """
    Add qpos and qvel into the info dict, for both the crane, motion platform,
    and both
    """

    info = unfreeze(info)

    info["qpos"] = jSG(data.qpos)
    info["qvel"] = jSG(data.qvel)
    info["qpos_crane"] = jSG(data.qpos[self._joint_dict["qpos_adr_crane"]])
    info["qvel_crane"] = jSG(data.qvel[self._joint_dict["dof_adr_crane"]])
    if self._config.use_motion_platform:
      info["qpos_motion_plat"] = jSG(data.qpos[self._joint_dict_mp["qpos_adr"]])
      info["qvel_motion_plat"] = jSG(data.qvel[self._joint_dict_mp["dof_adr"]])
    else:
      info["qpos_motion_plat"] = jSG(jnp.zeros((6)))
      info["qvel_motion_plat"] = jSG(jnp.zeros((6)))

    # testing: put detailed information in
    info["qacc"] = jSG(data.qacc[self._joint_dict["dof_adr_crane"]])
    info["qfrc_actuator"] = jSG(data.qfrc_actuator[self._joint_dict["dof_adr_crane"]])
    info["qfrc_bias"] = jSG(data.qfrc_bias[self._joint_dict["dof_adr_crane"]])
    info["qfrc_constraint"] = jSG(data.qfrc_constraint[self._joint_dict["dof_adr_crane"]])
    info["qfrc_passive"] = jSG(data.qfrc_passive[self._joint_dict["dof_adr_crane"]])
    info["qfrc_applied"] = jSG(data.qfrc_applied[self._joint_dict["dof_adr_crane"]])
    info["qM"] = jSG(data.qM)

    # jax.debug.print("lengths -> qfrc_actuator={a}, qfrc_bias={b}, qM={c}",
    #                 a=info["qfrc_actuator"].shape,
    #                 b=info["qfrc_bias"].shape,
    #                 c=info["qM"].shape)
    
    info = freeze(info)

    return info

  def add_domain_randomisation_to_info(self, info):
    """
    Add the domain randomised parameters to the info dict, both raw (absolute values)
    and normalised in the range [-1, +1].
    """

    # save the payload mass and scale factors in their raw form
    info["privileged_info"] = jSG(jnp.array([
        self._extra_randomisation_info["payload_mass"],             # 0 -> DO NOT CHANGE THIS ORDERING 
        self._extra_randomisation_info["payload_inertial_length"],  # 1 -> ADD NEW TO THE BOTTOM ONLY    
        self._extra_randomisation_info["cab_mass_scale"],           # 2  
        self._extra_randomisation_info["boom_mass_scale"],          # 3      
        self._extra_randomisation_info["slew_act_scale"],           # 4    
        self._extra_randomisation_info["luff_act_scale"],           # 5    
        self._extra_randomisation_info["hoist_act_scale"],          # 6   
        self._extra_randomisation_info["slew_gear_ratio"],          # 7  
        self._extra_randomisation_info["luff_gear_ratio"],          # 8   
        self._extra_randomisation_info["hoist_gear_ratio"],         # 9    
        self._extra_randomisation_info["slew_armature_scale"],      # 10      
        self._extra_randomisation_info["luff_armature_scale"],      # 11     
        self._extra_randomisation_info["hoist_armature_scale"],     # 12       
        self._extra_randomisation_info["slew_ctrl_scale"],          # 13     
        self._extra_randomisation_info["luff_ctrl_scale"],          # 14
        self._extra_randomisation_info["hoist_ctrl_scale"],         # 15     
    ]))

    # also normalise the privileged information given the limits
    norm_payload_mass = norm_min_max(self._extra_randomisation_info["payload_mass"],
                                     self._config.payload_min_mass,
                                     self._config.payload_max_mass)
    norm_inertial_len = norm_min_max(self._extra_randomisation_info["payload_inertial_length"],
                                     self._config.payload_min_inertial_length,
                                     self._config.payload_max_inertial_length)
    norm_cab_scale = norm_scale(self._extra_randomisation_info["cab_mass_scale"],
                                self._config.mass_max_scaling)
    norm_boom_scale = norm_scale(self._extra_randomisation_info["boom_mass_scale"],
                                 self._config.mass_max_scaling)
    norm_slew_scale = norm_scale(self._extra_randomisation_info["slew_act_scale"],
                                 self._config.actuator_max_scaling)
    norm_luff_scale = norm_scale(self._extra_randomisation_info["luff_act_scale"],
                                 self._config.actuator_max_scaling)
    norm_hoist_scale = norm_scale(self._extra_randomisation_info["hoist_act_scale"],
                                  self._config.actuator_max_scaling)
    norm_slew_gear_ratio = norm_min_max(self._extra_randomisation_info["slew_gear_ratio"],
                                        self._config.min_gear_ratio,
                                        self._config.max_gear_ratio)
    norm_luff_gear_ratio = norm_min_max(self._extra_randomisation_info["luff_gear_ratio"],
                                        self._config.min_gear_ratio,
                                        self._config.max_gear_ratio)
    norm_hoist_gear_ratio = norm_min_max(self._extra_randomisation_info["hoist_gear_ratio"],
                                         self._config.min_gear_ratio,
                                         self._config.max_gear_ratio)
    norm_slew_armature_scale = norm_min_max(self._extra_randomisation_info["slew_armature_scale"],
                                            self._config.min_armature_scale,
                                            self._config.max_armature_scale)
    norm_luff_armature_scale = norm_min_max(self._extra_randomisation_info["luff_armature_scale"],
                                            self._config.min_armature_scale,
                                            self._config.max_armature_scale)
    norm_hoist_armature_scale = norm_min_max(self._extra_randomisation_info["hoist_armature_scale"],
                                             self._config.min_armature_scale,
                                             self._config.max_armature_scale)
    norm_slew_ctrl_scale = norm_min_max(self._extra_randomisation_info["slew_ctrl_scale"],
                                        self._config.min_ctrlrange_scale,
                                        self._config.max_ctrlrange_scale)
    norm_luff_ctrl_scale = norm_min_max(self._extra_randomisation_info["luff_ctrl_scale"],
                                        self._config.min_ctrlrange_scale,
                                        self._config.max_ctrlrange_scale)
    norm_hoist_ctrl_scale = norm_min_max(self._extra_randomisation_info["hoist_ctrl_scale"],
                                         self._config.min_ctrlrange_scale,
                                         self._config.max_ctrlrange_scale)

    # save them also normalised into the space [-1, +1]
    info["privileged_info_normalised"] = jSG(jnp.array([
        norm_payload_mass,          # 0 -> MUST MATCH ORDER ABOVE!!
        norm_inertial_len,          # 1
        norm_cab_scale,             # 2
        norm_boom_scale,            # 3  
        norm_slew_scale,            # 4  
        norm_luff_scale,            # 5  
        norm_hoist_scale,           # 6  
        norm_slew_gear_ratio,       # 7      
        norm_luff_gear_ratio,       # 8      
        norm_hoist_gear_ratio,      # 9        
        norm_slew_armature_scale,   # 10          
        norm_luff_armature_scale,   # 11         
        norm_hoist_armature_scale,  # 12           
        norm_slew_ctrl_scale,       # 13     
        norm_luff_ctrl_scale,       # 14     
        norm_hoist_ctrl_scale,      # 15       
    ]))

    # # debugging: check on information
    # jax.debug.print("cab_scale={c}, max_scale={m}, norm={n}",
    #                 c=self._extra_randomisation_info["cab_mass_scale"],
    #                 m=self._config.mass_max_scaling,
    #                 n=norm_cab_scale)
    # jax.debug.print("slew_scale={c}, max_scale={m}, norm={n}",
    #                 c=self._extra_randomisation_info["slew_act_scale"],
    #                 m=self._config.actuator_max_scaling,
    #                 n=norm_slew_scale)

    return info

  def forward_kinematics(self, mjx_model, qpos, qvel, qpos_mp=None, qvel_mp=None):
    """
    Apply a given qpos to the crane model, and get the resulting sensor readings
    after computing the forward kinematics.
    """

    if self.debug_jit:
      print("forward_kinematics has been called to JIT")

    if self._config.use_motion_platform:
      if qpos_mp != None:
        qpos = jnp.concat([qpos_mp, qpos], axis=-1)
      else:
        qpos = jnp.concat([jnp.zeros(6), qpos], axis=-1)
      if qvel_mp != None:
        qvel = jnp.concat([qvel_mp, qvel], axis=-1)
      else:
        qvel = jnp.concat([jnp.zeros(6), qpos], axis=-1)

    data = mjx_init(mjx_model, qpos=qpos, qvel=qvel)
    info = self._update_sensors(data, {})
    info = self.add_q_values_to_info(data, info)
    info["action"] = jnp.zeros(len(self._key_actuators)) # action is not knowable

    obs = jnp.zeros(())
    reward = jnp.zeros(())
    done = jnp.zeros(())
    metrics = {}

    return mjx_env.State(data, obs, reward, done, metrics, info)

  @staticmethod
  def quat_to_matrix(quat):
    """
    Convert quaternion [w, x, y, z] (mujoco sensor style) to rotation matrix using 
    only JAX operations.
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    xx, xy, xz = x*x, x*y, x*z
    yy, yz, zz = y*y, y*z, z*z
    wx, wy, wz = w*x, w*y, w*z

    return jnp.array([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
    ])

  def vector_frame_to_global(self, vec, frame_pos, frame_quat):
    """
    Transform a vector from the crane frame into the global frame. If not using
    the motion platform then this is simply the identity function.
    """

    # base reference is 1m off the ground, base is 0.285m above reference
    frame_pos = frame_pos.at[2].set(frame_pos.at[2].get() - 0.285)

    mp_rot_matrix = self.quat_to_matrix(frame_quat)

    vec_global = jnp.dot(mp_rot_matrix, vec) + frame_pos

    # avoid any nans which can arise
    vec_global = jnp.nan_to_num(vec_global, nan=0.0)

    return vec_global

  def vector_global_to_frame(self, vec, frame_pos, frame_quat):
    """
    Transform a vector from global frame into the local frame given by frame_pos
    and frame_quat (these defined with respect to the global frame).
    """

    # base reference is 1m off the ground, base is 0.285m above reference
    frame_pos = frame_pos.at[2].set(frame_pos.at[2].get() - 0.285)

    mp_rot_matrix = self.quat_to_matrix(frame_quat)

    translation_vector = vec - frame_pos
    vec_local = jnp.dot(mp_rot_matrix.T, translation_vector)

    # avoid any nans which can arise
    vec_local = jnp.nan_to_num(vec_local, nan=0.0)

    return vec_local

  def payload_pos_from_joint_angles(self, qpos_actuated):
    """
    Compute the payload position relative to the base of the crane (local frame)
    by giving a position for each of the three joint motors. The other joints
    are not given, and are computed simply to ensure that the payload hangs
    vertically.
    """

    qpos = jnp.zeros((self._mj_model.qpos0.shape))
    qvel = jnp.zeros((self._mj_model.nv))

    # apply the given joint angles
    qpos = qpos.at[self._joint_dict["qpos_adr"]].set(qpos_actuated)

    # set the phi string hinge to the negative luff angle, so the 'string' starts straight
    qpos = qpos.at[self._joint_dict["luff-joint"]["qpos_adr"] + 2].set(
        qpos[self._joint_dict["luff-joint"]["qpos_adr"]]
    )

    # apply the initial position and velocity
    data = mjx_init(self.mjx_model, qpos=qpos, qvel=qvel)

    # extract the new payload position
    payload_position = mjx_env.get_sensor_data(self.mj_model, data, "payload_position")

    # get the payload position in local co-ordinates
    if self._config.use_motion_platform:

      payload_global = mjx_env.get_sensor_data(self.mj_model, data, "payload_position")
      base_pos = mjx_env.get_sensor_data(self.mj_model, data, "base_position")
      base_quat = mjx_env.get_sensor_data(self.mj_model, data, "base_orientation")

      payload_position = self.vector_global_to_frame(payload_global,
                                                     frame_pos=base_pos,
                                                     frame_quat=base_quat)
      
    # remove gradient information
    payload_position = jSG(payload_position)

    return payload_position

  def get_global_target(self, info):
    """
    Return the target position in the global frame.
    """
    if self._config.use_motion_platform:
      return self.vector_frame_to_global(info["target_position"],
                                         frame_pos=info["base_position"],
                                         frame_quat=info["base_orientation"])
    else:
      return jSG(info["target_position"].copy())

  # ----- rendering (override underlying mjx_env code) ----- #

  def render(
      self,
      trajectory: List[mjx_env.State],
      height: int = 240,
      width: int = 320,
      camera: Optional[str] = None,
      scene_option: Optional[mujoco.MjvOption] = None,
      modify_scene_fns: Optional[
          Sequence[Callable[[mujoco.MjvScene], None]]
      ] = None,
  ) -> Sequence[np.ndarray]:

    renderer = mujoco.Renderer(self.mj_model, height=height, width=width)
    camera = camera or -1

    def get_image(state, modify_scn_fn=None) -> np.ndarray:
      # convert state to regular mujoco model
      d = mujoco.MjData(self.mj_model)
      d.qpos, d.qvel = state.data.qpos, state.data.qvel
      d.mocap_pos, d.mocap_quat = state.data.mocap_pos, state.data.mocap_quat
      d.xfrc_applied = state.data.xfrc_applied
      # set the mocap marker for the target position
      mid = self._mocap_dict["target"]["mocap_id"]
      if "target_position_global" in state.info:
        d.mocap_pos[mid] = np.array(state.info["target_position_global"])
      # compute the model state and render
      mujoco.mj_forward(self.mj_model, d)
      renderer.update_scene(d, camera=camera, scene_option=scene_option)
      if modify_scn_fn is not None:
        modify_scn_fn(renderer.scene)
      return renderer.render()

    if isinstance(trajectory, list):
      out = []
      for i, state in enumerate(tqdm.tqdm(trajectory)):
        if modify_scene_fns is not None:
          modify_scene_fn = modify_scene_fns[i]
        else:
          modify_scene_fn = None
        out.append(get_image(state, modify_scene_fn))
    else:
      out = get_image(trajectory)

    renderer.close()
    return out

  # ----- class properties ----- #

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

if __name__ == "__main__":

  test_crane_env = False
  if test_crane_env:

    env = Crane()

    key = jrand.key(0)
    rng1, rng2 = jrand.split(key)

    state = env.reset(rng1)
    rand_act = jrand.uniform(rng2, env.n_actions)

    batch_size = 5
    key_batch = jrand.split(rng2, batch_size)

    # randomise the environment
    test_with_jit = False
    if test_with_jit:
      jit_rand = jax.jit(env.domain_randomise)
      batch_envs, in_axes = jit_rand(key_batch)
    else:
      batch_envs, in_axes = env.domain_randomise(key_batch)

    # print("Single env body mass shape:", env.mjx_model.body_mass.shape)
    # print("Batched env body mass shape:", batch_envs.body_mass.shape)
    # print("Batched env body inertia shape:", batch_envs.body_inertia.shape)

    # id = env._body_dict["payload"]["id"]
    # print("Body mass on payload:", env.mjx_model.body_mass[id])
    # print("Randomised batch payload masses:", batch_envs.body_mass[:, id])
    # print("Randomised batch payload inertias:", batch_envs.body_inertia[:, id, :])

    to_print = "Body = {0}, field = {1}. Original value {2} (shape={3})\nBatched value {4} (shape={5})\n"

    # print payload information
    id = env._body_dict["payload"]["id"]
    print(to_print.format(
        "Payload", "body_mass",
        env.mjx_model.body_mass[id], env.mjx_model.body_mass.shape,
        batch_envs.body_mass[:, id], batch_envs.body_mass.shape
    ))
    print(to_print.format(
        "Payload", "body_inertia",
        env.mjx_model.body_inertia[id, :], env.mjx_model.body_inertia.shape,
        batch_envs.body_inertia[:, id, :], batch_envs.body_inertia.shape
    ))

    # print boom information
    id = env._body_dict["boom"]["id"]
    print(to_print.format(
        "Boom", "body_mass",
        env.mjx_model.body_mass[id], env.mjx_model.body_mass.shape,
        batch_envs.body_mass[:, id], batch_envs.body_mass.shape
    ))
    print(to_print.format(
        "Boom", "body_inertia",
        env.mjx_model.body_inertia[id, :], env.mjx_model.body_inertia.shape,
        batch_envs.body_inertia[:, id, :], batch_envs.body_inertia.shape
    ))
    print(to_print.format(
        "Boom", "body_pos",
        env.mjx_model.body_pos[id, :], env.mjx_model.body_pos.shape,
        batch_envs.body_pos[:, id, :], batch_envs.body_pos.shape
    ))
    print(to_print.format(
        "Boom", "body_ipos",
        env.mjx_model.body_ipos[id, :], env.mjx_model.body_ipos.shape,
        batch_envs.body_ipos[:, id, :], batch_envs.body_ipos.shape
    ))

    # print cab information
    id = env._body_dict["cab"]["id"]
    print(to_print.format(
        "cab", "body_mass",
        env.mjx_model.body_mass[id], env.mjx_model.body_mass.shape,
        batch_envs.body_mass[:, id], batch_envs.body_mass.shape
    ))
    print(to_print.format(
        "cab", "body_inertia",
        env.mjx_model.body_inertia[id, :], env.mjx_model.body_inertia.shape,
        batch_envs.body_inertia[:, id, :], batch_envs.body_inertia.shape
    ))
    print(to_print.format(
        "cab", "body_pos",
        env.mjx_model.body_pos[id, :], env.mjx_model.body_pos.shape,
        batch_envs.body_pos[:, id, :], batch_envs.body_pos.shape
    ))
    print(to_print.format(
        "cab", "body_ipos",
        env.mjx_model.body_ipos[id, :], env.mjx_model.body_ipos.shape,
        batch_envs.body_ipos[:, id, :], batch_envs.body_ipos.shape
    ))

    exit()

    n_steps = 0

    for i in range(n_steps):
      print(i+1)
      state = env.step(state, rand_act)

  test_forward_kinematics = False
  if test_forward_kinematics:

    env = Crane()

    angles = np.array([0.2, 0.3, 0.4])
    payload_pos = env.payload_pos_from_joint_angles(angles)
    print(f"Angles {angles} gives position {payload_pos}")

    angles = np.array([-0.2, 0.3, 0.4])
    payload_pos = env.payload_pos_from_joint_angles(angles)
    print(f"Angles {angles} gives position {payload_pos}")

    angles = np.array([0.2, -0.3, 0.4])
    payload_pos = env.payload_pos_from_joint_angles(angles)
    print(f"Angles {angles} gives position {payload_pos}")

    angles = np.array([0.2, 0.3, 0.8])
    payload_pos = env.payload_pos_from_joint_angles(angles)
    print(f"Angles {angles} gives position {payload_pos}")

  test_mjspec = False
  if test_mjspec:

    xml_path = f"{pathhere}/mjcf/crane.xml"

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    spec = mujoco.MjSpec.from_file(xml_path)

    # make ALL the domain randomisation changes at least once for testing
    cab = spec.find_body("cab")
    # cab.first_geom().mass = 123 # only works if FIRST GEOM defines mass=" ". Inertia auto-recalculated, no ability to set manually.
    cab.mass = 1234 # only works if BODIES define <inertial/>.
    # only works if BODIES define <inertial/>. Allows manual choice of inertia.
    cab.inertia = np.array([1, 2, 3])
    cab.pos = np.array([-1, -2, -3])
    cab.ipos = np.array([-1.1, -2.1, -3.1])

    actuator = spec.find_actuator("slew-velocity")
    # print("actuator", actuator.__dict__)
    print(f"actuator gainprm = {actuator.gainprm}")
    print(f"actuator biasprm = {actuator.biasprm}")
    actuator.gainprm[0] *= 1.3
    actuator.biasprm[2] *= 1.3
    actuator.forcerange *= 2.0

    # recompile with the changes
    mj_model2 = spec.compile()

    # print the new xml snippet from spec
    # print(spec.to_xml())

    # # put the old and new models into mjx
    # mjx_model = mjx.put_model(mj_model)
    # mjx_model2 = mjx.put_model(mj_model2)

    # # print the models
    # print("MJX MODEL 1:", mjx_model)
    # print("\n", "-"*50, "\n\n")
    # print("\n", "-"*50, "\n\n")
    # print("\n", "-"*50, "\n\n")
    # print("MJX MODEL 2:", mjx_model2)

    # compare the difference between the two manually
    # https://www.diffchecker.com/text-compare/
    # find all changed vectors, both direct and indirect

  print_spec_changes = True
  if print_spec_changes:

    # --- Important --- #
    # ensure all the domain randomisation settings in config actually cause variation,
    # (i.e., are changed compared to normal), otherwise vectors may be missed

    cfg_override = { "xml_name" : "crane_8ft.xml" }

    crane_normal = Crane(config_overrides=cfg_override)
    crane_randomised = Crane(config_overrides=cfg_override)

    num_envs = 2
    key = jrand.key(0)
    key_batch = jrand.split(key, num_envs)
    mjx_models, vmap_dims, extras = crane_randomised.domain_randomise(rng=key_batch)

    normal_model = crane_normal.mj_model
    rand_model = crane_randomised.last_domain_randomised_model

    # put the old and new models into mjx
    mjx_normal = mjx.put_model(normal_model)
    mjx_rand = mjx.put_model(rand_model)

    # print the models
    print("NORMAL MJX MODEL 1:", mjx_normal)
    print("\n", "-"*50, "\n\n")
    print("\n", "-"*50, "\n\n")
    print("\n", "-"*50, "\n\n")
    print("RANDOMISED MJX MODEL 2:", mjx_rand)

    # compare the difference between the two manually
    # https://www.diffchecker.com/text-compare/
    # find all changed vectors, both direct and indirect