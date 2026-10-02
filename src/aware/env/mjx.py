import os
from mujoco_playground import registry
import jax
import jax.numpy as jnp
import jax.random as jrand
import random
from datetime import datetime
import time
import mediapy as media
import logging; logging.basicConfig(
    level=logging.INFO); logger = logging.getLogger(__name__)
from copy import deepcopy
import mujoco
import einops

from aware.env.crane.crane_env import Crane

# from: https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/wrapper_torch.py
def _jax_to_torch(tensor):
  from jax._src.dlpack import to_dlpack  # pylint: disable=import-outside-toplevel
  # pytype: disable=import-error # pylint: disable=import-outside-toplevel
  import torch.utils.dlpack as tpack

  tensor = to_dlpack(tensor)
  tensor = tpack.from_dlpack(tensor)
  return tensor

def _torch_to_jax(tensor):
  from jax._src.dlpack import from_dlpack  # pylint: disable=import-outside-toplevel
  # pytype: disable=import-error # pylint: disable=import-outside-toplevel
  import torch.utils.dlpack as tpack

  tensor = tpack.to_dlpack(tensor)
  tensor = from_dlpack(tensor)
  return tensor

class MJXEnv:

  def __init__(self, name, num_envs=None, rngseed=None, torch=False,
               enable_rendering=False, env_args=None, domain_randomise=False,
               render_all_environments=False, debug_nans=True, rerandomise_rate=None,
               structured_randomisation=False, structured_repeats=None,
               cfg_override_groups=None, debug_jit=False, fk_only=False,
               **kwargs):
    """
    Wrapper for mujoco playground environments. Adds some convienience by handling
    its own state and random seeds. Uses jax underneath, but set torch=True to convert
    outputs to and inputs from torch tensors. Supports vectorised environments.
    """

    # # for debugging: log recompilations to the terminal
    # jax.config.update("jax_log_compiles", True)

    # handle inputs
    self.name = name
    self.enable_rendering = enable_rendering
    self.num_envs = num_envs
    self.states_to_render = []
    self.torch = torch
    self.domain_randomise = domain_randomise
    self.rerandomise_rate = rerandomise_rate if not structured_randomisation else None
    self.structured_randomisation = structured_randomisation
    self.structured_repeats = structured_repeats # only used if structured_randomisation=True
    # only used if structured_randomisation=True
    self.cfg_override_groups = cfg_override_groups
    self.steps_since_last_rerandomise = 0 # only used for rerandomise rate
    self.debug_nans = debug_nans
    self.render_all_environments = render_all_environments
    self.debug_jit = debug_jit
    self.seed(rngseed)
    self.fk_over_domain_randomisation = False # exclude DR from forward kinematics

    logger.info(f"MJXEnv() is loading an environment:\n"
                f"  -> name = {name}\n"
                f"  -> num_envs = {num_envs}\n"
                f"  -> seed = {rngseed}\n"
                f"  -> pytorch_mode = {torch}\n"
                f"  -> domain_randomise = {domain_randomise}\n"
                f"  -> structured_randomisation = {structured_randomisation}\n"
                f"  -> rerandomise_rate = {rerandomise_rate}\n")
    t0 = time.process_time()

    if name.lower() == "crane":
      self.env = Crane(config_overrides=env_args, debug_jit=self.debug_jit)
      self.env_config = self.env._config
    else:
      self.env = registry.load(self.name, config_overrides=env_args)
      self.env_config = self.env._config

    if fk_only:
      # only using this class for forward kinematics, ignore step() and reset()
      self.fk_over_domain_randomisation = False # must be false, or we should run DR
      logger.info(
        f"Finished initialising {self.name} environment for forward_kinematics ONLY"
        f", after {time.process_time() - t0:.1f} seconds")
      return

    # jit and vmap the key environment functions
    if self.domain_randomise:
      # create a random set of environments (with vmap and jit)
      self.randomise(structured=structured_randomisation,
                     structured_repeats=structured_repeats,
                     cfg_override_groups=cfg_override_groups,
                     force_recompile=True) # jit here
    else:
      self.jit_reset = jax.jit(jax.vmap(self.env.reset))
      self.jit_step = jax.jit(jax.vmap(self.env.step))

    # prepare environment
    self.reset()

    # extract key information
    self.obs_dim = self.state.obs.shape[-1] # this is populated by reset above
    self.fps = 1.0 / self.env.dt

    # for crane, override action size to exclude motion platform
    if name.lower() == "crane":
      self.act_dim = len(self.env._key_actuators)
    else:
      self.act_dim = self.env.action_size

    logger.info(
        f"Finished initialising {self.name} environment, after {time.process_time() - t0:.1f} seconds")
    logger.debug(f"Environment settings:\n{self.get_params_dict()}")

  # ----- public functions ----- #

  def step(self, action: jnp.array):
    """
    Input action should be a jnp.array if self.torch=False, or a torch
    tensor if self.torch=True.

    Also auto-resets 

    state is a dataclass:
      - data: mjx.Data (mujoco data structure, not including mjModel)
      - obs: Observation (Union[jax.Array, Mapping[str, jax.Array]])
      - reward. jax.Array (float32)
      - done: jax.Array (float32)
      - metrics: Dict[str, jax.Array]
      - info: Dict[str, Any]
    """
    if self.torch: action = _torch_to_jax(action)

    self.state = self.jit_step(self.state, action)

    self.step_count += 1 # count for individual environments, for auto reset
    self.steps_since_last_rerandomise += 1 # for rerandomise rate only

    if self.enable_rendering:

      # if we want to render every environment
      if self.render_all_environments:
        render_state = [jax.tree.map(lambda x: x[i], self.state)
                        for i in range(self.num_envs)]
      else:
        render_state = jax.tree.map(lambda x: x[0], self.state)

      self.states_to_render.append(render_state)

    self.truncation = jnp.where(self.step_count >= self.env_config["episode_length"], 
                                jnp.array(True),
                                jnp.array(False))

    if self.debug_nans: self._debug_nans_in_state(self.state)

    # the output dictionary should concatenate both state.info and state.metrics
    out_info = {**self.state.info, **self.state.metrics, "truncation" : self.truncation}

    if self.torch: # return everything as torch tensors
      return (
          _jax_to_torch(self.state.obs),          # observation
          _jax_to_torch(self.state.reward),       # reward
          _jax_to_torch(self.state.done),         # terminal
          _jax_to_torch(out_info["truncation"]),  # truncated
          out_info,                               # info dict
      )

    else: # normal jax

      return (
          self.state.obs,                 # observation
          self.state.reward,              # reward
          self.state.done,                # terminal
          out_info["truncation"],         # truncated
          out_info,                       # info dict
      )

  def reset(self, reset_mask=None, key_batch=None, qpos=None, qvel=None,
            force_rerandomise=False):
    """
    Reset the environment. Pass in a reset mask to determine which environments
    are reset. Pass in a batch of keys to control the randomness.
    """

    if key_batch is not None:
      if key_batch.shape[0] != self.num_envs and qpos is None and qvel is None:
        raise RuntimeError(f"MJXEnv.reset() error: key_batch passed with shape {key_batch.shape}"
                           f", this does not match the number of environments = {self.num_envs}")

    # --- prepare for reset, handle inputs --- #

    # check if there are actually any states to reset
    if reset_mask is None or jnp.any(reset_mask):

      self.key, local_key = jrand.split(self.key)

      # batch keys, unless given a specific key batch
      if key_batch is None:
        if self.num_envs != None:
          batched_key = jrand.split(local_key, self.num_envs)
        else:
          batched_key = jnp.expand_dims(local_key, 0)
      else:
        batched_key = key_batch

      # check if we should re-randomise
      if ((self.domain_randomise and self.rerandomise_rate is not None)
          or force_rerandomise):
        full_eps_done = (self.steps_since_last_rerandomise) / \
            self.env_config['episode_length']
        if (full_eps_done > self.rerandomise_rate - 1e-6
            or force_rerandomise):
          # force randomisation and recompiling (note this is slow, ~3min per 1000 envs)
          logger.info(f"At reset(), with mask.any(), MJXEnv has completed {full_eps_done:.1f}"
                      f" equivalent full episodes ({self.steps_since_last_rerandomise} steps)"
                      f", exceeding rerandomise_rate = {self.rerandomise_rate}. Re-randomising now.")
          self.randomise(force_recompile=True, structured=False)
          self.steps_since_last_rerandomise = 0

      # --- reset the environment --- #

      if qpos is None and qvel is None:
        # normal case, reset to a random position
        reset_state = self.jit_reset(batched_key)
      else:
        # special case, reset to a given position
        if qpos is None:
          raise RuntimeError(f"MJXEnv.reset() error: qpos must be provided to reset() if setting qvel")
        if qvel is None:
          raise RuntimeError(f"MJXEnv.reset() error: qvel must be provided to reset() if setting qpos")
        # handle the case where inputs are not batched
        if len(qpos.shape) == 1:
          qpos = einops.repeat(qpos, "n -> b n", b=batched_key.shape[0])
          qvel = einops.repeat(qvel, "n -> b n", b=batched_key.shape[0])
        # handle jit
        if not hasattr(self, "jit_reset_state") or force_rerandomise:
          logger.info(f"qpos and qvel given in reset(), now jit-ing reset_state")
          if self.domain_randomise:
            self.jit_reset_state = jax.jit(self._reset_state_wrapper)
          else:
            self.jit_reset_state = jax.jit(jax.vmap(self.env.reset_to_state))
        # finally, apply the reset with given qpos and qvel
        reset_state = self.jit_reset_state(batched_key, qpos, qvel)

      # --- handle selective resets and additional fields --- #

      # add the truncated field to the state info
      reset_state_truncation = jnp.zeros((self.num_envs), dtype=bool)

      # reset all states
      if reset_mask is None or jnp.all(reset_mask):
        self.state = reset_state
        self.truncation = reset_state_truncation
        self.step_count = jnp.zeros((self.num_envs,), dtype=jnp.int16)

      # only reset some states
      else:
        def apply_reset(new, old):
          """
          Dynamically reshape reset_mask for correct broadcasting. For elements of
          state with 1 dimension we need reset_mask.shape = (num_envs, 1), for
          two dimensions we need (num_envs, 1, 1) etc
          """
          expanded_mask = jnp.reshape(
              reset_mask, (reset_mask.shape[0],) + (1,) * (new.ndim - 1))
          return jnp.where(expanded_mask, new, old)

        self.state = jax.tree.map(apply_reset, reset_state, self.state)

        self.truncation = jnp.where(
            reset_mask, False, reset_state_truncation)
        self.step_count = jnp.where(reset_mask, 0, self.step_count)

    if self.debug_nans: self._debug_nans_in_state(self.state)

    # --- return reset information --- #

    # the output dictionary should concatenate both state.info and state.metrics
    out_info = {**self.state.info, **self.state.metrics}

    # the output dictionary should concatenate both state.info and state.metrics
    out_info = {}
    out_info.update(self.state.info)
    out_info.update(self.state.metrics)

    if self.torch:
      return _jax_to_torch(self.state.obs), out_info
    else: 
      return self.state.obs, out_info

  def seed(self, rngseed=None):

    if not hasattr(self, "rngseed"): self.rngseed = rngseed

    if rngseed is None:
      if self.rngseed is not None: rngseed = self.rngseed
      else: rngseed = random.randint(0, 2_147_483_647)

    # save the initial seed, then create the jax prng key
    self.rngseed = rngseed
    self.key = jrand.key(self.rngseed)

  def randomise(self, force_recompile=False, structured=None, structured_repeats=None,
                cfg_override_groups=None):
    """
    Apply domain randomisation and generate a new batch of randomised environments.

    If force_recompile is true, the underlying environments will be re-randomised,
    by rerunning jit. However, if it is false, this function has no effect after
    the first time it is called.

    structured=False means parameters are randomised based on a random seed.
    structured=True means parameters are linearly interpolated from their min to max values.
    """

    logger.info(f"Applying domain randomisation now to {self.num_envs} environments")
    t0 = time.process_time()

    if not hasattr(self.env, "domain_randomise"):
      logger.error(
          f"Underlying environment (name={self.name}) does NOT have a 'domain_randomise' function.")
      raise RuntimeError(
          f"Failed to add domain randomisation, it is not supported by env={self.name}")

    if not hasattr(self, "randomise_fn_calls"):
      self.randomise_fn_calls = 1
    else:
      self.randomise_fn_calls += 1

    if self.randomise_fn_calls > 1 and not force_recompile:
      logger.warning(
          "Environments already randomised once. They are NOT re-randomised unless force_recompile=True. Environments have not been re-randomised.")
      return

    if structured:
      key_batch=None
    else:
      self.key, rng = jrand.split(self.key)
      key_batch = jrand.split(rng, self.num_envs)

    self.env_vmap, self.env_in_axes, self.env_extra = (
        self.env.domain_randomise(rng=key_batch, num_envs=self.num_envs, structured=structured,
                                  structured_repeats=structured_repeats,
                                  cfg_override_groups=cfg_override_groups))

    # need to recompile to force the new environments into compiled cache
    if force_recompile:
      if not self.debug_jit:
        self.jit_reset = jax.jit(self._reset_wrapper)
        self.jit_step = jax.jit(self._step_wrapper)
      else: # debug jit
        t0 = time.process_time()
        print("--- Starting to jit _reset_wrapper now ---")
        self.jit_reset = jax.jit(self._reset_wrapper)
        a = jrand.key(0)
        b = jrand.split(a, self.num_envs)
        x = self.jit_reset(b)
        jax.block_until_ready(x)
        t1 = time.process_time()
        print(f"--- Finish jit of _reset_wrapper after {t1-t0:.1f} seconds. ---")
        print("--- Starting to jit _step_wrapper now ---")
        self.jit_step = jax.jit(self._step_wrapper)
        c = jnp.zeros((self.num_envs, len(self.env._key_actuators)), dtype=jnp.float32)
        y = self.jit_step(x, c)
        jax.block_until_ready(y)
        t2 = time.process_time()
        print(f"--- Finish jit of _step_wrapper after {t2-t1:.1f} seconds. ---")
        print(f"Total time for jit: {t2-t0:.1f} seconds.")

    # # for debugging
    # self.print_domain_randomisation()

    t1 = time.process_time()
    logger.info(
        f"Finished creating {self.num_envs} domain randomised environments after {t1-t0:.1f} seconds")

  def close(self):
    return

  def render(self, height=480, width=480, camera_name=None, num_frames=None,
             finish_rendering=True):
    """
    Render the current episode up until now. A buffer of states is initialised as
    empty when reset() is called, and each call to step() adds a new state. This
    function passes that list of states to mujoco's renderer, and returns the frames.
    The camera_name can be set if desired to use a specific camera, otherwise the
    default mujoco view will be used.
    """

    if not self.enable_rendering:
      logger.info("MJXEnv.render() warning: self.enable_rendering=False, nothing rendered. Call MJXEnv.render_start() when you want to start saving frames.")
      return None
    elif len(self.states_to_render) == 0:
      logger.info(
          "MJXEnv.render() warning: self.states_to_render is empty, render must be called after step(). Nothing rendered")
      return None

    if num_frames == None: num_frames = len(self.states_to_render)

    logger.info(f"MJXEnv.render() is preparing to render {num_frames} frames")

    if self.render_all_environments:

      state_history = [[] for _ in range(self.num_envs)]
      for i in self.states_to_render:
        for j in range(self.num_envs):
          state_history[j].append(i[j])

      frames = []
      for i in state_history:
        frames.append(self.env.render(i, height=height, width=width, camera=camera_name))

    else:
      frames = self.env.render(
          self.states_to_render[:num_frames], height=height, width=width, camera=camera_name)

    if finish_rendering: self.render_end()

    return frames

  def render_start(self):
    """
    Indicate that future states should be saved for the purpose of rendering.
    """
    self.enable_rendering = True
    self.states_to_render = []

  def render_end(self):
    """
    Indicate that rendering is finished, and future states should not be saved.
    """
    self.enable_rendering = False
    self.states_to_render = []

  def save_render(self, savedir, savename="mujoco", add_timestamp=True):
    """
    Call render(), and then save the frames as a video at the given location.
    """
    if not self.enable_rendering:
      logger.warning(
          "MJXEnv.save_render() warning: self.enable_rendering=False, nothing saved.")
      return
    elif len(self.states_to_render) == 0:
      logger.warning(
          "MJXEnv.save_render() warning: self.states_to_render is empty, render must be called after step(). Nothing saved.")
      return

    frames = self.render()
    dt = self.env.dt

    if not os.path.exists(savedir):
      os.makedirs(savedir)

    if add_timestamp:
      savename = f"{savename}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    savepath = f"{savedir}/{savename}.mp4"

    logger.info(f"MJXEnv.save_render() is saving a video at path: {savepath}")

    media.write_video(savepath, frames, fps=1.0/dt)

    return

  def print_domain_randomisation(self, inds=None, print_params=True):
    """
    Get the key domain randomised parameters from the batch of environments, returns dict for logging 
    """

    if not hasattr(self, "env_vmap"):
      logger.warning(
          "MJXEnv.print_domain_randomisation() warning: self.env_vmap does NOT exist, no domain randomisation detected")
      return

    if inds == None:
      inds = list(range(self.num_envs))

    pay_id = self.env._body_dict["payload"]["id"]
    cab_id = self.env._body_dict["cab"]["id"]
    boom_id = self.env._body_dict["boom"]["id"]
    payload_mass = self.env_vmap.body_mass[inds, pay_id]
    cab_mass = self.env_vmap.body_mass[inds, cab_id]
    boom_mass = self.env_vmap.body_mass[inds, boom_id]

    slew_id = self.env._actuator_dict["slew-velocity"]["id"]
    luff_id = self.env._actuator_dict["luff-velocity"]["id"]
    hoist_id = self.env._actuator_dict["hoist-velocity"]["id"]
    slew_gain = self.env_vmap.actuator_gainprm[inds, slew_id, 0]
    luff_gain = self.env_vmap.actuator_gainprm[inds, luff_id, 0]
    hoist_gain = self.env_vmap.actuator_gainprm[inds, hoist_id, 0]

    dr_info = {}
    print("MJXEnv: Getting selected domain randomisation information:")
    for i, envid in enumerate(inds):

      if print_params:
        to_print = f"""Env {envid}:\n"""
        for key in self.env_extra:
          to_print += f" -> {key} = {self.env_extra[key][envid]}\n"
        print(to_print, end="", flush=True)

        # print(f"Env {envid}:\n"
        #       f" -> Masses: payload={payload_mass[i]:.3f}, cab={cab_mass[i]:.3f}, boom={boom_mass[i]:.3f}\n"
        #       f" -> Gains: slew={slew_gain[i]:.3f}, luff={luff_gain[i]:.3f}, hoist={hoist_gain[i]:.1f}\n"
        #       , end="", flush=True)

      dr_info[f'env_{envid}'] = {'mass':{'payload':payload_mass[i],
                                         'cab':cab_mass[i],
                                         'boom':boom_mass[i]},
                                 'gains': {'slew':slew_gain[i],
                                           'luff':luff_gain[i],
                                           'hoist':hoist_gain[i]}}
    return dr_info

  def get_params_dict(self):
    return {
        # class parameters
        "name" : self.name,
        "enable_rendering" : self.enable_rendering,
        "num_envs" : self.num_envs,
        "torch" : self.torch,
        "domain_randomise" : self.domain_randomise,
        "rerandomise_rate" : self.rerandomise_rate,
        "steps_since_last_rerandomise" : self.steps_since_last_rerandomise,
        "debug_nans" : self.debug_nans,
        "render_all_environments" : self.render_all_environments,
        "rngseed" : self.rngseed,
        "key" : self.key,
        # environment information
        "act_dim" : self.act_dim,
        "obs_dim" : self.obs_dim,
        "env_type" : "mujoco-playground",
        "env_config" : self.env_config,
    }

  def get_save_state(self):
    return {
        # class parameters
        "name" : self.name,
        "enable_rendering" : self.enable_rendering,
        "num_envs" : self.num_envs,
        "torch" : self.torch,
        "domain_randomise" : self.domain_randomise,
        "rerandomise_rate" : self.rerandomise_rate,
        "steps_since_last_rerandomise" : self.steps_since_last_rerandomise,
        "debug_nans" : self.debug_nans,
        "render_all_environments" : self.render_all_environments,
        "rngseed" : self.rngseed,
        "key" : self.key,
        # environment information
        "act_dim" : self.act_dim,
        "obs_dim" : self.obs_dim,
        "env_type" : "mujoco-playground",
        "env_config" : self.env_config,
    }

  def load_save_state(self, state_dict):
    """
    Load the environment give a save state
    """

    logger.info("MJXEnv() is being loaded now")
    t0 = time.process_time()

    # load the class variables from the given dictionary
    self.name = state_dict["name"]
    self.enable_rendering = state_dict["enable_rendering"]
    self.num_envs = state_dict["num_envs"]
    self.torch = state_dict["torch"]
    self.domain_randomise = state_dict["domain_randomise"]
    self.rerandomise_rate = state_dict["rerandomise_rate"]
    self.steps_since_last_rerandomise = state_dict["steps_since_last_rerandomise"]
    self.debug_nans = state_dict["debug_nans"]
    self.render_all_environments = state_dict["render_all_environments"]
    self.rngseed = state_dict["rngseed"]
    self.key = state_dict["key"]
    self.act_dim = state_dict["act_dim"]
    self.obs_dim = state_dict["obs_dim"]
    self.env_config = state_dict["env_config"]

    # now remake the environment, given the config
    if name.lower() == "crane":
      self.env = Crane(config_overrides=self.env_config)
      self.env_config = self.env._config
    else:
      self.env = registry.load(self.name, self.env_config)
      self.env_config = self.env._config

    # recompile the jitted step and reset, optionally randomise too
    if self.domain_randomise:
      self.randomise(force_recompile=True) # includes jax.jit
    else:
      self.jit_reset = jax.jit(jax.vmap(self.env.reset))
      self.jit_step = jax.jit(jax.vmap(self.env.step))

    self.reset()

    t1 = time.process_time()
    logger.info(f"MJXEnv() has finished loading after {t1-t0:.1f} seconds.")
    logger.debug(f"Environment settings:\n{self.get_params_dict()}")

  def forward_kinematics(self, qpos, qvel, qpos_mp=None, qvel_mp=None):
    """
    Run forward kinematics on the crane on each domain randomised environment.
    qpos should be a batched to match num_envs.
    """
    if not hasattr(self, "jit_fk"):
      self.jit_fk = jax.jit(self._forward_kinematics_wrapper)

    fk_state = self.jit_fk(qpos, qvel, qpos_mp, qvel_mp)

    return fk_state

  # ----- private functions ----- #

  def _env_fn(self, mjx_model, extras=None):
    """
    Function required for domain randomisation, creates a new env with which
    the mjx_model can be set to any choice. This pattern makes it compatible
    with JAX
    """
    original_spec = self.env.mj_spec
    self.env.mj_spec = None # can't pickle MjSpec object, fails in deepcopy()
    t0 = time.process_time()
    env = deepcopy(self.env) # deepcopy required to avoid JAX tracer leak
    t1 = time.process_time()
    env._mjx_model = mjx_model
    env._extra_randomisation_info = extras
    env.mj_spec = mujoco.MjSpec.from_file(env._xml_path) # recreate mjspec
    self.env.mj_spec = original_spec # restore the original to self.env
    return env

  def _reset_wrapper(self, rng):
    """
    Wrapper required for domain randomisation, to make it compatible with JAX,
    but not require passing the environment explicitly into reset.
    """
    def reset_inner(mjx_model, rng, extras=None):
      if self.debug_jit:
        t0 = time.process_time()
      env = self._env_fn(mjx_model, extras=extras)
      if self.debug_jit:
        t1 = time.process_time()
        print(f"Time to trace self._env_fn {t1-t0:.3f}")
      outputs = env.reset(rng)
      if self.debug_jit:
        t2 = time.process_time()
        print(f"Time to trace env.reset {t2-t1:.3f}")
      return outputs
    
    if self.debug_jit:
      t0 = time.process_time()
      print("_reset_wrapper() called to JIT")

    state = jax.vmap(reset_inner, in_axes=[self.env_in_axes, 0, 0])(
        self.env_vmap, rng, self.env_extra
    )

    if self.debug_jit:
      t1 = time.process_time()
      print(f"Time to trace reset_inner {t1-t0:.3f}")

    return state
  
  def _reset_state_wrapper(self, rng, qpos, qvel):
    """
    Wrapper required for domain randomisation, to make it compatible with JAX,
    but not require passing the environment explicitly into reset.
    """
    def reset_state_inner(mjx_model, rng, qpos, qvel, extras=None):
      if self.debug_jit:
        t0 = time.process_time()
      env = self._env_fn(mjx_model, extras=extras)
      if self.debug_jit:
        t1 = time.process_time()
        print(f"Time to trace self._env_fn {t1-t0:.3f}")
      outputs = env.reset_to_state(rng, qpos, qvel)
      if self.debug_jit:
        t2 = time.process_time()
        print(f"Time to trace env.reset {t2-t1:.3f}")
      return outputs
    
    if self.debug_jit:
      t0 = time.process_time()
      print("_reset_wrapper() called to JIT")

    state = jax.vmap(reset_state_inner, in_axes=[self.env_in_axes, 0, 0, 0, 0])(
        self.env_vmap, rng, qpos, qvel, self.env_extra
    )

    if self.debug_jit:
      t1 = time.process_time()
      print(f"Time to trace reset_inner {t1-t0:.3f}")

    return state

  def _step_wrapper(self, state, action):
    """
    Wrapper required for domain randomisation, to make it compatible with JAX,
    but not require passing the environment explicitly into reset.
    """

    def step_inner(mjx_model, state, action, extras=None):
      if self.debug_jit:
        t0 = time.process_time()
      env = self._env_fn(mjx_model, extras=extras)
      if self.debug_jit:
        t1 = time.process_time()
        print(f"Time to trace self._env_fn {t1-t0:.3f}")
      outputs = env.step(state, action)
      if self.debug_jit:
        t2 = time.process_time()
        print(f"Time to trace env.step {t2-t1:.3f}")
      return outputs
    
    if self.debug_jit:
      t0 = time.process_time()
      print("_step_wrapper() called to JIT")

    res = jax.vmap(step_inner, in_axes=[self.env_in_axes, 0, 0, 0])(
        self.env_vmap, state, action, self.env_extra
    )

    if self.debug_jit:
      t1 = time.process_time()
      print(f"Time to trace step_inner {t1-t0:.3f}")

    return res
  
  def _forward_kinematics_wrapper(self, qpos, qvel, qpos_mp=None, qvel_mp=None):
    """
    Wrapper for passing forward kinematics through all the domain randomised
    environments.
    """
    def fk_inner(mjx_model, qpos, qvel, qpos_mp, qvel_mp):
      return self.env.forward_kinematics(mjx_model, qpos, qvel, qpos_mp, qvel_mp)
    
    if qpos_mp == None:
      vmap_dims = [0, 0, None, None]
    else:
      vmap_dims = [0, 0, 0, 0]
    
    if self.fk_over_domain_randomisation:
      res = jax.vmap(fk_inner, in_axes=[self.env_in_axes, *vmap_dims])(
        self.env_vmap, qpos, qvel, qpos_mp, qvel_mp
      )
    else:
      res = jax.vmap(fk_inner, in_axes=[None, *vmap_dims])(
        self.env._mjx_model, qpos, qvel, qpos_mp, qvel_mp
      )

    return res

  def _debug_nans_in_state(self, state):
    """
    Check for nans in the state, these would occur from crane_env
    """

    if jax.numpy.isnan(self.state.reward).any():
      nan_mask = jax.numpy.isnan(self.state.reward)
      nan_count = jnp.sum(nan_mask.astype(int))
      nan_envs = jnp.nonzero(nan_mask)
      logger.warning(
          f"NaN found in reward output of crane_env.step(), for {nan_count} environments.\nEnv nums={nan_envs}.")
      done_envs = list(jnp.nonzero(self.state.done))
      logger.info(f"Environments where done is true: {done_envs}")

    if jax.numpy.isnan(self.state.obs).any():
      nan_mask = jax.numpy.isnan(self.state.obs)
      nan_count = jnp.sum(nan_mask.astype(int))
      env_nans = jnp.any(nan_mask, axis=1)
      num_envs = jnp.sum(env_nans.astype(int))
      nan_envs = jnp.nonzero(env_nans)
      logger.warning(
          f"NaN found in observation output of crane_env.step(), inside {num_envs} environments, with total count of {nan_count} nans.\nEnv nums={nan_envs}.")
      done_envs = jnp.nonzero(self.state.done)
      logger.info(f"Environments where done is true: {done_envs}")
      if self.env_config["use_roa"]:
        self._check_obs_nans_roa(self.state.obs) # function to check observation in detail

  def _check_obs_nans_roa(self, obs):
    """
    Checks an ROA observation for nans
    """

    a = self.env._config["roa_n_proprioceptive"]
    b = a + self.env._config["roa_n_scan"]
    c = b + self.env._config["roa_n_private_explicit"]
    d = c + self.env._config["roa_n_privileged_info"]
    n = int((a - 3) / 2) # assume obs_prop=[qpos, qvel, target_pos]

    obs_dict = {
        "qpos" : obs[:, :n],
        "qvel" : obs[:, n:2*n],
        "target_pos" : obs[:, 2*n:a],
        "scan" : obs[:, a:b],
        "private_explicit" : obs[:, b:c],
        "privileged_info" : obs[:, c:d],
        "past_timesteps" : obs[:, d:],
    }

    for key, value in obs_dict.items():
      nan_mask = jnp.isnan(value)
      if nan_mask.any():
        nan_count = jnp.sum(nan_mask.astype(int))
        env_nans = jnp.any(nan_mask, axis=1)
        num_envs = jnp.sum(env_nans.astype(int))
        nan_envs = jnp.nonzero(env_nans)[0]
        print(
            f"NaN found in observation, within {key}, inside {num_envs} environments ({nan_envs}), with total count {nan_count}")
        print(f" -> {key} vector from env {nan_envs[0]} is: {value[nan_envs[0]]}")

if __name__ == "__main__":

  test_crane = False
  if test_crane:

    key = jax.random.key(0)
    num_envs = 4

    env = MJXEnv("Crane", torch=False, num_envs=num_envs)

    first_state = env.reset()

    key, rng = jax.random.split(key, 2)
    action = jax.random.uniform(rng, (num_envs, env.act_dim))

    next_state = env.step(action)

    num_steps = 6

    for i in range(num_steps):

      key, rng = jax.random.split(key, 2)
      action = jax.random.uniform(rng, (num_envs, env.act_dim))

      obs, reward, t1, t2, info = env.step(action)

      reset_mask = jnp.logical_or(t1, t2)

      # # uncomment to test reset behaviour
      # if i % 2 == 0:
      #   reset_mask = reset_mask.at[0].set(1)
      #   reset_mask = reset_mask.at[-1].set(1)
      # if i % 10 == 0:
      #   reset_mask = jnp.ones(reset_mask.shape)
      # print("Reset mask is:", reset_mask)

      env.reset(reset_mask)

      # print(env.state)
      print(f"Step {i + 1} complete")

    reset_mask = jnp.zeros(num_envs)
    reset_mask = reset_mask.at[0].set(1)
    reset_mask = reset_mask.at[-1].set(1)

    env.reset(reset_mask)

  wobble_crane = False
  if wobble_crane:

    key = jax.random.key(0)
    num_envs = 2

    env = MJXEnv("Crane", torch=False, num_envs=num_envs)

    first_state = env.reset()

    key, rng = jax.random.split(key, 2)
    action = jax.random.uniform(rng, (num_envs, env.act_dim))

    next_state = env.step(action)

    num_steps = 400

    env.render_start()

    for i in range(num_steps):

      # key, rng = jax.random.split(key, 2)
      # action = jax.random.uniform(rng, (num_envs, env.act_dim))

      first_state = jax.tree.map(lambda x: x[0], env.state)

      action = []
      for j in range(env.act_dim):
        action.append(
            jnp.sin(
                env.state.data.time[0] * 2 * jnp.pi * \
                0.5 + j * 2 * jnp.pi / env.env.action_size
            )
        )
      action = jnp.array([action for i in range(num_envs)])
      if i < 101:
        action = -jnp.ones((num_envs, env.act_dim))
        action = action.at[:, 2].set(1)
      else:
        action = jnp.ones((num_envs, env.act_dim))
        action = action.at[:, 2].set(-1)
      # print("Action:", action)

      obs, reward, t1, t2, info = env.step(action)
      reset_mask = jnp.logical_or(t1, t2)
      # env.reset(reset_mask)
      if i % 20 == 0:
        print(
            f"Step {i + 1} complete. Payload position: {first_state.info['payload_position']}. (Action = {action})")
        first_state = jax.tree.map(lambda x: x[0], env.state)
        # print(f"Joint angles: {first_state.data.qpos}")
        # print(f"Payload position: {first_state.info['payload_position']}")
        # print(f"Target position: {first_state.info['target_position']}")

    env.save_render(savedir="media")

  test_domain_randomisation = True
  if test_domain_randomisation:

    key = jax.random.key(1)
    num_envs = 5
    env = MJXEnv("Crane", torch=False, num_envs=num_envs, domain_randomise=True)

    first_state = env.reset()
    key, rng = jax.random.split(key, 2)
    action = jax.random.uniform(rng, (num_envs, env.act_dim))
    next_state = env.step(action)

    num_steps = 3

    for i in range(num_steps):

      print(f"\n\n\n----- STEP {i + 1} -----")

      key, rng = jax.random.split(key, 2)
      action = jax.random.uniform(rng, (num_envs, env.act_dim))

      obs, reward, t1, t2, info = env.step(action)

      reset_mask = jnp.logical_or(t1, t2)
      if i % 2 == 0:
        reset_mask = reset_mask.at[0].set(1)
        reset_mask = reset_mask.at[-1].set(1)

      env.reset(reset_mask)
      env.randomise(force_recompile=True)
      env.print_domain_randomisation(print_params=True)

  benchmark_speeds = False
  if benchmark_speeds:

    key = jax.random.key(0)
    num_envs = 100
    repeat_num = 1000
    domain_randomise = True

    num_steps = repeat_num
    num_resets = repeat_num

    timer = time.process_time

    # env_names = ["Crane", "WalkerWalk", "HopperHop", "HumanoidWalk", "CartpoleBalance"]
    env_names = ["Crane"]

    head_str = f"{'Name':<12} | {'Init/s':<10} | {'->Step2/s':<10} | {'Per step/s':<10} | {'Reset/s':<10}\n"
    rows_str = "{0:<12} | {1:<10.3f} | {2:<10.3f} | {3:<10.3f} | {4:<10.3f}\n"
    print(head_str)
    table_str = f"""\nBenchmark of environment speeds, with num_envs = {num_envs}, num_step = {num_steps}, num_reset = {num_resets}\n\n"""
    table_str += head_str

    for name in env_names:

      t0 = timer()

      env = MJXEnv(name, num_envs=num_envs, domain_randomise=domain_randomise)

      jax.block_until_ready(env.state)
      t1 = timer()
      print(f"Finished initialisation after {t1 - t0:.3f} seconds.")

      first_state = env.reset()
      first_state = env.reset(jnp.ones(num_envs).at[0].set(0)) # check both permutations

      key, rng = jax.random.split(key, 2)
      action = jax.random.uniform(rng, (num_envs, env.act_dim))
      next_state = env.step(action)
      next_state = env.step(action)

      jax.block_until_ready(env.state)
      t2 = timer()
      print(f"Reached step 2 after {t2 - t1:.3f} seconds.")

      for i in range(num_steps):

        key, rng = jax.random.split(key, 2)
        action = jax.random.uniform(rng, (num_envs, env.act_dim))
        obs, reward, done, trunc, info = env.step(action)

      jax.block_until_ready(env.state)
      t3 = timer()
      print(f"Finished {num_steps} steps after {t3 - t2:.3f} seconds.")

      for i in range(num_resets):
        reset_mask = jnp.ones(num_envs)
        reset_mask = reset_mask.at[0].set(0)
        env.reset(reset_mask)

      # env.state[0].obs.block_until_ready()
      jax.block_until_ready(env.state)
      t4 = timer()
      print(f"Finished {num_resets} resets after {t4 - t3:.3f} seconds.")

      this_row = rows_str.format(name, t1-t0, t2-t1, (t3-t2) / \
                                 num_steps, (t4-t3) / num_resets)
      table_str += this_row
      print(rows_str.format(name, t1-t0, t2-t1, (t3-t2) / num_steps, (t4-t3) / num_resets))

    print("\n" + table_str)

    # num_randomise = 10
    # trand0 = timer()
    # for i in range(num_randomise):
    #   env.randomise()
    # jax.block_until_ready(env.env_vmap)
    # trand1 = timer()
    # print(f"Time taken for {num_randomise} randomisations = {trand1 - trand0:.3f} seconds, per randomisation: {(trand1 - trand0) / num_randomise:.3f} s")