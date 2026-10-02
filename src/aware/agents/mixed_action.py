import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
from aware.utils.training import anneal, AnnealSchedule
import logging; logging.basicConfig(
    level=logging.INFO); pylogging = logging.getLogger(__name__)
import einops


from aware.utils.runs import load_agent
from aware.utils.jax import torch_to_jax, jax_to_torch

# --- randomisation helper functions --- #

def split_rng(keys, num=2):
  return jax.vmap(lambda k: jrand.split(k, num), in_axes=0)(keys)

def get_uniform(rng, min=0.0, max=1.0):
  return jax.vmap(lambda r: jrand.uniform(r, minval=min, maxval=max), in_axes=0)(rng)

def get_gaussian(rng, std=1.0):
  return jax.vmap(lambda r: std * jrand.normal(r), in_axes=0)(rng)

def get_randint(rng, min=0, max=10):
  return jax.vmap(lambda r: jrand.randint(r, (), minval=min, maxval=max + 1), in_axes=0)(rng)

# --- individual classes for operators --- #

class SinOp:

  name = "Sinusoidal"

  def __init__(self, num_envs, num_actions, rng, noise_std=0.0, min_amp=0.0,
               max_amp=1.0, min_freq=100, max_freq=200, start_at_zero=True):
    """
    Operator that makes no motions, but can output random noise.
    """
    self.num_envs = num_envs
    self.num_actions = num_actions
    self.rng = rng
    self.noise_std = noise_std
    self.min_amp = min_amp
    self.max_amp = max_amp
    self.min_freq = min_freq
    self.max_freq = max_freq
    self.start_at_zero = start_at_zero
    self.reset()

  def reset(self, rng=None):
    """
    Re-randomise rng keys and action generating parameters
    """

    # create an initial key batch (overwritten in the loop below)
    rng = self.rng if rng is None else rng
    self.rng, rng_use = jrand.split(rng)
    key_batch = jrand.split(rng_use, self.num_envs)
    
    # create a list, one dictionary per action
    self.sin_op = []
    for i in range(self.num_actions):
      use_keys = split_rng(key_batch, num=6)
      key_batch = use_keys[:, 0] # for next loop
      rand_amps = get_uniform(use_keys[:, 1], min=self.min_amp, max=self.max_amp)
      rand_signs = jnp.where(get_uniform(use_keys[:, 2]) > 0.5,
                              -1 * jnp.ones(self.num_envs), jnp.ones(self.num_envs))
      rand_freq = get_randint(use_keys[:, 3], min=self.min_freq, max=self.max_freq)
      rand_step = get_randint(use_keys[:, 4], min=0, max=self.max_freq)
      self.sin_op.append({
        "amp" : rand_amps * rand_signs,
        "freq" : rand_freq,
        "curr_step" : 0 if self.start_at_zero else rand_step,
        "rng" : use_keys[:, 5],
      })

  def get_action(self, obs):
    """
    An operator which outputs sin wave commands
    """

    # create the empty action vector
    actions = jnp.zeros((self.num_envs, self.num_actions))

    # randomly calculate each action
    for i, d in enumerate(self.sin_op):

      # generate new rng
      use_keys = split_rng(d["rng"], 2)
      d["rng"] = use_keys[:, 0]

      # get the sinusoidal action, based on a step count
      mag = d["amp"] * jnp.sin(2 * jnp.pi * (d["curr_step"] / d["freq"]))
      actions = actions.at[:, i].set(mag + get_gaussian(use_keys[:, 1], std=self.noise_std))
      d["curr_step"] += 1

    actions = jnp.clip(actions, min=-1.0, max=1.0)

    return actions

class NoMotionOp:

  name = "No motion"

  def __init__(self, num_envs, num_actions, rng, noise_std=0.0):
    """
    Operator that makes no motions, but can output random noise.
    """
    self.num_envs = num_envs
    self.num_actions = num_actions
    self.rng = rng
    self.noise_std = noise_std
    self.reset()

  def reset(self, rng=None):
    """
    Re-randomise rng keys and action generating parameters
    """

    # create an initial key batch (overwritten in the loop below)
    self.rng, rng_use = jrand.split(self.rng)
    key_batch = jrand.split(rng_use, self.num_envs)
    
    # create a list, one dictionary per action
    self.no_motion_op = []
    for i in range(self.num_actions):
      use_keys = split_rng(key_batch, num=3)
      key_batch = use_keys[:, 0] # for next loop
      self.no_motion_op.append({
        "rng" : use_keys[:, 1],
      })

  def get_action(self, obs):
      """
      An operator which does nothing and outputs zeros for all control commands
      """

      actions = jnp.zeros((self.num_envs, self.num_actions))

      # randomly calculate each action
      for i, d in enumerate(self.no_motion_op):

        # generate new rng
        use_keys = split_rng(d["rng"], 2)
        d["rng"] = use_keys[:, 0]

        # get the random action
        actions = actions.at[:, i].set(0 + get_gaussian(use_keys[:, 1], 
                                                        std=self.noise_std))

      actions = jnp.clip(actions, min=-1.0, max=1.0)
    
      return actions

class RandOp:

  name = "Random"

  def __init__(self, num_envs, num_actions, rng, noise_std=0.0, min_amp=0.0,
               max_amp=1.0, min_steps_per_action=5, max_steps_per_action=5,
               ramp_mode=False, min_ramp_step=5, max_ramp_step=10):
    """
    Operator that outputs random actions, with optional **state-conditioned directional bias**
    away from joint limits. Works per joint, keeps actions in [-1, 1].
    """
    self.num_envs = num_envs
    self.num_actions = num_actions
    self.rng = rng
    self.noise_std = noise_std
    self.min_amp = min_amp
    self.max_amp = max_amp
    self.min_steps_per_action = min_steps_per_action
    self.max_steps_per_action = max_steps_per_action
    self.ramp_mode = ramp_mode 
    self.min_ramp_step = min_ramp_step
    self.max_ramp_step = max_ramp_step

    self.reset()

  def reset(self, rng=None):
    """
    Re-randomise rng keys and action generating parameters
    """

    # create an initial key batch (overwritten in the loop below)
    self.rng, rng_use = jrand.split(self.rng)
    key_batch = jrand.split(rng_use, self.num_envs)
    
    # create a list, one dictionary per action
    self.rand_op = []
    for i in range(self.num_actions):
      use_keys = split_rng(key_batch, num=4+(self.ramp_mode*4))
      key_batch = use_keys[:, 0] # for next loop
      rand_amps = get_uniform(use_keys[:, 1], min=self.min_amp, max=self.max_amp)
      rand_steps = get_randint(use_keys[:, 2], min=self.min_steps_per_action, 
                               max=self.max_steps_per_action)

      rand_op_settings = {
        "amp" : rand_amps,
        "n_steps" : rand_steps,
        "curr_step" : self.max_steps_per_action * jnp.ones((self.num_envs), dtype=int),
        "last_action" : jnp.zeros((self.num_envs)),
        "rng" : use_keys[:, 3],
      }

      if self.ramp_mode:
        ramp_settings = {
          'ramping': jnp.zeros(self.num_envs, dtype=jnp.bool_),
          'delta': jnp.zeros(self.num_envs),
          'ramp_steps': get_randint(use_keys[:, 4], min=self.min_ramp_step, max=self.max_ramp_step),
          'curr_ramp_step': jnp.zeros(self.num_envs),
          'frozen_next_action': jnp.zeros(self.num_envs),
          'frozen_curr_step': jnp.zeros(self.num_envs)
        }
        rand_op_settings = {**rand_op_settings, **ramp_settings}

      self.rand_op.append(rand_op_settings)

  def get_action(self, obs):
    """
    An operator which outputs fully random actions
    """

    # create the empty action vector
    actions = jnp.zeros((self.num_envs, self.num_actions))

    # randomly calculate each action
    for i, d in enumerate(self.rand_op):

      # generate new rng
      use_keys = split_rng(d["rng"], 4)
      d["rng"] = use_keys[:, 0]
      action_mag = get_uniform(use_keys[:, 1], min=-1, max=1)
      new_n_steps = get_randint(use_keys[:, 2], min=self.min_steps_per_action, max=self.max_steps_per_action)

      # increment steps, and reset in cases the step limit has been reached
      d["curr_step"] += 1
      reached = d["curr_step"] >= d["n_steps"]
      d["curr_step"] = jnp.where(reached, jnp.zeros(d["curr_step"].shape), d["curr_step"])
      d["n_steps"] = jnp.where(reached, new_n_steps, d["n_steps"])

      next_action = jnp.where(reached, action_mag, d["last_action"])

      if self.ramp_mode:
        begin_ramping = jnp.logical_and(reached, jnp.logical_not(d['ramping']))

        # freeze the target we want to ramp to
        d['frozen_next_action'] = jnp.where(begin_ramping, next_action, d['frozen_next_action'])
        d['frozen_curr_step'] = jnp.where(begin_ramping, d["curr_step"], d['frozen_curr_step'])

        d['ramping'] = jnp.where(begin_ramping, True, d['ramping'])
        
        denom = jnp.maximum(d['ramp_steps'], 1)
        delta = (d['frozen_next_action'] - d['last_action']) / denom
        d['delta'] = jnp.where(begin_ramping, delta, d['delta'])

        # advance the ramp 
        d['curr_ramp_step'] = jnp.where(d['ramping'], d['curr_ramp_step'] + 1, d['curr_ramp_step'])
        ramp_action = d['last_action'] + d['delta']

        # if ramping, use the ramped value
        next_action = jnp.where(d['ramping'], ramp_action, next_action)

        # clear the ramp state and reset the ramp step counter
        finished_ramp = d['curr_ramp_step'] >= d['ramp_steps']
        next_action = jnp.where(finished_ramp, d['frozen_next_action'], next_action)
        d['curr_step'] = jnp.where(finished_ramp, d['frozen_curr_step'], d['curr_step'])
        d['ramping'] = jnp.where(finished_ramp, False, d['ramping'])
        d['curr_ramp_step'] = jnp.where(finished_ramp, jnp.zeros_like(d['curr_ramp_step']), d['curr_ramp_step'])

      # set the random action
      actions = actions.at[:, i].set(next_action * d['amp']
                                     + get_gaussian(use_keys[:, 3], std=self.noise_std))
      d["last_action"] = next_action.copy()

    # final hard clip
    actions = jnp.clip(actions, min=-1.0, max=1.0)

    return actions

class RandWalkOp:

  name = "Rand_walk"

  def __init__(self, num_envs, num_actions, rng, theta=0.15, mu=0.0,
               walk_noise_std=0.2, min_amp=0.0, max_amp=1.0):
    self.num_envs = num_envs
    self.num_actions = num_actions
    self.rng = rng
    self.theta = theta
    self.mu = mu
    self.noise_std = walk_noise_std
    self.min_amp = min_amp
    self.max_amp = max_amp
    self.reset()

  def reset(self, rng=None):
    """
    Initialise OU noise state and random amplitudes for each environment and action
    """

    rng = self.rng if rng is None else rng

    self.rng, rng_use = jrand.split(rng)
    key_batch = jrand.split(rng_use, self.num_actions)

    self.ou_state = []
    for i in range(self.num_actions):
      rng = key_batch[i]
      rng, rng_ou, rng_amp = jrand.split(rng, 3)

      # OU process state initialized to mu (typically 0)
      ou_init = self.mu * jnp.ones((self.num_envs,))

      # Random amplitude for each environment
      amp = jrand.uniform(rng_amp, shape=(self.num_envs,),
                          minval=self.min_amp, maxval=self.max_amp)

      self.ou_state.append({
        "state": ou_init,
        "amp": amp,
        "rng": rng_ou
      })

  def get_action(self, obs):
    """
    Generate OU noise per action dimension, scaled by random amplitude
    """
    actions = jnp.zeros((self.num_envs, self.num_actions))
    dt = 1.0

    for i, ou in enumerate(self.ou_state):
      # Update RNG keys
      rng_keys = jrand.split(ou["rng"], 2)
      ou["rng"] = rng_keys[0]

      # Sample Gaussian noise
      noise = jrand.normal(rng_keys[1], (self.num_envs,))

      # OU process update
      prev_state = ou["state"]
      dx = self.theta * (self.mu - prev_state) * dt + self.noise_std * jnp.sqrt(dt) * noise
      new_state = prev_state + dx
      ou["state"] = jnp.clip(new_state, min=-1.0, max=1.0)

      # Apply amplitude scaling
      scaled_noise = new_state * ou["amp"]

      # Clip to [-1, 1] (assuming action bounds)
      actions = actions.at[:, i].set(scaled_noise)
    
    actions = jnp.clip(actions, min=-1.0, max=1.0)

    return actions

class PolicyOp:

  name = "Policy"

  def __init__(self, timestamp, id=None, loaded_policy_args=None):
    """
    Operator that makes no motions, but can output random noise.
    """

    if loaded_policy_args is None: loaded_policy_args = {}

    self.loaded_policy_fn = load_agent(timestamp, id=id, **loaded_policy_args)

    # this could be smoother 
    if hasattr(self.loaded_policy_fn, 'get_policy_action'):
      self.loaded_policy_fn = self.loaded_policy_fn.get_policy_action

    # special behaviour for ROA policy, which requires jax->torch
    elif self.loaded_policy_fn.name == "Agent_ROA_Base":
      self.infer_actions = self.loaded_policy_fn.get_inference_policy()
      self.loaded_policy_fn = self.get_roa_action

    else:
      raise RuntimeError(f"PolicyOp.__init__() error: ",
                         f"loaded policy from timestamp={timestamp} "
                         f"does not have a function 'get_policy_action', "
                         f"and is not an 'Agent_ROA_Base'.")
    
    self.reset()
    self.name = f"{self.name[:]}_{timestamp}"

  def reset(self, rng=None):
    """
    Re-randomise rng keys and action generating parameters
    """

  def get_action(self, obs):
    """
    An operator which outputs fully random actions
    """

    actions = self.loaded_policy_fn(obs)
    return actions

  def get_roa_action(self, obs_jax):
    """
    Get an action from a policy that expects pytorch observations
    """
    obs_torch = jax_to_torch(obs_jax)
    actions_torch = self.infer_actions(obs_torch, hist_encoding=True)
    actions_jax = torch_to_jax(actions_torch)
    return actions_jax

# --- overall agent which combines operators --- #

class MixedOperatorAgent:

  def __init__(self, 
               num_envs, 
               num_actions=3, 
               rngseed=0, 
               noise_std=0.02, 
               operators=None,
               mix_within_envs=False,
               auto_reset_after=None, 
               action_curriculum_len=None, 
               ):
    """
    Return actions in the interval [-1, +1] from a mix of different hardcoded
    operators, including no motion, sinusoid, and random actions.

    """
    logging.info("Initialising MixedOperatorAgent")
    self.num_actions = num_actions
    self.num_envs = num_envs
    self.rngseed = rngseed
    self.rng = jrand.key(rngseed)
    self.noise_std = noise_std
    self.mix_within_envs = mix_within_envs

    # apply default operator settings if not provided
    if operators is None:
      logging.info("No operator information given, using default settings")
      operators = [
        { "name": "sin", "frac": 0.3, "min_amp": 0.0, "max_amp": 1.0, 
          "min_freq": 100, "max_freq": 200, "start_at_zero": True },
        { "name": "rand", "frac": 0.3, "min_amp": 0.0, "max_amp": 1.0,
          "min_steps_per_action": 1, "max_steps_per_action": 10 },
        { "name": "rand_walk", "frac": 0.3, "min_amp": 0.0, "max_amp": 1.0,
          "walk_noise_std": 0.2, "theta": 0.1 },
        { "name": "no_motion", "frac": 0.1 },
      ]

    self.operators = []
    self.operator_fractions = []

    for op in operators:
      if op["name"] == "sin":
        self.operators.append(SinOp(
          num_envs=self.num_envs,
          num_actions=self.num_actions,
          rng=self.rng,
          noise_std=self.noise_std,
          min_amp=op["min_amp"],
          max_amp=op["max_amp"],
          min_freq=op["min_freq"],
          max_freq=op["max_freq"],
          start_at_zero=op["start_at_zero"],
        ))
      elif op["name"] == "rand":
        self.operators.append(RandOp(
          num_envs=self.num_envs,
          num_actions=self.num_actions,
          rng=self.rng,
          noise_std=self.noise_std,
          min_amp=op["min_amp"],
          max_amp=op["max_amp"],
          min_steps_per_action=op["min_steps_per_action"],
          max_steps_per_action=op["max_steps_per_action"],
          ramp_mode=op.get('ramp_mode', False),
          min_ramp_step = op.get('min_ramp_step', None),
          max_ramp_step = op.get('max_ramp_step', None),
        ))
      elif op["name"] == "rand_walk":
        self.operators.append(RandWalkOp(
          num_envs=self.num_envs,
          num_actions=self.num_actions,
          rng=self.rng,
          walk_noise_std=op["walk_noise_std"],
          min_amp=op["min_amp"],
          max_amp=op["max_amp"],
          theta=op["theta"],
        ))
      elif op["name"] == "no_motion":
        self.operators.append(NoMotionOp(
          num_envs=self.num_envs,
          num_actions=self.num_actions,
          rng=self.rng,
          noise_std=self.noise_std,
        ))
      elif op["name"] == "policy":
        self.operators.append(PolicyOp(
          timestamp=op["timestamp"],
          id=op["id"],
          loaded_policy_args=op["loaded_policy_args"],
        ))
      else:
        raise RuntimeError(f"MixedOperatorAgent.__init__() error: "
                           f"operator name = {op['name']} not recognised")
      
      # add the fraction of actions this operator will cover
      self.operator_fractions.append(op["frac"])

    # normalise fractions, in case they don't add up to 1.0
    fracs = np.array(self.operator_fractions)
    self.operator_fractions = fracs / np.sum(fracs)

    # parameters for doing auto-resets for episodes of constant length
    self.auto_reset_after = auto_reset_after
    self.actions_generated = 0

    # can pass in the length of an action curriculum
    if action_curriculum_len is not None:
      self.action_curriculum = AnnealSchedule(imin=0.1, imax=action_curriculum_len)
      self.action_curriculum_scale = anneal(0, self.action_curriculum)
      self.action_curriculum_index = 1
    else: self.action_curriculum = None

    self.reset()

    # log the types of operators and their fractions
    log_str = """MixedOperatorAgent fractions and settings:\n"""
    for i in range(len(self.operators)):
      log_str += f""" -> {self.operators[i].name} = {self.operator_fractions[i]:.3f}\n"""
    log_str += f""" -> mix_within_envs = {self.mix_within_envs}\n"""
    log_str += f""" -> noise_std = {self.noise_std}\n"""
    log_str += f""" -> auto_reset_after = {self.auto_reset_after}\n"""
    log_str += f""" -> action_curriculum_len = {action_curriculum_len}\n"""
    logging.info(log_str)

  def __call__(self, obs):
    return self.get_action(obs)
    
  def reset(self, key=None):
    """
    Reset all of the operators into an initial state. Generate new masks based
    on the provided operator fractions
    """

    # reset steps for auto-reset calling
    self.actions_generated = 0

    # track resets for curriculums
    if self.action_curriculum is not None:
      self.action_curriculum_scale = anneal(self.action_curriculum_index,
                                            self.action_curriculum)
      self.action_curriculum_index += 1

    # get fresh rng
    rng = self.rng if key is None else key

    # reset all the operators
    for i in range(len(self.operators)):
      self.rng, rng_use = jrand.split(rng)
      self.operators[i].reset(rng_use)
    
    self.rng, rng_use = jrand.split(rng)
    # normalise fraction unit vector (in case it doesn't add to one)
    fracs = np.array(self.operator_fractions)
    fracs = fracs / np.sum(fracs)

    # assign an integer to each operator, and divide by the fractions
    ops = jnp.arange(len(self.operators))
    if self.mix_within_envs:
      op_indicies = jrand.choice(rng_use, ops, shape=(self.num_envs, self.num_actions), 
                                 p=fracs)
    else:
      op_indicies = jrand.choice(rng_use, ops, shape=(self.num_envs,), p=fracs)
      op_indicies = einops.repeat(op_indicies, "b -> b a", a=self.num_actions)

    # generate masks for each operator action over all num_envs
    self.operator_masks = []
    for i in range(len(self.operators)):
      self.operator_masks.append(op_indicies == i)

  def get_action(self, obs):
    """
    Get actions randomly from the operators
    """

    # keep track of actions if doing auto-reset after certain num steps
    if self.auto_reset_after is not None:
      if self.actions_generated > self.auto_reset_after:
        self.reset()
    self.actions_generated += 1

    actions = jnp.zeros((self.num_envs, self.num_actions))

    # add each operators actions (subject to masking, so no overlaps)
    for i in range(len(self.operators)):
      op_actions = self.operators[i].get_action(obs)
      actions += self.operator_masks[i] * op_actions

    # apply any curriculum on overall action size
    if self.action_curriculum is not None:
      actions *= self.action_curriculum_scale
    
    return actions
