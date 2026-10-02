"""
RSSM using ROA for motion prediction
"""
import math

import einops
import elements
import aware.agents.embodied as embodied
import aware.agents.embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import jax.random as jrand
import ninjax as nj
import numpy as np
import optax
import re
import time
import random
import logging; logging.basicConfig(
    level=logging.INFO); pylogging = logging.getLogger(__name__)
import jax.profiler 

from omegaconf import DictConfig, OmegaConf
from typing import Dict
from jax import Array
import torch

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3

# @jax.custom_vjp
# def gradient_scaler(x, scale_factor):
#     return x

# def gradient_scaler_fwd(x, scale_factor):
#     # Return the input (x) and save the scale_factor for the backward pass
#     return x, scale_factor

# def gradient_scaler_bwd(res, g):
#     scale_factor = res
#     # Multiply the incoming gradient (g) by the scale factor
#     return g * scale_factor, None  # None is for the second argument (scale_factor)

class RSSM_ROA:

  """
  RSSM x ROA motion predictor
  """

  name ='rssm_roa'

  def __init__(self, 
               # core settings
               agent, 
               obs_dim, # encoder obs dim
               act_dim, # action obs dim
               dec_dim, # decoder obs dim
               privileged_info_dim,
               batch_length,
               loss_scales,
               seed=None, 
               submodule=False, 
               payload_prediction_mode=False,
               # RSSM training settings 
               online_training=False,
               replay_context=1, 
               act_mode=None, 
               obs_mode=None, 
               payload_model=None, # model for predicting the payload from joint angles 
               fk_regularizing=False, # regularize the predictions using the FK model
               # ROA settings
               use_privileged_info=True,
               roa_settings=None,
               # inference settings
               num_hist_timesteps_to_use=100, 
               imagination_horizon=50):
    
    if isinstance(agent, dict):
      agent = DictConfig(agent)

    available = jax.devices()
    elements.print(f'JAX devices ({jax.device_count()}):', available)
    
    self.seed = random.getrandbits(16) if seed is None else seed
    self.seed_inc = elements.Counter()
    self.batch_length = batch_length
    self.replay_context = replay_context
    self.online_training = online_training

    self.obs_dim = obs_dim
    self.dec_dim = obs_dim if dec_dim is None else dec_dim
    self.asym_dec = dec_dim is not None # set dec_dim to be 3 for assym
    self.target_obs = jnp.array(range(self.obs_dim))
    self.act_dim = act_dim
    self.privileged_info_dim = privileged_info_dim

    self.payload_prediction_mode = payload_prediction_mode
    if self.payload_prediction_mode and not self.asym_dec:
      raise ValueError("Assymmetrical decoder required for payload prediction mode!")

    logging.info(f"asym_dec: {self.asym_dec}, decoder output_size = {self.dec_dim}")

    self.obs_mode = obs_mode
    self.act_mode = act_mode

    if self.obs_mode == 'only_actuator':
      self.obs_dim = 6
      self.target_obs = jnp.array([0,1,4,7,8,11])
      logging.info(f"Obs mode set to only actuator, obs_dim = {self.obs_dim}")
    elif self.obs_mode == 'only_pos' or self.obs_mode == 'only_pos_full_recon':
      self.obs_dim = 7 
      self.target_obs = jnp.array([0,1,2,3,4,5,6])
      logging.info(f"Obs mode set to only position, obs_dim = {self.obs_dim}")
    elif self.obs_mode == 'only_slew':
      self.obs_dim = 1
      self.target_obs = jnp.array([0])
      logging.info(f"Obs mode set to only slew, obs_dim = {self.obs_dim}")
    elif self.obs_mode == 'only_luff':
      self.obs_dim = 1
      self.target_obs = jnp.array([1])
      logging.info(f"Obs mode set to only luff, obs_dim = {self.obs_dim}")
    elif self.obs_mode == 'no_double_pendulum':
      self.obs_dim = 10
      self.target_obs = jnp.array([0,1,2,3,4,7,8,9,10,11])
      logging.info(f"Obs mode set to no_double_pendulum, obs_dim = {self.obs_dim}")
    elif self.obs_mode == 'acceleration_model':
      self.obs_dim = 10
      self.target_obs = jnp.array([0,1,2,3,4,7,8,9,10,11])
    else:
      logging.info(f"Obs mode set to default, obs_dim = {self.obs_dim}")

    # training settings

    # state masking settings
    self.state_masking = agent.state_masking.enable
    self.state_masking_only_start = agent.state_masking.only_start
    self.state_masking_p = agent.state_masking.p
    logging.info(f"state masking settings: {self.state_masking}")
    
    # forward kinematics regularizing settings
    self.fk_regularizing = fk_regularizing
    self.payload_model = payload_model

    # closed loop training 
    autoregressive_cfg = agent.get('autoregressive_training', None)
    if autoregressive_cfg is not None:

      self.autoregressive_training = autoregressive_cfg.get('enable', False) # enable closed loop training
      self.autoregressive_training_scale = autoregressive_cfg.get('scale', 1.0) # scale for the loss
      self.autoregressive_training_step = autoregressive_cfg.get('step', 50000) # step to start training
      self.autoregressive_num_steps = autoregressive_cfg.get('num_steps', 20) # number of steps into future for training
      self.autoregressive_only_last = autoregressive_cfg.get('only_last', False) # only use the last step for los
      self.one_batch_autoregressive = autoregressive_cfg.get('one_batch', False) # train across the entire batch
      self.full_autoregressive_training = autoregressive_cfg.get('full', False) # only do autoregressive training

      # split closed loop training (use closed loop for actuated and open loop for free joints)
      self.split_loss_training = autoregressive_cfg.get('split', False)
      self.split_loss_training_step = autoregressive_cfg.get('split_step', 10000)

    else:
      self.autoregressive_training = False
      self.autoregressive_training_scale = 0.0
      self.autoregressive_training_step = 0.0
      self.full_autoregressive_training = False

      self.split_loss_training = False
      self.split_loss_training_step = 0.0

    # manage the scales with closed loop training
    if self.autoregressive_training:
      
      # split the loss between the free and actuated joints
      if self.split_loss_training:  
        # pop unused loss scales 
        loss_scales.pop('autoregressive', None)
        loss_scales.pop('vector_pos', None)
        loss_scales.pop('vector_vel', None)
        # add new loss scales (schedule will be controlled separately)
        loss_scales['free_joints'] = 0.5
        loss_scales['actuated_joints'] = 0.5

      # make sure autoregressive is a part of the loss scales
      else:
        if 'autoregressive' not in loss_scales:
          loss_scales['autoregressive'] = self.autoregressive_training_scale

    # remove autoregressive from loss scales if open loop training is disabled
    else:
      loss_scales.pop('autoregressive', None)

    if self.fk_regularizing and self.payload_model is None:
      logging.warning("RSSM_ROA: fk_regularizing enabled but no payload model passed, fk_regularizing will be disabled!")
      self.fk_regularizing = False
    
    # inference settings 
    self.num_hist_timesteps_to_use = num_hist_timesteps_to_use
    self.num_joint_angles = 7 # hardcoded for the crane
    self.add_next_action =  True
    self.add_latent_vector = False
    self.num_actions = self.act_dim
    self.integral_position = False
    self.imagination_horizon = imagination_horizon
    self.get_prediction_from_trajectory = self.sample_predictions_from_trajectory
    # self.get_prediction_from_trajectory = self.get_prediction_from_trajectory_main

    ## -- ROA SETUP -- ##

    self.use_privileged_info = use_privileged_info # includes in the obs space
    self.add_privileged_info = self.use_privileged_info # for compatibility with AgentPredictionDiscriminator
    roa_regex = ""

    roa_settings = roa_settings or {}
  
    if isinstance(roa_settings, DictConfig):
      roa_settings = OmegaConf.to_container(roa_settings, resolve=True)
    roa_config = elements.Config(roa_settings)

    self.enable_roa = roa_config.get('enable', True)

    # backwards compatibility, moved from a string 'roa_mode' to a boolean 'enable_roa'
    # this catches the older cases with roa_mode
    roa_mode = roa_config.get('roa_mode', 'classic')
    if roa_mode == 'disabled':
      self.enable_roa = False

    if roa_config.get('roa_mode', 'disabled') != 'disabled' and not self.use_privileged_info:
      raise ValueError("use_privileged_info is disabled but roa is enabled!")

    roa_modules = []

    self.roa_encoder = None
    self.roa_decoder = None
    self.roa_estimator = None
    self.latent_concat = None
    

    if self.enable_roa:
      
      self.privileged_latent_dim = roa_config.privileged_latent_dim # size of latent representation of privileged information
      self.latent_concat = roa_config.latent_concat # where to attach the latent privileged info, i.e. to observations or actions
      
      ## estimator settings ##

      # regularizer settings -> important
      self.anneal_roa_reg_step = roa_config.anneal_reg_step # step to begin annealing the reg loss
      self.anneal_roa_reg_value = roa_config.anneal_reg_value # value to set loss to

      # estimator loss
      self.estimator_huber_delta = roa_config.get('estimator_huber_delta', 1.0)
      self.estimator_transition_weight = roa_config.get('estimator_transition_weight', 1.0)
      self.estimator_burn_in = roa_config.get('estimator_burn_in', 10)

      # estimator confidence 
      self.use_estimator_confidence = roa_config.get('use_estimator_confidence', False)
      self.predictor_gets_confidence = roa_config.get('predictor_gets_confidence', False)
      self.estimator_confidence_per_feature = roa_config.get('estimator_confidence_per_feature', False)
      self.confidence_log_var_clamping_min = roa_config.get('confidence_log_var_clamping_min', 0.0)
      self.confidence_log_var_clamping_max =  roa_config.get('confidence_log_var_clamping_max', 1.0)

      # train with estimated latents
      self.train_with_estimated_latents = roa_config.get('train_with_estimated_latents', False) # train with the estimated latents
      self.train_with_estimated_latents_max_p = roa_config.get('train_with_estimated_latents_max_p', 0.0) # maximum probability that we will use estimated latents
      self.train_with_estimated_latents_max_step = roa_config.get('train_with_estimated_latents_max_step', 0.0) # step which we reach maximum probability 

      ## define the spaces ##

      if self.use_estimator_confidence:
        if self.estimator_confidence_per_feature: 
          self.confidence_dim = self.privileged_latent_dim
        else:
          self.confidence_dim = 1

      else:
        self.confidence_dim = 0

      priv_vector_space = {
        "priv_vector": elements.Space(np.float32, (self.privileged_info_dim,))
      }
      priv_latent_space = { # confidence will be removed from priv latent space if predictor does not get confidence 
        "priv_latent": elements.Space(np.float32, (self.privileged_latent_dim + (self.confidence_dim * self.predictor_gets_confidence)))
      }
      
      # encoder
      if roa_config.enc == 'mlp_enc':
        self.roa_encoder = ROA_MLP_Encoder(priv_space=priv_vector_space, 
                                           output_dim=self.privileged_latent_dim, 
                                           confidence_dim = self.confidence_dim*self.predictor_gets_confidence, # should be 0 if predictor does not get confidence
                                           output_confidence_min=self.confidence_log_var_clamping_min,
                                           **roa_config.mlp_enc_cfg, name="roa_enc")
        roa_regex += "|roa_enc"
      elif roa_config.enc == 'identity':
        self.roa_encoder = lambda x, *args, **kwargs : x['priv_vector'] 
      else:
        raise ValueError(f"roa encoder {roa_config.enc} not recognised!")
      roa_modules += [self.roa_encoder] if roa_config.enc != "identity" else []

      # estimator 
      if roa_config.estimator == 'cnn_estimator':
        self.roa_estimator = ROA_CNN_Estimator(privileged_latent_dim=self.privileged_latent_dim+self.confidence_dim, 
                                                **roa_config.cnn_estimator_cfg, name="roa_est")
        roa_regex += "|roa_est"
      elif roa_config.estimator == 'large_cnn_estimator':
        self.roa_estimator = ROA_LargeCNN_Estimator(privileged_latent_dim=self.privileged_latent_dim+self.confidence_dim, 
                                                **roa_config.large_cnn_estimator_cfg, name="roa_est")
        roa_regex += "|roa_est"
      elif roa_config.estimator == 'rnn_estimator':
        self.roa_estimator = ROA_RNN_Estimator(privileged_latent_dim=self.privileged_latent_dim+self.confidence_dim, 
                                                gru_cfg=roa_config.rnn_estimator_cfg.gru_cfg, name="roa_est")
        roa_regex += "|roa_est"
      elif roa_config.estimator == 'latent_head':
        self.roa_estimator = embodied.jax.MLPHead(space=priv_latent_space['priv_latent'], **roa_config.latent_head_cfg, name='roa_latent_head')
        roa_regex += "|roa_latent_head"
      elif roa_config.estimator == 'rssm_estimator':
        self.roa_estimator = ROA_RSSM_Estimator(roa_config.rssm_estimator_cfg, name='roa_est')
        roa_regex += "|roa_est"
      else:
        raise ValueError(f"roa estimator {roa_config.estimator} not recognised!")
      roa_modules += [self.roa_estimator]

      # decoder
      if roa_config.decoder == "priv_head":
          # self.roa_decoder = embodied.jax.MLPHead(space={'output':priv_vector_space['priv_vector']}, **roa_config.priv_head_cfg, name='roa_priv_head')
          self.roa_decoder = ROA_Decoder_Head(space={'output':priv_vector_space['priv_vector']}, priv_head_cfg=roa_config.priv_head_cfg, name='roa_priv_head')
          roa_regex += "|roa_priv_head"
      elif roa_config.decoder == "mlp_dec":
          self.roa_decoder = ROA_MLP_Decoder(priv_vector_space, **roa_config.mlp_dec_cfg, name='roa_decoder')
          roa_regex += "|roa_decoder"
      elif roa_config.decoder is not None:
        raise ValueError(f"roa decoder {roa_config.decoder} not recognised!")
      roa_modules += [self.roa_decoder] if roa_config.decoder is not None else []

      # experiments with ROA
      self.roa_rssm_timestep_weighting = roa_config.timestep_weighting
      self.roa_rssm_warmup_length = roa_config.warmup_length
      self.roa_decode_all_t = roa_config.get('decode_all_t', False)

      logging.info(f"Experimental settings:\n")
      logging.info(f"roa_decode_all_t: {self.roa_decode_all_t}")

    self.roa_config = roa_config # for future reference

    self.use_bayesian_set_encoder = roa_config.get('use_bayesian_set_encoder', False)
    assert not(self.use_bayesian_set_encoder and self.enable_roa), f"bayesian_set_encoder={self.use_bayesian_set_encoder} and {self.enable_roa}"

    if self.use_bayesian_set_encoder:
      assert self.use_privileged_info, "set_encoder enabled but use_privileged info disabled!"

      self.privileged_latent_dim = roa_config.privileged_latent_dim 
      self.use_estimator_confidence = False
      self.predictor_gets_confidence = False
      priv_latent_space = { # confidence will be removed from priv latent space if predictor does not get confidence 
        "priv_latent": elements.Space(np.float32, (self.privileged_latent_dim))
      }

      set_encoder_cfg = roa_config.get("set_encoder", {})
      set_encoder_hidden_units = set_encoder_cfg.get("hidden_units", [128, 128])
      self.set_encoder = BayesianSetEncoder(input_dim=(2*self.obs_dim + self.act_dim),
                                            lod=self.privileged_latent_dim,
                                            hidden_units=set_encoder_hidden_units,
                                            name='set_enc')
      roa_regex += "|set_enc"
      roa_modules.append(self.set_encoder)

    logging.info(f"roa components loaded: {roa_modules}")
              
    ## RSSM SETUP ##

    if isinstance(agent, DictConfig):
      agent = OmegaConf.to_container(agent, resolve=True)
    
    agent_config = elements.Config(agent)

    obs_space = {
        "vector": elements.Space(np.float32, (self.obs_dim,)),
        "is_first": elements.Space(bool),
        "is_last": elements.Space(bool),
      }
    
    if self.use_privileged_info and not self.use_bayesian_set_encoder: 
      obs_space = obs_space | priv_vector_space 

    if self.fk_regularizing:
      obs_space = obs_space | {"payload_position": elements.Space(np.float32, (3,))}

    if self.payload_prediction_mode:
      obs_space = obs_space | {"decoder_target": elements.Space(np.float32, (3,))}

    act_space = {
       "action": elements.Space(np.float32, (self.act_dim,), low=-1., high=1.)
    }

    if (self.enable_roa or self.use_bayesian_set_encoder) and self.latent_concat == 'act':
      act_space = act_space | priv_latent_space

    self.obs_space = {k: v for k, v in obs_space.items() if not k.startswith('log/')}
    self.act_space = {k: v for k, v in act_space.items() if k != 'reset'}
    
    exclude = ('is_first', 'is_last', 'is_terminal', 'reward', 'priv_vector', 'decoder_target', 'payload_position')
    enc_space = {k: v for k, v in self.obs_space.items() if k not in exclude}
    dec_space = { 
        "vector": elements.Space(np.float32, (self.dec_dim,)) # custom decoder space
    }

    if self.enable_roa and self.latent_concat == 'obs':
        enc_space = enc_space | priv_latent_space 
        dec_space = dec_space | priv_latent_space 

    self.enc = Encoder(enc_space, **agent_config.enc.simple, name='enc')
    self.dec = Decoder(dec_space, **agent_config.dec.simple, name='dec')
    self.dyn = RSSM(self.act_space, **agent_config.dyn.rssm, name='dyn')

    self.enc_symlog = self.enc.symlog
    if self.enc_symlog:
      self.output_transform = lambda x: nn.symexp(x)
    else:
      self.output_transform = lambda x: x

    ## TRAINING + OPTIMIZER SETUP ##

    self.modules = [self.dyn, self.enc, self.dec] + roa_modules
    self.opt = embodied.jax.Optimizer(
      self.modules, self._make_opt(**agent_config.opt), summary_depth=1, name='opt')
    
    # no ROA -> drop all ROA related scales
    if not self.enable_roa:
      self.scales = {k:v for k,v in loss_scales.items() if 'roa' not in k}
    # ROA with no decoder -> drop all ROA decoding scales
    elif self.roa_decoder is None:
      self.scales = {k:v for k,v in loss_scales.items() if 'roa_priv_recon' not in k}
    # ROA with decoder -> keep everyhing
    else:
      self.scales = {k:v for k,v in loss_scales.items()}

    # no need for regularistion if we are using estimator confidence
    if self.enable_roa and self.use_estimator_confidence:
      self.scales.pop('roa_reg', 0)
    
    # remove fk_reg
    if not self.fk_regularizing:
      self.scales = {k:v for k,v in self.scales.items() if k not in ['fk_reg']}

    if self.payload_prediction_mode:
      self.scales.pop("vector_vel", None)
      self.scales["vector"] = self.scales.pop("vector_pos")

    # check if we are enabling autoregressive RWM loss training 
    rwm_loss_cfg = agent.get("rwm_loss", None)
    if rwm_loss_cfg is not None:
      # reassign loss function to rwm
      self.rwm_loss = rwm_loss_cfg.get('enable', False)
      self.rwm_warmup = rwm_loss_cfg.get("rwm_warmup", 32)
      self.rwm_rollout = rwm_loss_cfg.get("rwm_rollout", 8)
      self.rwm_window_size = self.rwm_warmup + self.rwm_rollout
    else:
      self.rwm_loss = False
    
    if self.rwm_loss:
      # clean scales
      self.scales = {k:v for k,v in self.scales.items() if k in ['vector', 'dyn', 'rep'] or 'roa' in k}
      assert 'vector' in self.scales, "RWM loss requires 'vector' to be in loss scales"


    logging.info(f"Using loss scale factors:\n{self.scales}")

    self.spaces = {**self.obs_space, **self.act_space, **self.ext_space}
    
    ## NINJAX SETUP ##
    # this is not needed if RSSM is a submodule of another ninjax module  
    if not submodule: 
      self._train_jit = jax.jit(nj.pure(self._train))   
      self._forward_jit = jax.jit(nj.pure(self._forward), static_argnames=('single', 'decode'))
      self._imagine_forward_jit = jax.jit(nj.pure(self._imagine_forward), static_argnames='length')

      if self.enable_roa:
        self._roa_encode_jit = jax.jit(nj.pure(self.roa_encoder))
        self._roa_estimate_jit= jax.jit(nj.pure(self.roa_estimator))
        self._roa_decode_jit = jax.jit(nj.pure(self.roa_decoder), static_argnames='training')
      
      if self.use_bayesian_set_encoder:
        self._set_encoder_jit = jax.jit(nj.pure(self.set_encoder))
  
      dummy_state = {}
      dummy_batch_size = 128
      self.model_params = {}

      # base_params = nj.init(self.init_carry, static_argnums=1)(dummy_state, dummy_batch_size, seed=0)
      carry, prevact = self.init_carry(dummy_batch_size)

      if self.online_training:
        data = self._zeros(self.spaces, (dummy_batch_size, (self.batch_length + self.replay_context)))
      else:
        data = self._zeros(self.spaces, (dummy_batch_size, (self.batch_length)))
        data.pop('priv_latent', None) # this key is not needed
      
      self.params = nj.init(self._train)(dummy_state, carry, prevact, data, seed=0)
      model_regex = "^(enc|dyn|dec" + roa_regex + ")/"
      pattern = re.compile(model_regex)
      self.model_keys = [k for k in self.params.keys() if pattern.search(k)]
      self.model_params = {
                k: self.params[k].copy() for k in self.model_keys
      }

    logging.info(f"RSSM SETTINGS:\n"
                  f" -> action space: {self.act_space}\n"
                  f" -> observation space: {self.obs_space}\n"
                  f" -> encoder space: {enc_space}\n"
                  f" -> decoder space: {dec_space}\n"
                  f" -> rssm space: {self.act_space}\n"
                  f" --------------------------\n"
                  f" -> closed loop training: {self.autoregressive_training}\n"
                  f" -> closed loop training step: {self.autoregressive_training_step}\n"
                  f" -> closed loop training split: {self.split_loss_training}\n"
                  f" -> closed loop training split step: {self.split_loss_training_step}\n"
                  f" --------------------------\n"
                  f" -> scales: {self.scales}"
                  f" --------------------------\n"
                  f" -> use roa: {self.enable_roa}\n"
                  f" -> use bayesian set encoder: {self.use_bayesian_set_encoder}")

    if self.enable_roa:
      logging.info(f"ROA SETTINGS:\n"
                   f"-> use_estimator_confidence: {self.use_estimator_confidence} \n"
                   f"-> predictor_gets_confidence: {self.predictor_gets_confidence} \n"
                   f"-> estimator_confidence_per_feature: {self.estimator_confidence_per_feature} \n"
                   f"-> confidence_log_var_clamping_min: {self.confidence_log_var_clamping_min} \n"
                   f"-> confidence_log_var_clamping_max: {self.confidence_log_var_clamping_max} \n"
                   f"-> train_with_estimated_latents: {self.train_with_estimated_latents} \n"
                   f"-> train_with_estimated_latents_max_p: {self.train_with_estimated_latents_max_p} \n"
                   f"-> train_with_estimated_latents_max_step: {self.train_with_estimated_latents_max_step} \n")
    
  def __getattribute__(self, name):
    if name == "get_latent_estimator" and not (object.__getattribute__(self, 'enable_roa') or object.__getattribute__(self, 'use_bayesian_set_encoder')):
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'get_latent_estimator'")
    elif name == "get_latent_decoder" and object.__getattribute__(self, 'roa_decoder') is None:
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'get_latent_decoder'")
    else:
      return object.__getattribute__(self, name)

  @property
  def ext_space(self):
    """
    Extra bits in the parameter space
    """

    # add the vector noise, action noise, etc..
    spaces = {}
    spaces['vector_noise'] = elements.Space(np.float32, self.obs_dim)
    spaces['priv_vector_noise'] = elements.Space(np.float32, self.privileged_info_dim)
    spaces['priv_vector_stds'] = elements.Space(np.float32, self.privileged_info_dim)

    # spaces = {}
    # spaces['consec'] = elements.Space(np.int32)
    # spaces['stepid'] = elements.Space(np.uint8, 20)
    # if self.replay_context:
    #   spaces.update(elements.tree.flatdict(dict(
    #       enc=self.enc.entry_space,
    #       dyn=self.dyn.entry_space,
    #       dec=self.dec.entry_space)))
    return spaces
  
  # -- PUBLIC METHODS -- #

  # - core model functionality - #
  def forward(self, carry, obs, prevact, single=False, decode=True):
    """
    Move forward the latent state using real obs

    Returns:
      carry (enc, dyn, dec) -> new model latent state
      recon -> obs reconstructed (None when decode=False)
    """
    _, (carry, recon, feat) =  self._forward_jit(self.model_params, carry, obs, prevact, single=False, decode=decode, seed=self._seed())
    return carry, recon, feat
  
  def roa_encode(self, obs, **kwargs):
    """
    Computes the privileged latent directly from the privileged information. 
    """
    assert 'priv_vector' in obs, f"priv_vector not in obs keys: {obs.keys()}"
    _, (priv_latent) = self._roa_encode_jit(self.model_params, obs, seed=self._seed()) 
    return priv_latent  

  def roa_decode(self, priv_vector=None, observation=None, priv_latent=None, priv_head_mode='last', **kwargs):

    """
    Generates decodings of ROA latents.
    This function is highly general and can generate decodings in many different modes:

      1. Decode a precomputed latent passed through as 'priv_latent'
      2. Generate a new latent using the privileged information pass through in 'priv_latent'
      3. Generate a new latent using the privileged information passed through in the 'obs' array
      4. Generate a new latent using the obs array and the estimator
    """

    if self.roa_decoder is None:
      raise RuntimeError("No ROA decoder is defined!")
    
    # load obs for estimator or/and rssm decoder head
    if observation is not None:
      data = self.dreamer_format_array(observation)
      if len(data) != 2:
        raise ValueError("act missing from the observation array, i.e. obs length < obs_dim + act_dim") 
      obs_dict, act_dict = data  
      if 'priv_vector' in obs_dict:
        priv_vector = obs_dict['priv_vector']  
    # load precomputed priv latent to decode
    if priv_latent is not None:
        latents = priv_latent
    # or compute latent by encoding the priv_vector 
    elif priv_vector is not None:
      jnp_priv_vector = jnp.array(priv_vector)
      if jnp_priv_vector.ndim == 2:
        jnp_priv_vector = jnp.expand_dims(jnp_priv_vector, axis=1) # add T
      priv_dict = {'priv_vector': jnp_priv_vector}
      _, (latents) = self._roa_encode_jit(self.model_params, priv_dict, seed=self._seed())
      latents = latents[:, -1]
    # or estimate the latent using the obs + actions and passing through estimator network
    else:
      if observation is None:
        raise ValueError("obs input required to generate latent as priv_vector and priv_latent not provided!")
      estimate_array = jnp.concatenate([obs_dict['vector'], act_dict['action']], axis=-1)
      _, (latents) = self._roa_estimate_jit(self.model_params, estimate_array, seed=self._seed())
      if not self.predictor_gets_confidence:
        latents = latents[..., :self.privileged_latent_dim]
      if latents.ndim == 3:
        latents = latents[:, -1]
    
    # decoder that uses the RSSM hidden state to decode
    if self.roa_decoder == "priv_head": 
      if observation is None:
        raise ValueError("obs state is required to decode")
      B,T  = observation['vector'].shape[:2]
      act_dict['priv_latent'] = einops.repeat(latents, 'b d -> b t d', t=T) # repeat to concat to act_dict
      # forward pass the model
      carry, _ = self.init_carry(B)
      _, _, feat = self.forward(carry, obs_dict, act_dict)
      # decode features into latents
      featvec = feat2tensor(feat)
      _, (est_priv_vector) = self._roa_decode_jit(self.model_params, featvec, training=False, seed=self._seed())
      if priv_head_mode == 'last':
        est_priv_vector = est_priv_vector[:, -1]
      else:
        est_priv_vector = jnp.mean(est_priv_vector, axis=1)
    # normal decoder that simply decodes the RSSM latent
    else:
      _, (est_priv_vector) = self._roa_decode_jit(self.model_params, latents[..., :self.privileged_latent_dim], training=False, seed=self._seed())

    return est_priv_vector

  def roa_estimate(self, obs_vector, use_ground_truth_encoder=False, **kwargs):
    """
    Estimates the privileged latent using the observations.
    """

    if self.enable_roa:
      if use_ground_truth_encoder: # eval case in latent discriminator where obs_vector will be the privilidged information  
        if torch.is_tensor(obs_vector):
          assert obs_vector.shape[-1] == self.privileged_info_dim, obs_vector.shape[-1]
          jnp_obs_vector = jnp.array(obs_vector)
          if jnp_obs_vector.ndim == 2:
            jnp_obs_vector = jnp.expand_dims(jnp_obs_vector, axis=1) # add T
          obs_dict = {'priv_vector': jnp_obs_vector}
          _, (priv_latent) = self._roa_encode_jit(self.model_params, obs_dict, seed=self._seed())
          priv_latent = priv_latent[:, -1]
          return priv_latent
      else:
        if torch.is_tensor(obs_vector): 
          obs_vector = jnp.array(obs_vector)
        if obs_vector.shape[-1] > self.obs_dim + self.act_dim:
          obs_vector = obs_vector[..., :17] # include the actions

        assert obs_vector.shape[-1] == self.obs_dim + self.act_dim, f"invalid obs dim {obs_vector.shape[-1]}"
        _, (priv_latent) = self._roa_estimate_jit(self.model_params, obs_vector, seed=self._seed())
        if not self.predictor_gets_confidence:
          priv_latent = priv_latent[..., :self.privileged_latent_dim]
        if priv_latent.ndim == 3:
          priv_latent = priv_latent[:, -1]
        return priv_latent
      
    if self.use_bayesian_set_encoder:
        if use_ground_truth_encoder:
          logging.info(f"WARNING: Bayesian Set Encoder has no ground truth encoder, will use identity instead.")
          if torch.is_tensor(obs_vector):
            assert obs_vector.shape[-1] == self.privileged_info_dim, obs_vector.shape[-1]
            jnp_obs_vector = jnp.array(obs_vector)
            if jnp_obs_vector.ndim == 2:
              jnp_obs_vector = jnp.expand_dims(jnp_obs_vector, axis=1) # add T
            priv_latent = jnp_obs_vector[:, -1]
            return priv_latent
        # use the estimator
        else:
          if torch.is_tensor(obs_vector): 
            obs_vector = jnp.array(obs_vector)
          if obs_vector.shape[-1] > self.obs_dim + self.act_dim:
            obs_vector = obs_vector[..., :17] # include the actions

          assert obs_vector.shape[-1] == self.obs_dim + self.act_dim, f"invalid obs dim {obs_vector.shape[-1]}"

          # need to reshape into x, y
          ctx_x = obs_vector[:, :-1] # (obs + act)
          ctx_y = obs_vector[:, 1:, :self.obs_dim] # just obs
          set_encoder_obs = jnp.concatenate([ctx_x, ctx_y], axis=-1)
          _, (mu_z, cov_z) = self._set_encoder_jit(self.model_params, set_encoder_obs, seed=self._seed())
          if mu_z.ndim == 3:
            mu_z = mu_z[:, -1]
          return mu_z
      
  def imagine_forward(self, carry, act, length):
    """
    Move forward the latent state using only imagination
    """
    _, (carry, recon) =  self._imagine_forward_jit(self.model_params, carry, act, length, seed=self._seed())
    return carry, recon
  
  def train(self, carry, act, data, step=0):

    # run a training step, updating the model parameters
    self.params, mets = self._train_jit(self.params, carry, act, data, step, seed=self._seed())
    # update the model_params 
    self.model_params = {k: v for k, v in self.params.items() if k in self.model_keys}

    return carry, mets

  def denoise_trajectory(self, trajectory:Dict):
    """
    Denoise the trajectory through the observe function of the RSSM 
    """
    return trajectory

  # - evaluator methods - #

  def sample_predictions_from_trajectory(self, trajectory:Dict, index:int, horizon:int, 
                                         no_img=False, return_warmup=False, return_samples=False, 
                                         debug=False, num_samples=20, return_extras=False, latent_override=None, **unused_args) -> Array:
    
    """
    Experimental get prediction from trajectory.
    Uses 'trajectory' to warmup model up to 'index', then making 'horizon' predictions. 
  
    Args:
      trajectory (Dict): trajectory to use in warmup 
      index (int): point to start making predictions from (inclusive)
      horizon (int): number of predictions to make 
       
    """

    if unused_args:
      pylogging.debug("RSSM_MP.get_prediction_from_trajectory "
                      f"got unexpected arguments, which will be ignored: " + \
                      str([key for key in unused_args.keys()]))

    t0 = time.time()

    req_keys = ["joint_angles", "joint_velocities", "last_action"]
    assert all(k in trajectory for k in req_keys), f"trajectory is missing one or more of keys: {req_keys} "
    
    for key in req_keys:
      if len(trajectory[key].shape) == 2:
        trajectory[key] = jnp.expand_dims(trajectory[key], 0) 
      
    extras = {}
  
    # set variables
    batch_size = trajectory["joint_angles"].shape[0]
    x_oldest = index - self.num_hist_timesteps_to_use

    # assemble the observation from the trajectory
    qpos_hist = trajectory["joint_angles"][:, x_oldest : index, :]
    qvel_hist = trajectory["joint_velocities"][:, x_oldest : index, :]
    obs = np.concatenate([qpos_hist, qvel_hist], axis=-1)[..., self.target_obs]
    obs_processed = obs
    # obs_hist is all normalised 
    obs_hist = {'vector': obs_processed, 
                'is_first': jnp.zeros((batch_size, self.num_hist_timesteps_to_use), dtype=bool)}
    acts_hist = {'action': trajectory["last_action"][:, x_oldest:index]}
    acts_future = {'action' : trajectory["last_action"][:, index:index+horizon]}

    setup_time = time.time() - t0
    t1 = time.time()
    # init and set hidden state
    init_carry, _ = self.init_carry(batch_size)
    
    if self.enable_roa:

      if latent_override is not None:
        assert latent_override.ndim == 2, (f"expected num_dims for latent_override = 2, got {latent_override.ndim}"
                                           f"with shape {latent_override.shape}")
        hist_priv_latents = einops.repeat(latent_override, 'b d -> b t d', t=self.num_hist_timesteps_to_use)
        future_priv_latents =  einops.repeat(latent_override, 'b d -> b t d', t=horizon)
        if self.latent_concat == 'obs':
          raise NotImplementedError
        else:
          acts_hist['priv_latent'] = hist_priv_latents
          acts_future['priv_latent'] = future_priv_latents
      
      else:
        estimator_vector = jnp.concatenate([obs_hist['vector'], acts_hist['action']], axis=-1)
        priv_latents = self.roa_estimate(estimator_vector) # (B, 7)
        priv_latents_exp = jnp.expand_dims(priv_latents, 1) # (B, 1, 7)
        priv_latents_hist_repeat = jnp.repeat(priv_latents_exp, self.num_hist_timesteps_to_use, axis=1) # (B, 50, 7)
        if self.latent_concat == 'obs':
          obs_hist['priv_latent'] = priv_latents_hist_repeat
        elif self.latent_concat == 'act':
          acts_hist['priv_latent'] = priv_latents_hist_repeat
          priv_latents_future_repeat = jnp.repeat(priv_latents_exp, horizon, axis=1) 
          acts_future['priv_latent'] = priv_latents_future_repeat
    
    if self.use_bayesian_set_encoder:

      if latent_override is not None:
        assert latent_override.ndim == 2, (f"expected num_dims for latent_override = 2, got {latent_override.ndim}"
                                           f"with shape {latent_override.shape}")
        hist_priv_latents = einops.repeat(latent_override, 'b d -> b t d', t=self.num_hist_timesteps_to_use)
        future_priv_latents =  einops.repeat(latent_override, 'b d -> b t d', t=horizon)
        if self.latent_concat == 'obs':
          raise NotImplementedError
        else:
          acts_hist['priv_latent'] = hist_priv_latents
          acts_future['priv_latent'] = future_priv_latents

      else:
        T_ctx = T // 2
        x_vector_ctx = obs_hist['vector'][:, :T_ctx-1]
        act_vector_ctx = prevact['action'][:, :T_ctx-1]
        y_vector_ctx = acts_hist['vector'][: 1:T_ctx]
        set_obs = jnp.concatenate([x_vector_ctx, act_vector_ctx, y_vector_ctx], axis=-1)
        mu_z, cov_z, = self.set_encoder(set_obs)
        mu_z_exp = jnp.expand_dims(mu_z_exp, 1)
        mu_z_hist_repeat = jnp.repeat(mu_z_exp, self.num_hist_timesteps_to_use, axis=1)
        acts_hist['priv_latent'] = mu_z_hist_repeat
        muz_z_future_repeat = jnp.repeat(mu_z_exp, horizon, axis=1) 
        acts_future['priv_latent'] = muz_z_future_repeat

    rollouts = []
    for _ in range(num_samples):

      carry, warmup_recon, _ = self.forward(init_carry, obs_hist, acts_hist)  
      warmup_vector = self.output_transform(warmup_recon['vector']) # B, num_hist, D
      _, imagine_rollout = self.imagine_forward(carry[1], acts_future, horizon) # B, horizon, D
      imagine_vector = self.output_transform(imagine_rollout)
      
      sample_rollout = jnp.concatenate([warmup_vector, imagine_vector], axis=1) # B, num_hist + horizon, D
      rollouts.append(sample_rollout)

    rollouts = jnp.stack(rollouts, axis=0) # samples, B, num_hist + horizon, D

    # compute the standard deviation of the rollout
    rollout_std = jnp.std(rollouts, axis=0) # B, num_hist + horizon, D
    extras['predictor_stddev_denormalised'] = rollout_std[:, self.num_hist_timesteps_to_use:]

    if not return_warmup:
      rollouts = rollouts[:, :, self.num_hist_timesteps_to_use:]
    
    if return_samples:
      return_rollout = rollouts
    else:
      return_rollout = jnp.median(rollouts, axis=0)
    
    if return_extras:
      return return_rollout, extras 
    else:
      return return_rollout

  def get_latent_estimator(self, **kwargs):
    """
    Access point for the Agent_Latent_Discriminator class to use the RSSM's latent estimator.
    """
    return self.roa_estimate
  
  def get_latent_decoder(self, **kwargs):
    return self.roa_decode

    # elif self.rssm_latent_mode == 'stoch_2d':
    #   return feat['stoch']
    # elif self.rssm_latent_mode == 'stoch_1d':
    #   return feat['stoch'].reshape(feat['stoch'].shape[:2] + (-1,))
  
  # - util methods - # 
  def init_carry(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    carry = (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size))
    act = jax.tree.map(zeros, self.act_space)
    return carry, act
  
  def load_save_state(self, save_state):
    """
    Load the model save_state
    """
    self.model_params = save_state

  def get_observation_from_trajectory(self, trajectory, prediction_index):
    
    """
    Return the correct model observation from a given trajectory. Expects the
    trajectory to be a dictionary with the following fields:

    trajectory = {
      "joint_angles" : np.array(...),
      "joint_velocities" : np.array(...),
      "last_action" : np.array(...), # action that LED to angles/velocities at this timestep
    }

    prediction_index is the step for which PREDICTIONS will be made, so the 
    observation will cover timesteps up to but NOT including prediction_index.
    
    """ 
    # index the trajectory
    t = self.num_hist_timesteps_to_use
    prev_timesteps = slice(prediction_index - t, prediction_index)
    next_actions = slice(prediction_index - t, prediction_index)
    qpos_hist = trajectory["joint_angles"][:, prev_timesteps, :]
    qvel_hist = trajectory["joint_velocities"][:, prev_timesteps, :]
    act_hist = trajectory["last_action"][:, next_actions, :] 

    return np.concatenate([qpos_hist, qvel_hist, act_hist], axis=-1)
  
  def dreamer_format_array(self, x):
    """
    Formats an array to the Dreamer style obs, action dictionaries used by
    this model.  
    """
    
    x = jnp.array(x)
    B, T = x.shape[:2]

    if x.shape[-1] == self.obs_dim: # just create obs dictionary
      obs = {
        'vector': x,
        'is_first': jnp.zeros((B, T), dtype=bool).at[:, 0].set(True)
      }
      return [obs]
    elif x.shape[-1] == self.obs_dim + self.act_dim: # create obs and act dicitonary
      obs = {
        'vector': x[..., :self.obs_dim],
        'is_first': jnp.zeros((B, T), dtype=bool).at[:, 0].set(True)
      }
      act = {
        'action': x[..., self.obs_dim:]
      }
      return [obs, act]
    elif x.shape[-1] >= self.obs_dim + self.act_dim + self.privileged_info_dim:
      obs = {
        'vector': x[..., :self.obs_dim],
        'is_first': jnp.zeros((B, T), dtype=bool).at[:, 0].set(True),
        'priv_vector': x[..., self.obs_dim+self.act_dim:self.obs_dim+self.act_dim+self.privileged_info_dim]
      }
      act = {
        'action': x[..., self.obs_dim:self.obs_dim+self.act_dim]
      }
      return [obs, act]
    else:
      raise ValueError(f"input tensor of shape {x.shape} is too short / missing info (< self.obs_dim={self.obs_dim})")

  # -- PRIVATE METHODS -- #
     
  def _forward(self, carry, obs, prevact, single, decode=True):

    """
    Moves the model internal state forward.

    Set decode=False to skip the decoder when only the updated carry/feat are
    needed (e.g. an MPC belief update), avoiding a wasted decoder forward.
    """

    (enc_carry, dyn_carry, dec_carry) = carry
    kw = dict(training=False, single=single)

    reset = obs['is_first']
    obs['vector'] = obs['vector'][..., self.target_obs]

    deter_mask = jnp.zeros_like(reset, dtype=bool)
    stoch_mask = jnp.zeros_like(reset, dtype=bool)

    enc_carry, _, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, _, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, deter_mask=deter_mask, stoch_mask=stoch_mask, **kw)

    if not decode:
      carry = (enc_carry, dyn_carry, dec_carry)
      return carry, None, feat

    dec_carry, _, recons = self.dec(dec_carry, feat, reset, **kw)

    carry = (enc_carry, dyn_carry, dec_carry)
    return carry, recons, feat
  
  def _imagine_forward(self, carry, act, length):
    """
    Move the latent state forward using determinstic model state 
    h_{t-1} and prior action a_{t-1}.

    Returns the new model state (h_t, z_t) and imagined observation
    (x'_t)
    """
    kw = dict(training=False, single=False)
    carry, feat, _ = self.dyn.imagine(
        carry, act, length, lean=True, **kw)
    reset = jnp.zeros((act['action'].shape[:2]))
    _, _, recons = self.dec({}, feat, reset, **kw)
    obsrecons = recons['vector']
    carry = ({}, carry, {})
    return carry, obsrecons
  
  def _train(self, carry, prevact, data, step=0):
    """
    Trains the RSSM 
    """

    if self.online_training:
      raise NotImplementedError
    else:
      # form batch from data
      obs = {k: data[k] for k in self.obs_space}
      prevact = {'action': data['action']}
      vector_noise = data['vector_noise']
      priv_vector_noise = data.get('priv_vector_noise', None)
      priv_vector_stds = data.get('priv_vector_stds',None)
     
    metrics, loss_mets = self.opt(
        self._loss, carry, obs, prevact, vector_noise, priv_vector_noise, priv_vector_stds, step=step, training=True, has_aux=True)
    metrics.update(loss_mets) 
    return metrics
  
  def _loss(self, carry, obs, prevact, vector_noise, priv_vector_noise, priv_vector_stds, training, step):

    losses = {}
    metrics = {}
    B, T = obs['is_first'].shape

    ## -- ROA LOSS -- ##
    if self.enable_roa: 
      roa_losses, roa_metrics, predictor_latents = self._roa_loss(obs, prevact, priv_vector_noise, priv_vector_stds, step)
      losses |= roa_losses
      metrics |= roa_metrics

      if self.latent_concat == 'obs':
        obs['priv_latent'] = predictor_latents 
      elif self.latent_concat == 'act':
        prevact['priv_latent'] = predictor_latents 
    
    if self.use_bayesian_set_encoder:

      # split the obs and action into context and target (to match HiP-RSSM)
      T_ctx = T // 2
      x_vector_ctx = obs['vector'][:, :T_ctx-1]
      act_vector_ctx = prevact['action'][:, :T_ctx-1]
      y_vector_ctx = obs['vector'][:, 1:T_ctx]
      set_obs = jnp.concatenate([x_vector_ctx, act_vector_ctx, y_vector_ctx], axis=-1)
      mu_z, cov_z = self.set_encoder(set_obs)
      # set obs and act to the target 
      obs = {k:v[:, T_ctx:] for k, v in obs.items()}
      obs['is_first'] = obs['is_first'].at[:, 0].set(True)
      B, T = obs['is_first'].shape
      vector_noise = vector_noise[:, T_ctx:]
      prevact = {k:v[:, T_ctx:] for k, v in prevact.items()}
      prevact['priv_latent'] = einops.repeat(mu_z, "b d -> b T d", T=T)

    ## -- RSSM LOSS -- ##
    if self.rwm_loss:
      rwm_losses, rwm_metrics = self._rwm_loss(carry, obs, prevact, vector_noise)
      losses |= rwm_losses
      metrics |= rwm_metrics

    else:
      rssm_losses, rssm_metrics = self._rssm_loss(carry, obs, prevact, vector_noise)
      losses |= rssm_losses
      metrics |= rssm_metrics
      
    ## -- AGGREGATE LOSSES AND METRICS -- ##
    metrics.update({f'raw_loss/{k}': v.mean() for k, v in losses.items() if 'roa' not in k})
    assert set(losses.keys()) == set(self.scales.keys()), (
    sorted(losses.keys()), sorted(self.scales.keys()))
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])
    
    for k, v in metrics.items():
      if v.ndim > 1:
        raise ValueError(f"{k} has ndim {v.dim} which is incompatible with logger")

    return loss, metrics
      
  def _rssm_loss(self, carry, obs, prevact, vector_noise):
      
      w_enc_carry, w_dyn_carry, w_dec_carry = carry
      training = True

      reset = obs['is_first']
      obs['vector'] = obs['vector'][..., self.target_obs] # think this could be dangerous if target obs changes 

      B, T = reset.shape
      losses = {}
      metrics = {}

      deter_mask = jnp.zeros_like(reset, dtype=bool) # B, T
      stoch_mask = jnp.zeros_like(reset, dtype=bool) # B, T

      if self.state_masking:
        if self.state_masking_only_start:
          mask = jrand.uniform(nj.seed(), (B)) < self.state_masking_p
          deter_mask = deter_mask.at[:, 0].set(mask)
          stoch_mask = stoch_mask.at[:, 0].set(mask)
        else:
          mask = jrand.uniform(nj.seed(), (B, T)) < self.state_masking_p
          deter_mask = mask
          stoch_mask = mask

      metrics.update({'deter_resets': jnp.sum(deter_mask)})
      metrics.update({'stoch_resets': jnp.sum(stoch_mask)})
      metrics.update({'mean_act': jnp.mean(prevact['action'])})

      ### -- RSSM LOSS -- ###
      enc_carry, enc_entries, tokens = self.enc(
          w_enc_carry, obs, reset, training)

      dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
          w_dyn_carry, tokens, prevact, reset, deter_mask, stoch_mask, training)
      losses.update(los)
      metrics.update(mets)

      dec_carry, dec_entries, recons = self.dec(
          w_dec_carry, repfeat, reset, training)
      state_recon = recons['vector'] # extract the state reconstructions

      # if decoding output is different from the encoder input
      if self.asym_dec:
        assert 'decoder_target' in obs, f"obs missing decoder_target key which is required for asymetrical decoder!"
        target = obs['decoder_target']
        recon_loss = jnp.mean(state_recon.output.loss(target), axis=-1) # B, T
        losses['vector'] = recon_loss
        if 'reg_vector' in self.scales:
          recon_reg_loss = jnp.mean(state_recon.output.reg_loss(target), axis=-1) # sg on the model output and not the target 
          losses['reg_vector'] = recon_reg_loss
      # otherwise normal reconstruction loss
      else:
          space, value = self.obs_space['vector'], obs['vector']
          assert value.dtype == space.dtype, ('vector', space, value.dtype)
          target = value - vector_noise  
          recon_loss = state_recon.output.loss(sg(target)) # prevent aggregation over D (returns B,T,D)
          assert recon_loss.ndim == 3, f"recon_loss.ndim = {recon_loss.ndim}, expected 3"
          # compute different losses through aggregation:
          vector_loss = jnp.mean(recon_loss, axis=-1) 
          vector_pos_loss = jnp.mean(recon_loss[..., :7], axis=-1) # B,T
          vector_vel_loss = jnp.mean(recon_loss[..., 7:], axis=-1) # B,T
          vector_act_loss = jnp.mean(recon_loss[..., [0,1,4,7,8,11]], axis=-1) # B,T
          vector_free_loss = jnp.mean(recon_loss[..., [2,3,5,6,9,10,12,13]], axis=-1) # B,T
        
          # regularize reconstruction using forward kinematics 
          if self.fk_regularizing:
            assert 'payload_position' in obs, "payload_position missing from data in loss function which is required for fk_regularizing"
            assert 'fk_reg' in self.scales, "fk_reg missing from self.scales which is required for fk_regularizing"
            si_output = self.output_transform(state_recon.pred()) # B, T, D
            pos = si_output[..., :7] 
            vel = si_output[..., 7:14]
            # compute the payload_position
            payload_position = self.payload_model(pos, vel) 
            # compute the error 
            fk_reg_error = jnp.mean(jnp.abs(payload_position - obs['payload_position']), axis=-1)
            losses['fk_reg'] = fk_reg_error

          # splitting the loss between actuated and non-actuated components using open-loop and 1-step
          if self.split_loss_training:
            losses['free_joints'] = vector_free_loss 
            losses['actuated_joints'] = vector_act_loss
          else:
            losses['vector_pos']= jnp.where(self.full_autoregressive_training, 0, vector_pos_loss)
            losses['vector_vel']= jnp.where(self.full_autoregressive_training, 0, vector_vel_loss)
      
          # save other component results
          metrics.update({f"loss/vector_act_loss": vector_act_loss.mean()})
          metrics.update({f"loss/vector_free_loss": vector_free_loss.mean()})
          metrics.update({f"loss/vector_loss": vector_loss.mean()})
          

      ### -- AUTOREGRESSIVE TRAINING LOSSES -- ###
      if self.autoregressive_training:     
          """

          Toy example:
          BL = 10
          autoregressive_num_steps = 4


          obs: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9
          policy: a0, a1, a2, a3, a4, a5, a6, a7, a8, a9
          init_state: z-1
          observe_states: z0, z1, z2, z3, z4, z5, z6, z7, z8, z9
          
          1. N = 10 - 4 + 1 = 7
          2. vmap_states = [z-1, z0, z1, z2, z3, z4, z5]

          3 target_obs:
            1. [x0, x1, x2, x3]
            2. [x1, x2, x3, x4]
            3. [x2, x3, x4, x5]
            4. [x3, x4, x5, x6]
            5. [x4, x5, x6, x7]
            6. [x5, x6, x7, x8]
            7. [x6, x7, x8, x9]

          4. policy:
            1. [a0, a1, a2, a3]
            2. [a1, a2, a3, a4]
            3. [a2, a3, a4, a5]
            4. [a3, a4, a5, a6]
            5. [a4, a5, a6, a7]
            6. [a5, a6, a7, a8]
            7. [a6, a7, a8, a9]
          
          """

          N = self.batch_length - self.autoregressive_num_steps + 1
          reset_array = jnp.zeros((B, self.autoregressive_num_steps), dtype=jnp.bool_)

          # warmup state from trainer
          init_state = jax.tree.map(lambda x: jnp.expand_dims(x, axis=1), w_dyn_carry)  # B, D -> B, 1, D
          # states computed through observed rollout 
          observe_states = jax.tree.map(lambda x: x[:, :-self.autoregressive_num_steps], dyn_entries) # B, T, D -> B, T-N, D
          # combine, should be 1 behind the observations 
          vmap_states = jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=1), init_state, observe_states) # B, N, D

          # scan function for constructing the observation and action scan arrays
          def scan_step(carry, entry):
            x, i = carry
            y = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, i, self.autoregressive_num_steps, 1), x) 
            i+=1
            return (x, i), y
          
          scan_obs = obs['vector'] - vector_noise # these are the targets 
          _, target_obs = jax.lax.scan(scan_step, (scan_obs, 0), length=N) # B, T, D -> N, B, horizon, D
          _, policy = jax.lax.scan(scan_step, (prevact, 0), length=N) # B, T, D -> N, B, horizon, D
          
          def autoregressive_loss(warmup_state, policy, target): # -> B, T, D
            _, feat, _ = self.dyn.imagine(warmup_state, policy, self.autoregressive_num_steps, True)
            _, _, recons = self.dec({}, feat, reset_array)
            return recons['vector'].output.loss(sg(target))

          autoregressive_loss_vmap = jax.vmap(autoregressive_loss, in_axes=(1, 0, 0)) 
          autoregressive_loss = autoregressive_loss_vmap(vmap_states, policy, target_obs) # N, B, T, D
          autoregressive_loss = jnp.mean(autoregressive_loss, axis=0) # B, T, D

          if self.autoregressive_only_last:
            autoregressive_loss = autoregressive_loss[:, -1] # B, D
          else:
            autoregressive_loss = jnp.mean(autoregressive_loss, axis=1) # B, D

          if self.split_loss_training:
            # when past a certain step, only use the open loop loss for actuated joints
            autoregressive_actuated_loss = jnp.mean(autoregressive_loss[..., [0,1,4,7,8,11]], axis=-1)
            losses['actuated_joints'] = jnp.where(step > self.split_loss_training_step, 
                                                  autoregressive_actuated_loss, 
                                                  losses['actuated_joints'])
          else:
            losses['autoregressive'] = jnp.where(step > self.autoregressive_training_step, 
                                            autoregressive_loss*self.autoregressive_training_scale, 
                                            0)

      return losses, metrics

  def _rwm_loss(self, carry, obs, prevact, vector_noise):
        """
        Toy example:
        BL = 10
        M = 2
        N = 2


        obs: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9
        policy: a0, a1, a2, a3, a4, a5, a6, a7, a8, a9
        observe_states: z0, z1, z2, z3, z4, z5, z6, z7, z8, z9
        
        1. N = 10 - 4 + 1 = 7

        3. sliding window:
          1. [x0, x1, x2, x3]
          2. [x1, x2, x3, x4]
          3. [x2, x3, x4, x5]
          4. [x3, x4, x5, x6]
          5. [x4, x5, x6, x7]
          6. [x5, x6, x7, x8]
          7. [x6, x7, x8, x9]

        4. policy:
          1. [a0, a1, a2, a3]
          2. [a1, a2, a3, a4]
          3. [a2, a3, a4, a5]
          4. [a3, a4, a5, a6]
          5. [a4, a5, a6, a7]
          6. [a5, a6, a7, a8]
          7. [a6, a7, a8, a9]


          take history first 
          z_0 = f(x0, a0, z_-1), z_1 = f(x_1, a_1, z_0) 

          outer autoregression 
          ~z_2 = f(z_1, a_2)
          x_2 = d(z_2)

          inner autoregression
          z_2_x = f(x_2, a_2, z_1)
        """
        reset = obs['is_first']
        losses = {}
        roa_metrics = {}
        metrics = {}

        B, T = reset.shape

        # 1. Create all the windows from our dataset
        N = self.batch_length - self.rwm_window_size + 1 # number of windows we will have
        def window_scan_step(carry, entry):
          x, i = carry
          y = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, i, self.rwm_window_size, 1), x) 
          i+=1
          return (x, i), y
        _, obs_windows = jax.lax.scan(window_scan_step, (obs, 0), length=N) # B, T, D -> N, B, horizon, D
        _, policy_windows = jax.lax.scan(window_scan_step, (prevact, 0), length=N) # B, T, D -> N, B, horizon, D
        target_obs = obs['vector'] - vector_noise
        _, target_obs_windows = jax.lax.scan(window_scan_step, (target_obs, 0), length=N) # B, T, D -> N, B, horizon, D
        init_carry, _ = self.init_carry(B)
        reset = jnp.zeros((B, self.rwm_window_size), dtype=jnp.bool_).at[:, 0].set(True) # reset at the start of every window
        deter_mask = jnp.zeros_like(reset, dtype=bool) # B, T
        stoch_mask = jnp.zeros_like(reset, dtype=bool) # B, T

        # 2. implement the vmap loss function that will be vmapped over the windows
        def vmap_loss_step(obs_window, policy_window, target_obs_window):
          _, dyn_carry, _ = init_carry

          # 1. encodes the obs 
          # t0, t1, ..., tn
          enc_carry, enc_entries, tokens = self.enc(
            {}, obs_window, reset, True)
          # 2. dynamic states for KL loss
          # z0, z1, ..., zn
          _, dyn_entries, dyn_loss, repfeat, mets = self.dyn.loss(
            dyn_carry, tokens, policy_window, reset, deter_mask, stoch_mask, True)
          # 3. fetch the warmup state
          dyn_warmup = jax.tree.map(lambda x: x[:, self.rwm_warmup-1], dyn_entries)  
          # 4. perform outer autoregression
          policy_ar = jax.tree.map(lambda x: x[:, self.rwm_warmup:], policy_window)
          _, feat, _ = self.dyn.imagine(dyn_warmup, policy_ar, self.rwm_rollout, True)
          # 5. decode these features 
          _, _, recons = self.dec({}, feat, jnp.zeros((B, self.rwm_rollout), dtype=jnp.bool_), True) 
          # 6. extract the decodings and add them to the obs_window
          obs_recon = self.output_transform(recons['vector'].pred()) # B, rwm_rollout, D
          inner_obs = {'vector': jnp.concatenate([obs_window['vector'][:, :self.rwm_warmup], obs_recon], axis=1)}
          # 6. perform inner autoregression
          enc_carry, enc_entries, auto_tokens = self.enc(
            init_carry, inner_obs, reset, True)
          _, _, auto_feat = self.dyn.observe(
            dyn_carry, auto_tokens, policy_window, reset, True, deter_mask, stoch_mask)
          _, _, auto_recons = self.dec(
            {}, auto_feat, reset, True) 
          # 7. compute the loss for the recons 
          recon_error = auto_recons['vector'].loss(sg(target_obs_window)) # should return B, T, D
          # clip out the the warmup steps
          recon_loss = recon_error[:, self.rwm_warmup:]
          return recon_loss, dyn_loss, mets

        recon_loss, dyn_loss, mets = jax.vmap(vmap_loss_step, in_axes=(0, 0, 0))(obs_windows, policy_windows, target_obs_windows)

        # 3. apply averaging over the vmap dimension for the reconstruction loss and dynamics lsos
        recon_loss_mean = jax.tree.map(lambda x: jnp.mean(x, axis=(0, 2)), recon_loss)
        dyn_loss_mean = jax.tree.map(lambda x: jnp.mean(x, axis=(0, 2)), dyn_loss)
        mets = jax.tree.map(lambda x: jnp.mean(x, axis=0), mets)

        losses.update(dyn_loss_mean)
        losses.update({'vector': recon_loss_mean})

        return losses, metrics

  def _roa_loss(self, obs, prevact, priv_vector_noise, priv_vector_stds, step):
      
      B, T = priv_vector_noise.shape[:2]

      roa_losses = {}
      roa_metrics = {}

      non_noisy_priv_info = {"priv_vector": obs['priv_vector'] - priv_vector_noise}
      # these will be ignored if use_estimator_confidence is disabled
      true_noise = jnp.zeros((B, T, self.privileged_info_dim))
      no_noise = jnp.zeros((B, T, self.privileged_info_dim))

      if self.use_estimator_confidence:
        true_noise = priv_vector_stds
        if not self.estimator_confidence_per_feature: # average over the feature dim
          true_noise = jnp.mean(true_noise, axis=-1, keepdim=True) # B, 1
          no_noise = jnp.mean(no_noise, axis=-1, keepdim=True) # B, 1
      
      # priv latents used for the predictor
      noisy_priv_latents = self.roa_encoder(obs, noise_std=true_noise) # (B, T, D + conf) 
      noisy_priv_latents_repeat = einops.repeat(noisy_priv_latents[:, -1], 'B D -> B T D', T=T)  

      # priv latents used for the estimator loss function
      denoised_priv_latents = self.roa_encoder(non_noisy_priv_info, noise_std=no_noise) # (B, T, D + conf)
      
      # compute estimator loss
      if isinstance(self.roa_estimator, ROA_RSSM_Estimator):
        if self.use_estimator_confidence:
          raise NotImplementedError("Estimator confidence not implemented for RSSM")
        # the regression target is the encoding
        obs['decoder_target'] = priv_latents
        estimator_losses = self.roa_estimator.loss(obs, prevact) 
        roa_metrics.update({f"estimator_rssm/{k}":v.mean() for k, v in estimator_losses.items()})
        roa_reg_loss = estimator_losses['reg_vector'] # reg loss from the RSSM
        roa_est_pred_loss = estimator_losses['vector'] # pure prediction loss
        dyn_loss = estimator_losses['dyn']
        rep_loss = estimator_losses['rep']
        
        if self.roa_rssm_timestep_weighting == 'linear':
          T = roa_est_loss.shape[1]
          W = self.roa_rssm_warmup_length
          ramp_length = T - self.roa_rssm_warmup_length
          ramp = jnp.linspace(0.0, 1.0, ramp_length)
          scales = jnp.zeros(T).at[W:].set(ramp)
          roa_est_pred_loss *= scales
          roa_reg_loss *= scales
        
        roa_est_loss = (roa_est_pred_loss + dyn_loss + rep_loss)
        roa_reg_loss = (roa_reg_loss + sg(dyn_loss) + sg(rep_loss))

      else:
        # use only the last as the target
        target_latents = denoised_priv_latents[:, -1, :] # B, D
        estimator_input_vector = jnp.concatenate([obs['vector'], prevact['action']], axis=-1) # B, T, O + A
        # compute estimation of latents
        estimator_output = self.roa_estimator(estimator_input_vector) # output (B, D + conf)
        # use confidence loss function
        if self.use_estimator_confidence:
          # extract uncertainty
          estimated_uncertainty = estimator_output[:, self.privileged_latent_dim:] # B, conf
          estimated_uncertainty = jnp.clip(estimated_uncertainty, 
                                            self.confidence_log_var_clamping_min,
                                            self.confidence_log_var_clamping_max)
          # extract latents
          target_latents = target_latents[:, :self.privileged_latent_dim] # B, D
          estimated_latents = estimator_output[:, :self.privileged_latent_dim] # B, D
          # compute log likelihood loss
          latent_unc = jnp.exp(estimated_uncertainty) # variance 
          error_norm = jnp.power((estimated_latents - sg(target_latents)), 2) # B, D

          per_timestep_loss = ((1 / (2 * latent_unc)) * error_norm
                                + 0.5 * estimated_uncertainty)
          roa_losses['roa_est'] = per_timestep_loss.mean()
        # use normal estimator loss function
        else:
          raw_est_loss = huber_fn(sg(target_latents), estimator_output, self.estimator_huber_delta) # B, D
          raw_reg_loss = huber_fn(target_latents, sg(estimator_output), self.estimator_huber_delta) # B, D
          reg_loss_anneal = jnp.where(step >= self.anneal_roa_reg_step, 1.0, 0.0) # now just set reg term always to 0 if we are not at anneal step

          roa_losses['roa_est'] = jnp.mean(raw_est_loss, axis=-1)
          roa_losses['roa_reg'] = jnp.mean(raw_reg_loss * reg_loss_anneal, axis=-1)
      
      if self.roa_decoder is not None: 
        # try decode over all T and see the difference 
        if self.roa_decode_all_t:
          target = non_noisy_priv_info['priv_vector']
        else:
          target = non_noisy_priv_info['priv_vector'][:, -1] 
        # reconstruct from the model state 
        if self.roa_config.decoder == 'priv_head': 
          model_state = feat2tensor(repfeat)
          priv_vector_pred= self.roa_decoder(model_state, 2)['output']
          loss = priv_vector_pred.loss(target)
          pred_vector_mean = jnp.mean(jnp.linalg.norm(nn.symexp(priv_vector_pred.pred()), axis=-1))
          roa_metrics.update({f'decoded_priv_vector mean': pred_vector_mean})
          roa_losses["roa_priv_recon"] = loss
        # reconstruct from the priv latents
        else:
          if self.roa_decode_all_t:
            decoder_input = noisy_priv_latents_repeat[..., :self.privileged_latent_dim]
          else:
            decoder_input = noisy_priv_latents_repeat[:, -1, :self.privileged_latent_dim]
          
          # assert decoder_input.shape == target.shape, (decoder_input.shape, target.shape)
          priv_vector_pred = self.roa_decoder(decoder_input)['priv_vector'] 
          roa_losses['roa_priv_recon'] = priv_vector_pred.loss(target)

      # concatenate latents to observation for predictor
      if self.train_with_estimated_latents:
        estimated_priv_latents_repeat = einops.repeat(estimator_output, 'B D -> B T D', T=T)
        if not self.predictor_gets_confidence:
          estimated_priv_latents_repeat = estimated_priv_latents_repeat[..., :self.privileged_latent_dim]
        estimated_p = self.train_with_estimated_latents_max_p * jnp.minimum(step/self.train_with_estimated_latents_max_step, 1.0)
        estimated_mask = jrand.bernoulli(nj.seed(), p=estimated_p, shape=(B, 1, 1))
        predictor_latents = jnp.where(estimated_mask, 
                                      estimated_priv_latents_repeat, # stop gradient on the estimated latents  
                                      noisy_priv_latents_repeat)
      else:
        predictor_latents = noisy_priv_latents_repeat
                
      return roa_losses, roa_metrics, predictor_latents

  def _zeros(self, spaces, batch_shape):
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces.items()}
    for dim in reversed(batch_shape):
      data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data
  
  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,

      split_estimator_schedule: bool = False,

      estimator_lr: float = 4e-7,
      estimator_schedule: str = 'const',
      estimator_warmup: int = 1000,
      estimator_anneal: int = 0,

      decoder_lr: float = 4e-5,
      decoder_schedule: str = 'const',
      decoder_warmup: int = 1000,
      decoder_anneal: int = 0,

      **kwargs,

  ):

    logging.info("main_lr: {}".format(lr))
    logging.info("split schedule: {}".format(split_estimator_schedule))
    logging.info("estimator_lr: {}".format(estimator_lr))
    logging.info("decoder_lr: {}".format(decoder_lr))

    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'

    def create_schedule(lr, schedule, warmup, anneal):
      if schedule == 'const':
        sched = optax.constant_schedule(lr)
      elif schedule == 'linear':
        sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
      elif schedule == 'cosine':
        sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
      else:
        raise NotImplementedError(schedule)
      if warmup:
        ramp = optax.linear_schedule(0.0, lr, warmup)
        sched = optax.join_schedules([ramp, sched], [warmup])
      return sched
    
    main_scheduler = create_schedule(lr, schedule, warmup, anneal)

    if split_estimator_schedule:

      logging.info("Using split estimator schedule!")
      
      estimator_scheduler = create_schedule(estimator_lr, estimator_schedule, estimator_warmup, estimator_anneal)
      decoder_scheduler = create_schedule(decoder_lr, decoder_schedule, decoder_warmup, decoder_anneal)

      def label_fn(params_pytree):  
        def get_label_for_param(path, param_leaf):
            # Check if any part of the parameter's path is in our special set
            for key_part in path:
                if '_est' in key_part.key:
                    return 'estimator'
                if '_decoder' in key_part.key:
                    return 'decoder'
            return 'main'
        # Map this function over the params to get the label PyTree
        return jax.tree_util.tree_map_with_path(get_label_for_param, params_pytree)
      
      lr_transformer = optax.multi_transform(
        {
          'main': optax.scale_by_learning_rate(main_scheduler),
          'estimator': optax.scale_by_learning_rate(estimator_scheduler),
          'decoder': optax.scale_by_learning_rate(decoder_scheduler)
        },
        label_fn
      )
      chain.append(lr_transformer)
      return optax.chain(*chain)

    else:
      chain.append(optax.scale_by_learning_rate(main_scheduler))
      return optax.chain(*chain)
  
  def _seed(self):
    """
    Handles seed generation, incrementing from a base seed 
    """
    rng = np.random.default_rng(seed=[self.seed, int(self.seed_inc)])
    self.seed_inc.increment()
    return rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)
  
class RSSM(nj.Module):


  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  discrete_state: bool = True
  prior_dropout_p: float = 0.0
  posterior_dropout_p: float = 0.0
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    self.act_space = act_space
    self.kw = kw

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))

  def initial(self, bsize):

    deter = jnp.zeros([bsize, self.deter], f32)
    if self.discrete_state:
      stoch = jnp.zeros([bsize, self.stoch, self.classes], f32)
    else:
      stoch = jnp.zeros([bsize, self.stoch], f32)
      
    carry = nn.cast(dict(
        deter=deter,
        stoch=stoch))
    return carry

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe(self, carry, tokens, action, reset, training, deter_mask, stoch_mask, single=False):

    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, deter_mask, stoch_mask, training)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nj.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (tokens, action, reset, deter_mask, stoch_mask), unroll=unroll, axis=1)
      return carry, entries, feat

  def _observe(self, carry, tokens, action, reset, deter_mask, stoch_mask, training):
    
    # main reset 
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)

    # mask the individal states
    deter = nn.mask(deter, ~deter_mask)
    stoch = nn.mask(stoch, ~stoch_mask)

    # GRU algorithm
    deter = self._core(deter, stoch, action)
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
      x = nn.dropout(x, self.posterior_dropout_p, training=training)
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    # stoch = nn.cast(self._dist(logit).pred())
    entropy = nn.cast(self._dist(logit).output.entropy())
    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, stoch=stoch, logit=logit, entropy=entropy)
    entry = dict(deter=deter, stoch=stoch)

    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False, lean=False):

    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action) # (B * T, D)
      deter = self._core(carry['deter'], carry['stoch'], actemb)
      logit = self._prior(deter, training)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      # stoch = nn.cast(self._dist(logit).pred())
      carry = nn.cast(dict(deter=deter, stoch=stoch))
      if lean:
        # Inference path: the decoder only reads deter/stoch, so skip the
        # entropy op and avoid stacking logit/entropy across the scan.
        feat = nn.cast(dict(deter=deter, stoch=stoch))
      else:
        entropy = nn.cast(self._dist(logit).entropy())
        feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit, entropy=entropy))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True, lean=lean),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True, lean=lean),
            nn.cast(carry), nn.cast(policy), unroll=unroll, axis=1)

        # if not training:
        #   # We can also return all carry entries but it might be expensive.
        #   entries = dict(deter=feat['deter'], stoch=feat['stoch'])
        #   return carry, entries, feat, action
      return carry, feat, action

  def loss(self, carry, tokens, acts, reset, deter_mask, stoch_mask, training):
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training, deter_mask, stoch_mask,)
    prior = self._prior(feat['deter'], training)
    post = feat['logit']
    dyn = self._dist(sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(sg(prior)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    return carry, entries, losses, feat, metrics

  def _core(self, deter, stoch, action):
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))
    x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter

  def _prior(self, feat, training):
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
      x = nn.dropout(x, self.prior_dropout_p, training=training)
    return self._logit('priorlogit', x)
  
  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    if self.discrete_state:
      x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
      return x.reshape(x.shape[:-1] + (self.stoch, self.classes))
    else:
      return self.sub(name, nn.Linear, self.stoch*2, **kw)(x)
  
  def _dist(self, logits):

    if self.discrete_state: 
      out = embodied.jax.outs.OneHot(logits, self.unimix)
    else:
      mean = logits[..., :self.stoch]
      std = logits[..., self.stoch:]
      out = embodied.jax.outs.Normal(mean, std)

    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out

class Encoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, obs, reset, training, single=False):
    bdims = 1 if single else 2 # 2
    outs = []
    bshape = reset.shape 

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      outs.append(x)

    if self.imgkeys:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      assert all(x.dtype == jnp.uint8 for x in imgs)
      x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    tokens = x.reshape((*bshape, *x.shape[1:]))
    entries = {}
    return carry, entries, tokens

class Decoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  bspace: int = 8 
  outer: bool = False
  strided: bool = False
  output_loss: str = "symlog_mse"
  
  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
    self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, feat, reset, training=True, single=False):
    assert feat['deter'].shape[-1] % self.bspace == 0
    recons = {}
    bshape = reset.shape
    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
    inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
    inp = jnp.concatenate(inp, -1)

    if self.veckeys:
      spaces = {k: self.obs_space[k] for k in self.veckeys}
      outs = {k: self.output_loss for k in spaces}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
      x = x.reshape((*bshape, *x.shape[1:]))
      kw = dict(**self.kw, outscale=self.outscale)
      outs = self.sub('vec', embodied.jax.DictHead, spaces, outs, **kw)(x)
      recons.update(outs)

    entries = {}
    
    if not training:
        recons = {k: recons[k].pred() for k in self.veckeys}

    return carry, entries, recons

class ROA_MLP_Encoder(nj.Module):

  """
  ROA Encoder MLP
  """

  units: int = 128
  norm: str = 'rms'
  act: str = 'gelu'
  layers: int = 4
  symlog: bool = True
  
  def __init__(self, 
               priv_space, # input space
               output_dim, # output dimensions
               confidence_dim=0, # confidence_dim
               output_confidence_min=0.0, # min value for uncertainity 
               **kw): 

    self.obs_space = priv_space
    self.output_dim = output_dim
    
    self.confidence_dim = confidence_dim
    self.output_confidence = self.confidence_dim > 0
    self.output_uncertainty_min = output_confidence_min

    self.veckeys = {k for k in priv_space}
    self.kw = kw

  def __call__(self, obs, noise_std=0.0, single=False):

    # extracting the priv info from the obs, should be shape B, T, D
    vspace = {k: self.obs_space[k] for k in self.veckeys} 
    vecs = {k: obs[k] for k in self.veckeys}
    squish = nn.symlog if self.symlog else lambda x: x
    
    bdims = 2 # no, this should be a time sensitive measure!
    bshape = obs[list(vecs.keys())[0]].shape[:bdims] # shape (B, T)

    x = nn.DictConcat(vspace, 1, squish=squish)(vecs) # flatten vecs to have same feature dimension (i.e. B, T, D)
    x = x.reshape((-1, *x.shape[bdims:])) # collapse B, T to (B * T, D)

    kws = dict(**self.kw, act=self.act, norm=self.norm)
    y = self.sub('mlp_in', nn.MLP, self.layers-1, self.units, **kws)(x)
    out = self.sub('mlp_out', nn.Linear, self.output_dim)(y) 
    out = out.reshape((*bshape, *out.shape[1:])) # reshape to (B, T, D)

    if self.output_confidence:
      true_log_uncertainty = jnp.log(jnp.power(noise_std, 2) + 1e-6)
      if jnp.ndim(true_log_uncertainty) == 0: # expand to (B, N)
          true_log_uncertainty = jnp.broadcast_to(true_log_uncertainty, (out.shape[:-1] + (self.confidence_dim,)))

      true_log_uncertainty = jnp.maximum(true_log_uncertainty, self.output_uncertainty_min) # (B, N)
      out = jnp.concatenate([out, true_log_uncertainty], axis=-1)
    return out

class ROA_MLP_Decoder(nj.Module):

  """
  ROA Decoder MLP
  """

  units: int = 128
  norm: str = 'rms'
  act: str = 'gelu'
  layers: int = 4

  def __init__(self, space, output, **kw):
    
    self.output_space = space
    self.output_key = [k for k in space][0]
    self.output = {k: output for k in self.output_space}
    self.scale_fn = nn.symexp if 'symlog' in output else lambda x:x
    self.kw = kw

  def __call__(self, x, training=True):

    kw = dict(**self.kw, act=self.act, norm=self.norm)
    y = self.sub('mlp_in', nn.MLP, self.layers, self.units, **kw)(x)
    recons = self.sub('vec', embodied.jax.DictHead, self.output_space, self.output)(y)

    if not training:
      pred = self.scale_fn(recons[self.output_key].pred())
      return pred
    else:
      return recons

class ROA_Decoder_Head(nj.Module):
  """
  Wrapper for the output decoder head.
  """

  def __init__(self, space, priv_head_cfg):
    self.output_key = [k for k in space][0]
    self.scale_fn = nn.symexp if 'symlog' in priv_head_cfg.output else lambda x:x
    self.decoder = embodied.jax.MLPHead(space, **priv_head_cfg, name='roa_priv_head')
  
  def __call__(self, x, training=True):
    assert x.ndim == 3, f"number of dims in input x should be 3 (B, T, D), instead is {x.ndim}"
    y = self.decoder(x, 2)
    if not training:
      y_pred = self.scale_fn(y[self.space_key].pred())
      return y_pred
    else:
      return y

class ROA_CNN_Estimator(nj.Module):
  """
  A flexible and customizable ROA Latent Estimator using CNNs.

  The architecture of the convolutional layers can be specified via a
  configuration list during initialization, allowing for easy experimentation
  with different network depths and sizes.
  """

  act: str = 'gelu'
  symlog: bool = True

  def __init__(self, privileged_latent_dim, cnn_layers_config=None, **kw):
    """
    Initializes the customizable CNN estimator.

    Args:
      privileged_latent_dim (int): The dimension of the final output latent space.
      cnn_layers_config (list[dict], optional): A list of dictionaries, where
        each dictionary specifies the parameters for one Conv1D layer.
        Each dict should contain 'features', 'kernel_size', and 'strides'.
        If None, a default architecture is used. Defaults to None.
      **kw: Additional keyword arguments.
    """
    # super().__init__()
    self.privileged_latent_dim = privileged_latent_dim
    self.kw = kw

    # If no custom configuration is provided, use a default architecture
    # that matches your original implementation.
    if cnn_layers_config is None:
      self.cnn_layers_config = [
          {'depth': 32, 'kernel': 3, 'stride': 2, 'act':self.act},
          {'depth': 64, 'kernel': 3, 'stride': 2, 'act':self.act},
          {'depth': 128, 'kernel': 3, 'stride': 1, 'act':self.act},
      ]
    else:
      self.cnn_layers_config = [cnn_layers_config[f"l{i}"] for i in range(len(cnn_layers_config))]
    
  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, window_obs):
    """
    Forward pass of the CNN estimator.

    Args:
      window_obs (jnp.ndarray): Input tensor of shape (B, T, D).

    Returns:
      jnp.ndarray: Output tensor of shape (B, privileged_latent_dim).
    """
    squish = nn.symlog if self.symlog else lambda x: x
    bshape = window_obs.shape[0]
    
    # Ensure input is in the correct format
    x = squish(window_obs.astype(jnp.bfloat16))

    # --- Dynamically build and apply the convolutional stack ---
    # This loop iterates through your configuration and creates a layer for each entry.
    for i, layer_config in enumerate(self.cnn_layers_config):
      # Use self.sub() to create and manage the submodule
      # The layer_config dictionary is unpacked to pass its key-value pairs
      # as arguments to the nn.Conv1D constructor.
      conv_layer = self.sub(f'conv_{i}', nn.Conv1D, **layer_config)
      x = conv_layer(x)

    x = jnp.mean(x, axis=1) # global pooling
    # Final linear layer to project to the desired latent dimension
    y = self.sub('mlp_out', nn.Linear, self.privileged_latent_dim)(x)

    return y

class ROA_LargeCNN_Estimator(nj.Module):
  """
  Aligned CNN estimator, with dilation, attention and multiscale.
  """

  def __init__(self, 
               privileged_latent_dim,
               feature_dim, 
               timestep_dim, 
               output_dim, 
               symlog=True,
               num_layers=None,
               base_channels=32,
               channel_multiplier=1.5,
               activation_fn="elu",
               dropout_prob=0.2,
               layer_configs=None):

    self.feature_dim = feature_dim
    self.timestep_dim = timestep_dim
    self.output_dim = privileged_latent_dim
    self.base_channels = base_channels
    self.channel_multiplier = channel_multiplier
    self.activation_fn = activation_fn
    self.dropout_prob = dropout_prob
    self.squish = nn.symlog if symlog else lambda x: x

    if layer_configs is None:
      layer_configs = self._get_default_layer_configs(num_layers, timestep_dim)
    
    if num_layers is None:
      self.num_layers = len(layer_configs)
    else:
      assert num_layers == len(layer_configs)
      self.num_layers = num_layers
      
    self.layer_configs = layer_configs

    # Calculate channel progression
    self.channels = [feature_dim]
    for i in range(self.num_layers):
      next_channels = int(base_channels * (channel_multiplier ** i))
      self.channels.append(next_channels)

  def _get_default_layer_configs(self, num_layers, timestep_dim):
    configs = []
    for i in range(num_layers):
      if i == 0:
        config = {
            'use_multiscale': True,
            'use_dilation': False,
            'use_attention': False,
            'multiscale_kernels': (3, 5, 7)
        }
      elif i < num_layers - 1:
        config = {
            'use_multiscale': False,
            'use_dilation': True,
            'use_attention': bool((i % 2 == 1) * (i != 1)),
        }
      else:
        config = {
            'use_multiscale': False,
            'use_dilation': False,
            'use_attention': True,
        }
      configs.append(config)
    return configs

  def _get_dilations_for_layer(self, layer_idx, timestep_dim):
    if timestep_dim < 15:
      max_dilation_power = 1
    elif timestep_dim < 30:
      max_dilation_power = 2
    elif timestep_dim < 60:
      max_dilation_power = 3
    else:
      max_dilation_power = min(4 + layer_idx, 6)
    
    dilations = tuple(2 ** p for p in range(max_dilation_power + 1))
    return dilations

  def __call__(self, x, training=True):
    # x: (B, T, F)
    # No transpose needed as we work in NTC

    x = self.squish(x.astype(jnp.bfloat16))
    
    for i in range(self.num_layers):
      config = self.layer_configs[i]
      dilations = self._get_dilations_for_layer(i, self.timestep_dim)
      
      layer = self.sub(f'layer_{i}', nn.UnifiedConvLayer,
          in_channels=self.channels[i],
          out_channels=self.channels[i+1],
          kernel_size=3,
          use_multiscale=config.get('use_multiscale', True),
          use_dilation=config.get('use_dilation', True),
          use_attention=config.get('use_attention', True),
          multiscale_kernels=config.get('multiscale_kernels', (3, 5, 7)),
          dilations=dilations,
          dropout_prob=self.dropout_prob if i > 0 else 0.0,
          activation_fn=self.activation_fn
      )
      x = layer(x, training=training)

    # Global pooling
    x = x.mean(axis=1) # (B, C)
    
    # Classifier
    final_channels = self.channels[-1]
    x = self.sub('cls_fc1', nn.Linear, final_channels // 2)(x)
    x = self.sub('cls_bn', nn.Norm, 'layer')(x)
    x = nn.act(self.activation_fn)(x)
    x = nn.dropout(x, self.dropout_prob, training)
    x = self.sub('cls_fc2', nn.Linear, self.output_dim)(x)
    
    return x

class ROA_RSSM_Estimator(nj.Module):
  """
  Wrapper over the RSSM to use it as an estimator.  
  """

  def __init__(self, rssm_estimator_cfg):
    
    self.rssm = RSSM_ROA(**rssm_estimator_cfg, submodule=True)
    self.scales = rssm_estimator_cfg.loss_scales

  def __call__(self, x):
    """
    Given an input tensor of obs (B, T, D), return an estimated latent 

    1. initalise the latent state 
    2. convert b,t,d to dreamer format 
   
    """
    assert x.ndim == 3, f'expecting ndim = 3, instead got {x.ndim}'
    B,T,D = x.shape
    data = self.rssm.dreamer_format_array(x)
    assert len(data) == 2, f"x is too short, expecting length 14+3 but got {x.shape[-1]}"

    obs, act = data
    carry, _  = self.rssm.init_carry(B)
    _, preds, _ = self.rssm._forward(carry, obs, act, False)
    priv_vector = preds['vector']
    return priv_vector[:, -1]
  
  def loss(self, obs, prevact):
    """    
    Computes the estimator loss using the loss objective of the RSSM, i.e.
    a weighting of the reconstruction loss and KL loss.
    """
    B, T, _ = obs['vector'].shape
    carry, _ = self.rssm.init_carry(B)
    _, (_, _, outs, _) = self.rssm._loss(carry, obs, prevact, vector_noise=0, priv_vector_noise=0, training=True, step=0)

    return {k:v*self.scales[k] for k, v in outs['losses'].items()}

class ROA_RNN_Estimator(nj.Module):
  """
  RNN based estimator, using a GRU. 
  """

  symlog: bool = True
  
  def __init__(self, privileged_latent_dim, gru_cfg):

    self.privileged_latent_dim = privileged_latent_dim
    self.gru_cfg = gru_cfg
  
  def __call__(self, x):

    assert x.ndim == 3, f'expecting ndim = 3, instead got {x.ndim}'
    B,T,D = x.shape
    assert D == 17, 'expecting input of shape (B, T, 17)'
    x = nn.symlog(x.astype(jnp.bfloat16)) if self.symlog else x.astype(jnp.bfloat16)

    gru = self.sub('gru', nn.GRU, **self.gru_cfg)
    h = gru.initial(B)

    resets = jnp.zeros((B, T), dtype=jnp.bool_)
    h, y_gru = gru(h, x, resets)
    assert y_gru.shape == (B, T, self.gru_cfg['units']), f'expected shape (B, T, {self.gru_cfg["units"]}), instead got {y_gru.shape}'
    y_mlp = self.sub('mlp', nn.Linear, self.privileged_latent_dim)(y_gru)
    assert y_mlp.shape == (B, T, self.privileged_latent_dim), f'expected shape (B, T, {self.privileged_latent_dim}), instead got {y_mlp.shape}'
    return y_mlp

    
class BayesianSetEncoder(nj.Module):
  """
  JAX/ninjax port of the HiP-RSSM SetEncoder.

  Computes a posterior distribution over a latent task variable z from
  context pairs (x, y) using Bayesian Aggregation (BA). Can be used as
  a drop-in replacement for the ROA estimator in the RSSM pipeline.

  Architecture (mirrors PyTorch SetEncoder exactly):
    1. Flatten (B, T, D) → (B*T, D), concatenate x and y
    2. MLP with ReLU activations
    3. Optional L2 normalisation ("pre" on hidden, "post" on mean)
    4. Split into mean and log-variance heads
    5. Variance activation (softplus + epsilon, or elup1)
    6. Reshape → (B, T, lod)
    7. Bayesian Aggregation (or Mean Aggregation) over T → (B, lod)

  Usage as ROA estimator:
    The __call__ method accepts x of shape (B, T, obs_dim + act_dim).
    It splits obs from act internally, constructs context_X = x and
    context_Y = x[..., :target_dim], and returns mu_z of shape (B, lod).
  """

  symlog: bool = False

  def __init__(self, input_dim, lod, hidden_units,
               output_normalization='post',
               aggregator='BA',
               variance_act='softplus'):
    """
    Args:
      input_dim: Total input dimension (context_X + context_Y concatenated).
                 For the RSSM ablation this is (obs_dim + act_dim) + obs_dim = 2*obs_dim + act_dim.
      lod: Latent output dimension (task_dim / 2 in HiP-RSSM terms —
           the dimension of mu_z and cov_z each).
      hidden_units: List of hidden layer sizes, e.g. [128, 128].
      output_normalization: 'pre', 'post', or 'none'.
      aggregator: 'BA' (Bayesian Aggregation) or 'MA' (Mean Aggregation).
      variance_act: 'softplus' or 'elup1'.
    """
    self.input_dim = input_dim
    self.lod = lod
    self.hidden_units = hidden_units if isinstance(hidden_units, (list, tuple)) else list(hidden_units)
    self.output_normalization = output_normalization
    self.aggregator = aggregator
    self.variance_act = variance_act

  def __call__(self, x, y=None):
    """
    Forward pass.

    Can be called in two ways:
      1. set_encoder(x, y)  — explicit context_X and context_Y
      2. set_encoder(x)     — x is (B, T, obs+act), y is derived as x[..., :target_dim]
         (for ROA-style interface where a single obs+act vector is provided)

    Args:
      x: (B, T, x_dim) context inputs  (e.g. [obs, act])
      y: (B, T, y_dim) context outputs  (e.g. obs). If None, derived from x.

    Returns:
      mu_z:  (B, lod) — aggregated posterior mean
      cov_z: (B, lod) — aggregated posterior covariance (None if MA)
    """
    B, T, x_dim = x.shape

    if y is None:
      # ROA-style: x is the full (obs+act) vector; y is the obs portion
      # input_dim = x_dim + y_dim, and x_dim = obs_dim + act_dim, y_dim = obs_dim
      # so y_dim = input_dim - x_dim
      y_dim = self.input_dim - x_dim
      y = x[..., :y_dim]

    y_dim = y.shape[-1]

    # 1. Flatten: (B, T, D) → (B*T, D)
    x_flat = x.reshape(B * T, x_dim)
    y_flat = y.reshape(B * T, y_dim)

    # 2. Concatenate x and y
    h = jnp.concatenate([x_flat, y_flat], axis=-1)  # (B*T, input_dim)
    h = h.astype(nn.COMPUTE_DTYPE)

    # 3. Hidden layers: Linear + ReLU
    for i, units in enumerate(self.hidden_units):
      h = self.sub(f'hidden_{i}', nn.Linear, units)(h)
      h = jax.nn.relu(h)

    # 4. Pre-normalisation (L2 on hidden representation)
    if self.output_normalization.lower() == 'pre':
      h = h / (jnp.linalg.norm(h, ord=2, axis=-1, keepdims=True) + 1e-8)

    # 5. Mean head
    mean = self.sub('mean_layer', nn.Linear, self.lod)(h)

    # 6. Post-normalisation (L2 on mean)
    if self.output_normalization.lower() == 'post':
      mean = mean / (jnp.linalg.norm(mean.astype(jnp.float32), ord=2, axis=-1, keepdims=True) + 1e-8)

    # 7. Variance head
    log_var = self.sub('log_var_layer', nn.Linear, self.lod)(h)
    if self.variance_act == 'softplus':
      var = jax.nn.softplus(log_var) + 0.0001
    else:
      # elup1: x >= 0 → x+1, x < 0 → exp(x)
      var = jnp.where(log_var >= 0, log_var + 1.0, jnp.exp(log_var))

    # 8. Reshape: (B*T, lod) → (B, T, lod)
    mean = mean.reshape(B, T, self.lod)
    var = var.reshape(B, T, self.lod)

    # 9. Aggregate (in float32 for numerical stability)
    mu_z, cov_z = self._aggregate(mean.astype(jnp.float32),
                                   var.astype(jnp.float32))
    return mu_z, cov_z

  def _aggregate(self, obs_mean, obs_cov):
    """
    Bayesian Aggregation or Mean Aggregation over the time dimension.

    Mirrors torch SetEncoder.aggregate exactly:
      prior:  N(0, I)
      v = obs_mean − 0
      cov_w_inv = 1 / obs_cov
      cov_z  = 1 / (1 + Σ_t cov_w_inv_t)
      mu_z   = 0 + cov_z · Σ_t (cov_w_inv_t · v_t)

    Args:
      obs_mean: (B, T, lod)  per-point means
      obs_cov:  (B, T, lod)  per-point variances

    Returns:
      mu_z:  (B, lod)
      cov_z: (B, lod) or None
    """
    if self.aggregator == 'BA':
      # Prior N(0, I)
      initial_mean = jnp.zeros((1, 1, self.lod))  # (1, 1, lod)
      initial_cov  = jnp.ones((1, 1, self.lod))   # (1, 1, lod)

      v = obs_mean - initial_mean                   # (B, T, lod)
      cov_w_inv = 1.0 / obs_cov                     # (B, T, lod)

      # Bayesian update — aggregate all T observations
      cov_z_new = 1.0 / (1.0 / initial_cov + jnp.sum(cov_w_inv, axis=1, keepdims=True))
      mu_z_new  = initial_mean + cov_z_new * jnp.sum(cov_w_inv * v, axis=1, keepdims=True)

      return jnp.squeeze(mu_z_new, 1), jnp.squeeze(cov_z_new, 1)

    elif self.aggregator == 'MA':
      mu_z_new = jnp.mean(obs_mean, axis=1)
      return mu_z_new, None

    else:
      raise ValueError(f"Unknown aggregator: {self.aggregator}")

   
def feat2tensor(x:Dict):
  assert 'deter' in x and 'stoch' in x
  return jnp.concatenate([nn.cast(x['deter']), 
                          nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

def huber_fn(x, y, delta):
    diff = x - y
    abs_diff = jnp.abs(diff)
    quad = 0.5 * diff**2
    lin = delta * (abs_diff - 0.5 * delta)
    return jnp.where(abs_diff <= delta, quad, lin)