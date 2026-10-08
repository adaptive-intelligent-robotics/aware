import numpy as np
import logging; logging.basicConfig(
    level=logging.INFO); pylogger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.jit
import torch.jit
import torch.nn.functional as F
import einops
import math
from datetime import datetime
from omegaconf import ListConfig
import hydra
from omegaconf import OmegaConf
from functools import partial

import os
from aware import REPO_ROOT as path_to_root

# from agents.siMLPe_mlp import build_mlps_wrapper
from aware.utils.training import load_normalisation, apply_norm, revert_norm
from aware.utils.trajectory import data_to_trajectory
from aware.utils.jax import torch_to_jax, jax_to_torch
from aware.agents.mixed_action import MixedOperatorAgent
import aware.agents.networks.transformer as nets
from aware.agents.networks.simple import get_loss_fn, make_mlp, LargeCNN1D
from aware.agents.dataloader import DataBuffer, DataLoadWrapper

from aware.agents.networks.simple import PrivilegedContrastiveLoss
from aware.agents.networks.simple import InfoNCEPrivilegedLoss
from aware.agents.networks.simple import LinearSimilarityContrastiveLoss

# --- useful components --- #

def anneal(i, params):
  (xmin, xmax), (imin, imax) = params
  if i < imin: return xmin
  elif i > imax: return xmax
  else: 
    t = ((i - imin) / (imax - imin))
    return xmin + (xmax - xmin) * t

def infer_annealing(x, i):
  """
  Infer the value of alpha for a given setting 'x'
  """
    
  if isinstance(x, (tuple, list, ListConfig)):
    return anneal(i, x)
  else:
    return float(x)

def torch_rand_student_t(df, shape, device, dtype):
  """
  Generates Student's t-samples with unit variance (Var=1) using torch.distributions.
  """
  loc = torch.tensor(0.0, device=device, dtype=dtype)
  scale = torch.tensor(1.0, device=device, dtype=dtype)
  
  # --- CRITICAL CORRECTION ---
  # To normalize the variance to 1.0, we divide by the natural variance's square root.
  if df > 2:
      # Scale = sqrt(1 / Var(T_df)) = sqrt((df - 2) / df)
      scale = torch.sqrt(torch.tensor((df - 2) / df, device=device, dtype=dtype))

  # Initialize the location-scale distribution
  t_dist = torch.distributions.StudentT(
      df=df, 
      loc=loc, 
      scale=scale
  )
  
  # Sample the required shape (PyTorch's sample method takes the shape tuple)
  t_samples = t_dist.sample(sample_shape=shape)
  
  # Ensure device and dtype are correct (though often inherited)
  return t_samples.to(device, dtype)

def np_rand_student_t(df, shape, rng, dtype):
  """
  Generates Student's t-samples with unit variance (Var=1) using numpy.
  """
  # Sample from a Gamma distribution (chi-squared is a special case)
  # Gamma(df/2, df/2)
  gamma_samples = rng.gamma(df / 2.0, 2.0 / df, size=shape).astype(dtype)
  
  # The t-distribution is N(0, 1) / sqrt(Chi^2(df) / df)
  # Chi^2(df) / df is the same as our Gamma(df/2, df/2) sample?
  # Wait, numpy gamma is shape, scale. 
  # Chi-sq(df) is Gamma(df/2, 2).
  # We want X ~ Gamma(df/2, 2/df) so that E[X] = 1.
  
  # Standard normal samples
  gaussian_samples = rng.standard_normal(shape, dtype=dtype)
  
  # Student-t samples
  t_samples = gaussian_samples / np.sqrt(gamma_samples + 1e-9) # Add epsilon for stability
  
  # We must re-scale the variance. The variance of a t-dist is df/(df-2)
  # We scale it to have a variance of 1, to be a drop-in replacement for randn
  if df > 2:
      scale_factor = np.sqrt((df - 2) / df)
      t_samples = t_samples * scale_factor
  
  return t_samples
  
class NoiseGenerator:

  def __init__(self,

               # scale the noise generator
               std_scaling=1.0,               # scale factor applied to all incoming stddevs

               # enable/disable kinds of noise
               enable_gaussian_ou=False,      # use mixed gaussian/ornstein-uhlenbeck noise
               enable_offset=False,           # use random offsets to shift input data
               enable_spikes=False,           # random data spikes injected
               enable_fixed_readings=False,   # fix readings exactly randomly

               # determine how the noise is spread over the data, shape: (batch, timesteps, features)
               individualised_noise_std_range=(0.0, 1.0), # range of full stddev to uniformly sample for below two settings
               set_noise_per_batch=True,      # use different noise levels for each trajectory in batch
               set_noise_per_timestep=False,  # use different noise levels for each timestep in a trajectory
               set_noise_per_feature=False,   # use different noise levels for each feature dimension
               
               # control differenation of qpos noise
               enable_derivative_noise=False, # differentiate qpos to get qvel noise
               derivative_alpha=[0.1, 1.0],   # weight the differentiated noise vs regular
               derivative_dt=0.02,            # timestep to use for differentiation

               # control gaussian/ou noise
               alpha_range=(0.1, 0.9),        # range of split between gaussian/ou
               ou_theta=15.0,                 # mean tendency of ou noise (higher->gaussian)
               ou_dt=0.05,                    # timestep of ou, ratio random/drift = exp(-theta*dt)
               corr_rho=(0.5, 1.0),           # correlation coefficient if doing correlated noise

               # control differential noise
               differential_noise_range=[0.0, 1.0], # weighting of differential noise
               differential_dt=0.05,          # timestep (note this is not normalised)

               # control mean offset noise
               p_offset=0.25,                 # probability of a mean offset occuring
               offset_scale=1.0,              # mean offset scale relative to noise stddev

               # control spike noise    
               p_spike=0.01,                  # probability of large spike noise
               spike_max_duration=5,          # largest no. steps a spike can last for
               spike_magnitude_mean=10.0,     # size of spike times regular stddev (mean)
               spike_magnitude_std=2.5,       # size of spike times regular stddev (std)

               # control stuck sensor noise
               p_fixed=0.01,
               fixed_reading_max_duration=5,  # longest readings can be stuck (min=2)
               fixed_reading_identical=True,  # fixed readings have no other noise on them

               # np seed
               np_seed=123,

               **unused_args,
               ):
    """
    Class to generate realistic noise onto training signals.
    """

    if unused_args:
      pylogger.info("NoiseGenerator.__init__ got unexpected arguments, which will be ignored: " + \
                    str([key for key in unused_args.keys()]))
    
    # save input arguments
    self.std_scaling = std_scaling
    self.enable_gaussian_ou = enable_gaussian_ou
    self.enable_offset = enable_offset
    self.enable_spikes = enable_spikes
    self.enable_fixed_readings = enable_fixed_readings
    self.individualised_noise_std_range = individualised_noise_std_range
    self.set_noise_per_batch = set_noise_per_batch
    self.set_noise_per_timestep = set_noise_per_timestep
    self.set_noise_per_feature = set_noise_per_feature
    self.alpha_range = alpha_range
    self.ou_theta = ou_theta
    self.ou_dt = ou_dt
    self.corr_rho = corr_rho
    self.differential_noise_range = differential_noise_range
    self.dt = differential_dt
    self.p_offset = p_offset
    self.offset_scale = offset_scale
    self.p_spike = p_spike
    self.spike_max_duration = spike_max_duration
    self.spike_magnitude_mean = spike_magnitude_mean
    self.spike_magnitude_std = spike_magnitude_std
    self.p_fixed = p_fixed
    self.fixed_reading_max_duration = fixed_reading_max_duration
    self.fixed_reading_identical = fixed_reading_identical

    # save derivative noise args
    self.enable_derivative_noise = enable_derivative_noise
    self.derivative_alpha = derivative_alpha
    self.derivative_dt = derivative_dt

    # student t distribution
    self.enable_t_noise = False # disabled it only in 'add_noise_new'
    self.student_t_df = 2
    
    # additional hardcoded parameters
    self.enable_dt_jitter = True
    self.dt_jitter_proportion = 0.1
    self.qvel_noise_proportion = 0.05 # % of qvel noise std to be random (not differentiated)

    # support annealing schedule
    self.counter = 0
    self.current_std_scaling = 0.0
    self.increment(i=0)

    # numpy rng generator 
    self.rng = np.random.default_rng(np_seed)

  def increment(self, i=None):
    """
    Increment a step counter, only useful in the case where parameters
    are on an annealing schedule. Supports passing i or incrementing
    with +1 each time (if i=None).
    """

    if i is None:
      self.counter += 1
    else:
      self.counter = i

    # anneal the noise standard deviation scaling
    self.current_std_scaling = infer_annealing(self.std_scaling,
                                               i=self.counter)

  # paper baseline stable implementation
  def add_noise(self,
                observation: torch.Tensor, 
                std_dev: torch.Tensor, 
                enable_correlation: bool = False,
                corr_split_idx: torch.Tensor = None,
                ) -> torch.Tensor:
    """
    A generic function to apply configurable noise to a batch of observations.
    Optimized to pre-calculate random distributions outside of loops.
    """
    B, T, N = observation.shape
    device = observation.device
    dtype = observation.dtype

    # Reshape std_dev for broadcasting
    if std_dev.numel() == 1:
      total_std = std_dev.view(1, 1, 1).expand(1, 1, N)
    else:
      total_std = std_dev.to(device, dtype).view(1, 1, N)

    # multiply std tensor by scale factor
    total_std = self.current_std_scaling * total_std

    # handle applying randomisation per feature or per batch
    if (self.set_noise_per_feature or self.set_noise_per_batch or
        self.set_noise_per_timestep):
      # will we take randomised values over batch or features
      B_dim = B if self.set_noise_per_batch else 1
      T_dim = T if self.set_noise_per_timestep else 1
      N_dim = N if self.set_noise_per_feature else 1
      # uniformly sample random values over the specified dimensions [0.0, 1.0)
      rand_values = torch.rand((B_dim, T_dim, N_dim), device=device, dtype=dtype)
      # convert to range [min, max) with equation: (max - min) * x + min
      std_scalings = (((self.individualised_noise_std_range[1] 
                      - self.individualised_noise_std_range[0]) * rand_values)
                      + self.individualised_noise_std_range[0])
      # for correlated noise, share the std scaling across both
      if self.set_noise_per_feature and enable_correlation:
        std_scalings[:, :, corr_split_idx : N] = std_scalings[:, :, 0 : corr_split_idx]
      total_std = total_std * std_scalings

    # save the last set of standard deviations used, in case we want to check noise
    self.last_std_used = total_std.clone()

    total_noise = torch.zeros_like(observation)

    # --- 1. Correlated Student's t and Ornstein-Uhlenbeck (OU) Noise --- #
    if self.enable_gaussian_ou:
      alpha = torch.rand(B, 1, 1, device=device, dtype=dtype) * \
              (self.alpha_range[1] - self.alpha_range[0]) + \
              self.alpha_range[0]

      std_g = torch.sqrt(alpha) * total_std
      std_ou = torch.sqrt(1 - alpha) * total_std

      # a) Generate Student's t noise (heavy-tailed replacement for Gaussian)
      if not enable_correlation:
        # Uncorrelated Student's t noise
        gaussian_noise = torch_rand_student_t(
            self.student_t_df, (B, T, N), device=device, dtype=dtype
        ) * std_g
        total_noise += gaussian_noise
      else:
        # Correlated Student's t noise (Position/Velocity case)
        split_idx = corr_split_idx
        assert N == split_idx * 2, "For correlation, feature dimension N must be 2 * corr_split_idx"
        
        std_g_pos, std_g_vel = std_g[..., :split_idx], std_g[..., split_idx:]

        # 1. GENERATE BASE QPOS NOISE
        base_g_noise_pos = torch_rand_student_t(
            self.student_t_df, (B, T, split_idx), device=device, dtype=dtype
        )
        
        # QPOS NOISE is always based on the independent base noise
        gaussian_noise_pos = base_g_noise_pos * std_g_pos

        # --- DIFFERENTIAL NOISE LOGIC ---
        diff_range = self.differential_noise_range
        rand_prop_value = torch.rand(B, 1, 1, device=device, dtype=dtype)
        differential_noise_proportion = (((diff_range[1] - diff_range[0]) * rand_prop_value)
                                        + diff_range[0])
        
        # 2. CALCULATE DIFFERENTIAL QVEL NOISE (eta_diff)
        qpos_noise_shifted = torch.cat([torch.zeros(B, 1, split_idx, device=device, dtype=dtype), 
                                        gaussian_noise_pos[:, :-1, :]], dim=1)
        differential_qvel_noise = (gaussian_noise_pos - qpos_noise_shifted) / self.dt

        # --- CORRECTED SCALING LOGIC ---
        diff_noise_std = differential_qvel_noise.std(dim=1, keepdim=True)
        diff_noise_std = diff_noise_std.clamp(min=1e-9) # Handle zero std
        normalized_diff_noise = differential_qvel_noise / diff_noise_std
        differential_qvel_noise = normalized_diff_noise * std_g_vel
        # --- END CORRECTED SCALING LOGIC ---
        
        # 3. CALCULATE STANDARD CORRELATED QVEL NOISE (eta_corr)
        corr_rand = torch.rand((B, 1, split_idx), device=device, dtype=dtype)
        corr_rho = (((self.corr_rho[1] 
                    - self.corr_rho[0]) * corr_rand)
                    + self.corr_rho[0])
        
        # Correlated part of velocity noise (uses base_g_noise_pos)
        base_g_noise_vel_indep = torch_rand_student_t(
            self.student_t_df, (B, T, split_idx), device=device, dtype=dtype
        )
        base_g_noise_vel_corr = corr_rho * base_g_noise_pos + torch.sqrt(1 - corr_rho**2) * base_g_noise_vel_indep
        standard_correlated_qvel_noise = base_g_noise_vel_corr * std_g_vel
        
        # 4. MIX THE TWO NOISE SOURCES
        p = differential_noise_proportion.expand(B, T, split_idx)
        gaussian_noise_vel = (p * differential_qvel_noise + 
                              (1 - p) * standard_correlated_qvel_noise)
        
        total_noise += torch.cat([gaussian_noise_pos, gaussian_noise_vel], dim=-1)

        corr_rho = einops.rearrange(corr_rho, "b 1 t -> b t")
        self.last_diff_proportion_used = differential_noise_proportion.clone()

      # b) Generate Ornstein-Uhlenbeck noise (now based on Student's t)
      theta, dt = self.ou_theta, self.ou_dt
      ou_noise = torch.zeros_like(observation)
      
      exp_factor = torch.exp(torch.tensor(-theta * dt, device=device, dtype=dtype))
      sqrt_factor = torch.sqrt(1 - torch.exp(torch.tensor(-2 * theta * dt, device=device, dtype=dtype)))
      
      # --- OPTIMIZATION START: Pre-calculate noise distribution tensors ---
      # We need noise for T-1 steps. We generate it all at once to avoid 
      # creating the distribution object inside the loop.
      
      if not enable_correlation:
        # Pre-generate noise for all timesteps at once
        # Shape: (B, T-1, N)
        precalced_ou_noise = torch_rand_student_t(
            self.student_t_df, (B, T - 1, N), device=device, dtype=dtype
        )
      else:
        split_idx = corr_split_idx
        # Pre-generate noise for all timesteps at once for both components
        # Shape: (B, T-1, split_idx)
        precalced_base_pos = torch_rand_student_t(
            self.student_t_df, (B, T - 1, split_idx), device=device, dtype=dtype
        )
        precalced_base_vel_indep = torch_rand_student_t(
            self.student_t_df, (B, T - 1, split_idx), device=device, dtype=dtype
        )
      # --- OPTIMIZATION END ---

      for t in range(T - 1):
        if not enable_correlation:
          # Uncorrelated OU noise
          noise_factor = std_ou * sqrt_factor
          
          # Use pre-calculated noise for this timestep
          # Note: original code used noise_factor[:, 0, :], effectively locking std 
          # to the first timestep for the OU process. We preserve this behavior.
          random_comp = precalced_ou_noise[:, t, :] * noise_factor[:, 0, :]
          
          ou_noise[:, t + 1, :] = ou_noise[:, t, :] * exp_factor + random_comp
        else:
          # Correlated OU noise
          std_ou_pos, std_ou_vel = std_ou[..., :split_idx], std_ou[..., split_idx:]
          noise_factor_pos = std_ou_pos * sqrt_factor
          noise_factor_vel = std_ou_vel * sqrt_factor

          # Grab pre-calculated noise for this timestep
          base_ou_noise_pos_t = precalced_base_pos[:, t, :]
          base_ou_noise_vel_indep_t = precalced_base_vel_indep[:, t, :]
          
          base_ou_noise_vel_corr_t = corr_rho * base_ou_noise_pos_t + torch.sqrt(1 - corr_rho**2) * base_ou_noise_vel_indep_t
          
          ou_noise_pos_t, ou_noise_vel_t = ou_noise[:, t, :split_idx], ou_noise[:, t, split_idx:]
          
          # Preserving original behavior: using index 0 for noise factors
          next_ou_noise_pos = ou_noise_pos_t * exp_factor + base_ou_noise_pos_t * noise_factor_pos[:, 0, :]
          standard_next_ou_noise_vel = ou_noise_vel_t * exp_factor + base_ou_noise_vel_corr_t * noise_factor_vel[:, 0, :]
          
          # --- OU DIFFERENTIAL NOISE ---
          dt_ou = dt
          differential_next_ou_noise_vel = (next_ou_noise_pos - ou_noise_pos_t) / dt_ou

          # --- CORRECTED SCALING LOGIC ---
          diff_noise_std_ou = differential_next_ou_noise_vel.std(dim=0, keepdim=True)
          diff_noise_std_ou = diff_noise_std_ou.clamp(min=1e-9) # Handle zero std
          normalized_diff_noise_ou = differential_next_ou_noise_vel / diff_noise_std_ou
          # Preserving original behavior: using index 0 for std_ou_vel
          differential_next_ou_noise_vel = normalized_diff_noise_ou * std_ou_vel[:, 0, :]
          # --- END CORRECTED SCALING LOGIC ---

          # MIX THE TWO OU NOISE SOURCES
          p_ou = self.last_diff_proportion_used.squeeze(-1)
          next_ou_noise_vel = (p_ou * differential_next_ou_noise_vel + 
                                (1 - p_ou) * standard_next_ou_noise_vel)
          
          ou_noise[:, t + 1, :] = torch.cat([next_ou_noise_pos, next_ou_noise_vel], dim=-1)
    
      total_noise += ou_noise

    # --- 2. Persistent 'Zero-Error' Offsets --- #
    if self.enable_offset:
        
      offset_mask = (torch.rand(B, 1, N, device=device, dtype=dtype) < self.p_offset).float()
      # Base this on Student's t as well for heavy-tailed offsets
      offset = torch_rand_student_t(
          self.student_t_df, (B, 1, N), device=device, dtype=dtype
      ) * total_std * self.offset_scale
      total_noise += offset * offset_mask

    # --- 3. Random Spikes --- #
    if self.enable_spikes:

      spike_mask = (torch.rand(B, 1, N, device=device, dtype=dtype) < self.p_spike).float()
      
      # --- CHANGED: Use Laplace distribution for heavy-tailed spikes ---
      laplace_dist = torch.distributions.laplace.Laplace(
          self.spike_magnitude_mean, # loc
          self.spike_magnitude_std   # scale (std = scale * sqrt(2))
      )
      # Sample and ensure shape is (B, 1, N)
      spike_rand_scale = laplace_dist.sample((B, 1, N)).to(device, dtype).squeeze(-1)
      # --- END CHANGED ---

      spike_magnitudes = spike_rand_scale * total_std
      
      max_duration = self.spike_max_duration
      spike_durations = torch.randint(1, max_duration + 1, (B, 1, N), device=device)
      
      start_indices = torch.randint(0, T, (B, 1, N), device=device)
      end_indices = torch.min(start_indices + spike_durations, torch.tensor(T, device=device))
      
      time_indices = torch.arange(T, device=device, dtype=torch.long).view(1, T, 1).expand(B, -1, -1)
      spike_time_mask = (time_indices >= start_indices) & (time_indices < end_indices)
      
      spike_tensor = spike_magnitudes * spike_time_mask
      total_noise += spike_tensor * spike_mask

    # --- 4. 'Stuck' readings --- #
    if self.enable_fixed_readings:

      # determine which features in which batch will get a fixed event during their timesteps
      fixed_mask = (torch.rand(B, 1, N, device=device, dtype=dtype) < self.p_fixed)
      fixed_durations = torch.randint(2, self.fixed_reading_max_duration + 1, (B, 1, N), device=device)
      
      start_indices = torch.randint(0, T, (B, 1, N), device=device)
      end_indices = torch.min(start_indices + fixed_durations, torch.tensor(T, device=device))

      time_indices = torch.arange(T, device=device, dtype=torch.long).view(1, T, 1).expand(B, -1, -1)
      fixed_time_mask = (time_indices >= start_indices) & (time_indices < end_indices)

      hold_values = torch.gather(observation, 1, start_indices.expand(-1, T, -1)) # Gather all start values
      hold_values = hold_values[:, 0, :].unsqueeze(1) # Select the first one, shape (B, 1, N)

      # Combine the "does this feature drop?" mask with the "is this timestep dropping?" mask
      final_stuck_mask = (fixed_time_mask & fixed_mask)

      if self.fixed_reading_identical:
        hold_noise_values = torch.gather(total_noise, 1, start_indices.expand(-1, T, -1))
        hold_noise_values = hold_noise_values[:, 0, :].unsqueeze(1) # Shape (B, 1, N)
        target_stuck_value = hold_values + hold_noise_values
        
        # N_replace = (hold_obs + hold_noise) - Obs_current
        # (B, 1, N) - (B, T, N) -> (B, T, N)
        replacement_noise = target_stuck_value - observation
          
      else:
        # Target is the held obs + the *current* noise
        # N_replace = (hold_obs + N_current) - Obs_current
        # (B, 1, N) + (B, T, N) - (B, T, N) -> (B, T, N)
        replacement_noise = (hold_values + total_noise) - observation
          
      # Now, apply the calculated replacement noise to the total_noise tensor
      total_noise = torch.where(final_stuck_mask, 
                              replacement_noise, 
                              total_noise)

    return observation + total_noise

  # new function to generate clean white noise for the privileged information
  def add_white_noise(self,
                      observation: torch.Tensor, 
                      std_dev: torch.Tensor, 
                      enable_correlation: bool = False,
                      corr_split_idx: torch.Tensor = None,
                      ) -> torch.Tensor:
    """
    A generic function to apply configurable noise to a batch of observations.

    Args:
        observation (torch.Tensor): The input tensor with shape (B, T, N).
        std_dev (torch.Tensor): Standard deviations for the noise. Can be a scalar
                                or a tensor of shape (N,).
        enable_correlation (torch.Tensor): option to enable correlation betweeen features.
        corr_split_idx (torch.Tensor): index that seperates the base features from
                                       the correlated features, eg obs=[qpos, qvel],
                                       this should be N/2. qpos=base, qvel=correlated

    Returns:
        torch.Tensor: The observation tensor with added noise.
    """
    B, T, N = observation.shape
    device = observation.device
    dtype = observation.dtype

    # Use standard Gaussian (Normal) distribution for simple white noise
    def rand_sampler(shape):
      # torch.randn naturally gives unit variance
      return torch.randn(shape, device=device, dtype=dtype)

    # Reshape std_dev for broadcasting
    if std_dev.numel() == 1:
      total_std = std_dev.view(1, 1, 1).expand(1, 1, N)
    else:
      total_std = std_dev.to(device, dtype).view(1, 1, N)

    # multiply std tensor by scale factor
    total_std = self.current_std_scaling * total_std

    # handle applying randomisation per feature or per batch
    if (self.set_noise_per_feature or self.set_noise_per_batch or
        self.set_noise_per_timestep):
      # print(f"Varying noise per batch={self.set_noise_per_batch}, "
      #       f"per timestep={self.set_noise_per_timestep}, "
      #       f"per feature={self.set_noise_per_feature}")
      # will we take randomised values over batch or features
      B_dim = B if self.set_noise_per_batch else 1
      T_dim = T if self.set_noise_per_timestep else 1
      N_dim = N if self.set_noise_per_feature else 1
      # uniformly sample random values over the specified dimensions [0.0, 1.0)
      rand_values = torch.rand((B_dim, T_dim, N_dim), device=device, dtype=dtype)
      # convert to range [min, max) with equation: (max - min) * x + min
      std_scalings = (((self.individualised_noise_std_range[1] 
                        - self.individualised_noise_std_range[0]) * rand_values)
                        + self.individualised_noise_std_range[0])
      # for correlated noise, share the std scaling across both
      if self.set_noise_per_feature and enable_correlation:
        std_scalings[:, :, corr_split_idx : N] = std_scalings[:, :, 0 : corr_split_idx]
      # broadcast to get the new, scaled, standard deviations
      total_std = total_std * std_scalings # shape (B, 1, N)

    # save the last set of standard deviations used, in case we want to check noise
    self.last_std_used = total_std.clone()

    total_noise = torch.zeros_like(observation)

    # --- 1.  Gaussian Noise only --- #
    if self.enable_gaussian_ou:
      alpha = torch.rand(B, 1, 1, device=device, dtype=dtype) * \
               (self.alpha_range[1] - self.alpha_range[0]) + \
               self.alpha_range[0]

      std_g = torch.sqrt(alpha) * total_std
      std_ou = torch.sqrt(1 - alpha) * total_std

      # a) Generate Gaussian noise
      if not enable_correlation:
        # Uncorrelated noise
        # USE rand_sampler
        gaussian_noise = rand_sampler((B, T, N)) * std_g
        total_noise += gaussian_noise
      else:
        raise RuntimeError(f"NoiseGenerator.add_white_noise() cannot allow "
                           f"enable_correlation = True")
    else:
      raise RuntimeError(f"NoiseGenerator.add_white_noise() cannot allow "
                          f"enable_gaussian_ou = False")

    return observation + total_noise

  def add_noise_np(self,
                   observation: np.ndarray, 
                   std_dev: np.ndarray, 
                   enable_correlation: bool = False,
                   corr_split_idx: int = None,
                   ) -> np.ndarray:
      """
      A generic function to apply configurable noise to a batch of observations.
      (NumPy replication)

      Args:
          observation (np.ndarray): The input tensor with shape (B, T, N).
          std_dev (np.ndarray): Standard deviations for the noise. Can be a scalar
                                or a tensor of shape (N,).
          enable_correlation (bool): option to enable correlation betweeen features.
          corr_split_idx (int): index that seperates the base features from
                                         the correlated features.

      Returns:
          np.ndarray: The **total noise** tensor.
      """
      B, T, N = observation.shape
      dtype = observation.dtype

      # Reshape std_dev for broadcasting
      if std_dev.size == 1:
        # Use np.broadcast_to for memory efficiency, though direct multiply works
        total_std = np.broadcast_to(std_dev.reshape(1, 1, 1), (1, 1, N)).copy()
      else:
        total_std = std_dev.astype(dtype).reshape(1, 1, N)

      # multiply std tensor by scale factor
      total_std = self.current_std_scaling * total_std

      # handle applying randomisation per feature or per batch
      if (self.set_noise_per_feature or self.set_noise_per_batch or
          self.set_noise_per_timestep):
        
        B_dim = B if self.set_noise_per_batch else 1
        T_dim = T if self.set_noise_per_timestep else 1
        N_dim = N if self.set_noise_per_feature else 1
        
        # Use the class's random number generator (self.rng)
        rand_values = self.rng.random((B_dim, T_dim, N_dim), dtype=dtype)
        
        std_scalings = (((self.individualised_noise_std_range[1] 
                          - self.individualised_noise_std_range[0]) * rand_values)
                          + self.individualised_noise_std_range[0])
        
        if self.set_noise_per_feature and enable_correlation:
          std_scalings[:, :, corr_split_idx : N] = std_scalings[:, :, 0 : corr_split_idx]
        
        total_std = total_std * std_scalings # Broadcasting applies

      # save the last set of standard deviations used
      self.last_std_used = total_std.copy()

      total_noise = np.zeros_like(observation)

      # --- 1. Correlated Student's t and Ornstein-Uhlenbeck (OU) Noise --- #
      if self.enable_gaussian_ou:
        alpha = self.rng.random((B, 1, 1), dtype=dtype) * \
                 (self.alpha_range[1] - self.alpha_range[0]) + \
                 self.alpha_range[0]

        std_g = np.sqrt(alpha) * total_std
        std_ou = np.sqrt(1 - alpha) * total_std

        # a) Generate Student's t noise (heavy-tailed replacement for Gaussian)
        if not enable_correlation:
          # Uncorrelated Student's t noise
          gaussian_noise = np_rand_student_t(
              self.student_t_df, (B, T, N), self.rng, dtype
          ) * std_g
          total_noise += gaussian_noise
        else:
          # Correlated Student's t noise (Position/Velocity case)
          split_idx = corr_split_idx
          assert N == split_idx * 2, "For correlation, feature dimension N must be 2 * corr_split_idx"
          
          std_g_pos, std_g_vel = std_g[..., :split_idx], std_g[..., split_idx:]

          # 1. GENERATE BASE QPOS NOISE
          base_g_noise_pos = np_rand_student_t(
              self.student_t_df, (B, T, split_idx), self.rng, dtype
          )
          
          # QPOS NOISE is always based on the independent base noise
          gaussian_noise_pos = base_g_noise_pos * std_g_pos

          # --- DIFFERENTIAL NOISE LOGIC ---
          diff_range = self.differential_noise_range
          rand_prop_value = self.rng.random((B, 1, 1), dtype=dtype)
          differential_noise_proportion = (((diff_range[1] - diff_range[0]) * rand_prop_value)
                                           + diff_range[0])
          
          # 2. CALCULATE DIFFERENTIAL QVEL NOISE (eta_diff)
          qpos_noise_shifted = np.concatenate([np.zeros((B, 1, split_idx), dtype=dtype), 
                                               gaussian_noise_pos[:, :-1, :]], axis=1)
          differential_qvel_noise = (gaussian_noise_pos - qpos_noise_shifted) / self.dt

          # --- CORRECTED SCALING LOGIC ---
          diff_noise_std = differential_qvel_noise.std(axis=1, keepdims=True)
          diff_noise_std[diff_noise_std == 0] = 1.0 
          normalized_diff_noise = differential_qvel_noise / diff_noise_std
          differential_qvel_noise = normalized_diff_noise * std_g_vel
          # --- END CORRECTED SCALING LOGIC ---
          
          # 3. CALCULATE STANDARD CORRELATED QVEL NOISE (eta_corr)
          corr_rand = self.rng.random((B, 1, split_idx), dtype=dtype)
          corr_rho = (((self.corr_rho[1] 
                      - self.corr_rho[0]) * corr_rand)
                      + self.corr_rho[0])
          
          # Correlated part of velocity noise (uses base_g_noise_pos)
          base_g_noise_vel_indep = np_rand_student_t(
              self.student_t_df, (B, T, split_idx), self.rng, dtype
          )
          base_g_noise_vel_corr = corr_rho * base_g_noise_pos + np.sqrt(1 - corr_rho**2) * base_g_noise_vel_indep
          standard_correlated_qvel_noise = base_g_noise_vel_corr * std_g_vel
          
          # 4. MIX THE TWO NOISE SOURCES
          # Use np.broadcast_to to explicitly match shapes
          p = np.broadcast_to(differential_noise_proportion, (B, T, split_idx))
          
          gaussian_noise_vel = (p * differential_qvel_noise + 
                                (1 - p) * standard_correlated_qvel_noise)
          
          total_noise += np.concatenate([gaussian_noise_pos, gaussian_noise_vel], axis=-1)

          # Replicate the squeeze operation
          corr_rho = corr_rho.squeeze(axis=1) # Shape (B, 1, split_idx) -> (B, split_idx)
          
          self.last_diff_proportion_used = differential_noise_proportion.copy()

        # b) Generate Ornstein-Uhlenbeck noise (now based on Student's t)
        theta, dt = self.ou_theta, self.ou_dt
        ou_noise = np.zeros_like(observation)
        
        exp_factor = np.exp(np.array(-theta * dt, dtype=dtype))
        sqrt_factor = np.sqrt(1 - np.exp(np.array(-2 * theta * dt, dtype=dtype)))
        
        # --- OPTIMIZATION START: Pre-calculate noise distribution tensors ---
        # We need noise for T-1 steps. We generate it all at once to avoid 
        # creating the distribution object inside the loop.
        
        if not enable_correlation:
          # Pre-generate noise for all timesteps at once
          # Shape: (B, T-1, N)
          precalced_ou_noise = np_rand_student_t(
              self.student_t_df, (B, T - 1, N), self.rng, dtype
          )
        else:
          split_idx = corr_split_idx
          # Pre-generate noise for all timesteps at once for both components
          # Shape: (B, T-1, split_idx)
          precalced_base_pos = np_rand_student_t(
              self.student_t_df, (B, T - 1, split_idx), self.rng, dtype
          )
          precalced_base_vel_indep = np_rand_student_t(
              self.student_t_df, (B, T - 1, split_idx), self.rng, dtype
          )
        # --- OPTIMIZATION END ---

        for t in range(T - 1):
          if not enable_correlation:
            # Uncorrelated OU noise
            noise_factor = std_ou * sqrt_factor
            
            # Use pre-calculated noise for this timestep
            # Note: original code used noise_factor[:, 0, :], effectively locking std 
            # to the first timestep for the OU process. We preserve this behavior.
            random_comp = precalced_ou_noise[:, t, :] * noise_factor[:, 0, :]
            
            ou_noise[:, t + 1, :] = ou_noise[:, t, :] * exp_factor + random_comp
          else:
            # Correlated OU noise
            std_ou_pos, std_ou_vel = std_ou[..., :split_idx], std_ou[..., split_idx:]
            noise_factor_pos = std_ou_pos * sqrt_factor
            noise_factor_vel = std_ou_vel * sqrt_factor

            # Grab pre-calculated noise for this timestep
            base_ou_noise_pos_t = precalced_base_pos[:, t, :]
            base_ou_noise_vel_indep_t = precalced_base_vel_indep[:, t, :]
            
            base_ou_noise_vel_corr_t = corr_rho * base_ou_noise_pos_t + np.sqrt(1 - corr_rho**2) * base_ou_noise_vel_indep_t
            
            ou_noise_pos_t, ou_noise_vel_t = ou_noise[:, t, :split_idx], ou_noise[:, t, split_idx:]
            
            # Preserving original behavior: using index 0 for noise factors
            next_ou_noise_pos = ou_noise_pos_t * exp_factor + base_ou_noise_pos_t * noise_factor_pos[:, 0, :]
            standard_next_ou_noise_vel = ou_noise_vel_t * exp_factor + base_ou_noise_vel_corr_t * noise_factor_vel[:, 0, :]
            
            # --- OU DIFFERENTIAL NOISE ---
            dt_ou = dt
            differential_next_ou_noise_vel = (next_ou_noise_pos - ou_noise_pos_t) / dt_ou

            # --- CORRECTED SCALING LOGIC ---
            diff_noise_std_ou = differential_next_ou_noise_vel.std(axis=0, keepdims=True)
            diff_noise_std_ou[diff_noise_std_ou == 0] = 1.0
            normalized_diff_noise_ou = differential_next_ou_noise_vel / diff_noise_std_ou
            # Preserving original behavior: using index 0 for std_ou_vel
            differential_next_ou_noise_vel = normalized_diff_noise_ou * std_ou_vel[:, 0, :]
            # --- END CORRECTED SCALING LOGIC ---

            # MIX THE TWO OU NOISE SOURCES
            p_ou = self.last_diff_proportion_used.squeeze(axis=-1)
            
            next_ou_noise_vel = (p_ou * differential_next_ou_noise_vel + 
                                 (1 - p_ou) * standard_next_ou_noise_vel)
            
            ou_noise[:, t + 1, :] = np.concatenate([next_ou_noise_pos, next_ou_noise_vel], axis=-1)
      
        total_noise += ou_noise

      # --- 2. Persistent 'Zero-Error' Offsets --- #
      if self.enable_offset:
          
        offset_mask = (self.rng.random((B, 1, N), dtype=dtype) < self.p_offset).astype(dtype)
        # Base this on Student's t as well for heavy-tailed offsets
        offset = np_rand_student_t(
            self.student_t_df, (B, 1, N), self.rng, dtype
        ) * total_std * self.offset_scale
        total_noise += offset * offset_mask

      # --- 3. Random Spikes --- #
      if self.enable_spikes:

        spike_mask = (self.rng.random((B, 1, N), dtype=dtype) < self.p_spike).astype(dtype)
        
        # --- CHANGED: Use Laplace distribution for heavy-tailed spikes ---
        spike_rand_scale = self.rng.laplace(
            loc=self.spike_magnitude_mean, 
            scale=self.spike_magnitude_std, # numpy laplace uses scale directly
            size=(B, 1, N)
        ).astype(dtype)
        # --- END CHANGED ---

        spike_magnitudes = spike_rand_scale * total_std
        
        max_duration = self.spike_max_duration
        spike_durations = self.rng.integers(1, max_duration + 1, size=(B, 1, N))
        
        start_indices = self.rng.integers(0, T, size=(B, 1, N))
        end_indices = np.minimum(start_indices + spike_durations, T)
        
        time_indices = np.arange(T, dtype=np.int64).reshape(1, T, 1)
        spike_time_mask = (time_indices >= start_indices) & (time_indices < end_indices)
        
        spike_tensor = spike_magnitudes * spike_time_mask
        total_noise += spike_tensor * spike_mask

      # --- 4. 'Stuck' readings --- #
      if self.enable_fixed_readings:

        # determine which features in which batch will get a fixed event during their timesteps
        fixed_mask = (self.rng.random((B, 1, N), dtype=dtype) < self.p_fixed)
        fixed_durations = self.rng.integers(2, self.fixed_reading_max_duration + 1, size=(B, 1, N))
        
        start_indices = self.rng.integers(0, T, size=(B, 1, N))
        end_indices = np.minimum(start_indices + fixed_durations, T)

        time_indices = np.arange(T, dtype=np.int64).reshape(1, T, 1)
        fixed_time_mask = (time_indices >= start_indices) & (time_indices < end_indices)

        # Use np.take_along_axis to replicate torch.gather
        hold_values = np.take_along_axis(observation, start_indices, axis=1) # shape (B, 1, N)
        # Combine the "does this feature drop?" mask with the "is this timestep dropping?" mask
        final_stuck_mask = (fixed_time_mask & fixed_mask)

        if self.fixed_reading_identical:
          hold_noise_values = np.take_along_axis(total_noise, start_indices, axis=1)
          target_stuck_value = hold_values + hold_noise_values
          
          # N_replace = (hold_obs + hold_noise) - Obs_current
          # (B, 1, N) - (B, T, N) -> (B, T, N)
          replacement_noise = target_stuck_value - observation
            
        else:
          # Target is the held obs + the *current* noise
          # N_replace = (hold_obs + N_current) - Obs_current
          # (B, 1, N) + (B, T, N) - (B, T, N) -> (B, T, N)
          replacement_noise = (hold_values + total_noise) - observation
            
        # Now, apply the calculated replacement noise to the total_noise tensor
        total_noise = np.where(final_stuck_mask, 
                                replacement_noise, 
                                total_noise)

      return total_noise

  def add_white_noise_np(self,
                         observation: np.ndarray, 
                         std_dev: np.ndarray, 
                         enable_correlation: bool = False,
                         corr_split_idx: int = None,
                         ) -> np.ndarray:
      """
      A generic function to apply configurable noise to a batch of observations.
      (NumPy replication of add_white_noise)

      Args:
          observation (np.ndarray): The input tensor with shape (B, T, N).
          std_dev (np.ndarray): Standard deviations for the noise. Can be a scalar
                                or a tensor of shape (N,).
          enable_correlation (bool): option to enable correlation betweeen features.
          corr_split_idx (int): index that seperates the base features from
                                         the correlated features.

      Returns:
          np.ndarray: The **total noise** tensor.
      """
      B, T, N = observation.shape
      dtype = observation.dtype

      # Use standard Gaussian (Normal) distribution for simple white noise
      def rand_sampler(shape):
        # rng.standard_normal naturally gives unit variance
        return self.rng.standard_normal(shape, dtype=dtype)

      # Reshape std_dev for broadcasting
      if std_dev.size == 1:
        # Use np.broadcast_to for memory efficiency, though direct multiply works
        total_std = np.broadcast_to(std_dev.reshape(1, 1, 1), (1, 1, N)).copy()
      else:
        total_std = std_dev.astype(dtype).reshape(1, 1, N)

      # multiply std tensor by scale factor
      total_std = self.current_std_scaling * total_std

      # handle applying randomisation per feature or per batch
      if (self.set_noise_per_feature or self.set_noise_per_batch or
          self.set_noise_per_timestep):
        
        B_dim = B if self.set_noise_per_batch else 1
        T_dim = T if self.set_noise_per_timestep else 1
        N_dim = N if self.set_noise_per_feature else 1
        
        # Use the class's random number generator (self.rng)
        rand_values = self.rng.random((B_dim, T_dim, N_dim), dtype=dtype)
        
        std_scalings = (((self.individualised_noise_std_range[1] 
                          - self.individualised_noise_std_range[0]) * rand_values)
                          + self.individualised_noise_std_range[0])
        
        if self.set_noise_per_feature and enable_correlation:
          std_scalings[:, :, corr_split_idx : N] = std_scalings[:, :, 0 : corr_split_idx]
        
        total_std = total_std * std_scalings # Broadcasting applies

      # save the last set of standard deviations used
      self.last_std_used = total_std.copy()

      total_noise = np.zeros_like(observation)

      # --- 1.  Gaussian Noise only --- #
      if self.enable_gaussian_ou:
        alpha = self.rng.random((B, 1, 1), dtype=dtype) * \
                 (self.alpha_range[1] - self.alpha_range[0]) + \
                 self.alpha_range[0]

        std_g = np.sqrt(alpha) * total_std
        std_ou = np.sqrt(1 - alpha) * total_std

        # a) Generate Gaussian noise
        if not enable_correlation:
          # Uncorrelated noise
          # USE rand_sampler
          gaussian_noise = rand_sampler((B, T, N)) * std_g
          total_noise += gaussian_noise
        else:
          raise RuntimeError(f"NoiseGenerator.add_white_noise_np() cannot allow "
                             f"enable_correlation = True")
      else:
        raise RuntimeError(f"NoiseGenerator.add_white_noise_np() cannot allow "
                            f"enable_gaussian_ou = False")

      return total_noise


  def add_correlated_noise(self, # call this for qpos, qvel
                           observation: torch.Tensor, 
                           std_base: torch.Tensor, 
                           std_correlated: torch.Tensor,
                           use_np: bool = False,
                           ) -> torch.Tensor:
    """
    A specific wrapper to apply correlated noise.

    Args:
        observation (torch.Tensor): The input tensor with shape (B, T, N).
        std_base (torch.Tensor): Standard deviations of base feature, shape (N/2).
        std_correlated (torch.Tensor): Standard deviations for correlated features, shape (N/2,).

    Returns:
        torch.Tensor: The observation tensor with added noise.
    """
    
    if use_np:
      
      B, T, N = observation.shape
      N = int(N / 2)

      if std_base.size == 1:
        std_correlated = np.broadcast_to(std_base, (N,))
      if std_base.size == 1:
        std_correlated = np.broadcast_to(std_correlated, (N,))

      combined_std = np.concatenate([std_base, std_correlated], axis=0)
      return self.add_noise_np(observation, 
                               combined_std, 
                               enable_correlation=True, 
                               corr_split_idx=std_base.shape[0])

    else:

      B, T, N = observation.shape
      N = int(N / 2)

      # Reshape std_dev for broadcasting
      if std_base.numel() == 1:
        std_base = std_base.expand(N)

      # Reshape std_dev for broadcasting
      if std_correlated.numel() == 1:
        std_correlated = std_correlated.expand(N)

      # Combine std devs into a single vector
      combined_std = torch.cat([std_base, std_correlated], dim=0)
      return self.add_noise(observation, combined_std, enable_correlation=True,
                            corr_split_idx=std_base.shape[0])

# --- model components --- #

class SimplePrivilegedEncoder(nn.Module):

  name = "SimplePrivilegedEncoder"

  def __init__(self,
               input_dim,
               latent_dim,
               hidden_dim=64,
               hidden_layers=2,
               activation_fn="elu",
               use_layernorm=False,
               dropout_prob=0.0,
               device="cuda",
               **kwargs):
    """
    ROA inspired privileged information encoder. In practical terms, is simply
    a normal encoder-decoder.
    """

    if kwargs:
      pylogger.info(f"{self.name}.__init__ got unexpected arguments, which will be ignored: " +
                    str([key for key in kwargs.keys()]))

    super().__init__()

    self.device = device

    # encoder, puts privileged info into latent space
    self.encoder = make_mlp(
      input_dim=input_dim,
      output_dim=latent_dim,
      hidden_dim=hidden_dim,
      num_hidden=hidden_layers,
      activation_fn=activation_fn,
      use_layernorm=use_layernorm,
      use_dropout=True if dropout_prob > 1e-5 else False,
      dropout_prob=dropout_prob,
      activate_output=False,
    )

    # decoder, with reverse architecture
    self.decoder = make_mlp(
      input_dim=latent_dim,
      output_dim=input_dim,
      hidden_dim=hidden_dim,
      num_hidden=hidden_layers,
      activation_fn=activation_fn,
      use_layernorm=use_layernorm,
      use_dropout=True if dropout_prob > 1e-5 else False,
      dropout_prob=dropout_prob,
      activate_output=False,
    )

    self.to(self.device)

  def forward(self, x, decode=False):
    """
    Outputs the latents by default, but can set decode=True.
    """
    x = self.encoder(x)
    if decode:
      x = self.decoder(x)
    return x

  def get_save_state(self):
    """
    Return the save state of the module
    """
    return {
      "state_dict" : self.state_dict(),
    }
  
  def load_save_state(self, loaded_dict):
    """
    Load the save state of the module
    """
    self.load_state_dict(loaded_dict["state_dict"])

  def to(self, device):
    """
    Send the module to a specified device
    """
    self.device = device  # save the new device we are on
    super().to(device)    # call the underlying class method
    return self

class BasePredictor(nn.Module):

  def __init__(self,
               predict_qpos: bool = False,
               predict_qvel: bool = False,
               predict_acceleration: bool = True,
               predict_delta: bool = False,
               use_normalisation: bool = False,
               output_uncertainty: bool = False,
               uncertainty_per_feature: bool = False,
               multivariate_uncertainty: bool = False,
               predict_priv_info_num: int = 0,
               num_joint_angles: int = 7,
               physics_timestep: float = 2e-3,
               **kwargs):
    
    if kwargs:
      pylogger.info("BasePredictor.__init__ got unexpected arguments, which will be ignored: " +
                    str([key for key in kwargs.keys()]))

    super().__init__()

    self.predict_qpos = predict_qpos
    self.predict_qvel = predict_qvel
    self.predict_acceleration = predict_acceleration
    self.predict_delta = predict_delta
    self.use_normalisation = use_normalisation
    self.output_uncertainty = output_uncertainty
    self.uncertainty_per_feature = uncertainty_per_feature
    self.multivariate_uncertainty = multivariate_uncertainty
    self.n_q = num_joint_angles
    self.dt = physics_timestep
    self.predict_priv_info_num = predict_priv_info_num

    # indexes in incoming observation
    self.i_qpos = torch.arange(self.n_q)
    self.i_qvel = torch.arange(self.n_q, self.n_q * 2)

    # determine how many values we are predicting
    output_dim = 0
    if self.predict_qpos:
      self.pred_qpos = torch.arange(output_dim, output_dim + self.n_q)
      output_dim += self.n_q
    if self.predict_qvel: 
      self.pred_qvel = torch.arange(output_dim, output_dim + self.n_q)
      output_dim += self.n_q
    if self.predict_acceleration: 
      self.pred_qacc = torch.arange(output_dim, output_dim + self.n_q)
      output_dim += self.n_q

    if output_uncertainty:
      if multivariate_uncertainty:
        uncertainty_dim = (output_dim * (output_dim + 1)) // 2 # lower triangular matrix
      elif uncertainty_per_feature:
        uncertainty_dim = output_dim
      else:
        uncertainty_dim = 1
      self.pred_uncertainty = torch.arange(output_dim, output_dim + uncertainty_dim)
      output_dim += uncertainty_dim
      self.uncertainty_dim = uncertainty_dim
    else:
      self.uncertainty_dim = 0
      self.pred_uncertainty = torch.tensor([])

    # experimental: allow the predictor network to output the priv info (+ uncertainty)
    if self.predict_priv_info_num > 0:
      self.pred_priv_info = torch.arange(output_dim, output_dim + predict_priv_info_num)
      output_dim += predict_priv_info_num
      if output_uncertainty:
        if uncertainty_per_feature:
          priv_uncertainty_dim = predict_priv_info_num
        else:
          priv_uncertainty_dim = 1
        self.pred_priv_info_uncertainty = torch.arange(output_dim, output_dim + priv_uncertainty_dim)
        output_dim += priv_uncertainty_dim
        self.priv_uncertainty_dim = priv_uncertainty_dim

    if output_dim < self.n_q + output_uncertainty + predict_priv_info_num:
      raise RuntimeError(f"BasePredictor.__init__() error: "
                         f" predict qpos/qvel/qacc all set to False")
    
    # save the output dim
    self.output_dim = output_dim

    # insert the default values for dt if not using normalisation
    if not self.use_normalisation:
      self.norm = {
        "dt_qvel" : self.dt,
        "dt_qpos" : self.dt,
        "dt_qpos_squared" : 0.5 * (self.dt ** 2)
      }
    
  def net_forward(self, x):
    """
    Override this method to call the model
    """
    raise NotImplementedError()
    
  def forward(self, x, return_extras=False):
    """
    Pass the observation x through the model, and predict future states.
    Supports both single-step and multi-step predictions.
    """

    # prev values, in normalised units, from the observation
    qpos_prev = x[:, -1, self.i_qpos]  # (B, N)
    qvel_prev = x[:, -1, self.i_qvel]  # (B, N)

    # get the predictions of the model
    predictions = self.net_forward(x)
    
    # check if we have multi-step predictions
    is_multistep = predictions.dim() == 3  # (B, T, N)
    
    if is_multistep:
      B, timesteps, N = predictions.shape
    else:
      B, N = predictions.shape
      timesteps = 1
      predictions = predictions.unsqueeze(1)  # -> (B, 1, N)

    # initialize output tensors
    qpos = torch.zeros((B, timesteps, self.n_q), device=predictions.device)
    qvel = torch.zeros((B, timesteps, self.n_q), device=predictions.device)
    if self.predict_acceleration:
      qacc = torch.zeros((B, timesteps, self.n_q), device=predictions.device)

    # loop over each timestep in the prediction
    for t in range(timesteps):
      
      if self.predict_qpos:
        qpos[:, t] = predictions[:, t, self.pred_qpos]  # prediction, in normalised units
        if self.predict_delta:
          qpos[:, t] = qpos[:, t] + qpos_prev  # add in normalised unit space

      if self.predict_qvel:
        qvel[:, t] = predictions[:, t, self.pred_qvel]  # prediction, in normalised units
        if self.predict_delta:
          qvel[:, t] = qvel[:, t] + qvel_prev  # add in normalised units

      if self.predict_acceleration:
        qacc[:, t] = predictions[:, t, self.pred_qacc]  # prediction, in normalised units

        # now determine qpos and qvel (using timestep scaled into normalised space)
        qvel[:, t] = qvel_prev + self.norm["dt_qvel"] * qacc[:, t]
        qpos[:, t] = (qpos_prev + self.norm["dt_qpos"] * qvel_prev + 
                      self.norm["dt_qpos_squared"] * qacc[:, t])
          
      # if predicting only one of qpos/qvel, integrate/differentiate for the other
      elif not self.predict_qpos:
        qpos[:, t] = qpos_prev + self.norm["dt_qpos"] * qvel[:, t]
      elif not self.predict_qvel:
        qvel[:, t] = (qpos[:, t] - qpos_prev) / self.norm["dt_qpos"]

      # update prev values for next timestep
      qpos_prev = qpos[:, t]
      qvel_prev = qvel[:, t]

    # if original input was single-step, squeeze time dimension back out
    if not is_multistep:
      qpos = qpos.squeeze(1)  # (B, N)
      qvel = qvel.squeeze(1)  # (B, N)
      if self.predict_acceleration:
        qacc = qacc.squeeze(1)

    # combine qpos and qvel to create output
    out = torch.concat([qpos, qvel], dim=-1)

    if return_extras:
      extras = {}
      if self.predict_acceleration:
        extras["pred_qacc"] = qacc
      if self.output_uncertainty:
        extras["predictor_uncertainty"] = predictions[:, :, self.pred_uncertainty]
      if self.predict_priv_info_num > 0:
        est_latents = predictions[:, :, self.pred_priv_info]
        est_latents = torch.mean(est_latents, dim=1) # average over time
        extras["predictor_est_latents"] = est_latents
        if self.output_uncertainty:
          latent_unc = predictions[:, :, self.pred_priv_info_uncertainty]
          latent_unc = torch.mean(latent_unc, dim=1) # average over time
          extras["predictor_latent_uncertainty"] = latent_unc
      return out, extras
    else:
      return out
  
  def load_normalisation(self, norm_object):
    """
    Load normalisations given a normalisation object, from utils.training
    """

    pylogger.info(f"Loading normalisation dictionary into {self.name}")

    self.normalisations = norm_object

    if self.normalisations is None and self.use_normalisation:
      raise RuntimeError(f"{self.name}.load_normalisation() error: "
                         f"normalisation object is None, but use_normalisation=True")
    
    # create dictionary to save normalisations
    self.norm = norm_object.create_dict([
      ([f"qpos[{i}]" for i in range(self.n_q)], "qpos"),
      ([f"qvel[{i}]" for i in range(self.n_q)], "qvel"),
      ([f"qacc[{i}]" for i in range(self.n_q)], "qacc"),
      ([f"qM[{i}]" for i in range(self.n_q ** 2)], "M"),
      ([f"qfrc_in[{i}]" for i in range(self.n_q)], "C"),
      ([f"qfrc_out[{i}]" for i in range(self.n_q)], "J"),
    ])
    
    # determine the timestep vectors, given the normalisations, and add to norm dict
    self.norm["dt_qvel"] = self.dt * (self.norm["qacc"]["std"] / self.norm["qvel"]["std"])
    self.norm["dt_qpos"] = self.dt * (self.norm["qvel"]["std"] / self.norm["qpos"]["std"])
    self.norm["dt_qpos_squared"] = (0.5 * (self.dt ** 2) * 
                                    (self.norm["qacc"]["std"] / self.norm["qpos"]["std"]))

  def to(self, device):
    """
    Move the model to the specified device
    """
    self.device = device
    super().to(device)
    return self

class AttentionPredictor2(BasePredictor):

  name = "AttentionPredictor2"

  def __init__(self,
               # args for this class
               input_dim: int,
               hidden_dim: int = 64,
               prediction_timesteps: int = 1,
               num_layers: int = 2,
               num_heads: int = 4,
               dropout: float = 0.1,
               pos_encoding: bool = True,
               learnable_pos_encoding: bool = False,
               pooling_method: str = 'last',
               device: str = "cuda",
               # args for base predictor
               predict_qpos: bool = True,
               predict_qvel: bool = True,
               predict_acceleration: bool = False,
               predict_delta: bool = False,
               use_normalisation: bool = False,
               output_uncertainty: bool = False,
               uncertainty_per_feature: bool = False,
               multivariate_uncertainty: bool = False,
               num_joint_angles: bool = 7,
               physics_timestep: bool = 5e-3,
               **kwargs):
    
    if kwargs:
      print("AttentionPredictor2.__init__ got unexpected arguments, which will be ignored: " +
            str([key for key in kwargs.keys()]))

    # initialise the base predictor
    super().__init__(
      predict_qpos=predict_qpos,
      predict_qvel=predict_qvel,
      predict_acceleration=predict_acceleration,
      predict_delta=predict_delta,
      use_normalisation=use_normalisation,
      num_joint_angles=num_joint_angles,
      physics_timestep=physics_timestep,
      output_uncertainty=output_uncertainty,
      uncertainty_per_feature=uncertainty_per_feature,
      multivariate_uncertainty=multivariate_uncertainty,
    )

    self.prediction_timesteps = prediction_timesteps

    if self.prediction_timesteps > 1:
      pooling_method = "none" # do NOT pool, return features for all input timesteps

    self.net = nets.AttentionHistoryEncoder2(
        input_dim=input_dim,
        output_dim=self.output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        pos_encoding=pos_encoding,
        learnable_pos=learnable_pos_encoding,
        pooling_method=pooling_method,
        device=device,
      )
    
    self.to(device)
    
    pylogger.info(f"AttentionPredictor2 initialised")

  def net_forward(self, x):
    """
    Inference from the network
    """

    y = self.net(x)

    if self.prediction_timesteps > 1:
      if y.shape[1] < self.prediction_timesteps:
        raise RuntimeError(f"AttentionPredictor2.net_forward() error: "
                           f"network output has {y.shape[1]} timesteps, but "
                           f"prediction timesteps = {self.prediction_timesteps}")
      y = y[:, -self.prediction_timesteps:]

    return y

class LargeCNNEstimator(nn.Module):

  name = "LargeCNNEstimator"

  def __init__(self,
               input_dim,
               timestep_dim, 
               output_dim, 
               num_layers=None,
               base_channels=32,
               channel_multiplier=1.5,
               activation_fn="elu",
               dropout_prob=0.2,
               layer_configs=None,
               device="cuda"):
    
    super().__init__()

    self.net = LargeCNN1D(
      feature_dim=input_dim,
      timestep_dim=timestep_dim,
      output_dim=output_dim,
      num_layers=num_layers,
      base_channels=base_channels,
      channel_multiplier=channel_multiplier,
      activation_fn=activation_fn,
      dropout_prob=dropout_prob,
      layer_configs=layer_configs,
    )

    self.to(device)

  def forward(self, x):
    """
    Pass input x through the model.
    """

    return self.net(x)
  
  def to(self, device):
    """
    Move the model to the specified device
    """
    self.device = device
    super().to(device)
    return self

class AttentionDenoiser(nn.Module):

  name = "AttentionDenoiser"

  def __init__(self,
               input_dim: int,
               hidden_dim: int = 64,
               num_layers: int = 2,
               num_heads: int = 4,
               dropout: float = 0.1,
               pos_encoding: bool = True,
               learnable_pos_encoding: bool = False,
               device: str = "cuda",
               **kwargs):
    
    if kwargs:
      print(f"{self.name}.__init__ got unexpected arguments, which will be ignored: " +
            str([key for key in kwargs.keys()]))

    super().__init__()

    self.net = nets.AttentionHistoryEncoder2(
        input_dim=input_dim,
        output_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        pos_encoding=pos_encoding,
        learnable_pos=learnable_pos_encoding,
        pooling_method="none", # do not pool as we want identical output shape
        device=device,
      )
    
    self.to(device)
    
    pylogger.info(f"{self.name} initialised")

  def forward(self, x):
    """
    Inference from the network
    """

    y = self.net(x)

    if y.shape != x.shape:
      raise RuntimeError(f"{self.name}.forward() error: "
                         f"input.shape={x.shape} != output.shape={y.shape}")
    
    return y
  
# --- diffusion test --- #

# --- main models --- #

class LatentPredictor(nn.Module):

  def __init__(self, 
               
               # key parameters
               num_hist_timesteps_to_use=10,
               num_timesteps_predictor=None,
               num_timesteps_estimator=None,
               num_privileged_obs=6,
               num_latent_encoding=4,
               num_joint_angles=7,
               num_actions=3,
               device="cuda",

               # enable/disable features
               add_next_action=False,
               add_privileged_info=False,
               add_denoise_step=False,
               use_full_observation=False,
               from_sim_priv_info_indexes=None,
               zero_qvel_input=False,
               use_mini_batch_sampling=False,
               mini_batch_size=256,

               # experimental for baselines
               disable_latents_into_predictor=False,
               predictor_outputs_priv_info=False,

               # noise added to data
               use_noise=False,
               noise_std_SI_qpos=None,
               noise_std_SI_qvel=None,
               noise_std_SI_action=None,
               noise_std_SI_priv_info=None,
               default_noise_std=0.02,
               noise_params=None,

               # characterising uncertainty
               use_estimator_confidence=False,
               use_predictor_confidence=False,
               predictor_gets_confidence=False,
               use_edl_confidence=False,
               use_sampled_uncertainty=False,
               confidence_per_feature=False,
               multivariate_uncertainty=False,
               confidence_log_var_clamping=None,
               train_with_estimated_latents=False,
               train_with_estimated_latents_chance=0.1,
               mix_estimated_latents_elementwise=True,
               detach_estimated_latents_before_mix=True,
               add_calibrated_latent_noise=False,

               # which values is the model predicting
               predict_qpos=False,
               predict_qvel=False,
               predict_acceleration=True,
               predict_delta=False,

               # training hyperparameters
               use_normalisation=False,
               prediction_horizon=10,
               num_use_each_datapoint=1,
               taper_horizon_loss_to=1.0,
               max_grad_norm=1.0,
               use_gt_chance=0.5,
               estimator_update_rate=1,
               target_latent_norm=0.5,
               weight_decay=0.0,
               alphas=None,
               contrastive_style="linear",
               contrastive_temp=0.1,

               # learning hyperparameters
               learning_rate_encoder=1e-4,
               learning_rate_predictor=1e-4,
               learning_rate_estimator=1e-4,
               learning_rate_denoiser=1e-4,
               loss_fn="MSE",

               # hydra instantiated key components
               encoder=None,   # should be hydra configs
               estimator=None, # should be hydra configs
               predictor=None, # should be hydra configs
               denoiser=None,  # should be hydra configs

               # compatibility with legacy
               new_indexes=False,
               seperate_estimator_obs=False,

               **kwargs,
               ):
    """
    Estimator for future motions
    """

    if kwargs:
      pylogger.info("LatentPredictor.__init__ got unexpected arguments, which will be ignored: " + \
                    str([key for key in kwargs.keys()]))

    super().__init__()

    # key environment paraters
    self.num_joint_angles = num_joint_angles
    self.num_hist_timesteps_to_use = num_hist_timesteps_to_use
    self.num_timesteps_predictor = num_timesteps_predictor
    self.num_timesteps_estimator = num_timesteps_estimator
    self.num_actions = num_actions
    self.num_privileged_obs = num_privileged_obs
    self.num_latent_encoding = num_latent_encoding
    self.use_normalisation = use_normalisation
    self.device = device

    if self.num_hist_timesteps_to_use is None:
      if (self.num_timesteps_estimator is None or
          self.num_timesteps_predictor is None):
        raise RuntimeError(f"num_hist_timesteps_to_use=None, hence predictor and estimator "
                           f"timesteps must both be set. "
                           f"num_timesteps_estimator={self.num_timesteps_estimator}, "
                           f"num_timesteps_predictor={self.num_timesteps_predictor}")
      self.num_hist_timesteps_to_use = np.max([
        self.num_timesteps_estimator,
        self.num_timesteps_predictor
      ])
    else:
      if self.num_timesteps_predictor is None:
        self.num_timesteps_predictor = self.num_hist_timesteps_to_use
      if self.num_timesteps_estimator is None:
        self.num_timesteps_estimator = self.num_hist_timesteps_to_use

    # for compatibility - are these needed?
    self.n_q = self.num_joint_angles
    self.n_t = self.num_hist_timesteps_to_use

    # which values is the model predicting
    self.predict_qpos = predict_qpos
    self.predict_qvel = predict_qvel
    self.predict_acceleration = predict_acceleration
    self.predict_delta = predict_delta

    # optional components
    self.add_next_action = add_next_action
    self.add_privileged_info = add_privileged_info
    self.add_denoise_step = add_denoise_step
    self.use_full_observation = use_full_observation
    self.from_sim_priv_info_indexes = from_sim_priv_info_indexes
    self.zero_qvel_input = zero_qvel_input
    self.use_mini_batch_sampling = use_mini_batch_sampling
    self.mini_batch_size = mini_batch_size

    # experimental for baselines
    self.disable_latents_into_predictor = disable_latents_into_predictor
    self.predictor_outputs_priv_info = predictor_outputs_priv_info

    # define key parameters for adding noise to data
    self.use_noise = use_noise
    self.default_noise_std = default_noise_std
    self.noise_std_SI_qpos = noise_std_SI_qpos
    self.noise_std_SI_qvel = noise_std_SI_qvel
    self.noise_std_SI_action = noise_std_SI_action
    self.noise_std_SI_priv_info = noise_std_SI_priv_info

    # characterising uncertainty
    self.use_estimator_confidence = use_estimator_confidence
    self.use_predictor_confidence = use_predictor_confidence
    self.confidence_per_feature = confidence_per_feature
    self.multivariate_uncertainty = multivariate_uncertainty
    self.use_edl_confidence = use_edl_confidence
    self.predictor_gets_confidence = predictor_gets_confidence
    self.use_sampled_uncertainty = use_sampled_uncertainty
    self.confidence_log_var_clamping = confidence_log_var_clamping
    self.train_with_estimated_latents = train_with_estimated_latents
    self.train_with_estimated_latents_chance = train_with_estimated_latents_chance
    self.mix_estimated_latents_elementwise = mix_estimated_latents_elementwise
    self.detach_estimated_latents_before_mix = detach_estimated_latents_before_mix
    self.add_calibrated_latent_noise = add_calibrated_latent_noise

    # extra parameters for uncertainty, hardcoded
    self.default_nu = 100.0
    self.default_alpha = 4.0
    
    # default noise parameters: real world noise icra-attempt
    if noise_params is None:
      noise_params = {
      "master_std_scaling" : None,
      "qpos_qvel" : {
        'std_scaling': 1.0,
        'enable_gaussian_ou': True,
        'enable_offset': True,
        'enable_spikes': True,
        'enable_correlation': True,
        'alpha_range': (0.1, 0.9),
        'corr_rho': 0.8,
        'ou_theta': 15.0,
        'ou_dt': 0.05,
        'p_offset': 0.25,
        'offset_scale': 1.0,
        'p_spike': 0.01,
        'spike_max_duration': 5,
        'spike_magnitude_mean': 10.0,
        'spike_magnitude_std': 2.5,
      },
      "action" : {
        'std_scaling' : 1.0,
        'enable_gaussian_ou': True,
        'enable_offset': True,
        'alpha_range': (0.1, 0.9),
        'ou_theta': 15.0,
        'ou_dt': 0.05,
        'p_offset': 0.25,
        'offset_scale': 4.0,
      },
      "priv_info" : {
        'std_scaling' : 1.0,
        'enable_gaussian_ou': True,
        'alpha_range': (0.1, 0.9),
        'ou_theta': 15.0,
        'ou_dt': 0.05,
      },
    }

    # create noise generators based on noise parameters
    if ("master_std_scaling" in noise_params and 
        noise_params["master_std_scaling"] is not None):
      master_scale = { "std_scaling" : noise_params["master_std_scaling"] }
    else: master_scale = {}
    if ("master_std_range" in noise_params and 
        noise_params["master_std_range"] is not None):
      master_range = { "individualised_noise_std_range" : noise_params["master_std_range"] }
    else: master_range = {}
    self.noiser_qpos_qvel = NoiseGenerator(**dict(noise_params["qpos_qvel"]) | master_scale | master_range)
    self.noiser_action = NoiseGenerator(**dict(noise_params["action"]) | master_scale | master_range)
    self.noiser_priv_info = NoiseGenerator(**dict(noise_params["priv_info"]) | master_scale | master_range)

    if self.from_sim_priv_info_indexes is not None:
      if self.num_privileged_obs != self.from_sim_priv_info_indexes:
        pylogger.info(f"num_privileged_obs={self.num_privileged_obs}, whilst "
                      f"from_sim_priv_info_indexes={self.from_sim_priv_info_indexes}. "
                      f"Therefore, setting num_privileged_obs="
                      f"{len(from_sim_priv_info_indexes)}")
        self.num_privileged_obs = len(from_sim_priv_info_indexes)

    # key hyperparameters for training
    self.learning_rate_encoder = learning_rate_encoder
    self.learning_rate_predictor = learning_rate_predictor
    self.learning_rate_estimator = learning_rate_estimator
    self.learning_rate_denoiser = learning_rate_denoiser
    self.max_grad_norm = max_grad_norm
    self.loss_fn = get_loss_fn(loss_fn)(reduction="none") # no reduction, we apply tapers
    self.num_use_each_datapoint = num_use_each_datapoint
    self.estimator_update_rate = estimator_update_rate
    self.target_latent_norm = target_latent_norm
    self.weight_decay = weight_decay
    self.taper_horizon_loss_to = taper_horizon_loss_to
    self.prediction_horizon = prediction_horizon
    self.use_gt_chance = use_gt_chance
    self.alphas = alphas
    self.contrastive_style = contrastive_style
    self.contrastive_temp = contrastive_temp

    if self.contrastive_style.lower() == "linear":
      self.contrastive_fn = LinearSimilarityContrastiveLoss(temperature=contrastive_temp)
    elif self.contrastive_style.lower() == "binary":
      self.contrastive_fn = PrivilegedContrastiveLoss(temperature=contrastive_temp)
    elif self.contrastive_style.lower() == "nce":
      self.contrastive_fn = InfoNCEPrivilegedLoss(temperature=contrastive_temp)

    # legacy compatibility
    self.new_indexes = new_indexes
    self.seperate_estimator_obs = seperate_estimator_obs

    # initialise key class parameters
    self.dense_mass_matrix = True # only applies if regressing components
    self.normalisations = None
    self.updates_done = 0
    self.norm = {}
    self.qpos_actuated = torch.tensor([0, 1, 4], dtype=int)
    self.qpos_payload = torch.tensor([2, 3, 5, 6], dtype=int)
    self.qvel_actuated = self.qpos_actuated + self.n_q
    self.qvel_payload = self.qpos_payload + self.n_q
    self.act_joint_inds = torch.concat([
      self.qpos_actuated,
      self.qvel_actuated,
    ])
    self.payload_joint_inds = torch.concat([
      self.qpos_payload,
      self.qvel_payload,
    ])
    
    # --- handle indexing the observations --- #

    self._setup_indexes(new_indexes=new_indexes)

    # --- create the model components --- #

    # edit key settings in the configs, and then save them
    if predictor is not None:
      # default case
      predictor["input_dim"] = self.num_obs_post_encoder

      # cascade settings in the case that we use uncertainty in the predictor
      predictor["output_uncertainty"] = self.use_predictor_confidence
      predictor["uncertainty_per_feature"] = self.confidence_per_feature
      predictor["multivariate_uncertainty"] = self.multivariate_uncertainty
      # experimental: predictor outputs the latents directly
      if self.predictor_outputs_priv_info:
        predictor["predict_priv_info_num"] = self.num_latent_encoding

      self.predictor_hydra_configs = (
        OmegaConf.to_container(predictor, resolve=True, throw_on_missing=False))
      
    if estimator is not None:
      estimator["input_dim"] = self.num_obs_eval
      estimator["output_dim"] = self.num_latent_encoding
      # add any extra outputs for uncertainty values (either per vector, or per element)
      if self.use_estimator_confidence:
        estimator["output_dim"] += self.conf_inds.shape[0]
      self.estimator_hydra_configs = (
        OmegaConf.to_container(estimator, resolve=True, throw_on_missing=False))
      
    if encoder is not None:
      self.encoder_hydra_configs = (
        OmegaConf.to_container(encoder, resolve=True, throw_on_missing=False))
      
    if denoiser is not None:
      denoiser_len = self.n_q * 2
      self.denoiser_inds = torch.arange(denoiser_len)
      denoiser["input_dim"] = denoiser_len
      self.denoiser_hydra_configs = (
        OmegaConf.to_container(denoiser, resolve=True, throw_on_missing=False))

    if self.add_privileged_info:
      self.encoder = hydra.utils.instantiate(encoder)
      self.estimator = hydra.utils.instantiate(estimator)
    else:
      self.encoder = nn.Identity()
      self.estimator = nn.Identity()

    if self.add_denoise_step:
      # only instantiate if we have specifically added the denoiser
      self.denoiser = hydra.utils.instantiate(denoiser)
    else:
      self.denoiser = nn.Identity()

    self.predictor = hydra.utils.instantiate(predictor)

    # --- optimiser --- #

    if denoiser is not None:
      # create optimiser with denoiser network
      self.optimiser = optim.AdamW([
          {'params': self.encoder.parameters(), 
          'lr': self.learning_rate_encoder,
          "weight_decay" : self.weight_decay, },
          {'params': self.predictor.parameters(), 
          'lr': self.learning_rate_predictor,
          "weight_decay" : self.weight_decay, },
          {'params': self.estimator.parameters(), 
          'lr': self.learning_rate_estimator,
          "weight_decay" : self.weight_decay, },
          {'params': self.denoiser.parameters(),
          'lr': self.learning_rate_denoiser,
          "weight_decay" : self.weight_decay, }
      ])
    else:
      # normal branch: create the optimsier
      self.optimiser = optim.AdamW([
          {'params': self.encoder.parameters(), 
          'lr': self.learning_rate_encoder,
          "weight_decay" : self.weight_decay, },
          {'params': self.predictor.parameters(), 
          'lr': self.learning_rate_predictor,
          "weight_decay" : self.weight_decay, },
          {'params': self.estimator.parameters(), 
          'lr': self.learning_rate_estimator,
          "weight_decay" : self.weight_decay, },
      ])

    # --- finish --- #

    key_info = (
      f"LatentPredictor: \n"
      f"  --- key ---\n"
      f"  -> encoder_name={self.encoder.name if self.add_privileged_info else '<disabled>'}\n"
      f"  -> estimator_name={self.estimator.name if self.add_privileged_info else '<disabled>'}\n"
      f"  -> predictor_name={self.predictor.name}\n"
      f"  -> denoiser_name={self.denoiser.name if self.add_denoise_step else '<disabled>'}\n"
      f"  -> predict_qpos={self.predict_qpos}\n"
      f"  -> predict_qvel={self.predict_qvel}\n"
      f"  -> predict_acceleration={self.predict_acceleration}\n"
      f"  -> predict_delta={self.predict_delta}\n"
      f"  -> use_normalisation={self.use_normalisation}\n"
      f"  -> total number of parameters={sum(p.numel() for p in self.parameters())}\n"
      f"  -> predictor parameters={sum(p.numel() for p in self.predictor.parameters())}\n"
      f"  -> estimator parameters={sum(p.numel() for p in self.estimator.parameters())}\n"
      f"  -> encoder parameters={sum(p.numel() for p in self.encoder.parameters())}\n"
      f"  -> denoiser parameters={sum(p.numel() for p in self.denoiser.parameters())}\n"
      f"  --- extra ---\n"
      f"  -> num_joint_angles={self.n_q}\n"
      f"  -> num_hist_timesteps_to_use={self.n_t}\n"
      f"  -> num_timesteps_predictor={self.num_timesteps_predictor}\n"
      f"  -> num_timesteps_estimator={self.num_timesteps_estimator}\n"
      f"  -> num_actions={self.num_actions}\n"
      f"  -> num_privileged_obs={self.num_privileged_obs}\n"
      f"  -> num_latent_encoding={self.num_latent_encoding}\n"
      f"  -> num_obs_full={self.num_obs_full}\n"
      f"  -> num_obs_train={self.num_obs_train}\n"
      f"  -> num_obs_eval={self.num_obs_eval}\n"
      f"  -> num_obs_post_encoder={self.num_obs_post_encoder}\n"
      f"  -> new_indexes={self.new_indexes}\n"
      f"  -> use_full_observation={self.use_full_observation}\n"
      f"  -> contrastive_style={self.contrastive_style}\n"
    )
    
    for line in key_info.split("\n"):
      pylogger.info(line)
    
  # --- setup functions --- #

  def _indexes(self, x, mode="eval", required=True):
    """
    Return the indexes for an additional 'x' components of the observation,
    and keep track of the overall size of the observation. In addition, mark
    whether these indexes should be included in the 'eval' observation, the
    'train' observation, or the 'full' observation (hidden to the model, but
    accessible for the loss). If required is False, the indexes are only
    included in the 'full' observation if 'self.use_obs_full=True'.
    """
    start = self.num_obs_full
    if required:
      if mode == "eval":
        self.num_obs_train += x
        self.num_obs_eval += x
        self.num_obs_full += x
      elif mode == "train":
        self.num_obs_train += x
        self.num_obs_full += x
      elif mode == "full":
        self.num_obs_full += x
      else:
        raise RuntimeError(f"mode not recognised")
    elif self.use_full_observation:
      self.num_obs_full += x
    return torch.arange(start, self.num_obs_full)
  
  def _setup_indexes(self, new_indexes=False):
    """
    Set up indexes for slicing observations, calculate sizes for observations at
    training and eval time etc
    """

    self.num_obs_train = 0
    self.num_obs_eval = 0
    self.num_obs_full = 0

    # determine indexes for the predictor and estimator (eg [-5, -4, ..., -1])
    self.t_predictor = torch.arange(-self.num_timesteps_predictor, 0)
    self.t_estimator = torch.arange(-self.num_timesteps_estimator, 0)

    # define carefully the observation
    self.i_qpos = self._indexes(self.n_q, mode="eval", required=True)
    self.i_qvel = self._indexes(self.n_q, mode="eval", required=True)
    self.i_qs = torch.arange(2 * self.n_q)

    # do we add actions to the observation
    self.i_action = self._indexes(self.num_actions, mode="eval", 
                                  required=self.add_next_action)

    # do we train with privileged information, and encode it
    x = self.num_obs_train # obs before any priv info
    self.i_priv_obs = self._indexes(self.num_privileged_obs, mode="train",
                                    required=self.add_privileged_info)
    self.i_priv_enc = torch.arange(x, x + self.num_latent_encoding)

    # define the indexes of the latents themselves, and any uncertainty values
    self.latent_encoding_inds = torch.arange(self.num_latent_encoding)
    self.conf_inds = torch.tensor([]) # empty tensor
    if self.use_estimator_confidence:
      if self.use_edl_confidence:
        self.conf_inds = torch.arange(self.num_latent_encoding, self.num_latent_encoding + 3)
      else:
        if self.confidence_per_feature:
          self.conf_inds = torch.arange(self.num_latent_encoding, 2 * self.num_latent_encoding)
        else:
          self.conf_inds = torch.tensor([self.num_latent_encoding], dtype=int)
    
    if self.add_privileged_info:
      self.num_obs_post_encoder = self.num_obs_eval + self.num_latent_encoding
      if self.use_estimator_confidence and self.predictor_gets_confidence:
        self.num_obs_post_encoder += self.conf_inds.shape[0] # add confidence value/s to latent encoding
    else:
      self.num_obs_post_encoder = self.num_obs_eval
    # special case: predictor doesn't get latents at all
    if self.disable_latents_into_predictor:
      self.num_obs_post_encoder = self.num_obs_eval

    if self.dense_mass_matrix:
      m = self.n_q ** 2 # full mass matrix
    else:
      m = int((self.n_q * (self.n_q + 1)) / 2) # lower triangular of mass matrix

    # extra information
    if new_indexes:
      self.i_qacc = self._indexes(self.n_q, mode="full", required=self.use_full_observation)
      self.i_qfrc_in = self._indexes(self.n_q, mode="full", required=self.use_full_observation)
      self.i_qfrc_out = self._indexes(self.n_q, mode="full", required=self.use_full_observation)
      self.i_qfrc_actuator = self._indexes(self.n_q, mode="full", required=self.use_full_observation)
      self.i_qfrc_applied = self._indexes(self.n_q, mode="full", required=self.use_full_observation)
      self.i_qfrc_passive = self._indexes(self.n_q, mode="full", required=self.use_full_observation)
      self.i_qfrc_constraint = self._indexes(self.n_q, mode="full", required=self.use_full_observation)
      self.i_qM = self._indexes(m, mode="full", required=self.use_full_observation)
    else:
      # depreciated, will be deleted soon
      self.i_qfrc_out = self._indexes(self.n_q, mode="full", required=self.use_full_observation)
      self.i_qfrc_in = self._indexes(self.n_q, mode="full", required=self.use_full_observation)
      self.i_qfrc_con = self._indexes(self.n_q, mode="full", required=self.use_full_observation)
      self.i_qM = self._indexes(m, mode="full", required=self.use_full_observation)
      self.i_qacc = self._indexes(self.n_q, mode="full", required=self.use_full_observation)

    # indexes for masking out different parts of the observation
    self.i_obs_train = torch.arange(self.num_obs_train)
    self.i_obs_eval = torch.arange(self.num_obs_eval)
    self.i_obs_post_encoder = torch.arange(self.num_obs_post_encoder)
    self.i_obs_full = torch.arange(self.num_obs_full)

  # --- main functions --- #

  def forward_prediction(self, 
                         observation, 
                         estimator_observation=None,
                         return_extras=False,
                         eval=False,
                         priv_info_override=None,
                         latent_override=None,
                         gt_target=None,
                         priv_info_no_noise=None,
                         ):
    """
    Run forward and return a prediction of the next state given an observation
    with shape: [B, T, N], where B=batch_size, T=num_hist_timesteps_to_use, and
    N=feature_dimension.

    Optionally provide a different observation for the estimator, otherwise it
    uses the main observation.

    Important! The incoming observation can include privileged information. This
    function should extract only information that the model should see, and use
    just that.
    """
        
    # check inputs
    if latent_override is not None and priv_info_override is not None:
      raise RuntimeError(f"LatentPredictor.forward_prediction() error: "
                         f"latent_override and priv_info_override cannot both be set")
    if not eval and latent_override:
      raise RuntimeError(f"LatentPredictor.forward_prediction() error: "
                         f"latent_override set during training (eval=False)")
    if not eval and priv_info_override:
      raise RuntimeError(f"LatentPredictor.forward_prediction() error: "
                         f"priv_info_override set during training (eval=False)")

    # placeholders for optionally calculated variables
    est_encoding = None
    priv_encoding = None
    priv_info = None
    priv_encoding_no_noise = None
    decoded_latents = None
    estimator_uncertainty = None

    # do we mask out all incoming velocities
    if self.zero_qvel_input:
      if not hasattr(self, "debug_print_zero_qvel"):
        pylogger.info(f"\n\n{'-'*10} QVEL ZERO-ED forward_prediction {'-'*10}\n")
        self.debug_print_zero_qvel = True
      observation[:, :, self.i_qvel] = 0.0 # mask out incoming velocities
      if estimator_observation is not None:
        estimator_observation[:, :, self.i_qvel] = 0.0

    # --- experimental: add denoising step --- #
    if self.add_denoise_step:
      if not hasattr(self, "debug_print_denoiser"):
        pylogger.info(f"\n\n{'-'*10} DENOISED in forward_prediction {'-'*10}\n")
        self.debug_print_denoiser = True
      # detach observation from graph to avoid auto-regressive gradients
      obs_eval_denoiser = observation[:, :, self.denoiser_inds].clone().detach()
      denoised_obs = self.denoiser(obs_eval_denoiser) # pass through denoiser
      # apply the denoising step via addition of a zero gradient tensor
      # this avoids any denoising gradients entering the main observation
      observation[:, :, self.denoiser_inds] = (observation[:, :, self.denoiser_inds] 
                                     + (denoised_obs - observation[:, :, self.denoiser_inds]).detach())
      if estimator_observation is not None:
        estimator_observation[:, :, self.denoiser_inds] = observation[:, :, self.denoiser_inds]
    # --- end experimental: denoising step --- #

    # information available to model at eval time (i.e., no privileged information)
    obs_eval_predictor = observation[:, :, self.i_obs_eval].clone()
    obs_eval_predictor = obs_eval_predictor[:, self.t_predictor, :]

    if estimator_observation is None:
      obs_eval_estimator = observation[:, :, self.i_obs_eval].clone()
    else:
      obs_eval_estimator = estimator_observation[:, :, self.i_obs_eval]

    # handle addition of privileged information at training time, but not eval time
    if self.add_privileged_info:

      # at eval time -> estimate latents
      if eval:

        # if we are explicitly given latents, use these
        if latent_override is not None:
          # pylogger.info(f"latent_override passed in forward_prediction")
          est_encoding = latent_override
        # or if we are explicitly given the privileged information
        elif priv_info_override is not None:
          # pylogger.info(f"priv_info passed in forward_prediction")
          priv_info = priv_info_override.clone()
          est_encoding = self.encoder(priv_info)
          if self.use_estimator_confidence and self.predictor_gets_confidence:
            # add the 'uncertainty' value to the true encodings
            if self.use_edl_confidence:
              if self.confidence_per_feature:
                raise RuntimeError(F"confidence_per_feature not supported with use_edl_confidence")
              nu = self.default_nu
              alpha = self.default_alpha
              std = 0.01 # known standard deviation of input
              beta = (std ** 2) * (alpha - 1.0)
              est_encoding = torch.concat([
                est_encoding, 
                # assign a very low log-uncertainty to priv info encodings
                nu * torch.ones((est_encoding.shape[0], 1), device=self.device), # nu
                alpha * torch.ones((est_encoding.shape[0], 1), device=self.device), # alpha
                beta * torch.ones((est_encoding.shape[0], 1), device=self.device), # nu
              ], dim=-1)
            else:
              est_encoding = torch.concat([
                est_encoding, 
                # assign a very low log-uncertainty to priv info encodings
                torch.log(torch.zeros((est_encoding.shape[0], self.conf_inds.shape[0]),
                                       device=self.device, dtype=est_encoding.dtype) + 1e-6),
              ], dim=-1)
        else:
          # normal case, estimate latents from observation
          est_encoding = self.estimator(obs_eval_estimator)
          # strip out any confidence values prior to predictor input
          if self.use_estimator_confidence and not self.predictor_gets_confidence:
            estimator_uncertainty = est_encoding[:, self.conf_inds].clone() # extract uncertainty
            est_encoding = est_encoding[:, self.latent_encoding_inds] # remove uncertainty

        # in case our latent_override or priv_info_override was 1D, then broadcast to batch
        if est_encoding.ndim == 1:
            est_encoding = einops.repeat(est_encoding, "n -> b n", b=obs_eval_estimator.shape[0])
      
        if not self.predictor_outputs_priv_info:
          est_repeated = einops.repeat(est_encoding, "b p -> b t p", t=obs_eval_predictor.shape[1])
          post_enc_obs = torch.concat([obs_eval_predictor, est_repeated], dim=-1)

      # at training time -> use ground truth latents
      else:

        # if we are explicitly given overriding privileged information
        if priv_info_override is not None:
          priv_info = priv_info_override.clone()
          pylogger.warning(f"priv_info_override set at training time")
        elif latent_override is not None:
          raise RuntimeError(f"LatentPredictor.forward_prediction() error: "
                             f"eval=False, but latent_override != None. Cannot "
                             f"override latents at training time currently.")
        else:
          # normal case (during training), encode the privileged information
          priv_info = observation[:, -1, self.i_priv_obs].clone()

        # encode the privileged info (with noise) into the latent space
        priv_encoding = self.encoder(priv_info.clone())

        # EXPERIMENTAL: add explicit noise onto the priv_encoding, based on the current performance
        if self.add_calibrated_latent_noise:
          rmse_noise_scale = self.last_rmse_estimator_error.unsqueeze(0).expand_as(priv_encoding)
          synthetic_noise = torch.randn_like(priv_encoding) * rmse_noise_scale
          priv_encoding += synthetic_noise
          if not hasattr(self, "debug_add_calibrated_latent_noise"):
            self.debug_add_calibrated_latent_noise = True
          if self.debug_add_calibrated_latent_noise:
            print(f"--- ADDING CALIBRATED NOISE TO LATENT SPACE OF TRUE PRIV ENCODINGS ---")
            self.debug_add_calibrated_latent_noise = False

        # are we sampling latents prior to predictor/decoder input
        if self.use_sampled_uncertainty:

          # # [NEW STRATEGY]: Fixed Variance Injection
          # # We assign a small constant variance to the Teacher (Encoder).
          # # log_var = -4.0 corresponds to std_dev approx 0.135
          # # log_var = -6.0 corresponds to std_dev approx 0.05
          # fixed_logvar_val = -5.0
          
          # # Create a tensor of shape [Batch, Latent_Dim] filled with this value
          # logvar_priv = torch.full_like(priv_encoding, fill_value=fixed_logvar_val)

          # new! use the random stdevs (which aren't applied to priv info) to peturb latents
          stdev_latents = self.noiser_priv_info.last_std_used.detach().squeeze(1) # (B, 1, N) -> (B, N)
          logvar_priv = torch.log(torch.pow(stdev_latents, 2) + 1e-6)

          # Sample using the fixed noise
          std_priv = torch.exp(0.5 * logvar_priv)
          eps = torch.randn_like(std_priv)
          
          # This is the "Jittered" ground truth
          priv_encoding = priv_encoding + eps * std_priv 
          
          # for decoding, use the 'noisy' version to enforce a smooth latent space
          decoded_latents = self.encoder.decoder(priv_encoding)
        else:
          # normal case, get decoding deterministically
          decoded_latents = self.encoder.decoder(priv_encoding)

        # get a no noise encoding (for training the estimator)
        if priv_info_no_noise is not None:
          priv_encoding_no_noise = self.encoder(priv_info_no_noise.clone())

        # create the estimated privileged info encodings
        if return_extras or self.train_with_estimated_latents:
          est_encoding = self.estimator(obs_eval_estimator.detach())
          # est_encoding = self.estimator(obs_eval_estimator) # test

          # if using estimator confidence
          if self.use_estimator_confidence:
            # do we use confidence to define a distribution to sample latents from
            if self.use_sampled_uncertainty:

              # get the uncertainty values from the estimated encodings
              estimator_uncertainty = est_encoding[:, self.conf_inds].clone() # -> (B, num_conf_inds)
              if self.confidence_per_feature:
                std_est = torch.exp(0.5 * estimator_uncertainty)
              else:
                # use same standard deviation for every dimension
                std_est = torch.exp(0.5 * einops.repeat(estimator_uncertainty,
                                                        "b t 1 -> b t n", 
                                                        n=self.latent_encoding_inds.shape[-1]))
              
              # sample estimated encoding from distribution, defined by the given uncertainty
              eps = torch.randn_like(std_est)
              if self.predictor_gets_confidence:
                # sampled encoding does include confidence values
                sampled_est_encoding = est_encoding.clone()
                sampled_est_encoding[..., self.latent_encoding_inds] = (
                  est_encoding[..., self.latent_encoding_inds] + eps * std_est)
              else:
                # sampled encoding should not include confidence values
                sampled_est_encoding = (
                  est_encoding[..., self.latent_encoding_inds] + eps * std_est)

            # do we pass confidence directly into the predictor
            if self.predictor_gets_confidence:
              if self.use_edl_confidence:
                # add uncertainty values to the true privileged encodings
                # the true noise level has shape (B, 1, N) when applied per batch -> (B)
                true_noise = self.noiser_priv_info.last_std_used.detach().squeeze(1) # (B, 1, N) -> (B, N)
                if self.confidence_per_feature:
                  raise RuntimeError(f"confidence_per_feature not supported with use_edl_confidence")
                else:
                  true_noise = torch.mean(true_noise, dim=1, keepdim=True) # (B, N) -> (B, 1)
                nu = self.default_nu
                alpha = self.default_alpha
                beta = (true_noise ** 2) * (alpha - 1.0)
                priv_encoding = torch.concat([
                  priv_encoding, 
                  nu * torch.ones((priv_encoding.shape[0], 1), device=self.device),
                  alpha * torch.ones((priv_encoding.shape[0], 1), device=self.device),
                  beta, # already shape (B, 1)
                ], dim=-1)

              else:
                # add uncertainty values to the true privileged encodings
                # the true noise level has shape (B, 1, N) when applied per batch -> (B)
                true_noise = self.noiser_priv_info.last_std_used.detach().squeeze(1) # (B, 1, N) -> (B, N)
                if self.confidence_per_feature:
                  if self.add_calibrated_latent_noise:
                    true_log_uncertainty = torch.log(torch.pow(rmse_noise_scale, 2) + 1e-6)
                  else:
                    # normal case, take stddev on priv info
                    true_log_uncertainty = torch.log(torch.pow(true_noise, 2) + 1e-6)
                else:
                  # arithmetic mean of per dimension noise variances (to combine uncertainties)
                  avg_variance = torch.mean(torch.pow(true_noise, 2), dim=1, keepdim=True) # (B, 1)
                  true_log_uncertainty = torch.log(avg_variance + 1e-6)
                priv_encoding = torch.concat([priv_encoding, true_log_uncertainty], dim=-1)
            else:
              # strip out any uncertainty values from the estimated encodings
              estimator_uncertainty = est_encoding[:, self.conf_inds].clone() # -> (B, num_conf_inds)
              est_encoding = est_encoding[:, self.latent_encoding_inds]
          
        # construct the final observation to pass into the predictor
        if self.train_with_estimated_latents:
          # special case: mix in the estimated latents at training time
          chance = infer_annealing(self.train_with_estimated_latents_chance,
                                   self.updates_done)

          batch = priv_encoding.shape[0]
          encoding_plus_variance = priv_encoding.shape[1]
          if self.predictor_gets_confidence:
            encoding_num = encoding_plus_variance // 2
          else:
            encoding_num = encoding_plus_variance # no variance actually exists in priv_encoding

          # determine whether to swap means+variances individually (True) or whole vectors (False)
          if self.mix_estimated_latents_elementwise:
            # get random values for means, then repeat the vector so variances are paired
            rand_values = einops.repeat(torch.rand((batch, encoding_num), device=self.device),
                                          "b n -> b (2 n)")
          else:
            # get a random decision at the batch level
            rand_values = einops.repeat(torch.rand((batch), device=self.device),
                                        "b -> b n", n=encoding_plus_variance)
            
          use_estimated = rand_values < chance

          # for debugging, print each new update loop
          if not hasattr(self, "mixed_latent_debug_count"):
            self.mixed_latent_debug_count = self.updates_done - 1
          if self.mixed_latent_debug_count != self.updates_done:
            pylogger.info(f" !!! Using mixed latents: chance = {chance:.3f} !!! "
                          f"detach_estimated_latents_before_mix = {self.detach_estimated_latents_before_mix}, "
                          f"mix_estimated_latents_elementwise = {self.mix_estimated_latents_elementwise}")
            self.mixed_latent_debug_count += 1

          # randomly select batches to use estimated states
          if self.use_estimator_confidence and self.use_sampled_uncertainty:
            est_encoding_used = sampled_est_encoding.clone()
          else:
            est_encoding_used = est_encoding.clone()
          if self.detach_estimated_latents_before_mix:
            est_encoding_used = est_encoding_used.detach()
          mixed_encoding = torch.where(
            condition=use_estimated,
            input=est_encoding_used,
            other=priv_encoding,
          )
          mixed_repeated = einops.repeat(mixed_encoding, "b p -> b t p", t=obs_eval_predictor.shape[1])
          post_enc_obs = torch.concat([obs_eval_predictor, mixed_repeated], dim=-1)
        else:
          # normal case: predictor gets encoding of privileged information
          priv_repeated = einops.repeat(priv_encoding, "b p -> b t p", t=obs_eval_predictor.shape[1])
          post_enc_obs = torch.concat([obs_eval_predictor, priv_repeated], dim=-1)
    else:
      # the main observation is simply the non-privileged information
      post_enc_obs = obs_eval_predictor

    # special case: disable any latent input into the predictor
    if self.disable_latents_into_predictor:
      post_enc_obs = obs_eval_predictor

    forward_args = {}
    
    # run the predictor to predict the next state
    fwd_output = self.predictor.forward(post_enc_obs, 
                                        return_extras=return_extras,
                                        **forward_args)

    if return_extras:
      prediction, extras = fwd_output
    else:
      prediction = fwd_output
      extras = {} # no extra info from 'forward'
    
    if return_extras and self.add_privileged_info:
      if self.use_estimator_confidence:
        if self.predictor_gets_confidence:
          # seperate out the estimated latents and the uncertainty (wasn't done already)
          estimator_uncertainty = est_encoding[:, self.conf_inds]
          est_encoding = est_encoding[:, self.latent_encoding_inds]
          if priv_encoding is not None:
            priv_encoding = priv_encoding[:, self.latent_encoding_inds] # remove the 'uncertainty' we added
        extras["estimator_uncertainty"] = estimator_uncertainty
      extras["est_latents"] = est_encoding
      extras["true_latents"] = priv_encoding
      extras["priv_info"] = priv_info
      extras["true_latents_no_noise"] = priv_encoding_no_noise
      extras["decoded_latents"] = decoded_latents
    if return_extras and self.add_denoise_step:
      extras["denoised_obs"] = denoised_obs    

    # return the output, optionally adding extra info
    if return_extras:
      return prediction, extras
    else:
      return prediction

  def rollout_prediction(self, 
                         observation, 
                         n_predictions, 
                         ground_truth=None,
                         future_actions=None,
                         eval=False,
                         priv_info_override=None,
                         latent_override=None,
                         return_extras=False,
                         handle_normalisation=True,
                         enable_noise=False):
    """
    Roll forward a series of n_predictions, based on an initial history of
    observations of shape: [B, T, N], where B=batch_size, T=num_timesteps_hist_to_use,
    and N=feature_dimension.
    """

    # normalise all incoming inputs
    if handle_normalisation and self.use_normalisation:
      debug_norm = False # set to true to add debug info about normalisation
      if not hasattr(self, "debug_norm"):
        self.debug_norm = None
      else:
        debug_norm = False # only do it on the first call
      
      if debug_norm: print(f"About to normalise the 'observation' in rollout_prediction"
                           f", shape = {observation.shape}, eval = {eval}, "
                           f"norm_mean_eval.shape = {self.norm['eval']['mean'].shape}, "
                           f"norm_std_eval.shape = {self.norm['eval']['std'].shape}")
      if eval:
        observation = apply_norm(observation, debug=debug_norm,
                                 mean=self.norm['eval']['mean'], 
                                 std=self.norm['eval']['std'])
      else:
        print("WARNING: hit non-eval normalisation branch in rollout_prediction()")
        observation = apply_norm(observation, debug=debug_norm,
                                 mean=self.norm['full']['mean'], 
                                 std=self.norm['full']['std'])
      if ground_truth is not None:
        if debug_norm: print(f"About to normalise the 'ground_truth' in rollout_prediction"
                             f", shape = {ground_truth.shape}, eval = {eval}, "
                             f"norm_mean_pred.shape = {self.norm['pred']['mean'].shape}, "
                             f"norm_std_pred.shape = {self.norm['pred']['std'].shape}")
        ground_truth = apply_norm(ground_truth, debug=debug_norm,
                                  mean=self.norm['pred']['mean'],
                                  std=self.norm['pred']['std'])
      if future_actions is not None:
        if debug_norm: print(f"About to normalise the 'future_actions' in rollout_prediction"
                             f", shape = {future_actions.shape}, eval = {eval}, "
                             f"norm_mean_actions.shape = {self.norm['action']['mean'].shape}, "
                             f"norm_std_actions.shape = {self.norm['action']['std'].shape}")
        future_actions = apply_norm(future_actions, debug=debug_norm,
                                    mean=self.norm['action']['mean'],
                                    std=self.norm['action']['std'])
        
      if priv_info_override is not None:
        priv_info_override = apply_norm(priv_info_override, debug=debug_norm,
                                        mean=self.norm['priv_info']['mean'],
                                        std=self.norm['priv_info']['std'])
        
    # SPECIAL CASE! COPY LAST OBSERVED PRIV INFO AS GROUND TRUTH #
    # this accounts for the case where priv info changes during the rollout
    # but there is no way for the model to ever know it has changed
    if not eval and ground_truth is not None:
      ground_truth[:, :, self.i_priv_obs] = einops.repeat(observation[:, -1, self.i_priv_obs],
                                                          "b n -> b t n", t=ground_truth.shape[1])
    
    # handle if we will add training-time noise to the initial observation
    if (enable_noise and not eval):
      # the actions are overriden by the 'action_sequence' below, so no noise here
      noisy_observation = self.add_noise(observation.clone(), action_noise=False)
      noisy_ground_truth = ground_truth.clone()
      noisy_ground_truth = self.add_noise(noisy_ground_truth, action_noise=False)
    else:
      # no noise is actually added
      noisy_observation = observation.clone()
      if ground_truth is None:
        noisy_ground_truth = None
      else:
        noisy_ground_truth = ground_truth.clone()

    B, T, N = observation.shape
    predictions_rollout = torch.zeros((B, n_predictions, self.n_q * 2), 
                                       device=self.device)
    
    if hasattr(self.predictor, "prediction_timesteps"):
      prediction_timesteps = self.predictor.prediction_timesteps
    else:
      prediction_timesteps = 1

    # how many loops to get all the predictions we want
    num_steps = math.ceil(n_predictions / prediction_timesteps)

    # --- start new test for unified future actions handling --- #
    if self.add_next_action:

      future_actions_already_in_observation = 1
      req_future_actions = (num_steps * prediction_timesteps) - future_actions_already_in_observation
    
      # continous sequence of unique actions, first input all we observe from observation
      action_sequence = torch.zeros((B, T + req_future_actions, self.i_action.shape[0]),
                                    device=self.device)
      action_sequence[:, :T] = observation[:, :, self.i_action] # take from no noise obs

      # add in future actions, eval time
      if eval:
        if future_actions is None and n_predictions != 1:
          raise RuntimeError(F"LatentPredictor.rollout_prediction() error: "
                             f"future_actions is None, but must be given when eval=True")
        if future_actions.shape[1] != n_predictions - 1:
          raise RuntimeError(F"LatentPredictor.rollout_prediction() error: "
                             f"future_actions.shape={future_actions.shape}, "
                             f"expected number of timesteps to be number of predictions "
                             f"- 1. n_pred={n_predictions}")
        
        # insert the actions we know (zeros beyond, those predictions are dropped)
        action_insert = slice(T, T + n_predictions - 1) # we know these upcoming actions
        action_sequence[:, action_insert] = future_actions
        noisy_action_sequence = action_sequence # identical, no noise added (eval=True)
      
      # add in future actions, training time
      else:
        if future_actions is not None:
          raise RuntimeError(f"LatentPredictor.rollout_prediction() error: "
                             f"future_actions is not None, but eval=False")

        # insert the actions we know (zeros beyond, those predictions are dropped)
        action_insert = slice(T, T + n_predictions - 1) # we know these upcoming actions
        action_sequence[:, action_insert] = ground_truth[:, :-1, self.i_action]

        # handle adding noise to the action sequence at training time
        if (enable_noise and not eval):
          noisy_action_sequence = self.add_noise(action_sequence.clone(), action_noise=True)
        else:
          noisy_action_sequence = action_sequence

    # --- end new test for future actions prep --- #

    if self.use_gt_chance > 1e-5:
      gt_mask = torch.rand((B, self.n_q * 2), device=self.device) < self.use_gt_chance
      gt_mask = einops.repeat(gt_mask, "b n -> b t n", t=prediction_timesteps)

    # temporary for debugging!
    if prediction_timesteps != 1:
      if not hasattr(self, "temp_debug_flag"):
        self.temp_debug_flag = 0
      if self.temp_debug_flag % 1000 == 0:
        pylogger.info(f"rollout_prediction: \n"
                      f" -> n_predictions = {n_predictions}\n"
                      f" -> num_steps = {num_steps}\n"
                      f" -> prediction_timesteps = {prediction_timesteps}\n")
      self.temp_debug_flag += 1

    current_with_noise_obs = noisy_observation # actions not yet shifted
    current_no_noise_obs = observation
    estimator_obs = None # estimator uses same obs as predictor by default
    extras_rolled_out = {}

    for n in range(num_steps):

      # define indexes and details of current chunk of predictions
      pred_start = n * prediction_timesteps
      pred_end = min((n + 1) * prediction_timesteps, n_predictions)
      pred_inds = slice(pred_start, pred_end)
      n_pred_use = pred_end - pred_start

      # determine the indexes of the actions to pass with the observation
      if self.add_next_action:
        obs_action_start = n * prediction_timesteps # first action in observation
        obs_action_end = obs_action_start + T # observation has fixed length
        action_shift = prediction_timesteps - future_actions_already_in_observation

        # handle seperate observations for predictor vs estimator
        if self.seperate_estimator_obs:
          estimator_obs = current_with_noise_obs.clone()
          est_actions = slice(obs_action_start, obs_action_end)
          estimator_obs[:, :, self.i_action] = noisy_action_sequence[:, est_actions].clone()

        # shift actions in the observation, to handle longer horizon predictions in one
        seen_actions = slice(obs_action_start + action_shift,
                             obs_action_end + action_shift)
        current_with_noise_obs[:, :, self.i_action] = noisy_action_sequence[:, seen_actions].clone()
        current_no_noise_obs[:, :, self.i_action] = action_sequence[:, seen_actions].clone()

      # at training time, provide a ground truth target for diffusion models
      if not eval:
        # future = slice(pred_end, pred_end + prediction_timesteps)
        gt_target = ground_truth[:, pred_inds, self.i_qs].clone()
        # take the privileged info on the first loop only, then keep it
        # this means we regress to observable priv info, even if it changes
        if n == 0:
          priv_info_no_noise = current_no_noise_obs[:, -1, self.i_priv_obs].clone()
      else:
        gt_target = None
        priv_info_no_noise = None

      # get the next state/s prediction
      fwd_output = self.forward_prediction(current_with_noise_obs.clone(), 
                                           estimator_observation=estimator_obs.clone(),
                                           gt_target=gt_target, # only used in diffusion training
                                           eval=eval,
                                           return_extras=return_extras,
                                           latent_override=latent_override,
                                           priv_info_override=priv_info_override,
                                           priv_info_no_noise=priv_info_no_noise)
      
      # if returning extra info, save the rollouts of this as well
      if return_extras:
        new_prediction, extras = fwd_output
        if n == 0:
          for key in extras:
            if extras[key] is None: continue # safety check, skip any 'None' values
            if extras[key].ndim == 2:
              # shape (B, N), one value per forward call (e.g. estimated latents)
              extras_rolled_out[key] = torch.zeros((extras[key].shape[0], num_steps, 
                                                    extras[key].shape[-1]), device=self.device)
            elif extras[key].ndim == 3:
              if extras[key].shape[1] == n_predictions:
                # shape (B, T, N), one value per timestep (e.g. acceleration prediction)
                extras_rolled_out[key] = torch.zeros((extras[key].shape[0], n_predictions, 
                                                      extras[key].shape[-1]), device=self.device)
              else:
                # shape (B, Tx!=T, N), one batch of unknown size (Tx) per forward
                extras_rolled_out[key] = torch.zeros((extras[key].shape[0], num_steps, 
                                                      *extras[key].shape[1:]), 
                                                     device=self.device)
                if self.add_denoise_step and key == "denoised_obs":
                  extras_rolled_out["true_obs"] = torch.zeros((extras[key].shape[0], num_steps, 
                                                               *extras[key].shape[1:]), 
                                                              device=self.device)

            else:
              raise RuntimeError(f"LatentPredictor.rollout_prediction() error: "
                                 f"extras[{key}].shape = {extras[key].shape}, "
                                 f"expected 2 or 3 dimensions only")
        for key in extras:
          if extras[key] is None: continue # safety check, skip any 'None' values
          if extras[key].ndim == 2:
            extras_rolled_out[key][:, n] = extras[key] # one value per 'forward''
          elif extras[key].ndim == 3:
            if extras[key].shape[1] == n_predictions:
              extras_rolled_out[key][:, pred_inds] = extras[key] # one value per timestep
            else:
              extras_rolled_out[key][:, n] = extras[key] # one 'batch' of values per 'forward'
              if self.add_denoise_step and key == "denoised_obs":
                extras_rolled_out["true_obs"][:, n] = (current_no_noise_obs[:, :, self.i_qs]
                                                       .clone().detach())
          else:
              raise RuntimeError(f"LatentPredictor.rollout_prediction() error: "
                                 f"extras[{key}].shape = {extras[key].shape}, "
                                 f"expected 2 or 3 dimensions only")
      else:
        new_prediction = fwd_output
      
      # maintain support for 2D single step predictions
      if new_prediction.ndim == 2:
        new_prediction = new_prediction.unsqueeze(1) # (B, N) -> (B, 1, N)

      # save the model prediction at this rollout step
      assert new_prediction.shape == (B, prediction_timesteps, self.n_q * 2)
      predictions_rollout[:, pred_inds] = new_prediction[:, :n_pred_use] # clip end if needed

      # if we continue predicting, prepare the next observation
      if n + 1 < num_steps:

        n_obs = current_with_noise_obs.shape[1]
        update_inds = slice(n_obs - prediction_timesteps, n_obs)

        next_with_noise_obs = torch.zeros_like(current_with_noise_obs)
        next_with_noise_obs[:, :-prediction_timesteps] = current_with_noise_obs[:, prediction_timesteps:]

        next_no_noise_obs = torch.zeros_like(current_no_noise_obs)
        next_no_noise_obs[:, :-prediction_timesteps] = current_no_noise_obs[:, prediction_timesteps:]

        # recursively update observation, evaluation time
        if eval:
          next_with_noise_obs[:, update_inds, self.i_qs] = new_prediction

        # recursively update observation, training time
        else:
    
          next_with_noise_obs[:, update_inds] = noisy_ground_truth[:, pred_inds] # assign full ground truth (inc. priv info)
          next_with_noise_obs[:, update_inds, self.i_qs] = new_prediction  # overwrite with the new prediction

          next_no_noise_obs[:, update_inds] = ground_truth[:, pred_inds] # assign full ground truth (inc. priv info)
          next_no_noise_obs[:, update_inds, self.i_qs] = new_prediction  # overwrite with the new prediction

          # mask out predictions with true states so learning is easier
          if ground_truth is not None and self.use_gt_chance > 1e-5:
            next_with_noise_obs[:, update_inds, self.i_qs] = torch.where(
                gt_mask,                                        # constant mask
                noisy_ground_truth[:, pred_inds, self.i_qs],    # cond is true
                next_with_noise_obs[:, update_inds, self.i_qs], # cond if false
            )
            next_no_noise_obs[:, update_inds, self.i_qs] = torch.where(
                gt_mask,                                        # constant mask
                ground_truth[:, pred_inds, self.i_qs],          # cond is true
                next_no_noise_obs[:, update_inds, self.i_qs],   # cond if false
            )

        # prepare for next loop
        current_with_noise_obs = next_with_noise_obs
        current_no_noise_obs = next_no_noise_obs

    # do we denormalise, should happen at eval time but not at training time
    if handle_normalisation and self.use_normalisation:
      if debug_norm: print(f"About to revert normalisations for 'predictions_rollout'"
                           f" in rollout_prediction, shape = {predictions_rollout.shape}")
      predictions_rollout = revert_norm(predictions_rollout, debug=debug_norm,
                                        mean=self.norm["pred"]["mean"],
                                        std=self.norm["pred"]["std"])
      # add a normalised field for prediction uncertainty
      if "predictor_uncertainty" in extras_rolled_out and return_extras and self.use_predictor_confidence:
        if debug_norm:
            pylogger.info(f"De-normalising the predictor uncertainty as well, with "
                          f"multivariate={self.multivariate_uncertainty}, "
                          f"confidence_per={self.confidence_per_feature}")
        if self.multivariate_uncertainty:
          # 1. Get the raw Cholesky parameters
          # Shape: (..., cholesky_dim) e.g., (B, Rollouts, Horizon, 28)
          raw_cholesky = extras_rolled_out["predictor_uncertainty"]
          
          # 2. Flatten batch dimensions for matrix construction
          batch_shape = raw_cholesky.shape[:-1]
          input_dim = raw_cholesky.shape[-1]
          # Solve D*(D+1)/2 = input_dim to find D (e.g., 28 -> 7)
          # This is the reverse of n(n+1)/2
          output_dim = int((-1 + (1 + 8 * input_dim)**0.5) / 2) 

          flat_cholesky = raw_cholesky.reshape(-1, input_dim)
          
          # 3. Reconstruct L (Identical to Loss Function logic)
          tril_indices = torch.tril_indices(row=output_dim, col=output_dim, offset=0, 
                                            device=raw_cholesky.device)
          L = torch.zeros(flat_cholesky.shape[0], output_dim, output_dim, 
                          device=raw_cholesky.device)
          L[:, tril_indices[0], tril_indices[1]] = flat_cholesky
          
          # Apply Softplus + Epsilon to diagonal (MUST match your Loss function)
          diag_indices = range(output_dim)
          L[:, diag_indices, diag_indices] = torch.nn.functional.softplus(
              L[:, diag_indices, diag_indices]
          ) + 1e-6
          
          # 4. Calculate Marginal Standard Deviations
          # The std dev of variable i is the L2 norm of the ith row of L
          # Shape: (N, D)
          marginal_stddevs_flat = torch.linalg.norm(L, dim=-1)
          
          # 5. Reshape back to original batch structure
          # Shape: (..., D) e.g., (B, Rollouts, Horizon, 7)
          pred_stddev = marginal_stddevs_flat.reshape(*batch_shape, output_dim)

          # 6. Denormalise
          extras_rolled_out["predictor_stddev_denormalised"] = revert_norm(
              pred_stddev,
              mean=torch.zeros_like(self.norm["pred"]["mean"]), # Mean shift doesn't affect spread
              std=self.norm["pred"]["std"],
          )

        elif self.confidence_per_feature:
          pred_stddev = torch.pow(torch.exp(extras_rolled_out["predictor_uncertainty"]), 0.5)
          extras_rolled_out["predictor_stddev_denormalised"] = revert_norm(
            pred_stddev,
            mean=torch.zeros_like(self.norm["pred"]["mean"]),
            std=self.norm["pred"]["std"],
          )
        else:
          # cast the same uncertainty to each dimension seperately (via denormalisation)
          pred_stddev = torch.pow(torch.exp(extras_rolled_out["predictor_uncertainty"]), 0.5)
          extras_rolled_out["predictor_stddev_denormalised"] = revert_norm(
            einops.repeat(pred_stddev, "b t pred_h 1 -> b t pred_h n", n=pred_stddev.shape[-1]),
            mean=torch.zeros_like(self.norm["pred"]["mean"]),
            std=self.norm["pred"]["std"],
          )
      # experimental: if latents come out of the predictor
      if self.predictor_outputs_priv_info and return_extras:
        # pylogger.warning(f"rollout_prediction is swapping predictor latents for estimator!!")
        extras["est_latents"] = extras["predictor_est_latents"]
        extras["latent_uncertainty"] = extras["predictor_latent_uncertainty"]

    if return_extras:
      return predictions_rollout, extras_rolled_out
    else:
      return predictions_rollout

  def update(self, trajectory_batch):
    """
    Apply backprop to improve the model
    """

    # ensure we are in training mode
    self.train()

    # set up noise to be added to observation throughout this one rollout
    if self.use_noise:
      debug_noise = True
      self.prepare_noise(debug_noise=debug_noise)
      
    # metrics to track, such as errors, loss statistics
    tracked_metrics = {}

    # data structure for saving info, add an 'alpha' for every possible loss
    loss_alphas = self.get_loss_alphas()

    # will we return extra information from our forward predictions
    return_extras = bool(self.add_privileged_info + self.use_full_observation)

    # will we update the estimator at this step
    update_estimator = self.updates_done % self.estimator_update_rate == 0

    # determine split of timesteps in batch
    B, T, N = trajectory_batch.shape
    n_hist = self.n_t
    n_pred = self.prediction_horizon
    update_ratio = max(1, int(self.n_t / self.num_use_each_datapoint)) # must be at least 1
    n_updates = ((T - n_pred - n_hist) // update_ratio) + 1

    # new for advanced sampling
    max_start_idx = T - n_pred - n_hist # largest timestep we can start from

    if self.use_mini_batch_sampling:
      n_updates = int(n_updates * (B / self.mini_batch_size)) # scale up updates by batch divisions
      pylogger.info(f"Using new sampling scheme, with mini_batch={self.mini_batch_size}")
    elif self.num_use_each_datapoint < 1.0:
      raise RuntimeError(f"self.num_use_each_datapoing={self.num_use_each_datapoint} "
                         f"but use_mini_batch_sampling=False. In this case, the num "
                         f"use each datapoint should be an integer greater than 1")

    # for debugging
    pylogger.info(f"n_updates = {n_updates}, with update ratio = {update_ratio}"
                   f", n_pred = {n_pred}, total timesteps T = {T}, "
                   f"num_use_each_datapoint = {self.num_use_each_datapoint}, "
                   f"update_estimator={update_estimator}")
    
    # create a dictionary of metrics which we will return
    loss_metrics = { 
      "total_loss" : 0,
    }

    self.rmse_estimator_error = None
    self.last_rmse_estimator_error = 0.25 * torch.ones((self.num_latent_encoding), device="cuda")
    self.debug_add_calibrated_latent_noise = True

    # shuffle the temporal order (only affects old sampling method)
    update_indices = torch.randperm(n_updates)

    for n, t_num in enumerate(update_indices):

      # fresh loss for this loop
      loss = {}

      # sample randomly in time, in mini-batches
      if self.use_mini_batch_sampling:

        # sample for batch and time dimensions, both have shape (mini_batch_size)
        batch_indices = torch.randint(0, B, (self.mini_batch_size,), device=self.device)
        time_indices = torch.randint(0, max_start_idx + 1, (self.mini_batch_size,), device=self.device)
        
        # Create time indices for the history and future, both shape (mini_batch_size, n_hist)
        past_indices = time_indices.unsqueeze(1) + torch.arange(n_hist, device=self.device).unsqueeze(0)
        future_indices = time_indices.unsqueeze(1) + torch.arange(n_hist, n_hist + n_pred, device=self.device).unsqueeze(0)
 
        # expand batch indexes to match in the time dimension, both shape (mini_batch_size, n_hist)
        b_idx_past = batch_indices.unsqueeze(1).expand(-1, n_hist) 
        b_idx_future = batch_indices.unsqueeze(1).expand(-1, n_pred)
        
        # index out the randomised batches and randomised sequences of time indexes
        observations = trajectory_batch[b_idx_past, past_indices].clone()
        full_ground_truth = trajectory_batch[b_idx_future, future_indices].clone()

      # sample in temporal sequence
      else:

        # determine indexes for previous history, and future predictions
        in_start = t_num.item() * update_ratio
        in_end = in_start + self.n_t
        out_start = in_end
        out_end = out_start + n_pred
        past = slice(in_start, in_end)
        future = slice(out_start, out_end)
        pylogger.debug(f"n={n}, t_num={t_num.item()}, in -> {in_start} : {in_end}, out -> {out_start} : {out_end} (T={T})")

        # prepare to pass in everything, including hidden information
        observations = trajectory_batch[:, past].clone()        # keep batch clean
        full_ground_truth = trajectory_batch[:, future].clone() # keep batch clean

      # Normalize only the sampled windows. Normalizing trajectory_batch before
      # sampling creates a second full-size GPU tensor and can exhaust memory.
      if self.use_normalisation:
        if n == 0:
          pylogger.info(f"Normalising sampled trajectory windows in update()")
        observations = apply_norm(
            observations,
            mean=self.norm["full"]["mean"],
            std=self.norm["full"]["std"],
            debug=(n == 0),
        )
        full_ground_truth = apply_norm(
            full_ground_truth,
            mean=self.norm["full"]["mean"],
            std=self.norm["full"]["std"],
            debug=False,
        )

      true_traj = full_ground_truth[..., self.i_qs].clone()

      # query the model and get the predicted trajectories
      pred_outputs = self.rollout_prediction(
        observation=observations,
        n_predictions=n_pred,
        future_actions=None, # passed in via ground truth
        ground_truth=full_ground_truth,
        return_extras=return_extras,
        handle_normalisation=False, # already handled above for training time
        eval=False, # can never be true at training time
        enable_noise=self.use_noise,
      )

      if return_extras:
        predicted_traj, extras = pred_outputs
      else:
        predicted_traj = pred_outputs

      # calculate the prediction losses (add also actuated and regular)
      # mae_loss_fn = torch.nn.L1Loss(reduction="none") # no reduction, we apply tapers
      # loss["prediction"] = self.get_tapered_loss(predicted_traj, true_traj, loss_fn=mae_loss_fn)
      loss["prediction"] = self.get_tapered_loss(predicted_traj, true_traj.detach())
      loss["actuated"] = self.get_tapered_loss(predicted_traj[:, :, self.act_joint_inds], 
                                               true_traj[:, :, self.act_joint_inds])
      loss["payload"] = self.get_tapered_loss(predicted_traj[:, :, self.payload_joint_inds], 
                                              true_traj[:, :, self.payload_joint_inds])
      
      if self.use_predictor_confidence:
        
        # predicted_uncertainty.shape (B, num_rollouts, pred_horizon, num_uncertainties)
        # for example, predicted_traj (256, 30, 14), and uncertainty (256, 3, 10, 1)
        predicted_uncertainty = extras["predictor_uncertainty"]
        _, num_rollouts, horizon, num_unc_params = predicted_uncertainty.shape

        # multivariate uncertainty case
        if self.multivariate_uncertainty:

          if n == 0:
            pylogger.info(f"Calculating the multi-variate loss for the predictions")
            
          # 1. Align Trajectories to Uncertainty Structure
          # predicted_traj comes in as (B, Total_Time, D), e.g., (256, 30, 7)
          # We must split Total_Time into (num_rollouts, horizon) using einops
          traj_aligned = einops.rearrange(
              predicted_traj, 
              "b (r h) d -> b r h d", 
              r=num_rollouts, h=horizon
          )
          true_aligned = einops.rearrange(
              true_traj.detach(), 
              "b (r h) d -> b r h d", 
              r=num_rollouts, h=horizon
          )
          output_dim = traj_aligned.shape[-1]

          # 2. Flatten Batch/Time dims for Vectorized Matrix Construction
          # We treat every timestep in every rollout as an independent sample for the loss
          # Shape becomes (N_samples, ...)
          flat_uncertainty = predicted_uncertainty.reshape(-1, num_unc_params) 
          flat_traj = traj_aligned.reshape(-1, output_dim)
          flat_true = true_aligned.reshape(-1, output_dim)

          # 3. Construct Lower Triangular Matrix (L)
          tril_indices = torch.tril_indices(row=output_dim, col=output_dim, offset=0, 
                                            device=predicted_uncertainty.device)
          L = torch.zeros(flat_uncertainty.shape[0], output_dim, output_dim, 
                          device=predicted_uncertainty.device)
          L[:, tril_indices[0], tril_indices[1]] = flat_uncertainty
          
          # 4. Enforce Positive Definiteness on Diagonal
          diag_indices = torch.arange(output_dim, device=predicted_uncertainty.device)
          L[:, diag_indices, diag_indices] = torch.nn.functional.softplus(
              L[:, diag_indices, diag_indices]
          ) + 1e-6

          # 5. Calculate Loss
          # MultivariateNormal handles the Cholesky math (log_det, Mahalanobis) efficiently
          dist = torch.distributions.MultivariateNormal(loc=flat_traj, scale_tril=L)
          log_prob = dist.log_prob(flat_true) # Returns shape (N_samples,)
          
          loss["prediction_confidence"] = -log_prob.mean()

        # regular diagonal covariance matrix
        else:

          # model outputs the log variance
          predicted_uncertainty_log_var = predicted_uncertainty

          if self.confidence_log_var_clamping is not None:
            predicted_uncertainty_log_var = torch.clamp(predicted_uncertainty_log_var,
                                                        self.confidence_log_var_clamping[0],
                                                        self.confidence_log_var_clamping[1])
          pred_unc = torch.exp(predicted_uncertainty_log_var) # -> sigma**2, shape (B, T, N)
          # calculate the MSE loss at each timestep, per feature
          pred_error_norm = torch.pow(predicted_traj - true_traj.detach(), 2) # shape (B, T, N)
          # compute error for each rollout, and along the whole horizon
          pred_error_norm = einops.rearrange(pred_error_norm, "b (r h) n -> b r h n",
                                             r=num_rollouts, h=horizon)
          # calculate the negative-loss-likelihood at each timestep
          pred_per_timestep_loss = ((1 / (2 * pred_unc)) * pred_error_norm
                                    + 0.5 * predicted_uncertainty_log_var)
          loss["prediction_confidence"] = pred_per_timestep_loss.mean() # get mean loss (don't feed via MSE)
      
      # diffusion losses (if present)
      if "predicted_noise" in extras:
        loss["diffusion"] = self.get_loss(extras["predicted_noise"], extras["actual_noise"])
      
      # enforce a loss to ensure that predicted velocity is the derivative of predicted position
      if self.use_normalisation:
        SI_predicted_traj = revert_norm(predicted_traj, debug=False,
                                        mean=self.norm["pred"]["mean"],
                                        std=self.norm["pred"]["std"])
      else:
        SI_predicted_traj = predicted_traj
      SI_predicted_qpos = SI_predicted_traj[:, :, self.i_qpos]
      SI_predicted_qvel = SI_predicted_traj[:, :, self.i_qvel]
      diff_velocity = torch.diff(SI_predicted_qpos, dim=1) / self.predictor.dt
      loss["integrate_pos"] = self.get_tapered_loss(SI_predicted_qvel[:, :-1], 
                                                    diff_velocity)
      
      # loss terms when using the ROA style privileged encoder
      if self.add_privileged_info:
        
        # compute values needed for below loss calculation
        if not hasattr(self, "latent_target_repeated"):
          self.latent_target_repeated = einops.repeat(
            torch.tensor([self.target_latent_norm], device=self.device),
            "1 -> b t", b=B, t=n_pred
          )

        # take the privileged ground truth value from the last observed true data
        # n_encodings = extras["true_latents"].shape[1]
        # priv_ground_truth = trajectory_batch[:, out_start - 1, self.i_priv_obs] # value at end of obs
        # priv_gt_repeated = einops.repeat(priv_ground_truth, "b n -> b t n", t=n_encodings)

        # get the latent predictions
        priv_info_with_noise = extras["priv_info"] # priv info WITH noise (decoder doesn't care about real truth)
        true_latents_with_noise = extras["true_latents"]
        true_latents_without_noise = extras["true_latents_no_noise"]
        decoded_latents = extras["decoded_latents"]
        est_latents = extras["est_latents"]

        # experimental for baselines: predictor outputs latents
        if self.predictor_outputs_priv_info:
          # shape (B, T, N)
          est_latents = extras["predictor_est_latents"]

        # regularise the latents with a log-barrier, with a target minimum 'target_norm'
        # a per element average of 'target_latent_norm' gives the following 'target_norm'
        # latent shape = (batch, n_pred, feature_dim)
        d = true_latents_without_noise.shape[-1]
        latent_norms = true_latents_without_noise.norm(dim=-1)
        target_norm = math.sqrt(d) * self.target_latent_norm # min or m*x - exp(x)
        ratio = torch.clamp(latent_norms / target_norm, min=1e-6, max=1e6)
        loss["latent_norm"] = -torch.log(ratio).mean() + ratio.mean()
        loss["latent_norm_mse"] = (latent_norms - target_norm).pow(2).mean()

        # # test extra latent control
        # eigenvals = self.get_eigenvalues(true_latents)
        # condition_number = eigenvals.max() / eigenvals.min()
        # dim_importance_ratio = 10.0 # most important dim can have 10x more impact than least
        # loss["latent_anisotrophy"] = torch.relu(condition_number - dim_importance_ratio)

        # key losses for training the encoder
        loss["regularisation"] = self.get_tapered_loss(est_latents.detach(), true_latents_without_noise)
        # decoder should decode what was encoded, i.e., noisy priv info, not perfect priv info
        loss["decoder"] = self.get_tapered_loss(decoded_latents, priv_info_with_noise)

        # save metrics regarding the latents
        if n == 0:
          tracked_metrics["true_latent_norm"] = torch.mean(true_latents_without_noise.norm(dim=-1)).item()
          tracked_metrics["est_latent_norm"] = torch.mean(est_latents.norm(dim=-1)).item()
        else:
          tracked_metrics["true_latent_norm"] += torch.mean(true_latents_without_noise.norm(dim=-1)).item()
          tracked_metrics["est_latent_norm"] += torch.mean(est_latents.norm(dim=-1)).item()

        if update_estimator:
          if n == 0:
            pylogger.info(f"Updating estimator (update={self.updates_done}, "
                          f"rate={self.estimator_update_rate})")
            
          # penalise temporal changes in latents
          if est_latents.shape[1] > 1:
            std_devs = torch.std(est_latents, dim=1)
            loss["consistency"] = self.get_loss(std_devs, torch.zeros_like(std_devs))
            loss["smoothing"] = self.get_loss(est_latents[:, :-1],
                                               est_latents[:, 1:])

          # update the estimator to track the encoder 'true' latents
          loss["estimator"] = self.get_loss(est_latents, true_latents_without_noise.detach())

          # calculate the normalised estimator loss (scale independent)
          mean_true_latents = torch.mean(true_latents_without_noise, dim=0).detach()
          std_true_latents = torch.std(true_latents_without_noise, dim=0).detach()
          loss["estimator_norm"] = self.get_loss(
            (est_latents - mean_true_latents) / std_true_latents,
            ((true_latents_without_noise - mean_true_latents) / std_true_latents).detach(),
          )

          # if the estimator is outputting uncertainty values wrt to the latents
          if self.use_estimator_confidence:

            # evidential deep learning, quantify uncertainty
            if self.use_edl_confidence:

              # extract the distributional parameters
              # print(f'extras["estimator_uncertainty"].shape = {extras["estimator_uncertainty"].shape}')
              nu_raw = extras["estimator_uncertainty"][:, :, 0:1]
              alpha_raw = extras["estimator_uncertainty"][:, :, 1:2]
              beta_raw = extras["estimator_uncertainty"][:, :, 2:3]

              # clamp to a reasonable range to avoid explosion
              nu_raw = torch.clamp(nu_raw, min=-10.0, max=10.0)

              # process for stability (ensure positivity, and alpha >= 1)
              nu = F.softplus(nu_raw) + 1e-6
              beta = F.softplus(beta_raw) + 1e-6
              alpha = F.softplus(alpha_raw) + 1.0 + 1e-6

              # calculate error
              error = torch.abs(est_latents - true_latents_without_noise.detach()) # shape (B, T, N)

              term_1 = 0.5 * torch.log(torch.pi / nu)
              term_2 = -alpha * torch.log(beta)
              term_3 = (alpha + 0.5) * torch.log(beta + 0.5 * nu * (error ** 2))
              term_4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

              nll_loss = (term_1 + term_2 + term_3 + term_4).mean()
              reg_loss = (error * (2.0 * nu + alpha)).mean()

              loss["estimator_confidence"] = nll_loss + 1e-3 * reg_loss

            # negative log-likelihood loss
            else:

              # the uncertainty value is represented by the log variance
              latent_uncertainty_log_var = extras["estimator_uncertainty"]

              # experimental: support latents and uncertainty output from the predictor
              if self.predictor_outputs_priv_info:
                if n == 0:
                  pylogger.warning(f"predictor_outputs_priv_info=True, with use_estimator_confidence=True")
                if "predictor_latent_uncertainty" in extras:
                  # shape (B, T, N)
                  latent_uncertainty_log_var = extras["predictor_latent_uncertainty"]
                else:
                  raise RuntimeError(f"LatentPredictor.update() error: "
                                     f"self.predictor_outputs_priv_info=True, and "
                                     f"self.use_estimator_confidence=True, "
                                     f"but predictor does NOT output any confidence values")

              if not self.confidence_per_feature:
                latent_uncertainty_log_var = einops.repeat(latent_uncertainty_log_var,
                                                       "b t 1 -> b t n", n=self.num_latent_encoding)

              if self.confidence_log_var_clamping is not None:
                latent_uncertainty_log_var = torch.clamp(latent_uncertainty_log_var,
                                                         self.confidence_log_var_clamping[0],
                                                         self.confidence_log_var_clamping[1])
              
              latent_unc = torch.exp(latent_uncertainty_log_var) # -> sigma**2, shape (B, T, N)
              # calculate the MSE loss at each timestep, per feature
              error_norm = torch.pow(est_latents - true_latents_without_noise.detach(), 2) # shape (B, T, N)
              # calculate the negative-loss-likelihood at each timestep
              per_timestep_loss = ((1 / (2 * latent_unc)) * error_norm
                                    + 0.5 * latent_uncertainty_log_var)
              loss["estimator_confidence"] = per_timestep_loss.mean() # get mean loss (don't feed via MSE)
              
              if est_latents.shape[1] > 1:
                # get the squared error between adjacent latent predictions
                smooth_norm = torch.pow(est_latents[:, :-1] - est_latents[:, 1:], 2) # shape (B, T-1, N)
                # average adjacent uncertainties to get our 'tolerance' to changes, shape (B, T-1, N)
                smooth_unc = latent_unc[:, :-1] + latent_unc[:, 1:] # sum the two variances
                per_timestep_smooth_loss = ((1 / (2 * smooth_unc)) * smooth_norm
                                            + 0.5 * torch.log(smooth_unc))
                loss["smoothing_confidence"] = per_timestep_smooth_loss.mean()

              # keep a running total of how well the estimator is doing
              error_per_estimated_latent = est_latents.detach() - true_latents_without_noise.detach()
              rmse_error = torch.sqrt(torch.mean(error_per_estimated_latent.pow(2), dim=(0, 1)))
              if self.rmse_estimator_error is None:
                self.rmse_estimator_error = rmse_error
              else:
                self.rmse_estimator_error + rmse_error
              if n == n_updates - 1:
                self.last_rmse_estimator_error = self.rmse_estimator_error / n

              # save metrics regarding the confidence
              if n == 0:
                tracked_metrics["average_confidence_var"] = torch.mean(latent_unc).item()
                tracked_metrics["stddev_confidence_var"] = torch.std(latent_unc).item()
              else:
                tracked_metrics["average_confidence_var"] += torch.mean(latent_unc).item()
                tracked_metrics["stddev_confidence_var"] += torch.std(latent_unc).item()

          # # contrastive loss to encourage encoder to well form the latent space
          # if loss_alphas["contrastive"] > 1e-5:
          #   loss["contrastive"] = self.physics_informed_infonce(
          #     true_latents=true_latents_without_noise,
          #     priv_ground_truth=priv_ground_truth,
          #     temperature=0.07,
          #     num_negatives=256,
          #     similarity_threshold=0.90,
          #     batch_size=1024,
          #   )

      # loss terms for a denoiser network
      if self.add_denoise_step:
        # denoised_obs shape = (B, n_encodings, num_timesteps_hist, n_q * 2)
        loss["denoise"] = self.get_loss(extras["denoised_obs"].reshape(B, -1, self.denoiser_inds.shape[0]),
                                        extras["true_obs"].reshape(B, -1, self.denoiser_inds.shape[0]))
        # temporal smoothing loss, so that the output trajectory is smooth
        temp_changes = extras["denoised_obs"][:, :, 1:] - extras["denoised_obs"][:, :, :-1]
        loss["denoise_smoothing"] = torch.mean(torch.pow(temp_changes, 2)) # manual MSE error

      # optional additional loss terms if full observation enabled
      if self.use_full_observation:
        
        # # acceleration loss
        # if "pred_qacc" in extras:
        #   true_qacc = trajectory_batch[:, future, self.i_qacc]
        #   loss["acceleration"] = self.get_tapered_loss(extras["pred_qacc"], true_qacc)

        # mass matrix and torque vector losses
        if "pred_M" in extras:
          if self.dense_mass_matrix:
            true_M = full_ground_truth[..., self.i_qM]
          else:
            true_M_lower = full_ground_truth[..., self.i_qM]
            bm, tm, vm = true_M_lower.shape
            true_M = self.unpack_qM(true_M_lower.reshape(bm * tm, vm))
            true_M = true_M.reshape(bm, tm, -1)
          true_C = full_ground_truth[..., self.i_qfrc_in]
          true_J = full_ground_truth[..., self.i_qfrc_out]

          loss["pred_M"] = self.get_tapered_loss(extras["pred_M"], true_M)
          loss["pred_C"] = self.get_tapered_loss(extras["pred_C"], true_C)
          loss["pred_J"] = self.get_tapered_loss(extras["pred_J"], true_J)

      # now aggregate the overall loss
      total_loss = torch.zeros(())

      for i, key in enumerate(loss):

        # calculate the overall loss
        if loss_alphas[key] > 1e-6:  # Only include if alpha > 0
          total_loss = total_loss + (loss[key] * loss_alphas[key])

        # log all losses for debugging (even if they did not contribute)
        if key in loss_metrics:
          loss_metrics[key] += loss[key].item()
        else:
          loss_metrics[key] = loss[key].item()
      loss_metrics["total_loss"] += total_loss.item()

      # perform backpropagation
      self.optimiser.zero_grad()
      total_loss.backward()
      nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
      self.optimiser.step()

      # # old - seperate optimisers for predictor and estimator
      # self.predictor_optimiser.zero_grad()
      # self.estimator_optimiser.zero_grad()

      # total_loss.backward()
            
      # # 3. Clip gradients for each group SEPARATELY
      # nn.utils.clip_grad_norm_([p for p in self.predictor.parameters()] + 
      #                           [p for p in self.encoder.parameters()], 
      #                           self.max_grad_norm)
      
      # # This second clip call will NOT cause an error. It's a second pass over
      # # a different parameter group's gradients, which are all present after
      # # the single backward() pass.
      # if update_estimator:
      #     nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)

      # # 4. Perform a SEPARATE step for each optimizer
      # self.predictor_optimiser.step()
      
      # # The estimator only gets its step if its losses were part of the total_loss
      # if update_estimator:
      #   self.estimator_optimiser.step()

      # compute some additional statistics
      with torch.no_grad():
        pred_error = true_traj - predicted_traj
        if self.use_normalisation:
          pred_error = revert_norm(pred_error, debug=False,
                                   mean=self.norm["pred"]["mean"],
                                   std=self.norm["pred"]["std"])
        joint_angle_error = torch.abs(pred_error).mean().item()
        actuated_pos_error = torch.abs(pred_error[:, :, self.qpos_actuated]).mean().item()
        actuated_vel_error = torch.abs(pred_error[:, :, self.qvel_actuated]).mean().item()
        payload_pos_error = torch.abs(pred_error[:, :, self.qpos_payload]).mean().item()
        payload_vel_error = torch.abs(pred_error[:, :, self.qvel_payload]).mean().item()

      if n == 0:
        tracked_metrics["joint_angle_error"] = joint_angle_error
        tracked_metrics["actuated_pos_error"] = actuated_pos_error
        tracked_metrics["actuated_vel_error"] = actuated_vel_error
        tracked_metrics["payload_pos_error"] = payload_pos_error
        tracked_metrics["payload_vel_error"] = payload_vel_error
      else:
        tracked_metrics["joint_angle_error"] += joint_angle_error
        tracked_metrics["actuated_pos_error"] += actuated_pos_error
        tracked_metrics["actuated_vel_error"] += actuated_vel_error
        tracked_metrics["payload_pos_error"] += payload_pos_error
        tracked_metrics["payload_vel_error"] += payload_vel_error

    # count number of complete updates, for estimator every X updates
    self.updates_done += 1

    for key in loss_metrics:
      loss_metrics[key] /= n_updates # get average over updates

    # add additional subfields with more information
    for key in tracked_metrics:
      tracked_metrics[key] /= n_updates # get average over updates
    loss_metrics["metrics"] = tracked_metrics
    loss_metrics["alphas"] = loss_alphas

    return loss_metrics
  
  def estimate_latents(self, observation, handle_normalisation=True,
                       use_ground_truth_encoder=False, return_confidence=False):
    """
    Return the latents based on a state history. If use_ground_truth_encoder=True, 
    then observation should instead be the ground truth privileged information.
    """

    # normalise all incoming inputs
    if handle_normalisation and self.use_normalisation:

      debug_norm_estimator = True # set to true to add debug info about normalisation
      if not hasattr(self, "debug_norm_estimator"):
        self.debug_norm_estimator = None
      else:
        debug_norm_estimator = False # only do it on the first call

      # if we are using the true encoder (requires ground truth priv info as observation)
      if use_ground_truth_encoder:
        if debug_norm_estimator:
          pylogger.info(f"About to normalise the ground_truth_priv_info in "
                        f"estimate_latents (use_ground_truth_encoder=True), "
                        f"observation shape = {observation.shape}, "
                        f"norm_mean_priv_info.shape = {self.norm['priv_info']['mean'].shape}, "
                        f"norm_std_priv_info.shape = {self.norm['priv_info']['std'].shape}")
        observation = apply_norm(observation, debug=debug_norm_estimator,
                                 mean=self.norm['priv_info']['mean'],
                                 std=self.norm['priv_info']['std'])
        
      # if we are using the estimator (standard case)
      else:
        if debug_norm_estimator: 
          pylogger.info(f"About to normalise the observation in estimate_latents, "
                        f"observation shape = {observation.shape}, "
                        f"norm_mean_eval.shape = {self.norm['eval']['mean'].shape}, "
                        f"norm_std_eval.shape = {self.norm['eval']['std'].shape}")
        observation = apply_norm(observation, debug=debug_norm_estimator,
                                 mean=self.norm['eval']['mean'], 
                                 std=self.norm['eval']['std'])
        
    # special case: encode a provided privileged information vector
    if use_ground_truth_encoder:
      latents = self.encoder(observation)

    # experimental: latents come out of predictor network
    elif self.predictor_outputs_priv_info:
      # pylogger.warning(f"estimate_latents() is calling rollout_prediciton, because "
      #                  f"predictor_outputs_priv_info=True")
      # how many predictions do we make in one call to rollout_prediction
      if hasattr(self.predictor, "prediction_timesteps"):
        prediction_timesteps = self.predictor.prediction_timesteps
      else:
        prediction_timesteps = 1
      # only single step prediction, fine for next action to be zero
      B, T, N = observation.shape
      action_dummy = torch.zeros((B, prediction_timesteps - 1, 3), device=observation.device)
      # need to run rollout to get latents
      preds, extras = self.rollout_prediction(observation, 
                                              n_predictions=prediction_timesteps,
                                              eval=True,
                                              handle_normalisation=False,
                                              future_actions=action_dummy,
                                              return_extras=True)
      latents = extras["predictor_est_latents"] # shape (B, 1, N)
      latents = torch.squeeze(latents, dim=1) # shape (B, N)

    # if we have a normal observation, which is a history of states
    else:

      # mask out velocity values if required
      if self.zero_qvel_input:
        if not hasattr(self, "debug_print_zero_qvel_est_latents"):
          pylogger.info(f"\n\n{'-'*10} QVEL ZERO-ED estimate_latents {'-'*10}\n")
          self.debug_print_zero_qvel_est_latents = True
        observation[:, :, self.i_qvel] = 0.0

      # optional addition of denoising step
      if self.add_denoise_step:
        if not hasattr(self, "debug_print_denoiser_latents"):
          pylogger.info(f"\n\n{'-'*10} DENOISED in estimate_latents {'-'*10}\n")
          self.debug_print_denoiser_latents = True
        observation[:, :, self.denoiser_inds] = self.denoiser(observation[:, :, self.denoiser_inds])
    
      # finally, pass the observation into the estimator
      latents = self.estimator(observation)

      # default: return only latents, and not uncertainty values
      if self.use_estimator_confidence:
        if not return_confidence:
          latents = latents[..., self.latent_encoding_inds]

    return latents

  def decode_latents(self, latents, handle_normalisation=True):
    """
    Return the decodings of the latents.

    When it comes to confidence, the decoder never recieves a confidence
    value. The encoder does not output a confidence, and the decoder takes
    the same shape as the encoder. So confidence values should always be
    removed in the current implementation.
    """

    # handle automatically stripping out confidence values (handle case 'latents'
    # is passed in from either encoder or estimator)
    if self.use_estimator_confidence:
      if latents.shape[-1] != self.num_latent_encoding + self.conf_inds.shape[0]:
        pylogger.debug(f"latents with input shape = {latents.shape} "
                       f"doesn't have the extra expected confidence value")
      else:
        latents = latents[..., self.latent_encoding_inds] # strip out the confidence value

    # decode the latents to the privileged information
    decodings = self.encoder.decoder(latents)

    # de-normalise all outgoing outputs
    if handle_normalisation and self.use_normalisation:

      debug_norm_decoder = True # set to true to add debug info about normalisation
      if not hasattr(self, "debug_norm_decoder"):
        self.debug_norm_decoder = None
      else:
        debug_norm_decoder = False # only do it on the first call

      if debug_norm_decoder:
        pylogger.info(f"About to denormalise the latents in decode_latents, "
                      f"decodings shape = {decodings.shape}, "
                      f"norm_mean_priv_info.shape = {self.norm['priv_info']['mean'].shape}, "
                      f"norm_std_priv_info.shape = {self.norm['priv_info']['std'].shape}")
      decodings = revert_norm(decodings, debug=debug_norm_decoder,
                              mean=self.norm['priv_info']['mean'],
                              std=self.norm['priv_info']['std'])
      
    return decodings

  def run_denoiser(self, observation, handle_normalisation=True):
    """
    Pass an observation through the denoiser
    """

    # normalise all incoming inputs
    if handle_normalisation and self.use_normalisation:
      debug_norm = False # set to true to add debug info about normalisation
      if not hasattr(self, "debug_norm"):
        self.debug_norm = None
      else:
        debug_norm = False # only do it on the first call
      
      if debug_norm: print(f"About to normalise the 'observation' in rollout_prediction"
                           f", shape = {observation.shape}, eval = {eval}, "
                           f"norm_mean_eval.shape = {self.norm['eval']['mean'].shape}, "
                           f"norm_std_eval.shape = {self.norm['eval']['std'].shape}")
        
      observation = apply_norm(observation, debug=debug_norm,
                               mean=self.norm['eval']['mean'], 
                               std=self.norm['eval']['std'])
      
    # mask out velocity values after normalisation (so they stay at zero)
    if self.zero_qvel_input:
      if not hasattr(self, "debug_print_zero_qvel_run_denoiser"):
        pylogger.info(f"\n\n{'-'*10} QVEL ZERO-ED run_denoiser {'-'*10}\n")
        self.debug_print_zero_qvel_run_denoiser = True
      observation[:, :, self.i_qvel] = 0.0
    
    # run the denoiser
    observation[:, :, self.denoiser_inds] = self.denoiser(observation[:, :, self.denoiser_inds])

    # now revert normalisation
    if handle_normalisation and self.use_normalisation:
      observation = revert_norm(observation, debug=debug_norm,
                                mean=self.norm['eval']['mean'], 
                                std=self.norm['eval']['std'])
      
    return observation

  # --- utility functions --- #

  def get_loss(self, x, y):
    """
    Get the loss using a mean reduction (loss_fn has reduction='None' to be compatible
    with tapering)
    """
    elemwise_loss = self.loss_fn(x, y)
    loss = elemwise_loss.mean()
    return loss

  def get_taper(self, x):
    """
    Get a taper in the time dimension of a target value, expanding into the batch
    and feature dimensions. Add an extra step so if we taper from 1.0->0.0, we don't
    include 0.0 in the taper (this would exclude data from the loss).
    """
    B, T, N = x.shape
    taper_bare = torch.linspace(start=1.0, end=self.taper_horizon_loss_to, 
                                steps=T + 1, device=self.device)[:-1]
    taper = einops.repeat(taper_bare, "t -> b t n", b=B, n=N)
    return taper
  
  def get_tapered_loss(self, x, y, loss_fn=None):
    """
    Apply a taper in the time dimension to a target, and then apply the loss
    function and return the loss.
    """

    taper = self.get_taper(x)
    if loss_fn is not None:
      elemwise_loss = loss_fn(x, y)
    else:
      elemwise_loss = self.loss_fn(x, y)
    loss = (elemwise_loss * taper).mean()
    return loss

  def prepare_noise(self, debug_noise=False):
    """
    Prepare noise information for training, called every update()
    """

    # prepare dt
    dt_qpos = 0.05 * (self.norm["qvel"]["std"] / self.norm["qpos"]["std"])
    self.noiser_qpos_qvel.dt = dt_qpos

    # noise standard deviations across dimensions, create these only once
    if not hasattr(self, "use_noise_std_qpos"):
          
      pylogger.info(f"Preparing noise std vectors for qpos, qvel, etc...")

      if self.noise_std_SI_qpos is not None:
        self.use_noise_std_qpos = torch.tensor(self.noise_std_SI_qpos, device=self.device)
        if self.use_normalisation:
          self.use_noise_std_qpos = apply_norm(
            self.use_noise_std_qpos,
            mean=torch.tensor(0.0, device=self.device),
            std=self.norm["qpos"]["std"].to(self.device),
            debug=debug_noise,
          )
      else:
        self.use_noise_std_qpos = torch.scalar_tensor(self.default_noise_std,
                                                      device=self.device)

    if not hasattr(self, "use_noise_std_qvel"):
      if self.noise_std_SI_qvel is not None:
        self.use_noise_std_qvel = torch.tensor(self.noise_std_SI_qvel, device=self.device)
        if self.use_normalisation:
          self.use_noise_std_qvel = apply_norm(
            self.use_noise_std_qvel,
            mean=torch.tensor(0.0, device=self.device),
            std=self.norm["qvel"]["std"].to(self.device),
            debug=debug_noise,
          )
      else:
        self.use_noise_std_qvel = torch.scalar_tensor(self.default_noise_std,
                                                      device=self.device)

    if not hasattr(self, "use_noise_std_action"):
      if self.noise_std_SI_action is not None:
        self.use_noise_std_action = torch.tensor(self.noise_std_SI_action, device=self.device)
        if self.use_normalisation:
          self.use_noise_std_action = apply_norm(
            self.use_noise_std_action,
            mean=torch.tensor(0.0, device=self.device),
            std=self.norm["action"]["std"].to(self.device),
            debug=debug_noise,
          )
      else:
        self.use_noise_std_action = torch.scalar_tensor(self.default_noise_std,
                                                        device=self.device)

    if not hasattr(self, "use_noise_std_priv_info"):
      if self.noise_std_SI_priv_info is not None:
        self.use_noise_std_priv_info = torch.tensor(self.noise_std_SI_priv_info, device=self.device)
        if self.use_normalisation:
          self.use_noise_std_priv_info = apply_norm(
            self.use_noise_std_priv_info,
            mean=torch.tensor(0.0, device=self.device),
            std=self.norm["priv_info"]["std"].to(self.device),
            debug=debug_noise,
          )
      else:
        self.use_noise_std_priv_info = torch.scalar_tensor(self.default_noise_std,
                                                           device=self.device)

    # increment the noisers, in case the std_scaling is on an annealing schedule
    self.noiser_qpos_qvel.increment(i=self.updates_done)
    self.noiser_action.increment(i=self.updates_done)
    self.noiser_priv_info.increment(i=self.updates_done)

    # reset this flag, to debug noise every update()
    self.log_noise = debug_noise

  def add_noise(self, observation, action_noise=False):
    """
    Add noise onto the observation
    """
    if not hasattr(self, "log_noise"):
      self.log_noise = True

    if self.log_noise:
      pylogger.info(
        f"Adding noise with (units: normalised):\n"
        f" -> qpos_std (scaling={self.noiser_qpos_qvel.current_std_scaling:.3f}"
        f", range={self.noiser_qpos_qvel.individualised_noise_std_range})"
        f" = {self.use_noise_std_qpos}\n"
        f" -> qvel_std (scaling={self.noiser_qpos_qvel.current_std_scaling:.3f}"
        f", range={self.noiser_qpos_qvel.individualised_noise_std_range})"
        f" = {self.use_noise_std_qvel}\n"
        f" -> action_std (scaling={self.noiser_action.current_std_scaling:.3f}"
        f", range={self.noiser_action.individualised_noise_std_range})"
        f" = {self.use_noise_std_action}\n"
        f" -> priv_info_std (scaling={self.noiser_priv_info.current_std_scaling:.3f}"
        f", range={self.noiser_priv_info.individualised_noise_std_range})"
        f" = {self.use_noise_std_priv_info}\n"
      )
      self.log_noise = False

    if not action_noise:

      observation[:, :, self.i_qs] = self.noiser_qpos_qvel.add_correlated_noise(
        observation[:, :, self.i_qs],
        std_base=self.use_noise_std_qpos,
        std_correlated=self.use_noise_std_qvel,
      )

      observation[:, :, self.i_action] = self.noiser_action.add_noise(
        observation[:, :, self.i_action],
        std_dev=self.use_noise_std_action,
      )

      # add only clean white noise to the privileged information (no OU component)
      if self.use_sampled_uncertainty:
        # pylogger.debug("Sampled uncertainty! Applying 5% noise on priv info")
        observation[:, :, self.i_priv_obs] = self.noiser_priv_info.add_white_noise(
          observation[:, :, self.i_priv_obs],
          std_dev=self.use_noise_std_priv_info * 0.05, # only apply small noise on raw values
        )
        # get the randomised stddevs
        _ = self.noiser_priv_info.add_white_noise(
          observation[:, :, self.i_priv_obs],
          std_dev=self.use_noise_std_priv_info, # large stdev stored internally
        )
      else:
        # normal, non-experimental case
        observation[:, :, self.i_priv_obs] = self.noiser_priv_info.add_white_noise(
          observation[:, :, self.i_priv_obs],
          std_dev=self.use_noise_std_priv_info,
        )

    else:

      observation[:, :, :] = self.noiser_action.add_noise(
        observation[:, :, :],
        std_dev=self.use_noise_std_action,
      )

    return observation

  def get_loss_alphas(self):
    """
    Get the loss coefficients
    """

    # bind the number of updates into the function call
    global infer_annealing
    infer = partial(infer_annealing, i=self.updates_done)

    # core losses for state prediction
    loss_alphas = {
      "prediction" : infer(self.alphas["prediction"]),
      "actuated" : infer(self.alphas["actuated"]),
      "payload" : infer(self.alphas["payload"]),
      "prediction_confidence" : infer(self.alphas["prediction_confidence"]),
      "integrate_pos" : infer(self.alphas["integrate_pos"]),
      "diffusion" : infer(self.alphas["diffusion"]),
    }

    # loss terms for ROA style privileged encoder
    if self.add_privileged_info:

      loss_alphas = loss_alphas | {
        "estimator" : infer(self.alphas["estimator"]),
        "estimator_norm" : infer(self.alphas["estimator_norm"]),
        "estimator_confidence" : infer(self.alphas["estimator_confidence"]),
        "latent_norm" : infer(self.alphas["latent_norm"]),
        "latent_norm_mse" : infer(self.alphas["latent_norm_mse"]),
        "latent_anisotrophy" : infer(self.alphas["latent_anisotrophy"]),
        "regularisation" : infer(self.alphas["regularisation"]),
        "consistency" : infer(self.alphas["consistency"]),
        "smoothing" : infer(self.alphas["smoothing"]),
        "smoothing_confidence" : infer(self.alphas["smoothing_confidence"]),
        "decoder" : infer(self.alphas["decoder"]),
        "contrastive" : infer(self.alphas["contrastive"]),
      }

    # loss terms for denoising observations
    if self.add_denoise_step:
      loss_alphas = loss_alphas | {
        "denoise" : infer(self.alphas["denoise"]),
        "denoise_smoothing" : infer(self.alphas["denoise_smoothing"]),
      }

    # additional loss terms if we include extra info in the observation
    if self.use_full_observation:

      loss_alphas = loss_alphas | {
        "acceleration" : infer(self.alphas["acceleration"]),
        "pred_M" : infer(self.alphas["pred_M"]),
        "pred_C" : infer(self.alphas["pred_C"]),
        "pred_J" : infer(self.alphas["pred_J"]),
      }

    return loss_alphas

  def load_normalisation(self, norm_object):
    """
    Load normalisations given a normalisation object, from utils.training
    """

    pylogger.info(f"Loading normalisation object into LatentPredictor")

    self.normalisations = norm_object

    if self.normalisations is None:
      if self.use_normalisation:
        pylogger.warning(f"Normalisation object is none, but use normalisation=True"
                         f", since no normalisation object, normalisation is disabled.")
      self.use_normalisation = False
      self.norm = {}
      return
    
    # create dictionary to save normalisations
    self.norm = norm_object.create_dict([
      (self.return_obs_index_list(self.i_obs_full), "full"),
      (self.return_obs_index_list(self.i_qs), "pred"),
      (self.return_obs_index_list(self.i_obs_eval), "eval"),
      (self.return_obs_index_list(self.i_action), "action"),
      (self.return_obs_index_list(self.i_priv_obs), "priv_info"),
    ])

    # now load normalisation on sub-component
    self.predictor.load_normalisation(norm_object)

    # check for clashes
    for key in self.norm:
      if key in self.predictor.norm and key not in ["means", "stds", "names"]:
        pylogger.warning(f"WARNING: key={key} present in both Agent.normalisations "
                         f"and predictor.normalisations. This key will now be "
                         f"OVERWRITTEN by the agent version.")

    # save all of the normalisation information into one dict
    self.norm = self.predictor.norm | self.norm

    # for debugging: print the exact normalisation dict
    # print(f"Normalisation dict: {self.norm}")

  def return_obs_index_list(self, index_list):
    """
    Call the index naming function in a loop, to return a list
    """
    names = []
    for i in index_list:
      names.append(self.return_obs_index_name(i))
    return names

  def return_obs_index_name(self, i):
    """
    Given an index i, check what the name of that element is in the observation
    for the model. For example, if the observation has structure:
    obs = [
      qpos[0],
      qpos[1],
      qvel[0],
      qvel[1],
    ]
    then this class has members:
      self.i_qpos = torch.arange(0, 2)
      self.i_qvel = torch.arange(2, 4)

    and this function should return:
      -> i=0, return 'qpos[0]'
      -> i=1, return 'qpos[1]'
      -> i=2, return 'qvel[0]'
      -> i=3, return 'qvel[1]'
      -> i=4, return 'no match, i=4'
    """
  
    if not isinstance(i, torch.Tensor):
      i = torch.tensor(i)
    
    ind_vars = [name for name in dir(self) if name.startswith("i_") and not callable(getattr(self, name))]
    
    for var in ind_vars:
      # these are fields which overlap others, ignore them
      if var in ["i_qs", 
                 "i_obs",
                 "i_obs_eval",
                 "i_obs_train",
                 "i_obs_full",
                 "i_obs_post_encoder",
                 "i_priv_enc"]: continue
      indexes = getattr(self, var)
      check = torch.isin(i, indexes)
      if check.any():
        # Find which element in indexes matches i
        match_positions = torch.where(indexes == i)[0]
        if len(match_positions) > 0:
          elem = match_positions[0].item()  # take first match
          return f"{var[2:]}[{elem}]"
    
    return f"no match, i={i}"

  def get_save_state(self):
    """
    Return the save state of the module
    """
    return {
        "encoder" : self.encoder.state_dict(),
        "estimator" : self.estimator.state_dict(),
        "predictor" : self.predictor.state_dict(),
        "denoiser" : self.denoiser.state_dict() if self.add_denoise_step else None,
        # "prediction_optimiser_state_dict" : self.predictor_optimiser.state_dict(),
        # "estimator_optimiser_state_dict" : self.estimator_optimiser.state_dict(),
        "optimiser_state_dict" : self.optimiser.state_dict(),
        "normalisations" : self.normalisations,
        "norm_dict" : self.norm, # union of ours and predictor normalisation dict
        "updates_done" : self.updates_done,
    }

  def load_save_state(self, loaded_dict):
    """
    Load the module from a loaded state
    """
    self.encoder.load_state_dict(loaded_dict["encoder"])
    self.estimator.load_state_dict(loaded_dict["estimator"])
    self.predictor.load_state_dict(loaded_dict["predictor"])

    if self.add_denoise_step:
      self.denoiser.load_state_dict(loaded_dict["denoiser"])

    self.updates_done = loaded_dict["updates_done"]

    if "optimiser_state_dict" in loaded_dict:
      # pass
      self.optimiser.load_state_dict(loaded_dict["optimiser_state_dict"])
    else:
      pass
      # self.predictor_optimiser.load_state_dict(loaded_dict["prediction_optimiser_state_dict"])
      # self.estimator_optimiser.load_state_dict(loaded_dict["estimator_optimiser_state_dict"])

    if "norm_dict" in loaded_dict:
      self.norm = loaded_dict["norm_dict"]
      # backwards compatibility
      if "actions" in self.norm:
        self.norm["action"] = self.norm["actions"]
      self.predictor.norm = self.norm

    # keep for backwards compatibility (redundant from 2025-07-04)
    elif "normalisations" in loaded_dict:
      self.normalisations = loaded_dict["normalisations"]
      self.load_normalisation(self.normalisations)

  def to(self, device):
    """
    Send the agent to the specified device
    """
    self.device = device  # save the new device we are on
    super().to(device)    # call the underlying class method
    return self
  
# --- agent to wrap model --- #

class Agent_Latent_Estimator:

  name = "Agent_Latent_Estimator"

  def __init__(self, 
              
               # environment settings
               num_envs,
               num_env_steps_per_update,

               # hydra instantiation settings
               network=None,
               operator=None,
               eval_mode=False,

               # observation settings
               use_full_observation=False,
               from_sim_priv_info_indexes=None,

               # options for loading datasets
               load_from_dataset=False,
               dataset_path=None,
               dataloader_num_threads=2,

               # options for saving datasets
               save_to_dataset=False,
               save_data_folder=None,
               save_data_mode="numpy",

               # options for generating new data
               noise_qpos_std=0.0,
               noise_qvel_std=0.0,
               noise_priv_info_std=0.0,

               seed=None,
               device="cuda",
               **kwargs):
    """
    Agent wrapper class implement a motion predictor, where an underlying ROA
    agent performs the actions.
    """

    if kwargs:
      pylogger.info("Agent_Latent_Estimator.__init__ got unexpected arguments, which will be ignored: " + \
                    str([key for key in kwargs.keys()]))

    self.num_envs = num_envs
    self.num_env_steps_per_update = num_env_steps_per_update
    self.rngseed = seed # does nothing currently
    self.mixed_agent = operator
    self.device = device
    self.use_full_observation = use_full_observation
    self.from_sim_priv_info_indexes = from_sim_priv_info_indexes

    # the main model should be hydra instantiated
    self.model = network
    self.model.train() # enter training mode

    # expose key parts of model
    self.num_joint_angles = self.model.n_q
    self.num_hist_timesteps_to_use = self.model.n_t
    self.add_privileged_info = self.model.add_privileged_info

    # note this is insufficient for full reproducibility in pytorch
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)

    # handle variables for datasets
    self.save_to_dataset = save_to_dataset
    self.load_from_dataset = load_from_dataset
    self.noise_qpos_std = noise_qpos_std
    self.noise_qvel_std = noise_qvel_std
    self.noise_priv_info_std = noise_priv_info_std

    # migrate relative paths to absolute
    if dataset_path is not None and not os.path.isabs(dataset_path):
      dataset_path = f"{path_to_root}/{dataset_path}"
    if not os.path.isabs(save_data_folder):
      save_data_folder = f"{path_to_root}/{save_data_folder}/"

    # save main paths
    self.dataset_path = dataset_path
    self.save_data_folder = (f"{save_data_folder}/"
                             f"{datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}")

    self.dataset_cfg = None
    
    # prepare data loading, unless we are in eval mode
    if eval_mode:

      pylogger.info(f"Loading in eval_mode, no data buffer or loader created")

    else:
      # load configs of the dataset
      if self.load_from_dataset:
        self.dataset_cfg = OmegaConf.load(f"{self.dataset_path}/config.yaml")

      # are we loading normalisations to apply across the dataset
      if self.model.use_normalisation:
        self.dataset_normalisations = load_normalisation(
          dataset_path, 
          filename="dataset_statistics",
          mean_field="Mean",
          std_field="Stddev",
          use_torch=True, 
          device=self.device
        )
        self.model.load_normalisation(self.dataset_normalisations)
      else:
        self.dataset_normalisations = None

      # create a storage buffer to save states
      self.storage = DataBuffer(
          num_envs=num_envs,
          traj_len=num_env_steps_per_update,
          data_shape=(self.model.num_obs_full,),
          device=self.device,
          save_data=self.save_to_dataset,
          dataset_path=self.save_data_folder,
          save_style=save_data_mode,
      )

      # create a dataloader
      if self.load_from_dataset:
        self.dataloader = DataLoadWrapper(
          dataset_path=self.dataset_path,
          datafile_name="data_numpy",
          data_indexes=np.arange(self.model.num_obs_full),
          data_mode="torch",
          num_workers=dataloader_num_threads,
          device=self.device,
          shuffle=True,
        )

    # create a mixed action agent
    if operator is not None:
      self.mixed_agent = operator
      self.use_mixed_actions = True
    else:
      raise RuntimeError(f"Operator is none")
      self.mixed_agent = MixedOperatorAgent(num_envs=num_envs,
                                            num_actions=self.model.num_actions)
      
    pylogger.info(f"Agent_Latent_Estimator has finished initialisation")

  def get_save_state(self, infos=None):
    """
    Get the state of the agent which can be saved.
    """
    to_save = {
        "name" : self.name,
        'model': self.model.get_save_state(),
        "seed" : self.rngseed,
        "device" : self.device,
        'infos': infos,
    }

    return to_save

  def load_save_state(self, loaded_dict, device=None):
    """
    Load the state of the agent
    """
    self.model.load_save_state(loaded_dict["model"])
    self.rngseed = loaded_dict['seed']
    self.device = loaded_dict['device']

    self.to(device if device is not None else self.device)
    pylogger.info(f"Moved to device={self.device}")

    return loaded_dict['infos']

  def to(self, device):
    """
    Move the agent to a specified device
    """
    if hasattr(self, "model"):
      self.model.to(device)
    if hasattr(self, "storage"):
      self.storage.to(device)
    self.device = device

  def load_data(self, i_episode=None):
    """
    Load data into the storage buffer from a dataset
    """

    # new, use pytorch dataloader
    data = self.dataloader.load() # loads the next file

    B, T, N = self.storage.data.shape

    # assign the data into our storage buffer (maintains compatibility with using physics)
    self.storage.data = data
    self.storage.num_envs = data.shape[0]
    self.storage.traj_len = data.shape[1]
    self.storage.data_shape = data.shape
    self.storage.traj_step = self.storage.traj_len

    return

  def update(self):
    """
    Update the model using data from the storage buffer
    """

    if not self.storage.is_full():
      raise RuntimeError(f"Agent_Latent_Estimator.update() error: "
                         f"self.storage buffer is not full when update() called.")
    
    # save the generated trajectories
    if self.save_to_dataset:
      inds = torch.arange(self.storage.data.shape[-1])
      names = []
      for i in inds:
        names.append(self.model.return_obs_index_name(i))
      self.storage.save(names=names)
      self.storage.empty()
      return {} # skip any updates, save data only

    # update the model
    update_dict = self.model.update(self.storage.get())
    self.storage.empty()

    # re-randomise the mixed action agent
    if self.use_mixed_actions:
      self.mixed_agent.reset()

    return update_dict

  def get_random_action_mix(self, obs_torch):
    """
    Return actions by mixing together actions from several different sources
    randomly, eg policy + sinousoid + random noise
    """
    obs_jax = torch_to_jax(obs_torch)
    actions_jax = self.mixed_agent.get_action(obs_jax)
    actions_torch = jax_to_torch(actions_jax)
    return actions_torch

  def make_observation(self, obs, actions, info):
    """
    Construct and add an observation to the buffer
    """
    def add_noise(vals, std):
      if std > 1e-5:
        vals = vals + torch.normal(mean=torch.zeros_like(vals), std=std)
      return vals
    
    def add_to_obs(obs, new_vals, noise_std=None):
      if noise_std is not None:
        new_vals = add_noise(new_vals, noise_std)
      if obs is None:
        new_obs = new_vals
        n1 = 0
      else:
        new_obs = torch.concat([
          obs,
          new_vals,
        ], dim=-1)
        n1 = obs.shape[-1]
      n2 = new_obs.shape[-1]
      inds = torch.arange(n1, n2)
      return new_obs, inds

    # extract the qpos and qvel information from the observation
    batch_num = obs.shape[0]
    q = self.num_joint_angles
    qpos = obs[:, : q]
    qvel = obs[:, q : q * 2]

    # create the observation
    new_obs, self.i_qpos = add_to_obs(None, qpos, noise_std=self.noise_qpos_std)
    new_obs, self.i_qvel = add_to_obs(new_obs, qvel, noise_std=self.noise_qvel_std)

    if self.use_full_observation or self.model.add_next_action:
      new_obs, self.i_action = add_to_obs(new_obs, actions)

    if self.use_full_observation or self.model.add_privileged_info:
      priv_info = jax_to_torch(info["privileged_info_normalised"])
      # check if only certain indexes of privileged info should be used in observation
      if not hasattr(self, "priv_info_indexes"):
        if self.from_sim_priv_info_indexes is not None:
          self.priv_info_indexes = np.array(self.from_sim_priv_info_indexes, dtype=int)
          pylogger.info(f"Special case: use selected privileged info in observation. "
                        f"Priv info.shape={priv_info.shape}, indexes selected are "
                        f"{self.priv_info_indexes}")
        else:
          self.priv_info_indexes = np.arange(priv_info.shape[1])
          pylogger.info(f"Default case: use all privileged info in observation")
      # index the privileged info and add it to the observation
      new_obs, self.i_priv_obs = add_to_obs(new_obs, priv_info[:, self.priv_info_indexes],
                                            noise_std=self.noise_priv_info_std)
      
    if self.use_full_observation:
      if (len(info['qM'].shape[1:]) == 2 and not self.model.dense_mass_matrix
          or len(info['qM'].shape[1:]) == 1 and self.model.dense_mass_matrix):
        raise RuntimeError(f"Agent_Latent_Estimator.get_action() error: "
                           f"mass matrix from mujoco, 'qM', has shape = {info['qm'].shape}"
                           f", but dense_mass_matrix={self.model.dense_mass_matrix}. Either"
                           f" change this, or use/remove <option jacobian='sparse'> in XML.")
      
      # get the detailed data out
      if self.model.dense_mass_matrix:
        mass_matrix = jax_to_torch(info["qM"]).reshape(batch_num, -1)
      else:
        mass_matrix = jax_to_torch(info["qM"])
      qacc = jax_to_torch(info["qacc"])
      qfrc_actuator = jax_to_torch(info["qfrc_actuator"])
      qfrc_bias = jax_to_torch(info["qfrc_bias"])
      qfrc_applied = jax_to_torch(info["qfrc_applied"])
      qfrc_passive = jax_to_torch(info["qfrc_passive"])
      qfrc_constraint = jax_to_torch(info["qfrc_constraint"])
      qfrc_all_ext = qfrc_actuator + qfrc_applied + qfrc_passive + qfrc_constraint

      # NEW: add it all to the observation
      if self.model.new_indexes:
        new_obs, self.i_qacc = add_to_obs(new_obs, qacc)
        new_obs, self.i_qfrc_in = add_to_obs(new_obs, qfrc_bias)
        new_obs, self.i_qfrc_out = add_to_obs(new_obs, qfrc_all_ext)
        new_obs, self.i_qfrc_actuator = add_to_obs(new_obs, qfrc_actuator)
        new_obs, self.i_qfrc_applied = add_to_obs(new_obs, qfrc_applied)
        new_obs, self.i_qfrc_passive = add_to_obs(new_obs, qfrc_passive)
        new_obs, self.i_qfrc_constraint = add_to_obs(new_obs, qfrc_constraint)
        new_obs, self.i_qM = add_to_obs(new_obs, mass_matrix)

      else:
        # OLD version
        new_obs, self.i_qfrc_out = add_to_obs(new_obs, qfrc_all_ext)
        new_obs, self.i_qfrc_in = add_to_obs(new_obs, qfrc_bias)
        new_obs, self.i_qfrc_constraint = add_to_obs(new_obs, qfrc_constraint)
        new_obs, self.i_qM = add_to_obs(new_obs, mass_matrix)
        new_obs, self.i_qacc = add_to_obs(new_obs, qacc)

    # debugging
    if not hasattr(self, "debug_obs"):
      inds = torch.arange(self.model.num_obs_full)
      names = self.return_obs_index_list(inds)
      print(f"Observation created, shape = {new_obs.shape}")
      for i in range(len(inds)):
        print(f"ind={inds[i]}, name={names[i]}")
      self.debug_obs = None
    
    return new_obs

  def get_action(self, obs, info=None):
    """
    Get the next actions for the environment, and update the replay buffer with
    all of the information from this step.

    The incoming obs is in the default format, [qpos, qvel]. The 'info' dict contains
    much additional information.
    """

    # get the actions based on this observation
    with torch.no_grad():
      actions = self.get_random_action_mix(obs)

    # create and add an observation
    new_obs = self.make_observation(obs, actions, info)
    self.storage.add(new_obs)

    return actions

  def predict_rollout(self, obs, n_steps=1, actions=None, priv_info_override=None,
                      latent_override=None, return_extras=False):
    """
    Return a prediction for the specified number of steps, with shape:
      Input: obs.shape = (B, N)
      Output: predictions.shape = (B, T, N)

      where B = batch size, T = n_steps, and N = num obs features.

      For predictors which use the future actions, these should be passed
      in, with shape (B, T - 1, num_actions). Actions should start at two
      steps into the future (since one step into the future should already
      be contained in obs).

    If use_torch=False, expects JAX (or numpy) tensor input, returns JAX tensor.
    If use_torch=True, expects a torch tensor input, returns torch tensor
    """

    # confirm obs is a numpy array, and convert to torch
    if isinstance(obs, (np.ndarray)):
      obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
    else:
      raise RuntimeError(f"Agent_Latent_Estimator.predict_rollout() error: "
                         f"obs not a numpy array, it is ({type(obs)}). Only numpy is supported")
    
    # handle type conversions
    if actions is not None and isinstance(actions, (np.ndarray)):
      actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
    if priv_info_override is not None and isinstance(priv_info_override, (np.ndarray)):
      priv_info_override = torch.tensor(priv_info_override, device=self.device, dtype=torch.float32)
    if latent_override is not None and isinstance(latent_override, (np.ndarray)):
      latent_override = torch.tensor(latent_override, device=self.device, dtype=torch.float32)

    # ensure no actions are passed if actions are not added to observations
    if not self.model.add_next_action:
      actions = None

    # put into eval mode
    self.model.eval()

    with torch.no_grad():

      prediction_output = self.model.rollout_prediction(obs, n_steps, eval=True,
                                                        future_actions=actions,
                                                        priv_info_override=priv_info_override,
                                                        latent_override=latent_override,
                                                        return_extras=return_extras)
      
      if return_extras:
        predictions, extras = prediction_output
      else:
        predictions = prediction_output
      
    if isinstance(predictions, (tuple, list)):
      raise RuntimeError(f"Agent_Latent_Estimator.predict() error: "
                         f"predictions is a tuple/list")

    # print(f"predictions = {predictions}")
    # print(f"predictions.shape = {predictions.shape}", flush=True)

    predictions = predictions.numpy(force=True)

    if return_extras:
      for key in extras:
        if extras[key] is not None:
          extras[key] = extras[key].numpy(force=True)
      return predictions, extras
    else:
      return predictions
  
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
  
    t = self.model.num_hist_timesteps_to_use

    if prediction_index < t:
      raise RuntimeError(f"Agent_Latent_Estimator.get_observation_from_trajectory() error: "
                         f"prediction_index={prediction_index} is less than "
                         f"num_hist_timsteps_to_use={t}.")
    
    prev_timesteps = slice(prediction_index - t, prediction_index)
    next_actions = slice(prediction_index - t + 1, prediction_index + 1)
    qpos_hist = trajectory["joint_angles"][:, prev_timesteps, :]
    qvel_hist = trajectory["joint_velocities"][:, prev_timesteps, :]
    act_hist = trajectory["last_action"][:, next_actions, :] # actions from next step

    if self.model.add_next_action:
      observation = np.concatenate([qpos_hist, qvel_hist, act_hist], axis=-1)
    else:
      observation = np.concatenate([qpos_hist, qvel_hist], axis=-1)

    return observation
  
  def get_prediction_from_trajectory(self, trajectory, index=None, horizon=None,
                                     priv_info_override=None, latent_override=None,
                                     fix_estimated_latents=False, return_extras=False,
                                     **unused_args):
    """
    Return a prediction rollout at the specified index (index is the first
    prediction, it is NOT seen by the model), for the given horizon.
    """

    if unused_args:
      pylogger.debug("Agent_Latent_Estimator.get_prediction_from_trajectory "
                     f"got unexpected arguments, which will be ignored: " + \
                     str([key for key in unused_args.keys()]))

    if index is None:
      index = self.model.num_hist_timesteps_to_use
    if horizon is None:
      horizon = 1

    obs_hist = self.get_observation_from_trajectory(trajectory, prediction_index=index)

    if self.model.add_next_action:
      # the action for horizon=1 is in the above observation, so we need horizon-1 more actions
      future_actions = trajectory["last_action"][:, index + 1 : index + horizon, :]
      if future_actions.shape[1] < horizon - 1:
        raise RuntimeError(f"Agent_Latent_Estimator.get_prediction_from trajectory() error: "
                           f"index={index} does not give horizon={horizon} future actions.")
    else:
      future_actions = None
    
    if fix_estimated_latents:
      if latent_override is not None or priv_info_override is not None:
        raise RuntimeError(f"Agent_Latent_Estimator.get_prediction_from_trajectory() error: "
                           f"fix_estimated_latents=True, but latent_override != None"
                           f", these settings cannot be both used together")
      latent_override = self.estimate_latents_from_trajectory(trajectory, index=index,
                                                              average_num=1)

    rollout_output = self.predict_rollout(obs_hist, n_steps=horizon, actions=future_actions,
                                          priv_info_override=priv_info_override, 
                                          latent_override=latent_override,
                                          return_extras=return_extras)

    return rollout_output
  
  def get_latent_estimator(self, device=None):
    """
    If the agent uses a latent encoder for privileged information, return that
    network, otherwise return None
    """

    if self.model.add_privileged_info:
      self.model.eval()
      if device is not None:
        self.model.to(device)
      return self.model.estimate_latents
    
    else:
      pylogger.warning(f"Agent_Latent_Estimator.get_latent_estimator() warning: "
                       f"self.add_privileged_info = False, returning None")
      return None
    
  def get_latent_decoder(self, device=None):
    """
    If the agent has a decoder for latent privileged information, return that
    network, otherwise return None
    """

    if self.model.add_privileged_info:
      self.model.eval()
      if device is not None:
        self.model.to(device)
      return self.model.decode_latents
    
    else:
      pylogger.warning(f"Agent_Latent_Estimator.get_latent_decoder() warning: "
                       f"self.add_privileged_info = False, returning None")
      return None

  def estimate_latents_from_trajectory(self, trajectory, index=None, average_num=1):
    """
    Return a set of latent estimations from a given trajectory, with prediction
    made up to (but not including) the given index.
    """

    if not self.model.add_privileged_info:
      pylogger.warning(f"model.add_privileged_info=False, cannot estimate latents. Returning None")
      return None

    if index is None:
      index = self.model.num_hist_timesteps_to_use
    if average_num is None:
      average_num = trajectory["joint_angles"].shape[1] - index

    self.model.eval()

    with torch.no_grad():
      for i in range(average_num):

        # get the observation at this step
        obs_hist = self.get_observation_from_trajectory(trajectory, prediction_index=index + i)

        # query the latent estimator
        state_obs = torch.tensor(obs_hist, device=self.device, dtype=torch.float32)
        estimated_latents = self.model.estimate_latents(state_obs, return_confidence=True)

        # cumulatively add to get a final average
        if i == 0:
          average_latents = torch.zeros_like(estimated_latents)
        average_latents = torch.add(average_latents, estimated_latents)

    if i > 1:
      average_latents = torch.divide(average_latents, i)

    return estimated_latents

  def denoise_trajectory(self, trajectory):
    """
    Return a denoised trajectory, if the model has a denoiser, otherwise, simply
    returns the provided trajectory
    """

    if not self.model.add_denoise_step:
      pylogger.warning(f"{self.name}.denoise_trajectory() warning: "
                       f"model does not have a denoiser, returning given trajectory")
      return trajectory

    pylogger.info(f"Preparing to denoise a trajectory")

    t = self.num_hist_timesteps_to_use
    
    # convert the input trajectory into a standard observation
    qpos_hist = trajectory["joint_angles"]
    qvel_hist = trajectory["joint_velocities"]

    if self.model.add_next_action:
      # shift actions backwards (last state gets wrong action which never resolves)
      act_hist = np.roll(trajectory["last_action"], shift=-1, axis=1)
      observation = np.concatenate([qpos_hist, qvel_hist, act_hist], axis=-1)
    else:
      observation = np.concatenate([qpos_hist, qvel_hist], axis=-1)

    observation = torch.tensor(observation, device=self.device)

    # now loop through and apply the denoising
    num_total = observation.shape[1]
    num_loops = int(np.ceil(num_total / t))

    denoised_obs = torch.zeros_like(observation)

    for i in range(num_loops):

      # write into the denoised traj, and handle overshoot at the end
      if i == num_loops - 1:
        inds = torch.arange(num_total - t, num_total)
      else:
        inds = torch.arange(i * t, (i + 1) * t)

      # run the model to denoise
      with torch.no_grad():
        denoised_chunk = self.model.run_denoiser(observation[:, inds])

      # add in the denoised position and velocity (not action)
      denoised_obs[:, inds] = denoised_chunk

    # mean_qpos = torch.mean(denoised_obs[:, :, self.model.i_qpos])
    # std_qpos = torch.std(denoised_obs[:, :, self.model.i_qpos])
    # mean_qvel = torch.mean(denoised_obs[:, :, self.model.i_qvel])
    # std_qvel = torch.std(denoised_obs[:, :, self.model.i_qvel])

    # print(f"QPOS denoising, mean={mean_qpos.item():.3f}, std={std_qpos.item():.3f}")
    # print(f"QVEL denoising, mean={mean_qvel.item():.3f}, std={std_qvel.item():.3f}")

    # convert the final output back into a trajectory
    denoised_traj = data_to_trajectory(
      denoised_obs.numpy(force=True),
      i_qpos=self.model.i_qpos,
      i_qvel=self.model.i_qvel,
      i_action=self.model.i_action,
      use_priv_info=False
    )

    return denoised_traj

  # --- utility functions --- #

  def return_obs_index_list(self, index_list):
    """
    Call the index naming function in a loop, to return a list
    """
    names = []
    for i in index_list:
      names.append(self.return_obs_index_name(i))
    return names

  def return_obs_index_name(self, i, key="i_", exclusions=None):
    """
    Given an index i, check what the name of that element is in the observation
    for the model. For example, if the observation has structure:
    obs = [
      qpos[0],
      qpos[1],
      qvel[0],
      qvel[1],
    ]
    then this class has members:
      self.i_qpos = torch.arange(0, 2)
      self.i_qvel = torch.arange(2, 4)

    and this function should return:
      -> i=0, return 'qpos[0]'
      -> i=1, return 'qpos[1]'
      -> i=2, return 'qvel[0]'
      -> i=3, return 'qvel[1]'
      -> i=4, return 'unmatched[4]'
    """
  
    if not isinstance(i, torch.Tensor):
      i = torch.tensor(i)
    
    ind_vars = [name for name in dir(self) if name.startswith(key) and not callable(getattr(self, name))]

    if exclusions is None:
      exclusions = []
    
    for var in ind_vars:
      # these are fields which overlap others, ignore them
      if var in exclusions: continue
      indexes = getattr(self, var)
      check = torch.isin(i, indexes)
      if check.any():
        # Find which element in indexes matches i
        match_positions = torch.where(indexes == i)[0]
        if len(match_positions) > 0:
          elem = match_positions[0].item()  # take first match
          return f"{var[2:]}[{elem}]"
    
    return f"unmatched[{i}]"
  