import torch
import numpy as np
import einops
import matplotlib.pyplot as plt
import jax.numpy as jnp
from functools import partial
import time
import io
import logging; logging.basicConfig(
    level=logging.INFO); pylogging = logging.getLogger(__name__)

from aware.utils.runs import load_agent

def empty_latents(observation, handle_normalisation=True, latent_dim=7, **kwargs):
  """
  For agents with no ability to predict latents, simply return zeros
  """
  # B, T, N = observation.shape
  return torch.zeros((observation.shape[0], latent_dim))
  # torch.manual_seed(np.random.randint(low=0, high=100_000))
  # return torch.rand((observation.shape[0], latent_dim))

class Agent_Latent_Discriminator:

  name = "Agent_Latent_Discriminator"

  def __init__(self, timestamp=None, id=None, device="cuda", agent=None,
               agent_cfg=None, agent_file_starts=None, dummy_latents=False,
               **kwargs):
    """
    This class loads a latent encoder ROA model from a specified timestamp, and
    then uses it to discriminate whether changes have been made to the system
    parameters.
    """

    if agent is None:
      self.agent, self.agent_cfg = load_agent(timestamp, agent_file_starts=agent_file_starts, 
                                              id=id, device=device, return_configs=True,
                                              **kwargs)
    else:
      self.agent = agent
      self.agent_cfg = agent_cfg

    self.device = device
    self.last_index = None
    self.last_view = None

    self.set_latent_networks(dummy_latents=dummy_latents)

    # get vital information that the agent should expose
    try:
      self.t = self.agent.num_hist_timesteps_to_use
    except AttributeError as e:
      raise AttributeError(f"Agent_Latent_Discriminator.__init__() error: "
                           f"self.agent with name={self.agent.name} does not have the "
                           f"required field 'num_hist_timesteps_to_use'.\nError msg: {e}")
    
  def set_latent_networks(self, dummy_latents=False):
    """
    Assign the functionality for the latent encoder etc
    """

    if not dummy_latents:
      if hasattr(self.agent, "get_latent_estimator"): 
        # extract the encoder etc (note the decoder does the whole encoder+decode pass)
        self.encoder = self.agent.get_latent_estimator(device=self.device)
        # check that the encoder is not disabled (add_privileged_info=False)
        if self.encoder is not None:
          if hasattr(self.agent, 'get_latent_decoder'):
            self.decoder = self.agent.get_latent_decoder(device=self.device)
          else:
            pylogging.warning(f"Loaded agent with name {self.agent.name} has no decoder attribute, using dummy latents decoder")
            self.decoder = empty_latents
          self.gt_priv_encoder = partial(self.encoder, use_ground_truth_encoder=True) # bind kwarg to function call
          dummy_latents = False # latents are enabled
        else:
          pylogging.warning(f"Loaded agent with name {self.agent.name} has encoder=None"
                            f", dummy_latents enabled")
          dummy_latents = True
      else:
        pylogging.warning(f"Loaded agent with name {self.agent.name} has no attribute "
                          f"'get_latent_estimator', dummy_latents enabled")
        dummy_latents = True

    if dummy_latents:
      pylogging.info(f"Latent discriminator set into dummy mode, all latents will be zero")
      self.encoder = empty_latents
      self.decoder = empty_latents
      self.gt_priv_encoder = empty_latents
      self.dummy_latents = True
    else:
      self.dummy_latents = False

  def get_latent(self, observation, decode=False, gt_latents=None,
                 return_confidence=False):
    """
    Return the latent vector as a numpy array, given an input observation as a
    numpy array. The observation should have structure:
    
    roa_obs = jnp.concat([
      jnp.zeros((batch_num, any_amount*)),
      jnp.concat([
        qpos_history_for_n_steps,
        qvel_history_for_n_steps,
        action_history_for_n_steps (perhaps zeros for this)
      ], axis=-1).reshape(batch_num, -1)
    ])

    *The initial zeros can be any length, as the observation is indexed by the
    encoder like this obs = obs[-num_prop * num_timesteps :].
    """
    # temporary fix for clarity
    privileged_info = gt_latents

    with torch.no_grad():
      
      # special case when decoding using the rssm
      if decode and self.agent.name == 'rssm_roa':
        latent = self.decoder(observation=observation, priv_vector=privileged_info)

      else:
        if privileged_info is None:
          # standard case, estimate the latents from state history
          state_obs = torch.tensor(observation, device=self.device, dtype=torch.float32)
          latent = self.encoder(state_obs, return_confidence=return_confidence)
        else:
          # special case, return the true encodings using ground truth privileged info
          priv_info_obs = torch.tensor(privileged_info, device=self.device, dtype=torch.float32)
          latent = self.gt_priv_encoder(priv_info_obs)
        
        if decode:
          latent = self.decoder(latent)
    
    if torch.is_tensor(latent):
      return latent.numpy(force=True)
    else: # JAX
      return np.array(latent)

  def get_latent_from_trajectory(self, trajectory, index=None, decode=False,
                                 gt_latents=None, return_confidence=False):
    """
    Return the latent vector for a specific trajectory, up to (but NOT
    including) the provided index.
    """
    if "precomputed_latents" in trajectory:
      latent = trajectory["precomputed_latents"][:, index, :]  # (B, M)
      if return_confidence:
        latent = np.concatenate([latent, np.zeros_like(latent)], axis=-1)
      return latent

    if index is None:
      index = self.t

    x_prev = index
    x_oldest = index - self.t
    batch_num = trajectory["joint_angles"].shape[0]

    # assemble the observation from the trajectory
    qpos_hist = trajectory["joint_angles"][:, x_oldest : x_prev, :]
    qvel_hist = trajectory["joint_velocities"][:, x_oldest : x_prev, :]

    act_hist = trajectory["last_action"][:, x_oldest + 1 : x_prev + 1, :]
    obs = np.concatenate([qpos_hist, qvel_hist, act_hist], axis=-1)

    latent = self.get_latent(obs, decode=decode, gt_latents=gt_latents,
                             return_confidence=return_confidence)

    return latent
  
  def get_latent_over_time(self, trajectory, index=None, num_steps=None, decode=False,
                           gt_latents=None, return_confidence=False):
    """
    Get the latent over time from a trajectory
    """

    if index is None:
      index = self.t

    B, T, _ = trajectory["joint_angles"].shape
    if num_steps is None:
      num_steps = T - index

    if num_steps < 1:
      raise RuntimeError(f"{self.name}.get_latent_over_time() error: "
                         f"num_steps = {num_steps} (< 0), T={T}, index={index}")

    if "precomputed_latents" in trajectory:
      full_latent = trajectory["precomputed_latents"][:, index : index + num_steps, :]  # (B, T, M)
      if return_confidence:
        full_latent = np.concatenate([full_latent, np.zeros_like(full_latent)], axis=-1)
      return full_latent

    for i in range(num_steps):

      # query the model for the latents
      latent = self.get_latent_from_trajectory(trajectory, index=index + i,
                                               decode=decode, gt_latents=gt_latents,
                                               return_confidence=return_confidence)
      if i == 0:
        B, N = latent.shape
        full_latent = np.zeros((B, num_steps, N))

      full_latent[:, i, :] = latent

    return full_latent

  def calculate_measurement(self, latents, reference_dict=None, method="norm",
                            reference_stddev=None):
    """
    Calculate the measurement or measurements on vector of latents, shape (B, T, N)
    """

    B, T, N = latents.shape
    eps = 1e-6
    
    if method in ["norm", "area", "cosine"]:

      if reference_dict is not None:
 
        if reference_dict["method"] == "mean_all":
          reference = einops.repeat(reference_dict["mean"], "n -> b t n", b=B, t=T)
        elif reference_dict["method"] == "mean":
          reference = einops.repeat(reference_dict["mean"], "b n -> b t n", t=T)
        else:
          raise RuntimeError(f"Agent_Latent_Discriminator.calculate_measurement() error: "
                             f"method={method}, but reference_method="
                             f"{reference_dict['method']}, these are incompatible. "
                             f"'{method}' requires 'mean' or 'mean_all'")
        # apply the reference
        difference = latents - reference
      else:
        difference = latents
      
      # calculate the measurement
      if method == "norm":
        measurements = np.linalg.norm(difference, axis=2)
      elif method == "area":
        measurements = np.trapezoid(difference, dx=1.0, axis=2)
      elif method == "cosine":
        if reference_dict is not None:
          norm_reference = reference / (np.linalg.norm(reference, axis=2, keepdims=True) + eps)
        else:
          norm_reference = np.zeros_like(latents[:, 0])
          norm_reference[:, 0] = 1.0
          norm_reference = np.expand_dims(norm_reference, axis=1)
        norm_latents = latents / (np.linalg.norm(latents, axis=-1, keepdims=True) + eps)
        cos_sim = np.sum(norm_reference * norm_latents, axis=2)  # shape: (B, T)
        measurements = 1.0 - cos_sim

    elif method == "passthrough":
      measurements = latents # do nothing

    elif method == "mahalanobis":
      
      if reference_dict is None:
        raise RuntimeError(f"Agent_Latent_Discrimiantor.calculate_measurements() error: "
                           f"reference_dict == None, but is required for method={method}")

      # Normalize test points
      Xc = ((latents - reference_dict["means"][:, None, :])
            / reference_dict["mad"][:, None, :])
      
      # Mahalanobis: sqrt( (x)ᵀ Σ⁻¹ (x) )
      left = np.einsum('bti,bij->btj', Xc, reference_dict["covs_inv"]) 
      measurements = np.sqrt(np.einsum('bti,bti->bt', left, Xc))

    elif method == "mahalanobis-variance":
      
      if reference_dict is None:
        raise RuntimeError(f"Agent_Latent_Discrimiantor.calculate_measurements() error: "
                           f"reference_dict == None, but is required for method={method}")

      # latents: (B, T, N), means: (N,), mad: (N,)
      # We use [None, None, :] to broadcast the global reference across batch and time
      Xc = ((latents - reference_dict["means"][None, None, :])
            / reference_dict["mad"][None, None, :])

      if reference_stddev is not None:
        # sigma_ref shape: (1, N, N) -> we squeeze it or use [0] to get (N, N)
        sigma_ref = reference_dict["cov"][0] 
        
        # Calculate variance and NORMALIZE by the reference MAD^2
        # This keeps the test uncertainty in the same 'unit space' as the reference cov
        raw_var = np.mean((reference_stddev ** 2), axis=1) # (B, N)
        normed_var = raw_var / (reference_dict["mad"] ** 2) # (B, N)
        
        # Add diagonal variance to the reference covariance matrix
        # batch_covs will be (B, N, N)
        batch_covs = sigma_ref + np.array([np.diag(v) for v in normed_var])
        
        # Using linalg.inv on a batch is efficient
        cov_inv = np.linalg.inv(batch_covs) # (B, N, N)
      else:
        cov_inv = reference_dict["covs_inv"] # (1, N, N)
      
      # Mahalanobis: sqrt( (x)ᵀ Σ⁻¹ (x) )
      # left: (B, T, N) @ (B, N, N) -> (B, T, N)
      left = np.einsum('bti,bij->btj', Xc, cov_inv) 
      measurements = np.sqrt(np.einsum('bti,bti->bt', left, Xc))

    elif method == "bhattacharyya":
      
      if reference_dict is None or reference_stddev is None:
          raise RuntimeError("Bhattacharyya requires reference_dict and variance predictions.")

      # 1. Prepare Reference Statistics (p)
      # sigma_p: (N, N)
      sigma_p = reference_dict["cov"][0]
      mu_p = reference_dict["means"]
      mad = reference_dict["mad"]
      logdet_p = reference_dict["logdet_ref"]

      # 2. Prepare Test Statistics (q)
      # Normalize inputs by Reference MAD to stay in the same "unit space"
      # mu_q: (B, T, N) - We keep time dimension for the mean-shift term
      mu_q_norm = (latents - mu_p[None, None, :]) / mad[None, None, :]
      
      # sigma_q: (B, N) - Averaged over time T to simplify matrix inversion
      # We square stddev to get variance, then normalize by MAD^2
      raw_var = np.mean(reference_stddev**2, axis=1) 
      sigma_q_diag = raw_var / (mad**2) # (B, N)

      # 3. Calculate Average Covariance (Sigma_avg)
      # Sigma_avg = (Sigma_p + Sigma_q) / 2
      # We use broadcasting to add diagonal sigma_q to full matrix sigma_p
      # (1, N, N) + (B, N, N) -> (B, N, N)
      eye_N = np.eye(sigma_p.shape[0])
      sigma_q_mat = sigma_q_diag[:, None, :] * eye_N[None, :, :]
      sigma_avg = 0.5 * (sigma_p[None, :, :] + sigma_q_mat)

      # 4. Term 1: Mahalanobis-like Distance (Mean Shift)
      # 1/8 * (mu_p - mu_q)^T * Sigma_avg^-1 * (mu_p - mu_q)
      sigma_avg_inv = np.linalg.inv(sigma_avg) # (B, N, N)
      
      # Einsum: (B, T, N) @ (B, N, N) -> (B, T, N)
      left_term = np.einsum('bti,bij->btj', -mu_q_norm, sigma_avg_inv) 
      # Dot product: (B, T, N) . (B, T, N) -> (B, T)
      term1 = (1/8) * np.einsum('bti,bti->bt', left_term, -mu_q_norm)

      # 5. Term 2: Covariance Divergence (The "Shape" check)
      # 0.5 * ln( |Sigma_avg| / sqrt(|Sigma_p| * |Sigma_q|) )
      # Decomposed: 0.5 * (ln|Sigma_avg| - 0.5*ln|Sigma_p| - 0.5*ln|Sigma_q|)
      
      # Calculate log determinants
      _, logdet_avg = np.linalg.slogdet(sigma_avg) # (B,)
      
      # logdet of diagonal matrix sigma_q is sum of log of diagonal elements
      # sigma_q_diag elements are variances (sigma^2)
      logdet_q = np.sum(np.log(sigma_q_diag + 1e-9), axis=1) # (B,)

      term2 = 0.5 * (logdet_avg - 0.5 * logdet_p - 0.5 * logdet_q)

      # 6. Combine
      # term2 is (B,), so we broadcast to (B, T)
      measurements = np.sqrt(np.maximum(0, term1 + term2[:, None]))

    elif method == "mahalanobis-scaling":
    
      if reference_dict is None:
        raise RuntimeError(f"Agent_Latent_Discriminantor.calculate_measurements() error: "
                           f"reference_dict == None, but is required for method={method}")

      # 1. Flatten the reference stats to (N,) and (N, N)
      # This ensures they broadcast correctly against (B, T, N)
      ref_means = reference_dict["means"][0] 
      ref_mad = reference_dict["mad"][0]
      cov_inv = reference_dict["covs_inv"][0]

      # 2. Calculate raw deviation
      # Result shape: (B, T, N)
      deviation = latents - ref_means

      if reference_stddev is not None:
        # 3. Uncertainty-Aware Normalization
        # Result shape: (B, T, N)
        total_sigma = np.sqrt(ref_mad ** 2 + reference_stddev ** 2)
        Xc = deviation / total_sigma
      else:
        # Fallback to standard MAD normalization
        Xc = deviation / ref_mad
      
      # Mahalanobis: sqrt( (x)ᵀ Σ⁻¹ (x) )
      left = np.einsum('bti,ij->btj', Xc, cov_inv) 
      measurements = np.sqrt(np.einsum('bti,bti->bt', left, Xc))

    elif method == "maha-stddev":
      
      if reference_dict is None:
        raise RuntimeError(f"Agent_Latent_Discrimiantor.calculate_measurements() error: "
                           f"reference_dict == None, but is required for method={method}")

      # Normalize test points
      Xc = ((latents - reference_dict["means"][:, None, :])
            / reference_dict["stddev"][:, None, :])
      
      # Mahalanobis: sqrt( (x)ᵀ Σ⁻¹ (x) )
      left = np.einsum('bti,bij->btj', Xc, reference_dict["covs_inv"]) 
      measurements = np.sqrt(np.einsum('bti,bti->bt', left, Xc))

    elif method == "maha-robust":
      
      if reference_dict is None:
        raise RuntimeError(f"Agent_Latent_Discrimiantor.calculate_measurements() error: "
                           f"reference_dict == None, but is required for method={method}")

      # Normalize test points
      Xc = latents - reference_dict["means"][:, None, :]
      
      # Mahalanobis: sqrt( (x)ᵀ Σ⁻¹ (x) )
      left = np.einsum('bti,bij->btj', Xc, reference_dict["covs_inv"]) 
      measurements = np.sqrt(np.einsum('bti,bti->bt', left, Xc))

    elif method == "var":

      if reference_dict is None:
        raise RuntimeError(f"Agent_Latent_Discrimiantor.calculate_measurements() error: "
                           f"reference_dict == None, but is required for method={method}")

      # --- 1. Load VAR model and residual stats ---
      p = reference_dict["p"]
      A1 = reference_dict["A1"]
      intercept = reference_dict["intercept"]
      res_means = reference_dict["res_means"]
      # res_mad = reference_dict["res_mad"]
      res_cov_inv = reference_dict["res_cov_inv"]
      use_diff = reference_dict["use_diff"]

      if use_diff:
        latents = latents[:, 1:, :] - latents[:, :-1, :]

      B, T, N = latents.shape
      measurements = np.zeros((B, T))

      # We can only calculate residuals from timestep p onwards
      if T <= p:
        raise RuntimeError(f"method={method} failed, T={T} which is <= p={p}")
        # Not enough data to calculate any residuals
        return measurements

      # --- 2. Vectorized Residual Calculation ---

      # Get all "previous" states: e_{t-1}
      # Shape: (B, T-p, N)
      prev_all = latents[:, p-1:-1, :]

      # Get all "current" states: e_t
      # Shape: (B, T-p, N)
      curr_all = latents[:, p:, :]

      # Predict all future states: \hat{e}_t = e_{t-1} @ A1 + c
      # We use einsum for the batched matrix multiply
      # (B, T-p, N) @ (N, N) -> (B, T-p, N)
      prediction_all = np.einsum('bti,ij->btj', prev_all, A1) + intercept

      # Calculate all residuals: r_t = e_t - \hat{e}_t
      # Shape: (B, T-p, N)
      residual_all = curr_all - prediction_all

      # --- 3. Calculate Mahalanobis Distance of Residuals ---

      # Normalize residuals: r_c = (r_t - \mu_res) / \sigma_mad
      # Shape: (B, T-p, N)
      normed_res_all = (residual_all - res_means)# / res_mad

      # Mahalanobis: sqrt( (r_c)ᵀ Σ⁻¹ (r_c) )
      # Left part: (r_c)ᵀ Σ⁻¹
      # (B, T-p, N) @ (1, N, N) -> (B, T-p, N)
      left = np.einsum('bti,bij->btj', normed_res_all, res_cov_inv)

      # Full quadratic form: ((r_c)ᵀ Σ⁻¹) (r_c)
      # (B, T-p, N) * (B, T-p, N) -> (B, T-p)
      dist_sq = np.einsum('bti,bti->bt', left, normed_res_all)

      # Store the sqrt of the distance
      # We put this into the `measurements` array, offset by p
      measurements[:, p:] = np.sqrt(dist_sq)

      # `measurements` now has 0s for t=0..p-1, and the anomaly score for t=p..T

    elif method == "l2":
      # compute norm of the error vector
      if reference_dict is None:
        raise RuntimeError(f"Agent_Latent_Discrimiantor.calculate_measurements() error: "
                           f"reference_dict == None, but is required for method=l2")
      pylogging.info(f"Computing l2 measurement")
      means = reference_dict["means"]
      measurements = np.linalg.norm(latents - means, axis=-1)
      
    elif method == "l1":
      if reference_dict is None:
        raise RuntimeError(f"Agent_Latent_Discrimiantor.calculate_measurements() error: "
                           f"reference_dict == None, but is required for method=l1")
      pylogging.info(f"Computing l1 measurement")
      means = reference_dict["means"]
      measurements = np.linalg.norm(latents - means, axis=-1, ord=1)

    elif method == "l2-standardised":
      if reference_dict is None:
        raise RuntimeError(f"Agent_Latent_Discrimiantor.calculate_measurements() error: "
                           f"reference_dict == None, but is required for method=l2-standardised")
      pylogging.info(f"Computing l2-standardised measurement")
      means = reference_dict["means"]
      mad = reference_dict["mad"]
      measurements = np.linalg.norm((latents - means) / mad, axis=-1)
  
    elif method == "l1-standardised":
      if reference_dict is None:
        raise RuntimeError(f"Agent_Latent_Discrimiantor.calculate_measurements() error: "
                           f"reference_dict == None, but is required for method=l1-standardised")
      pylogging.info(f"Computing l1-standardised measurement")
      means = reference_dict["means"]
      mad = reference_dict["mad"]
      measurements = np.linalg.norm((latents - means) / mad, axis=-1, ord=1)

    else:
      raise RuntimeError(f"Agent_Latent_Discrimiantor.calculate_measurement() error: "
                         f"method={method} not recognised")

    
    return measurements

  def get_measurement_from_trajectory(self, trajectory, index=None, num_measurements=None,
                                      method="norm", window_size=1, relative_change=False,
                                      use_reference=False, reference_obj=None,
                                      reference_type="auto", decode=False, 
                                      reference_auto_num_samples=100, use_jax=False,
                                      computing_reference=False, 
                                      return_confidence=False,
                                      return_decodings=False,
                                      return_extra=False):
    """
    Get latent measurements over time for a trajectory.

    Index is the timestep in the trajectory for which the first measurement is given.
    For example, at index = 20, that is the measurement which arises from the previous
    x steps of data, inclusive (x being the amount required for one measurement).

    num_measurements is the number of measurements returned by this function, starting
    at index. The window size is the width of rolling average over measurements.
    relative_change=True means this function returns the change between adjacent
    measurements rather than the raw measurements themselves.
    decode=True means latents are passed first through the decoder.

    Either:
      - provide a reference trajectory, from which the average latents will be calculated
        and used as the reference values.
      - provide directly the reference latents.
    """

    # the minimum history required to make inference with the model
    min_hist = self.t + (window_size - 1)

    if index is None:
      index = min_hist + int(relative_change) # need extra point for relative change
    elif index < 0:
      index = trajectory["joint_angles"].shape[1] + index
    elif index < min_hist:
      pylogging.warning(f"Agent_Latent_Discriminator.get_measurement_from_trajectory() error: "
                        f"cannot have index={index} less than {min_hist} "
                        f"(self.t={self.t} + window_size={window_size})")
      raise RuntimeError(f"Agent_Latent_Discriminator.get_measurement_from_trajectory() error: "
                         f"cannot have index={index} less than {min_hist} "
                         f"(self.t={self.t} + window_size={window_size})")
      if return_extra:
        return None, None
      else:
        return None
      
    # the maximum number of points in the trajectory we can get measurements for
    index -= int(relative_change) # go backwards one extra if we need change @ index
    max_measure = trajectory["joint_angles"].shape[1] - index
  
    if num_measurements is None:
      num_measurements = max_measure
    else:
      num_measurements += int(relative_change) # need 2 measurements for 1 relative change
    if max_measure < 1:
      raise RuntimeError(f"Agent_Latent_Discriminator.get_measurement_over_time() error: "
                         f"trajectory length = {trajectory['joint_angles'].shape[1]} is too"
                         f" short for the given index of {index}.")
    if num_measurements > max_measure:
      raise RuntimeError(f"Agent_Latent_Discriminator.get_measurement_over_time() error: "
                         f"num_measurements={num_measurements}, with T={trajectory['joint_angles'].shape[1]} "
                         f"timesteps in the trajectory, maximum number of measurements is "
                         f"{max_measure} when index={index} (minimum history={min_hist}, "
                         f"self.t={self.t} + window_size={window_size})")
      
    # we need to roll forwards from a previous point in the trajectory
    # we look into the past to make sure we have ground truth data for all points to compare against
    past_index = index - (window_size - 1)
    num_steps = num_measurements + (window_size - 1)

    # careful! only write these when computing actual measurement, not reference
    if not computing_reference:
      # save to simplify aligning (this is the number of measurements 'dropped' from front of traj)
      self.last_index = index
      self.last_view = window_size + self.t

    # convert the trajectory to latents
    latents = self.get_latent_over_time(trajectory, index=past_index, num_steps=num_steps,
                                        decode=decode, return_confidence=return_confidence)
    
    # seperate latents from confidence
    if (return_confidence and hasattr(self.agent, "model")
        and hasattr(self.agent.model, "use_estimator_confidence")
        and self.agent.model.use_estimator_confidence):
      latent_uncertainty = latents[:, :, self.agent.model.conf_inds] # save confidence value (shape B, T, N)
      latents = latents[:, :, self.agent.model.latent_encoding_inds] # strip out confidence value from latent vector
    # rssm case -> it does not have attribute model (it is a model itself)
    elif (return_confidence and 
          hasattr(self.agent, "use_estimator_confidence") and
          self.agent.use_estimator_confidence):
          latent_uncertainty = latents[:, :, self.agent.privileged_latent_dim:] # strip out confidence value from latent vector
          latents = latents[:, :, :self.agent.privileged_latent_dim] # save confidence value (shape B, T, N)
    else: latent_uncertainty = None
    
    # now apply the window size, to apply smoothing
    smooth_latents = self.timestep_rolling_average(latents, window_size=window_size)

    (B, T, N) = smooth_latents.shape

    if use_reference:
      if reference_type == "computed":
        if not isinstance(reference_obj, dict) and "method" in reference_obj:
          raise RuntimeError(f"Agent_Latent_Discriminator.get_measurement_from_trajectory() error: "
                             f"reference_type=computed, but reference obj is not a reference dict")
        # we should have been passed the reference latent information dictionary
        reference_dict = reference_obj
        pylogging.info(f"Reference dict has been passed directly")
      else:
        if reference_type == "auto":
          args = { "reference_latents" : latents[:, :reference_auto_num_samples]}
          pylogging.info(f"Reference being auto generated from first "
                         f"{reference_auto_num_samples} samples")
        else:
          if reference_type == "trajectory":
            if not isinstance(reference_obj, dict) and "joint_angles" in reference_obj:
              raise RuntimeError(f"Agent_Latent_Discriminator.get_measurement_from_trajectory() error: "
                                f"reference_type=trajectory, but reference obj is not a trajectory")
            args = { "trajectory" : reference_obj }
            pylogging.info(f"Reference trajectory has been passed")
          elif reference_type == "latents":
            if not isinstance(reference_obj, np.ndarray) and len(reference_obj.shape) != 3:
              raise RuntimeError(f"Agent_Latent_Discriminator.get_measurement_from_trajectory() error: "
                                f"reference_type=latents, but reference obj is not valid")
            args = { "reference_latents" : reference_obj }
            pylogging.info(f"Reference latents have been passed directly")

        if method in ["norm", "area", "cosine"]:
          ref_method = "mean"
        elif method == "mahalanobis":
          ref_method = "mahalanobis"
        else:
          raise RuntimeError(f"Agent_Latent_Discriminator.get_measurement_from_trajectory() error: "
                            f"method={method} not recognised when 'use_reference'==True")

        # compute the reference latent information dictionary
        reference_dict = self.get_reference_values(**args, method=ref_method, decode=decode,
                                                   use_jax=use_jax)
    else:
      # no reference is being used
      reference_dict = None

    # now, calculate the discimrination measurement
    measurements = self.calculate_measurement(smooth_latents, reference_dict, method)

    # are we taking the relative change between measurements
    if relative_change:
      measurements = np.abs(self.calculate_measurement_change(measurements))

    decoded_latents = None
    if return_decodings:
      decoded_latents = self.decoder(latents)

    extra_info = {
      "latents" : latents,
      "decoded_latents": decoded_latents,
      "latent_uncertainty" : latent_uncertainty,
      "index" : self.last_index,
      "view" : self.last_view,
    }

    if return_extra:
      return measurements, extra_info
    else:
      return measurements

  def get_reference_values(self, trajectory=None, decode=False, method="mean_all",
                           reference_latents=None, use_jax=False, reference_stddev=None):
    """
    Get a reference value for each latent (i.e. its average) over a trajectory
    which is deemed to have good standard data
    """

    # generate reference latents if given a trajectory
    if trajectory is not None and reference_latents is None:
      reference_latents = self.get_measurement_from_trajectory(
        trajectory=trajectory,
        computing_reference=True, # preserve last_index/view
        index=None,
        num_measurements=None,
        use_reference=False,
        use_jax=use_jax,
        window_size=1,
        method="passthrough",
        decode=decode,
      )

    # otherwise, use latents already passed in
    elif reference_latents is not None:
      if trajectory is not None:
        pylogging.warning(f"trajectory != None even though reference_latents "
                          f"have been passed. Ignoring the trajectory")
        
    else:
      raise RuntimeError(f"Either 'trajectory' or 'reference_latents' must be set")    

    if method == "mean_all":

      # get the total mean overall across every batch
      mean_latents = np.mean(reference_latents, axis=(0, 1))
      std_latents = np.std(reference_latents, axis=(0, 1))
      pylogging.info(f"get_reference_values: mean latents = {mean_latents}")
      pylogging.info(f"get_reference_values: std latents = {std_latents}")

      output = {
        "mean" : mean_latents,  # shape (N)
        "std" : std_latents,    # shape (N)
      }

    elif method == "mean":

      # keep the means per batch
      mean_latents = np.mean(reference_latents, axis=1)
      std_latents = np.std(reference_latents, axis=1)

      output = {
        "mean" : mean_latents,  # shape (B, N)
        "std" : std_latents,    # shape (B, N)
      }
    
    elif method == "passthrough":

      output = {
        "latents" : reference_latents, # shape (B, T, N)
      }

    elif method == "mahalanobis":

      pylogging.debug(f"Computing mahalanobis reference statistics")
      t0 = time.perf_counter()

      if use_jax:
        lib = jnp
      else:
        lib = np

      B, T, N = reference_latents.shape
      eps = 1e-6

      # Reshape to (B*T, N) - treat all samples equally
      reference_reshaped = einops.rearrange(reference_latents, "b t n -> (b t) n")
      
      # Robust statistics
      means = lib.median(reference_reshaped, axis=0, keepdims=True)  # shape (1, N)
      mad = lib.median(lib.abs(reference_reshaped - means), axis=0, keepdims=True)
      mad = lib.where(mad < eps, eps, mad)
      
      # Option 1a: Use MAD-normalized covariance (more robust)
      normed = (reference_reshaped - means) / mad
      cov = lib.cov(normed.T)  # More standard way to compute covariance

      cov_inv = lib.linalg.inv(cov + eps * lib.eye(N))
      
      output = {
          "means" : np.array(means),                   # shape (1, N)
          "mad" : np.array(mad),                       # shape (1, N) 
          "covs_inv" : np.array(cov_inv[None, :, :]),  # shape (1, N, N)
      }

    elif method == "mahalanobis-variance":

      pylogging.debug(f"Computing mahalanobis reference statistics")
      t0 = time.perf_counter()

      lib = jnp if use_jax else np
      B, T, N = reference_latents.shape
      eps = 1e-6

      reference_reshaped = einops.rearrange(reference_latents, "b t n -> (b t) n")
      
      # 1. Robust statistics
      means = lib.median(reference_reshaped, axis=0) # (N,)
      mad = lib.median(lib.abs(reference_reshaped - means[None, :]), axis=0)
      mad = lib.where(mad < eps, eps, mad)
      
      # 2. Global Covariance
      normed = (reference_reshaped - means[None, :]) / mad[None, :]
      cov = lib.cov(normed.T)  # (N, N)

      # don't do this, keep reference distribution clean, no model variance predictions
      # # 3. Add Intrinsic Variance
      # if reference_stddev is not None:
      #   # Match shapes and normalize
      #   stddevs_reshaped = einops.rearrange(reference_stddev, "b t n -> (b t) n")
      #   normed_vars = (stddevs_reshaped ** 2) / (mad[None, :] ** 2)
        
      #   avg_intrinsic_var = lib.mean(normed_vars, axis=0) # (N,)
      #   cov = cov + lib.diag(avg_intrinsic_var)

      # 4. Final Inverse
      cov_inv = lib.linalg.inv(cov + eps * lib.eye(N))
      
      output = {
          "cov" : np.array(cov[None, :, :]),           # (1, N, N)
          "means" : np.array(means),                   # (N,)
          "mad" : np.array(mad),                       # (N,)
          "covs_inv" : np.array(cov_inv[None, :, :]),   # (1, N, N)
      }

    elif method == "bhattacharyya":

      pylogging.debug(f"Computing bhattacharyya reference statistics")
      
      lib = jnp if use_jax else np
      B, T, N = reference_latents.shape
      eps = 1e-6

      reference_reshaped = einops.rearrange(reference_latents, "b t n -> (b t) n")
      
      # 1. Robust Stats (Means & MAD)
      means = lib.median(reference_reshaped, axis=0) # (N,)
      mad = lib.median(lib.abs(reference_reshaped - means[None, :]), axis=0)
      mad = lib.where(mad < eps, eps, mad)
      
      # 2. Reference Covariance (Sigma_p)
      # Note: We do NOT add intrinsic variance here; we want the clean manifold structure
      normed = (reference_reshaped - means[None, :]) / mad[None, :]
      cov = lib.cov(normed.T)  # (N, N)
      
      # 3. Pre-compute Log Determinant for Term 2
      # slogdet returns (sign, logdet); we take [1]
      _, logdet_cov = lib.linalg.slogdet(cov)

      output = {
          "cov" : np.array(cov[None, :, :]), # (1, N, N)
          "means" : np.array(means),         # (N,)
          "mad" : np.array(mad),             # (N,)
          "logdet_ref": np.array(logdet_cov) # scalar
      }

    elif method == "mahalanobis-scaling":

      pylogging.debug(f"Computing mahalanobis reference statistics")
      t0 = time.perf_counter()

      if use_jax:
        lib = jnp
      else:
        lib = np

      B, T, N = reference_latents.shape
      eps = 1e-6

      # Reshape to (B*T, N) - treat all samples equally
      reference_reshaped = einops.rearrange(reference_latents, "b t n -> (b t) n")
      
      # Robust statistics
      means = lib.median(reference_reshaped, axis=0, keepdims=True)  # shape (1, N)
      mad = lib.median(lib.abs(reference_reshaped - means), axis=0, keepdims=True)
      mad = lib.where(mad < eps, eps, mad)
      
      # Option 1a: Use MAD-normalized covariance (more robust)
      normed = (reference_reshaped - means) / mad
      cov = lib.cov(normed.T)  # More standard way to compute covariance

      cov_inv = lib.linalg.inv(cov + eps * lib.eye(N))
      
      output = {
          "means" : np.array(means),                   # shape (1, N)
          "mad" : np.array(mad),                       # shape (1, N) 
          "covs_inv" : np.array(cov_inv[None, :, :]),  # shape (1, N, N)
      }

    elif method == "maha-stddev":

      pylogging.debug(f"Computing mahalanobis reference statistics (using stddev)")
      t0 = time.perf_counter()

      use_jax = True
      if use_jax:
        lib = jnp
      else:
        lib = np

      B, T, N = reference_latents.shape
      eps = 1e-6

      # Reshape to (B*T, N) - treat all samples equally
      reference_reshaped = einops.rearrange(reference_latents, "b t n -> (b t) n")
      
      # Robust statistics
      means = lib.median(reference_reshaped, axis=0, keepdims=True)  # shape (1, N)
      stddev = lib.std(reference_reshaped, axis=0, keepdims=True)
      stddev = lib.where(stddev < eps, eps, stddev)
      
      # Option 1a: Use MAD-normalized covariance (more robust)
      normed = (reference_reshaped - means) / stddev
      cov = lib.cov(normed.T)  # More standard way to compute covariance

      output = {
          "means" : np.array(means),                   # shape (1, N)
          "stddev" : np.array(stddev),                 # shape (1, N) 
          "covs_inv" : np.array(cov_inv[None, :, :]),  # shape (1, N, N)
      }
      
    elif method == "maha-robust":

      from sklearn.covariance import MinCovDet
      pylogging.debug(f"Computing mahalanobis reference statistics (robust approach)")
      t0 = time.perf_counter()

      use_jax = True
      if use_jax:
        lib = jnp
      else:
        lib = np

      B, T, N = reference_latents.shape
      n_t = 1
      if n_t > 1:
        reference_latents = self.create_overlapping_windows(reference_latents, H=n_t,
                                                            use_jax=use_jax)
        # collapse new window dimension (note that T -> T - n_t + 1)
        reference_latents = einops.rearrange(reference_latents, "b t h n -> b t (h n)")

      B, T, N = reference_latents.shape
      eps = 1e-6

      # Reshape to (B*T, N) - treat all samples equally
      reference_reshaped = einops.rearrange(reference_latents, "b t n -> (b t) n")

      # --- Removed Robust Statistics (Median/MAD) ---
      # means = lib.median(reference_reshaped, axis=0, keepdims=True)  # shape (1, N)
      # mad = lib.median(lib.abs(reference_reshaped - means), axis=0, keepdims=True)
      # mad = lib.where(mad < eps, eps, mad)
      # normed = (reference_reshaped - means) / mad
      # cov = lib.cov(normed.T)
      # cov_inv = lib.linalg.inv(cov + eps * lib.eye(N))
      # ---

      # +++ Added MCD Robust Statistics +++
      #
      # NOTE: sklearn's MinCovDet requires NumPy arrays.
      # We must convert from JAX array to NumPy array if use_jax=True.
      # This computation will run on the CPU.
      reference_reshaped_np = np.array(reference_reshaped)

      # 1. Initialize and fit the MCD estimator
      #    support_fraction determines the proportion of "clean" data to use.
      #    A common value is ~0.75, but you can tune it.
      mcd = MinCovDet(support_fraction=0.75, random_state=42)
      mcd.fit(reference_reshaped_np)

      # 2. Get the robust mean (location) and covariance
      #    mcd.location_ is the robust mean, shape (N,)
      #    mcd.covariance_ is the robust covariance matrix, shape (N, N)
      robust_means = mcd.location_   # This is our new 'means'
      robust_cov = mcd.covariance_  # This is our new 'cov'

      # 3. Invert the robust covariance matrix (using NumPy)
      #    We still add eps for numerical stability.
      cov_inv = np.linalg.inv(robust_cov + eps * np.eye(N))

      # 4. Reshape to match your original output shapes
      robust_means = robust_means.reshape(1, N) # shape (1, N)
      cov_inv = cov_inv[None, :, :,]             # shape (1, N, N)
      # +++

      output = {
          "means" : np.array(robust_means),   # shape (1, N)
          "mad" : None,                       # 'mad' is no longer used in this approach
          "covs_inv" : np.array(cov_inv),     # shape (1, N, N)
          "num_timepoints" : n_t,             # number of adjacent timepoints in each vector
      }
      
    elif method == "var":

      # Set a small epsilon for numerical stability
      eps = 1e-6
      # Define the lag order for the VAR model. p=1 is the most common.
      p = 1
      use_diff = False

      # pylogging and time are assumed to be defined
      pylogging.debug(f"Computing VAR(p={p}) reference statistics")
      t0 = time.perf_counter()

      # We will use numpy for the OLS fit.
      lib = np

      B, T0, N = reference_latents.shape

      if use_diff:
        reference_latents = reference_latents[:, 1:, :] - reference_latents[:, :-1, :]

      # --- 1. Prepare data for OLS ---
      # We need to create Y (current values) and X (lagged values)
      # Y = e_t, X = [e_{t-1}, 1] (the 1 is for the intercept)
      # We must do this for each trajectory independently to respect boundaries.

      # Y: all timesteps from p onwards
      # Shape: (B, T0-p, N) -> (B*(T0-p), N)
      Y = reference_latents[:, p:, :].reshape(-1, N)

      # X: all timesteps from p-1 up to the one before the end
      # Shape: (B, T0-p, N) -> (B*(T0-p), N)
      X_lagged = reference_latents[:, p-1:-1, :].reshape(-1, N)

      # Add intercept term
      X_intercept = lib.ones((Y.shape[0], 1))
      X = lib.hstack([X_lagged, X_intercept])

      # --- 2. Solve for VAR parameters (OLS) ---
      # Y = X @ beta, where beta = [A1, c]
      # We use lstsq for numerical stability (more robust than direct inv)
      # beta will have shape (N+1, N)
      try:
          beta, _, _, _ = lib.linalg.lstsq(X, Y, rcond=None)
      except lib.linalg.LinAlgError as e:
          pylogging.error(f"VAR linear algebra fit failed: {e}")
          # Handle error appropriately, maybe raise it
          raise e

      # Extract parameters
      # A1 is the (N, N) matrix for e_{t-1}
      A1 = beta[:-1, :]
      # intercept is the (N,) vector
      intercept = beta[-1, :]

      # --- 3. Calculate Residuals ---
      # R = Y_actual - Y_predicted = Y - X @ beta
      R = Y - X @ beta  # Shape (B*(T0-p), N)

      pylogging.debug(f"VAR fit complete. Calculating residual statistics...")

      # --- 4. Get Standard Statistics of Residuals ---
      # This is the corrected section. We no longer use MAD.
      # We model the raw residuals directly.

      # The mean of OLS residuals is ~0, but we compute it for
      # numerical precision.
      res_means = lib.mean(R, axis=0, keepdims=True)  # shape (1, N)

      # Covariance of the *raw* residuals
      res_cov = lib.cov(R.T)
      res_cov_inv = lib.linalg.inv(res_cov + eps * lib.eye(N))

      # --- 5. Store all model parameters ---
      # We store the raw residual mean and covariance.
      # We no longer need 'res_mad'.
      output = {
          "p": p,
          "A1": np.array(A1),  # shape (N, N)
          "intercept": np.array(intercept), # shape (N,)
          "res_means": np.array(res_means), # shape (1, N)
          "res_cov_inv": np.array(res_cov_inv[None, :, :]), # shape (1, N, N)
          "use_diff" : use_diff,
      }

      pylogging.debug(f"VAR reference built in {time.perf_counter() - t0:.4f}s")

    elif method in ['l2', 'l1', 'l2-standardised', 'l1-standardised']:

      pylogging.info(f"Computing l2/l1/l2-standardised/l1-standardised reference statistics (same as mahalanobis)")
      t0 = time.perf_counter()

      if use_jax:
        lib = jnp
      else:
        lib = np

      B, T, N = reference_latents.shape
      eps = 1e-6

      # Reshape to (B*T, N) - treat all samples equally
      reference_reshaped = einops.rearrange(reference_latents, "b t n -> (b t) n")
      
      # Robust statistics
      means = lib.median(reference_reshaped, axis=0, keepdims=True)  # shape (1, N)
      mad = lib.median(lib.abs(reference_reshaped - means), axis=0, keepdims=True)
      mad = lib.where(mad < eps, eps, mad)
      
      # Option 1a: Use MAD-normalized covariance (more robust)
      normed = (reference_reshaped - means) / mad
      cov = lib.cov(normed.T)  # More standard way to compute covariance

      cov_inv = lib.linalg.inv(cov + eps * lib.eye(N))
      
      output = {
          "means" : np.array(means),                   # shape (1, N)
          "mad" : np.array(mad),                       # shape (1, N) 
          "covs_inv" : np.array(cov_inv[None, :, :]),  # shape (1, N, N)
      }

    else:
      raise RuntimeError(f"Agent_Latent_Discriminator.get_reference_values() error: "
                         f"method={method} not recognised")
      
    # save info about how the references were generated
    output["method"] = method
    output["decode"] = decode
      
    return output

  # --- utility functions --- #
    
  def timestep_rolling_average(self, tensor, window_size):
    """
    Compute rolling average over time axis (T), returning shape (B, T - window_size + 1, N).
    """
    B, T, N = tensor.shape
    if window_size < 1 or window_size > T:
        raise ValueError("window_size must be in range [1, T]")

    if window_size == 1:
        return tensor  # Return identity

    # Calculate cumulative sum along time axis
    cumsum = np.cumsum(tensor, axis=1)
    
    # Compute window sums by subtracting appropriate cumulative sums
    window_sums = np.zeros((B, T - window_size + 1, N))
    window_sums = cumsum[:, window_size-1:, :].copy()
    window_sums[:, 1:, :] -= cumsum[:, :-window_size, :]
    
    return window_sums / window_size

  def calculate_measurement_change(self, tensor):
    """
    Converts a tensor into an amount of change between each pair of subsequent
    elements.
    """

    B, N = tensor.shape

    if N < 2:
      raise RuntimeError(f"Agent_Latent_Discriminator.calculate_measurement_change() error: "
                         f"tensor.shape = {tensor.shape}, but the last dimension has"
                         f" size less than 2. At least two points are required for "
                         f"calculating the amount of change.")
    
    return tensor[:, 1:] - tensor[:, :-1]

  def create_overlapping_windows(self, data, H, use_jax=False):
    """
    Vectorized version using advanced indexing for better performance.
    
    Creates overlapping windows by copying memory using NumPy's advanced
    indexing capabilities, which is more efficient than explicit loops.

    Args:
        data (np.ndarray): Input array of shape (B, T, N).
        H (int): The horizon (window size).

    Returns:
        np.ndarray: New array of shape (B, T - H + 1, H, N) containing copied data.
        
    Raises:
        ValueError: If H <= 0, H > T, or data is not 3-dimensional.
    """
    
    if data.ndim not in [2, 3]:
        raise ValueError(f"Input data must be 2 or 3-dimensional, got {data.ndim}D")
    
    if not isinstance(H, int) or H <= 0:
        raise ValueError("Horizon H must be a positive integer")
    
    if use_jax:
      lib = jnp
    else:
      lib = np
    
    B, T = data.shape[:2]
    
    if H > T:
        raise ValueError(f"Horizon H ({H}) cannot be greater than the number of timesteps T ({T})")
    
    # Calculate output dimensions
    num_windows = T - H + 1
    
    # Create index arrays for advanced indexing
    # window_starts: [0, 1, 2, ..., num_windows-1]
    window_starts = lib.arange(num_windows)
    # offsets: [0, 1, 2, ..., H-1] 
    offsets = lib.arange(H)
    
    # Create indices for all windows at once
    # Broadcasting: (num_windows, 1) + (1, H) = (num_windows, H)
    indices = window_starts[:, lib.newaxis] + offsets[lib.newaxis, :]
    
    # Use advanced indexing to extract all windows at once
    # data[:, indices, :] has shape (B, num_windows, H, N)
    windowed_data = data[:, indices]
    
    return windowed_data

  def fig_to_img(self, fig):
    """
    Helper function to convert a matplotlib figure to an image to be 
    exported. 
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    img = plt.imread(buf, format='png')[..., :3]
    buf.close()
    if img.dtype != np.uint8:
          img = (255 * np.clip(img, 0, 1)).astype(np.uint8)
    plt.close(fig)
    return img

# --- depreciated functions --- #
