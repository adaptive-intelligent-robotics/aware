import torch
import numpy as np
import einops
import math
import time
import jax
import jax.numpy as jnp
import logging; logging.basicConfig(
    level=logging.INFO); pylogging = logging.getLogger(__name__)

from aware.utils.runs import load_agent

class Agent_Prediction_Discriminator:

  name = "Agent_Prediction_Discriminator"

  def __init__(self, timestamp=None, agent_file_starts=None, id=None, device="cuda", 
               agent=None, agent_cfg=None, **kwargs):
    """
    This class loads a motion predictor model and uses it as a discriminator to
    determine if any unexpected changes have occured in the environment.
    """

    if agent is None:
      self.agent, self.agent_cfg = load_agent(timestamp, agent_file_starts=agent_file_starts, 
                                              id=id, device=device, return_configs=True,
                                              **kwargs)
    else:
      self.agent = agent
      self.agent_cfg = agent_cfg
    
    if hasattr(self.agent, 'get_prediction_from_trajectory'):
      self.get_prediction_from_trajectory = self.agent.get_prediction_from_trajectory
    else:
      pylogging.warning(f"Agent_Prediction_Discriminator.__init__() error: "
                        f"loaded_agent (timestamp={timestamp}, id={id}) does not "
                        f"have a function called 'get_prediction_from_trajectory'")
      self.get_prediction_from_trajectory = None

    self.device = device
    self.last_index = None
    self.last_view = None

    # get vital information that the agent should expose
    try:
      self.t = self.agent.num_hist_timesteps_to_use
    except AttributeError as e:
      raise AttributeError(f"Agent_Prediction_Discriminator.__init__() error: "
                           f"self.agent with name={self.agent.name} does not have the "
                           f"required field 'num_hist_timesteps_to_use'.\nError msg: {e}")

  def trajectory_to_rollout(self, trajectory, start_index, horizon=0, num_steps=None):
    """
    Convert a trajectory into our rollout format (horizon * [qpos, qvel])
    """
    if num_steps is None:
      num_steps = trajectory["joint_angles"].shape[1] - start_index

    inds = slice(start_index, start_index + num_steps + horizon)
    qpos_hist = trajectory["joint_angles"][:, inds, :]
    qvel_hist = trajectory["joint_velocities"][:, inds, :]
    rollout = np.concatenate([qpos_hist, qvel_hist], axis=-1)

    return rollout
  
  def get_predictions_over_time(self, trajectory, index=None, num_steps=None, 
                                horizon=10, prediction_args={}, return_extras=False):
    """
    Get the predictions over time from a trajectory. The first prediction is made
    AT index.
    """

    # print(f'total_num_steps: {num_steps}')
    global_start_time = time.time()
    mean_time = 0
    total_time = 0

    B, T, _ = trajectory["joint_angles"].shape
    if num_steps is None:
      num_steps = T - index - horizon

    extras = {}
  
    for i in range(num_steps):

      start_time = time.time()
      rollout, extra_info = self.agent.get_prediction_from_trajectory(trajectory, 
                                                                        index=index + i, 
                                                                        horizon=horizon,
                                                                        return_extras=True,
                                                                        **prediction_args)
      
      if i == 0:
        # create data structure to hold all rolled out predictions
        B, H, N = rollout.shape
        full_rollout = np.zeros((B, num_steps, H, N))
        if return_extras:
          for key in extra_info:
            if extra_info[key] is not None:
              extras[key] = np.zeros((B, num_steps, *extra_info[key].shape[1:]))
      
      full_rollout[:, i, :] = rollout
      if return_extras:
        for key in extras:
          extras[key][:, i] = extra_info[key]

      total_time += time.time()-start_time
      mean_time = total_time / (i+1)
      # print(f"running mean_time: {mean_time}")
    
    run_time = time.time() -  global_start_time
    pylogging.debug(f"run_times = {run_time}")
    pylogging.debug(f"mean prediction time: {mean_time}")

    if return_extras:
      return full_rollout, extras
    else:
      return full_rollout

  def get_prediction_difference(self, true_trajectory, predictions, method="norm",
                                position=False, velocity=True, normalise_errors=False,
                                use_jax=False, aggregation=None, mode=None):
    """
    Determine the difference between measured trajectories and predictions.

    WARNING: Method expected in format: {aggregation}-{target_joints}

    """

    t0 = time.perf_counter()
    info = {} # dict for collecting additonal information
    pylogging.debug(f"method={method}, aggregation={aggregation}, mode={mode}")
    method = method.split("-")

    # override with explicit arguments, rather than method='norm-slew'
    if aggregation is None:
      aggregation = method[0]
    else:
      pylogging.debug(f"Aggregation override set as: {aggregation} (method={method})")
    if mode is None:
      mode = method[1] if len(method) > 1 else ""
    else:
      pylogging.debug(f"Mode override set as: {mode} (method={method})")

    if use_jax:
      lib = jnp
    else:
      lib = np

    if len(true_trajectory.shape) == 3:
      B1, T1, N1 = true_trajectory.shape
      B2, T2, N2 = predictions.shape
      true_trajectory = lib.expand_dims(true_trajectory, axis=2)
      predictions = lib.expand_dims(predictions, axis=2)
    elif len(true_trajectory.shape) == 4:
      B1, T1, W1, N1 = true_trajectory.shape
      B2, T2, W2, N2 = predictions.shape

      if W1 != W2:
        raise RuntimeError(f"Agent_Prediction_Discriminator.get_prediction_difference() error: "
                          f"window size of true_trajectory ({W1}), does not equal "
                          f"window size of predictions ({W2})")

    if B1 != B2:
      raise RuntimeError(f"Agent_Prediction_Discriminator.get_prediction_difference() error: "
                         f"batch size of true_trajectory ({B1}), does not equal "
                         f"batch size of predictions ({B2})")
    
    if T1 != T2:
      raise RuntimeError(f"Agent_Prediction_Discriminator.get_prediction_difference() error: "
                         f"timesteps in true_trajectory ({T1}), does not equal "
                         f"timesteps in predictions ({T2})")
    
    if N1 != N2:
      raise RuntimeError(f"Agent_Prediction_Discriminator.get_prediction_difference() error: "
                         f"feature size of true_trajectory ({N1}), does not equal "
                         f"feature size of predictions ({N2})")
    
    # if normalisation_dict is not None:
    #   assert 'means' in normalisation_dict, 'means missing from normalisation_dict'
    #   assert 'std' in normalisation_dict, 'std missing from normalisation_dict'

    #   true_trajectory = (true_trajectory - normalisation_dict['means']) / normalisation_dict['std']
    #   predictions = (predictions -  normalisation_dict['means']) / normalisation_dict['std']


    if mode == "actuators":
      if position and velocity:
        inds = lib.array([0, 1, 4, 7, 8, 11], dtype=int)
      elif position:
        inds = lib.array([0, 1, 4], dtype=int)
      elif velocity:
        inds = lib.array([7, 8, 11], dtype=int)
      else: raise RuntimeError("either position or velocity must be true")
    elif mode == "slew":
      if position and velocity:
        inds = lib.array([0, 7], dtype=int)
      elif position:
        inds = lib.array([0], dtype=int)
      elif velocity:
        inds = lib.array([7], dtype=int)
      else: raise RuntimeError("either position or velocity must be true")
    elif mode == "luff":
      if position and velocity:
        inds = lib.array([1, 8], dtype=int)
      elif position:
        inds = lib.array([1], dtype=int)
      elif velocity:
        inds = lib.array([8], dtype=int)
      else: raise RuntimeError("either position or velocity must be true")
    elif mode == "hoist":
      if position and velocity:
        inds = lib.array([4, 11], dtype=int)
      elif position:
        inds = lib.array([4], dtype=int)
      elif velocity:
        inds = lib.array([11], dtype=int)
      else: raise RuntimeError("either position or velocity must be true")
    elif mode == "payload":
      if position and velocity:
        inds = lib.array([2, 3, 5, 6, 9, 10, 12, 13], dtype=int)
      elif position:
        inds = lib.array([2, 3, 5, 6], dtype=int)
      elif velocity:
        inds = lib.array([9, 10, 12, 13], dtype=int)
      else: raise RuntimeError("either position or velocity must be true")
    elif mode == "no_double_pendulum":
      if position and velocity:
        inds = lib.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 11], dtype=int)
      elif position:
        inds = lib.array([0, 1, 2, 3, 4], dtype=int)
      elif velocity:
        inds = lib.array([7, 8, 9, 10, 11], dtype=int)
      else: raise RuntimeError("either position or velocity must be true")
    elif mode == "exp":
      if position and velocity:
        inds = lib.array([0, 1, 2, 3, 7, 8, 9, 10], dtype=int)
      elif position:
        inds = lib.array([0, 1, 2, 3, 4], dtype=int)
      elif velocity:
        inds = lib.array([7, 8, 9, 10, 11], dtype=int)
      else: raise RuntimeError("either position or velocity must be true")
    elif mode in ["", "all"]:
      if position and velocity:
        inds = lib.arange(14)
      elif position:
        inds = lib.arange(7)
      elif velocity:
        inds = lib.arange(7, 14)
      else: raise RuntimeError("either position or velocity must be true")
    else:
      raise RuntimeError(f"Agent_Prediction_Discriminator.get_prediction_difference() error: "
                         f"mode={mode} not recognised, should be '', 'all', 'slew', 'actuators' etc")
    
    # save the inds used - useful for double discriminator
    # print(f"Prediction discrim inds = {inds}")
    info["joint_diff_inds"] = inds

    # Objective = return difference with shape (B, T)
    # This is one measurement per timestep, repeated for every batch
    # Hence, need to average across feature dim N, and horizon H
    # Axes:  0, 1, 2, 3
    # Shape: B, T, H, N

    # normalise across all dimensions - hardcoded values from real data
    norm_factors = lib.array([
      0.7625,
      0.3621,
      0.1428,
      0.3740,
      0.4317,
      0.0195,
      0.0135,
      0.0658,
      0.0327,
      0.4779,
      0.3340,
      0.0833,
      0.1890,
      0.1426,
    ])

    # makes no difference when using mahalanbois distance
    if normalise_errors:
      print(F"prediction discriminator is normalising the errors!")
      true_trajectory = lib.divide(true_trajectory, norm_factors)
      predictions = lib.divide(predictions, norm_factors)
    
    t1 = time.perf_counter()
    joint_difference = true_trajectory - predictions
    t2 = time.perf_counter()
    
    info['joint_difference'] = joint_difference
    info['true_trajectory'] = true_trajectory
    info['pred_trajectory'] = predictions

    joint_difference = joint_difference[:, :, :, inds]

    # print(f"method = {method}, aggregation = {aggregation}, position = {position}, velocity = {velocity}, "
    #       f"inds = {inds}, joint_difference.shape = {joint_difference.shape}")

    if aggregation == "norm":
      # norm over joint angle differences, averaged over all steps
      diff = lib.mean(lib.linalg.norm(joint_difference, axis=3), axis=2)
    elif aggregation == "norm_per_component":
      if mode == 'no_double_pendulum':
        act_component_idx = lib.array([0, 1, 4, 5, 6, 9])
        payload_component_idx = lib.array([2, 3, 7, 8])
        act_diff = lib.mean(lib.linalg.norm(joint_difference[..., act_component_idx], axis=3), axis=2) 
        payload_diff = lib.mean(lib.linalg.norm(joint_difference[..., payload_component_idx], axis=3), axis=2) 
        diff = lib.stack([act_diff, payload_diff],axis=2)
      else:
        raise NotImplementedError(f"norm_per_component not implemented for {mode}")
    elif aggregation == "average":
      # average over joint angle differences, averaged over all steps
      diff = lib.mean(lib.mean(np.abs(joint_difference), axis=3), axis=2)
    elif aggregation == "end":
      # norm over joint angle differences, taken at the last timestep
      diff = lib.linalg.norm(np.abs(joint_difference), axis=3)[:, :, -1]
    elif aggregation == "per_joint":
      # get the average difference per joint, without a norm or similar
      diff = lib.mean(np.abs(joint_difference), axis=2) # average over horizon only -> (B, T, N)
    elif aggregation == "none":
      diff = lib.zeros((B1, T1)) # skip difference here, values are passed via info dict
    else:
      raise RuntimeError(f"Agent_Prediction_Discriminator.get_prediction_difference() error: "
                        f"aggregation={aggregation} not recognised")
    
    info["measurement"] = diff
    
    t3 = time.perf_counter()

    pylogging.debug(f"get_prediction_difference(), mode={mode}, aggregation={aggregation}:\n"
                    f" -> time taken for determining mode and prep: {t1 - t0:.3f}s\n"
                    f" -> time taken for actual joint error subtraction: {t2 - t1:.3f}s\n"
                    f" -> time taken for final aggregation: {t3 - t2:.3f}s\n"
                    f" -> total time taken: {t3 - t0:.3f}s"
                    )

    return diff, info
  
  def get_measurement_from_trajectory(self, trajectory, index=None, num_measurements=None,
                                      window_size=1, method="norm", relative_change=False,
                                      horizon=10, use_jax=False, return_extra=False,
                                      prediction_args={},
                                      **prediction_difference_args):
    """
    Get the error measurement over time for a trajectory. 

    Index is the timestep in the trajectory for which the first measurement is given.
    For example, at index = 20, that is the measurement which arises from the previous
    x steps of data, inclusive (x being the amount required for one measurement).

    num_measurements is the number of measurements returned by this function, starting
    at index.
    relative_change=True means this function returns the change between adjacent
    measurements rather than the raw measurements themselves.
    The horizon is how far into the future predictions are made, 
    The window size is the width of rolling average over measurements.
    """

    timer = time.perf_counter
    t0 = timer()
    # the minimum history required to make inference with the model
    min_hist = self.t + window_size + horizon - 1 # 299

    if index is None:
      index = min_hist + int(relative_change) # need extra point for relative change
    elif index < 0:
      index = trajectory["joint_angles"].shape[1] + index
    elif index < min_hist:
      pylogging.warning(f"Agent_Prediction_Discriminator.get_measurement_from_trajectory() error: "
                        f"cannot have index={index} less than {min_hist} "
                        f"(self.t={self.t} + window_size={window_size} + horizon={horizon})")
      raise RuntimeError(f"Agent_Prediction_Discriminator.get_measurement_from_trajectory() error: "
                         f"cannot have index={index} less than {min_hist} "
                         f"(self.t={self.t} + window_size={window_size} + horizon={horizon})")
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
    # if max_measure < 0: 
    #   raise RuntimeError(f"Agent_Prediction_Discriminator.get_measurement_over_time() error: "
    #                      f"trajectory length = {trajectory['joint_angles'].shape[1]} is too"
    #                      f" short for the given index of {index}.")
    # if num_measurements > max_measure:
    #   raise RuntimeError(f"Agent_Prediction_Discriminator.get_measurement_over_time() error: "
    #                      f"num_measurements={num_measurements}, with T={trajectory['joint_angles'].shape[1]} "
    #                      f"timesteps in the trajectory, maximum number of measurements is "
    #                      f"{max_measure} when index={index} (minimum history={min_hist}, "
    #                      f"self.t={self.t} + window_size={window_size} + horizon={horizon})")
      
    # we need to roll forwards from a previous point in the trajectory
    # we look into the past to make sure we have ground truth data for all points to compare against
    past_index = index - (horizon + window_size - 1) 
    num_steps = num_measurements + (window_size - 1) 

    # save to simplify aligning (this is the number of measurements 'dropped' from front of traj)
    self.last_index = index
    self.last_view = horizon + window_size + self.t

    if hasattr(self.agent, 'model') and  hasattr(self.agent.model, 'add_privileged_info'):
      add_privileged_info = self.agent.model.add_privileged_info
    else:
      if hasattr(self.agent, 'add_privileged_info'):
        add_privileged_info = self.agent.add_privileged_info
      else: 
        add_privileged_info = False

    # last minute test: if we fix latents over WHOLE trajectory?
    if ("predict_latents_only_once" in prediction_args and
        prediction_args["predict_latents_only_once"] and
        hasattr(self.agent, "get_latent_estimator")):
      
      if "fix_estimated_latents" in prediction_args:
          prediction_args["fix_estimated_latents"] = False # these settings clash, disable

      # Transformer AWARE models
      if hasattr(self.agent, "model"):

        # do we have latents enabled
        if (hasattr(self.agent.model, "add_privileged_info") and
            self.agent.model.add_privileged_info):
      
          pylogging.info(f"IMPORTANT! Predicting latents only once in prediction discriminator")
          
          if "fix_estimated_latents" in prediction_args:
            prediction_args["fix_estimated_latents"] = False # these settings clash, disable

          # check if the predictor gets the confidence
          if self.agent.model.predictor_gets_confidence:
            return_confidence = True
          else:
            return_confidence = False

          # get the estimator
          estimator = self.agent.get_latent_estimator()


          # get the observation
          obs_hist = self.agent.get_observation_from_trajectory(trajectory, 
                                                                prediction_index=past_index)
          with torch.no_grad():
            # standard case, estimate the latents from state history
            state_obs = torch.tensor(obs_hist, device=self.device, dtype=torch.float32)
            estimated_latents = estimator(state_obs, return_confidence=return_confidence)

          # pass in the same set of estimated latents every time
          prediction_args["latent_override"] = estimated_latents 

      # RSSM
      else:

        # check if the predictor gets the confidence
        if self.agent.predictor_gets_confidence:
          return_confidence = True
        else:
          return_confidence = False

        # get the observation and estimate the latents
        estimator = self.agent.get_latent_estimator()
        obs_hist = self.agent.get_observation_from_trajectory(trajectory, 
                                                              prediction_index=past_index)
        estimated_latents = estimator(obs_hist, return_confidence=return_confidence)

        # pass in the same set of estimated latents every time
        prediction_args["latent_override"] = estimated_latents 

    # test using an overall latent override
    if ("predictor_latent_override" in prediction_args and
        prediction_args["predictor_latent_override"] is not None and
        hasattr(self.agent, "get_latent_estimator")):
      pylogging.warning(f"\n\n==== PREDICTOR LATENT OVERRIDE SET ====\n\n")
      prediction_args["latent_override"] = prediction_args["predictor_latent_override"]
      # catch RSSM case
      if not hasattr(self.agent, "model"):
        # get expected size (as with fix estimated latents above)
        if self.agent.predictor_gets_confidence: return_confidence = True
        else: return_confidence = False
        estimator = self.agent.get_latent_estimator()
        obs_hist = self.agent.get_observation_from_trajectory(trajectory, 
                                                              prediction_index=past_index)
        estimated_latents = estimator(obs_hist, return_confidence=return_confidence)
        # cast to every single timestep
        n_t = estimated_latents.shape[0] # get only the shape
        prediction_args["latent_override"] = einops.repeat(
          prediction_args["latent_override"], "n -> t n", t=n_t)

    # convert the trajectory into a ground truth and set of horizon length rollouts
    predictions, pred_extras = self.get_predictions_over_time(trajectory, index=past_index,
                                                              num_steps=num_steps, horizon=horizon,
                                                              prediction_args=prediction_args,
                                                              return_extras=True)
    B, T, H, N = predictions.shape # per T we have an H step horizon of N joint predictions
    t1 = timer()

    true_traj_raw = self.trajectory_to_rollout(trajectory, start_index=past_index, 
                                               num_steps=index + (num_measurements - 1) - past_index)
    true_traj_stacked = self.create_overlapping_windows_safe(true_traj_raw, H=horizon,
                                                             use_jax=use_jax)
    # true_traj_stacked = self.create_overlapping_windows_strided(true_traj_raw, H=horizon)

    t2 = timer()

    # print(f"index={index}, min_hist={min_hist}, past_index={past_index}, window_size={window_size}, horizon={horizon}")
    # print(f"predictions.shape = {predictions.shape}")
    # print(f"true_traj_raw.shape = {true_traj_raw.shape}")
    # print(f"true_traj_stacked.shape = {true_traj_stacked.shape}")

    B2, T2, H2, N2 = true_traj_stacked.shape

    if B2 != B: raise RuntimeError(f"(predictions) B={B} != B2={B2} (true traj)")
    if T2 != T: raise RuntimeError(f"(predictions) T={T} != T2={T2} (true traj)")
    if H2 != H: raise RuntimeError(f"(predictions) H={H} != H2={H2} (true traj)")
    if N2 != N: raise RuntimeError(f"(predictions) N={N} != N2={N2} (true traj)")

    # process measurement differences on GPU, major bottleneck otherwise
    if use_jax:
      D = (2,) if 'per_component' in method else ()

      # 1. Pre-allocate the main measurements array (CPU)
      measurements = np.zeros((B, num_steps) + D)
      
      # Prepare for info dictionary pre-allocation
      info_buffers = {}
      joint_diff_inds = None # To store the static metadata
      initialized_info = False

      tl_0 = timer()

      # batches with shape beyond [1000, 980, 10, 14] crash due to out of memory
      max_batch = 100 
      num_loops = math.ceil(B / max_batch)

      for n in range(num_loops):
        tl_1 = timer()
        start = max_batch * n
        end = min(max_batch * (n + 1), B) 
        sub_ind = slice(start, end)

        true_traj_batch = true_traj_stacked[sub_ind]
        predictions_batch = predictions[sub_ind]

        # move onto the GPU
        true_traj_jax = jax.device_put(jnp.array(true_traj_batch))
        predictions_jax = jax.device_put(jnp.array(predictions_batch))

        # get measurements for this batch
        batch_measurements, batch_info = self.get_prediction_difference(
            true_traj_jax, 
            predictions_jax, 
            method=method,
            use_jax=True,
            **prediction_difference_args
        )

        # 2. Transfer measurements to CPU immediately
        # This prevents GPU memory buildup
        measurements[sub_ind] = np.array(batch_measurements)

        # 3. Handle 'info' dictionary
        # On the very first loop, we learn the shape of 'info' items and pre-allocate
        if not initialized_info:
            for key, val in batch_info.items():
                if key == "joint_diff_inds":
                    # Capture static metadata once
                    joint_diff_inds = val 
                    continue
                
                # Convert first batch to numpy to get correct shape/dtype
                sample_np = np.array(val)
                # Pre-allocate full buffer on CPU
                # shape[1:] assumes the first dimension is the batch dimension
                info_buffers[key] = np.zeros((B,) + sample_np.shape[1:], dtype=sample_np.dtype)
            
            initialized_info = True

        # 4. Fill the pre-allocated buffers
        for key, val in batch_info.items():
            if key == "joint_diff_inds": continue
            info_buffers[key][sub_ind] = np.array(val)

        # 5. Aggressively free GPU references
        del batch_measurements, batch_info, true_traj_jax, predictions_jax
        
        # Optional: Force garbage collection if memory is extremely tight
        # import gc; gc.collect() 

        tl_2 = timer()

      # 6. Reconstruct the final info dictionary
      # No concatenation happens here; just pointing to the buffers we filled
      info = info_buffers
      if joint_diff_inds is not None:
          info["joint_diff_inds"] = joint_diff_inds

      tl_3 = timer()

      pylogging.debug(f"Time taken for GPU measure = {tl_2 - tl_0:.3f}s")
      pylogging.debug(f"Time taken for measurement processing = {tl_3 - tl_2:.3f}s")

    else:
      # evaluate on the CPU, warning - slow for v. large batches
      measurements, info = self.get_prediction_difference(true_traj_stacked, 
                                                          predictions, 
                                                          method=method,
                                                          **prediction_difference_args)
      
    t3 = timer()
        
    # apply the window size rolling average
    measurements = self.timestep_rolling_average(measurements, window_size=window_size)

    t4 = timer()

    # are we taking the relative change between measurements
    if relative_change:
      measurements = self.calculate_measurement_change(measurements)

    t5 = timer()

    pylogging.info(f"Timings for 'get_measurement_from_trajectory:'")
    pylogging.info(f"Time taken to query model: {(t1 - t0):.3f}s")
    pylogging.info(f"Time taken to handle reference: {(t2 - t1):.3f}s")
    pylogging.info(f"Time taken to calculate difference (method={method}, use_jax={use_jax}): "
                   f"{(t3 - t2):.3f}s")
    pylogging.info(f"Time taken to rolling average: {(t4 - t3):.3f}s")
    if relative_change:
      pylogging.debug(f"Time taken to convert to relative change: {(t5 - t4):.3f}s")

    extra_info = info | pred_extras | {
      "predictions" : predictions,
      "index" : self.last_index,
      "view" : self.last_view,
    }

    if return_extra:
      return measurements, extra_info
    else:
      return measurements

  # --- utility functions --- #

  def timestep_rolling_average(self, tensor, window_size):
    """
    Compute rolling average over time axis (T), returning shape (B, T - window_size + 1).
    """
   
    B, T = tensor.shape[:2] # tensor of measurements along time, batched
    if window_size < 1 or window_size > T:
      raise ValueError("window_size must be in range [1, T]")

    if window_size == 1:
      return tensor  # Return identity

    # Calculate cumulative sum along time axis
    cumsum = np.cumsum(tensor, axis=1)
    
    # Compute window sums by subtracting appropriate cumulative sums
    window_sums = np.zeros((B, T - window_size + 1))
    window_sums = cumsum[:, window_size-1:].copy()
    window_sums[:, 1:] -= cumsum[:, :-window_size]
    
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
  
  def create_overlapping_windows_safe(self, data, H, use_jax=False):
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
    
    if data.ndim != 3:
        raise ValueError(f"Input data must be 3-dimensional, got {data.ndim}D")
    
    if not isinstance(H, int) or H <= 0:
        raise ValueError("Horizon H must be a positive integer")
    
    if use_jax:
      lib = jnp
    else:
      lib = np
    
    B, T, N = data.shape
    
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
    windowed_data = data[:, indices, :]
    
    return windowed_data
