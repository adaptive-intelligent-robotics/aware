import numpy as np
from copy import deepcopy
import einops
import logging; logging.basicConfig(
    level=logging.INFO); pylogging = logging.getLogger(__name__)

from aware.agents.prediction_discriminator import Agent_Prediction_Discriminator
from aware.agents.latent_discriminator import Agent_Latent_Discriminator

def robust_softplus(x):
    # For x > 34, softplus(x) is essentially x within float64 precision
    return np.where(x > 34, x, np.log1p(np.exp(x)))

class Agent_Double_Discriminator:

  name = "Agent_Double_Discriminator"
  
  def __init__(self, timestamp=None, agent_file_starts=None, id=None, device="cuda", 
               agent=None, pred_discrim=None, latent_discrim=None, dummy_latents=False,
               **kwargs):
    """
    Discriminator which uses both predictions and latents. Joint errors and latents
    are concat into one vector, and treated like a larger collection of latents.
    """

    if pred_discrim is None:
      self.pred_discrim = Agent_Prediction_Discriminator(
        timestamp=timestamp,
        agent_file_starts=agent_file_starts,
        id=id,
        device=device,
        agent=agent,
        **kwargs
      )
    else: self.pred_discrim = pred_discrim
    
    if latent_discrim is None:
      self.latent_discrim = Agent_Latent_Discriminator(
        timestamp=timestamp,
        agent_file_starts=agent_file_starts,
        id=id,
        device=device,
        agent=agent,
        dummy_latents=dummy_latents,
        **kwargs
      )
    else: self.latent_discrim = latent_discrim
    self.dummy_latents = self.latent_discrim.dummy_latents

    self.t = max(self.pred_discrim.t, self.latent_discrim.t)
    self.agent_cfg = self.pred_discrim.agent_cfg # expose a consistent API
    self.last_index = None
    self.last_view = None

  def get_measurement_from_trajectory(self, trajectory, index=None, num_measurements=None,
                                      window_size=1, method="mahalanobis",
                                      double_mode="concat", joint_diff_method="average",
                                      return_extra=False, relative_change=False,
                                      use_reference=True, reference_obj=None,
                                      reference_type="auto", reference_auto_num_samples=100,
                                      reference_auto_num_trajectories=10,
                                      reference_study_repeat_num=10,
                                      computing_reference=False, use_jax=False,
                                      denoise_reference_trajectory=False,
                                      add_confidence_as_measurement=False,
                                      add_confidence_as_feature=False,
                                      multiply_by_confidence=False,
                                      scale_by_uncertainty=False,
                                      return_latent_decodings=False,
                                      pred_args={}, latent_args={}):
    """
    Return measurements from a trajectory based on a given index to start from
    and the desired number of measurements
    """

    relative_change = False

    # define method characteristics
    reference_methods = ["mahalanobis", "mahalanobis-variance", "mahalanobis-scaling",
                         "bhattacharyya", "maha-robust", "maha-stddev", "var", "l2", "l1", "l2-standardised", "l1-standardised"]
    single_point_methods = ["norm", "area", "cosine"]
    # if method not in reference_methods

    allowable_modes = ["latent_only", "errors_only", "concat", "seperate"]
    if double_mode not in allowable_modes:
      raise RuntimeError(f"{self.name}.get_measurement_from_trajectory() error: "
                         f"double_mode={double_mode} not recognised. The possible "
                         f"options are: {allowable_modes}")

    # the minimum history required to make inference with the model
    if "horizon" not in pred_args:
      pred_args["horizon"] = 10 # default, but should be set
      pylogging.warning(f"'horizon' not set, has been set as 10. "
                        f"This value should be set explicitly in 'pred_args'")
    min_hist_pred = self.pred_discrim.t + pred_args["horizon"] + (window_size - 1)
    min_hist_latent = self.latent_discrim.t + (window_size - 1)

    # models should align, take the one which requireds more history
    min_hist = max(min_hist_pred, min_hist_latent)
    if index is None:
      index = min_hist + int(relative_change) # need extra point for relative change
    elif index < 0:
      index = trajectory["joint_angles"].shape[1] + index
    elif index < min_hist:
      pylogging.warning(f"Agent_Double_Discriminator.get_measurement_from_trajectory() error: "
                        f"cannot have index={index} less than {min_hist} "
                        f"(self.t={self.t} + window_size={window_size})")
      # set the index to the new minimum, to avoid an error
      index = min_hist
      pylogging.warning(f"\n\n\n{'='*10} DOUBLE DISCRIM: SETTING START INDEX = {min_hist} {'='*10}\n\n")
      
    # --- query the discriminators --- #

    latent_extras = {}
    pred_extras = {}
    double_extras = {}

    # used for plotting and detemermining manipulation window sizes
    latent_last_view = 0
    pred_last_view = 0

    if double_mode != "latent_only":

      if denoise_reference_trajectory:
        denoise_traj = self.pred_discrim.agent.denoise_trajectory(deepcopy(trajectory))
        pred_trajectory = deepcopy(denoise_traj)
      else:
        pred_trajectory = trajectory

      pred_measure, pred_extras = self.pred_discrim.get_measurement_from_trajectory(
        trajectory=pred_trajectory,
        index=index,
        num_measurements=num_measurements,
        window_size=1, # ensure no smoothing yet
        relative_change=False,
        return_extra=True,
        use_jax=use_jax,
        **pred_args | {
          # "aggregation" : "none", # avoid any calculation, as pred_measure as not used
        },
      )
      pred_last_view = self.pred_discrim.last_view

      if denoise_reference_trajectory:
        pred_extras["denoised_trajectory"] = denoise_traj
        pred_extras["trajectory_before_denoise"] = deepcopy(trajectory)

      # check if the predictor outputs an uncertainty, and process into confidence score
      if "predictor_uncertainty" in pred_extras:
        predictor_uncertainty = pred_extras["predictor_uncertainty"]
        # reshape to get one confidence per horizon timestep
        if predictor_uncertainty.ndim == 5:
          predictor_uncertainty = einops.rearrange(predictor_uncertainty, 
                                            "b t roll chunk x -> b t (roll chunk) x")
        elif predictor_uncertainty.ndim != 4:
          raise RuntimeError(f"predictor_uncertainty dimension should be 4 or 5, shape = "
                              f"{predictor_uncertainty.shape}, expected (B, T, roll/?, chunk, x)")
        # # process the uncertainty, assuming it is output as a log variance
        # pred_variance = np.exp(predictor_uncertainty)
        # k = 1
        # pred_confidence = np.exp(-k * pred_variance)
        # pred_extras["predictor_confidence_per"] = pred_confidence # per horizon timestep, per feature
        # pred_extras["predictor_confidence"] = np.mean(pred_confidence, axis=(2, 3))

      # can we use the standard deviations from the model (in real units) to adjust our errors
      if "predictor_stddev_denormalised" in pred_extras:
        predictor_stddev_denormalised = pred_extras["predictor_stddev_denormalised"]
        # reshape to get one confidence per horizon timestep
        if predictor_stddev_denormalised.ndim == 5:
          predictor_stddev_denormalised = einops.rearrange(predictor_stddev_denormalised, 
                                            "b t roll chunk x -> b t (roll chunk) x")
        elif predictor_stddev_denormalised.ndim < 4:
          raise RuntimeError(f"predictor_stddev_denormalised dimension should be 4 or 5, shape = "
                              f"{predictor_stddev_denormalised.shape}, expected (B, T, roll/?, chunk, x)")
        # shape B, T, H, N
        pred_extras["predictor_stddev_denormalised"] = predictor_stddev_denormalised # per horizon timestep, per feature

      if (use_reference or method not in single_point_methods 
          or method == "average_seperate"):

        # extract crucial information
        joint_diffs = pred_extras["joint_difference"] # shape B, T, H, N2

        if scale_by_uncertainty and "predictor_stddev_denormalised" in pred_extras:
          pylogging.warning("NORMALISING THE PREDICTOR ERRORS BASED ON UNCERTAINTY")
          joint_diffs = joint_diffs / pred_extras["predictor_stddev_denormalised"]

        # determine how to collapse the H-step horizon down to one measurement per joint
        if joint_diff_method == "average":
          joint_diffs = np.mean(np.abs(joint_diffs), axis=2) # average over the horizon
        elif joint_diff_method == "norm":
          joint_diffs = np.linalg.norm(joint_diffs, axis=2) # norm of error vector over whole horizon (per joint)
        elif joint_diff_method == "end":
          joint_diffs = np.abs(joint_diffs)[:, :, -1] # take abs value of last error datapoint at horizon
        elif joint_diff_method == "area":
          area_true = np.trapz(pred_extras['true_trajectory'], axis=2) # to shape B, T, N2
          area_pred = np.trapz(pred_extras['pred_trajectory'], axis=2) # to shape B, T, N2
          joint_diffs = np.abs(area_true - area_pred)
        # experimental: test using a covariance only method
        elif joint_diff_method == "covariance_method":
          joint_diffs = pred_extras['true_trajectory'][:, :, 0, :] # shape (B, T, H, N) -> (B, T, N)
          bb, tt, nn = joint_diffs.shape
          # add in the actions
          qpos = trajectory["joint_angles"][:, index:index + tt, :]
          qvel = trajectory["joint_velocities"][:, index:index + tt, :]
          actions = trajectory["last_action"][:, index:index + tt, :]
          # qvel = np.zeros_like(qvel)
          print(f"\n\n\nadding actions\n\n\n")
          joint_diffs = np.concatenate((qpos, qvel, actions), axis=-1)
        else:
          raise RuntimeError(f"{self.name}.get_measurement_from_trajectory() error: "
                             f"joint_diff_method={joint_diff_method} not recognised")

        # use the indexing specified in the prediction discriminator
        inds = pred_extras["joint_diff_inds"]
        pylogging.info(f"Double Discrim inds are: {inds}")
        joint_diffs = joint_diffs[:, :, inds]

      else:
        # go straight to reference, no need to get intermediate errors
        joint_diffs = pred_extras["measurement"]

      # test: window size to smooth error vectors
      pred_window_size = None
      if pred_window_size is not None:
        averaged = self.pred_discrim.timestep_rolling_average(joint_diffs,
                                                              window_size=pred_window_size)
        joint_diffs[:, pred_window_size -1:] = averaged
        joint_diffs[:, :pred_window_size - 1] = 0.0 # to see on plots

      if "predictor_confidence" in pred_extras and add_confidence_as_feature:
        joint_diffs = np.concatenate((
          joint_diffs, 
          np.expand_dims(pred_extras["predictor_confidence"], axis=2),
          # np.expand_dims(np.std(pred_extras["predictor_confidence_per"], axis=(2,3)), axis=2),
        ), axis=2)
        print(f'==== predictor confidence added as feature!! range = (min={np.min(pred_extras["predictor_confidence"])}, max={np.max(pred_extras["predictor_confidence"])}) ====')

    if double_mode != "errors_only":

      latent_measure, latent_extras = self.latent_discrim.get_measurement_from_trajectory(
        trajectory=trajectory,
        index=index,
        num_measurements=num_measurements,
        window_size=1, # ensure no smoothing yet
        relative_change=False,
        return_extra=True,
        use_jax=use_jax,
        return_decodings=return_latent_decodings,
        ** latent_args | {
          "use_reference" : False, # override
          "method" : "norm", # select method which allows 'use_reference=False'
        },
      )
      
      latent_last_view = self.latent_discrim.last_view

      if return_latent_decodings:
        assert "decoded_latents" in latent_extras

      # extract crucial information
      latents = latent_extras["latents"] # shape B, T, N1
      # process any latent uncertainty into a latent confidence score (if present)
      if latent_extras["latent_uncertainty"] is not None:
        # edl models
        if (hasattr(self.latent_discrim.agent, "model") and
            hasattr(self.latent_discrim.agent.model, "use_edl_confidence") and
            self.latent_discrim.agent.model.use_edl_confidence):
          # uncertainty shape -> (B, T, 3)
          # print(f"uncertainty.shape = {latent_extras['latent_uncertainty'].shape}")
          nu_raw = latent_extras["latent_uncertainty"][:, :, 0] # shape B, T
          alpha_raw = latent_extras["latent_uncertainty"][:, :, 1] # shape B, T
          beta_raw = latent_extras["latent_uncertainty"][:, :, 2] # shape B, T
          # process for stability (ensure positivity, and alpha >= 1)
          nu = robust_softplus(nu_raw) + 1e-6
          beta = robust_softplus(beta_raw) + 1e-6
          alpha = robust_softplus(alpha_raw) + 1.0 + 1e-6
          # quantify uncertainty
          aleatoric_unc = beta / (alpha - 1.0)
          epistemic_unc = beta / (nu * (alpha - 1.0))
          total_unc = aleatoric_unc + epistemic_unc
          # quantify confidence
          k = 1
          aleotoric_conf = np.exp(-k * aleatoric_unc)
          epistemic_conf = np.exp(-k * epistemic_unc)
          total_conf = np.exp(-k * total_unc)
          # alternative option - use evidence directly for confidence
          evidence_conf = 1 - np.exp(-k * nu)
          # determine which confidence value to use
          latent_confidence = epistemic_conf
        else:
          latent_uncertainty = latent_extras["latent_uncertainty"]
          latent_extras["latent_stddev"] = np.sqrt(np.exp(latent_uncertainty))
          if latent_uncertainty.ndim == 3:
            # average over the feature dimension if there is one
            latent_uncertainty = np.mean(latent_uncertainty, axis=2)
        #   # quantify confidence
        #   k = 5
        #   latent_confidence = np.exp(-k * np.exp(latent_uncertainty))
        # # save confidence into extra features
        # latent_extras["latent_confidence"] = latent_confidence
        if add_confidence_as_feature:
          latents = np.concatenate([latents, np.expand_dims(latent_uncertainty, axis=2)], axis=2) # shapes (B, T, N)s
          print(f'==== latent confidence added as feature!! range = (min={np.min(latent_extras["latent_confidence"])}, max={np.max(latent_extras["latent_confidence"])}) ====')

      # experimental, insert confidence score into 'latents' prior to mahalanobis calc.
      concat = False
      if concat:
        latents = np.concatenate((latents, latent_uncertainty), axis=-1)
        # latents = np.concatenate((latents, np.mean(latent_uncertainty, axis=2, keepdims=True)), axis=-1)
        print(f"\n\n==== latent confidence concat on latent vector ====\n")
    
    # careful! only write these when computing actual measurement, not reference
    if not computing_reference:
      # expose these convenience variables for alignment in plotting
      self.last_index = index
      self.last_view = max(latent_last_view, pred_last_view)

    # determine how to combine the two measurements
    if double_mode == "concat":

      # concatenate into one vector
      double_vec = np.concatenate((latents, joint_diffs), axis=2) # shape B, T, N1 + N2
      measure_dims = [double_vec]
      modes = ["concat"]

    elif double_mode == "seperate":

      measure_dims = [latents, joint_diffs]
      modes = ["latent_only", "errors_only"]

    elif double_mode == "latent_only":

      measure_dims = [latents]
      modes = ["latent_only"]

    elif double_mode == "errors_only":

      measure_dims = [joint_diffs]
      modes = ["errors_only"]
      
    else:
      raise RuntimeError(f"Agent_Double_Discriminator.get_measurement_from_trajectory() error: "
                         f"double_mode={double_mode} not recognised")
        
    # --- handle references (required for proper measurements) --- #
    

    final_measurements = []

    for measure_vec, this_mode in zip(measure_dims, modes):

      # use model variance in covariance matrix for mahalanobis distance
      measurement_stddevs = None # default is not using the measurement stddevs
      if method in ["mahalanobis-variance", "mahalanobis-scaling", "bhattacharyya"]:
        got_latent = "latent_stddev" in latent_extras
        got_pred = "predictor_stddev_denormalised" in pred_extras
        if this_mode == "latent_only":
          if got_latent:
            measurement_stddevs = latent_extras["latent_stddev"]
            pylogging.warning(f"FOUND LATENT VARIANCE with method={method}")
        if this_mode == "errors_only":
          if got_pred:
            # initial shape: B, T, H, N -> to B, T, N (averaging over horizon)
            measurement_stddevs = np.max(pred_extras["predictor_stddev_denormalised"], axis=2)
            pylogging.warning(f"FOUND PREDICTOR VARIANCE with method={method}")  
        if this_mode == "concat":
          if got_latent: latent_stddevs = latent_extras["latent_stddev"]
          else: latent_stddevs = np.zeros_like(latents)
          if got_pred: pred_stddevs = np.max(pred_extras["predictor_stddev_denormalised"], axis=2)
          else: pred_stddevs = np.zeros_like(joint_diffs)
          if got_latent or got_pred:
            measurement_stddevs = np.concatenate([latent_stddevs, pred_stddevs], axis=-1)
            pylogging.warning(f"FOUND: LATENT ({got_latent}) AND PREDICTOR VARIANCE ({got_pred}) "
                              f"(mode=concat) with method={method}")  

      if use_reference and reference_type != "auto-reference-study":
        if reference_type == "computed":
          if not isinstance(reference_obj, dict) and "method" in reference_obj:
            raise RuntimeError(f"Agent_Double_Discriminator.get_measurement_from_trajectory() error: "
                               f"reference_type=computed, but reference obj is not a reference dict")
          # check this double mode has been correctly pre-computed
          if this_mode not in reference_obj:
            raise RuntimeError(f"Agent_Double_Discriminator.get_measurement_from_trajectory() error: "
                               f"reference_type=computed, but reference dict does not contain our "
                               f"current double_mode={this_mode}. It has keys: {[key for key in reference_obj]}")
          # take the reference dictionary for this double mode
          reference_dict = reference_obj[this_mode]
          pylogging.info(f"Reference dict has been passed directly")
        else:
          if reference_type == "auto":
            args = { "reference_latents" : measure_vec[:, :reference_auto_num_samples]}
            pylogging.info(f"Reference being auto generated from first "
                          f"{reference_auto_num_samples} samples")
          else:
            if reference_type == "trajectory":
              if not isinstance(reference_obj, dict) and "joint_angles" in reference_obj:
                raise RuntimeError(f"Agent_Double_Discriminator.get_measurement_from_trajectory() error: "
                                  f"reference_type=trajectory, but reference obj is not a trajectory")
              args = { "trajectory" : reference_obj }
              pylogging.info(f"Reference trajectory has been passed")
            elif reference_type == "latents":
              if not isinstance(reference_obj, np.ndarray) and len(reference_obj.shape) != 3:
                raise RuntimeError(f"Agent_Double_Discriminator.get_measurement_from_trajectory() error: "
                                  f"reference_type=latents, but reference obj is not valid")
              args = { "reference_latents" : reference_obj }
              pylogging.info(f"Reference latents have been passed directly")

          if method in single_point_methods:
            ref_method = "mean"
          elif method in reference_methods:
            ref_method = method
          else:
            raise RuntimeError(f"Agent_Double_Discriminator.get_measurement_from_trajectory() error: "
                              f"method={method} not recognised when 'use_reference'==True")

          # compute the reference latent information dictionary
          reference_dict = self.get_reference_values(**args, 
                                                     method=ref_method,
                                                     decode=(True if "decode" in latent_args and
                                                             latent_args["decode"] else False),
                                                     pred_args=pred_args,
                                                     latent_args=latent_args,
                                                     use_jax=use_jax,
                                                     # very important! pass args for the recursive call of
                                                     # of this function, both MUST run with same settings
                                                     # for mahalanobis to be valid
                                                     double_mode=this_mode,
                                                     joint_diff_method=joint_diff_method,
                                                     # experimental features
                                                     reference_stddev=measurement_stddevs,
                                                     )

        measure_vec = self.latent_discrim.calculate_measurement(measure_vec, 
                                                                reference_dict,
                                                                method=method,
                                                                # experimental features
                                                                reference_stddev=measurement_stddevs)
        
        # # apply post-procesing and build up our list of measurements
        # measure_vec = self.pred_discrim.timestep_rolling_average(measure_vec, 
        #                                                          window_size=window_size)
        # if relative_change:
        #   measure_vec = np.abs(self.latent_discrim.calculate_measurement_change(measure_vec))

        # --- new: confidence additions to measurement --- #
        if (this_mode != "errors_only" and
            "latent_confidence" in latent_extras):
          confidence_score = latent_extras["latent_confidence"]
          if multiply_by_confidence:
            # multiply to 'stabilise' latent estimation
            measure_vec *= confidence_score
            print(f"==== latent confidence multiplied!! range = (min={np.min(confidence_score)}, max={np.max(confidence_score)}) ====")
          if add_confidence_as_measurement:
            # stack for discriminating/plotting using both (i.e. independent thresholds)
            measure_vec = np.stack([measure_vec, confidence_score], axis=2)
            print(f"==== latent confidence added as measurement!! resultant shape = {measure_vec.shape} ====")

        # check if the predictor outputs an uncertainty
        if (this_mode != "latent_only"):
          if "predictor_confidence" in pred_extras:
            pred_confidence = pred_extras["predictor_confidence"]
            if multiply_by_confidence:
              print(f"==== predictor confidence multiplied!! range = (min={np.min(pred_confidence)}, max={np.max(pred_confidence)}) ====")
              measure_vec *= pred_confidence
            if add_confidence_as_measurement:
              # stack for discriminating/plotting using both (i.e. independent thresholds)
              measure_vec = np.stack([measure_vec, pred_confidence], axis=2)
              print(f"==== predictor confidence added as measurement!! resultant shape = {measure_vec.shape} ====")
        # --- end: confidence additions to measurement --- #


        # save the final result
        if measure_vec.ndim == 2:
          measure_vec = np.expand_dims(measure_vec, axis=2)

        final_measurements.append(measure_vec)

      elif use_reference and reference_type == "auto-reference-study":

        # key defining characteristics of the reference study
        num_repeats = reference_study_repeat_num
        num_ref_traj = reference_auto_num_trajectories
        seed = 100

        # ensure recreate same rng in multiple loops of measure_vec
        rng = np.random.default_rng(seed)

        # we need to exclude trajectories where manipulations are occuring
        manip_active = trajectory["manipulation_bool"][:, index:index + measure_vec.shape[1]]

        # if trajectory has invalid data (e.g. dropped frames), exclude from reference trajectories
        if "valid_data" in trajectory:
          data_invalid = ~np.array(trajectory["valid_data"][:, index:index + measure_vec.shape[1]], dtype=bool)
          data_invalid = np.any(data_invalid, axis=2) # any invalid reason along last dimension
          manip_active = np.logical_or(manip_active, data_invalid)

        clean_trajectories = np.argwhere(~np.any(manip_active, axis=1)).reshape(-1)

        pylogging.info(f"Performing a reference study with {num_repeats} repeats, and "
                       f"{num_ref_traj} reference trajectories (out of {len(clean_trajectories)})"
                       f", with seed={seed}, this_mode={this_mode}")

        # prepare to study how the measurement is affected by different references
        if "reference_study" not in double_extras:
          double_extras["reference_study"] = {}
          for n in range(num_repeats):
            double_extras["reference_study"][n] = []

        for n in range(num_repeats):

          # get a new random selection of non manipulated trajectories for a reference
          random_clean_inds = rng.permutation(clean_trajectories)[:num_ref_traj]
          # print(f"Random 'clean' inds for real trajectory reference: {random_clean_inds}"
          #       f", taken {num_ref_traj} / {len(clean_trajectories)} non-manipulated trajectories")

          # take these non-manipulated trajectories as our references
    
          if method in reference_methods:
            ref_method = method
          else:
            raise RuntimeError(f"Agent_Double_Discriminator.get_measurement_from_trajectory() error: "
                               f"method={method} not recognised when 'use_reference'==True"
                               f" (triggered from reference study)")

          args = { "reference_latents" : measure_vec[random_clean_inds] }
          # compute the reference latent information dictionary (should be identical to above!)
          reference_dict = self.get_reference_values(**args, 
                                                    method=ref_method,
                                                    decode=(True if "decode" in latent_args and
                                                            latent_args["decode"] else False),
                                                    pred_args=pred_args,
                                                    latent_args=latent_args,
                                                    use_jax=use_jax,
                                                    # very important! pass args for the recursive call of
                                                    # of this function, both MUST run with same settings
                                                    # for mahalanobis to be valid
                                                    double_mode=this_mode,
                                                    joint_diff_method=joint_diff_method,
                                                    # experimental features
                                                    reference_stddev=measurement_stddevs)
          
          # calculate the resultant measurement from this particular reference
          this_measurement = self.latent_discrim.calculate_measurement(measure_vec, 
                                                                      reference_dict,
                                                                      method=method,
                                                                      # experimental features
                                                                      reference_stddev=measurement_stddevs)
          
          # --- new: confidence additions to measurement --- #
          if (this_mode != "errors_only" and
              "latent_confidence" in latent_extras):
            confidence_score = latent_extras["latent_confidence"]
            if multiply_by_confidence:
              # multiply to 'stabilise' latent estimation
              this_measurement *= confidence_score
              if n == 0: print(f"==== latent confidence multiplied!! range = (min={np.min(confidence_score)}, max={np.max(confidence_score)}) ====")
            if add_confidence_as_measurement:
              # stack for discriminating/plotting using both (i.e. independent thresholds)
              this_measurement = np.stack([this_measurement, confidence_score], axis=2)
              if n == 0: print(f"==== latent confidence added as measurement!! resultant shape = {this_measurement.shape} ====")

          # check if the predictor outputs an uncertainty
          if (this_mode != "latent_only"):
            if "predictor_confidence" in pred_extras:
              pred_confidence = pred_extras["predictor_confidence"]
              if multiply_by_confidence:
                if n == 0: print(f"==== predictor confidence multiplied!! range = (min={np.min(pred_confidence)}, max={np.max(pred_confidence)}) ====")
                this_measurement *= pred_confidence
              if add_confidence_as_measurement:
                # stack for discriminating/plotting using both (i.e. independent thresholds)
                this_measurement = np.stack([this_measurement, pred_confidence], axis=2)
                if n == 0: print(f"==== predictor confidence added as measurement!! resultant shape = {this_measurement.shape} ====")
          # --- end: confidence additions to measurement --- #

          # # apply post-processing
          # this_measurement = self.pred_discrim.timestep_rolling_average(this_measurement, 
          #                                                               window_size=window_size)
          # if relative_change:
          #   this_measurement = np.abs(self.latent_discrim.calculate_measurement_change(this_measurement))
          
          # save the final result
          if this_measurement.ndim == 2:
            this_measurement = np.expand_dims(this_measurement, axis=2)
          double_extras["reference_study"][n].append(this_measurement.copy())

        # take an abritrary example from the study to be returned (for plotting etc)
        final_measurements = double_extras["reference_study"][0]

      else:

        # --- new: confidence additions to measurement --- #
        if (this_mode != "errors_only" and
            "latent_confidence" in latent_extras):
          confidence_score = latent_extras["latent_confidence"]
          if multiply_by_confidence:
            # multiply to 'stabilise' latent estimation
            measure_vec *= confidence_score
            print(f"==== latent confidence multiplied!! range = (min={np.min(confidence_score)}, max={np.max(confidence_score)}) ====")
          if add_confidence_as_measurement:
            # stack for discriminating/plotting using both (i.e. independent thresholds)
            measure_vec = np.stack([measure_vec, confidence_score], axis=2)
            print(f"==== latent confidence added as measurement!! resultant shape = {measure_vec.shape} ====")

        # check if the predictor outputs an uncertainty
        if (this_mode != "latent_only"):
          if "predictor_confidence" in pred_extras:
            pred_confidence = pred_extras["predictor_confidence"]
            if multiply_by_confidence:
              print(f"==== predictor confidence multiplied!! range = (min={np.min(pred_confidence)}, max={np.max(pred_confidence)}) ====")
              measure_vec *= pred_confidence
            if add_confidence_as_measurement:
              # stack for discriminating/plotting using both (i.e. independent thresholds)
              measure_vec = np.stack([measure_vec, pred_confidence], axis=2)
              print(f"==== predictor confidence added as measurement!! resultant shape = {measure_vec.shape} ====")
        # --- end: confidence additions to measurement --- #
     
        # # apply post-procesing and build up our list of measurements
        # measure_vec = self.pred_discrim.timestep_rolling_average(measure_vec, 
        #                                                          window_size=window_size)
        # if relative_change:
        #   measure_vec = np.abs(self.latent_discrim.calculate_measurement_change(measure_vec))

        final_measurements.append(measure_vec)

    # --- #

    # special catch: if done reference study, need to combine measurement vectors
    if use_reference and reference_type == "auto-reference-study":
      for n in range(num_repeats):
        if len(double_extras["reference_study"][n]) > 1:
          double_extras["reference_study"][n] = np.concatenate(double_extras["reference_study"][n], axis=-1)
        else:
          double_extras["reference_study"][n] = double_extras["reference_study"][n][0]

    # finally, prepare the measure vector
    if len(final_measurements) > 1:
      measurements = np.concatenate(final_measurements, axis=-1)
    else:
      measurements = final_measurements[0]

    # # code for completely random measurements per step
    # # recommened to set 'reference_study_repeat_num' to 100, to average over 100
    # print(f"\n\n\nWARNING!!!! COMPLETELY RANDOM MEASUREMENTS!!!!\n\n")
    # B, T, N = final_measurements[0].shape
    # measurements = np.random.rand(B, T, 1) # random measurements
    # if use_reference and reference_type == "auto-reference-study":
    #   for n in range(num_repeats):
    #     double_extras["reference_study"][n] = np.random.rand(B, T, 1)

    if return_extra:
      return measurements, latent_extras | pred_extras | double_extras
    else:
      return measurements

  def get_reference_values(self, trajectory=None, decode=False, method="mahalanobis",
                           reference_latents=None, pred_args={}, latent_args={},
                           precompute_for_double_mode=None, use_jax=False,
                           reference_stddev=None,
                           **get_measurement_args):
    """
    Get reference values for both prediction and discrim
    """

    # pre-compute case (for deployment only), get reference for all double modes
    if precompute_for_double_mode is not None:
      # determine how to combine the two measurements
      if precompute_for_double_mode == "concat":
        modes = ["concat"]
      elif precompute_for_double_mode == "seperate":
        modes = ["latent_only", "errors_only"]
      elif precompute_for_double_mode == "latent_only":
        modes = ["latent_only"]
      elif precompute_for_double_mode == "errors_only":
        modes = ["errors_only"]
      else:
        raise RuntimeError(f"Agent_Double_Discriminator.get_reference_values() error: "
                           f"precompute_for_double_mode={precompute_for_double_mode} not recognised")
    else:
      # default case, modes are handled in 'get_measurement_from_trajectory'
      modes = [None]

    final_ref_dict = {}
    ref_latents_passed = True if reference_latents is not None else False

    for this_mode in modes:
  
      # generate reference latents if given a trajectory
      if trajectory is not None and not ref_latents_passed:
        # instead of latents, it is latents and average joint errors
        reference_latents = self.get_measurement_from_trajectory(
          trajectory=trajectory,
          computing_reference=True, # preserve last_index/view
          **get_measurement_args,
          index=None,
          num_measurements=None,
          double_mode=this_mode if this_mode is not None else "concat",
          use_jax=use_jax,
          use_reference=False,
          return_extra=False,
          window_size=1,
          pred_args=pred_args,
          latent_args=latent_args,
        )

      # otherwise, use latents already passed in
      elif reference_latents is not None:
        if trajectory is not None:
          pylogging.warning(f"trajectory != None even though reference_latents "
                            f"have been passed. Ignoring the trajectory")
          
      else:
        raise RuntimeError(f"Either 'trajectory' or 'reference_latents' must be set")
      
      # get the reference dict
      ref_dict = self.latent_discrim.get_reference_values(
        reference_latents=reference_latents,
        reference_stddev=reference_stddev,
        decode=decode,
        method=method,
        use_jax=use_jax,
      )

      # eval case, we don't prepare for any particular modes
      if this_mode is None:
        final_ref_dict = ref_dict
      # pre-computed deployment case, prepare for any required double modes
      else:
        final_ref_dict[this_mode] = ref_dict

    return final_ref_dict
