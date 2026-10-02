# python imports
import logging; logging.basicConfig(
    level=logging.INFO); pylogging = logging.getLogger(__name__)
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import textwrap
import itertools


import math
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import time
import einops
import jax.numpy as jnp
import jax.lax
from jax.typing import ArrayLike as JaxArrayLike

# repo imports
from aware.utils.trajectory import deep_dict_update, load_trajectory
from aware.utils.training import calculate_average_precision, calculate_auroc, cross_validate_best_f1
from aware.utils.logger import create_table, fig_to_img
from aware.agents.prediction_discriminator import Agent_Prediction_Discriminator
from typing import Dict, List

# get the location of this file
from aware import REPO_ROOT as path_to_root

class DiscriminatorEval:
  """
  Class for evaluating latent and motion predictor discriminators.
  """

  # ---- measurement methods ---- #

  def per_measurement_thresholding_classification(self, 
                                                  measurements, 
                                                  thresholds, 
                                                  manipulation_active, 
                                                  combine_mode="none",
                                                  signal_high_steps=1,
                                                  use_jax=True):
    """
    Performs a grid search over a set of thresholds for multiple measurements to classify 
    trajectories.

    For each combination of thresholds, it classifies every timestep and then evaluates
    performance on a per-trajectory basis (e.g., TP, TN, FP, FN).
    """

    t0 = time.perf_counter()

    if use_jax:
      lib = jnp
    else:
      lib = np

    if measurements.ndim == 2:
      measurements = lib.expand_dims(measurements, axis=2) # from [B, T] -> [B, T, 1]

    # if we combine measurements by adding them, collapse to one measurement
    if combine_mode == "sum":
      measurements = lib.sum(measurements, axis=-1, keepdims=True)
      pylogging.info(f"measure_combine_mode=sum, shape={measurements.shape}")

    # repeats, batch, timesteps, measurements
    R, B, T, num_measurements = measurements.shape

    gt_positive = lib.any(manipulation_active, axis=1)
    gt_negative = ~gt_positive

    # pre-broadcast to handle additional R dimension
    gt_positive_b = lib.expand_dims(gt_positive, axis=0) # (B) -> (1, B)
    gt_negative_b = lib.expand_dims(gt_negative, axis=0) # (B) -> (1, B)
    manipulation_active_b = manipulation_active[None, :, :] # (B, T) -> (1, B, T)    
    not_manipulation_active_b = ~manipulation_active_b      # (B, T) -> (1, B, T)

    # each threshold gets a dimension, corresponding that the order of measurements
    results_shape = [len(thresholds)] * num_measurements + [R, B] # result per trajectory
    true_pos = np.zeros(results_shape, dtype=bool)
    true_neg = np.zeros(results_shape, dtype=bool)
    false_pos = np.zeros(results_shape, dtype=bool)
    false_neg = np.zeros(results_shape, dtype=bool)

    # create an easy lookup to convert a value to its index in thresholds
    value_to_index = {value: i for i, value in enumerate(thresholds)}

    # loop over every single combination of thresholds (num_thresholds ^ num_measurements)
    combinations = itertools.product(thresholds, repeat=num_measurements)

    for threshold_tuple in combinations:

      # determine the indexes that correspond to this combination of thresholds
      indexes = tuple(value_to_index[value] for value in threshold_tuple)

      # check each measurement against it's corresponding thresholds at this iteration
      all_classifications = lib.abs(measurements) > lib.array(threshold_tuple) # shape R, B, T, N

      # perform a rolling average 'AND', so measurements must stay high to count
      all_classifications = self.rolling_all(all_classifications, window_size=signal_high_steps,
                                             axis=2, use_jax=use_jax)

      # now determine how to combine into a final judgement
      if all_classifications.shape[3] > 1:
        # we must decide how to combine our thresholded classifications
        if combine_mode == "none":
          # take only the first measurement (no combining)
          classified = all_classifications[:, :, :, 0]
        elif combine_mode == "or":
          # any measurements were above their threshold
          classified = lib.any(all_classifications, axis=3) # shape R, B, T
        elif combine_mode == "and":
          # all measurements were above their threshold
          classified = lib.all(all_classifications, axis=3) # shape R, B, T
        # elif combine_mode.startswith("trust_order"):
        #   order = [int(x) for x in combine_mode.split("-")[2:]]
        #   classified = lib.zeros(all_classifications.shape[:2])
        else:
          raise RuntimeError(f"DiscriminatorEval.per_measurement_thresholding_classifier() error: "
                              f"combine_mode={combine_mode} not recognised")
      else:
        # we only have measurement, so reduce redundant dimension (R, B, T, 1) -> (R, B, T)
        classified = lib.squeeze(all_classifications, axis=-1)

      # now check along the timestep dimension, if classification co-incides with manipulation
      # classified is (R, B, T). manipulation_active_b is (1, B, T)

      # detection masks: inside manipulation window, outside, and across whole trajectory
      correct_detection = lib.any(classified & manipulation_active_b, axis=2) # Shape (R, B)
      has_false_alarms = lib.any(classified & not_manipulation_active_b, axis=2) # Shape (R, B)
      has_any_detection = lib.any(classified, axis=2) # Shape (R, B)
      
      # Now combine (R, B) with (1, B) ground truth
      
      # true positive: trajectories with manipulation, correct detection, no false alarms
      true_pos[indexes] = (gt_positive_b & correct_detection & ~has_false_alarms)
      
      # true negative: trajectories without manipulation, no detections
      true_neg[indexes] = (gt_negative_b & ~has_any_detection)
      
      # false positive: detection in trajectory without manipulation, or a false alarm
      false_pos[indexes] = (
          (gt_negative_b & has_any_detection) |  # Clean trajectories with any detections
          (gt_positive_b & has_false_alarms)     # Manipulation trajectories with false alarms
      )

      # false negative: trajectories with manipulation, no correct detection, no false alarms
      false_neg[indexes] = (gt_positive_b & ~correct_detection & ~has_false_alarms)

    t1 = time.perf_counter()

    pylogging.debug(f"Time taken for threshold classification = {t1 - t0:.3f}s"
                    f" (use_jax={use_jax})")

    return true_pos, true_neg, false_pos, false_neg

  def calculate_results(self, measurements, manipulation_active, maximise="f1", 
                        max_threshold="auto", num_thresholds=100,
                        threshold_balancing=False, measure_combine_mode="and",
                        signal_high_steps=1, average_type="none", window_size=1,
                        exponential_average_alpha=0.1, ref_study=False,
                        cross_validation_n=5,
                        use_jax=True):
    """
    Get the final results of the discrimination test
    """

    # handle input dimensions
    if measurements.ndim == 2: # shape (B, T) -> (1, B, T, 1)
      measurements = np.expand_dims(measurements, axis=(0, 3))
    elif measurements.ndim == 3 and not ref_study: # shape (B, T, N)
      measurements = np.expand_dims(measurements, axis=0)
    elif measurements.ndim == 3 and ref_study: # shape (R, B, T)
      measurements = np.expand_dims(measurements, axis=3)
    elif measurements.ndim != 4: # shape (R, B, T, N), perfect
      raise RuntimeError(f"DiscriminatorEval.calculate_results() error: "
                         f"measurements.shape = {measurements.shape}")

    # now we have this correct shape (R=repeats, B=batch, T=timesteps)
    R, B, T, num_measurements = measurements.shape

    if threshold_balancing and num_measurements > 1:
      num_thresholds_original = num_thresholds
      num_thresholds = int(np.ceil(np.power(num_thresholds, 1.0 / num_measurements)))
      pylogging.warning(f"theshold_balancing=True, and num_thresholds="
                        f"{num_thresholds_original}, that means using "
                        f"{num_thresholds} thresholds for each of {num_measurements} measurements")

    # determine the thresholds
    if max_threshold == "auto":
      percentile_measurement = np.nanpercentile(np.abs(measurements), 97.5)
      thresholds = np.linspace(start=0.0, stop=percentile_measurement, num=num_thresholds - 1)
      thresholds = np.append(thresholds, 1.01 * np.nanmax(np.abs(measurements)))

    elif max_threshold == "auto-full-negative":
      min_score = np.nanmin(measurements)
      max_score = np.nanmax(measurements)
    
      thresholds = np.linspace(start=min_score, stop=max_score, num=num_thresholds)
      thresholds[-1] = max_score + 1e-9

    else:
      thresholds = np.linspace(start=0.0, stop=max_threshold, num=num_thresholds)

    if use_jax:
      measurements = jnp.array(measurements)
      manipulation_active = jnp.array(manipulation_active)

    # do we apply averaging along the time dimension
    if average_type != "none":
      measurements = self.rolling_average(measurements, window_size=window_size,
                                          average_type=average_type, use_jax=use_jax,
                                          alpha=exponential_average_alpha, axis=2)
    
    # evaluate the discrimination success using the thresholds (handles any num measurements)
    true_pos, true_neg, false_pos, false_neg = self.per_measurement_thresholding_classification(
      measurements=measurements, 
      thresholds=thresholds, 
      manipulation_active=manipulation_active,
      combine_mode=measure_combine_mode,
      signal_high_steps=signal_high_steps,
      use_jax=use_jax,
    )

    # these output with shape (Th1, ..., ThN, R, B), so sum over B dimension (axis=-1)
    # true_pos.shape = [len(thresholds)] * num_measurements + [R, B]
    tp_sum = true_pos.sum(axis=-1)
    tn_sum = true_neg.sum(axis=-1)
    fp_sum = false_pos.sum(axis=-1)
    fn_sum = false_neg.sum(axis=-1)

    # All metrics will have shape (Th1, ..., ThN, R)
    precision = tp_sum / (tp_sum + fp_sum + 1e-10)
    recall = tp_sum / (tp_sum + fn_sum + 1e-10)
    f1 = 2 * ((precision * recall) / (precision + recall + 1e-10))
    success_rate = (tp_sum + tn_sum) / B
    
    # determine which criteria we should find the maximum of
    if maximise.lower() == "f1":
      results = f1
    elif maximise.lower() == "sr":
      results = success_rate
    elif maximise.lower() == "precision":
      results = precision
    elif maximise.lower() == "recall":
      results = recall
    else:
      raise RuntimeError(f"DiscriminatorEval.calculate_results() error: "
                         f"maximise={maximise} not recognised.")
    
    # Find best arg *per study (R)*
    threshold_shape = results.shape[:-1] # (Th1, ..., ThN)
    threshold_dims_flat = np.prod(threshold_shape)

    # Reshape to (FlatThresholds, R) and find argmax for each R
    argmax_flat_per_study = np.argmax(
        results.reshape(threshold_dims_flat, R), axis=0
    ) # Shape (R,)
    
    # Convert flat indices back to tuples
    # Shape (N, R)
    best_arg_per_study = np.array(
        np.unravel_index(argmax_flat_per_study, threshold_shape)
    ) 
    
    # Shape (N, R)
    best_threshold_per_study = thresholds[best_arg_per_study]
      
    # Helper to extract the best value for each study
    def get_best_per_study(metric_array, flat_indices):
      # metric_array is (Th1, ..., ThN, R)
      # flat_indices is (R,)
      flat_metric = metric_array.reshape(threshold_dims_flat, R)
      # Use take_along_axis, adding a new axis to flat_indices
      return np.take_along_axis(flat_metric, flat_indices[None, :], axis=0).squeeze(axis=0)

    # All are shape (R,)
    best_f1_per_study = get_best_per_study(f1, argmax_flat_per_study)
    best_precision_per_study = get_best_per_study(precision, argmax_flat_per_study)
    best_recall_per_study = get_best_per_study(recall, argmax_flat_per_study)
    best_success_rate_per_study = get_best_per_study(success_rate, argmax_flat_per_study)
    final_result_per_study = get_best_per_study(results, argmax_flat_per_study)

    # Calculate average precision per study (looping is safest if func isn't vectorized)
    avg_precision_per_study = []
    avg_auroc_per_study = []
    avg_xval_f1_per_study = []
    avg_xval_sr_per_study = []
    for i in range(R):
      # precision[..., i] selects all threshold dims for the i-th study
      avg_precision_per_study.append(
        calculate_average_precision(precision[..., i], recall[..., i])
      )
      # true_pos shape = (Th1, Th2, ..., ThN, R, B), so index R
      avg_auroc_per_study.append(
        calculate_auroc(true_pos[..., i, :], true_neg[..., i, :], false_pos[..., i, :], false_neg[..., i, :])
      )
      # get metrics based on cross validation
      xval_metrics = cross_validate_best_f1(true_pos[..., i, :], true_neg[..., i, :], false_pos[..., i, :], false_neg[..., i, :],
                                            n_splits=cross_validation_n)
      avg_xval_f1_per_study.append(xval_metrics["f1"])
      avg_xval_sr_per_study.append(xval_metrics["success_rate"])

    avg_precision_per_study = np.array(avg_precision_per_study) # Shape (R,)
    avg_auroc_per_study = np.array(avg_auroc_per_study)
    avg_xval_f1_per_study = np.array(avg_xval_f1_per_study)
    avg_xval_sr_per_study = np.array(avg_xval_sr_per_study)

    # index in repeat to use
    index_to_use = 0

    best_arg = tuple(best_arg_per_study[:, index_to_use]) # (N, R) -> (N,) -> tuple
        
    res_info = {
      # Key summary metrics (scalar) for this *single* repeat
      "threshold" : best_threshold_per_study[:, index_to_use], # (N, R) -> (N,)
      "average_precision" : avg_precision_per_study[index_to_use], # (R,) -> scalar
      "auroc" : avg_auroc_per_study[index_to_use], # (R,) -> scalar
      "f1" : avg_xval_f1_per_study[index_to_use],
      "old_f1" : best_f1_per_study[index_to_use],
      "precision" : best_precision_per_study[index_to_use],
      "recall" : best_recall_per_study[index_to_use],
      "success_rate" : avg_xval_sr_per_study[index_to_use], 
      "old_success_rate" : best_success_rate_per_study[index_to_use],
      
      # Raw data for this *single* repeat
      "data" : {
        "best_arg" : best_arg,
        "maximise" : maximise,
        "measurements" : np.array(measurements[index_to_use]), # (R,B,T,N) -> (B,T,N)
        "num_measurements" : num_measurements,
        "threshold" : thresholds,
        
        # (Th1, ..., ThN, R) -> (Th1, ..., ThN)
        "f1" : f1[..., index_to_use], 
        "precision" : precision[..., index_to_use],
        "recall" : recall[..., index_to_use],
        "success_rate" : success_rate[..., index_to_use],
        
        # (Th1, ..., ThN, R, B) -> (Th1, ..., ThN, B)
        "true_pos" : true_pos[..., index_to_use, :], 
        "true_neg" : true_neg[..., index_to_use, :],
        "false_pos" : false_pos[..., index_to_use, :],
        "false_neg" : false_neg[..., index_to_use, :],
      },
    }

    if ref_study:
      
      # 2. *Overwrite* top-level metrics with the aggregated *means*
      res_info["average_precision"] = np.mean(avg_precision_per_study)
      res_info["auroc"] = np.mean(avg_auroc_per_study)
      res_info["f1"] = np.mean(avg_xval_f1_per_study)
      res_info["old_f1"] = np.mean(best_f1_per_study)
      res_info["sr"] = np.mean(avg_xval_sr_per_study)
      res_info["old_sr"] = np.mean(best_success_rate_per_study)
      res_info["precision"] = np.mean(best_precision_per_study)
      res_info["recall"] = np.mean(best_recall_per_study)

      # 3. *Add* aggregated stddevs
      res_info["average_precision_std"] = np.std(avg_precision_per_study)
      res_info["auroc_std"] = np.std(avg_auroc_per_study)
      res_info["f1_std"] = np.std(best_f1_per_study)
      res_info["sr_std"] = np.std(best_success_rate_per_study)
      res_info["precision_std"] = np.std(best_precision_per_study)
      res_info["recall_std"] = np.std(best_recall_per_study)
      
      # 4. *Add* the raw per-threshold, per-study data
      res_info['threshold_data'] = {
        'precision': precision,                       # (Th1, ..., ThN, R)
        'recall': recall,                             # (Th1, ..., ThN, R)
        'true_pos': true_pos,                         # (Th1, ..., ThN, R, B)
        'true_neg': true_neg,                         # (Th1, ..., ThN, R, B)
        'false_pos': false_pos,                       # (Th1, ..., ThN, R, B)
        'false_neg': false_neg,                       # (Th1, ..., ThN, R, B)
        'best_arg': best_arg_per_study,               # (N, R)
        'average_precision': avg_precision_per_study  # (R,)
      }
    
    return res_info

  def measure_discrimination(self, agent, trajectory=None, 
                             num_thresholds=100, max_threshold="auto", 
                             discrim_locks_high=False, start_index=None,
                             measure_combine_mode="and", threshold_balancing=False,
                             signal_high_steps=1, average_type="none",
                             averaging_window_size=1, return_raw=False, use_jax=True,
                             exponential_average_alpha=0.1, cross_validation_n=5,
                             skip_measurement=False, invalid_data_mode=None,
                             measure_args={}):
    """
    Measure the discrimination performance of an agent over a trajectory which
    contains manipulations, indicated by the trajectory['manipulation_bool'] field.
    """

    timer_fcn = time.perf_counter
    t0 = timer_fcn()

    if return_raw:
      measurements, info = agent.get_measurement_from_trajectory(
        trajectory=trajectory,
        return_extra=True,
        index=start_index,
        **measure_args,
      )
    else:
      measurements = agent.get_measurement_from_trajectory(
        trajectory=trajectory,
        return_extra=False,
        index=start_index,
        **measure_args,
      )
      info = {} # no extra info returned from above
    
    if skip_measurement:
      return info

    t1 = timer_fcn()
    # clip the manipulation bool vector to be aligned with the measurements
    m_start = agent.last_index
    m_end = agent.last_index + measurements.shape[1]
    manip_vec = trajectory["manipulation_bool"][:, m_start : m_end]

    if invalid_data_mode is not None and "valid_data" in trajectory:
      pylogging.info(f"Checking for invalid data in DiscriminatorEval.measure_discrimination()"
                     f", taking average over all invalid features (state/slew+luff/hoist).\n"
                     f"invalid_data_mode = {invalid_data_mode}")
      valid_vec = np.all(trajectory["valid_data"][:, m_start : m_end], axis=2)
    if invalid_data_mode == "manipulation":
      manip_vec = np.logical_or(manip_vec, ~valid_vec)
    
    # widen the manipulation step wave since the end of manipulation is in view for a while
    safety_padding = 5 # extra padding to allow agent to classify manipulation as having occured
    if discrim_locks_high:
      end_later = m_end # stays high for whole rest of trajectory
    else:
      end_later = agent.last_view + safety_padding # low when manipulated samples aren't in history
    manip_vec_wide = self.widen_manipulation_vector(manip_vec,
                                                    start_earlier=0 + safety_padding, 
                                                    end_later=end_later)    
    manipulation_active = manip_vec_wide.astype(int) # convert from float to int

    if invalid_data_mode in ["zero_narrow", "zero"] and "valid_data" in trajectory:
      start_padding = 0 # we only mark invalid if we have 10 dropped samples
      end_padding = 0 # no end padding needed in theory
      if invalid_data_mode == "zero":
        end_at = agent.last_view + end_padding
      elif invalid_data_mode == "zero_narrow":
        end_at = 0
      invalid_vec_wide = self.widen_manipulation_vector(~valid_vec,
                                                        start_earlier=0 + start_padding,
                                                        end_later=end_at)
      valid_vec_wide = ~np.expand_dims(np.array(invalid_vec_wide, dtype=bool), axis=2)
    else: valid_vec_wide = None

    # are we studying a variety of different measurements in a reference study
    if return_raw and "reference_study" in info:

      # stack measurements into one large tensor
      # Get the keys (0, 1, 2, ...) and sort them to guarantee order
      sorted_keys = sorted(info["reference_study"].keys())

      # Build a list of the arrays, in order
      arrays_to_stack = [info["reference_study"][key] for key in sorted_keys]

      # Now, stack the list of arrays
      stacked_measurements = np.stack(arrays_to_stack, axis=0)
      num = stacked_measurements.shape[0] # This is R (e.g., 10)
      mnum = stacked_measurements.shape[3] # this is N, number of measurements

      if valid_vec_wide is not None:
        valid_vec_wide_repeat = einops.repeat(valid_vec_wide, "b t 1-> r b t n", r=num, n=mnum)
        stacked_measurements *= valid_vec_wide_repeat

      # Make one single, vectorized call
      res_info = self.calculate_results(
          stacked_measurements, # Shape (R, B, T, N)
          manipulation_active, 
          maximise="f1", 
          max_threshold=max_threshold, 
          num_thresholds=num_thresholds,
          threshold_balancing=threshold_balancing,
          measure_combine_mode=measure_combine_mode, 
          signal_high_steps=signal_high_steps,
          average_type=average_type,
          window_size=averaging_window_size,
          exponential_average_alpha=exponential_average_alpha,
          cross_validation_n=cross_validation_n,
          use_jax=use_jax,
          ref_study=True # indicate we are performing a reference study
      )

      # extract a processed measurement (typically from [-1] in study), for plotting
      measurements = res_info["data"]["measurements"]

      if valid_vec_wide is not None:
        measurements *= valid_vec_wide

      pylogging.info(
          f"After a reference study with num={num}, the averaged values are:\n"
          f""" -> average_precision = {res_info["average_precision"]:.3f} (stddev = {res_info["average_precision_std"]:.3f})\n"""
          f""" -> auroc = {res_info["auroc"]:.3f} (stddev = {res_info["auroc_std"]:.3f})\n"""
          f""" -> f1 = {res_info["f1"]:.3f} (stddev = {res_info["f1_std"]:.3f})\n"""
          f""" -> sr = {res_info["sr"]:.3f} (stddev = {res_info["sr_std"]:.3f})\n"""
          f""" -> precision = {res_info["precision"]:.3f} (stddev = {res_info["precision_std"]:.3f})\n"""
          f""" -> recall = {res_info["recall"]:.3f} (stddev = {res_info["recall_std"]:.3f})\n"""
      )

    # normal case: we have one measurement
    else:

      if manip_vec_wide is not None:
        measurements *= valid_vec_wide

      # determine how well the discriminator has performed
      res_info = self.calculate_results(
        measurements, 
        manipulation_active, 
        maximise="f1", 
        max_threshold=max_threshold, 
        num_thresholds=num_thresholds,
        threshold_balancing=threshold_balancing,
        measure_combine_mode=measure_combine_mode, 
        signal_high_steps=signal_high_steps,
        average_type=average_type,
        window_size=averaging_window_size,
        exponential_average_alpha=exponential_average_alpha,
        use_jax=use_jax,
        ref_study=False,
      )

      pylogging.info(
        f"After a single evaluation of discrimination accuracy:\n"
        f""" -> average_precision = {res_info["average_precision"]:.3f}\n"""
        f""" -> auroc = {res_info["auroc"]:.3f}\n"""
        f""" -> f1 = {res_info["f1"]:.3f}\n"""
        f""" -> sr = {res_info["success_rate"]:.3f}\n"""
        f""" -> precision = {res_info["precision"]:.3f}\n"""
        f""" -> recall = {res_info["recall"]:.3f}\n"""
      )

    t2 = timer_fcn()
    discrim_time = t1 - t0
    time_per = discrim_time / (measurements.shape[1])
    freq = 1 / time_per
    pylogging.info(f"Time for discrimination = {discrim_time:.3f}s, "
                   f"time per point (T={measurements.shape[1]}) = {time_per * 1e3:.3f}ms"
                   f", model frequency = {freq:.1f}Hz. Time for evaluation code to "
                   f"run = {t2 - t1:.3f}s. Note: batch size (B={measurements.shape[0]})"
                   f" not considered in this approximate analysis.")

    # return info and any extra info in 'extra'
    return info | res_info | {

      # log the time taken for the measurements to be made
      "total_time" : discrim_time,
      "approx_time_per_point" : time_per,
      "approx_frequency" : freq,

      # raw data only returned if specified
      "measurements" : measurements if return_raw else None,
      "manipulation_active" : manipulation_active if return_raw else None,
      "valid_vec" : valid_vec_wide if return_raw else None,
      "m_start" : m_start if return_raw else None,
      "m_end" : m_end if return_raw else None,
    }

  def default_evaluation(self, double_discrim, eval_name, trajectory_name=None,
                         trajectory=None, traj_label=None, start_index=100, 
                         double_mode="seperate", eval_double=False, eval_latent=True, 
                         eval_predictor=True, print_out=True, discrim_args=None, 
                         use_jax=None, classification_args=None):
    """
    Perform the default evaulation of a discriminator (which should be loaded into
    the double discriminator class).

    Inputs:
      double_discrim: the double discriminator to evaluate
      eval_name: name to give this evaluation
      trajectory_name: name of trajectory to load from 'eval_trajectories'
      trajectory: trajectory to evaluate. Must be provided if no 'trajectory_name'
      traj_label: name to give trajectory in table of results if 'print_out=True'
      start_index: index in the trajectory to begin measurement
      double_mode: method for combining measurements in the full double discrim
      eval_double: do we evaluate the double mode
      eval_latent: do we evaluate the latent discrimination
      eval_predictor: do we evaluate the prediction discirmination
      print_out: print a table summarising the comparisons
      discrim_args: arguments to pass into the double discriminator
      classification_args: arguments to control thresholds and classification

    Important! Default discrim_args and classification_args are hardcoded below.
    """

    # define default arguments into the discriminator
    default_discrim_args = {
      "double_mode" : double_mode,
      "method" : "mahalanobis",
      "use_jax" : use_jax if use_jax is not None else False,
      "use_reference" : True,
      "reference_type" : "auto-reference-study",
      "reference_auto_num_trajectories" : 10,
      "reference_study_repeat_num" : 10,
      "window_size" : 1,
      "joint_diff_method" : "average",
      "pred_args" : {
        "horizon" : 20,
        "position" : True,
        "velocity" : True,
        "prediction_args" : {
          "predict_latents_only_once" : True,
        },
      },
      "latent_args" : {
      }
    }

    # define default arguments into the threshold classification
    default_classification_args = {
      "num_thresholds" : 1000, # with threshold balancing, 1000 for 1 measurement, 32 * 32 for 2
      "max_threshold" : "auto",
      "threshold_balancing" : True, # spread num_thresholds out when doing multiple measurements
      "measure_combine_mode" : "or",
      "signal_high_steps" : 1,
      "average_type" : "expo", # 'mean', 'median', 'expo'
      "averaging_window_size" : 1,
      "exponential_average_alpha" : 0.1, # for average_type="expo"
      "cross_validation_n" : 5, # N-fold cross validation for F1 score
      "use_jax" : use_jax if use_jax is not None else True,
      "invalid_data_mode" : None,
    }

    # account for any overrides
    if discrim_args is not None:
      discrim_args = deep_dict_update(default_discrim_args, discrim_args, strict=False)
    else:
      discrim_args = default_discrim_args

    if classification_args is not None:
      classification_args = deep_dict_update(default_classification_args, classification_args,
                                             strict=False)
    else:
      classification_args = default_classification_args
  
    # load the desired trajectory
    if trajectory is None:
      if trajectory_name is None:
        raise RuntimeError(f"DiscriminatorEval.default_evaluation() error: "
                           f"either provide a trajectory, or a trajectory name to load "
                           f"from eval_trajectories. The default fallback has been removed")
      trajectory = load_trajectory(f"{path_to_root}/eval_trajectories/{trajectory_name}")

    name_extensions = []
    double_modes = []
    output_data = {}

    has_latents = not double_discrim.dummy_latents
    has_predictor = double_discrim.pred_discrim.get_prediction_from_trajectory is not None

    if eval_double:
      if has_latents and has_predictor:
        name_extensions.append("double")
        double_modes.append(double_mode)
      elif has_latents:
        name_extensions.append("latent")
        double_modes.append("latent_only")
      elif has_predictor:
        name_extensions.append("predictor")
        double_modes.append("errors_only")

    if eval_latent and has_latents:
      name_extensions.append("latent")
      double_modes.append("latent_only")
    
    if eval_predictor and has_predictor:
      name_extensions.append("predictor")
      double_modes.append("errors_only")
    
    # create a table to print to summarise the comparison if requested
    if print_out:
      # create a table to summarise results
      label_columns = ["Name", "Type"]
      if traj_label is not None: label_columns += ["Trajectory"]
      data_columns = ["AP", "AUROC", "F1 score", "SR", "Hz"]
      min_cell_width = 8
      header_str, row_str = create_table(
        widths=[len(eval_name), len("predictor")] 
                + ([max(len("Trajectory"), len(traj_label))] if traj_label is not None else [])
                + [max(min_cell_width, len(x)) for x in data_columns], 
        types=[str for x in label_columns] + [float for x in data_columns], 
        float_fmt=".3f"
      )
      table_str = """\n--- Evaluation results table ---\n"""
      table_str += header_str.format(*label_columns, *data_columns)

    for i in range(len(double_modes)):

      this_name = f"{eval_name} {name_extensions[i]}"
      
      if print_out:
        pylogging.info(f"default_evaluation for: {this_name}")
      
      output_data[name_extensions[i]] = self.plot_measurements(
        agent=double_discrim,
        trajectory=trajectory,
        start_index=start_index,
        return_data=True,
        envs=None,
        seed=None,
        sharey=False,
        discrim_locks_high=False,
        discrim_args=discrim_args | { "double_mode" : double_modes[i] },
        classification_args=classification_args,
      )

      output_data[name_extensions[i]]["model"] = eval_name
      output_data[name_extensions[i]]["eval_name"] = this_name

      if print_out:
        labels = [eval_name, name_extensions[i]]
        if traj_label is not None: 
          labels += [traj_label]
        this_row = row_str.format(
          *labels,
          output_data[name_extensions[i]]["average_precision"],
          output_data[name_extensions[i]]["auroc"],
          output_data[name_extensions[i]]["f1"],
          output_data[name_extensions[i]]["success_rate"],
          output_data[name_extensions[i]]["approx_frequency"],
        )
        table_str += this_row

    if print_out:
      pylogging.info(table_str)

    return output_data

  # --- plotting functions --- #

  def plot_measurements(self, 
                        discrim:Dict=None, 
                        envs:List=None, 
                        sharey:bool=False, 
                        start_index:int=None,
                        measurements_to_plot:int=None,
                        measurement_names:List=None,
                        seed:int=123,
                        show_dr_data:bool=False,
                        custom_env_labels:List=None,
                        custom_operator_labels:List=None, 
                        plot_stacked_component_error:bool=False,
                        stacked_joint_indicies:List=None,
                        stacked_joint_labels:List=None, 
                        agent:Agent_Prediction_Discriminator=None,
                        trajectory:Dict=None,
                        return_data:bool=False,
                        title_prefix:str=None,
                        default_env_grid_num:int=4,
                        discrim_args={},
                        classification_args={},
                        **other_args):
    """
    Plots discrimination measurements, showing the discrimination signal vs the manipulation bool and calculated threshold. 
    If discrimination data dict is not provided and agent can be passed to compute the data.

    Args:
      discrim (Dict): data returned from measure_discrimination containing the threshold, F1, etc.. to plot.
      envs (List): environments (batch axis) to plot.
      sharey (bool): share y axis across all plots.
      measurements_to_plot (int): if discriminating using multiple measurements, the index of the measurement to plot. Will plot all measurements if None.
      measurement_names (List): list of names for each measurement to be plot
      seed (int): seed for randomly picking environments to plot if envs is not specified.
      show_dr_data (bool): label with domain randomised data (payload mass, gains, etc) in plots.
      custom_env_labels (List): labels for each environment index to show on plot.
      custom_operator_labels (List) labels for each operator to show on plot.
      plot_stacked_component_error (bool): plot as a stack plot the per component (i.e. per joint) error. 
      stacked_joint_indicies (List): list of joint indexes to plot in the stacked plot.
      stacked_joint_labels (List): list of joint labels per index to plot in the stacked plot.
      agent (Agent_Prediction_Discriminator): agent to generate discrimination data using measure_discrimination. 
      trajectory (Dict): trajectory to discriminate with. 
      return_data (bool): returns the discrimination data.
      title_prefix: adds text to the start of the title
      default_env_grid_num: when env=None, a grid of how many plots should be shown
    """

    if discrim is None:
      assert agent is not None, 'agent required to generate discrimination measurements'
      # evaluate the performance of the discriminator
      discrim = self.measure_discrimination(
        trajectory=trajectory,
        agent=agent,
        start_index=start_index,
        return_raw=True,
        measure_args=discrim_args,
        **classification_args,
        **other_args,
      )
      discrim["measure_args"] = discrim_args


    if classification_args.get('skip_measurement', False):
      return discrim

    # needed for data introspection later
    best_arg = discrim["data"]["best_arg"]
    total = discrim['measurements'].shape[0]

    # if not told which environments to plot, get examples of all cases
    if envs is None:
      if seed is None: seed = np.random.randint(0, 100_000_000)
      rng = np.random.RandomState(seed)
      true_pos_inds = rng.permutation(np.argwhere(discrim['data']['true_pos'][best_arg] > 0.5))
      true_neg_inds = rng.permutation(np.argwhere(discrim['data']['true_neg'][best_arg] > 0.5))
      false_pos_inds = rng.permutation(np.argwhere(discrim['data']['false_pos'][best_arg] > 0.5))
      false_neg_inds = rng.permutation(np.argwhere(discrim['data']['false_neg'][best_arg] > 0.5))
           
      # assemble the structured exxamples
      x = default_env_grid_num # num of each to show
      envs = np.concatenate([
        true_pos_inds[:x],
        true_neg_inds[:x],
        false_pos_inds[:x],
        false_neg_inds[:x],
      ]).reshape(-1) # change shape from (1, X) -> (X)

    # get only the indexes we care about
    if not isinstance(envs, (list, tuple, np.ndarray)):
      envs = [envs]
    envs = np.array(envs, dtype=int)

    if len(envs) == 0: # no data to plot
      return discrim
    
    fig, axs, rows, cols = self.subplots_grid(len(envs), sharey=sharey,
                                              constrained_layout=True)
    legend_handles = []
    legend_labels = []
    layer_handles_captured = False

    num_measurements = discrim["data"]["num_measurements"]
    if measurement_names is None:
      if num_measurements == 1:
        measurement_names = ["Measurement"]
      else:
        measurement_names = [f"Measurement {x}" for x in range(num_measurements)]
    elif isinstance(measurement_names, str):
      measurement_names = [measurement_names]
    while len(measurement_names) < num_measurements:
      measurement_names.append("Unnamed measurement")

    # define plotting styles for multiple measurements and thresholds
    if discrim["data"]["num_measurements"] > 4:
      raise RuntimeError(f"Plotting only support num_measurements <= 4 currently. "
                         f"You had {discrim['data']['num_measurements']}. To add "
                         f"support you need plot styles and colours for 4 or more lines")
    measure_linestyles = ["-", "--", ":", "--"]
    threshold_colours = [
        "#005F73",
        "#ACF618",
        "#64D3EE",
        "#E5A3C9"
      ]
    
    # extra measurements for plotting
    measurements = discrim["measurements"]

    for i, e in enumerate(envs):

      x = np.arange(discrim["m_start"], discrim["m_end"])
      min_y = np.min(measurements[e])
      max_y = np.max(measurements[e])
      manip_active = np.where(discrim["manipulation_active"],
                              1.05 * max_y,
                              0.75 * min_y)


      if plot_stacked_component_error:
        collections, layer_labels_local = self.plot_stacked_error(joint_difference=discrim['joint_difference'][e],
                                                                  x = x,
                                                                  joint_indicies=stacked_joint_indicies,
                                                                  window_size=measure_args.get('window_size', 1),
                                                                  ax=axs[i],
                                                                  joint_labels=stacked_joint_labels,
                                                                  )
        for coll in collections:
          coll.set_zorder(0)  

        if not layer_handles_captured:
            legend_handles.extend(list(collections))
            legend_labels.extend(layer_labels_local)
            layer_handles_captured = True

      if measurements[e].ndim < 2:
        meas_line, = axs[i].plot(x, measurements[e], color="#111111", zorder=3)

        if i == 0:
          legend_handles.append(meas_line)
          legend_labels.append(measurement_names[0])
      else:

        for mnum in range(measurements[e].shape[-1]):

          if (measurements_to_plot is None or
              (isinstance(measurements_to_plot, (tuple, list)) and 
               mnum in measurements_to_plot) or
              measurements_to_plot == mnum):

            measure_line, = axs[i].plot(x, measurements[e][:, mnum], 
                                       color="#111111", zorder=3,
                                       ls=measure_linestyles[mnum])
            if i == 0:
              legend_handles.append(measure_line)
              legend_labels.append(measurement_names[mnum])
              
      manip_line, = axs[i].plot(
          x, manip_active[e],
          linestyle=":", color="#D81B60", zorder=3
      )
      if i == 0:
          legend_handles.append(manip_line)
          legend_labels.append("Manipulation")

      for tnum, threshold in enumerate(discrim["threshold"]):
        thr_line, = axs[i].plot(
            x, np.clip(threshold * np.ones(x.shape), a_min=0.0, a_max=1.1 * np.max(measurements[e])),
            linestyle="-.", color=threshold_colours[tnum], zorder=3
        )
        if i == 0:
          legend_handles.append(thr_line)
          legend_labels.append(f"{measurement_names[tnum]} threshold")
      
      # add labels to the graph of the classification
      if custom_env_labels is None:
        e_label = e
      else:
        e_label = custom_env_labels[e]
  
      # text = f"Env={e_label}\nOperator={operator} -> "
      text = f"Env={e_label}\n"
      if discrim["data"]["true_pos"][best_arg][e]: text += "true_pos"
      if discrim["data"]["true_neg"][best_arg][e]: text += "true_neg"
      if discrim["data"]["false_pos"][best_arg][e]: text += "false_pos"
      if discrim["data"]["false_neg"][best_arg][e]: text += "false_neg"

      if show_dr_data:
        dr_data = (f"\npayload_mass: {trajectory['mass'][e, 0, 0]}\n"
                   f"slew_gain: {trajectory['gain'][e, 0, 0]}\n"
                   f"luff_gain: {trajectory['gain'][e, 0, 1]}\n"
                   f"hoist_gain: {trajectory['gain'][e, 0, 2]}")
        text += dr_data

      axs[i].text(0.15, 0.97, text,
        verticalalignment='top',
        horizontalalignment='right',
        transform=axs[i].transAxes,
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1, alpha=0.5))

    fig.legend(
        legend_handles, legend_labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),  # move legend fully outside axes
        fontsize=11,
        frameon=True
    )

    threshold_str = str([f"{t:.3f}" for t in discrim["threshold"]])

    # format nicely the measurement args
    def build_up_string(dictionary, string, limit=50, avoid=[]):
      gap = ", "
      keylist = list(dictionary)
      for key in keylist:
        if key in avoid: continue
        if isinstance(dictionary[key], dict):
          if dictionary[key] == {}:
            string += f"{key}: {'{}'}{gap}"
          else:
            string += f"{key}: {'{'}"
            string = build_up_string(dictionary[key], string, limit=limit,
                                    avoid=avoid)
            string += f"{'}'}{gap}"
        else:
          to_add = f"{key}={dictionary[key]}{gap}"
          if len(to_add) > limit:
            string += f"{key}=...{gap}"
          else:
            string += to_add
      return string[:-len(gap)]
    

    m_args = build_up_string(discrim["measure_args"], "",
                             avoid=["reference_obj", "num_thresholds", "discrim_locks_high",
                                    "max_threshold", "use_reference", "reference_type"])
    m_args_wrapped = textwrap.fill(m_args, width=80)

    # add a title on the figure which contains all the key information
    if title_prefix is not None: title_prefix += "\n"
    fig.suptitle(f"{title_prefix if title_prefix is not None else ''}"
                 f"Best f1 score = {discrim['f1']:.3f} with "
                 f"threshold = {threshold_str}. "
                 f"\nCorresponding precision = {discrim['precision']:.3f}, "
                 f"recall = {discrim['recall']:.3f}, "
                 f"success_rate = {discrim['success_rate'] * 1e2:.1f}%"
                 f"\nBreakdown TP / TN / FP / FN = "
                 f"({(discrim['data']['true_pos'][best_arg].sum() / total) * 1e2:.1f} / "
                 f"{(discrim['data']['true_neg'][best_arg].sum() / total) * 1e2:.1f} / "
                 f"{(discrim['data']['false_pos'][best_arg].sum() / total) * 1e2:.1f} / "
                 f"{(discrim['data']['false_neg'][best_arg].sum() / total) * 1e2:.1f})% "
                 f"""\nMeasure args: [{m_args_wrapped}]"""
                 ,
                 fontsize=16)
    
    fig.set_size_inches(2 + 4 * cols + 2, 2 + 3 * rows)  # extra width for legend

    if return_data:
      img = fig_to_img(fig)
      discrim['plot'] = img
      return discrim
    
    else:
      plt.show()

  # --- utility functions --- #

  def widen_manipulation_vector(self, trajectories, start_earlier, end_later,
                                lock_high=False):
    """
    Vectorized version of the widening function for better performance.
    Uses a different approach with shift operations to handle asymmetric widening.
    
    Args:
        trajectories: numpy array of shape (B, T) with 0.0 and 1.0 values
        start_earlier: number of steps to extend backwards (start earlier)
        end_later: number of steps to extend forwards (end later)
    
    Returns:
        shifted_trajectories: numpy array of shape (B, T) with widened manipulation periods
    """
    B, T = trajectories.shape
    hthresh = 0.5
    result = np.zeros_like(trajectories, dtype=np.float32)

    if lock_high:
        # Detect rising edges
        padded = np.concatenate([np.zeros((B, 1), dtype=trajectories.dtype), trajectories], axis=1)
        diff = np.diff(padded, axis=1)
        rising = diff > hthresh

        any_rising = np.any(rising, axis=1)
        first_rising_idx = np.argmax(rising, axis=1)
        start_idx = np.clip(first_rising_idx - start_earlier, 0, T)

        # Time index
        time = np.arange(T)[None, :]  # (1, T)
        lock_mask = time >= start_idx[:, None]  # (B, T)
        lock_mask &= any_rising[:, None]        # disable mask where no rising edge

        result[lock_mask] = 1.0
    
    else:

      # Widening mode (standard behavior)
      batch_indices, time_indices = np.where(trajectories > hthresh)
      
      if len(batch_indices) == 0:
          return trajectories.copy()
      
      # For each high position, mark the widened region
      for offset in range(-start_earlier, end_later + 1):
          # Calculate new time indices with offset
          new_time_indices = time_indices + offset
          
          # Filter to keep only valid indices
          valid_mask = (new_time_indices >= 0) & (new_time_indices < T)
          valid_batch_indices = batch_indices[valid_mask]
          valid_time_indices = new_time_indices[valid_mask]
          
          # Set these positions to 1
          result[valid_batch_indices, valid_time_indices] = 1.0
    
    return result

  def subplots_grid(self, x, **subplots_args):

    """
    Return a flattened axis object based on the number of items to plot, x
    """
    rows = math.ceil(math.sqrt(x))
    cols = math.ceil(x / rows)
    fig, axs = plt.subplots(rows, cols, **subplots_args)
    if x == 1: axs = [axs]
    else: axs = axs.flatten()

    return fig, axs, rows, cols
  
  def plot_stacked_error(
      self,
      joint_difference: list[ArrayLike | JaxArrayLike], 
      method:str='norm',                  # how signal is/was computed              
      window_size:int = 1,               
      x:ArrayLike=None,                             
      joint_indicies:List = None,
      joint_labels:List = None,
      order_mode:str="by_max",
      order:List = None,                 
      ax:Axes=None,
  ):
      """
      Creates a stacked error plot, plotting the error per component that contributes to the discrimination 
      signal. 

      Args: 
        joint_difference (np.ndarray | jnp.ndarray): raw joint difference between predicted and actual, should be shape (T, H, D).
        method (str): how the discrimination signal was computed.
        window_size (int): size of window used in smoothing out measurements.  
        x (np.ndarray): x axis values.
        joint_indicies (List): joint indicies to plot on the stack plot. 
        joint_labels (List): joint labels to plot.
        order_mode (str): how to order the stacked plots.
        order (List): custom ordering if order_mode = 'specified'
        ax (Axes): axes to plot the stacked error over. 
      """

      jd = np.array(joint_difference)
      
      assert jd.ndim == 3, f"expecting joint difference to be of dim 3, instead got {jd.ndim}"

      # Compute per‑joint magnitude signal.
      if method == 'norm':
          S = jd ** 2                 # (T, H, D)
          C = np.sqrt(S.mean(axis=1)) # (T, D) mean over horizon
      else:
          raise NotImplementedError
      
      # select certain joints
      if joint_indicies is not None: 
        if isinstance(joint_indicies, list):
            joint_indicies = np.array(joint_indicies)
        C = C[..., joint_indicies] 
      
      # aggregate joints over window size:
      if window_size > 1:
          cumsum = np.cumsum(C, axis=0)  # cumulative over time
          C = cumsum[window_size-1:].copy()
          C[1:] -= cumsum[:-window_size]
          C /= window_size

      # ordering 
      if order_mode == "by_max":
        col_max = np.max(C, axis=0)
        order = np.argsort(-col_max)
      elif order_mode == "given":
        order = np.arange(C.shape[1])
      elif order_mode == "specified":
        order = order
      else:
          raise ValueError("order_mode must be 'by_max', 'given', 'specified' ")

      C = C[:, order]
      
      # plotting
      layers = np.transpose(C) # stackplot wants (D, T)
      created_fig = False

      if ax is None:
          fig, ax = plt.subplots()
          created_fig = True
      else:
          fig = ax.figure

      # ensure consistent colors across axes
      base_colors = list(plt.cm.tab10.colors)
      if len(base_colors) < C.shape[1]:
          repeats = math.ceil(C.shape[1] / len(base_colors))
          base_colors = (base_colors * repeats)[:C.shape[1]]
      # reorder colors to match `order`
      colors = [base_colors[j] for j in order]

      if joint_labels:
        layer_labels = [joint_labels[i] for i in order]
      else:       
        layer_labels = [f"Joint {j}" for j in order]

      collections = ax.stackplot(
          x, layers,
          labels=layer_labels,
          edgecolor='white', linewidth=0.5,
          colors=colors
      )

      if created_fig:
          fig.tight_layout()

      return collections, layer_labels

  def rolling_all(self, boolean_tensor, window_size, use_jax=False, axis=1):
    """
    Computes a rolling 'AND' and pads the result to match the input size.

    A timestep in the output is True only if all values within the corresponding
    window (from that timestep onward) are True. The output is padded with
    False at the end to match the original tensor's time dimension length.

    Args:
        boolean_tensor (np.ndarray): The input boolean tensor.
        window_size (int): The size of the rolling window.
        use_jax (bool): Whether using JAX.
        axis (int): The time dimension axis to roll over (default is 1).

    Returns:
        np.ndarray: A boolean tensor with the same shape as the input.
    """
    T = boolean_tensor.shape[axis]

    if not 1 <= window_size <= T:
      raise ValueError(f"window_size ({window_size}) must be in range [1, {T}]"
                       f" for axis {axis}")

    if window_size == 1:
      return boolean_tensor
    
    if use_jax:
      lib = jnp
    else:
      lib = np

    int_tensor = boolean_tensor.astype(int)
    # Perform cumsum along the specified time axis
    cumsum = lib.cumsum(int_tensor, axis=axis)
    
    # --- Dynamic Slicing ---
    
    # Build slicer for: cumsum[:, window_size-1:, ...] (for axis=1)
    slicer_start = [slice(None)] * boolean_tensor.ndim
    slicer_start[axis] = slice(window_size - 1, None)
    
    # Build slicer for: cumsum[:, :-window_size, ...] (for axis=1)
    slicer_src = [slice(None)] * boolean_tensor.ndim
    slicer_src[axis] = slice(None, -window_size)
    
    # Build slicer for: window_sums[:, 1:, ...] (for axis=1)
    slicer_dest = [slice(None)] * boolean_tensor.ndim
    slicer_dest[axis] = slice(1, None)
    
    # --- End Dynamic Slicing ---
    
    window_sums = cumsum[tuple(slicer_start)].copy()
    
    if use_jax:
      window_sums = window_sums.at[tuple(slicer_dest)].subtract(cumsum[tuple(slicer_src)])
    else:
      window_sums[tuple(slicer_dest)] -= cumsum[tuple(slicer_src)]
      
    result = window_sums == window_size

    # --- Efficient Padding ---
    # The result is shorter than the input by `window_size - 1` elements.
    num_to_pad = window_size - 1
    
    # We define the padding width for each axis.
    pad_width = [(0, 0)] * result.ndim
    # Pad only after the specified time axis
    pad_width[axis] = (0, num_to_pad) 

    # Pad the result with False values (0 for booleans).
    return lib.pad(result, pad_width, mode='constant', constant_values=False)

  def rolling_average(self, measurement_tensor, window_size=5, average_type="mean", 
                      use_jax=False, alpha=0.1, axis=1):
    """
    Compute a rolling average of the measurement along a specific axis.
    - "mean" or "median": Standard rolling window average.
    - "expo": Exponential moving average, controlled by `alpha`.
    
    Args:
        measurement_tensor (array): The input signal tensor.
        window_size (int): The size of the window for "mean" or "median".
        average_type (str): "mean", "median", or "expo".
        use_jax (bool): Whether to use JAX (jnp) or NumPy (np).
        alpha (float): Smoothing factor for "expo" (0 < alpha <= 1).
        axis (int): The time dimension axis to roll over (default is 1).
    
    Returns:
        array: The averaged tensor with the same shape as input.
    """

    if axis < 0:
      axis = measurement_tensor.ndim + axis

    T = measurement_tensor.shape[axis]

    if use_jax:
      lib = jnp
    else:
      lib = np

    if average_type == "mean" or average_type == "median":

      if T - window_size + 1 < 1:
        raise RuntimeError(f"DiscriminatorEval.rolling_average() error: "
                            f"T={T}, window_size={window_size} on axis {axis}, "
                            f"window size is too large for the current T")

      # Create overlapping windows along the specified time axis
      measurement_stacked = self.create_overlapping_windows(
          measurement_tensor, H=window_size, use_jax=use_jax, axis=axis
      )
      
      output_tensor = lib.zeros_like(measurement_tensor)
      
      # The new 'H' dimension is at 'axis + 1'
      window_axis = axis + 1
      
      # calculate the rolling average
      if average_type == "mean":
        average = lib.mean(measurement_stacked, axis=window_axis)
      elif average_type == "median":
        average = lib.median(measurement_stacked, axis=window_axis)
      
      # Build a dynamic slicer to insert the result
      # e.g., for axis=1: output_tensor[:, window_size - 1:, ...]
      slicer = [slice(None)] * output_tensor.ndim
      slicer[axis] = slice(window_size - 1, None)
      
      # insert the measurements at the end, pre-pad with zeros
      if use_jax:
        output_tensor = output_tensor.at[tuple(slicer)].set(average)
      else:
        output_tensor[tuple(slicer)] = average

      return output_tensor

    elif average_type == "expo":
      # Exponential moving average

      # broadcast alpha, to handle if we get one per measurement
      if isinstance(alpha, (tuple, list)):
        if len(alpha) != measurement_tensor.shape[-1]:
          raise RuntimeError(f"DiscriminatorEval.rolling_average() error: "
                            f"alpha={alpha}, but measurement_tensor.shape = "
                            f"{measurement_tensor.shape}. The last dimension of the "
                            F"measurements should match the number of alpha, if giving multiple")
        alpha = lib.array(alpha)
      else: alpha = lib.array([alpha])
      dims = measurement_tensor.ndim
      einstr = "x -> "
      for d in range(dims - 2): einstr += "1 "
      einstr += "x"
      alpha = einops.rearrange(alpha, einstr)
      
      # Slicer to get the first 'num_mean' items along the time axis
      num_mean = 5
      slicer_init = [slice(None)] * measurement_tensor.ndim
      slicer_init[axis] = slice(None, num_mean)
      
      # Initial "carry" (E_prev) is the mean of the first few timesteps
      initial_carry = lib.mean(measurement_tensor[tuple(slicer_init)], axis=axis)

      if use_jax:
        # JAX implementation uses jax.lax.scan
        
        def ewma_step(E_prev, x_t):
            # x_t is a slice of the tensor *without* the time dimension
            # Both x_t and E_prev must have the same shape, e.g., (R, B, N)
            E_t = alpha * x_t + (1.0 - alpha) * E_prev
            return E_t, E_t # (new_carry, new_output_slice)

        # We must scan over the time dimension (axis)
        
        # --- FIX: Dynamic Transpose ---
        # We need to move the time 'axis' to the front (axis 0)
        # and keep all other axes in their original relative order.
        
        original_axes = list(range(measurement_tensor.ndim))
        
        # e.g., if ndim=4, axis=2:
        # original_axes = [0, 1, 2, 3]
        # axes_to_scan = [2] + [0, 1, 3] = [2, 0, 1, 3]
        axes_to_scan = [axis] + [i for i in original_axes if i != axis]
        
        # This transposes from (R, B, T, N) -> (T, R, B, N)
        measurement_tensor_T_first = lib.transpose(measurement_tensor, axes_to_scan)
        
        # Run the scan
        # initial_carry has shape (R, B, N)
        # measurement_tensor_T_first has shape (T, R, B, N)
        # Therefore, x_t (slice) will have shape (R, B, N)
        # This now matches initial_carry.
        _, ewma_T_first = jax.lax.scan(ewma_step, initial_carry, measurement_tensor_T_first)
        
        # Transpose back
        # We need the inverse permutation
        # e.g., if axes_to_scan was [2, 0, 1, 3], we need [1, 2, 0, 3]
        axes_to_original = list(np.argsort(axes_to_scan))
        
        # This transposes from (T, R, B, N) -> (R, B, T, N)
        ewma_output = lib.transpose(ewma_T_first, axes_to_original)
        
        return ewma_output
        
      else:
        # NumPy implementation uses a standard for-loop
        ewma_output = lib.zeros_like(measurement_tensor)
        E_prev = initial_carry
        
        for t in range(T):
            # Dynamic slicer for x_t = measurement_tensor[:, t, ...] (for axis=1)
            slicer_t = [slice(None)] * measurement_tensor.ndim
            slicer_t[axis] = t
            x_t = measurement_tensor[tuple(slicer_t)]
            
            E_t = alpha * x_t + (1.0 - alpha) * E_prev
            
            # Dynamic slicer for ewma_output[:, t, ...] = E_t (for axis=1)
            ewma_output[tuple(slicer_t)] = E_t
            
            E_prev = E_t # Update carry for next iteration
            
        return ewma_output

    else:
      raise RuntimeError(f"DiscriminatorEval.rolling_average() error: "
                          f"average_type={average_type} not recognised")

  def create_overlapping_windows(self, data, H, use_jax=False, axis=1):
    """
    Vectorized version using advanced indexing for better performance.
    
    Creates overlapping windows along a specified axis 'axis'.
    The new 'H' dimension will be inserted at 'axis + 1'.

    Args:
        data (np.ndarray): Input array of any shape.
        H (int): The horizon (window size).
        use_jax (bool): Whether to use JAX or NumPy.
        axis (int): The axis to create windows over (default is 1).

    Returns:
        np.ndarray: New array of shape 
                    (*dims_before_axis, T - H + 1, H, *dims_after_axis)
        
    Raises:
        ValueError: If H <= 0 or H > T.
    """
    
    if not isinstance(H, int) or H <= 0:
        raise ValueError("Horizon H must be a positive integer")
    
    if use_jax:
      lib = jnp
    else:
      lib = np
    
    # Handle negative axis
    if axis < 0:
        axis = data.ndim + axis
        
    T = data.shape[axis]
    
    if H > T:
        raise ValueError(f"Horizon H ({H}) cannot be greater than the number of "
                         f"timesteps T ({T}) on axis {axis}")
    
    # Calculate output dimensions
    num_windows = T - H + 1
    
    # Create index arrays for advanced indexing
    window_starts = lib.arange(num_windows)
    offsets = lib.arange(H)
    
    # Create indices for all windows at once
    # Broadcasting: (num_windows, 1) + (1, H) = (num_windows, H)
    indices = window_starts[:, lib.newaxis] + offsets[lib.newaxis, :]
    
    # Build a dynamic slicer tuple
    # e.g., for axis=1, this becomes (slice(None), indices, slice(None), ...)
    slicer = [slice(None)] * data.ndim
    slicer[axis] = indices
    
    # Use advanced indexing to extract all windows at once
    windowed_data = data[tuple(slicer)]
    
    # Output shape is (*dims_before_axis, num_windows, H, *dims_after_axis)
    return windowed_data

  # --- depreciated functions --- #
