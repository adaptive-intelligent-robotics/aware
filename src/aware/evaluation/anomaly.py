"""
Anomaly detection evaluation on the released AWARE crane dataset, as reported in the paper.
Used by scripts/evaluate.py, and periodically by the trainers during training.
"""
import logging; pylogger = logging.getLogger(__name__)
from copy import deepcopy

import numpy as np

from aware import REPO_ROOT as path_to_root
from aware.evaluation.discriminator import DiscriminatorEval
from aware.utils.trajectory import load_trajectory, index_trajectory_batch

# released trajectories, see the README for a description of the dataset
DATASETS = {
  "low_noise" : "data/crane_human_operator_240traj_low_noise.pkl",   # Vicon state estimation
  "high_noise" : "data/crane_human_operator_240traj_high_noise.pkl", # CCTV state estimation
}

def load_dataset(name):
  """
  Load one of the released datasets by name ('low_noise' or 'high_noise'), or a path.
  """
  path = DATASETS[name] if name in DATASETS else name
  return load_trajectory(f"{path_to_root}/{path}")

def reference_latent_mean(double_discrim, trajectory, num_ref_traj=80, seed=100):
  """
  Mean estimated latent over randomly selected nominal (non-manipulated) trajectories.
  This is given to the motion predictor in place of per-step latent estimates.
  """
  rng = np.random.default_rng(seed)
  manip_active = np.any(trajectory["manipulation_bool"], axis=1)
  clean_trajectories = np.argwhere(~manip_active).reshape(-1)
  random_clean_inds = rng.permutation(clean_trajectories)[:num_ref_traj]
  ref_traj = index_trajectory_batch(trajectory, indexes=random_clean_inds)
  pylogger.info(f"Reference latents from {ref_traj['joint_angles'].shape[0]} / "
                f"{len(clean_trajectories)} non-manipulated trajectories")
  ref_dict = double_discrim.latent_discrim.get_reference_values(
    trajectory=deepcopy(ref_traj),
    method="mahalanobis",
  )
  return np.squeeze(ref_dict["means"], axis=0) # shape (1, N) -> (N)

def evaluate(double_discrim, trajectory, name, traj_label=None, start_index=100,
             num_thresholds=1024, print_out=True):
  """
  Evaluate anomaly detection with the double discriminator, returning a dictionary of
  results for each signal type: 'double', 'latent' and 'predictor'. Each contains the
  keys 'average_precision', 'auroc', 'f1', 'sr' and 'approx_frequency'.
  """
  latent_override = reference_latent_mean(double_discrim, trajectory)

  return DiscriminatorEval().default_evaluation(
    double_discrim=double_discrim,
    eval_name=name,
    trajectory=deepcopy(trajectory),
    traj_label=traj_label,
    start_index=start_index,
    eval_double=True,
    eval_latent=True,
    eval_predictor=True,
    print_out=print_out,
    double_mode="seperate",
    use_jax=True,
    classification_args={
      "num_thresholds" : num_thresholds, # spread over measurements, so 1024 -> 32 * 32 for 2
      "max_threshold" : "auto",
      "threshold_balancing" : True,
      "measure_combine_mode" : "or",
      "invalid_data_mode" : "zero",
    },
    discrim_args={
      "method" : "mahalanobis",
      "reference_auto_num_trajectories" : 10,
      "joint_diff_method" : "average",
      "denoise_reference_trajectory" : False,
      "add_confidence_as_measurement" : False,
      "add_confidence_as_feature" : False,
      "multiply_by_confidence" : False,
      "scale_by_uncertainty" : True,
      "pred_args" : {
        "horizon" : 20,
        "position" : True,
        "velocity" : True,
        "prediction_args" : {
          "fix_estimated_latents" : False,
          "predict_latents_only_once" : False,
          "predictor_latent_override" : latent_override,
        },
      },
      "latent_args" : {},
    },
  )
