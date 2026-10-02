import numpy as np
from dataclasses import dataclass
import torch
import jax.numpy as jnp
from sklearn.model_selection import StratifiedKFold
import logging; logging.basicConfig(
    level=logging.INFO); pylogging = logging.getLogger(__name__)

from aware.utils.modelsaver import ModelSaver

def apply_norm(x, mean, std, debug=False):
  """
  Apply normalisations to a tensor x
  """
  if debug:
    # if not scalars, check vectors have identical length
    if len(x.shape) > 0 and len(mean.shape) > 0 and x.shape[-1] != mean.shape[0]: 
        raise RuntimeError(f"x.shape={x.shape}, mean.shape={mean.shape}")
    if len(x.shape) > 0 and len(std.shape) > 0 and x.shape[-1] != std.shape[0]: 
        raise RuntimeError(f"x.shape={x.shape}, std.shape={std.shape}")
    
  xbar = (x - mean) / std

  if debug:
    mean_before = torch.mean(to_torch(x)).item()
    std_before = torch.std(to_torch(x)).item()
    mean_after = torch.mean(to_torch(xbar)).item()
    std_after = torch.std(to_torch(xbar)).item()
    logging.info(f"Apply normalisation:\n"
                 f" -> before (u={mean_before:.4f}, s={std_before:.4f})\n"
                 f" -> after (u={mean_after:.4f}, s={std_after:.4f})")
    
  return xbar

def revert_norm(x, mean, std, debug=False):
  """
  Revert normalisations on a tensor x
  """
  if debug:
    if x.shape[-1] != mean.shape[0]: 
        raise RuntimeError(f"x.shape={x.shape}, mean.shape={mean.shape}")
    if x.shape[-1] != std.shape[0]: 
        raise RuntimeError(f"x.shape={x.shape}, std.shape={std.shape}")
    
  xbar = (x * std) + mean

  if debug:
    mean_before = torch.mean(to_torch(x)).item()
    std_before = torch.std(to_torch(x)).item()
    mean_after = torch.mean(to_torch(xbar)).item()
    std_after = torch.std(to_torch(xbar)).item()
    logging.info(f"Revert normalisation:\n"
                 f" -> before (u={mean_before:.4f}, s={std_before:.4f})\n"
                 f" -> after (u={mean_after:.4f}, s={std_after:.4f})")
    
  return xbar

def to_torch(x):
    """
    Return a new torch tensor verion of the input array, assuming the input array
    is either a JAX array or a numpy array.
    """
    if isinstance(x, torch.Tensor): return x
    return torch.tensor(np.array(x.copy()))

def calculate_average_precision(precision, recall, return_extras=False, interpolated=True):
    """
    Calculate the average precision given precision and recall
    """

    # handle the multiple dimension case (multiple measurements)
    if precision.ndim > 1:
        precision = precision.reshape(-1)
        recall = recall.reshape(-1)

    # add the anchor point of full precision with 0.0 recall
    P = np.concatenate(([1.0], precision))
    R = np.concatenate(([0.0], recall))

    # strictly sort, so that identical recall values are sorted by their precision
    sorted_inds = np.lexsort((-P, R))
    R = R[sorted_inds]
    P = P[sorted_inds]

    # smooth out jagged surface
    if interpolated:
        for i in range(len(P) - 2, -1, -1):
            P[i] = max(P[i], P[i + 1])

    recall_changes = np.where(R[1:] != R[:-1])[0]
    avg_precision = np.sum((R[recall_changes + 1] - R[recall_changes]) * P[recall_changes + 1])

    if return_extras:

        # Get all unique points that define the interpolated curve
        # This includes the start (0, 1) and all points at recall changes
        plot_indices = np.concatenate(([0], recall_changes + 1))
        
        # get the precision-recall vectors for plotting
        info = {
            "precision" : P[plot_indices],
            "recall" : R[plot_indices],
        }
        # add in anchor point for plotting
        info["recall"] = np.concatenate((info["recall"], [1.0]))
        info["precision"] = np.concatenate((info["precision"], [0.0]))
        return avg_precision, info
    else:
        return avg_precision

def calculate_auroc(tp, tn, fp, fn, return_extras=False):
    """
    Calculate the Area Under the Receiver Operating Characteristic (AUROC) curve.
    Inputs must be numpy arrays representing the counts at different thresholds.
    """
    
    # 1. Flatten inputs
    tp = np.asarray(tp).sum(axis=-1).reshape(-1).astype(int)
    fp = np.asarray(fp).sum(axis=-1).reshape(-1).astype(int)
    tn = np.asarray(tn).sum(axis=-1).reshape(-1).astype(int)
    fn = np.asarray(fn).sum(axis=-1).reshape(-1).astype(int)

    # 2. Calculate Rates
    # We use np.divide with 'where' to handle cases with 0 positives or negatives
    actual_positives = tp + fn
    actual_negatives = fp + tn
    
    # Check if we have valid data (at least one positive and one negative total)
    # If not, the metric is undefined. We return 0.0 or raise an error.
    if np.all(actual_positives == 0) or np.all(actual_negatives == 0):
        # Depending on preference, you could raise ValueError here
        print(f"invalid data")
        return (0.0, {}) if return_extras else 0.0

    tpr = np.divide(tp, actual_positives, out=np.zeros_like(tp, dtype=float), where=actual_positives!=0)
    fpr = np.divide(fp, actual_negatives, out=np.zeros_like(fp, dtype=float), where=actual_negatives!=0)

    # 3. Add anchor points (0,0) and (1,1)
    tpr_curve = np.concatenate(([0.0], tpr, [1.0]))
    fpr_curve = np.concatenate(([0.0], fpr, [1.0]))

    # 4. Sort strictly by FPR, then TPR
    # lexsort sorts by the last array passed (primary), then the previous (secondary)
    # This ensures that for the same FPR, we go from lower TPR to higher TPR (vertical line up)
    sorted_inds = np.lexsort((tpr_curve, fpr_curve))
    tpr_curve = tpr_curve[sorted_inds]
    fpr_curve = fpr_curve[sorted_inds]

    # 5. Calculate Area (Trapezoidal Rule)
    auroc = np.trapz(tpr_curve, fpr_curve)

    if return_extras:
        # CORRECTION: Do not filter by unique FPR. 
        # ROC curves are step functions; filtering removes the "vertical" steps 
        # and distorts the visualization. 
        # We only remove exact duplicates (same FPR AND same TPR).
        
        # Check differences with the next point
        diffs = np.diff(fpr_curve, append=fpr_curve[-1] + 1) != 0 # FPR changes
        diffs |= np.diff(tpr_curve, append=tpr_curve[-1] + 1) != 0 # TPR changes
        
        # Always keep the last point
        mask = diffs
        mask[-1] = True 

        info = {
            "tpr": tpr_curve[mask],
            "fpr": fpr_curve[mask],
        }
        return auroc, info
    else:
        return auroc

def cross_validate_best_f1(true_pos, true_neg, false_pos, false_neg, n_splits=5):
    """
    Performs Stratified K-Fold CV to find the unbiased F1 score.
    Automatically derives ground truth labels from the input arrays.
    
    Args:
        true_pos, ... : Boolean arrays of shape (..., Batch)
    """
    
    # 1. Shape Handling (Flattening the Grid)
    batch_size = true_pos.shape[-1]
    
    # Reshape from (..., Batch) -> (Total_Configs, Batch)
    tp_flat = true_pos.reshape(-1, batch_size)
    tn_flat = true_neg.reshape(-1, batch_size)
    fp_flat = false_pos.reshape(-1, batch_size)
    fn_flat = false_neg.reshape(-1, batch_size)
    
    num_configs = tp_flat.shape[0]
    
    # 2. Derive Ground Truth Labels (The Fix)
    # A sample is Positive if it is either a TP or FN at index 0 (or any index).
    # We use boolean OR (|) to reconstruct the ground truth vector.
    gt_labels = tp_flat[0, :] | fn_flat[0, :]
    
    # 3. Setup Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    final_tp_mask = np.zeros(batch_size, dtype=bool)
    final_tn_mask = np.zeros(batch_size, dtype=bool)
    final_fp_mask = np.zeros(batch_size, dtype=bool)
    final_fn_mask = np.zeros(batch_size, dtype=bool)

    # pylogging.info(f"Starting {n_splits}-Fold CV on {batch_size} samples...")

    # 4. The Cross-Validation Loop
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(np.zeros(batch_size), gt_labels)):
        
        # --- A. TRAINING PHASE ---
        train_tp = np.sum(tp_flat[:, train_idx], axis=1)
        train_fp = np.sum(fp_flat[:, train_idx], axis=1)
        train_fn = np.sum(fn_flat[:, train_idx], axis=1)
        
        epsilon = 1e-10
        train_precision = train_tp / (train_tp + train_fp + epsilon)
        train_recall    = train_tp / (train_tp + train_fn + epsilon)
        train_f1        = 2 * (train_precision * train_recall) / (train_precision + train_recall + epsilon)
        
        best_idx = np.argmax(train_f1)
        
        # --- B. TESTING PHASE ---
        final_tp_mask[test_idx] = tp_flat[best_idx, test_idx]
        final_tn_mask[test_idx] = tn_flat[best_idx, test_idx]
        final_fp_mask[test_idx] = fp_flat[best_idx, test_idx]
        final_fn_mask[test_idx] = fn_flat[best_idx, test_idx]

    # 5. Final Aggregation
    total_tp = np.sum(final_tp_mask)
    total_tn = np.sum(final_tn_mask)
    total_fp = np.sum(final_fp_mask)
    total_fn = np.sum(final_fn_mask)
    
    precision = total_tp / (total_tp + total_fp + 1e-10)
    recall    = total_tp / (total_tp + total_fn + 1e-10)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-10)
    accuracy  = (total_tp + total_tn) / batch_size

    metrics = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "success_rate": accuracy,
        "tp" : total_tp,
        "tn" : total_tn,
        "fp" : total_fp,
        "fn" : total_fn,
    }
    
    return metrics

@dataclass
class AnnealSchedule:
  imin: int = 0
  imax: int = 100
  xmin: float = 0.0
  xmax: float = 1.0

def anneal(i, schedule):
  if i < schedule.imin: return schedule.xmin
  elif i > schedule.imax: return schedule.xmax
  else: 
    t = ((i - schedule.imin) / (schedule.imax - schedule.imin))
    return schedule.xmin + (schedule.xmax - schedule.xmin) * t
  
def symlog(x: torch.Tensor, constant: float = 1.0) -> torch.Tensor:
    """
    Applies the symmetric logarithm transformation to a PyTorch tensor.

    The function handles positive, negative, and zero values gracefully.
    It's defined as:
    y = sign(x) * log(|x| + constant)

    Args:
        x (torch.Tensor): The input tensor of shape (B, T, X).
        constant (float): A small positive constant to add inside the logarithm
                          to handle values near zero and control the compression.
                          Commonly 1.0 or a value related to the scale of small
                          values in your data.

    Returns:
        torch.Tensor: The symlog-transformed tensor.
    """
    if constant <= 0:
        raise ValueError("The 'constant' parameter must be positive.")

    # Calculate the sign of the tensor
    s = torch.sign(x)

    # Calculate the absolute value
    abs_x = torch.abs(x)

    # Apply the log transformation to the absolute value plus the constant
    log_transformed = torch.log(abs_x + constant)

    # Reapply the sign
    y = s * log_transformed
    return y

def symexp(y: torch.Tensor, constant: float = 1.0) -> torch.Tensor:
    """
    Applies the inverse symmetric logarithm transformation (symmetric exponential)
    to a PyTorch tensor.

    It's the inverse of the symlog function:
    x = sign(y) * (exp(|y|) - constant)

    Args:
        y (torch.Tensor): The symlog-transformed tensor of shape (B, T, X).
        constant (float): The same constant used in the original symlog transformation.

    Returns:
        torch.Tensor: The original (untransformed) tensor.
    """
    if constant <= 0:
        raise ValueError("The 'constant' parameter must be positive.")

    # Calculate the sign of the transformed tensor
    s = torch.sign(y)

    # Calculate the absolute value
    abs_y = torch.abs(y)

    # Apply the exponential transformation to the absolute value
    exp_transformed = torch.exp(abs_y)

    # Subtract the constant and reapply the sign
    x = s * (exp_transformed - constant)
    return x

def load_normalisation(loadpath, filename="data_torch", min_std=1e-3,
                       mean_field="Mean", std_field="Stddev",
                       use_torch=False, device="cuda", debug=False):
    """
    Load a normalisation from a file, and return a normalisation object.
    """

    pylogging.info(f"Preparing to load normalisations from: {loadpath}/{filename}")
    modelsaver = ModelSaver(loadpath)
    loaded_text = modelsaver.read_textfile(filename)

    table_dict = {}

    lines = loaded_text.split("\n")
    data_lines = [line.strip() for line in lines if line.strip()]
    
    for line in data_lines:
        # Split the line by '|' and clean up whitespace
        parts = [part.strip() for part in line.split('|')]
        
        # skip lines without any '|', as not part of the table
        if len(parts) < 2:
            continue

        # first table column must be the name of the variable
        name = parts[0]

        # check for the table column headers, and create fields for each
        if name == "Name":
            table_headings = parts[:]
            for heading in table_headings:
                table_dict[heading] = []
        else:
            for i, heading in enumerate(table_headings):
                if heading == "Name":
                    table_dict[heading].append(parts[i])
                else:
                    table_dict[heading].append(float(parts[i]))

    # now create the normalisation object
    names = table_dict["Name"]
    means = table_dict[mean_field]
    stds = table_dict[std_field]

    norm = Normalisation(names=names, means=means, stds=stds, debug=debug,
                         min_std=min_std, use_torch=use_torch, device=device)
    
    return norm
    
class Normalisation:

    def __init__(self, names=None, means=None, stds=None, min_std=1e-3, 
                 use_torch=False, device="cuda", debug=True):
        """
        Create an object which handles normalisations for a set of variables given
        by names, each with a mean and standard deviation.

        Args:
            - names: a list of unique names, eg ["qpos[0]", "qpos[1]", ....]
            - means: corresponding means in the exact order of names
            - stds: corresponding standard deviations in the exact order of stds
            - min_std: minimum std deviation value, since this is divided by
        """

        if use_torch:
            self.means = torch.tensor(means).to(device)
            self.stds = torch.clamp(torch.tensor(stds), min=min_std).to(device)
            self.obs_means = self.means.clone()
            self.obs_stds = self.stds.clone()
        else:
            self.means = jnp.array(means)
            self.stds = jnp.clip(jnp.array(stds), min=min_std)
            self.obs_means = self.means.copy()
            self.obs_stds = self.stds.copy()

        self.use_torch = use_torch
        self.names = names
        self.min_std = min_std
        self.device = device
        self.debug = debug

        pylogging.info(f"Normalisation object created:\n"
                       f"  -> number of names = {len(self.names)}\n"
                       f"  -> min_std = {self.min_std}\n"
                       f"  -> use_torch = {self.use_torch}\n"
                       f"  -> device = {self.device}\n")
        
        # add extra information about the normalisation parameters for debugging
        if self.debug:
            log_str = """All normalisation fields:\n"""
            for i in range(len(self.names)):
                log_str += (f"  ({i}) -> {self.names[i]}, mean={self.means[i]:.3f}, "
                            f"std={self.stds[i]:.3f}\n")
            for line in log_str.splitlines():
                pylogging.info(line)

    def set_obs(self, obs_names, with_return=False):
        """
        Sets an observation with a specified set of names, in the same format as
        'self.names', and including only names which are within 'self.names', but
        in any order.
        """

        # set the mean and standard deviations corresponding to these names
        self.obs_names = obs_names
        self.obs_means, self.obs_stds = self.get_means_stds(self.obs_names)

        if with_return:
            return self.obs_means, self.obs_stds
    
    def get_means_stds(self, names):
        """
        Get the means and standard deviations of a set of names
        """

        if self.use_torch:
            means = torch.zeros(len(names), device=self.device)
            stds = torch.zeros(len(names), device=self.device)
        else:
            means =np.zeros(len(names))
            stds = np.zeros(len(names)) 

        if self.debug: log_str = """Normalisation.get_means_stds() values:\n"""
        for i, name in enumerate(names):
            for j, n in enumerate(self.names):
                if n == name:
                    means[i] = self.means[j]
                    stds[i] = self.stds[j]
                    if self.debug:
                        log_str += (f"  ({i}) -> {self.names[j]}, mean={self.means[j]:.3f}, "
                                    f"std={self.stds[j]:.3f}\n")
                    break
                    
        if self.debug: pylogging.info(log_str)

        if not self.use_torch:
            # jax arrays are immutable, so convert at the end
            means = jnp.array(means)
            stds = jnp.array(stds)

        return means, stds

    def apply_normalisation(self, obs, names=None, debug=False):
        """
        Apply normalisation to an observation
        """
        if names is not None:
            if debug: pylogging.info(f"Changing to new obs with names = {names}")
            self.set_obs(names)

        return apply_norm(obs, self.obs_means, self.obs_stds, debug=debug)
    
    def revert_normalisation(self, obs, names=None, debug=False):
        """
        Revert normalisation on an observation
        """
        if names is not None:
            if debug: pylogging.info(f"Changing to new obs with names = {names}")
            self.set_obs(names)

        return revert_norm(obs, self.obs_means, self.obs_stds, debug=debug)
    
    def create_dict(self, list_of_names=None):
        """
        Create a dictionary to save normalisations for a series of names,
        eg list_of_names = [
            (["qpos[0]", "qpos[1]", "qpos[2]", etc], "qpos"),
            (["qvel[0]", "qvel[2]", "qvel[3]", etc], "qvel"),
            ...
        ]
        """

        norm_dict = {
            "names" : self.names,
            "means" : self.means,
            "stds" : self.stds,
        }

        for names, label in list_of_names:
            norm_dict[label] = {}
            (norm_dict[label]["mean"], 
             norm_dict[label]["std"]) = self.get_means_stds(names)
            
        return norm_dict