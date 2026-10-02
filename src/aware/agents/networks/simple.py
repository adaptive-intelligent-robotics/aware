# python imports
import logging; pylogger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----- utility functions ----- #

def get_activation_fn(act_name):
  """
  Convert a named activation function into its pytorch equivalent. If not given
  a string, assumes that the activation is already converted.
  """
  if isinstance(act_name, str):
    if act_name.lower() == "elu":
      return nn.ELU
    elif act_name.lower() == "selu":
      return nn.SELU
    elif act_name.lower() == "relu":
      return nn.ReLU
    elif act_name.lower() == "crelu":
      return nn.ReLU
    elif act_name.lower() == "lrelu":
      return nn.LeakyReLU
    elif act_name.lower() == "tanh":
      return nn.Tanh
    elif act_name.lower() == "sigmoid":
      return nn.Sigmoid
    else:
      raise RuntimeError(f"get_activation_fn() error: "
                         f"invalid activation function: received '{act_name}'")
  else:
    return act_name

def get_loss_fn(loss_name):
  """
  Convert a named loss function into its pytorch equivalent. If not given
  a string, assumes that the loss is already converted.
  """
  if isinstance(loss_name, str):
    if loss_name.lower() == "mse":
      return nn.MSELoss
    elif loss_name.lower() == "l1":
      return nn.L1Loss
    elif loss_name.lower() in ["huber", "smoothl1"]:
      return nn.HuberLoss
    else:
      raise RuntimeError(f"get_loss_fn() error: "
                         f"invalid loss function: received '{loss_name}'")
  return loss_name

def make_mlp(
    input_dim,
    output_dim,
    hidden_dim,
    num_hidden,
    activation_fn=nn.ELU,
    use_layernorm=False,
    use_dropout=False,
    dropout_prob=0.1,
    activate_output=False,
    init_weights=True,
):
  """
  Make a simple MLP with proper weight initialization for common activation functions.

  Args:
      input_dim: Input dimension
      output_dim: Output dimension  
      hidden_dim: Hidden layer dimension
      num_hidden: Number of hidden layers
      activation_fn: Activation function (nn.Module class or string)
      use_layernorm: Whether to use layer normalization
      use_dropout: Whether to use dropout
      dropout_prob: Dropout probability
      activate_output: Whether to apply activation to output layer
      init_weights: Whether to initialize weights based on activation function
  """

  # if activation_fn is a string input, convert to torch function
  activation_fn = get_activation_fn(activation_fn)

  layers = []

  # First layer
  layers.append(nn.Linear(input_dim, hidden_dim))
  if use_layernorm:
    layers.append(nn.LayerNorm(hidden_dim))
  layers.append(activation_fn())
  if use_dropout:
    layers.append(nn.Dropout(dropout_prob))

  # Hidden layers
  for _ in range(num_hidden):
    layers.append(nn.Linear(hidden_dim, hidden_dim))
    if use_layernorm:
      layers.append(nn.LayerNorm(hidden_dim))
    layers.append(activation_fn())
    if use_dropout:
      layers.append(nn.Dropout(dropout_prob))

  # Final layer
  layers.append(nn.Linear(hidden_dim, output_dim))
  if activate_output:
    layers.append(activation_fn())

  model = nn.Sequential(*layers)

  # Initialize weights based on activation function
  if init_weights:
    initialise_weights(model, activation_fn)

  return model

def initialise_weights(model, activation_fn):
  """Initialize weights based on the activation function used."""
  activation_name = activation_fn.__name__.lower()

  # Define initialization strategies
  he_activations = {'relu', 'leakyrelu', 'elu', 'swish', 'silu', 'mish'}
  xavier_activations = {'tanh', 'sigmoid', 'gelu'}

  for module in model.modules():
    if isinstance(module, nn.Linear):
      if activation_name in he_activations:
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
      elif activation_name == 'selu':
        nn.init.normal_(module.weight, 0, math.sqrt(1.0 / module.weight.size(1)))
      else:  # xavier for tanh/sigmoid/gelu and unknown activations
        nn.init.xavier_normal_(module.weight)

      if module.bias is not None:
        nn.init.constant_(module.bias, 0)

# --- loss functions --- #

class PrivilegedContrastiveLoss:

  def __init__(self, 
               similarity_threshold=0.2,  # Threshold for considering samples "similar"
               dissimilarity_threshold=0.8,  # Threshold for considering samples "different"
               margin=1.0,
               temperature=0.1,
               adaptive_thresholds=True,
               normalize_distances=True):
    """
    Contrastive loss for privileged information in [-1, +1] range

    Args:
        similarity_threshold: Percentile threshold for similar pairs (0-1)
        dissimilarity_threshold: Percentile threshold for dissimilar pairs (0-1)  
        margin: Margin for negative pairs in contrastive loss
        temperature: Temperature for distance scaling
        adaptive_thresholds: Whether to compute thresholds adaptively per batch
        normalize_distances: Whether to normalize distances by dimensionality
    """
    self.similarity_threshold = similarity_threshold
    self.dissimilarity_threshold = dissimilarity_threshold
    self.margin = margin
    self.temperature = temperature
    self.adaptive_thresholds = adaptive_thresholds
    self.normalize_distances = normalize_distances

  def compute_privileged_distances(self, priv_info):
    """
    Compute L2 distances between privileged information vectors

    Args:
        priv_info: [B, T, D_priv] privileged information
    Returns:
        distances: [B*T, B*T] pairwise distance matrix
        max_possible_distance: theoretical maximum distance
    """
    B, T, D_priv = priv_info.shape
    priv_flat = priv_info.view(B * T, D_priv)

    # Compute pairwise L2 distances
    distances = torch.cdist(priv_flat, priv_flat, p=2)

    # For uniform [-1, +1], max distance is sqrt(D_priv * 4) = 2*sqrt(D_priv)
    max_possible_distance = 2.0 * math.sqrt(D_priv)

    if self.normalize_distances:
      distances = distances / max_possible_distance

    return distances, max_possible_distance

  def get_similarity_masks(self, distances, batch_size_flat):
    """
    Create masks for similar and dissimilar pairs based on distance thresholds
    """
    # Remove self-comparisons
    identity_mask = torch.eye(batch_size_flat, device=distances.device).bool()

    if self.adaptive_thresholds:
      # Use percentiles of actual distance distribution
      flat_distances = distances[~identity_mask]
      sim_threshold = torch.quantile(flat_distances, self.similarity_threshold)
      dissim_threshold = torch.quantile(flat_distances, self.dissimilarity_threshold)
    else:
      # Fixed thresholds based on normalized distance
      if self.normalize_distances:
        sim_threshold = self.similarity_threshold  # Already 0-1 range
        dissim_threshold = self.dissimilarity_threshold
      else:
        # Scale by max possible distance
        max_dist = distances.max().item()
        sim_threshold = self.similarity_threshold * max_dist
        dissim_threshold = self.dissimilarity_threshold * max_dist

    # Create masks
    similar_mask = (distances <= sim_threshold) & ~identity_mask
    dissimilar_mask = (distances >= dissim_threshold) & ~identity_mask

    return similar_mask, dissimilar_mask, sim_threshold, dissim_threshold

  def loss(self, latents, priv_info, return_stats=False):
    """
    Main contrastive loss computation

    Args:
        latents: [B, T, D_lat] latent encodings
        priv_info: [B, T, D_priv] privileged information
        return_stats: whether to return debugging statistics

    Returns:
        loss: scalar contrastive loss
        stats: dict with debugging info (if return_stats=True)
    """
    B, T, D_lat = latents.shape
    batch_size_flat = B * T

    # Flatten tensors
    latents_flat = latents.view(batch_size_flat, D_lat)

    # Compute privileged information distances
    priv_distances, max_priv_dist = self.compute_privileged_distances(priv_info)

    # Get similarity masks
    similar_mask, dissimilar_mask, sim_thresh, dissim_thresh = self.get_similarity_masks(
        priv_distances, batch_size_flat
    )

    # Compute latent distances
    latent_distances = torch.cdist(latents_flat, latents_flat, p=2)

    # Positive loss: similar privileged info should have similar latents
    if similar_mask.sum() > 0:
      pos_latent_distances = latent_distances[similar_mask]
      positive_loss = pos_latent_distances.mean()
    else:
      positive_loss = torch.tensor(0.0, device=latents.device)

    # Negative loss: dissimilar privileged info should have dissimilar latents
    if dissimilar_mask.sum() > 0:
      neg_latent_distances = latent_distances[dissimilar_mask]
      # Apply margin-based loss (encourage distance > margin)
      negative_loss = torch.clamp(self.margin - neg_latent_distances, min=0).mean()
    else:
      negative_loss = torch.tensor(0.0, device=latents.device)

    total_loss = positive_loss + negative_loss

    if return_stats:
      stats = {
          'positive_loss': positive_loss.item(),
          'negative_loss': negative_loss.item(),
          'n_similar_pairs': similar_mask.sum().item(),
          'n_dissimilar_pairs': dissimilar_mask.sum().item(),
          'sim_threshold': sim_thresh.item() if torch.is_tensor(sim_thresh) else sim_thresh,
          'dissim_threshold': dissim_thresh.item() if torch.is_tensor(dissim_thresh) else dissim_thresh,
          'avg_latent_distance': latent_distances[~torch.eye(batch_size_flat, device=latents.device).bool()].mean().item(),
          'avg_priv_distance': priv_distances[~torch.eye(batch_size_flat, device=latents.device).bool()].mean().item(),
          'max_priv_distance': max_priv_dist
      }
      return total_loss, stats

    return total_loss

class InfoNCEPrivilegedLoss:

  def __init__(self, temperature=0.1, k_negatives=None):
    """
    InfoNCE loss using privileged information similarity as ground truth

    Args:
        temperature: temperature parameter for softmax
        k_negatives: number of negatives to sample (None = use all)
    """
    self.temperature = temperature
    self.k_negatives = k_negatives

  def loss(self, latents, priv_info, return_stats=False):
    """
    Compute InfoNCE loss where similarity is determined by privileged info
    """
    B, T, D_lat = latents.shape
    batch_size_flat = B * T

    latents_flat = latents.view(batch_size_flat, D_lat)
    priv_flat = priv_info.view(batch_size_flat, -1)

    # Normalize latents for cosine similarity
    latents_norm = F.normalize(latents_flat, dim=-1)

    # Compute privileged info similarities (higher = more similar)
    priv_distances = torch.cdist(priv_flat, priv_flat, p=2)
    priv_similarities = torch.exp(-priv_distances / self.temperature)

    # Compute latent similarities
    latent_similarities = torch.matmul(latents_norm, latents_norm.T) / self.temperature

    # InfoNCE loss: latent similarities should match privileged similarities
    # Use privileged similarities as soft targets
    loss = F.kl_div(
        F.log_softmax(latent_similarities, dim=-1),
        F.softmax(priv_similarities, dim=-1),
        reduction='batchmean'
    )

    if return_stats:

      # Mask out self-comparisons (diagonal)
      mask = ~torch.eye(batch_size_flat, device=latents.device).bool()
      target_sim_masked = priv_similarities[mask]
      latent_sim_masked = latent_similarities[mask]

      with torch.no_grad():
        # Compute correlation between target and actual similarities
        correlation = torch.corrcoef(torch.stack(
            [target_sim_masked, latent_sim_masked]))[0, 1]

        stats = {
            'similarity_correlation': correlation.item() if not torch.isnan(correlation) else 0.0,
            'target_sim_mean': target_sim_masked.mean().item(),
            'target_sim_std': target_sim_masked.std().item(),
            'latent_sim_mean': latent_sim_masked.mean().item(),
            'latent_sim_std': latent_sim_masked.std().item(),
            'avg_priv_distance': priv_distances[mask].mean().item(),
        }
      return loss, stats
    else:
      return loss

class LinearSimilarityContrastiveLoss:

  def __init__(self, 
               temperature=0.1,
               similarity_scale='cosine',  # 'cosine', 'exp', or 'linear'
               latent_similarity='cosine',  # 'cosine' or 'l2'
               loss_type="mse",
               normalize_privileged=True):
    """
    Contrastive loss using linear scaling of similarities instead of binary thresholds

    Args:
        temperature: temperature parameter for similarity scaling
        similarity_scale: how to convert privileged distances to similarities
            - 'cosine': use cosine-like similarity [0,1]  
            - 'exp': exponential decay similarity
            - 'linear': linear mapping from distance to similarity
        latent_similarity: how to compute latent similarities
            - 'cosine': cosine similarity between normalized latents
            - 'l2': negative L2 distance (higher = more similar)
        normalize_privileged: whether to normalize privileged distances
    """
    self.temperature = temperature
    self.similarity_scale = similarity_scale
    self.latent_similarity = latent_similarity
    self.normalize_privileged = normalize_privileged
    self.loss_type = loss_type

  def privileged_to_similarity(self, priv_distances, max_distance=None):
    """
    Convert privileged information distances to similarity targets [0,1]

    Args:
        priv_distances: [N, N] pairwise distance matrix
        max_distance: maximum possible distance for normalization

    Returns:
        similarities: [N, N] similarity matrix where 1=identical, 0=maximally different
    """
    if self.normalize_privileged and max_distance is not None:
      # Normalize distances to [0, 1] range
      normalized_distances = priv_distances / max_distance
    else:
      normalized_distances = priv_distances

    if self.similarity_scale == 'cosine':
      # Cosine-like: similarity = 1 - normalized_distance
      similarities = torch.clamp(1.0 - normalized_distances, min=0.0, max=1.0)

    elif self.similarity_scale == 'exp':
      # Exponential decay: similarity = exp(-distance/temperature)
      similarities = torch.exp(-normalized_distances / self.temperature)

    elif self.similarity_scale == 'linear':
      # Linear mapping: map [0, max_dist] to [1, 0]
      max_dist = normalized_distances.max() if normalized_distances.max() > 0 else 1.0
      similarities = torch.clamp(1.0 - normalized_distances / max_dist, min=0.0, max=1.0)

    else:
      raise ValueError(f"Unknown similarity_scale: {self.similarity_scale}")

    return similarities

  def compute_latent_similarities(self, latents):
    """
    Compute similarity matrix for latents

    Args:
        latents: [N, D] flattened latents

    Returns:
        similarities: [N, N] latent similarity matrix
    """
    if self.latent_similarity == 'cosine':
      # Cosine similarity: normalize then compute dot product
      latents_norm = F.normalize(latents, dim=-1)
      similarities = torch.matmul(latents_norm, latents_norm.T)
      # Scale to [0, 1] range: (cosine + 1) / 2
      similarities = (similarities + 1.0) / 2.0

    elif self.latent_similarity == 'l2':
      # Negative L2 distance, scaled to similarity
      distances = torch.cdist(latents, latents, p=2)
      max_dist = distances.max() if distances.max() > 0 else 1.0
      # Convert distance to similarity: similarity = 1 - distance/max_distance  
      similarities = torch.clamp(1.0 - distances / max_dist, min=0.0, max=1.0)

    else:
      raise ValueError(f"Unknown latent_similarity: {self.latent_similarity}")

    return similarities

  def loss(self, latents, priv_info, return_stats=False):
    """
    Compute contrastive loss using linear similarity matching

    Args:
        latents: [B, T, D_lat] latent encodings
        priv_info: [B, T, D_priv] privileged information in [-1, +1] range
        loss_type: 'mse', 'kl', or 'huber' for similarity matching
        return_stats: whether to return debugging statistics

    Returns:
        loss: scalar contrastive loss
        stats: dict with debugging info (if return_stats=True)
    """
    B, T, D_lat = latents.shape
    _, _, D_priv = priv_info.shape
    batch_size_flat = B * T

    # Flatten tensors
    latents_flat = latents.view(batch_size_flat, D_lat)
    priv_flat = priv_info.view(batch_size_flat, D_priv)

    # Compute privileged information distances and similarities
    priv_distances = torch.cdist(priv_flat, priv_flat, p=2)

    # Maximum possible distance for uniform [-1, +1]: 2*sqrt(D_priv)
    max_possible_distance = 2.0 * math.sqrt(D_priv)

    # Convert privileged distances to similarity targets
    target_similarities = self.privileged_to_similarity(
        priv_distances, max_possible_distance)

    # Compute current latent similarities
    latent_similarities = self.compute_latent_similarities(latents_flat)

    # Mask out self-comparisons (diagonal)
    mask = ~torch.eye(batch_size_flat, device=latents.device).bool()

    target_sim_masked = target_similarities[mask]
    latent_sim_masked = latent_similarities[mask]

    # Compute similarity matching loss
    if self.loss_type == 'mse':
      loss = F.mse_loss(latent_sim_masked, target_sim_masked)
    elif self.loss_type == 'kl':
      # Treat similarities as probabilities (add small epsilon for numerical stability)
      eps = 1e-8
      target_probs = target_sim_masked + eps
      latent_probs = latent_sim_masked + eps
      # Normalize to make them proper probabilities
      target_probs = target_probs / target_probs.sum()
      latent_probs = latent_probs / latent_probs.sum()
      loss = F.kl_div(latent_probs.log(), target_probs, reduction='batchmean')
    elif self.loss_type == 'huber':
      loss = F.huber_loss(latent_sim_masked, target_sim_masked, delta=0.1)
    else:
      raise ValueError(f"Unknown loss_type: {self.loss_type}")

    if return_stats:
      with torch.no_grad():
        # Compute correlation between target and actual similarities
        correlation = torch.corrcoef(torch.stack(
            [target_sim_masked, latent_sim_masked]))[0, 1]

        stats = {
            'similarity_correlation': correlation.item() if not torch.isnan(correlation) else 0.0,
            'target_sim_mean': target_sim_masked.mean().item(),
            'target_sim_std': target_sim_masked.std().item(),
            'latent_sim_mean': latent_sim_masked.mean().item(),
            'latent_sim_std': latent_sim_masked.std().item(),
            'avg_priv_distance': priv_distances[mask].mean().item(),
            'max_priv_distance': max_possible_distance,
        }
      return loss, stats

    return loss

# --- vanilla 1D CNN --- #

class CNN1D(nn.Module):

  def __init__(self, 
               feature_dim, 
               timestep_dim, 
               output_dim, 
               activation_fn="elu"):
    """
    Adaptive 1D CNN that automatically adjusts architecture based on input dimensions.

    Args:
        T (int): Number of timesteps (10-50 typical)
        N (int): Feature dimension (14-25 typical)  
        output_dim (int): Final output dimension
        activation (str): Activation function ('relu', 'elu', 'tanh')
    """
    super().__init__()

    self.T = timestep_dim
    self.N = feature_dim
    self.output_dim = output_dim
    self.activation_fn = get_activation_fn(activation_fn)

    # Calculate channel progression based on N
    # Start with N input channels, expand then compress
    self.base_channels = max(16, self.N)  # Ensure minimum capacity
    self.mid_channels = min(64, 2 * self.base_channels)  # Expand but cap
    self.final_channels = max(8, self.N // 2)  # Compress for final processing

    # Build adaptive conv layers
    self.conv_layers = self._build_conv_layers()

    # Calculate output size after convolutions
    conv_output_dim = self._calculate_conv_output_dim()

    # Final linear layer
    self.fc = nn.Linear(conv_output_dim, output_dim)

  def _calculate_kernel_stride(self, input_length):
    """
    Calculate appropriate kernel size and stride for given input length.

    Expects input length from 10-50
    """
    kernel_size = max(3, min(8, input_length // 4))
    stride = max(1, kernel_size // 2)

    return kernel_size, stride

  def _build_conv_layers(self):
    """
    Build convolutional layers adaptively based on T.
    """
    layers = []
    current_length = self.T
    in_channels = self.N

    # First conv layer: expand channels
    kernel1, stride1 = self._calculate_kernel_stride(current_length)
    layers.extend([
        nn.Conv1d(in_channels=in_channels, 
                  out_channels=self.mid_channels,
                  kernel_size=kernel1, 
                  stride=stride1,
                  padding=kernel1//2),
        self.activation_fn()
    ])

    # Update current length after first conv
    current_length = (current_length + 2*(kernel1//2) - kernel1) // stride1 + 1

    # Second conv layer: maintain or reduce channels
    if current_length > 5:  # Only add if we have enough length remaining
      kernel2, stride2 = self._calculate_kernel_stride(current_length)
      layers.extend([
          nn.Conv1d(in_channels=self.mid_channels,
                    out_channels=self.base_channels, 
                    kernel_size=kernel2,
                    stride=stride2,
                    padding=kernel2//2),
          self.activation_fn()
      ])
      current_length = (current_length + 2*(kernel2//2) - kernel2) // stride2 + 1

    # Third conv layer: final compression (only if we have sufficient length)
    if current_length > 3:
      kernel3 = min(3, current_length)
      layers.extend([
          nn.Conv1d(in_channels=self.base_channels,
                    out_channels=self.final_channels,
                    kernel_size=kernel3,
                    stride=1,
                    padding=kernel3//2),
          self.activation_fn()
      ])

    layers.append(nn.Flatten())
    return nn.Sequential(*layers)

  def _calculate_conv_output_dim(self):
    """Calculate the output size after all conv layers by running a dummy forward pass."""
    with torch.no_grad():
      dummy_input = torch.randn(1, self.N, self.T)
      dummy_output = self.conv_layers(dummy_input)
      return dummy_output.shape[1]

  def forward(self, x):
    """
    Forward pass.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, N)

    Returns:
        torch.Tensor: Output tensor of shape (B, output_dim)
    """
    # Conv1d expects (B, C, L) but we receive (B, T, N)
    # We need to transpose to (B, N, T) where N becomes channels and T becomes length
    x = x.transpose(1, 2)  # (B, T, N) -> (B, N, T)

    # Apply conv layers
    x = self.conv_layers(x)

    # Apply final linear layer
    x = self.fc(x)

    return x

# --- improved 1D CNN --- #

class ChannelAttention(nn.Module):
  """Lightweight attention mechanism for temporal sequences."""

  def __init__(self, channels, reduction=8):
    super().__init__()
    self.channels = channels
    self.reduction = reduction

    # Squeeze-and-excitation style attention
    self.fc1 = nn.Linear(channels, channels // reduction)
    self.fc2 = nn.Linear(channels // reduction, channels)
    self.activation = nn.ReLU()

  def forward(self, x):
    """
    Args:
        x: (B, C, T) tensor
    Returns:
        x: (B, C, T) tensor with attention applied
    """
    # Global average pooling over time dimension
    attention = x.mean(dim=2)  # (B, C)

    # Squeeze and excitation
    attention = self.fc1(attention)
    attention = self.activation(attention)
    attention = self.fc2(attention)
    attention = torch.sigmoid(attention)

    # Apply attention
    attention = attention.unsqueeze(2)  # (B, C, 1)
    return x * attention

class UnifiedConvLayer(nn.Module):
  """Unified convolutional layer with optional multiscale, dilation, and attention components."""

  def __init__(self, 
               in_channels, 
               out_channels, 
               kernel_size=3,
               use_multiscale=True,
               use_dilation=True,
               use_attention=True,
               multiscale_kernels=[3, 5, 7],
               dilations=[1, 2, 4],
               dropout_prob=0.1,
               activation_fn=nn.ELU):
    super().__init__()

    self.use_multiscale = use_multiscale
    self.use_dilation = use_dilation
    self.use_attention = use_attention

    # Calculate how to split channels across different processing paths
    num_paths = 0
    if use_multiscale:
      num_paths += len(multiscale_kernels)
    if use_dilation:
      num_paths += len(dilations)
    if not use_multiscale and not use_dilation:
      num_paths = 1  # Standard conv

    # Each path gets equal share of output channels
    channels_per_path = out_channels // num_paths
    remaining_channels = out_channels % num_paths

    self.conv_paths = nn.ModuleList()
    self.bn_paths = nn.ModuleList()

    path_idx = 0

    # Multiscale convolutions
    if use_multiscale:
      for i, k in enumerate(multiscale_kernels):
        path_channels = channels_per_path + (remaining_channels if path_idx == 0 else 0)
        remaining_channels = 0 if path_idx == 0 else remaining_channels

        self.conv_paths.append(
            nn.Conv1d(in_channels, path_channels, k, padding=k//2)
        )
        self.bn_paths.append(nn.BatchNorm1d(path_channels))
        path_idx += 1

    # Dilated convolutions
    if use_dilation:
      for i, d in enumerate(dilations):
        path_channels = channels_per_path + (remaining_channels if path_idx == 0 else 0)
        remaining_channels = 0 if path_idx == 0 else remaining_channels

        padding = (kernel_size - 1) * d // 2
        self.conv_paths.append(
            nn.Conv1d(in_channels, path_channels, kernel_size, 
                      padding=padding, dilation=d)
        )
        self.bn_paths.append(nn.BatchNorm1d(path_channels))
        path_idx += 1

    # Standard convolution (if neither multiscale nor dilation)
    if not use_multiscale and not use_dilation:
      self.conv_paths.append(
          nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
      )
      self.bn_paths.append(nn.BatchNorm1d(out_channels))

    # Attention mechanism
    if use_attention:
      self.attention = ChannelAttention(out_channels)

    # Layer normalization and residual projection
    self.layer_norm = nn.LayerNorm(out_channels)

    # Residual connection projection (if channel dimensions don't match)
    if in_channels != out_channels:
      self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)
    else:
      self.residual_proj = None

    self.activation = activation_fn()
    self.dropout = nn.Dropout(dropout_prob)

  def forward(self, x):
    """
    Args:
        x: (B, C, T) tensor
    Returns:
        x: (B, out_channels, T) tensor
    """
    residual = x

    # Apply convolution paths
    path_outputs = []
    for conv, bn in zip(self.conv_paths, self.bn_paths):
      path_out = conv(x)
      path_out = bn(path_out)
      path_out = self.activation(path_out)
      path_outputs.append(path_out)

    # Concatenate all paths
    out = torch.cat(path_outputs, dim=1)

    # Apply attention if enabled
    if self.use_attention:
      out = self.attention(out)

    # Apply dropout
    out = self.dropout(out)

    # Residual connection
    if self.residual_proj is not None:
      residual = self.residual_proj(residual)

    out = out + residual

    # Layer normalization (convert to (B, T, C) for LayerNorm, then back)
    out = out.transpose(1, 2)  # (B, C, T) -> (B, T, C)
    out = self.layer_norm(out)
    out = out.transpose(1, 2)  # (B, T, C) -> (B, C, T)

    return out

class LargeCNN1D(nn.Module):
  
  def __init__(self, 
               feature_dim, 
               timestep_dim, 
               output_dim, 
               num_layers=None,
               base_channels=32,
               channel_multiplier=1.5,
               activation_fn="elu",
               dropout_prob=0.2,
               layer_configs=None):
    """
    Enhanced 1D CNN with unified layer architecture.

    Args:
        feature_dim (int): Number of input features
        timestep_dim (int): Number of timesteps
        output_dim (int): Final output dimension
        num_layers (int): Number of unified conv layers
        base_channels (int): Base number of channels
        channel_multiplier (float): Channel growth factor between layers
        activation_fn (str): Activation function name
        layer_configs (list): List of dicts specifying config for each layer.
                             If None, uses sensible defaults.
                             Each dict can have keys: 'use_multiscale', 'use_dilation', 'use_attention'
    """
    super().__init__()

    self.T = timestep_dim
    self.N = feature_dim
    self.output_dim = output_dim
    self.activation_fn = get_activation_fn(activation_fn)

    # Default layer configurations if not provided
    if layer_configs is None:
      layer_configs = self._get_default_layer_configs(num_layers, timestep_dim)

    if num_layers is None:
      self.num_layers = len(layer_configs)
    elif num_layers != len(layer_configs):
      raise RuntimeError(f"LargeCNN1D.__init__() error: "
                         f"num_layers={num_layers}, layer_configs length="
                         f"{len(layer_configs)}, these must match, or set "
                         f"either of them to None")

    # Ensure we have config for each layer
    assert len(layer_configs) == num_layers, "layer_configs must have same length as num_layers"
    self.layer_configs = layer_configs

    # Calculate channel progression
    self.channels = [feature_dim]
    for i in range(num_layers):
      next_channels = int(base_channels * (channel_multiplier ** i))
      self.channels.append(next_channels)

    # Build unified layers
    self.layers = nn.ModuleList()
    for i in range(num_layers):
      config = layer_configs[i]

      # Adjust dilations based on layer depth and timestep length
      dilations = self._get_dilations_for_layer(i, timestep_dim)

      layer = UnifiedConvLayer(
          in_channels=self.channels[i],
          out_channels=self.channels[i+1],
          kernel_size=3,
          use_multiscale=config.get('use_multiscale', True),
          use_dilation=config.get('use_dilation', True),
          use_attention=config.get('use_attention', True),
          multiscale_kernels=config.get('multiscale_kernels', [3, 5, 7]),
          dilations=dilations,
          dropout_prob=dropout_prob if i > 0 else 0.0, # skip dropout on first layer
          activation_fn=self.activation_fn
      )
      self.layers.append(layer)

    # Global pooling and classifier
    final_channels = self.channels[-1]
    self.global_pool = nn.AdaptiveAvgPool1d(1)
    self.classifier = nn.Sequential(
        nn.Linear(final_channels, final_channels // 2),
        nn.BatchNorm1d(final_channels // 2),
        self.activation_fn(),
        nn.Dropout(dropout_prob),
        nn.Linear(final_channels // 2, output_dim)
    )

  def _get_default_layer_configs(self, num_layers, timestep_dim):
    """Generate sensible default configurations for each layer."""
    configs = []

    for i in range(num_layers):
      # First layer: Focus on multiscale feature extraction
      if i == 0:
        config = {
            'use_multiscale': True,
            'use_dilation': False,
            'use_attention': False,
            'multiscale_kernels': [3, 5, 7]
        }
      # Middle layers: Use dilation for long-range dependencies
      elif i < num_layers - 1:
        config = {
            'use_multiscale': False,
            'use_dilation': True,
            # attention every other layer, except 2nd
            'use_attention': (i % 2 == 1) * (i != 1),  
        }
      # Last layer: Focus on attention and refinement
      else:
        config = {
            'use_multiscale': False,
            'use_dilation': False,
            'use_attention': True,
        }

      configs.append(config)

    return configs

  def _get_dilations_for_layer(self, layer_idx, timestep_dim):
    """Calculate appropriate dilations for each layer."""

    if timestep_dim < 15:
      max_dilation_power = 1 # [1, 2]
    elif timestep_dim < 30:
      max_dilation_power = 2 # [1, 2, 4]
    elif timestep_dim < 60:
      max_dilation_power = 3 # [1, 2, 4, 8]
    else: # timestep_dim >= 60 (Long sequences)
      # E.g., for layer 0: [1..16], layer 1: [1..32], etc.
      max_dilation_power = min(4 + layer_idx, 6)

    # gnerate dilations using a list comprehension
    dilations = [2 ** p for p in range(max_dilation_power + 1)]

    return dilations

  def forward(self, x):
    """
    Forward pass.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, N)

    Returns:
        torch.Tensor: Output tensor of shape (B, output_dim)
    """
    # Conv1d expects (B, C, L) but we receive (B, T, N)
    # Transpose to (B, N, T) where N becomes channels and T becomes length
    x = x.transpose(1, 2)  # (B, T, N) -> (B, N, T)

    # Apply unified layers
    for layer in self.layers:
      x = layer(x)

    # Global pooling and classification
    x = self.global_pool(x)  # (B, C, 1)
    x = x.squeeze(2)  # (B, C)
    x = self.classifier(x)

    return x

# Example usage and comparison
if __name__ == "__main__":
  # Example parameters for a robot with 20 features and up to 50 timesteps
  feature_dim = 20
  timestep_dim = 50  # Updated to test with longer sequences
  output_dim = 10
  batch_size = 8

  # Create models
  original_model = CNN1D(feature_dim, timestep_dim, output_dim)

  # large model
  new_model = LargeCNN1D(
      feature_dim=feature_dim, 
      timestep_dim=timestep_dim, 
      output_dim=output_dim, 
      num_layers=6,
  )

  # Test input
  x = torch.randn(batch_size, timestep_dim, feature_dim)

  # Forward pass
  with torch.no_grad():
    output1 = original_model(x)
    output2 = new_model(x)

  print(f"Original model output shape: {output1.shape}")
  print(f"New model output shape: {output2.shape}")
  print(
      f"Original model parameters: {sum(p.numel() for p in original_model.parameters())}")
  print(f"New model parameters: {sum(p.numel() for p in new_model.parameters())}")