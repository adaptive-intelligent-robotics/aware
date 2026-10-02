# python imports
import logging; pylogger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import math

# --- attention encoder --- #

# --- reworked attention encoder --- #

class TemporalSelfAttention2(nn.Module):
  """
  Self-attention module that operates on temporal sequences.
  """
  def __init__(self, 
               hidden_dim: int, 
               num_heads: int = 4, 
               dropout: float = 0.1,
               use_causal_mask: bool = False):
    super().__init__()

    if hidden_dim % num_heads != 0:
      raise ValueError(
          f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.use_causal_mask = use_causal_mask

    self.attention = nn.MultiheadAttention(
        embed_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True
    )

    # Output projection and normalization
    self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    self.norm1 = nn.LayerNorm(hidden_dim)
    self.norm2 = nn.LayerNorm(hidden_dim)

    # Feed-forward network
    self.ffn = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 4, hidden_dim),
        nn.Dropout(dropout)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape [B, T, hidden_dim]
    Returns:
        output: Attention-processed tensor of shape [B, T, hidden_dim]
    """
    
    # if we enforce temporal consistency, where steps can only attend backwards in time
    attn_mask = None
    if self.use_causal_mask:
      seq_len = x.size(1)
      attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

    # Self-attention with residual connection
    attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
    x = self.norm1(attn_output + x)

    # Feed-forward with residual connection
    ffn_output = self.ffn(x)
    output = self.norm2(ffn_output + x)

    return output

class PositionalEncodingAttEnc2(nn.Module):
  """
  Positional encoding for temporal sequences.
  Supports both sinusoidal and learnable positional encodings.
  """
  def __init__(self, hidden_dim: int, max_len: int = 1000, dropout: float = 0.1, 
               learnable: bool = False):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    self.learnable = learnable

    if learnable:
      # Learnable positional embeddings
      pos_init_std = 0.1
      self.pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden_dim) * pos_init_std)
    else:
      # Sinusoidal positional encoding
      position = torch.arange(max_len).unsqueeze(1).float()
      div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-math.log(10000.0) / hidden_dim))

      pe = torch.zeros(1, max_len, hidden_dim)
      pe[0, :, 0::2] = torch.sin(position * div_term)

      # Fixed: ensure we don't go out of bounds for cosine terms
      if hidden_dim % 2 == 1:
        pe[0, :, 1::2] = torch.cos(position * div_term)
      else:
        pe[0, :, 1::2] = torch.cos(position * div_term)

      self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape [B, T, hidden_dim]
    """
    seq_len = x.size(1)

    if self.learnable:
      x = x + self.pos_embedding[:, :seq_len, :]
    else:
      x = x + self.pe[:, :seq_len, :]

    return self.dropout(x)

class TemporalAttentionEncoder2(nn.Module):
  """
  Multi-layer temporal attention encoder.
  """
  def __init__(self, 
               input_dim: int, 
               hidden_dim: int, 
               num_layers: int = 2, 
               num_heads: int = 4, 
               dropout: float = 0.1,
               pos_encoding: bool = True,
               learnable_pos: bool = False,
               use_causal_mask: bool = False):
    super().__init__()

    if num_layers < 1:
      raise ValueError("num_layers must be at least 1")

    self.input_projection = nn.Linear(input_dim, hidden_dim)
    self.add_positional_encoding = pos_encoding

    # Positional encoding (applied AFTER input projection but BEFORE attention)
    if pos_encoding:
      self.pos_encoder = PositionalEncodingAttEnc2(
          hidden_dim=hidden_dim, 
          dropout=dropout,
          learnable=learnable_pos
      )

    # All attention layers now work with hidden_dim consistently
    self.attention_layers = nn.ModuleList([
        TemporalSelfAttention2(
            hidden_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            use_causal_mask=use_causal_mask,
        ) for _ in range(num_layers)
    ])

  def forward(self, x: torch.Tensor, prepend_token: torch.Tensor = None) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape [B, T, input_dim]
        prepend_token: An optional token of shape [1, 1, hidden_dim] to prepend.
    Returns:
        encoded: Output tensor of shape [B, T, hidden_dim] (T+1 if prepend_token used)
    """
    batch_size = x.shape[0]

    # Project input to hidden dimension
    x = self.input_projection(x)

    # + If a token is provided, prepend it to the sequence.
    if prepend_token is not None:
      # Expand the token to match the batch size
      expanded_token = prepend_token.expand(batch_size, -1, -1)
      x = torch.cat((expanded_token, x), dim=1)

    # Apply positional encoding AFTER projection and potential token prepending
    if self.add_positional_encoding:
      x = self.pos_encoder(x)

    # Apply attention layers
    for layer in self.attention_layers:
      x = layer(x)

    return x

class AttentionHistoryEncoder2(nn.Module):
  """
  Complete attention-based encoder for temporal state histories.

  Args:
      input_dim: Feature dimension of each timestep
      output_dim: Output feature dimension
      hidden_dim: Hidden dimension for attention layers
      num_layers: Number of attention layers
      num_heads: Number of attention heads
      dropout: Dropout probability
      pos_encoding: Whether to use positional encoding
      learnable_pos: Whether to use learnable positional encodings (vs sinusoidal)
      use_causal_mask: If True, applies a causal mask to prevent attending to future steps.
      pooling_method: How to aggregate temporal information ('last', 'mean', 'max', 'none', 'cls')
  """

  def __init__(self,
               input_dim: int,
               output_dim: int,
               hidden_dim: int = 64,
               num_layers: int = 2,
               num_heads: int = 4,
               dropout: float = 0.1,
               pos_encoding: bool = True,
               learnable_pos: bool = False,
               use_causal_mask: bool = False,
               pooling_method: str = 'last',
               device: str = "cuda"):

    super().__init__()

    # Validation
    valid_pooling = {'last', 'mean', 'max', 'none', 'cls'}
    if pooling_method not in valid_pooling:
      raise ValueError(f"pooling_method must be one of {valid_pooling}")

    self.pooling_method = pooling_method
    self.device = device
    self.cls_token = None

    if self.pooling_method == "cls":
      self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    # Core components
    self.temporal_encoder = TemporalAttentionEncoder2(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        pos_encoding=pos_encoding,
        learnable_pos=learnable_pos,
        use_causal_mask=use_causal_mask,
    )

    self.output_layer = nn.Linear(hidden_dim, output_dim)

    self.to(self.device)

  def _validate_input(self, x: torch.Tensor) -> None:
    """Validate input tensor shape and properties."""
    if x.dim() != 3:
      raise ValueError(f"Expected 3D input tensor [B, T, N], got {x.dim()}D tensor")

    batch_size, seq_len, feature_dim = x.shape
    expected_feature_dim = self.temporal_encoder.input_projection.in_features

    if feature_dim != expected_feature_dim:
      raise ValueError(
          f"Expected feature dimension {expected_feature_dim}, got {feature_dim}"
      )

    if seq_len == 0:
      raise ValueError("Sequence length cannot be 0")

  def _pool_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
    """
    Pool temporal features according to specified method.

    Args:
        x: Tensor of shape [B, T, hidden_dim]
    Returns:
        pooled: Tensor of shape [B, hidden_dim]
                or shape (B, T, hidden_dim) if self.pooling_method='none'
    """
    if self.pooling_method == 'last':
      return x[:, -1, :]
    elif self.pooling_method == 'mean':
      return x.mean(dim=1)
    elif self.pooling_method == 'max':
      return x.max(dim=1)[0]
    elif self.pooling_method == "cls":
      return x[:, 0, :] # select the output of the CLS token at index 0
    elif self.pooling_method == "none":
      return x # keep all timesteps
    else:
      raise ValueError(f"Unknown pooling method: {self.pooling_method}")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the attention encoder.

    Args:
        x: Input tensor of shape [B, T, input_dim]
           B = batch size
           T = sequence length (time steps)  
           input_dim = feature dimension at each timestep

    Returns:
        output: Encoded tensor of shape [B, output_dim]  
                or shape (B, T, output_dim) if self.pooling_method='none'
    """
    self._validate_input(x)

    # + The forward pass is now simplified.
    # + We pass the raw input `x` and conditionally pass the `cls_token`.
    # + The temporal_encoder now handles the logic internally.
    token_to_pass = self.cls_token if self.pooling_method == 'cls' else None
    x = self.temporal_encoder(x, prepend_token=token_to_pass)

    # Pool temporal information
    x = self._pool_temporal_features(x)  # [B, hidden_dim]

    # Final output projection
    output = self.output_layer(x)  # [B, output_dim]

    return output

  def to(self, device):
    """
    Move to the specified device
    """
    self.device = device
    super().to(device)
    return self

# --- transformer --- #

# --- GRU version of AttentionHistoryEncoder2 --- #
