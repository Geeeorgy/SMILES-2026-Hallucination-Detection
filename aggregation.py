"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    

    L, T, D = hidden_states.shape
    real_positions = attention_mask.nonzero(as_tuple=False)
    # Split layers into early, intermediate and late
    s1 = L // 3
    s2 = 2 * L // 3

    first_pos = int(real_positions[0].item())
    last_pos = int(real_positions[-1].item())

    # Pool over layers
    early = hidden_states[:s1]
    early_pooled = early.mean(dim=0)
    early_mean = early_pooled[first_pos:last_pos+1].mean(dim=0)
    early_var = early_pooled[first_pos:last_pos+1].var(dim=0)

    # Pool over intermediate layers
    inter = hidden_states[s1:s2]
    inter_pooled = inter.mean(dim=0)
    inter_mean = inter_pooled[first_pos:last_pos+1].mean(dim=0)
    inter_var = inter_pooled[first_pos:last_pos+1].var(dim=0)
    
    # Pool over last layers
    late = hidden_states[s2:]
    late_pooled = late.mean(dim=0)
    late_mean = late_pooled[first_pos:last_pos+1].mean(dim=0)
    late_var = late_pooled[first_pos:last_pos+1].var(dim=0)

    feature = torch.cat([early_mean, early_var,
                          inter_mean, inter_var,
                            late_mean, late_var])


    return feature
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # tensor shape = L,T,D
    L, T, D = hidden_states.shape
    real_positions = attention_mask.nonzero(as_tuple=False)

    first_pos = int(real_positions[0].item())
    last_pos = int(real_positions[-1].item())

    hidden_states = hidden_states[:, first_pos:last_pos+1]

    # cosine similarity vs the last layer
    last_layer = hidden_states[-1]
    mean_cos = []
    var_cos = []

    # Layer norm mean and var
    norm_mean = []
    norm_var = []
    for l in range(L):
      if l != 1:
        cos = F.cosine_similarity(hidden_states[l], last_layer, dim=-1)
        mean_cos.append(cos.mean())
        var_cos.append(cos.var())
    
        cos_mean_feats = torch.stack(mean_cos)
        cos_var_feats = torch.stack(var_cos)
      norms = torch.norm(hidden_states[l], dim = -1)
      norm_mean.append(norms.mean())
      norm_var.append(norms.var())
    
    geometric_feats = torch.cat([torch.stack(mean_cos),
                                torch.stack(var_cos),
                                torch.stack(norm_mean),
                                torch.stack(norm_var)])


    return geometric_feats


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
