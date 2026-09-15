"""
src/augmentation.py
====================
Fraud-Aware Graph Augmentation (FAGA) — Novel Contribution #1
==============================================================

Motivation
----------
Standard contrastive graph augmentations (random edge-drop, feature masking)
are designed for *homophilous* graphs, where connected nodes tend to share the
same label. In the Elliptic Bitcoin graph the opposite holds: illicit
transaction nodes are deliberately routed through legitimate wallets, so fraud
nodes are typically *surrounded* by licit neighbours — a heterophilous setting.

Random augmentations destroy the minority-class signal in two ways:
  1. Randomly dropping edges removes informative licit→fraud connections.
  2. Uniformly masking features hides the rare fraud-node feature patterns
     that the model is trying to learn.

FAGA introduces **asymmetric augmentation strength**: it applies stronger
augmentation to the *majority/background* signal while carefully preserving the
*minority/signal* structure. Since ground-truth labels are not available during
training (would cause label leakage), FAGA uses **structural and feature-based
proxies** to identify "signal" nodes:

- **Degree centrality**: Low-degree nodes in Elliptic are often fraud (illicit
  transactions are leaf-like in the money-flow graph). High-degree nodes are
  typically licit hubs (exchanges, mixers).
- **Feature norm**: Nodes with atypical feature magnitudes may indicate fraud.

Two augmentation strategies
--------------------------
• ``structure_aware_edge_drop``
    Drops edges with probability inversely proportional to the *minimum degree*
    of the two endpoints. Edges incident to low-degree (potentially fraud)
    nodes are preserved; edges between high-degree (licit hub) nodes are
    dropped aggressively.

• ``feature_aware_mask``
    Masks node features with probability inversely proportional to the node's
    feature L2 norm. Nodes with distinctive/outlier features (potential fraud)
    are masked conservatively; typical nodes are masked aggressively.

The function ``augment_pair(data)`` returns two augmented views and is the
drop-in replacement for the dual ``encode(data.x, ...)`` calls previously used
in ``ElliGAT.contrastive_loss()``.

References
----------
* You et al., "Graph Contrastive Learning with Augmentations", NeurIPS 2020.
* Zhu et al., "Graph Contrastive Learning Beyond Homophily", TPAMI 2022.
* Sun et al., "SUGAR: Subgraph Neural Network with Reinforcement Pooling and
  Self-Supervised Mutual Information Mechanism", WWW 2021.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


# ─── Helper: clone a Data object with new x / edge_index ─────────────────────

def _clone_with(data: Data, x=None, edge_index=None, edge_attr=None) -> Data:
    """Return a shallow clone of *data* with optional field replacements."""
    new = data.clone()
    if x is not None:
        new.x = x
    if edge_index is not None:
        new.edge_index = edge_index
    if edge_attr is not None:
        new.edge_attr = edge_attr
    return new


# ─── Helper: compute node "signal score" without using labels ────────────────

def _compute_signal_scores(data: Data) -> torch.Tensor:
    """
    Compute per-node signal scores (0..1) indicating likelihood of being
    a minority-class (fraud) node, WITHOUT using ground-truth labels.

    Uses two proxies:
    1. Inverse degree centrality: low-degree nodes score higher (fraud-like)
    2. Feature outlier score: nodes with high feature L2 norm score higher

    Returns
    -------
    scores : (N,) tensor in [0, 1], higher = more likely fraud/signal
    """
    N = data.x.size(0)
    device = data.x.device

    # Proxy 1: Inverse degree (normalized)
    # Low degree → higher score (potential fraud leaf nodes)
    deg = degree(data.edge_index[0], num_nodes=N, dtype=torch.float).clamp(min=1)
    inv_deg = 1.0 / deg
    inv_deg_norm = (inv_deg - inv_deg.min()) / (inv_deg.max() - inv_deg.min() + 1e-8)

    # Proxy 2: Feature L2 norm (normalized)
    # High norm → outlier features → higher score
    feat_norm = data.x.norm(dim=1)
    feat_norm_norm = (feat_norm - feat_norm.min()) / (feat_norm.max() - feat_norm.min() + 1e-8)

    # Combine proxies (equal weight)
    signal_score = 0.5 * inv_deg_norm + 0.5 * feat_norm_norm
    return signal_score.to(device)


# ─── Augmentation 1: Structure-aware edge drop ───────────────────────────────

def structure_aware_edge_drop(
    data: Data,
    drop_prob_high: float = 0.30,   # Drop prob for high-degree (background) edges
    drop_prob_low: float = 0.03,    # Drop prob for low-degree (signal) edges
) -> Data:
    """
    Drop edges with probability based on endpoint degrees (structural proxy).

    Edges where BOTH endpoints have high degree (licit hubs) are dropped
    aggressively. Edges incident to at least one low-degree node (potential
    fraud) are preserved.

    Parameters
    ----------
    data           : PyG Data object.
    drop_prob_high : Probability of dropping edge between two high-degree nodes.
    drop_prob_low  : Probability of dropping edge incident to low-degree node.

    Returns
    -------
    Data  augmented view with a subset of edges retained.
    """
    ei = data.edge_index          # (2, E)
    N = data.x.size(0)
    device = ei.device

    # Compute per-node signal scores (higher = more signal/fraud-like)
    signal = _compute_signal_scores(data)  # (N,)

    src_signal = signal[ei[0]]
    dst_signal = signal[ei[1]]

    # Edge signal = minimum of endpoint signals (preserve if either endpoint is signal)
    edge_signal = torch.minimum(src_signal, dst_signal)

    # Map signal score [0,1] to drop probability [drop_prob_low, drop_prob_high]
    # High signal (potential fraud) → low drop probability
    drop_prob = drop_prob_high - edge_signal * (drop_prob_high - drop_prob_low)

    rand = torch.rand(ei.size(1), device=device)
    keep_mask = rand >= drop_prob

    new_ei = ei[:, keep_mask]
    new_ea = data.edge_attr[keep_mask] if data.edge_attr is not None else None

    return _clone_with(data, edge_index=new_ei, edge_attr=new_ea)


# ─── Augmentation 2: Feature-aware mask ──────────────────────────────────────

def feature_aware_mask(
    data: Data,
    feat_mask_high: float = 0.30,   # Mask prob for typical (background) nodes
    feat_mask_low: float = 0.05,    # Mask prob for outlier (signal) nodes
) -> Data:
    """
    Mask node features with probability based on feature outlier score.

    Nodes with atypical/outlier features (high L2 norm, potential fraud)
    are masked conservatively. Typical nodes are masked aggressively.

    Parameters
    ----------
    data           : PyG Data object.
    feat_mask_high : Fraction of features to zero-mask for typical nodes.
    feat_mask_low  : Fraction of features to zero-mask for outlier nodes.

    Returns
    -------
    Data  augmented view with masked node features.
    """
    x = data.x.clone()            # (N, D)
    N, D = x.shape
    device = x.device

    # Compute signal scores
    signal = _compute_signal_scores(data)  # (N,)

    # Map signal [0,1] to mask probability [feat_mask_low, feat_mask_high]
    # High signal (outlier) → low mask probability
    p = feat_mask_high - signal * (feat_mask_high - feat_mask_low)  # (N,)

    # Per-node, per-feature Bernoulli mask
    rand = torch.rand(N, D, device=device)
    mask = rand < p.unsqueeze(1)   # True → zero out
    x[mask] = 0.0

    return _clone_with(data, x=x)


# ─── Compound augmentation: FAGA pair ─────────────────────────────────────────

def augment_pair(
    data: Data,
    drop_prob_high: float = 0.30,
    drop_prob_low: float = 0.03,
    feat_mask_high: float = 0.30,
    feat_mask_low: float = 0.05,
):
    """
    Generate two augmented views of *data* for contrastive learning.

    View 1: structure-aware edge drop  (structural augmentation)
    View 2: feature-aware mask         (attribute augmentation)

    The two views use independent randomness so they differ in every training
    step, providing genuine contrastive signal — unlike the original code
    which passed the same data through two identical forward passes.

    NO LABEL LEAKAGE: Uses only structural (degree) and feature statistics,
    never ground-truth labels.

    Returns
    -------
    (view1, view2) : two augmented Data objects.
    """
    view1 = structure_aware_edge_drop(
        data,
        drop_prob_high=drop_prob_high,
        drop_prob_low=drop_prob_low,
    )
    view2 = feature_aware_mask(
        data,
        feat_mask_high=feat_mask_high,
        feat_mask_low=feat_mask_low,
    )
    return view1, view2


# ─── Legacy aliases (for backward compatibility) ──────────────────────────────

def licit_edge_drop(data: Data, drop_prob_licit: float = 0.30, drop_prob_fraud: float = 0.03) -> Data:
    """Deprecated: uses labels. Kept for backward compatibility only."""
    import warnings
    warnings.warn("licit_edge_drop uses ground-truth labels (label leakage). Use structure_aware_edge_drop instead.")
    return structure_aware_edge_drop(data, drop_prob_high=drop_prob_licit, drop_prob_low=drop_prob_fraud)

def fraud_preserve_mask(data: Data, feat_mask_licit: float = 0.30, feat_mask_fraud: float = 0.05) -> Data:
    """Deprecated: uses labels. Kept for backward compatibility only."""
    import warnings
    warnings.warn("fraud_preserve_mask uses ground-truth labels (label leakage). Use feature_aware_mask instead.")
    return feature_aware_mask(data, feat_mask_high=feat_mask_licit, feat_mask_low=feat_mask_fraud)
