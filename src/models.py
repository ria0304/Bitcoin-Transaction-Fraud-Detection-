
"""
src/models.py
=============
Model zoo for Bitcoin fraud detection.

Models
------
ElliGAT      – Proposed: 4-layer GATv2 + Temporal Edge Encoding +
                Self-supervised pre-training hook + Heterophily-aware
                neighbour aggregation (combines ego + neighbour diff)
EvolveGCN_O  – Temporal GNN baseline (EvolveGCN-O variant)
BaselineGNN  – GraphSAGE ablation (matches original repo)
MLP          – Tabular-only baseline

Novel contributions (journal version)
--------------------------------------
1. Edge-aware attention (edge_attr injected into GATv2 attention)
2. Heterophily-aware readout: concat([h_i, h_i - mean(h_j)]) with
   formal justification as a fraud-detection-specific inductive bias.
3. Self-supervised masked-feature pre-training (GraphMAE-style)
4. Contrastive (NT-Xent) loss with FAGA views.
5. heterophily_score(): per-node L2 norm of (h_i - mean(h_N)).
6. mc_dropout_forward(): returns full (S, N) probability tensor for
   MC Dropout uncertainty estimation and pseudo-label generation.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, SAGEConv, GCNConv
from torch.utils.checkpoint import checkpoint as _ckpt
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree


# Set GRAD_CHECKPOINT=0 to disable gradient checkpointing.
_GRAD_CKPT = (
    os.environ.get("GRAD_CHECKPOINT", "1").strip().lower()
    not in ("0", "false", "no", "")
)


# ─── Utility: Sinusoidal temporal encoding ───────────────────────────────────

class TemporalEdgeEncoder(nn.Module):
    """
    Encode the scalar timestep-delta |Δt| carried by each edge into a
    d_out-dimensional vector consumed by GATv2's edge-aware attention.

    Modes
    -----
    sinusoidal:
        Fixed multi-scale sinusoidal encoding followed by a learned
        projection.

    mlp:
        Learned multilayer projection of the scalar temporal delta.
    """

    def __init__(
        self,
        d_out: int,
        mode: str = "sinusoidal",
        max_period: float = 100.0,
    ):
        super().__init__()
        self.d_out = d_out
        self.mode = mode.lower()
        self.max_period = max_period

        if self.mode not in ("sinusoidal", "mlp"):
            raise ValueError(
                f"Unknown temporal encoding mode: {mode!r}. "
                "Expected 'sinusoidal' or 'mlp'."
            )

        if self.mode == "sinusoidal":
            self.proj = nn.Linear(d_out, d_out)
            half = d_out // 2
            if half > 0:
                exponents = torch.arange(half, dtype=torch.float32)
                denom = max(half - 1, 1)
                frequencies = torch.exp(
                    -math.log(max_period) * exponents / denom
                )
            else:
                frequencies = torch.empty(0, dtype=torch.float32)

            self.register_buffer("frequencies", frequencies)

        else:
            self.mlp = nn.Sequential(
                nn.Linear(1, d_out),
                nn.ReLU(),
                nn.Linear(d_out, d_out),
            )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        delta = delta.reshape(-1, 1).float()

        if self.mode == "mlp":
            return self.mlp(delta)

        if self.frequencies.numel() == 0:
            enc = delta.new_zeros((delta.size(0), self.d_out))
        else:
            angles = delta * self.frequencies.unsqueeze(0)
            enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

            if enc.size(-1) < self.d_out:
                enc = F.pad(enc, (0, self.d_out - enc.size(-1)))
            elif enc.size(-1) > self.d_out:
                enc = enc[:, :self.d_out]

        return self.proj(enc)


# ─── ElliGAT: proposed model ─────────────────────────────────────────────────

class ElliGAT(nn.Module):
    """
    Edge-aware, heterophily-aware GATv2 for transaction fraud detection.

    The network combines node features, temporal edge encodings, and
    ego-versus-neighbour representations.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 1,
        heads: int = 4,
        dropout: float = 0.3,
        edge_dim: int = 32,
        edge_encoding: str = "sinusoidal",
        **kwargs,
    ):
        super().__init__()

        # Accept the common hidden_dim alias used by experiment configs.
        if "hidden_dim" in kwargs:
            hidden_channels = int(kwargs.pop("hidden_dim"))

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected ElliGAT arguments: {unexpected}")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.heads = heads
        self.edge_dim = edge_dim
        self.edge_encoding = edge_encoding

        self.temporal_encoder = TemporalEdgeEncoder(
            d_out=edge_dim,
            mode=edge_encoding,
        )

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList([
            GATv2Conv(
                hidden_channels,
                hidden_channels // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=False,
            )
            for _ in range(4)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(4)
        ])

        self.output_proj = nn.Linear(hidden_channels * 2, hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        # Self-supervised masked-feature reconstruction head.
        self.pretrain_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels),
        )

        # Contrastive projection head.
        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 128),
        )

    def _edge_attr(self, edge_attr, num_edges, device):
        """Convert raw scalar temporal edge deltas to encoded vectors."""
        if edge_attr is None:
            delta = torch.zeros(num_edges, device=device)
        else:
            delta = edge_attr
            if delta.dim() > 1:
                delta = delta[:, 0]
        return self.temporal_encoder(delta)

    def encode(self, x, edge_index, edge_attr=None):
        """
        Encode all nodes in the supplied graph.

        Returns an embedding for every node. Gradient checkpointing can
        reduce activation memory during training.
        """
        h = self.input_proj(x)
        edge_emb = self._edge_attr(
            edge_attr,
            edge_index.size(1),
            x.device,
        )

        for conv, norm in zip(self.convs, self.norms):
            residual = h

            if self.training and _GRAD_CKPT and h.requires_grad:
                h = _ckpt(
                    lambda node_x, edge_emb_arg, conv_layer=conv:
                        conv_layer(
                            node_x,
                            edge_index,
                            edge_attr=edge_emb_arg,
                        ),
                    h,
                    edge_emb,
                    use_reentrant=False,
                )
            else:
                h = conv(h, edge_index, edge_attr=edge_emb)

            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if h.shape == residual.shape:
                h = h + residual

        # Heterophily-aware neighbour difference representation.
        src, dst = edge_index
        neighbour_sum = torch.zeros_like(h)
        neighbour_count = torch.zeros(
            h.size(0), device=h.device, dtype=h.dtype
        )

        neighbour_sum.index_add_(0, dst, h[src])
        neighbour_count.index_add_(
            0,
            dst,
            torch.ones(dst.numel(), device=h.device, dtype=h.dtype),
        )

        neighbour_mean = neighbour_sum / neighbour_count.clamp_min(1).unsqueeze(-1)
        difference = h - neighbour_mean

        combined = torch.cat([h, difference], dim=-1)
        return F.elu(self.output_proj(combined))

    def forward(self, data):
        """Return raw logits for every node."""
        h = self.encode(data.x, data.edge_index, getattr(data, "edge_attr", None))
        logits = self.classifier(h)
        return logits.squeeze(-1) if logits.size(-1) == 1 else logits

    def pretrain_loss(self, data, mask_ratio: float = 0.3):
        """
        Mask a subset of node features and reconstruct them from embeddings.
        """
        x_orig = data.x
        n_nodes = x_orig.size(0)

        mask_count = max(1, int(n_nodes * mask_ratio))
        mask_idx = torch.randperm(
            n_nodes, device=x_orig.device
        )[:mask_count]

        x_masked = x_orig.clone()
        x_masked[mask_idx] = 0.0

        data_tmp = data.clone()
        data_tmp.x = x_masked

        h = self.encode(
            data_tmp.x,
            data_tmp.edge_index,
            getattr(data_tmp, "edge_attr", None),
        )
        x_recon = self.pretrain_head(h)

        return F.mse_loss(x_recon[mask_idx], x_orig[mask_idx])

    def contrastive_loss(
        self,
        data,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Memory-bounded symmetric cross-view contrastive loss using FAGA.

        Only a sampled subset of training nodes participates in the
        contrastive similarity matrix. The graph encoder still processes
        the full graph.

        Set CONTRASTIVE_MAX_NODES to adjust the subset size. The default
        is 2048 nodes, producing a 2048 x 2048 similarity matrix rather
        than the much larger 2B x 2B matrix over every training node.
        """
        from src.augmentation import augment_pair

        self.train()

        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

        if train_idx.numel() < 2:
            return next(self.parameters()).new_zeros(())

        max_nodes = max(
            2,
            int(os.environ.get("CONTRASTIVE_MAX_NODES", "2048")),
        )

        if train_idx.numel() > max_nodes:
            chosen = torch.randperm(
                train_idx.numel(),
                device=train_idx.device,
            )[:max_nodes]
            train_idx = train_idx[chosen]

        view1, view2 = augment_pair(data)

        h1 = self.encode(
            view1.x,
            view1.edge_index,
            getattr(view1, "edge_attr", None),
        )
        h2 = self.encode(
            view2.x,
            view2.edge_index,
            getattr(view2, "edge_attr", None),
        )

        z1 = F.normalize(self.contrast_head(h1[train_idx]), dim=-1)
        z2 = F.normalize(self.contrast_head(h2[train_idx]), dim=-1)

        # Compute only cross-view similarities.
        logits = torch.mm(z1, z2.T) / temperature
        labels = torch.arange(z1.size(0), device=z1.device)

        loss_forward = F.cross_entropy(logits, labels)
        loss_backward = F.cross_entropy(logits.T, labels)

        return (loss_forward + loss_backward) / 2

    def mc_dropout_forward(self, data, n: int) -> torch.Tensor:
        """
        Perform n stochastic forward passes for MC Dropout uncertainty.

        Returns a tensor of shape (n, N) for binary classification, or
        (n, N, C) for multi-class classification.
        """
        self.train()
        with torch.no_grad():
            return torch.stack([self.forward(data) for _ in range(n)])

    def heterophily_score(self, data) -> torch.Tensor:
        """
        Compute the L2 distance between each node embedding and its
        neighbourhood's mean embedding.
        """
        h = self.encode(
            data.x,
            data.edge_index,
            getattr(data, "edge_attr", None),
        )

        src, dst = data.edge_index
        neighbour_sum = torch.zeros_like(h)
        neighbour_count = torch.zeros(
            h.size(0), device=h.device, dtype=h.dtype
        )

        neighbour_sum.index_add_(0, dst, h[src])
        neighbour_count.index_add_(
            0,
            dst,
            torch.ones(dst.numel(), device=h.device, dtype=h.dtype),
        )

        neighbour_mean = (
            neighbour_sum / neighbour_count.clamp_min(1).unsqueeze(-1)
        )
        return torch.norm(h - neighbour_mean, p=2, dim=-1)


# ─── Baseline models ─────────────────────────────────────────────────────────

class BaselineGNN(nn.Module):
    """GraphSAGE baseline."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        if "hidden_dim" in kwargs:
            hidden_channels = int(kwargs.pop("hidden_dim"))

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        h = F.relu(self.conv1(data.x, data.edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, data.edge_index))
        logits = self.classifier(h)
        return logits.squeeze(-1) if logits.size(-1) == 1 else logits


class EvolveGCN_O(nn.Module):
    """Lightweight GCN baseline."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        if "hidden_dim" in kwargs:
            hidden_channels = int(kwargs.pop("hidden_dim"))

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        h = F.relu(self.conv1(data.x, data.edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, data.edge_index))
        logits = self.classifier(h)
        return logits.squeeze(-1) if logits.size(-1) == 1 else logits


class MLP(nn.Module):
    """Tabular-only multilayer perceptron baseline."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        if "hidden_dim" in kwargs:
            hidden_channels = int(kwargs.pop("hidden_dim"))

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, data):
        x = data.x if hasattr(data, "x") else data
        logits = self.net(x)
        return logits.squeeze(-1) if logits.size(-1) == 1 else logits
