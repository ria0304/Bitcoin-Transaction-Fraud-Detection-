"""
src/pseudo_label.py
====================
Pseudo-Label Propagation (PLP) — Novel Contribution #3
=======================================================

Motivation
----------
The Elliptic dataset contains 203,769 transactions, but only ~22.9% carry
ground-truth labels. The remaining 77.1% are loaded into the graph
(data.y == -1) but are never used for training in the baseline code. PLP
uses MC Dropout confidence to assign pseudo-labels to unlabelled nodes and
re-trains with a down-weighted pseudo-label loss (phase 3).

Three-phase protocol
--------------------
  Phase 1: Self-supervised pre-training            (existing)
  Phase 2: Supervised fine-tuning on labelled nodes (existing)
  Phase 3: Pseudo-label propagation                 (NEW)
    3a. MC Dropout on all nodes -> mean probability + uncertainty (std).
    3b. Assign pseudo-labels to unlabelled nodes whose std < confidence_threshold.
    3c. Re-train with PseudoLabelLoss weighting pseudo nodes by lambda_pl.

References
----------
* Lee, "Pseudo-Label: Simple and Efficient Semi-Supervised Learning",
  ICML Workshops 2013.
* Zhou et al., "Learning with Local and Global Consistency", NeurIPS 2004.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


# ─── Pseudo-label generation ─────────────────────────────────────────────────

def generate_pseudo_labels(
    model,
    data: Data,
    n_mc_samples: int = 30,
    confidence_threshold: float = 0.10,
    prob_low: float = 0.20,
    prob_high: float = 0.80,
):
    """
    Run MC Dropout on *model* and assign pseudo-labels to unlabelled nodes
    that pass a confidence gate.

    Parameters
    ----------
    model                : Trained ElliGAT (or any model with forward / mc_dropout_forward).
    data                 : Full graph. Unlabelled nodes have data.y == -1.
    n_mc_samples         : Number of Monte Carlo forward passes.
    confidence_threshold : Maximum allowed MC std for pseudo-labelling.
    prob_low             : Probability ceiling for assigning pseudo-label 0 (licit).
    prob_high            : Probability floor for assigning pseudo-label 1 (fraud).

    Returns
    -------
    (pseudo_mask, pseudo_y)
    pseudo_mask : (N,) bool tensor — True for nodes that received a pseudo-label.
    pseudo_y    : (N,) float tensor — pseudo-label values (0.0 / 1.0).
    """
    device = data.x.device

    # MC Dropout forward passes
    if hasattr(model, "mc_dropout_forward"):
        logits_mc = model.mc_dropout_forward(data, n_mc_samples)   # (S, N)
        probs_mc  = torch.sigmoid(logits_mc)                        # (S, N)
    else:
        model.train()
        with torch.no_grad():
            probs_mc = torch.stack([
                torch.sigmoid(model(data)) for _ in range(n_mc_samples)
            ], dim=0)

    mean_prob = probs_mc.mean(dim=0)   # (N,)
    std_prob  = probs_mc.std(dim=0)    # (N,)

    unlabelled = (data.y == -1)
    confident  = std_prob <= confidence_threshold

    is_pseudo_licit = unlabelled & confident & (mean_prob <= prob_low)
    is_pseudo_fraud = unlabelled & confident & (mean_prob >= prob_high)

    pseudo_mask = is_pseudo_licit | is_pseudo_fraud
    pseudo_y    = torch.zeros_like(mean_prob)
    pseudo_y[is_pseudo_fraud] = 1.0

    n_pl       = int(pseudo_mask.sum().item())
    n_pf       = int(is_pseudo_fraud.sum().item())
    n_pl_licit = int(is_pseudo_licit.sum().item())
    print(
        f"    [PLP] Pseudo-labels assigned: {n_pl:,}  "
        f"({n_pf:,} fraud, {n_pl_licit:,} licit)  "
        f"out of {int(unlabelled.sum()):,} unlabelled nodes"
    )

    model.eval()
    return pseudo_mask, pseudo_y


# ─── Phase-3 re-training ─────────────────────────────────────────────────────

def pseudo_label_finetune(
    model,
    data: Data,
    pseudo_mask: torch.Tensor,
    pseudo_y: torch.Tensor,
    lambda_pl: float = 0.20,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> None:
    """
    Phase-3 re-training: fine-tune *model* on real labels + pseudo-labels,
    with pseudo-nodes down-weighted by lambda_pl.

    Total loss = L_supervised + lambda_pl * L_pseudo
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_loss  = float("inf")
    best_state = None

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        logits = model(data)

        # Supervised loss on real training labels
        tr_logits = logits[data.train_mask]
        tr_labels = data.y[data.train_mask]
        l_sup = F.binary_cross_entropy_with_logits(tr_logits, tr_labels)

        # Pseudo-label loss on pseudo-labelled unlabelled nodes
        if pseudo_mask.any():
            pl_logits = logits[pseudo_mask]
            pl_labels = pseudo_y[pseudo_mask]
            l_pseudo  = F.binary_cross_entropy_with_logits(pl_logits, pl_labels)
        else:
            l_pseudo = torch.tensor(0.0, device=logits.device)

        loss = l_sup + lambda_pl * l_pseudo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(
                f"    [PLP finetune] ep {epoch:03d} | "
                f"L_sup={l_sup.item():.4f}  "
                f"L_pseudo={l_pseudo.item():.4f}  "
                f"total={loss.item():.4f}"
            )

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"    [PLP] Phase-3 complete.  Best loss: {best_loss:.5f}")


# ─── Convenience wrapper ──────────────────────────────────────────────────────

def run_plp(
    model,
    data: Data,
    n_mc_samples: int = 30,
    confidence_threshold: float = 0.10,
    lambda_pl: float = 0.20,
    epochs: int = 50,
    lr: float = 1e-4,
):
    """
    Full PLP pipeline: generate pseudo-labels then re-train (phase 3).

    Returns (pseudo_mask, pseudo_y) for downstream offline validation.
    """
    print("  [PLP] Generating pseudo-labels via MC Dropout …")
    pseudo_mask, pseudo_y = generate_pseudo_labels(
        model, data,
        n_mc_samples=n_mc_samples,
        confidence_threshold=confidence_threshold,
    )

    if not pseudo_mask.any():
        print("  [PLP] No confident pseudo-labels found — skipping phase 3.")
        return pseudo_mask, pseudo_y

    print(f"  [PLP] Phase-3 re-training ({epochs} epochs, lambda_pl={lambda_pl}) …")
    pseudo_label_finetune(
        model, data, pseudo_mask, pseudo_y,
        lambda_pl=lambda_pl, epochs=epochs, lr=lr,
    )

    return pseudo_mask, pseudo_y
