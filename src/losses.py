"""
src/losses.py
=============
Loss functions for imbalanced fraud detection.

FocalLoss      – Down-weights easy negatives (gamma=2.5, alpha=0.80)
AsymmetricLoss – Asymmetric focusing: harder penalty on FN than FP
CombinedLoss   – FocalLoss + contrastive auxiliary term
CombinedASLLoss– AsymmetricLoss + contrastive + pretrain aux (journal)
PseudoLabelLoss– Weighted BCE for pseudo-labelled unlabelled nodes (PLP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) tuned for severe class imbalance.
    α=0.80 weights fraud (minority) class more heavily.
    γ=2.5 focuses on hard examples.
    """

    def __init__(self, alpha: float = 0.80, gamma: float = 2.5, pos_weight=None):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        pt      = torch.exp(-bce)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss    = alpha_t * (1.0 - pt) ** self.gamma * bce
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (Ben-Baruch et al., 2021) adapted for binary fraud detection.
    Applies different γ values for positive (fraud) and negative (licit) examples,
    effectively making missed frauds costlier than false alarms.

    γ_pos=0  → no down-weighting of hard positives (catch every fraud)
    γ_neg=4  → aggressively down-weight easy negatives (reduce FP noise)
    """

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip      = clip           # probability shift to avoid log(0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs    = torch.sigmoid(logits)
        probs_m  = probs.clamp(min=self.clip)     # prevent log(0) on negatives

        # Positive branch: standard BCE, no focal down-weighting
        loss_pos = -targets * torch.log(probs_m)

        # Negative branch: hard-example focusing with γ_neg
        prob_neg = (1.0 - probs).clamp(min=self.clip)
        loss_neg = -(1.0 - targets) * (
            (1.0 - probs_m) ** self.gamma_neg * torch.log(prob_neg)
        )
        return (loss_pos + loss_neg).mean()


class CombinedASLLoss(nn.Module):
    """Novelty (ElliGAT-ASL): AsymmetricLoss + contrastive + pretrain aux terms."""

    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05,
                 lambda_c=0.30, lambda_p=0.10):
        super().__init__()
        self.asl = AsymmetricLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
        self.lambda_c = lambda_c
        self.lambda_p = lambda_p

    def forward(self, logits, targets, l_contrast=None, l_pretrain=None):
        loss = self.asl(logits, targets)
        if l_contrast is not None:
            loss = loss + self.lambda_c * l_contrast
        if l_pretrain is not None:
            loss = loss + self.lambda_p * l_pretrain
        return loss


class CombinedLoss(nn.Module):
    """
    Combines FocalLoss with a contrastive auxiliary term.

        L = L_focal + λ_contrast * L_contrast + λ_pretrain * L_pretrain

    The contrastive and pretrain losses are computed outside (in trainer.py)
    and passed in as pre-computed scalars.
    """

    def __init__(
        self,
        alpha: float   = 0.80,
        gamma: float   = 2.5,
        lambda_c: float = 0.30,
        lambda_p: float = 0.10,
        pos_weight=None,
    ):
        super().__init__()
        self.focal    = FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
        self.lambda_c = lambda_c
        self.lambda_p = lambda_p

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        l_contrast: torch.Tensor | None = None,
        l_pretrain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.focal(logits, targets)
        if l_contrast is not None:
            loss = loss + self.lambda_c * l_contrast
        if l_pretrain is not None:
            loss = loss + self.lambda_p * l_pretrain
        return loss


class PseudoLabelLoss(nn.Module):
    """
    Weighted Binary Cross-Entropy for Pseudo-Label Propagation (PLP).

    Combines the standard supervised loss on real labels with a down-weighted
    pseudo-label loss on confident unlabelled nodes:

        L = L_supervised + lambda_pl * L_pseudo

    The lambda_pl coefficient (default 0.20) ensures pseudo-labelled nodes
    exert a fraction of the gradient pressure of real labelled nodes,
    preventing confirmation-bias accumulation while still propagating
    useful structural signal from the unlabelled majority.

    This is equivalent to the pseudo-label approach of Lee (2013) adapted
    for node classification on a graph with heterophilous structure.

    References
    ----------
    * Lee, "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning
      Method for Deep Neural Networks", ICML Workshops 2013.
    """

    def __init__(self, lambda_pl: float = 0.20, asl_gamma_neg: float = 4.0):
        super().__init__()
        self.lambda_pl = lambda_pl
        # Use ASL for the supervised part (carries over from CombinedASLLoss)
        self.asl       = AsymmetricLoss(gamma_pos=0.0, gamma_neg=asl_gamma_neg)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pl_logits: torch.Tensor | None = None,
        pl_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits     : (n_labelled,) logits on real training nodes
        targets    : (n_labelled,) real labels (0 / 1)
        pl_logits  : (n_pseudo,) logits on pseudo-labelled nodes (optional)
        pl_targets : (n_pseudo,) pseudo-labels (0.0 / 1.0)  (optional)
        """
        loss = self.asl(logits, targets)
        if pl_logits is not None and pl_targets is not None:
            l_pseudo = torch.nn.functional.binary_cross_entropy_with_logits(
                pl_logits, pl_targets
            )
            loss = loss + self.lambda_pl * l_pseudo
        return loss
