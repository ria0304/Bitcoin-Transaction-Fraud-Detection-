"""
src/losses.py
=============
Loss functions for imbalanced fraud detection.

FocalLoss    – Down-weights easy negatives (γ=2.5, α=0.80)
AsymmetricLoss – Asymmetric focusing: harder penalty on FN than FP
CombinedLoss – FocalLoss + contrastive auxiliary term
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
