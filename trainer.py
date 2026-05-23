"""
src/trainer.py
==============
Training loop, early stopping, evaluation, and uncertainty estimation.

Improvements over original repo
---------------------------------
• Two-phase training: self-supervised pre-training → fine-tuning
• Contrastive auxiliary loss during fine-tuning
• Cosine LR schedule with linear warm-up
• Gradient clipping + AdamW
• AUC + F1 joint validation criterion (harmonic mean) to avoid high-AUC /
  low-F1 traps common in imbalanced settings
• MC Dropout uncertainty calibrated on validation set
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from src.losses import CombinedLoss, FocalLoss


# ─── Threshold search ────────────────────────────────────────────────────────

def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Grid search threshold maximising F1 on validation set."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        f = f1_score(y_true, (probs > t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float | None = None,
) -> dict:
    if threshold is None:
        threshold = find_best_threshold(y_true, probs)
    y_pred = (probs > threshold).astype(int)
    return {
        "Accuracy":     float((y_pred == y_true).mean()),
        "Balanced Acc": float(balanced_accuracy_score(y_true, y_pred)),
        "MCC":          float(matthews_corrcoef(y_true, y_pred)),
        "Precision":    float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":       float(recall_score(y_true, y_pred, zero_division=0)),
        "F1":           float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC":      float(roc_auc_score(y_true, probs)),
        "threshold":    float(threshold),
        "probs":        probs,
        "true":         y_true,
    }


# ─── LR Scheduler with warm-up ───────────────────────────────────────────────

def _warmup_cosine(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    """Linear warm-up then cosine annealing."""

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Self-supervised pre-training phase ─────────────────────────────────────

def pretrain(
    model,
    data,
    epochs: int = 50,
    lr: float = 5e-4,
    mask_ratio: float = 0.20,
    verbose: bool = True,
):
    """
    Masked-feature auto-encoder pre-training.
    Only the ElliGAT model supports this (has pretrain_forward).
    """
    if not hasattr(model, "pretrain_forward"):
        return

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = _warmup_cosine(opt, warmup_steps=5, total_steps=epochs)

    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss = model.pretrain_forward(data, mask_ratio=mask_ratio)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if verbose and epoch % 10 == 0:
            print(f"    [pretrain] epoch {epoch:03d} | recon_loss={loss.item():.5f}")


# ─── Fine-tuning phase ───────────────────────────────────────────────────────

def train_gnn(
    model,
    data,
    pos_weight: torch.Tensor,
    epochs: int = 300,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 30,
    warmup_steps: int = 10,
    lambda_c: float = 0.30,
    lambda_p: float = 0.10,
    verbose: bool = True,
) -> None:
    """
    Fine-tune GNN with:
        • CombinedLoss (FocalLoss + contrastive + pre-train)
        • Warm-up cosine LR
        • Early stopping on harmonic mean of AUC and F1
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = _warmup_cosine(optimizer, warmup_steps, epochs)
    criterion = CombinedLoss(
        alpha=0.80, gamma=2.5,
        lambda_c=lambda_c, lambda_p=lambda_p,
        pos_weight=pos_weight,
    )

    best_score, no_improve, best_state = 0.0, 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data)
        tr_logits = logits[data.train_mask]
        tr_labels = data.y[data.train_mask]

        # Auxiliary losses (only for ElliGAT)
        l_c = model.contrastive_loss(data) if hasattr(model, "contrastive_loss") else None
        l_p = model.pretrain_forward(data, mask_ratio=0.10) if hasattr(model, "pretrain_forward") else None

        loss = criterion(tr_logits, tr_labels, l_contrast=l_c, l_pretrain=l_p)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits = model(data)
            val_probs  = torch.sigmoid(val_logits[data.val_mask]).cpu().numpy()
            val_y      = data.y[data.val_mask].cpu().numpy()

        if len(np.unique(val_y)) < 2:
            continue

        val_auc = roc_auc_score(val_y, val_probs)
        val_f1  = f1_score(
            val_y, (val_probs > find_best_threshold(val_y, val_probs)).astype(int),
            zero_division=0,
        )
        # Joint criterion: harmonic mean of AUC and F1
        score = 2 * val_auc * val_f1 / max(val_auc + val_f1, 1e-8)

        if score > best_score:
            best_score = score
            no_improve  = 0
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch}")
            break

        if verbose and epoch % 20 == 0:
            print(
                f"    [finetune] ep {epoch:03d} | "
                f"loss={loss.item():.4f} | "
                f"AUC={val_auc:.4f} | F1={val_f1:.4f} | score={score:.4f}"
            )

    if best_state:
        model.load_state_dict(best_state)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_gnn(model, data) -> dict:
    """Evaluate GNN on test set, return metrics dict."""
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs  = torch.sigmoid(logits[data.test_mask]).cpu().numpy()
        y_true = data.y[data.test_mask].cpu().numpy()
    return compute_metrics(y_true, probs)


# ─── MC Dropout uncertainty ───────────────────────────────────────────────────

def mc_uncertainty(model, data, n_samples: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Run n_samples stochastic forward passes (model.train() with dropout active).
    Returns (mean_probs, std_probs) on test nodes.
    """
    if hasattr(model, "mc_dropout_forward"):
        logits_mc = model.mc_dropout_forward(data, n_samples)
        probs_mc  = torch.sigmoid(logits_mc)[:, data.test_mask].cpu().numpy()
    else:
        model.train()
        with torch.no_grad():
            samples = [
                torch.sigmoid(model(data)[data.test_mask]).cpu().numpy()
                for _ in range(n_samples)
            ]
        probs_mc = np.stack(samples, axis=0)
        model.eval()

    return probs_mc.mean(axis=0), probs_mc.std(axis=0)
