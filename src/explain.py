"""
src/explain.py
==============
Fraud Subgraph Explainability via GNNExplainer — Novel Contribution #4
=======================================================================

Motivation
----------
High-stakes deployment of fraud detection models requires interpretability.
Regulators (FATF, FinCEN) increasingly require that automated AML/CFT decisions
be explainable to compliance officers. Existing GNN work on Elliptic provides
no attribution of *why* a node is predicted as illicit.

We apply GNNExplainer (Ying et al., 2019) to the trained ElliGAT model to:
  1. Find the minimal subgraph + feature mask accounting for each prediction.
  2. Aggregate explanations to find recurring fraud-predictive motifs.
  3. Report Fidelity+ and Fidelity- scores as quantitative explanation quality.

Fidelity scores
---------------
* Fidelity+ : prediction change when important edges are REMOVED.
  High Fidelity+ means the explanation subgraph was crucial.
* Fidelity- : prediction change when ONLY important edges are KEPT.
  High Fidelity- means the complement edges were not contributing.

References
----------
* Ying et al., "GNNExplainer: Generating Explanations for Graph Neural
  Networks", NeurIPS 2019.
* Amara et al., "GraphFramEx: Towards Systematic Evaluation of
  Explainability Methods for GNNs", LoG 2022.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

try:
    from torch_geometric.explain import Explainer, GNNExplainer
    HAS_EXPLAINER = True
except ImportError:
    HAS_EXPLAINER = False


# ─── Fidelity computation ─────────────────────────────────────────────────────

def compute_fidelity(
    model,
    data: Data,
    node_idx: int,
    important_edges: torch.Tensor,
    important_feats: torch.Tensor,
) -> dict:
    """
    Compute Fidelity+ and Fidelity- for a single node explanation.

    Parameters
    ----------
    model           : Trained ElliGAT.
    data            : Full graph.
    node_idx        : Index of the node being explained.
    important_edges : (E,) bool mask — True for the K most important edges.
    important_feats : (D,) feature importance weights in [0,1].

    Returns
    -------
    dict with fidelity_plus, fidelity_minus, and original/modified probs.
    """
    model.eval()
    with torch.no_grad():
        prob_orig = float(torch.sigmoid(model(data)[node_idx]).item())

        # Fidelity+: remove important edges
        data_minus = data.clone()
        data_minus.edge_index = data.edge_index[:, ~important_edges]
        if data.edge_attr is not None:
            data_minus.edge_attr = data.edge_attr[~important_edges]
        prob_minus = float(torch.sigmoid(model(data_minus)[node_idx]).item())

        # Fidelity-: keep ONLY important edges
        data_only = data.clone()
        data_only.edge_index = data.edge_index[:, important_edges]
        if data.edge_attr is not None:
            data_only.edge_attr = data.edge_attr[important_edges]
        prob_only = float(torch.sigmoid(model(data_only)[node_idx]).item())

    return {
        "fidelity_plus":  abs(prob_orig - prob_minus),
        "fidelity_minus": abs(prob_orig - prob_only),
        "prob_orig":      prob_orig,
        "prob_minus":     prob_minus,
        "prob_only":      prob_only,
    }


# ─── GNNExplainer wrapper ─────────────────────────────────────────────────────

def explain_fraud_nodes(
    model,
    data: Data,
    n_nodes: int = 50,
    top_k_edges: int = 10,
    epochs: int = 200,
) -> dict:
    """
    Run GNNExplainer on the n_nodes highest-probability fraud test nodes.

    Returns a dict with aggregated fidelity scores and per-node results.
    Sets 'available': False if torch_geometric.explain is not installed.
    """
    if not HAS_EXPLAINER:
        print(
            "  [Explain] torch_geometric.explain not found "
            "(requires PyG >= 2.3). Skipping explainability step."
        )
        return {"available": False}

    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(data))

    test_fraud_mask = data.test_mask & (data.y == 1)
    if not test_fraud_mask.any():
        print("  [Explain] No fraud nodes in test set — skipping.")
        return {"available": False}

    fraud_idx = test_fraud_mask.nonzero(as_tuple=True)[0]
    sorted_idx = fraud_idx[probs[fraud_idx].argsort(descending=True)]
    explain_nodes = sorted_idx[:n_nodes].tolist()

    print(f"  [Explain] Running GNNExplainer on {len(explain_nodes)} fraud nodes …")

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    node_results, fid_plus, fid_minus = [], [], []

    for i, node_idx in enumerate(explain_nodes):
        try:
            expl = explainer(
                x=data.x,
                edge_index=data.edge_index,
                index=int(node_idx),
                edge_attr=data.edge_attr,
            )

            # Top-K important edges
            if expl.edge_mask is not None:
                edge_scores = expl.edge_mask.cpu()
                k = min(top_k_edges, len(edge_scores))
                topk_idx  = edge_scores.topk(k).indices
                imp_edges = torch.zeros(len(edge_scores), dtype=torch.bool)
                imp_edges[topk_idx] = True
            else:
                imp_edges = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
                topk_idx  = torch.tensor([], dtype=torch.long)

            feat_imp = (
                expl.node_mask[node_idx].cpu()
                if expl.node_mask is not None
                else torch.ones(data.x.shape[1])
            )

            fid = compute_fidelity(model, data, int(node_idx), imp_edges, feat_imp)
            fid_plus.append(fid["fidelity_plus"])
            fid_minus.append(fid["fidelity_minus"])

            node_results.append({
                "node_idx":       int(node_idx),
                "prob_orig":      fid["prob_orig"],
                "fidelity_plus":  fid["fidelity_plus"],
                "fidelity_minus": fid["fidelity_minus"],
                "top_k_edges":    topk_idx.tolist(),
                "feat_imp":       feat_imp.numpy(),
            })

        except Exception as exc:
            print(f"    [Explain] node {node_idx}: {exc}")
            continue

        if (i + 1) % 10 == 0:
            print(f"    Explained {i+1}/{len(explain_nodes)} nodes …")

    if not fid_plus:
        return {"available": True, "node_results": [], "n_explained": 0}

    results = {
        "available":           True,
        "n_explained":         len(fid_plus),
        "fidelity_plus_mean":  float(np.mean(fid_plus)),
        "fidelity_plus_std":   float(np.std(fid_plus)),
        "fidelity_minus_mean": float(np.mean(fid_minus)),
        "fidelity_minus_std":  float(np.std(fid_minus)),
        "node_results":        node_results,
    }

    print(
        f"\n  [Explain] Fidelity+ = {results['fidelity_plus_mean']:.4f} "
        f"+/- {results['fidelity_plus_std']:.4f}  |  "
        f"Fidelity- = {results['fidelity_minus_mean']:.4f} "
        f"+/- {results['fidelity_minus_std']:.4f}"
    )
    return results


# ─── Top feature attribution summary ─────────────────────────────────────────

def top_fraud_features(
    node_results: list,
    feature_names: list | None = None,
    top_k: int = 10,
) -> list:
    """
    Aggregate feature importance scores across explained fraud nodes and
    return the top-K most discriminative features as (name, mean_importance) tuples.
    """
    if not node_results:
        return []

    D = len(node_results[0]["feat_imp"])
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(D)]

    stacked  = np.stack([r["feat_imp"] for r in node_results], axis=0)
    mean_imp = stacked.mean(axis=0)

    ranked = sorted(zip(feature_names, mean_imp.tolist()),
                    key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
