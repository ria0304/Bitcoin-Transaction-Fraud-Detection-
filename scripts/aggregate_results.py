#!/usr/bin/env python3
"""
Aggregate per-seed pipeline outputs into one mean ± std table.

Each CI matrix job runs the pipeline with a single seed and writes
``<results_dir>/results-seed-<seed>/metrics.json``.  This script merges them.

Usage:  python scripts/aggregate_results.py <results_dir> <out_dir>

Writes to <out_dir>:
    metrics_aggregate.json   mean/std/per-seed values for every model+metric
    per_seed.csv             one row per (seed, model, metric)
    summary.md               markdown tables (also printed to stdout)

Std is the population std (ddof=0), matching np.std in main.py.
Wilcoxon tests need scipy; they are skipped with a note if it is missing.
"""
import csv
import glob
import json
import math
import os
import re
import sys

HEADLINE = ["ROC-AUC", "F1", "MCC", "Balanced Acc", "Precision", "Recall"]
ORDER = ["MLP", "RandomForest", "XGBoost", "LightGBM", "BaselineGNN",
         "EvolveGCN", "ROLAND", "TGN", "ElliGAT", "ElliGAT+PLP", "MetaEnsemble"]


def mean_std(vals):
    m = sum(vals) / len(vals)
    return m, math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def load(results_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "metrics.json"), recursive=True)):
        m = re.search(r"seed-(\d+)", path)
        with open(path) as f:
            data = json.load(f)
        seed = int(m.group(1)) if m else (data.get("_config", {}).get("seeds") or [len(runs)])[0]
        runs[seed] = data
        print(f"[aggregate] loaded seed {seed} from {path}", file=sys.stderr)
    return runs


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    results_dir, out_dir = sys.argv[1:]
    os.makedirs(out_dir, exist_ok=True)

    runs = load(results_dir)
    if not runs:
        print("::error::No metrics.json found - every seed job failed or produced no metrics.")
        sys.exit(1)
    seeds = sorted(runs)

    # model -> metric -> {seed: value}
    table = {}
    for seed, data in runs.items():
        for model, mets in data.items():
            if model.startswith("_") or not isinstance(mets, dict):
                continue
            for metric, v in mets.items():
                val = v.get("mean") if isinstance(v, dict) else v
                if isinstance(val, (int, float)) and math.isfinite(val):
                    table.setdefault(model, {}).setdefault(metric, {})[seed] = float(val)

    agg = {}
    for model, mets in table.items():
        agg[model] = {}
        for metric, per_seed in mets.items():
            vals = [per_seed[s] for s in sorted(per_seed)]
            m, sd = mean_std(vals)
            agg[model][metric] = {"mean": m, "std": sd, "n": len(vals),
                                  "per_seed": {str(s): per_seed[s] for s in sorted(per_seed)}}

    with open(os.path.join(out_dir, "metrics_aggregate.json"), "w") as f:
        json.dump({"seeds": seeds, "models": agg}, f, indent=2)
    with open(os.path.join(out_dir, "per_seed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "model", "metric", "value"])
        for model, mets in agg.items():
            for metric, d in mets.items():
                for s, v in d["per_seed"].items():
                    w.writerow([s, model, metric, v])

    models = [m for m in ORDER if m in agg] + [m for m in agg if m not in ORDER]
    cols = [c for c in HEADLINE if any(c in agg[m] for m in models)]
    extra = sorted({k for m in models for k in agg[m]} - set(HEADLINE))

    lines = [f"## Aggregate over {len(seeds)} seed(s): {', '.join(map(str, seeds))}", ""]
    if len(seeds) == 1:
        lines += ["> Only one seed available - std is 0 by construction.", ""]

    def render(columns):
        out = ["| Model | " + " | ".join(columns) + " |",
               "|---|" + "---|" * len(columns)]
        for m in models:
            cells = []
            for c in columns:
                d = agg[m].get(c)
                cells.append(f"{d['mean']:.4f} ± {d['std']:.4f}" if d else "n/a")
            out.append(f"| {m} | " + " | ".join(cells) + " |")
        return out

    lines += render(cols)
    if extra:
        lines += ["", "### Other metrics", ""] + render(extra)

    expl = [(s, runs[s].get("_explainability")) for s in seeds if runs[s].get("_explainability")]
    if expl:
        lines += ["", "### GNNExplainer fidelity (mean over seeds)", ""]
        for key in ("fidelity_plus_mean", "fidelity_minus_mean"):
            vals = [e[key] for _, e in expl if key in e]
            if vals:
                m, sd = mean_std(vals)
                lines.append(f"- {key}: {m:.4f} ± {sd:.4f} (n={len(vals)})")

    # Paired Wilcoxon signed-rank on ROC-AUC across seeds (ElliGAT vs rest)
    lines += ["", "### Wilcoxon signed-rank (ElliGAT vs competitor, ROC-AUC, paired by seed)", ""]
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    base = agg.get("ElliGAT", {}).get("ROC-AUC", {}).get("per_seed")
    if wilcoxon is None:
        lines.append("_scipy not installed - skipped._")
    elif not base:
        lines.append("_ElliGAT ROC-AUC missing - skipped._")
    else:
        rows = []
        for comp in models:
            if comp == "ElliGAT":
                continue
            other = agg[comp].get("ROC-AUC", {}).get("per_seed", {})
            common = sorted(set(base) & set(other), key=int)
            if len(common) < 2:
                continue
            a = [base[s] for s in common]
            b = [other[s] for s in common]
            if all(x == y for x, y in zip(a, b)):
                continue
            _, p = wilcoxon(a, b)
            rows.append(f"| {comp} | {len(common)} | {p:.4f} |")
        if rows:
            lines += ["| Competitor | n seeds | p (two-sided) |", "|---|---|---|"] + rows
            lines += ["", "_With n=5 paired seeds the smallest attainable two-sided exact p is "
                          "0.0625, so p<0.05 is impossible; use more seeds if significance is claimed._"]
        else:
            lines.append("_Needs at least 2 seeds - skipped._")

    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
