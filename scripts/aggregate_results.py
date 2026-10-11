
#!/usr/bin/env python3
"""
Aggregate per-seed pipeline outputs into one mean +/- std table.

Each CI matrix job runs the pipeline with a single seed and writes:
    <results_dir>/results-seed-<seed>/metrics.json

Usage:
    python scripts/aggregate_results.py <results_dir> <out_dir>

Writes to <out_dir>:
    metrics_aggregate.json   mean/std/per-seed values for every model+metric
    per_seed.csv             one row per (seed, model, metric)
    summary.md               Markdown tables (also printed to stdout)

Standard deviation is the population standard deviation (ddof=0).
Wilcoxon tests need scipy; they are skipped with a note if scipy is missing.
"""

import csv
import glob
import json
import math
import os
import re
import sys

HEADLINE = [
    "ROC-AUC",
    "F1",
    "MCC",
    "Balanced Acc",
    "Precision",
    "Recall",
]

ORDER = [
    "MLP",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "BaselineGNN",
    "EvolveGCN",
    "ROLAND",
    "TGN",
    "ElliGAT",
    "ElliGAT+PLP",
    "MetaEnsemble",
]


def mean_std(vals):
    """Return population mean and standard deviation."""
    if not vals:
        return 0.0, 0.0

    mean = sum(vals) / len(vals)
    std = math.sqrt(
        sum((value - mean) ** 2 for value in vals) / len(vals)
    )
    return mean, std


def load(results_dir):
    """Load metrics.json files recursively, indexed by seed."""
    runs = {}

    pattern = os.path.join(results_dir, "**", "metrics.json")
    paths = sorted(glob.glob(pattern, recursive=True))

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[aggregate] WARNING: Could not read {path}: {exc}",
                file=sys.stderr,
            )
            continue

        if not isinstance(data, dict):
            print(
                f"[aggregate] WARNING: Expected a JSON object in {path}; "
                "skipping.",
                file=sys.stderr,
            )
            continue

        match = re.search(r"seed-(\d+)", path)

        if match:
            seed = int(match.group(1))
        else:
            configured_seeds = data.get("_config", {}).get("seeds", [])
            if isinstance(configured_seeds, list) and configured_seeds:
                seed = int(configured_seeds[0])
            else:
                seed = len(runs)

        if seed in runs:
            print(
                f"[aggregate] WARNING: Duplicate metrics for seed {seed}; "
                f"using {path}.",
                file=sys.stderr,
            )

        runs[seed] = data
        print(
            f"[aggregate] Loaded seed {seed} from {path}",
            file=sys.stderr,
        )

    return runs


def report_missing_metrics(results_dir):
    """Provide useful diagnostics when no valid seed results were loaded."""
    print(
        "::error::No valid metrics.json files were found.",
        file=sys.stderr,
    )
    print(
        f"[aggregate] Results directory: {os.path.abspath(results_dir)}",
        file=sys.stderr,
    )

    if not os.path.isdir(results_dir):
        print(
            "[aggregate] The results directory does not exist. "
            "Check the pipeline output path.",
            file=sys.stderr,
        )
        return

    print(
        "[aggregate] Expected files such as "
        "'results-seed-42/metrics.json'.",
        file=sys.stderr,
    )

    try:
        entries = sorted(os.listdir(results_dir))
    except OSError as exc:
        print(
            f"[aggregate] Could not list results directory: {exc}",
            file=sys.stderr,
        )
        return

    if not entries:
        print("[aggregate] The results directory is empty.", file=sys.stderr)
    else:
        print("[aggregate] Entries in results directory:", file=sys.stderr)
        for entry in entries[:50]:
            print(f"  - {entry}", file=sys.stderr)

    print(
        "[aggregate] Check the seed job logs for the original failure. "
        "An aggregation error usually means the metrics were never written "
        "or could not be read.",
        file=sys.stderr,
    )


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)

    results_dir, out_dir = sys.argv[1:]

    os.makedirs(out_dir, exist_ok=True)

    runs = load(results_dir)

    if not runs:
        report_missing_metrics(results_dir)
        sys.exit(1)

    seeds = sorted(runs)

    # Structure:
    # model -> metric -> {seed: value}
    table = {}

    for seed, data in runs.items():
        for model, metrics in data.items():
            if model.startswith("_") or not isinstance(metrics, dict):
                continue

            for metric, value in metrics.items():
                if isinstance(value, dict):
                    value = value.get("mean")

                if (
                    isinstance(value, (int, float))
                    and math.isfinite(value)
                ):
                    table.setdefault(model, {}).setdefault(
                        metric, {}
                    )[seed] = float(value)

    # Calculate aggregate statistics for each model and metric.
    aggregated = {}

    for model, metrics in table.items():
        aggregated[model] = {}

        for metric, per_seed in metrics.items():
            ordered_seeds = sorted(per_seed)
            values = [per_seed[seed] for seed in ordered_seeds]
            mean, std = mean_std(values)

            aggregated[model][metric] = {
                "mean": mean,
                "std": std,
                "n": len(values),
                "per_seed": {
                    str(seed): per_seed[seed]
                    for seed in ordered_seeds
                },
            }

    # Save aggregate JSON.
    aggregate_path = os.path.join(
        out_dir, "metrics_aggregate.json"
    )

    with open(aggregate_path, "w", encoding="utf-8") as file:
        json.dump(
            {"seeds": seeds, "models": aggregated},
            file,
            indent=2,
        )

    # Save one row per seed/model/metric.
    csv_path = os.path.join(out_dir, "per_seed.csv")

    with open(
        csv_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "model", "metric", "value"])

        for model, metrics in aggregated.items():
            for metric, details in metrics.items():
                for seed, value in details["per_seed"].items():
                    writer.writerow([seed, model, metric, value])

    models = (
        [model for model in ORDER if model in aggregated]
        + [
            model
            for model in aggregated
            if model not in ORDER
        ]
    )

    columns = [
        column
        for column in HEADLINE
        if any(column in aggregated[model] for model in models)
    ]

    extra_columns = sorted(
        {
            metric
            for model in models
            for metric in aggregated[model]
        }
        - set(HEADLINE)
    )

    lines = [
        f"## Aggregate over {len(seeds)} seed(s): "
        f"{', '.join(map(str, seeds))}",
        "",
    ]

    if len(seeds) == 1:
        lines.extend(
            [
                "> Only one seed is available; standard deviation "
                "is 0 by construction.",
                "",
            ]
        )

    def render(selected_columns):
        """Render a Markdown table for the selected metrics."""
        if not selected_columns:
            return ["No matching metrics were found.", ""]

        output = [
            "| Model | " + " | ".join(selected_columns) + " |",
            "|---|" + "---|" * len(selected_columns),
        ]

        for model in models:
            cells = []

            for column in selected_columns:
                details = aggregated[model].get(column)

                if details:
                    cells.append(
                        f"{details['mean']:.4f} "
                        f"+/- {details['std']:.4f}"
                    )
                else:
                    cells.append("n/a")

            output.append(
                f"| {model} | " + " | ".join(cells) + " |"
            )

        return output

    lines.extend(render(columns))

    if extra_columns:
        lines.extend(["", "### Other metrics", ""])
        lines.extend(render(extra_columns))

    # Aggregate explainability metrics when present.
    explainability = [
        (seed, runs[seed].get("_explainability"))
        for seed in seeds
        if runs[seed].get("_explainability")
    ]

    if explainability:
        lines.extend(
            [
                "",
                "### GNNExplainer fidelity (mean over seeds)",
                "",
            ]
        )

        for key in ("fidelity_plus_mean", "fidelity_minus_mean"):
            values = [
                result[key]
                for _, result in explainability
                if isinstance(result, dict)
                and isinstance(result.get(key), (int, float))
                and math.isfinite(result[key])
            ]

            if values:
                mean, std = mean_std(values)
                lines.append(
                    f"- {key}: {mean:.4f} +/- {std:.4f} "
                    f"(n={len(values)})"
                )

    # Paired Wilcoxon signed-rank tests:
    # ElliGAT versus each competitor, using ROC-AUC across shared seeds.
    lines.extend(
        [
            "",
            "### Wilcoxon signed-rank "
            "(ElliGAT vs competitor, ROC-AUC, paired by seed)",
            "",
        ]
    )

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    baseline = (
        aggregated.get("ElliGAT", {})
        .get("ROC-AUC", {})
        .get("per_seed")
    )

    if wilcoxon is None:
        lines.append("_scipy not installed - skipped._")
    elif not baseline:
        lines.append("_ElliGAT ROC-AUC missing - skipped._")
    else:
        rows = []

        for competitor in models:
            if competitor == "ElliGAT":
                continue

            other = (
                aggregated.get(competitor, {})
                .get("ROC-AUC", {})
                .get("per_seed", {})
            )

            common_seeds = sorted(
                set(baseline) & set(other),
                key=int,
            )

            if len(common_seeds) < 2:
                continue

            first = [baseline[seed] for seed in common_seeds]
            second = [other[seed] for seed in common_seeds]

            if all(a == b for a, b in zip(first, second)):
                continue

            try:
                _, p_value = wilcoxon(first, second)
            except ValueError as exc:
                print(
                    f"[aggregate] WARNING: Wilcoxon test failed for "
                    f"{competitor}: {exc}",
                    file=sys.stderr,
                )
                continue

            rows.append(
                f"| {competitor} | {len(common_seeds)} "
                f"| {p_value:.4f} |"
            )

        if rows:
            lines.extend(
                [
                    "| Competitor | n seeds | p (two-sided) |",
                    "|---|---|---|",
                ]
            )
            lines.extend(rows)
            lines.extend(
                [
                    "",
                    "_With n=5 paired seeds, the smallest attainable "
                    "two-sided exact p-value is 0.0625. More seeds "
                    "are needed for a conventional p < 0.05 claim._",
                ]
            )
        else:
            lines.append("_Needs at least 2 non-identical paired seeds; skipped._")

    summary = "\n".join(lines) + "\n"

    summary_path = os.path.join(out_dir, "summary.md")

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(summary)

    print(summary)

    print(
        f"[aggregate] Wrote: {aggregate_path}",
        file=sys.stderr,
    )
    print(
        f"[aggregate] Wrote: {csv_path}",
        file=sys.stderr,
    )
    print(
        f"[aggregate] Wrote: {summary_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
