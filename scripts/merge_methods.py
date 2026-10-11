#!/usr/bin/env python3
"""
Merge per-method metrics.json files into ONE metrics.json per seed.

Stepwise workflow layout (downloaded artifacts):
    <in_dir>/method-<seed>-<method>/metrics.json

Output (one folder per seed, same layout the aggregator already expects):
    <out_dir>/seed-<seed>/metrics.json

Usage:
    python scripts/merge_methods.py <in_dir> <out_dir>
"""
import glob
import json
import os
import re
import sys


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    in_dir, out_dir = sys.argv[1:]

    per_seed = {}
    for path in sorted(glob.glob(os.path.join(in_dir, "**", "metrics.json"), recursive=True)):
        m = re.search(r"method-(\d+)-([A-Za-z0-9_]+)", path)
        if not m:
            print(f"[merge] skip (name does not match): {path}", file=sys.stderr)
            continue
        seed = int(m.group(1))
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[merge] WARNING unreadable {path}: {exc}", file=sys.stderr)
            continue
        merged = per_seed.setdefault(seed, {"_config": {"methods": []}})
        for key, val in data.items():
            if key == "_config":
                merged["_config"].update({k: v for k, v in val.items() if k != "methods"})
                merged["_config"]["methods"] += val.get("methods", [])
            else:
                merged[key] = val
        print(f"[merge] seed {seed}: +{[k for k in data if not k.startswith('_')]} from {path}")

    if not per_seed:
        print("::error::No per-method metrics.json found to merge.", file=sys.stderr)
        sys.exit(1)

    for seed, merged in per_seed.items():
        d = os.path.join(out_dir, f"seed-{seed}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        models = [k for k in merged if not k.startswith("_")]
        print(f"[merge] seed {seed}: {len(models)} models -> {d}/metrics.json")


if __name__ == "__main__":
    main()
