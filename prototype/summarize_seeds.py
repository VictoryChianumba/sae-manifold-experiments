"""
#21 follow-up: compute per-row confidence intervals from the 10-seed
legible/iso/adaptive results, and write markdown summary tables.

NEXT_SESSION.md #21 spec called for "compute confidence intervals per row" —
the 10-seed run produced per-seed lists in each results.json (e.g.
ve['1.0|years'] is a list of 10 floats), but the pareto plots only show the
seed-averaged means. This script converts those per-seed lists into
mean ± SEM tables and flags rows that beat PCA at the ~95% confidence level
(mean - 2·SEM > PCA_mean + 2·SEM, treating the SAE and PCA as paired
per-seed samples and using SEM of the per-seed difference).

Usage (no GPU, no pod):
  python prototype/summarize_seeds.py --tag meta-llama-3.1-8b_L16

Output files (one per stage, alongside its results.json):
  cache/<tag>/<stage>/summary_seeds10.md
And a stage-spanning roll-up at:
  cache/<tag>/seeds10_summary.md
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


# Per-stage table layout: what config columns label the row, what metric
# columns we tabulate, and where to find them in results.json.
STAGES = {
    "legible_coord": dict(
        config_keys=["lams"],
        config_label=lambda c: f"λ={c[0]}",
        metric_arr="ve",     # also r2lin, r2knn — handled below
    ),
    "iso_parsimony": dict(
        config_keys=["configs"],
        config_label=lambda c: f"iso={c[0]} pars={c[1]}",
        metric_arr="ve",
    ),
    "adaptive_parsimony": dict(
        config_keys=["configs"],
        config_label=lambda c: f"iso={c[0]} gate={c[1]}",
        metric_arr="ve",
    ),
}

MANIFOLDS = ["years", "age", "temperature", "colors", "geography"]
METRICS = ["ve", "r2lin", "r2knn"]


def _stat(xs):
    """Mean and standard error of the mean (SEM = std / sqrt(N), unbiased
    std). Returns (mean, sem, n)."""
    a = np.asarray(xs, dtype=float)
    n = len(a)
    mean = float(a.mean())
    # Use unbiased std (ddof=1) — small-sample correction, matters at N=10.
    sem = float(a.std(ddof=1) / math.sqrt(n))
    return mean, sem, n


def _paired_diff_stat(sae_xs, pca_xs):
    """Paired-difference statistic: per-seed (SAE − PCA), then mean ± SEM
    of the differences. This is the right test because seed is shared
    between SAE and PCA (same train/eval split). Returns (mean_d, sem_d,
    crosses_2sem: bool).

    crosses_2sem flags a "real" win in either direction at ~95% paired CI:
      mean_d - 2·sem_d > 0   or   mean_d + 2·sem_d < 0
    """
    a = np.asarray(sae_xs, dtype=float)
    b = np.asarray(pca_xs, dtype=float)
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    mean_d = float(d.mean())
    sem_d = float(d.std(ddof=1) / math.sqrt(n))
    lo, hi = mean_d - 2 * sem_d, mean_d + 2 * sem_d
    crosses = "↑" if lo > 0 else ("↓" if hi < 0 else "·")
    return mean_d, sem_d, crosses


def _format(mean, sem):
    """Pretty-print mean ± SEM with sensible precision."""
    return f"{mean:.3f}±{sem:.3f}"


def _summarize_stage(cache_dir, stage, spec):
    res_path = cache_dir / stage / "results.json"
    if not res_path.exists():
        return None, f"(no results.json for {stage})"
    d = json.load(open(res_path))
    seeds = d["seeds"]
    headline_N = d["headline_N"]
    pca_r2 = d["pca_r2"]      # per-manifold list across seeds (label R²)
    pca_ve = d.get("pca_ve", None)  # per-manifold list, optional

    # Reconstruct row labels (the config column).
    if "lams" in spec["config_keys"]:
        row_keys = [str(l) for l in d["lams"]]
        row_labels = [spec["config_label"]([float(l)]) for l in row_keys]
    else:
        cfgs = d["configs"]
        row_keys = ["|".join(str(c) for c in cfg) for cfg in cfgs]
        row_labels = [spec["config_label"](cfg) for cfg in cfgs]

    md = [f"# {stage} — 10-seed summary (headline_N={headline_N})\n"]
    md.append(f"Seeds: {seeds}.  Cells = mean ± SEM over 10 seeds.\n")
    md.append("Symbols on the R² rows: **↑** = paired Δ vs PCA exceeds 2·SEM (SAE>PCA), "
              "**↓** = paired Δ exceeds 2·SEM the other way (PCA>SAE), **·** = inside the band.\n")

    for metric in METRICS:
        arr = d.get(metric)
        if arr is None:
            continue
        md.append(f"\n## {metric}\n")
        # Header
        header = ["config"] + MANIFOLDS + ["mean*"]
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join(["---"] * len(header)) + "|")

        for rk, rlabel in zip(row_keys, row_labels):
            cells = [rlabel]
            row_means = []
            for mname in MANIFOLDS:
                key = f"{rk}|{mname}"
                xs = arr.get(key)
                if xs is None or len(xs) == 0:
                    cells.append("—")
                    continue
                mean, sem, _ = _stat(xs)
                cell = _format(mean, sem)
                # If we're on r2lin/r2knn, also paired-diff vs PCA.
                if metric in ("r2lin", "r2knn"):
                    pca_xs = pca_r2.get(mname, [])
                    if len(pca_xs) >= len(xs):
                        _, _, crosses = _paired_diff_stat(xs, pca_xs)
                        cell = f"{cell} {crosses}"
                if metric == "ve" and pca_ve and mname in pca_ve:
                    _, _, crosses = _paired_diff_stat(xs, pca_ve[mname])
                    cell = f"{cell} {crosses}"
                cells.append(cell)
                if mname != "colors":  # mean* excludes cyclic per project convention
                    row_means.append(mean)
            if row_means:
                cells.append(f"{np.mean(row_means):.3f}")
            else:
                cells.append("—")
            md.append("| " + " | ".join(cells) + " |")

        # PCA reference row
        if metric in ("r2lin", "r2knn") and metric == "r2lin":
            pca_cells = ["**PCA ref**"]
            pca_row_means = []
            for mname in MANIFOLDS:
                xs = pca_r2.get(mname, [])
                if not xs:
                    pca_cells.append("—")
                    continue
                mean, sem, _ = _stat(xs)
                pca_cells.append(_format(mean, sem))
                if mname != "colors":
                    pca_row_means.append(mean)
            pca_cells.append(f"{np.mean(pca_row_means):.3f}" if pca_row_means else "—")
            md.append("| " + " | ".join(pca_cells) + " |")

    md.append("\n")
    md_text = "\n".join(md)
    out_path = cache_dir / stage / "summary_seeds10.md"
    out_path.write_text(md_text)
    return md_text, str(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="meta-llama-3.1-8b_L16",
                   help="Cache tag (subdir of cache/)")
    a = p.parse_args()

    cache_dir = Path(__file__).resolve().parent.parent / "cache" / a.tag
    if not cache_dir.exists():
        raise SystemExit(f"No cache dir: {cache_dir}")

    rollup = [f"# 10-seed CI summary — {a.tag}\n",
              "Generated by `prototype/summarize_seeds.py`. Mean ± SEM across 10 seeds. "
              "Paired-difference cross-PCA flag (↑/↓/·) uses per-seed (SAE − PCA) at 2·SEM "
              "(~95% CI on the paired difference).\n"]
    for stage, spec in STAGES.items():
        md, path = _summarize_stage(cache_dir, stage, spec)
        if md is None:
            rollup.append(f"\n## {stage}\n{path}\n")
            continue
        rollup.append(f"\n## {stage}\nFull table: `{path}`\n")
        # Embed the body (drop the per-stage h1).
        body = "\n".join(md.split("\n")[1:])
        rollup.append(body)

    roll_path = cache_dir / "seeds10_summary.md"
    roll_path.write_text("\n".join(rollup))
    print(f"Wrote {roll_path}")


if __name__ == "__main__":
    main()
