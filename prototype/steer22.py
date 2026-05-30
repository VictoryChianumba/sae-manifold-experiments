"""
#22 follow-up to `steer.py`: multi-seed, second-target, unaligned-axis, fp32 controls.

The #21 sweep gave a clean "wider α range" answer — at 8B the legible-axis curve
asymptotes at α=+5 and never crosses zero, so the modulation cap is real (not "just
needed more α"). This script adds the three robustness checks the NEXT_SESSION.md
#22 spec called for:

  - Multi-seed (3 seeds of the aligned factored SAE) for both the legible axis and
    the equal-norm random control — does the slope hold across seeds?
  - Second target: years contrast (` twenty` vs ` nineteen`) — is the modulation cap
    specific to temperature/hot-cold or a general 8B steering property?
  - Unaligned-axis baseline (lam_label=0 factored SAE, same architecture and routing
    as the legible one, just no label supervision) — a tighter control than random.
  - fp32 control for the canonical cell (temperature/legible/seed=0) — rules out the
    asymptote being a bf16 precision artifact.

Run on the CUDA pod after the 10-seed #21 sweep has trained its SAEs:
  SAE_MODEL_NAME=NousResearch/Meta-Llama-3.1-8B SAE_LAYER=16 \\
  SAE_D_MODEL=4096 SAE_CACHE_TAG=meta-llama-3.1-8b_L16 \\
  python prototype/steer22.py

Output:  cache/<tag>/steer22/{steering22.png, results22.json, run.log}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from data import CACHE_DIR, DEVICE, LAYER, MODEL_NAME
from fair_comparison import load_split, factored_eval
from legible_coord import train_factored_legible, _mixture_arrays
from steer import readout_logit_contrast, _single_token_id


OUT_DIR = CACHE_DIR / "steer22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIXTURE = ["years", "age", "temperature", "colors", "geography"]
COORD_DIM = 3
ALPHAS = list(range(-6, 7))

# Two readout contrasts. Prompts are chosen so BOTH POS and NEG are plausible
# next-token continuations — otherwise the contrast measures prompt-priors not
# steering.
TARGETS = {
    "temperature": dict(
        prompts=[
            "The weather outside today is really",
            "Stepping outside, the air felt very",
            "Honestly, the temperature right now is extremely",
        ],
        pos=" hot", neg=" cold",
    ),
    "years": dict(
        prompts=[
            "She was born in",
            "The book was published in",
            "I last visited the city in",
        ],
        # POS=2000s era ("twenty twenty"), NEG=1900s era ("nineteen ninety-five").
        # Pushing +v along the legible years axis should bias toward the later era.
        pos=" twenty", neg=" nineteen",
    ),
}


def _build_vector(model, mean_std, per, target):
    """Same recipe as steer.legible_steering_vector but parameterized by target.
    Returns (v_raw, ||v||, dom_chart)."""
    mean, std = mean_std
    d = per[target]
    _, Ztr, Zte, dom = factored_eval(model, (mean, std),
                                     d["Xtr"], d["Xte"], d["mean_m"])
    ok = ~np.isnan(d["ytr"])
    probe = LinearRegression().fit(Ztr[ok], d["ytr"][ok])
    w = probe.coef_.astype(np.float32)
    w_hat = w / (np.linalg.norm(w) + 1e-8)

    Xn = torch.from_numpy((d["Xtr"] - mean) / std)
    with torch.no_grad():
        _, _, coords = model(Xn)
        z = coords[:, dom, :]
        wt = torch.from_numpy(w_hat)
        g0 = model.charts[dom](z)
        g1 = model.charts[dom](z + wt)
        delta_n = (g1 - g0).mean(0)
    v_raw = (delta_n * std).numpy().astype(np.float32)
    return v_raw, float(np.linalg.norm(v_raw)), int(dom)


def _load_llm_dtype(dtype):
    """Variant of data.load_llm that lets us pick the model dtype explicitly,
    so we can compare bf16 vs fp32 without changing data.py."""
    import nnsight
    from transformers import AutoTokenizer
    print(f"Loading {MODEL_NAME} on {DEVICE} (dtype={dtype})...")
    if DEVICE == "cuda":
        model = nnsight.LanguageModel(MODEL_NAME, device_map="auto",
                                      torch_dtype=dtype, dispatch=True)
    else:
        model = nnsight.LanguageModel(MODEL_NAME, device_map=DEVICE,
                                      torch_dtype=dtype, dispatch=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    for p in model.model.parameters():
        p.requires_grad = False
    return model, tok


def _sweep(llm, tok, label_for_log, vec, target_cfg):
    """One α sweep: vec is the raw steering vector, target_cfg = TARGETS[target]."""
    pos_id = _single_token_id(tok, target_cfg["pos"])
    neg_id = _single_token_id(tok, target_cfg["neg"])
    curve = []
    for a in ALPHAS:
        vals = [readout_logit_contrast(llm, tok, p, a * vec, pos_id, neg_id)
                for p in target_cfg["prompts"]]
        curve.append(float(np.mean(vals)))
    slope = float(np.polyfit(np.array(ALPHAS, float), curve, 1)[0])
    z0 = ALPHAS.index(0)
    delta_max = curve[-1] - curve[z0]
    print(f"    {label_for_log:36}  slope/α={slope:+.4f}  Δlogit(α=+6)={delta_max:+.3f}  curve[α=0]={curve[z0]:+.3f}")
    return dict(alphas=ALPHAS, curve=curve, slope=slope)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # -- 1. Train factored SAEs ------------------------------------------------
    # 3 aligned (lam_label=1) at seeds 0/1/2, plus 1 unaligned (lam_label=0) at seed 0.
    bundles = {}  # key -> (model, (mean,std), per, label)
    for seed in [0, 1, 2]:
        print(f"\n== train lam_label=1 seed={seed} ==")
        per, _ = load_split(MIXTURE, seed)
        X, m, y, names = _mixture_arrays(per)
        sae, ms = train_factored_legible(X, m, y, names,
                                         coord_dim=COORD_DIM, lam_label=1.0, seed=seed)
        bundles[f"aligned_seed{seed}"] = (sae, ms, per)

    print(f"\n== train lam_label=0 seed=0 (unaligned baseline) ==")
    per, _ = load_split(MIXTURE, 0)
    X, m, y, names = _mixture_arrays(per)
    sae, ms = train_factored_legible(X, m, y, names,
                                     coord_dim=COORD_DIM, lam_label=0.0, seed=0)
    bundles["unaligned_seed0"] = (sae, ms, per)

    # -- 2. Build steering vectors --------------------------------------------
    # vectors[(axis_kind, seed, target)] = (v, ||v||, dom)
    vectors = {}
    for key, (sae, ms, per) in bundles.items():
        kind = "legible" if key.startswith("aligned") else "unaligned"
        seed = int(key.split("seed")[-1])
        for target in TARGETS:
            v, vn, dom = _build_vector(sae, ms, per, target)
            vectors[(kind, seed, target)] = (v, vn, dom)
            print(f"  v[{kind} seed={seed} target={target}]  ||v||={vn:.3f}  dom={dom}")

    # Equal-norm random control: one per target, normed to the legible-seed0 vector.
    rng = np.random.default_rng(0)
    for target in TARGETS:
        canon_v, canon_norm, _ = vectors[("legible", 0, target)]
        rv = rng.standard_normal(canon_v.shape).astype(np.float32)
        rv *= canon_norm / (np.linalg.norm(rv) + 1e-8)
        vectors[("random", 0, target)] = (rv, canon_norm, -1)

    # -- 3. bf16 sweep over every (kind, seed, target) ------------------------
    print("\n== load model bf16 ==")
    llm, tok = _load_llm_dtype(torch.bfloat16)
    sweeps = {}
    for target, cfg in TARGETS.items():
        print(f"  target={target}  pos='{cfg['pos']}'  neg='{cfg['neg']}'")
        for (kind, seed, t), (v, vn, dom) in vectors.items():
            if t != target:
                continue
            sweeps[("bf16", kind, seed, target)] = _sweep(
                llm, tok, f"bf16 {kind} seed={seed}", v, cfg
            ) | dict(v_norm=vn, dom=dom)

    # -- 4. fp32 control for the canonical cell -------------------------------
    del llm
    torch.cuda.empty_cache()
    print("\n== load model fp32 (canonical-cell control) ==")
    llm32, tok32 = _load_llm_dtype(torch.float32)
    canon_v, canon_vn, canon_dom = vectors[("legible", 0, "temperature")]
    sweeps[("fp32", "legible", 0, "temperature")] = _sweep(
        llm32, tok32, "fp32 legible seed=0", canon_v, TARGETS["temperature"]
    ) | dict(v_norm=canon_vn, dom=canon_dom)
    # Also random control in fp32 to factor out dtype.
    rv = rng.standard_normal(canon_v.shape).astype(np.float32)
    rv *= canon_vn / (np.linalg.norm(rv) + 1e-8)
    sweeps[("fp32", "random", 0, "temperature")] = _sweep(
        llm32, tok32, "fp32 random seed=0", rv, TARGETS["temperature"]
    ) | dict(v_norm=canon_vn, dom=-1)
    del llm32
    torch.cuda.empty_cache()

    # -- 5. Persist + plot ----------------------------------------------------
    rp = OUT_DIR / "results22.json"
    rp.write_text(json.dumps({
        f"{dt}/{k}/seed{s}/{t}": v
        for (dt, k, s, t), v in sweeps.items()
    }, indent=2))
    print(f"\nSaved {rp}")

    # Plot: 2 cols (one per target) × 1 row of curves; show every bf16 curve plus the
    # fp32 control in temperature. Each curve is Δ logit vs α=0.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    color_for = {"legible": "tab:red", "unaligned": "tab:blue", "random": "tab:gray"}
    style_for = {"bf16": "-", "fp32": "--"}
    for ax, target in zip(axes, TARGETS):
        z0 = ALPHAS.index(0)
        for (dt, kind, seed, t), s in sweeps.items():
            if t != target:
                continue
            rel = [v - s["curve"][z0] for v in s["curve"]]
            label = f"{dt} {kind} s={seed}"
            ax.plot(ALPHAS, rel, style_for[dt] + "o", color=color_for[kind],
                    alpha=0.85 if dt == "bf16" else 0.6,
                    markersize=4, label=label)
        ax.axhline(0, color="k", lw=0.7, alpha=0.5)
        ax.set_title(f"target = {target}  ({TARGETS[target]['pos'].strip()} vs {TARGETS[target]['neg'].strip()})")
        ax.set_xlabel("α (× steering vector)")
        ax.set_ylabel(f"Δ logit({TARGETS[target]['pos'].strip()}−{TARGETS[target]['neg'].strip()}) vs α=0")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"§9b.7 #22 robustness: multi-seed × second-target × unaligned-axis × fp32-control",
                 fontsize=12)
    fig.tight_layout()
    pp = OUT_DIR / "steering22.png"
    fig.savefig(pp, dpi=130, bbox_inches="tight")
    print(f"Saved {pp}")


if __name__ == "__main__":
    main()
