"""
Causal/steering leg: is the legible coordinate the *concept*, not just decodable?

Everything else shows the chart coordinate can be *read*. This asks whether it is
*causal*: build a steering vector from the legible axis, add it to a readout
prompt's activation, and see whether the model's behaviour moves along the concept.

Method (standard activation steering, but the direction comes from our legible
coordinate):
  1. Train the aligned factored SAE (lam=1) on the manifold mixture; for the target
     manifold, find its dominant chart and the **legible axis** ŵ = the (unit)
     direction in coordinate space that the held-out label probe maps to the label.
  2. **Steering vector** v = activation displacement for +1 step along ŵ, averaged
     over the manifold (in raw activation units):  v = std · mean_x[ g_m(z+ŵ) − g_m(z) ].
  3. Patch a readout prompt's layer-L last-token activation: x' = x + α·v (add the
     delta — the reconstruction is lossy, so we steer, not replace), run the rest of
     the model, and read a behavioural **logit contrast** (e.g. " hot" − " cold").
  4. Sweep α. The legible axis should move the contrast **smoothly and monotonically**.
  5. **Control**: a random direction of equal norm should NOT — otherwise we've shown
     nothing.

Run (CPU, from repo root; loads SmolLM2-135M):
  SAE_MODEL_NAME=HuggingFaceTB/SmolLM2-135M SAE_LAYER=19 \
  SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/steer.py

Run (CUDA pod, Llama-3.1-8B layer 16 — task #13a):
  SAE_MODEL_NAME=NousResearch/Meta-Llama-3.1-8B SAE_LAYER=16 \
  SAE_D_MODEL=4096 SAE_CACHE_TAG=meta-llama-3.1-8b_L16 \
  python prototype/steer.py
  (SAE_DEVICE auto-selects cuda; cache lands in cache/<tag>/steer/.)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from data import CACHE_DIR, DEVICE, LAYER, load_llm
from fair_comparison import load_split, factored_eval
from legible_coord import train_factored_legible, _mixture_arrays

RESULTS_DIR = CACHE_DIR / "steer"
MIXTURE = ["years", "age", "temperature", "colors", "geography"]

# Target manifold + behavioural readout: a prompt whose next token reflects the
# concept, and a contrast token pair that should move in opposite directions.
TARGET = "temperature"
READOUT_PROMPTS = [
    "The weather outside today is really",
    "Stepping outside, the air felt very",
    "Honestly, the temperature right now is extremely",
]
POS_WORD, NEG_WORD = " hot", " cold"      # +v should raise hot, lower cold
ALPHAS = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]


def legible_steering_vector(coord_dim=3, seed=0):
    """Train aligned factored SAE; return (v_raw, ||v_raw||, dom_chart, model bits)
    for the TARGET manifold's legible axis."""
    per, _ = load_split(MIXTURE, seed)
    Xtr_mix, mid_mix, y_mix, names = _mixture_arrays(per)
    model, (mean, std) = train_factored_legible(
        Xtr_mix, mid_mix, y_mix, names, coord_dim=coord_dim, lam_label=1.0, seed=seed)

    d = per[TARGET]
    _, Ztr, Zte, dom = factored_eval(model, (mean, std), d["Xtr"], d["Xte"], d["mean_m"])
    ok = ~np.isnan(d["ytr"])
    probe = LinearRegression().fit(Ztr[ok], d["ytr"][ok])
    w = probe.coef_.astype(np.float32)
    w_hat = w / (np.linalg.norm(w) + 1e-8)               # legible axis (unit, in coord space)

    # Activation displacement per +1 step along w_hat, averaged over the manifold.
    Xn = torch.from_numpy((d["Xtr"] - mean) / std)
    with torch.no_grad():
        _, _, coords = model(Xn)
        z = coords[:, dom, :]                            # [N, cd]
        wt = torch.from_numpy(w_hat)
        g0 = model.charts[dom](z)
        g1 = model.charts[dom](z + wt)
        delta_n = (g1 - g0).mean(0)                      # normalized-activation delta
    v_raw = (delta_n * std).numpy().astype(np.float32)   # raw-activation steering vector
    return v_raw, float(np.linalg.norm(v_raw)), dom


@torch.no_grad()
def readout_logit_contrast(model, tok, prompt, v_alpha, pos_id, neg_id):
    """Patch layer-L last-token activation by v_alpha, return logit(pos) - logit(neg)."""
    # vt must match the model's device + dtype: at 135M (CPU/fp32) this is a no-op,
    # at 8B on CUDA the model is typically bf16 and a default-fp32-CPU vt would die
    # with a device-or-dtype mismatch inside the trace.
    mdtype = next(model.model.parameters()).dtype
    vt = torch.tensor(v_alpha, dtype=mdtype, device=DEVICE)
    with model.trace(prompt):
        model.model.layers[LAYER].output[0][..., -1, :] += vt
        logits = model.output.logits[..., -1, :].save()
    lg = logits.float().reshape(-1)
    return float(lg[pos_id] - lg[neg_id])


def _single_token_id(tok, word):
    ids = tok.encode(word, add_special_tokens=False)
    return ids[-1] if len(ids) else tok.encode(word.strip(), add_special_tokens=False)[-1]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Computing legible steering vector...")
    v_raw, v_norm, dom = legible_steering_vector()
    print(f"  target={TARGET} dom_chart={dom} ||v||={v_norm:.2f}")

    rng = np.random.default_rng(0)
    v_rand = rng.standard_normal(v_raw.shape).astype(np.float32)
    v_rand *= v_norm / (np.linalg.norm(v_rand) + 1e-8)    # equal-norm random control

    print("Loading model...")
    model, tok = load_llm()
    pos_id, neg_id = _single_token_id(tok, POS_WORD), _single_token_id(tok, NEG_WORD)
    print(f"  contrast tokens: '{POS_WORD}'={pos_id}  '{NEG_WORD}'={neg_id}")

    # contrast(alpha), averaged over readout prompts, for legible vs random.
    curves = {}
    for label, vec in [("legible axis", v_raw), ("random control", v_rand)]:
        means = []
        for a in ALPHAS:
            vals = [readout_logit_contrast(model, tok, p, a * vec, pos_id, neg_id)
                    for p in READOUT_PROMPTS]
            means.append(float(np.mean(vals)))
            print(f"  {label:14} α={a:+d}  logit({POS_WORD.strip()}-{NEG_WORD.strip()})={means[-1]:+.3f}")
        curves[label] = means

    # Center each curve on its α=0 value so we read the *change* the steering induces.
    z0 = ALPHAS.index(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, c in curves.items():
        rel = [v - c[z0] for v in c]
        ax.plot(ALPHAS, rel, "-o",
                color="tab:red" if "legible" in label else "tab:gray", label=label)
    ax.axhline(0, color="k", lw=0.7, alpha=0.5)
    ax.set_xlabel("steering strength α (× legible/control vector)")
    ax.set_ylabel(f"Δ logit( '{POS_WORD.strip()}' − '{NEG_WORD.strip()}' )  vs α=0")
    ax.set_title(f"Steering along the legible '{TARGET}' coordinate\n"
                 "(hotter → +; monotone for the concept axis, flat for random)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = RESULTS_DIR / "steering.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"Saved {p}")

    # Headline numbers: slope of the legible curve and the control, by least squares.
    A = np.array(ALPHAS, float)
    slopes = {}
    for label, c in curves.items():
        slope = float(np.polyfit(A, c, 1)[0])
        slopes[label] = slope
        print(f"  {label}: logit-contrast slope per unit α = {slope:+.3f}")

    # Persist the numbers so the writeup can cite them without re-running.
    results = {
        "target": TARGET, "pos_word": POS_WORD, "neg_word": NEG_WORD,
        "alphas": ALPHAS, "curves": curves, "slopes": slopes,
        "v_norm": v_norm, "dom_chart": dom,
        "readout_prompts": READOUT_PROMPTS,
    }
    rp = RESULTS_DIR / "results.json"
    rp.write_text(json.dumps(results, indent=2))
    print(f"Saved {rp}")


if __name__ == "__main__":
    main()
