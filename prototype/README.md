# Prototype: a factored "which-manifold + where-on-it" SAE

A toy of a **manifold-native** alternative to the standard SAE, motivated by the
finding in `../REPRODUCTION.md` that a dictionary of straight atoms shatters/dilutes
curved concept manifolds.

## Idea

Replace the dictionary-of-rays with an **atlas of charts** (`factored_sae.py`):

- a soft **router** `r(x)` → distribution over `M` charts — *which manifold*
- per-chart **coordinate** `z_m(x) ∈ R^coord_dim` — *where on it*
- per-chart **nonlinear decoder** `g_m(z_m)` → activation — a *curved* chart

```
x_hat = bias + Σ_m  a_m · g_m(z_m)
```

The router is the learned, **unsupervised** analogue of the paper's post-hoc feature
clustering — here it is baked into the architecture. (Router = MoE-style expert
selection; charts = local coordinate maps = an atlas. The two ideas unify.)

## Run

```bash
SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/factored_sae.py
# ablations:
SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/factored_sae.py --linear-charts
SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/factored_sae.py --linear-charts --coord-dim 1
```

Trains in ~1–2 min on CPU on the concept-manifold activations already in `cache/`.

## What it measures

- **Discovery** — does the router separate the manifolds unsupervised? (`purity`:
  fraction of a manifold's points landing on its dominant chart)
- **Geometry** — does each chart's coordinate *parameterize* the manifold?
  (`coord→label R²`: linear regression from the chart's coordinate to the manifold's
  ground-truth label, e.g. year/age/temperature)

## Result — the ablation ladder

Six manifolds (years, age, temperature, days, colors, geography), 12 charts,
SmolLM2-135M activations:

| chart type (per cluster) | overall recon VE | age R² | temp R² | years R² |
|---|---|---|---|---|
| **1 ray** (`--linear-charts --coord-dim 1`, SAE-like) | 0.978 | 0.08 | 0.01 | 0.04 |
| **3-D subspace** (`--linear-charts`) | 0.985 | 0.72 | 0.78 | 0.50 |
| **3-D curved chart** (default, nonlinear) | 0.996 | 0.95 | 0.86 | 0.42 |

Two takeaways, both echoing the paper:

1. **Reconstruction ≠ captured geometry.** The ray model reconstructs at 97.8%
   variance yet its coordinate recovers nothing (R²≈0) — the dilution/shattering
   failure reproduced inside this architecture.
2. **The lever is structural.** Subspace-per-cluster (baking in the clustering)
   takes coordinate-recovery from ~0 to ~0.7–0.8; **curvature** (nonlinear charts)
   adds the final lift on the genuinely curved manifolds.

## Caveats

- **Toy scale**: SmolLM2-135M activations; ~5.4k points; results are noisy
  (e.g. `years` splits across charts; nonlinear vs subspace R² ordering wobbles).
- **Cyclic-label artifact**: `colors` (hue) and `days` (time-of-day) wrap around, so
  *linear* regression on the raw label scores ≈0 even when the coordinate is good.
  A cyclic-aware score (predict sin/cos) would fix this — not yet implemented.
- Not a scalable SAE: no sparsity in the SAE sense, trained on tiny curated data.

## The fair comparison (`fair_comparison.py`)

The ablation ladder above is *suggestive but confounded* (coord-dim capacity,
in-sample R², no real-SAE baseline, one seed). `fair_comparison.py` is the
controlled redo — everything **held out**, scored in **raw activation units**
against a single per-manifold denominator, **3 seeds**:

- **Primary metric = the paper's own subspace-capture VE(N)**, made
  architecture-agnostic: for each manifold, the fraction of its *test* activation
  variance explained by an **N-dimensional per-point code**. PCA-N is the optimal
  linear ceiling; a linear dictionary can at best reach it; only a *curved* decoder
  can exceed it.
- **Curvature is the only free variable** between the two factored conditions
  (linear vs nonlinear charts at identical `coord_dim`, `n_charts`, training).
- **Two real SAE baselines**: the off-the-shelf C4-trained SAE, and a standard
  BatchTopK SAE *retrained on the same mixture train split* (isolates architecture
  from training data).

```bash
SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/fair_comparison.py \
    --seeds 0 1 2 --coord-dims 2 3
# -> cache/fair_comparison/{results.json, ve_curves.png, run.log}
```

### Result — held-out VE at N=3 (mean ± sd over 3 seeds)

| method | years | age | temperature | colors | geography |
|---|---|---|---|---|---|
| PCA (linear ceiling) | 0.39 | **0.79** | **0.79** | 0.74 | 0.52 |
| SAE-C4 geometric | 0.09 | 0.06 | 0.09 | 0.09 | 0.04 |
| SAE-mix geometric | 0.10 | 0.27 | 0.50 | 0.53 | 0.20 |
| SAE-mix statistical | 0.05 | 0.03 | 0.38 | 0.37 | 0.09 |
| Factored **LINEAR** | 0.09 | −0.07 | 0.05 | 0.57 | 0.12 |
| Factored **NONLINEAR** | **0.84** | 0.29 | 0.68 | **0.80** | **0.76** |

Curvature delta (`nl − lin`) is large and positive everywhere: years +0.75,
geography +0.65, temperature +0.63, age +0.36, colors +0.23 — robust across
seeds and at `coord_dim = 2` as well.

**The honest verdict — a split decision, leaning positive:**

1. **Curvature is real and decisive at matched dimensionality.** The nonlinear
   charts beat the linear charts at the *same* `coord_dim` on every manifold, by
   margins many× the seed spread. This is the cleanest possible isolation of the
   paper's thesis: the failure is the *straightness* of the atoms, not the
   clustering or the parameter count (the linear-chart control has both).
2. **On genuinely curved manifolds, curvature beats even the linear ceiling.**
   Factored-NL exceeds **PCA-N** on years (+0.45), geography (+0.25), colors
   (+0.07) — a curved 3-coord code reconstructs held-out activations better than
   *any* linear 3-D subspace can. That is a non-trivial positive result, not just
   "matches the remedy the paper already names."
3. **On essentially-linear manifolds it loses, by design.** age (−0.50) and
   temperature (−0.11) are near-1-D lines; PCA already saturates and the nonlinear
   chart *underfits* them (tiny manifolds: age = 69 train points). Curvature helps
   exactly where there is curvature — and the held-out split correctly penalises it
   where there isn't.
4. **The interpretability leg does NOT come for free — report this loudly.** The
   secondary *held-out label R²* (same regressor, all conditions) shows the
   nonlinear coords decode the label **worse** than PCA or even the linear charts on
   years (NL 0.16 lin / 0.29 kNN vs PCA 0.75) and geography. Better reconstruction
   geometry ≠ a more legible coordinate: the curved chart spends its fidelity on
   bending through activation space, not on laying the label out linearly. So this
   is a Pareto move on *geometry fidelity* but **not** (yet) on
   *interpretability/parsimony* — the bar we set ourselves. kNN recovers some of the
   gap, so the label is present in the coords but nonlinearly embedded.

Bottom line: **curvature genuinely buys held-out geometric fidelity beyond the
linear ceiling on curved manifolds** — the central claim survives a fair test. But
the win is reconstruction-geometry, and the interpretability payoff the whole
premise rests on is *not* demonstrated and partly contradicted. Earning that is the
real next step, not a foregone conclusion.

## Making the coordinate legible (`legible_coord.py`)

The unmet bar above was *interpretability*: the nonlinear coordinate reconstructed
well but decoded the label poorly. `legible_coord.py` tests whether a coordinate
can be **both** by adding a weak **label-alignment** term — a per-manifold linear
probe on the dominant chart's coordinate, trained on TRAIN labels only
(per-manifold standardized) — and sweeping its weight `lam_label`. The reported
label R² still uses a *fresh* held-out linear probe (same protocol as
`fair_comparison`), so alignment only *shapes* the coordinate; R² still tests
generalization.

```bash
SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/legible_coord.py \
    --seeds 0 1 2 --lams 0 0.3 1 3 10
# -> cache/legible_coord/{results.json, pareto.png, run.log}
```

### Result — legibility is nearly free (3 seeds, coord_dim = 3)

| lam_label | VE@3 (mean) | label R² linear, non-cyclic mean | years R² | geography R² |
|---|---|---|---|---|
| 0 (unsupervised) | 0.68 | 0.55 | 0.16 | 0.41 |
| 0.3 | 0.67 | 0.66 | 0.24 | 0.69 |
| **1** | **0.68** | **0.89** | **0.89** | **0.80** |
| 3 | 0.64 | 0.94 | 0.98 | 0.87 |
| 10 | 0.63 | 0.95 | 0.99 | 0.91 |

**Verdict — the interpretability bar is met (for curved manifolds).** At
`lam_label = 1`, held-out label R² jumps from 0.16→0.89 (years) and 0.41→0.80
(geography) **with essentially no reconstruction cost** (VE@3 mean 0.68→0.68; years
0.84→0.83). At that operating point the curved-manifold coordinate **beats PCA on
both axes simultaneously** — fidelity (years VE 0.83 vs PCA-3 0.39; geography 0.76
vs 0.52) *and* legibility (years R² 0.89 vs PCA 0.75; geography 0.80 vs 0.60). So
the earlier fidelity↔legibility tension was an artifact of the *unsupervised*
coordinate being free to bend arbitrarily; a weak orientation prior removes it.

Honest caveats (see `pareto.png`):
- **Push too hard and the tradeoff reappears.** At `lam = 10`, R² saturates (~0.95)
  but VE erodes (geography 0.76→0.69, colors 0.80→0.77). The clean operating point
  is the knee, `lam ≈ 1`; alignment and reconstruction only *compete* past it.
- **This uses weak label supervision.** The router/charts stay unsupervised; only
  the within-chart *axis* is oriented by the label. That fits a "probe a known
  concept" setting, not unsupervised discovery — be clear about which claim is made.
- **`age` stays weak** (VE ≈ 0.29, and it degrades at high `lam`): 69 train points
  is too few for the MLP, so alignment and reconstruction fight over scarce data.
- **`colors` needs cyclic handling** — resolved in the section below.

## Cyclic-aware scoring AND alignment for colors

Colours are labelled by **hue**, which wraps (0.99 and 0.01 are both red), so a
*line* is the wrong target — both the scoring and the alignment have to know about
the loop.

**Scoring** (`fair_comparison.label_score`, `CYCLIC_LABEL = {"colors": 1.0}`):
regress the N-dim representation onto `(cos θ, sin θ)` of the hue angle, multi-output
R², same linear + kNN regressors; non-cyclic manifolds unchanged. This alone showed
the earlier ≈0 was a *scoring* artifact — PCA actually recovers 0.30 of the hue
circle — but did **not** rescue the factored coordinate.

**Alignment** (`legible_coord.train_factored_legible`): the per-manifold probe now
targets a *concept-shaped* quantity — a standardized scalar for non-cyclic labels,
but `(cos θ, sin θ)` for cyclic ones (a 2-output probe). A linear probe `w·z + b`
cannot orient a coordinate onto a circle; a `cos/sin` target can.

### Result — cyclic alignment makes colours legible (3 seeds, cyclic R²)

| lam_label | colors, LINEAR align (old) | colors, CYCLIC align (new) | colors VE@3 |
|---|---|---|---|
| 0 | 0.13 | 0.13 | 0.80 |
| 0.3 | 0.13 | 0.24 | 0.80 |
| 1 | 0.15 | 0.33 | 0.80 |
| 3 | 0.17 | **0.41** | 0.80 |
| 10 | 0.34 | **0.54** | 0.76 |

With the matched **cyclic** alignment, colours' held-out cyclic R² climbs from 0.13
to 0.54 and **crosses its PCA reference (0.30) at `lam ≈ 1`** while VE stays at 0.80
(above PCA-3's colours VE ≈ 0.74) — so colours, too, becomes a modest Pareto win,
exactly where the linear probe had left it stuck below PCA. This confirms the
diagnosis was mechanical: legibility needs a *concept-shaped* target, not just a
fair score.

Honest residue: colours is still the **hardest** manifold in absolute terms
(R² ~0.4–0.5, vs ~0.9 for years/geography). Hue is only one of three varying colour
attributes (lightness/saturation also move) over ~1.8 k noisy points, so the loop is
partly entangled; cyclic alignment lifts it clearly but does not fully linearise it.
The `lam = 10` over-alignment knee (VE erosion) is unchanged.

## Unsupervised isometry legibility (`iso_coord.py`) — a negative result

Can a coordinate be made legible **without any labels**? `iso_coord.py` drops the
label-alignment crutch and instead adds a label-free **isometry / arc-length**
penalty (cf. Gropp et al., *Isometric Autoencoders*, 2020): along a random unit
direction in coordinate space, each chart decoder's finite-difference speed
`‖J_m u‖` is pinned to a fixed target (router-weighted), making the decoder
constant-speed so the coordinate becomes arc-length. The fixed (not free) target is
what stops the trivial `z → const` collapse a variance-only penalty allows. No label
is used in training; held-out label R² (cyclic-aware) is a pure generalization test.

### Result — isometry does *not* recover legibility (3 seeds)

| lam_iso | VE@3 (mean) | label R², non-cyclic mean* | years | geography | colors (cyclic) |
|---|---|---|---|---|---|
| 0 (unsup.) | 0.68 | 0.55 | 0.16 | 0.41 | 0.13 |
| 1 | 0.65 | 0.63 | 0.45 | 0.35 | 0.19 |
| 3 | 0.60 | 0.60 | 0.31 | 0.39 | 0.11 |
| 10 | 0.32 | 0.55 | 0.11 | 0.51 | 0.21 |
| 30 | −0.06 | 0.60 | 0.39 | 0.11 | 0.18 |

**Verdict — negative, and clearly so.** The legibility mean* never moves outside
seed noise (0.55 → 0.63 peak at `lam = 1`, then flat), and that tiny bump already
costs reconstruction (VE 0.68 → 0.65). The isometry points **never reach the PCA
reference** on legibility for any manifold (see `pareto.png`), and any weight large
enough to actually reshape the coordinate **destroys reconstruction** (VE → 0.32 at
`lam = 10`, negative at 30). Compare the supervised result: a *concept-shaped* label
prior took mean* from 0.55 → 0.89 at `lam = 1` with **zero** VE cost.

Why it fails — and it's instructive:
- **Arc-length ≠ linear-in-coordinate.** Isometry makes equal coordinate steps equal
  manifold steps, but a 3-D isometric coordinate can still *wind* arbitrarily through
  coordinate space; linear R² needs the label to be ~linear in the coords, which
  isometry does not impose. Arc-length is necessary, not sufficient, for legibility.
- **It fights reconstruction.** Forcing uniform decoder speed distorts the chart's
  fit to a non-uniformly-curved manifold, so VE collapses well before legibility
  improves.
- **No help for the entangled/cyclic cases** (colors stuck ~0.1–0.2): isometry says
  nothing about *which* coordinate direction is the concept.

So legibility here is cheap **with** a weak, concept-shaped label prior but not
recoverable from this label-free isometry prior *alone* — which motivated adding
the missing ingredient below.

## Isometry + parsimony (`iso_parsimony.py`) — label-free, and it works on low-D manifolds

The isometry post-mortem said arc-length is necessary but not sufficient: a 3-D
isometric coordinate can still *wind*. The fix is to also stop it having spare dims
to wind in. `iso_parsimony.py` adds a **coordinate-parsimony** prior to isometry,
both still label-free:

- **parsimony**: minimize the **participation ratio** of the coordinate's per-dim
  variance, `PR = (Σ vᵢ)² / Σ vᵢ² ∈ [1, cd]` — a *scale-invariant* count of active
  dims (an L1-on-std penalty can be gamed by shrinking the coord and growing the
  decoder; PR cannot). Driving PR→1 collapses unused dims; reconstruction keeps the
  needed ones. We grid over `(lam_iso, lam_pars)` so each prior's marginal effect
  shows (iso=0 row = parsimony alone; pars=0 row = isometry alone).

### Result — a clean split by intrinsic dimensionality (3 seeds, label-free)

Held-out label R² (R² *not* a training target — no labels used):

| config (iso, pars) | years | temperature | age | geography | colors | active-dims (PR) |
|---|---|---|---|---|---|---|
| (0, 0) unsup. | 0.16 | 0.79 | 0.84 | 0.41 | 0.13 | ~2.3 |
| (0, 8) parsimony | 0.55 | 0.82 | 0.84 | 0.01 | 0.05 | ~1.3 |
| (1, 8) iso+parsimony | **0.70** | **0.89** | 0.84 | 0.08 | 0.07 | ~1.4 |
| PCA reference | 0.75 | 0.92 | 0.91 | 0.60 | 0.30 | — |

Parsimony does exactly what it says — the held-out coordinate's active-dim count
(PR) collapses from ~2.3 toward 1 — and the effect on legibility is **cleanly split
by the manifold's intrinsic dimension**:

- **It works, label-free, on the genuinely low-D manifolds.** Years (a helix that
  is ~1-D in arc length) jumps **0.16 → 0.70** and temperature **0.79 → 0.89** —
  both now near their PCA reference (0.75 / 0.92), and years is in the
  neighbourhood of the *supervised* result (0.89). This is the first time the
  unsupervised route recovered real legibility: collapse the spare dims, make the
  survivor arc-length, and a clean 1-D concept lines up linearly **with no labels**.
- **It backfires on the intrinsically multi-D manifolds.** A single global parsimony
  weight over-collapses geography (~2-3D on the sphere: 0.41 → 0.01) and colors (a
  2-D cyclic loop: → 0.07), destroying both their legibility *and* their VE — you
  cannot force a sphere onto one straight axis.
- **So the aggregate is flat** (non-cyclic mean R² ~0.55 → ~0.63, within seed noise)
  because the low-D gains and multi-D losses cancel, and VE drops overall as
  parsimony bites (0.68 → ~0.53 at `pars = 8`).

**Verdict — a qualified positive.** Parsimony *was* the missing ingredient the
isometry post-mortem predicted: with it, label-free legibility is genuinely
recoverable, but only when the parsimony strength matches the manifold's intrinsic
dimension. A global weight can't know that, so it helps 1-D manifolds and harms
higher-D ones. The clear next step is an **intrinsic-dimension-adaptive** parsimony
(let each chart learn how many coordinate dims it needs) rather than one global knob.

## Possible next steps

- **Intrinsic-dimension-adaptive parsimony**: a per-chart learned/annealed target
  dimensionality (or a nested/Matryoshka coordinate with a per-dim KL gate) so each
  manifold keeps exactly the dims it needs — fixing the over-collapse of geography/
  colors while keeping the years/temperature win.
- A proper isometric-AE (exact Jacobian + encoder pseudo-inverse term) rather than
  the finite-difference surrogate here.
- A **causal/steering** leg: move along a (now-legible) chart coordinate → smooth
  predicted-output change.
- Group-sparse (top-k charts) selection instead of soft softmax.
