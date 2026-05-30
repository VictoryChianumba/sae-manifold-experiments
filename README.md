# sae-manifold-experiments

Experiments building on Bhalla et al., "Do Sparse Autoencoders Capture Concept Manifolds?" (Goodfire, 2026). Personal project, exploratory scope, M1 + a few rented A100 sessions.

## The question

The paper shows that SAEs capture curved concept manifolds (years as a helix, colors as a wheel, geography on a sphere) by tiling them with localized linear features — a regime they call dilution. They argue future architectures should treat geometric objects, not directions, as the basic unit, and explicitly flag that they have new architectures in the works.

This repo is a solo attempt to build one such architecture as a learning exercise — a factored "atlas-of-charts" SAE — and to honestly characterize what it does and doesn't buy at small scale (SmolLM2-135M) and on the paper's actual model (Llama-3.1-8B layer 16).

## The architecture

A standard SAE describes activations with a dictionary of straight directions. An atlas-SAE replaces that with:

- a router r(x) → distribution over M charts ("which manifold"),
- a per-chart coordinate z_m(x) ∈ R^cd ("where on it"),
- a per-chart nonlinear decoder g_m(z_m) → activation (a curved local chart).

Reconstruction is x̂ = bias + g_dom(z_dom(x)) + bias under hard top-1 routing. Each chart is a small MLP, so a single chart can bend along a curved manifold that a linear dictionary would need many atoms to tile poorly.

## What's in the repo

Five findings, in order of how robustly I'd defend them. Most of the substantive wins concentrate on one manifold (geography); the other four manifolds either tie PCA or are honest negatives. I'll be transparent about that throughout.

### 1. Supervised concept-shaped legibility crosses PCA cleanly on geography at 8B

`legible_coord` adds a weak per-manifold label-alignment term (standardized scalar for non-cyclic labels, (cos θ, sin θ) for cyclic ones) to the factored-SAE training. At λ=10 on Llama-3.1-8B layer 16, 10 seeds:

- geography: R² 0.776 ± 0.010 vs PCA 0.472 ± 0.010 (~30 SEM gap, +0.30 lift), with VE also rising 0.46 → 0.52.
- years: R² 0.963 ± 0.005 vs PCA 0.948 ± 0.003 (marginal but clean, ~3 SEM).
- temperature, colors, age: tie PCA within seed noise.

The geography result is the cleanest Pareto crossing the project produced. An earlier 3-seed snapshot suggested supervised legibility crossed PCA on 4/5 manifolds; with proper seed counts the win is concentrated on geography, marginal on years, and absent elsewhere. The "PCA already saturates at 8B" framing from intermediate drafts of this writeup was also partially wrong — PCA's geography R² is 0.47, not saturated, and that's specifically where supervision pays off.

### 2. Label-free recovery crosses PCA on geography at 8B (smaller, but real)

`iso_parsimony` (isometry + participation-ratio penalty on coordinate variance, both label-free) at iso=1/pars=0 on geography: R² 0.542 ± 0.024 vs PCA 0.472 ± 0.010 — first 2·SEM-significant unsupervised crossing of PCA at 8B in the project. Narrow (one config, one manifold), but the geometry result and the methodology are both clean.

Adaptive gating (`adaptive_parsimony`, learned per-dim gates) also crosses PCA on geography (~+0.10) but actively hurts colors at 2·SEM (0.716 ± 0.056 vs PCA 0.925 ± 0.001) — learned gates aren't a free architectural addition.

### 3. Methodological warning #1: soft routing destroys what you're trying to measure

Early version of this work evaluated reconstruction using the full router-weighted mixture across all 12 charts and compared its variance-explained against a 3-dimensional PCA baseline. That isn't matched-dim: the soft mixture has 12 × 3 = 36 latent dims plus an input-dependent gating signal, vs PCA-3's 3. The early "factored-NL beats PCA by +0.45 on years" headline was inflated by this mismatch.

Catching this required noticing that a strictly-linear factored model was beating PCA-3 by +0.15 on held-out data — mathematically impossible at matched dim, since PCA-3 is the optimal 3-d linear projection by construction. That number was the bug's signature.

After the fix (hard top-1 routing, eval restricted to the dominant chart's reconstruction), the matched-dim VE story shrank: factored-NL beats PCA-3 on years (+0.12) and loses on geography, colors, temperature, age. The NL−LIN delta survives at +0.3 to +2.0 across the board, so curvature is doing real work against the linear-chart baseline — just not enough to beat the linear ceiling on most manifolds.

The transferable warning: any MoE-style SAE variant using soft routing for differentiability is potentially diluting both reconstruction comparability and per-chart interpretability into the gating mechanism. Hard routing alone — same architecture otherwise — lifted unsupervised label decodability on years from 0.16 to 0.57.

### 4. Causal steering: the supervision is what makes the axis causal

`steer.py` builds a steering vector from the legible temperature axis, adds α·v to a readout prompt's layer-16 last-token activation, and reads logit(" hot") − logit(" cold"). At 8B, 3 SAE training seeds:

- Legible-axis slope: +0.022 / +0.090 / +0.288 — all positive, sign-consistent, but magnitude varies ~13× across seeds.
- Unaligned-axis control (same architecture, same routing, no label supervision, all 3 seeds): slope −0.031, flat-or-wrong-direction.
- Random-direction control (equal norm): flat.

The unaligned-axis comparison is the genuine causal finding: the label supervision specifically — not the architecture, not the routing, not the choice of direction — is what creates a directionally consistent causal axis. That's a cleaner attribution than "legible axis vs random," which doesn't isolate which component matters.

At wide α-range (±6), 1/3 of seeds crosses the cold→hot decision boundary (control), 2/3 stop at modulation. An earlier single-seed framing called the 8B result "modulation only" in contrast to a 135M "control" finding; with seeds, that qualitative gap probably doesn't hold — the 135M result was also 1-seed.

A second contrast (years, " twenty" vs " nineteen") didn't replicate the temperature finding — legible and unaligned axes track together. Likely cause: years splits across multiple charts unstably across seeds, and the contrast tokens are polysemous. Temperature is the test that worked.

### 5. Methodological warning #2: single-seed results are provisional, including negatives

Three times in this project a single-seed result misled the writeup: the inflated factored-NL VE (turned out to be a protocol bug), the "6× over random control" steering ratio (turned out to be 1.7×–13× depending on seed and α-range), and the "modulation, not control" 8B steering verdict (turned out to be 1/3 control, 2/3 modulation across seeds).

At the scales and noise levels of this kind of work, anything reported from a single seed is provisional regardless of whether the result is positive or negative. The temptation to draw conclusions from a single run is strongest after an honest reframe pass, when "modulation only" feels like the calibrated answer — but the calibrated answer requires the seed count to back it up.

## What didn't survive

- The original "+0.45 VE on years over PCA at matched dim" headline. Bug.
- "Free legibility: years 0.16 → 0.89." Half the lift was the hard-routing protocol fix, half was supervision. Reframed.
- "Supervised legibility crosses PCA on 4/5 manifolds." 3-seed-snapshot artifact; with 10 seeds, the win concentrates on geography (+0.30) and is marginal-to-absent elsewhere.
- "Gates close uniformly regardless of intrinsic dim" (adaptive-parsimony negative). Post-fix, gates do differentiate — but in the wrong direction, because per-chart dim count measures "how much of the manifold this chart sees" rather than intrinsic dim.
- "Modulation cap at 8B steering, period." Holds for 2/3 seeds; 1/3 crosses zero. The qualitative 135M↔8B "control→modulation" gap is probably not real under seed-aware framing.

## What this isn't

Not a contribution against the Goodfire paper. Their Ising-coupling clustering (Section 6 of the same paper) and this repo's factored atlas are parallel responses to the same diagnosis — SAEs dilute curved manifold geometry — tackled with different machinery: theirs keeps the SAE and recovers structure post-hoc; this one replaces the SAE with a routed nonlinear architecture so dilution doesn't happen in the first place. Theirs runs at scale on real LLM features and finds novel manifolds; this one runs at small scale on five concept manifolds and gets one clean geography crossing. The genuine followup work — the "new architectures in the works" the paper flags in its conclusion — isn't public yet and probably supersedes both strands. Not a novel architecture in any field sense — soft-routed MoE-style SAEs exist; factored decoders exist. Not a publishable result.

## What this is

A worked example of building an interpretability research pipeline end-to-end on constrained hardware, finding a load-bearing bug in one's own protocol, re-running the corrected experiment that hurts the headline, expanding to proper seed counts when single-seed verdicts looked too clean, and reporting what survives without inflation. Two methodological warnings (soft-routing dilution; single-seed verdicts unreliable in both directions) are transferable beyond this rig. The geography results are real. Most of the other manifolds tie PCA or honestly lose. That's the project.

## Reproducing

[install / run instructions]
