"""
End-to-end driver for the SAE-manifold experiment suite — built so a rented GPU
runs the whole pipeline **unattended** instead of us debugging interactively on the
clock.  Model-agnostic: it reads the model config from the same env vars as the rest
of the pipeline (`SAE_MODEL_NAME`, `SAE_LAYER`, `SAE_D_MODEL`, `SAE_DEVICE`), so the
*identical* command runs SmolLM2-135M on a laptop CPU and Llama-3.1-8B on a GPU.

Per layer it runs these stages in order (each = a subprocess with a per-(model,layer)
cache tag, so nothing collides and a crash resumes from the last finished stage):

  extract     data.py                  -> cache/<tag>/<manifold>.pt
  background  background.py            -> cache/<tag>/background_acts_<bg>.dat
  sae         train_sae.py             -> cache/<tag>/sae_<d_sae>_k<k>.pt
  fair        fair_comparison.py  (#13) -> cache/<tag>/fair_comparison/results.json
  legible     legible_coord.py         -> cache/<tag>/legible_coord/results.json
  iso         iso_parsimony.py         -> cache/<tag>/iso_parsimony/results.json
  adaptive    adaptive_parsimony.py(#12)-> cache/<tag>/adaptive_parsimony/results.json

Stages are idempotent: an existing non-empty output is skipped unless --force.
`--stages` selects a subset; `--layers` loops the whole suite over several layers
(the #13 layer-sensitivity sweep); `--seeds` is forwarded to every experiment (more
seeds = the #17 statistical-rigor pass).  #14 (in-the-wild router), #15 (group-sparse
charts) and #16 (proper isometric-AE) are not yet implemented; add them as new stages
once their scripts exist.

Examples
--------
  # CPU smoke test on the small model (reuses cached 135M activations):
  SAE_DEVICE=cpu SAE_D_MODEL=576 SAE_MODEL_NAME=HuggingFaceTB/SmolLM2-135M SAE_LAYER=19 \
      uv run python prototype/run_all.py --stages fair legible adaptive --seeds 0 \
      --cache-tag "" --dry-run

  # Full real-model run on a GPU (Llama-3.1-8B, the paper's model):
  uv run python prototype/run_all.py --layers 16 --seeds 0 1 2 --bg-tokens 200000
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROTO = REPO / "prototype"
CACHE_BASE = REPO / "cache"

DEFAULT_MANIFOLDS = ["years", "age", "temperature", "colors", "geography"]


def model_slug(model_name):
    """meta-llama/Llama-3.1-8B -> llama-3.1-8b (a filesystem-safe cache tag stem)."""
    return model_name.split("/")[-1].lower().replace("_", "-")


def cache_dir_for(tag):
    return (CACHE_BASE / tag) if tag else CACHE_BASE


def _nonempty(p: Path):
    return p.exists() and (p.is_dir() or p.stat().st_size > 0)


def build_stages(args, tag, layer, d_sae, sae_path, cdir):
    """Each stage: (name, argv, output_to_check_for_skip).  argv is run with the
    same interpreter (sys.executable) inside the per-layer env."""
    seeds = [str(s) for s in args.seeds]
    bg = str(args.bg_tokens)
    manifold_pts = [cdir / f"{m}.pt" for m in args.manifolds]
    return [
        ("extract", [str(REPO / "data.py"), "--manifold", *args.manifolds],
         manifold_pts),
        ("background", [str(REPO / "background.py"), "--n-tokens", bg],
         [cdir / f"background_acts_{bg}.dat"]),
        ("sae", [str(REPO / "train_sae.py"), "--n-tokens", bg,
                 "--expansion-factor", str(args.expansion), "--k", str(args.k),
                 "--epochs", str(args.sae_epochs)],
         [sae_path]),
        ("fair", [str(PROTO / "fair_comparison.py"), "--manifolds", *args.manifolds,
                  "--seeds", *seeds, "--coord-dims", "2", "3",
                  "--c4-sae", str(sae_path)],
         [cdir / "fair_comparison" / "results.json"]),
        ("legible", [str(PROTO / "legible_coord.py"), "--manifolds", *args.manifolds,
                     "--seeds", *seeds, "--lams", "0", "0.3", "1", "3", "10"],
         [cdir / "legible_coord" / "results.json"]),
        ("iso", [str(PROTO / "iso_parsimony.py"), "--manifolds", *args.manifolds,
                 "--seeds", *seeds, "--lam-isos", "0", "1", "--lam-pars", "0", "2", "8"],
         [cdir / "iso_parsimony" / "results.json"]),
        ("adaptive", [str(PROTO / "adaptive_parsimony.py"), "--manifolds", *args.manifolds,
                      "--seeds", *seeds, "--lam-isos", "0", "1",
                      "--lam-gates", "0", "2", "4", "8"],
         [cdir / "adaptive_parsimony" / "results.json"]),
    ]


def run_stage(name, argv, env, logdir):
    logdir.mkdir(parents=True, exist_ok=True)
    logpath = logdir / f"{name}.log"
    cmd = [sys.executable, *argv]
    print(f"    $ {' '.join(cmd)}")
    print(f"    log -> {logpath}")
    t0 = time.time()
    with open(logpath, "w") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf,
                              stderr=subprocess.STDOUT, cwd=str(REPO))
    dt = time.time() - t0
    return proc.returncode, dt, logpath


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="Layers to sweep (default: just SAE_LAYER from env)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--manifolds", nargs="+", default=DEFAULT_MANIFOLDS)
    ap.add_argument("--stages", nargs="+", default=None,
                    help="Subset/order of stages (default: all)")
    ap.add_argument("--bg-tokens", type=int, default=200_000,
                    help="Background C4 tokens for the standard-SAE baseline")
    ap.add_argument("--sae-epochs", type=int, default=40)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--cache-tag", default=None,
                    help="Override the auto <model>_L<layer> tag (rarely needed; "
                         "use '' to write to the flat cache/ dir)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run stages even if their output already exists")
    ap.add_argument("--keep-going", action="store_true",
                    help="Continue to the next layer if a stage fails")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan (commands, tags, skip decisions) and exit")
    args = ap.parse_args()

    model_name = os.environ.get("SAE_MODEL_NAME", "meta-llama/Llama-3.1-8B")
    d_model = int(os.environ.get("SAE_D_MODEL", "4096"))
    d_sae = args.expansion * d_model
    layers = args.layers or [int(os.environ.get("SAE_LAYER", "19"))]
    slug = model_slug(model_name)

    print(f"Driver: model={model_name} (slug={slug}) d_model={d_model} d_sae={d_sae}")
    print(f"        layers={layers} seeds={args.seeds} bg_tokens={args.bg_tokens}")
    print(f"        manifolds={args.manifolds}")
    print(f"        device={os.environ.get('SAE_DEVICE', 'auto')}  force={args.force}\n")

    summary = dict(model=model_name, d_model=d_model, d_sae=d_sae, layers=layers,
                   seeds=args.seeds, bg_tokens=args.bg_tokens, runs=[])

    for layer in layers:
        tag = args.cache_tag if args.cache_tag is not None else f"{slug}_L{layer}"
        cdir = cache_dir_for(tag)
        sae_path = cdir / f"sae_{d_sae}_k{args.k}.pt"
        env = {**os.environ, "SAE_LAYER": str(layer), "SAE_CACHE_TAG": tag,
               "PYTHONUNBUFFERED": "1"}
        stages = build_stages(args, tag, layer, d_sae, sae_path, cdir)
        if args.stages:
            wanted = set(args.stages)
            stages = [s for s in stages if s[0] in wanted]

        print(f"=== layer {layer}  tag='{tag}'  cache={cdir} ===")
        run_rec = dict(layer=layer, tag=tag, stages=[])
        for name, argv, outputs in stages:
            done = all(_nonempty(Path(o)) for o in outputs)
            if done and not args.force:
                print(f"  [skip] {name}: output exists ({outputs[0]})")
                run_rec["stages"].append(dict(stage=name, status="skipped"))
                continue
            if args.dry_run:
                print(f"  [plan] {name}: {sys.executable} {' '.join(argv)}")
                run_rec["stages"].append(dict(stage=name, status="planned"))
                continue
            print(f"  [run ] {name}")
            rc, dt, logpath = run_stage(name, argv, env, cdir / "logs")
            status = "ok" if rc == 0 else f"FAILED(rc={rc})"
            print(f"  [{'done' if rc == 0 else 'FAIL'}] {name}  ({dt:.0f}s)  {status}")
            run_rec["stages"].append(dict(stage=name, status=status,
                                          seconds=round(dt, 1), log=str(logpath)))
            if rc != 0:
                print(f"  !! stage '{name}' failed — see {logpath}")
                if not args.keep_going:
                    summary["runs"].append(run_rec)
                    _write_summary(summary)
                    print("\nAborting (use --keep-going to continue past failures).")
                    sys.exit(1)
                break
        summary["runs"].append(run_rec)

    _write_summary(summary)
    print("\nAll requested work complete.")


def _write_summary(summary):
    CACHE_BASE.mkdir(parents=True, exist_ok=True)
    path = CACHE_BASE / "run_all_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {path}")


if __name__ == "__main__":
    main()
