"""Run a parameter scan of the HistFactory ONNX benchmark and plot timings.

The scan dispatches the actual training and benchmarking to subprocesses so
that the PyTorch and ROOT/SOFIE ONNX runtimes don't share a Python process
(same reason `train_and_export.py` and `benchmark.py` are split). For each
scan point:

  * If the scan variable is `n_shared` or `n_per_channel`, retrain the
    surrogate (the NN architecture depends on K and M).
  * Run `benchmark.py` with the current parameters.
  * Parse the per-backend "mean : T s   (std S, min M)" lines.

Results are saved to a JSON file and rendered as a 2-panel matplotlib plot:
top panel = wall time per `minimize()` for both backends (log scale, with
std error bars); bottom panel = speed-up ratio CPU / Codegen.

Examples
--------

Scan over the number of shared nuisances K (retrains each point):

    python scan.py --scan-var n_shared --values 0,2,4,8 \\
        --channels 4 --n-bins 16 --n-obs-scale 5.0 --repeats 3 \\
        --out-tag K

Scan over channels (no retraining):

    python scan.py --scan-var channels --values 1,2,4,8 \\
        --n-bins 16 --n-obs-scale 5.0 --repeats 3 \\
        --out-tag channels

The two output files for tag `<tag>` are `scan_<tag>.json` and
`scan_<tag>.png`, written next to this script.

Note
----

Both `train_and_export.py` and `benchmark.py` are launched as subprocesses
using `sys.executable`, so this script must be invoked with a Python that has
`torch` (for training) and a ROOT-aware Python (with `thisroot.sh` sourced)
for the benchmark. In practice that means: source `thisroot.sh` before
running `scan.py`.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Match a "mean : <T> s   (std <S>, min <M>)" line, anchored to the section
# header above it so we can pick the CPU vs Codegen block deterministically.
RE_BLOCK_MEAN = re.compile(
    r"^=== (?P<header>CPU backend|Codegen \+ AD backend).*?$\n"
    r"(?:.*\n)*?\s*mean\s*:\s*(?P<mean>[\d.eE+-]+)\s+s\s+\(std\s+(?P<std>[\d.eE+-]+)",
    re.MULTILINE,
)

# Pretty axis labels per scan variable.
SCAN_LABEL = {
    "channels": "Number of channels N",
    "n_bins": "Bins per channel B",
    "n_shared": "Shared nuisances K",
    "n_per_channel": "Per-channel nuisances M",
    "n_obs_scale": "Yield scale",
}


def parse_output(stdout):
    """Extract CPU and Codegen mean/std seconds from a benchmark stdout."""
    found = {}
    for m in RE_BLOCK_MEAN.finditer(stdout):
        key = "cpu" if m.group("header").startswith("CPU") else "codegen"
        found[key] = (float(m.group("mean")), float(m.group("std")))
    if "cpu" not in found or "codegen" not in found:
        raise RuntimeError(
            f"Could not parse CPU and Codegen timings from benchmark output:\n{stdout}"
        )
    cpu_mean, cpu_std = found["cpu"]
    cg_mean, cg_std = found["codegen"]
    return {
        "cpu_mean": cpu_mean, "cpu_std": cpu_std,
        "codegen_mean": cg_mean, "codegen_std": cg_std,
    }


def run_train(K, M, hidden):
    print(f"\n>>> Training surrogate (K={K}, M={M}, hidden={hidden}) ...", flush=True)
    cmd = [
        sys.executable, str(HERE / "train_and_export.py"),
        "--n-shared", str(K),
        "--n-per-channel", str(M),
        "--hidden", str(hidden),
    ]
    # Stream training output directly; it is long-running and useful to see.
    subprocess.run(cmd, check=True)


def run_bench(channels, n_bins, n_obs_scale, repeats, seed):
    print(
        f"\n>>> Benchmark (channels={channels}, n_bins={n_bins}, "
        f"scale={n_obs_scale}, repeats={repeats}, seed={seed}) ...",
        flush=True,
    )
    cmd = [
        sys.executable, str(HERE / "benchmark.py"),
        "--channels", str(channels),
        "--n-bins", str(n_bins),
        "--n-obs-scale", str(n_obs_scale),
        "--repeats", str(repeats),
        "--seed", str(seed),
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("[stderr]", res.stderr, file=sys.stderr)
    return parse_output(res.stdout)


def make_plot(summary, out_path, xscale="linear"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = summary["points"]
    scan_var = summary["scan_var"]
    xs = [p["value"] for p in points]
    cpu_y = [p["cpu_mean"] for p in points]
    cpu_e = [p["cpu_std"] for p in points]
    cg_y = [p["codegen_mean"] for p in points]
    cg_e = [p["codegen_std"] for p in points]
    speedup = [c / g if g > 0 else float("nan") for c, g in zip(cpu_y, cg_y)]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.5, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2]},
    )
    ax_top.errorbar(
        xs, cpu_y, yerr=cpu_e, marker="o", linestyle="-",
        label="CPU (numerical gradients)", capsize=3, color="tab:blue",
    )
    ax_top.errorbar(
        xs, cg_y, yerr=cg_e, marker="s", linestyle="-",
        label="Codegen + AD (Clad)", capsize=3, color="tab:orange",
    )
    ax_top.set_ylabel("Wall time per minimize() [s]")
    ax_top.set_yscale("log")
    if xscale == "log":
        ax_top.set_xscale("log")
    ax_top.legend(loc="best")
    ax_top.grid(True, which="both", alpha=0.3)

    fixed = summary["fixed_params"]
    fixed_str = (
        f"channels={fixed['channels']}, n_bins={fixed['n_bins']}, "
        f"K={fixed['n_shared']}, M={fixed['n_per_channel']}, "
        f"yield_scale={fixed['n_obs_scale']}, repeats={fixed['repeats']}"
    )
    # Replace the scanned variable's value with "<scan>" so the title is honest.
    title_var = {"n_shared": "K", "n_per_channel": "M"}.get(scan_var, scan_var)
    fixed_str = re.sub(
        rf"\b{re.escape(title_var)}=[\w.+-]+",
        f"{title_var}=<scan>",
        fixed_str,
    )
    ax_top.set_title(
        f"HistFactory ONNX benchmark — scan over {SCAN_LABEL[scan_var]}\n{fixed_str}",
        fontsize=9,
    )

    ax_bot.plot(xs, speedup, marker="d", linestyle="-", color="tab:green")
    ax_bot.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
    ax_bot.set_xlabel(SCAN_LABEL[scan_var])
    ax_bot.set_ylabel("CPU / Codegen")
    if xscale == "log":
        ax_bot.set_xscale("log")
    ax_bot.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-var", required=True, choices=list(SCAN_LABEL))
    parser.add_argument("--values", required=True,
                        help="Comma-separated scan values")
    # Fixed parameters (overridden for the scan dimension).
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--n-bins", type=int, default=16)
    parser.add_argument("--n-shared", type=int, default=4)
    parser.add_argument("--n-per-channel", type=int, default=1)
    parser.add_argument("--n-obs-scale", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=32,
                        help="MLP hidden width (used when retraining)")
    parser.add_argument("--skip-retrain", action="store_true",
                        help="Skip retraining even when scanning n_shared / "
                             "n_per_channel (assumes the surrogate on disk "
                             "already matches every scan point).")
    parser.add_argument("--out-tag", default="scan",
                        help="Tag used in output filenames: scan_<tag>.json, "
                             "scan_<tag>.png")
    parser.add_argument("--xscale", default="linear", choices=["linear", "log"],
                        help="X-axis scale on the plot.")
    args = parser.parse_args()

    typed = int if args.scan_var != "n_obs_scale" else float
    values = [typed(v.strip()) for v in args.values.split(",")]

    requires_retrain = (
        args.scan_var in ("n_shared", "n_per_channel") and not args.skip_retrain
    )

    fixed_params = {
        "channels": args.channels,
        "n_bins": args.n_bins,
        "n_shared": args.n_shared,
        "n_per_channel": args.n_per_channel,
        "n_obs_scale": args.n_obs_scale,
        "repeats": args.repeats,
        "seed": args.seed,
        "hidden": args.hidden,
    }

    points = []
    for v in values:
        cfg = dict(fixed_params)
        cfg[args.scan_var] = v
        if requires_retrain:
            run_train(cfg["n_shared"], cfg["n_per_channel"], args.hidden)
        timings = run_bench(
            cfg["channels"], cfg["n_bins"], cfg["n_obs_scale"],
            cfg["repeats"], cfg["seed"],
        )
        points.append({"value": v, "config": cfg, **timings})
        print(
            f"  -> {args.scan_var}={v}: "
            f"CPU={timings['cpu_mean']:.3f}+/-{timings['cpu_std']:.3f} s, "
            f"Codegen={timings['codegen_mean']:.3f}+/-{timings['codegen_std']:.3f} s, "
            f"speedup={timings['cpu_mean']/timings['codegen_mean']:.2f}x"
        )

    summary = {
        "scan_var": args.scan_var,
        "values": values,
        "fixed_params": fixed_params,
        "points": points,
    }
    out_json = HERE / f"scan_{args.out_tag}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved scan data -> {out_json}")

    out_plot = HERE / f"scan_{args.out_tag}.png"
    make_plot(summary, out_plot, xscale=args.xscale)


if __name__ == "__main__":
    main()
