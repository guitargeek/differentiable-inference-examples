"""Run a parameter scan of the HistFactory ONNX benchmark and plot timings.

The scan dispatches the actual training and benchmarking to subprocesses so
that the PyTorch and ROOT/SOFIE ONNX runtimes don't share a Python process
(same reason `train_and_export.py` and `benchmark.py` are split). For each
scan point:

  * If the scan variable is `n_shared`, `n_per_channel`, or `shared_frac`,
    retrain the surrogate (the NN architecture depends on K and M).
  * Run `benchmark.py` with the current parameters.
  * Parse the per-backend "mean : T s   (std S, min M)" lines.

Results are saved to a JSON file and rendered as a 2-panel ROOT plot:
top panel = wall time per `minimize()` for both backends (log scale, with
std error bars); bottom panel = speed-up ratio CPU / Codegen.

Examples
--------

Scan over the number of shared nuisances K (retrains each point):

    python scan.py --scan-var n_shared --values 0,2,4,8 \\
        --channels 8 --n-bins 16 --n-obs-scale 5.0 --repeats 3 \\
        --out-tag K

Scan over channels (no retraining):

    python scan.py --scan-var channels --values 1,2,4,8 \\
        --n-bins 16 --n-obs-scale 5.0 --repeats 3 \\
        --out-tag channels

Scan over the *shared fraction* of nuisances at fixed total nuisance count
T = K + N*M (retrains each point because K and M both change):

    python scan.py --scan-var shared_frac --values 0,4,8,12,16 \\
        --channels 8 --n-bins 16 --total-nuisances 16 \\
        --n-obs-scale 5.0 --repeats 3 --out-tag shared_frac

`--values` is the list of K values to visit; M is auto-computed as
(T - K) / N at each point. Each K must satisfy (T - K) % N == 0. The
x-axis on the plot is the shared fraction K / T, so f=0 is "all nuisances
per-channel" and f=1 is "all nuisances shared". This is the regime where
AD's advantage is clearest: every shared-parameter perturbation invalidates
the per-bin cache in *all* channels at once, while a per-channel
perturbation only touches one channel.

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
    "shared_frac": "Shared fraction K / (K + N #upoint M)",
}

# Scan variables that change the NN architecture and therefore force a
# retrain of the surrogate at every scan point.
RETRAIN_VARS = {"n_shared", "n_per_channel", "shared_frac"}


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


def run_bench(channels, n_bins, n_obs_scale, repeats, seed,
              random_start, shifted_start):
    start_label = (
        "random" if random_start
        else f"shifted={shifted_start}" if shifted_start is not None
        else "default"
    )
    print(
        f"\n>>> Benchmark (channels={channels}, n_bins={n_bins}, "
        f"scale={n_obs_scale}, repeats={repeats}, seed={seed}, "
        f"start={start_label}) ...",
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
    if random_start:
        cmd.append("--random-start")
    elif shifted_start is not None:
        cmd += ["--shifted-start", str(shifted_start)]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("[stderr]", res.stderr, file=sys.stderr)
    return parse_output(res.stdout)


def make_plot(summary, out_path, xscale="linear"):
    import array

    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)

    points = summary["points"]
    scan_var = summary["scan_var"]
    n = len(points)
    # For shared_frac the user-supplied "value" is K, but the natural x-axis
    # is the fraction K / total. Each point caches the fraction it visits.
    if scan_var == "shared_frac":
        xs_list = [float(p["shared_frac"]) for p in points]
    else:
        xs_list = [float(p["value"]) for p in points]
    xs = array.array("d", xs_list)
    cpu_y = array.array("d", [p["cpu_mean"] for p in points])
    cpu_e = array.array("d", [p["cpu_std"] for p in points])
    cg_y = array.array("d", [p["codegen_mean"] for p in points])
    cg_e = array.array("d", [p["codegen_std"] for p in points])
    speedup = array.array(
        "d", [c / g if g > 0 else 0.0 for c, g in zip(cpu_y, cg_y)]
    )
    zeros = array.array("d", [0.0] * n)

    canvas = ROOT.TCanvas("c_scan", "scan", 900, 720)
    split = 1.2 / (3.0 + 1.2)  # bottom-pad fraction
    pad_top = ROOT.TPad("pad_top", "", 0, split, 1, 1)
    pad_bot = ROOT.TPad("pad_bot", "", 0, 0, 1, split)
    pad_top.SetBottomMargin(0.02)
    pad_top.SetTopMargin(0.18)
    pad_top.SetLeftMargin(0.13)
    pad_top.SetRightMargin(0.04)
    pad_top.SetLogy()
    pad_top.SetGrid()
    pad_bot.SetTopMargin(0.04)
    pad_bot.SetBottomMargin(0.32)
    pad_bot.SetLeftMargin(0.13)
    pad_bot.SetRightMargin(0.04)
    pad_bot.SetGrid()
    if xscale == "log":
        pad_top.SetLogx()
        pad_bot.SetLogx()
    pad_top.Draw()
    pad_bot.Draw()

    # X-axis range, padded slightly so markers don't sit on the frame.
    if xscale == "log":
        xmin = min(xs_list) / 1.2
        xmax = max(xs_list) * 1.2
    else:
        span = max(xs_list) - min(xs_list)
        pad = 0.05 * span if span > 0 else max(1.0, 0.1 * abs(max(xs_list)))
        xmin = min(xs_list) - pad
        xmax = max(xs_list) + pad

    # ---- Top panel: CPU and Codegen wall times with error bars ----
    pad_top.cd()
    g_cpu = ROOT.TGraphErrors(n, xs, cpu_y, zeros, cpu_e)
    g_cg = ROOT.TGraphErrors(n, xs, cg_y, zeros, cg_e)
    g_cpu.SetMarkerStyle(20)
    g_cpu.SetMarkerSize(1.1)
    g_cpu.SetMarkerColor(ROOT.kAzure + 2)
    g_cpu.SetLineColor(ROOT.kAzure + 2)
    g_cpu.SetLineWidth(2)
    g_cg.SetMarkerStyle(21)
    g_cg.SetMarkerSize(1.1)
    g_cg.SetMarkerColor(ROOT.kOrange + 7)
    g_cg.SetLineColor(ROOT.kOrange + 7)
    g_cg.SetLineWidth(2)

    mg = ROOT.TMultiGraph()
    mg.Add(g_cpu, "LP")
    mg.Add(g_cg, "LP")
    mg.Draw("A")
    mg.GetXaxis().SetLimits(xmin, xmax)
    mg.GetXaxis().SetLabelSize(0)
    mg.GetXaxis().SetTickLength(0.03)
    mg.GetYaxis().SetTitle("Wall time per minimize() [s]")
    mg.GetYaxis().SetTitleOffset(1.2)
    mg.GetYaxis().SetTitleSize(0.045)
    mg.GetYaxis().SetLabelSize(0.04)

    leg = ROOT.TLegend(0.16, 0.62, 0.55, 0.78)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(g_cpu, "CPU (numerical gradients)", "lp")
    leg.AddEntry(g_cg, "Codegen + AD (Clad)", "lp")
    leg.Draw()

    fixed = summary["fixed_params"]
    if scan_var == "shared_frac":
        # K and M both vary; advertise the conserved total instead.
        fixed_str = (
            f"channels={fixed['channels']}, n_bins={fixed['n_bins']}, "
            f"K + N#upointM = {summary['total_nuisances']} (K, M = <scan>), "
            f"yield_scale={fixed['n_obs_scale']}, repeats={fixed['repeats']}"
        )
    else:
        fixed_str = (
            f"channels={fixed['channels']}, n_bins={fixed['n_bins']}, "
            f"K={fixed['n_shared']}, M={fixed['n_per_channel']}, "
            f"yield_scale={fixed['n_obs_scale']}, repeats={fixed['repeats']}"
        )
        title_var = {"n_shared": "K", "n_per_channel": "M"}.get(scan_var, scan_var)
        fixed_str = re.sub(
            rf"\b{re.escape(title_var)}=[\w.+-]+",
            f"{title_var}=<scan>",
            fixed_str,
        )
    title_top = ROOT.TLatex()
    title_top.SetNDC()
    title_top.SetTextAlign(22)
    title_top.SetTextSize(0.05)
    title_top.DrawLatex(
        0.54, 0.93,
        f"Binned fit ONNX benchmark - scan over {SCAN_LABEL[scan_var]}",
    )
    title_top.SetTextSize(0.035)
    title_top.DrawLatex(0.54, 0.86, fixed_str)

    pad_top.RedrawAxis()

    # ---- Bottom panel: speed-up ratio ----
    pad_bot.cd()
    g_speed = ROOT.TGraph(n, xs, speedup)
    g_speed.SetMarkerStyle(33)
    g_speed.SetMarkerSize(1.6)
    g_speed.SetMarkerColor(ROOT.kGreen + 2)
    g_speed.SetLineColor(ROOT.kGreen + 2)
    g_speed.SetLineWidth(2)
    g_speed.SetTitle("")
    g_speed.Draw("ALP")
    g_speed.GetXaxis().SetLimits(xmin, xmax)
    g_speed.GetXaxis().SetTitle(SCAN_LABEL[scan_var])
    g_speed.GetXaxis().SetTitleSize(0.12)
    g_speed.GetXaxis().SetTitleOffset(1.05)
    g_speed.GetXaxis().SetLabelSize(0.10)
    g_speed.GetYaxis().SetTitle("CPU / Codegen")
    g_speed.GetYaxis().SetTitleSize(0.10)
    g_speed.GetYaxis().SetTitleOffset(0.55)
    g_speed.GetYaxis().SetLabelSize(0.09)
    g_speed.GetYaxis().SetNdivisions(505)

    line = ROOT.TLine(xmin, 1.0, xmax, 1.0)
    line.SetLineColor(ROOT.kGray + 2)
    line.SetLineStyle(2)
    line.Draw()
    g_speed.Draw("LP SAME")

    pad_bot.RedrawAxis()

    canvas.SaveAs(str(out_path))
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
    parser.add_argument("--total-nuisances", type=int, default=None,
                        help="For --scan-var shared_frac: the conserved total "
                             "T = K + N*M of nuisance parameters across the scan. "
                             "Required for shared_frac, ignored otherwise.")
    parser.add_argument("--random-start", action="store_true",
                        help="Forward --random-start to benchmark.py: each run "
                             "starts from a different randomized point in "
                             "parameter space (shared between CPU and Codegen). "
                             "Default starts at the truth, which makes Minuit "
                             "converge in a handful of steps and underestimates "
                             "wall time.")
    parser.add_argument("--shifted-start", type=float, nargs="?",
                        const=1.0, default=None, metavar="KICK",
                        help="Forward --shifted-start to benchmark.py: a "
                             "deterministic kick of KICK pre-fit sigmas from "
                             "the prior mode (alternating signs by parameter "
                             "index). Identical at every scan point, so trends "
                             "across the scan stay clean. Mutually exclusive "
                             "with --random-start. Default kick: 1.0.")
    args = parser.parse_args()
    if args.random_start and args.shifted_start is not None:
        raise SystemExit("--random-start and --shifted-start are mutually exclusive.")

    typed = int if args.scan_var != "n_obs_scale" else float
    values = [typed(v.strip()) for v in args.values.split(",")]

    requires_retrain = (
        args.scan_var in RETRAIN_VARS and not args.skip_retrain
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
        "random_start": args.random_start,
        "shifted_start": args.shifted_start,
    }

    # For the shared-fraction scan, validate inputs and pre-resolve (K, M)
    # per point. Each value v is interpreted as K; M is fixed by the
    # conservation constraint K + N*M = T.
    resolved_per_point = {}  # value -> (K, M, fraction)
    if args.scan_var == "shared_frac":
        if args.total_nuisances is None:
            raise SystemExit(
                "--scan-var shared_frac requires --total-nuisances T."
            )
        T = args.total_nuisances
        N = args.channels
        for v in values:
            K = int(v)
            if K < 0 or K > T:
                raise SystemExit(
                    f"shared_frac value K={K} is outside [0, T={T}]."
                )
            rem = T - K
            if rem % N != 0:
                raise SystemExit(
                    f"shared_frac value K={K} incompatible with channels={N} "
                    f"and T={T}: (T - K)={rem} is not divisible by N. "
                    f"Pick K values congruent to T mod N "
                    f"(valid Ks: {list(range(T % N, T + 1, N))})."
                )
            M = rem // N
            resolved_per_point[v] = (K, M, K / T if T > 0 else 0.0)

    points = []
    for v in values:
        cfg = dict(fixed_params)
        if args.scan_var == "shared_frac":
            K, M, _ = resolved_per_point[v]
            cfg["n_shared"] = K
            cfg["n_per_channel"] = M
        else:
            cfg[args.scan_var] = v
        if requires_retrain:
            run_train(cfg["n_shared"], cfg["n_per_channel"], args.hidden)
        timings = run_bench(
            cfg["channels"], cfg["n_bins"], cfg["n_obs_scale"],
            cfg["repeats"], cfg["seed"],
            args.random_start, args.shifted_start,
        )
        point = {"value": v, "config": cfg, **timings}
        if args.scan_var == "shared_frac":
            K, M, frac = resolved_per_point[v]
            point["shared_frac"] = frac
            point["K"] = K
            point["M"] = M
        points.append(point)

        if args.scan_var == "shared_frac":
            K, M, frac = resolved_per_point[v]
            label = f"shared_frac={frac:.3f} (K={K}, M={M})"
        else:
            label = f"{args.scan_var}={v}"
        print(
            f"  -> {label}: "
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
    if args.scan_var == "shared_frac":
        summary["total_nuisances"] = args.total_nuisances
    out_json = HERE / f"scan_{args.out_tag}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved scan data -> {out_json}")

    out_plot = HERE / f"scan_{args.out_tag}.png"
    make_plot(summary, out_plot, xscale=args.xscale)


if __name__ == "__main__":
    main()
