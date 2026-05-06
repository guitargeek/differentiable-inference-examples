"""Benchmark RooFit minimization of an ONNX-surrogate likelihood, scaled
across an arbitrary number of channels with shared nuisance parameters.

Each channel is a `RooWrapperPdf` around a `RooONNXFunc` that computes the
learned likelihood ratio `r(x; theta) = exp(z(x; theta))` from the network
exported by `train_and_export.py`. All channels are combined into a single
`RooSimultaneous` likelihood. Three classes of parameters are fit:

  * `mu`           : POI, **shared** across channels.
  * `nu_1, ..., nu_K` : nuisance parameters, **shared** across channels and
                        each constrained by a unit Gaussian (the standard HEP
                        nuisance setup). They go through the surrogate net by
                        shifting the effective location by `0.1 * sum_k nu_k`.
  * `alpha_i`      : per-channel skewness, **independent** across channels.

So with `N` channels and `K` nuisances we fit `1 + K + N` free parameters.
Of those, `1 + K` are shared — perturbing any of them invalidates *every*
event's NN evaluation in the per-event cache that RooFit uses for finite
differences. This is the regime where AD analytically beats FD: AD costs
~1 forward + 1 backward per gradient step, FD costs ~`(2*(K+1) + 2*N + 1)`
forward passes (roughly).

The same fit is performed with two backends:

  1. CPU backend (default `createNLL(data)`).
     - Standard RooFit graph; gradients are numerical (finite differences).
  2. Codegen + AD backend (`createNLL(data, EvalBackend::Codegen())`).
     - The whole NLL is JIT-compiled into a single C++ function and Clad
       generates an analytical gradient using the custom-derivative pullback
       implemented for RooONNXFunc.

For both, we time `RooMinimizer::minimize("Minuit2")`. The first call is used
as a warm-up (it pays JIT / Clad compilation costs); subsequent calls are
timed and averaged.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.stats import skewnorm

import ROOT  # noqa: E402  -- assumes thisroot.sh has been sourced

HERE = Path(__file__).resolve().parent
ONNX_PATH = HERE / "surrogate_lhr.onnx"
META_PATH = HERE / "surrogate_meta.npz"


def load_meta():
    if not ONNX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Run train_and_export.py first to produce surrogate_lhr.onnx and "
            "surrogate_meta.npz"
        )
    return np.load(META_PATH)


def channel_alpha_truths(n_channels, alpha_min, alpha_max, seed=0):
    """Per-channel true alpha values, well-separated and safely inside the prior."""
    rng = np.random.default_rng(seed)
    margin = 0.25 * (alpha_max - alpha_min)
    lo = alpha_min + margin
    hi = alpha_max - margin
    if n_channels == 1:
        grid = np.array([0.5 * (lo + hi)])
    else:
        grid = np.linspace(lo, hi, n_channels)
    jitter = rng.uniform(-0.05, 0.05, size=n_channels) * (hi - lo)
    return (grid + jitter).astype(np.float64)


def build_model(meta, n_channels, n_obs_per_channel, seed):
    """Build a RooSimultaneous over `n_channels` channels with `K` shared
    nuisances, plus a `RooArgSet` of unit-Gaussian constraints to plug into
    `createNLL` via `ExternalConstraints(...)`."""
    x_min = float(meta["x_min"]); x_max = float(meta["x_max"])
    mu_min = float(meta["mu_min"]); mu_max = float(meta["mu_max"])
    alpha_min = float(meta["alpha_min"]); alpha_max = float(meta["alpha_max"])
    mu_true = float(meta["mu_true"])
    K = int(meta["n_nuisances"])
    nu_range = float(meta["nu_range"])

    x = ROOT.RooRealVar("x", "x", 0.0, x_min, x_max)
    mu = ROOT.RooRealVar("mu", "mu", 0.0, mu_min, mu_max)

    # K shared nuisance parameters, each with a unit-Gaussian constraint.
    nuisances = []
    constraints = []
    nu_zero = ROOT.RooConstVar("nu_zero", "nu_zero", 0.0)
    nu_one = ROOT.RooConstVar("nu_one", "nu_one", 1.0)
    keep_global = [nu_zero, nu_one]
    for k in range(K):
        nu_k = ROOT.RooRealVar(
            f"nu_{k}", f"nu_{k}", 0.0, -nu_range, nu_range
        )
        con_k = ROOT.RooGaussian(
            f"con_nu_{k}", f"constraint nu_{k}", nu_k, nu_zero, nu_one
        )
        nuisances.append(nu_k)
        constraints.append(con_k)
    constraint_set = ROOT.RooArgSet(*constraints) if constraints else ROOT.RooArgSet()

    rng = np.random.default_rng(seed)
    alpha_truths = channel_alpha_truths(n_channels, alpha_min, alpha_max, seed=seed + 1)

    cat = ROOT.RooCategory("chan", "chan")
    pdfs_map = {}
    data_map = {}
    keep = [*keep_global, *constraints, *nuisances]
    alphas = []

    for i, alpha_true_i in enumerate(alpha_truths):
        alpha_i = ROOT.RooRealVar(
            f"alpha_{i}", f"alpha_{i}", 0.0, alpha_min, alpha_max
        )
        # theta tensor passed to the ONNX surrogate, in the order it was trained:
        # (mu, alpha, nu_1, ..., nu_K).
        theta_args = [mu, alpha_i, *nuisances]
        onnx_func_i = ROOT.RooONNXFunc(
            f"lhr_func_{i}",
            "",
            [[x], theta_args],
            str(ONNX_PATH),
        )
        pdf_i = ROOT.RooWrapperPdf(
            f"lhr_pdf_{i}", f"lhr_pdf_{i}", onnx_func_i, True
        )

        # Sample observed data at the truth: per-channel alpha, true mu,
        # nuisances at their nominal value (zero).
        obs_i = skewnorm.rvs(
            a=float(alpha_true_i),
            loc=mu_true,
            scale=1.0,
            size=n_obs_per_channel,
            random_state=rng,
        ).astype(np.float64)
        obs_i = obs_i[(obs_i >= x_min) & (obs_i <= x_max)]
        ds_i = ROOT.RooDataSet.from_numpy(
            {"x": obs_i}, [x], name=f"data_{i}"
        )

        chan_name = f"chan_{i}"
        cat.defineType(chan_name, i)
        pdfs_map[chan_name] = pdf_i
        data_map[chan_name] = ds_i
        keep.extend([onnx_func_i, pdf_i, alpha_i, ds_i])
        alphas.append(alpha_i)

    simul = ROOT.RooSimultaneous("simul", "simul", pdfs_map, cat)
    combined = ROOT.RooDataSet(
        "combined", "combined", [x], Index=cat, Import=data_map
    )

    return {
        "x": x, "mu": mu, "alphas": alphas, "nuisances": nuisances,
        "constraint_set": constraint_set, "cat": cat,
        "simul": simul, "combined": combined,
        "alpha_truths": alpha_truths, "mu_true": mu_true,
        "K": K, "keep": keep,
    }


def reset_params(mu, alphas, nuisances):
    mu.setVal(0.0)
    for a in alphas:
        a.setVal(0.0)
    for nu in nuisances:
        nu.setVal(0.0)


def time_minimize(nll, mu, alphas, nuisances):
    reset_params(mu, alphas, nuisances)
    minim = ROOT.RooMinimizer(nll)
    minim.setErrorLevel(0.5)
    #minim.setPrintLevel(-1)
    minim.setPrintLevel(1)
    # Strategy 0: skip Minuit's numerical Hessian / extra checks. Without this,
    # Minuit2 falls back to numerical Hessians and the AD gradient cannot
    # actually save calls.
    minim.setStrategy(0)
    t0 = time.perf_counter()
    status = minim.minimize("Minuit2")
    dt = time.perf_counter() - t0
    result = minim.save()
    ROOT.SetOwnership(result, True)
    return dt, status, result


def bench(label, nll, mu, alphas, nuisances, n_repeats):
    print(f"\n=== {label} ===")
    dt0, status0, _ = time_minimize(nll, mu, alphas, nuisances)
    print(f"  warm-up : status={status0}  wall = {dt0:.3f} s")
    times = []
    last_result = None
    for i in range(n_repeats):
        dt, status, res = time_minimize(nll, mu, alphas, nuisances)
        times.append(dt)
        last_result = res
        print(f"  run {i + 1:2d}  : status={status}  wall = {dt:.3f} s")
    arr = np.array(times)
    print(f"  mean    : {arr.mean():.3f} s   (std {arr.std():.3f}, min {arr.min():.3f})")
    return arr.mean(), arr, last_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=4,
                        help="Number of simultaneous channels")
    parser.add_argument("--n-obs", type=int, default=2000,
                        help="Observed events per channel")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Timed minimize() calls per backend (after a warm-up)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    meta = load_meta()
    K = int(meta["n_nuisances"])
    n_pars = 1 + K + args.channels
    print(f"Channels         : {args.channels}")
    print(f"Shared nuisances : {K}")
    print(f"N obs / channel  : {args.n_obs}")
    print(f"Free parameters  : {n_pars}  "
          f"(1 mu + {K} nu_k + {args.channels} alpha_i; "
          f"{1 + K} shared, {args.channels} per-channel)")

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    m = build_model(meta, args.channels, args.n_obs, args.seed)
    mu = m["mu"]; alphas = m["alphas"]; nuisances = m["nuisances"]
    simul = m["simul"]; combined = m["combined"]
    cset = m["constraint_set"]

    print(f"True mu          : {m['mu_true']:.4f}")
    print("True alpha_i     : " + ", ".join(f"{a:.3f}" for a in m["alpha_truths"]))
    print("True nu_k        : 0  (with N(0,1) constraints)")

    if K > 0:
        nll_default = simul.createNLL(combined, ROOT.RooFit.ExternalConstraints(cset))
        nll_codegen = simul.createNLL(
            combined,
            ROOT.RooFit.EvalBackend.Codegen(),
            ROOT.RooFit.ExternalConstraints(cset),
        )
    else:
        nll_default = simul.createNLL(combined)
        nll_codegen = simul.createNLL(combined, ROOT.RooFit.EvalBackend.Codegen())
    ROOT.SetOwnership(nll_default, True)
    ROOT.SetOwnership(nll_codegen, True)

    mean_default, _, res_default = bench(
        "CPU backend (numerical gradients)",
        nll_default, mu, alphas, nuisances, args.repeats,
    )
    mean_codegen, _, res_codegen = bench(
        "Codegen + AD backend (analytical gradient via Clad)",
        nll_codegen, mu, alphas, nuisances, args.repeats,
    )

    print("\n=== Summary ===")
    print(f"  channels={args.channels}  K_nuisances={K}  "
          f"free params={n_pars}  obs/chan={args.n_obs}")
    print(f"  CPU       mean wall : {mean_default:.3f} s")
    print(f"  Codegen   mean wall : {mean_codegen:.3f} s")
    if mean_codegen > 0:
        print(f"  speed-up (CPU / Codegen) : {mean_default / mean_codegen:.2f}x")

    def fmt(res, name):
        v = res.floatParsFinal().find(name)
        return f"{v.getVal():+.4f} +/- {v.getError():.4f}" if v else "--"

    print("  fit results (last run):")
    for label, res in [("CPU      ", res_default), ("Codegen  ", res_codegen)]:
        parts = [f"mu = {fmt(res, 'mu')}"]
        for k in range(K):
            parts.append(f"nu_{k} = {fmt(res, f'nu_{k}')}")
        for i in range(args.channels):
            parts.append(f"alpha_{i} = {fmt(res, f'alpha_{i}')}")
        print(f"    {label}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
