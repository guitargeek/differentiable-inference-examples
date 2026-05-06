"""Benchmark RooFit minimization of an ONNX-driven HistFactory-style binned
likelihood, with the morph factor learned by a neural network.

Layout of one channel
---------------------

For each channel `c`, with `B` bins:

    nominal(b)   : RooHistFunc with the nominal per-bin yield y^nom_b.
    morph(b)     : HistFactory `ParamHistFunc` whose `B` bin parameters are
                   `B` distinct `RooONNXFunc` instances (all pointing at the
                   *same* ONNX file but each fed a different bin position and
                   the channel's per-channel nuisances). The NN output is the
                   morph factor r(b, theta) >= 0.
    binWidth(b)  : `RooBinWidthFunction` returning 1/binWidth at evaluation
                   time. In binned-likelihood mode it collapses to 1.0 and
                   triggers the `BinnedLikelihoodActiveYields` flag, so that
                   raw values are interpreted as yields by the codegen NLL.
    shape(b)     : RooProduct( nominal * morph * binWidth ).
    pdf_c        : RooRealSumPdf(name, "", [shape], [const1], extended=True).
                   The `BinnedLikelihood` attribute is set so the codegen
                   backend takes the per-bin Poisson shortcut.

Combined model
--------------
* All `pdf_c` go into one `RooSimultaneous` indexed by a `RooCategory chan`.
* Per-bin observed counts are Poisson-sampled at the truth and packed into a
  combined `RooDataHist` indexed by the same category.
* Constraints: one unit-Gaussian per nuisance (shared and per-channel),
  combined into a single `RooArgSet` and passed via `ExternalConstraints`.

Three classes of parameters are fit:
    mu               : POI, **shared** across channels.
    nu_1, ..., nu_K  : nuisances, **shared** across channels, Gaussian-constrained.
    alpha_{c,m}      : per-channel nuisances, **independent** per channel,
                       Gaussian-constrained.

So with `N` channels, `K` shared nuisances, `M` per-channel nuisances, the
number of free parameters is `1 + K + N*M`.

The same fit is performed with two backends:
  1. CPU backend (default `createNLL(data)`).
  2. Codegen + AD backend (`createNLL(data, EvalBackend::Codegen())`).

Wall time of `RooMinimizer::minimize("Minuit2")` is reported for each.
"""

import argparse
import time
from pathlib import Path

import numpy as np

import ROOT  # noqa: E402  -- assumes thisroot.sh has been sourced

HERE = Path(__file__).resolve().parent
ONNX_PATH = HERE / "surrogate_morph.onnx"
META_PATH = HERE / "surrogate_meta.npz"


def load_meta():
    if not ONNX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Run train_and_export.py first to produce surrogate_morph.onnx and "
            "surrogate_meta.npz"
        )
    return np.load(META_PATH)


# ---------------------------------------------------------------------------
# Truth-side helpers (numpy mirrors of train_and_export.py).
# ---------------------------------------------------------------------------
def bin_centers_x(meta, n_bins):
    edges = np.linspace(float(meta["x_min"]), float(meta["x_max"]), n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def bin_positions_norm(meta, n_bins):
    x_min = float(meta["x_min"]); x_max = float(meta["x_max"])
    return bin_centers_x(meta, n_bins) * (2.0 / (x_max - x_min))


def template_b_s(meta, n_bins):
    """Return per-bin (B_b, S_b) using the same parametric forms as the trainer."""
    xc = bin_centers_x(meta, n_bins)
    B = np.full(n_bins, float(meta["b_level"]))
    S = float(meta["s_amp"]) * np.exp(
        -0.5 * ((xc - float(meta["s_mu"])) / float(meta["s_sigma"])) ** 2
    )
    return B, S


def bump_values(bin_pos, centers, nu_amp, bump_sigma):
    if centers.size == 0:
        return np.zeros(bin_pos.shape + (0,))
    diff = bin_pos[..., None] - centers[None, :]
    return nu_amp * np.exp(-0.5 * (diff / bump_sigma) ** 2)


def truth_yields(meta, n_bins, mu, nu, alpha_c):
    """Per-bin truth yield for one channel, given the channel's alpha vector."""
    bp = bin_positions_norm(meta, n_bins)
    B, S = template_b_s(meta, n_bins)
    factor = np.ones(n_bins)
    K = int(meta["n_shared"]); M = int(meta["n_per_channel"])
    nu_amp = float(meta["nu_amp"]); bump_sigma = float(meta["bump_sigma"])
    if K > 0:
        h = bump_values(bp, np.asarray(meta["shared_centers"]), nu_amp, bump_sigma)
        factor = factor + (np.asarray(nu) * h).sum(axis=-1)
    if M > 0:
        g = bump_values(bp, np.asarray(meta["channel_centers"]), nu_amp, bump_sigma)
        factor = factor + (np.asarray(alpha_c) * g).sum(axis=-1)
    return (B + mu * S) * factor


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(meta, n_channels, n_bins, obs_scale, seed):
    """Build the simultaneous HistFactory-style model + binned data + constraints."""
    x_min = float(meta["x_min"]); x_max = float(meta["x_max"])
    mu_min = float(meta["mu_min"]); mu_max = float(meta["mu_max"])
    mu_true = float(meta["mu_true"])
    K = int(meta["n_shared"])
    M = int(meta["n_per_channel"])
    nu_range = float(meta["nu_range"])

    rng = np.random.default_rng(seed)

    # Observable with explicit binning -- ParamHistFunc reads its bin count
    # from the observable's binning.
    x = ROOT.RooRealVar("x", "x", 0.0, x_min, x_max)
    x.setBins(n_bins)

    # POI: shared across channels.
    mu = ROOT.RooRealVar("mu", "mu", 1.0, mu_min, mu_max)

    # Shared nuisances + their unit Gaussian constraints.
    nu_zero = ROOT.RooConstVar("zero_const", "0", 0.0)
    nu_one = ROOT.RooConstVar("one_const", "1", 1.0)
    keep = [nu_zero, nu_one]

    nuisances = []
    constraints = []
    for k in range(K):
        nu_k = ROOT.RooRealVar(f"nu_{k}", f"nu_{k}", 0.0, -nu_range, nu_range)
        con_k = ROOT.RooGaussian(
            f"con_nu_{k}", f"constraint nu_{k}", nu_k, nu_zero, nu_one
        )
        nuisances.append(nu_k)
        constraints.append(con_k)

    # Per-bin position constants (shared across channels).
    bin_pos = bin_positions_norm(meta, n_bins)
    bin_pos_consts = []
    for b in range(n_bins):
        c_b = ROOT.RooConstVar(f"binpos_{b}", f"binpos_{b}", float(bin_pos[b]))
        bin_pos_consts.append(c_b)
    keep.extend(bin_pos_consts)

    # Truth alphas: well separated across channels, deterministic given seed.
    if M > 0:
        alpha_truths = rng.uniform(-0.8, 0.8, size=(n_channels, M))
    else:
        alpha_truths = np.zeros((n_channels, 0))

    # Build channels.
    cat = ROOT.RooCategory("chan", "chan")
    pdfs_map = {}
    data_hists = {}      # name -> TH1 with observed counts
    alphas_per_chan = [] # list of list[RooRealVar]; len = n_channels

    # Nominal per-bin yields from the parametric truth at the reference point.
    nominal = truth_yields(
        meta, n_bins, float(meta["mu_ref"]),
        np.zeros(K), np.zeros(M),
    ) * float(obs_scale)

    for c in range(n_channels):
        # Per-channel alpha parameters.
        alphas_c = []
        for m in range(M):
            a_cm = ROOT.RooRealVar(
                f"alpha_{c}_{m}", f"alpha_{c}_{m}", 0.0, -nu_range, nu_range
            )
            alphas_c.append(a_cm)
            con_cm = ROOT.RooGaussian(
                f"con_alpha_{c}_{m}", f"constraint alpha_{c}_{m}",
                a_cm, nu_zero, nu_one,
            )
            constraints.append(con_cm)
        alphas_per_chan.append(alphas_c)

        # theta arg list per bin: (mu, nu_1..nu_K, alpha_{c,1}..alpha_{c,M}).
        theta_args = [mu, *nuisances, *alphas_c]

        # One RooONNXFunc per bin, all pointing at the same ONNX file.
        morph_funcs = []
        for b in range(n_bins):
            f_b = ROOT.RooONNXFunc(
                f"morph_c{c}_b{b}", "",
                [[bin_pos_consts[b]], theta_args],
                str(ONNX_PATH),
            )
            morph_funcs.append(f_b)

        # ParamHistFunc with the morph functions as bin parameters.
        # The class lives in the global namespace despite the
        # `RooStats/HistFactory/ParamHistFunc.h` header path.
        morph_phf = ROOT.ParamHistFunc(
            f"morph_phf_c{c}", "",
            ROOT.RooArgList(x),
            ROOT.RooArgList(*morph_funcs),
        )

        # Nominal RooHistFunc for the channel.
        nom_hist = ROOT.TH1D(f"nom_h_c{c}", "", n_bins, x_min, x_max)
        for b in range(n_bins):
            nom_hist.SetBinContent(b + 1, float(nominal[b]))
        nom_dh = ROOT.RooDataHist(
            f"nom_dh_c{c}", "", ROOT.RooArgList(x), nom_hist
        )
        nom_hf = ROOT.RooHistFunc(
            f"nom_hf_c{c}", "", ROOT.RooArgSet(x), nom_dh
        )

        # Bin-width function: in normal eval returns 1/binWidth (shape -> density);
        # in binned-likelihood mode collapses to 1.0 and triggers the
        # "BinnedLikelihoodActiveYields" flag so codegen interprets raw pdf
        # values as yields.
        binw = ROOT.RooBinWidthFunction(f"binw_c{c}", "", nom_hf, True)

        # shape = nominal * morph * binWidthInverse  (a density at eval time;
        # yield-per-bin in binned-likelihood mode).
        shape = ROOT.RooProduct(
            f"shape_c{c}", "",
            ROOT.RooArgList(nom_hf, morph_phf, binw),
        )

        # Per-channel extended PDF as a RooRealSumPdf with a single component.
        pdf_c = ROOT.RooRealSumPdf(
            f"pdf_c{c}", "",
            ROOT.RooArgList(shape),
            ROOT.RooArgList(nu_one),
            True,  # extended
        )
        pdf_c.setAttribute("BinnedLikelihood")

        # Build the per-channel observed data via Poisson-sampled bin counts.
        truth_c = truth_yields(
            meta, n_bins, mu_true,
            np.zeros(K), alpha_truths[c],
        ) * float(obs_scale)
        obs_c = rng.poisson(truth_c).astype(np.float64)
        obs_hist = ROOT.TH1D(f"obs_h_c{c}", "", n_bins, x_min, x_max)
        for b in range(n_bins):
            obs_hist.SetBinContent(b + 1, float(obs_c[b]))

        chan_name = f"chan_{c}"
        cat.defineType(chan_name, c)
        pdfs_map[chan_name] = pdf_c
        data_hists[chan_name] = obs_hist
        keep.extend([
            *morph_funcs, morph_phf, nom_hist, nom_dh, nom_hf,
            binw, shape, pdf_c, obs_hist, *alphas_c,
        ])

    # Combined simultaneous PDF and binned data.
    simul = ROOT.RooSimultaneous("simul", "simul", pdfs_map, cat)
    combined = ROOT.RooDataHist(
        "combined", "combined", ROOT.RooArgList(x),
        Index=cat, Import=data_hists,
    )

    constraint_set = ROOT.RooArgSet(*constraints) if constraints else ROOT.RooArgSet()
    keep.extend([*constraints, *nuisances])

    return {
        "x": x, "mu": mu, "nuisances": nuisances,
        "alphas_per_chan": alphas_per_chan, "alpha_truths": alpha_truths,
        "constraint_set": constraint_set, "cat": cat,
        "simul": simul, "combined": combined,
        "mu_true": mu_true, "K": K, "M": M, "keep": keep,
    }


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
def reset_params(mu, nuisances, alphas_per_chan):
    mu.setVal(1.0)
    for nu in nuisances:
        nu.setVal(0.0)
    for alphas_c in alphas_per_chan:
        for a in alphas_c:
            a.setVal(0.0)


def time_minimize(nll, mu, nuisances, alphas_per_chan):
    reset_params(mu, nuisances, alphas_per_chan)
    minim = ROOT.RooMinimizer(nll)
    minim.setErrorLevel(0.5)
    minim.setPrintLevel(-1)
    minim.setStrategy(0)
    t0 = time.perf_counter()
    status = minim.minimize("Minuit2")
    dt = time.perf_counter() - t0
    result = minim.save()
    ROOT.SetOwnership(result, True)
    return dt, status, result


def bench(label, nll, mu, nuisances, alphas_per_chan, n_repeats):
    print(f"\n=== {label} ===")
    dt0, status0, _ = time_minimize(nll, mu, nuisances, alphas_per_chan)
    print(f"  warm-up : status={status0}  wall = {dt0:.3f} s")
    times = []
    last_result = None
    for i in range(n_repeats):
        dt, status, res = time_minimize(nll, mu, nuisances, alphas_per_chan)
        times.append(dt)
        last_result = res
        print(f"  run {i + 1:2d}  : status={status}  wall = {dt:.3f} s")
    arr = np.array(times)
    print(
        f"  mean    : {arr.mean():.3f} s   "
        f"(std {arr.std():.3f}, min {arr.min():.3f})"
    )
    return arr.mean(), arr, last_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=2,
                        help="Number of simultaneous channels")
    parser.add_argument("--n-bins", type=int, default=8,
                        help="Bins per channel")
    parser.add_argument("--n-shared", type=int, default=None,
                        help="Required to match the trained surrogate. "
                             "If unset, takes the value baked into the metadata.")
    parser.add_argument("--n-per-channel", type=int, default=None,
                        help="Required to match the trained surrogate. "
                             "If unset, takes the value baked into the metadata.")
    parser.add_argument("--n-obs-scale", type=float, default=1.0,
                        help="Multiplicative scale on the truth yields. "
                             "Higher -> more events -> tighter likelihood.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Timed minimize() calls per backend (after warm-up)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    meta = load_meta()
    K_meta = int(meta["n_shared"]); M_meta = int(meta["n_per_channel"])
    if args.n_shared is not None and args.n_shared != K_meta:
        raise SystemExit(
            f"--n-shared={args.n_shared} does not match the trained surrogate "
            f"(K={K_meta}). Re-run train_and_export.py with the right K."
        )
    if args.n_per_channel is not None and args.n_per_channel != M_meta:
        raise SystemExit(
            f"--n-per-channel={args.n_per_channel} does not match the trained "
            f"surrogate (M={M_meta}). Re-run train_and_export.py with the right M."
        )

    n_pars = 1 + K_meta + args.channels * M_meta
    print(f"Channels             : {args.channels}")
    print(f"Bins / channel       : {args.n_bins}")
    print(f"Shared nuisances K   : {K_meta}")
    print(f"Per-channel M        : {M_meta}")
    print(f"Yield scale          : {args.n_obs_scale}")
    print(
        f"Free parameters      : {n_pars}  "
        f"(1 mu + {K_meta} nu_k + {args.channels}*{M_meta} alpha_cm; "
        f"{1 + K_meta} shared, {args.channels * M_meta} per-channel)"
    )

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    m = build_model(meta, args.channels, args.n_bins, args.n_obs_scale, args.seed)
    mu = m["mu"]; nuisances = m["nuisances"]
    alphas_per_chan = m["alphas_per_chan"]
    simul = m["simul"]; combined = m["combined"]
    cset = m["constraint_set"]

    print(f"True mu              : {m['mu_true']:.4f}")
    print("True alpha_{c,m}     : ")
    for c, alphas_c_truth in enumerate(m["alpha_truths"]):
        if M_meta > 0:
            print(f"  channel {c}: " + ", ".join(f"{a:+.3f}" for a in alphas_c_truth))
    print("True nu_k            : 0  (with N(0,1) constraints)")

    # Build NLLs for both backends.
    if cset.size() > 0:
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
        nll_default, mu, nuisances, alphas_per_chan, args.repeats,
    )
    mean_codegen, _, res_codegen = bench(
        "Codegen + AD backend (analytical gradient via Clad)",
        nll_codegen, mu, nuisances, alphas_per_chan, args.repeats,
    )

    print("\n=== Summary ===")
    print(
        f"  channels={args.channels}  bins={args.n_bins}  K={K_meta}  M={M_meta}  "
        f"free params={n_pars}  yield scale={args.n_obs_scale}"
    )
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
        for k in range(K_meta):
            parts.append(f"nu_{k} = {fmt(res, f'nu_{k}')}")
        for c in range(args.channels):
            for mm in range(M_meta):
                parts.append(f"alpha_{c}_{mm} = {fmt(res, f'alpha_{c}_{mm}')}")
        print(f"    {label}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
