# ONNX HistFactory minimization benchmark

A binned, multi-channel HistFactory-style benchmark that mirrors
`onnx_sbi_benchmark/` but for the **interpolation/morphing** step typical of
HistFactory templates. The morphing function is replaced by a single
neural-network surrogate, instantiated once per channel as a `RooONNXFunc`
that takes the (rescaled) observable directly as its bin-position input.
Two NLL builds are compared:

1. **CPU backend** (default `createNLL(data)`): standard RooFit graph, Minuit2
   gets gradients by finite differences; the binned-likelihood optimisation
   skips the per-bin integral.
2. **Codegen + AD backend** (`createNLL(data, EvalBackend::Codegen())`): the
   whole NLL is JIT-compiled and Clad produces an analytical gradient using
   the custom-derivative pullback that ships with `RooONNXFunc`.

## Truth model

For channel `c` and bin `b`:

```
y_{c,b}(theta) = (B_b + mu * S_b) * (1 + sum_k nu_k h_k(b)
                                       + sum_m alpha_{c,m} g_m(b))
```

* `B_b` flat background, `S_b` Gaussian signal at the bin centre.
* `h_k`, `g_m` localised Gaussian bumps (shared / per-channel).
* `mu` POI; `nu_k` shared nuisances; `alpha_{c,m}` per-channel nuisances. Each
  nuisance carries a unit Gaussian constraint, plugged into the NLL via
  `ExternalConstraints`.

Truth data: per-bin observed counts are sampled from `Poisson(y_{c,b}^truth)`
with all nuisances at zero and `mu = mu_true` (`MU_TRUE` in
`train_and_export.py`).

## NN structure

* One scalar-output ONNX network shared across all (channel, bin) pairs.
* Inputs: `bin_pos` (one number, normalised to `[-1, 1]`) and `theta = (mu,
  nu_1..nu_K, alpha_1..alpha_M)`.
* Trained as a regression on `log r(b, theta) = log(y(b, theta) / y^nom(b))`
  where the nominal point is `mu = mu_ref = 1`, `nu = 0`, `alpha = 0`. The
  ONNX wrapper exports `r = exp(log_r)` so the morph factor is strictly
  positive.

## Likelihood

Per channel:

```
shape_c = nominal_HistFunc(y^nom)  *  RooONNXFunc(x_norm, theta_c)  *  RooBinWidthFunction
pdf_c   = RooRealSumPdf(shape_c, const_one, extended=True)
pdf_c.setAttribute("BinnedLikelihood")
```

* A single `RooONNXFunc` per channel takes the observable itself (rescaled
  via a `RooFormulaVar` to the network's `[-1, 1]` input range) as the
  bin-position input. In binned-likelihood mode the NLL evaluates `shape`
  once per bin with `x` set to that bin's centre, so this single ONNX call
  produces the per-bin morph factor without an explicit `ParamHistFunc`
  indirection or `N_bins` constant copies of the network.
* `RooBinWidthFunction` divides by the bin width during plain evaluation so
  the product behaves as a density. In binned-likelihood mode it collapses to
  `1.0` and triggers `BinnedLikelihoodActiveYields`, telling the codegen NLL
  to interpret raw values as yields directly.

Combined: `RooSimultaneous` over channels; the constraint product is built
externally and passed via `RooFit::ExternalConstraints`.

## Configurable knobs

`train_and_export.py`:
* `--n-shared K`         number of shared bump nuisances `h_k`.
* `--n-per-channel M`    number of per-channel bump nuisances `g_m`.
* `--hidden`             MLP hidden width.

`benchmark.py`:
* `--channels N`         number of simultaneous channels.
* `--n-bins B`           bins per channel.
* `--n-obs-scale`        multiplicative scale on truth yields (controls
                         per-channel statistics).
* `--n-shared`, `--n-per-channel` are read from the metadata; passing them
  on the command line just sanity-checks the trained surrogate matches.

So the free-parameter count is `1 + K + N*M` — `1 + K` shared, `N*M`
per-channel. Increasing `K` (shared) is the regime where AD wins, since
shared-parameter perturbations invalidate every channel's per-bin cache.

## Layout

```
onnx_histfactory_benchmark/
├── README.md
├── train_and_export.py    # PyTorch: regress log-morph, export ONNX
├── benchmark.py            # RooFit: simultaneous binned fit, time both backends
└── scan.py                 # Drives parameter scans + ROOT plots
```

`train_and_export.py` writes:
* `surrogate_morph.onnx` — ONNX graph with two input tensors (`bin_pos`
  shape `[1, 1]`, `theta` shape `[1, 1+K+M]`) and a scalar output `r`.
* `surrogate_meta.npz` — bin templates, bump centres, prior bounds, truth
  parameters.

The two scripts are split for the same reason as the SBI benchmark: the ONNX
runtimes that PyTorch and ROOT (TMVA SOFIE) use clash if loaded into the same
process.

## Running

`train_and_export.py` needs Python with `torch`, `numpy`. It does not need
ROOT:

```bash
python train_and_export.py [--n-shared 2] [--n-per-channel 1] [--hidden 32]
```

`benchmark.py` needs ROOT (with `tmva-sofie=ON` and `roofit_clad=ON`). Source
`thisroot.sh` first:

```bash
. <root_build>/bin/thisroot.sh
python benchmark.py [--channels N] [--n-bins B] [--n-obs-scale 1.0] [--repeats 3]
```

## Scans + plots

`scan.py` drives parameter sweeps and produces a 2-panel ROOT plot
(top: wall time per `minimize()` for both backends, log scale, with std error
bars; bottom: CPU/Codegen speed-up). Output goes to `scan_<tag>.json` and
`scan_<tag>.png` next to the script.

Scan over the number of shared nuisances K (retrains the surrogate at each
point, since the NN architecture depends on K):

```bash
. <root_build>/bin/thisroot.sh
python scan.py --scan-var n_shared --values 0,2,4,8 \
    --channels 8 --n-bins 16 --n-per-channel 1 --n-obs-scale 5.0 \
    --repeats 3 --out-tag K
```

Scan over channels (no retraining needed):

```bash
python scan.py --scan-var channels --values 1,2,4,8 \
    --n-bins 16 --n-obs-scale 5.0 --repeats 3 --out-tag channels
```

Scan over the *shared fraction* of nuisances at fixed total nuisance count
`T = K + N*M`. K is the comma-separated list of values to visit; M is
auto-computed at each point as `(T - K) / N`. Each chosen K must satisfy
`(T - K) % N == 0`. The plot's x-axis is the fraction `K / T` ∈ [0, 1]:

```bash
python scan.py --scan-var shared_frac --values 0,4,8,12,16 \
    --channels 8 --n-bins 16 --total-nuisances 16 \
    --n-obs-scale 5.0 --repeats 3 --out-tag shared_frac
```

This is the regime that most clearly motivates AD: every shared-parameter
perturbation invalidates the per-bin cache in *all* channels at once, while
a per-channel perturbation only touches one channel's cache. So the
CPU/Codegen ratio should grow with the shared fraction even though the
total parameter count stays put.

The other scan variables are `n_bins`, `n_per_channel`, and `n_obs_scale`.
Pass `--skip-retrain` if you've already trained a surrogate that matches the
scan points and want to reuse it.

## Notes

* The benchmark uses `Minuit2` with `setStrategy(0)`. The default strategy 1
  computes a numerical Hessian as part of MIGRAD, re-introducing FD and
  largely defeating the AD gradient.
* The first `minimize()` call after `createNLL(..., Codegen())` pays a one-off
  Clad gradient-generation cost; the benchmark always runs a warm-up call
  before the timed repeats.
* Each `RooONNXFunc` instance creates its own SOFIE session. With one
  instance per channel the construction step grows linearly in `N_channels`
  (independent of `N_bins`).
