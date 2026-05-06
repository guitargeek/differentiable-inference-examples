# ONNX SBI minimization benchmark

This benchmark measures the runtime of a RooFit minimization where the
likelihood is a parametrised neural-network surrogate for the likelihood
ratio of an analytical signal hypothesis function. It compares two ways
of building the NLL with `RooONNXFunc`:

1. **CPU backend** (default `createNLL(data)`): the standard RooFit graph,
   evaluated by the CPU backend. Minuit2 obtains gradients by finite
   differences. RooFit's per-event caching is active.
2. **Codegen + AD backend** (`createNLL(data, EvalBackend::Codegen())`):
   the whole NLL is JIT-compiled into a single C++ function and Clad
   generates an analytical gradient using the custom-derivative pullback
   that ships with `RooONNXFunc`.

Both backends use exactly the same `RooONNXFunc`, `RooWrapperPdf`,
`RooSimultaneous`, `RooDataSet`, and Minuit2 settings. The only difference
is the `EvalBackend`.

## Signal hypothesis

The "true" model is the skew-normal distribution

```
f(x; mu, alpha) = 2 * phi(x - mu) * Phi(alpha * (x - mu))
```

with two parameters: location `mu` and skewness `alpha`. With non-zero
`alpha` the likelihood landscape is asymmetric and the minimization is
non-trivial.

On top of that, an arbitrary number of **Gaussian-bump nuisance parameters**
modulate the shape:

```
p(x; mu, alpha, nu) = skewnorm(x; mu, alpha) * (1 + sum_k nu_k * b_k(x))
```

where each `b_k(x)` is a Gaussian bump centred at a different x value.
Each `nu_k` therefore has a localised, distinguishable effect on the
distribution — mimicking the way HEP nuisance parameters describe
independent experimental systematics. In the fit each `nu_k` carries a
unit-Gaussian constraint via `RooFit::ExternalConstraints`.

## Multi-channel scaling study

`benchmark.py` builds a `RooSimultaneous` over a configurable number of
channels with three classes of parameters:

* `mu`            : POI, **shared** across channels.
* `nu_1, ..., nu_K` : nuisances, **shared** across channels and
                      Gaussian-constrained.
* `alpha_i`       : per-channel skewness, **independent** across channels.

So with `N` channels and `K` nuisances we fit `1 + K + N` free
parameters; `1 + K` of them are shared. Perturbing any shared parameter
invalidates *every* event's NN evaluation in the per-event cache that
RooFit uses for finite differences. This is the regime where AD wins:

```
T_cpu  ~  ((1 + 2*K + 2)*N + 2*N) * n_obs * forward_cost
T_ad   ~  N * n_obs * (forward + backward)_cost
```

so AD's advantage scales with `K` (number of shared parameters), since
that controls how many NLL re-evaluations FD has to do per gradient step.

This setup mirrors a realistic HEP fit: a shared POI, many shared
nuisance parameters, and a small per-channel parameter. (In a real
analysis each channel would have its own neural-network template; we
re-use one network for simplicity.)

## Layout

```
onnx_sbi_benchmark/
├── README.md
├── train_and_export.py    # PyTorch: train classifier, export ONNX, save metadata
└── benchmark.py            # RooFit: simultaneous fit, time each backend
```

`train_and_export.py` writes:

* `surrogate_lhr.onnx` — ONNX graph with two input tensors
  (`obs` shape `[1,1]`, `theta` shape `[1, 2 + K]`) and a scalar output
  giving `r(x; theta) = exp(z(x; theta))`.
* `surrogate_meta.npz` — prior bounds, `n_nuisances`, default truths.

The two scripts are split because the ONNX runtimes that PyTorch and
ROOT (TMVA SOFIE) use can clash if loaded into the same process.

## Running

`train_and_export.py` needs Python with `torch`, `numpy`, and `scipy`.
It does **not** need ROOT:

```bash
python train_and_export.py [--hidden 32] [--n-nuisances 4]
```

`--n-nuisances K` controls how many bump-nuisance inputs the surrogate
takes; `K=0` reproduces the no-nuisance setup. `--hidden` controls the
MLP width.

`benchmark.py` needs ROOT (with `tmva-sofie=ON` and `roofit_clad=ON`).
Source `thisroot.sh` first:

```bash
. <root_build>/bin/thisroot.sh
python benchmark.py [--channels N] [--n-obs M] [--repeats K]
```

A typical scaling scan over channels:

```bash
for N in 1 2 4 8 16 32; do
  python benchmark.py --channels $N --n-obs 2000 --repeats 3 \
      | grep -E "channels=|mean wall|speed-up"
done
```

Vary `--n-nuisances` in the training step and re-run the scan to map
out how the AD advantage grows with the number of shared parameters.

## What it prints

For each backend the script reports the wall time of one warm-up
`minimize("Minuit2")` plus N timed repeats, the per-backend mean and
standard deviation, and a side-by-side comparison of the final fit
parameters. Both backends are expected to converge to the same point —
what differs is the cost.

## Indicative numbers

On an Opus laptop, `--n-obs 2000`, default 32-32-1 MLP:

| K (shared nuisances) | N (channels) | free params | CPU | Codegen+AD | speed-up |
|---|---|---|---|---|---|
| 0 | 4 | 5 | 1.08 s | 1.26 s | 0.86× |
| 4 | 4 | 9 | 5.51 s | 3.18 s | **1.73×** |
| 8 | 4 | 13 | 5.46 s | 3.47 s | **1.57×** |
| 8 | 8 | 17 | 8.43 s | 5.88 s | **1.44×** |
| 16 | 8 | 25 | 80.19 s | 65.99 s | **1.22×** |

`K = 0` is the regime where caching saves FD on the per-channel
parameters. Adding shared nuisances flips the balance because every
shared-parameter perturbation invalidates the cache for every event.

## Notes

* The benchmark runs Minuit2 with `setStrategy(0)`. The default
  strategy (1) computes a numerical Hessian as part of MIGRAD, which
  re-introduces finite differences and largely defeats the analytical
  gradient. Strategy 0 trusts the AD gradient and is what you want
  whenever you actually care about the AD speed-up.
* The first `minimize()` call after `createNLL(..., Codegen())` pays a
  one-off Clad gradient-generation cost. The benchmark always runs a
  warm-up call before the timed repeats.
* The fit's converged minimum may not be exactly at the truth: SBI
  surrogates accumulate small per-event LR biases, and a thousands-of-
  events likelihood amplifies them. Both backends converge to the
  *same* (biased) minimum, which is what the timing comparison cares
  about. Train a wider network (`--hidden 64` or `128`) to reduce the
  bias if needed.
