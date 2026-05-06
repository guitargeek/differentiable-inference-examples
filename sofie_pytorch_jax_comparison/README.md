# SOFIE + Clad vs PyTorch vs JAX

Micro-benchmark comparing forward and input-gradient throughput of three CPU
inference paths on the same fully-connected MLP (~70k parameters):

* **PyTorch** — eager autograd, with `torch.jit.script` + `torch.jit.freeze`,
  single-threaded.
* **JAX** — `jax.jit` for the forward pass, `jax.jit(grad(...))` for the
  input gradient.
* **SOFIE + Clad** — the model is exported to ONNX, regenerated as C++ via
  `TMVA::Experimental::SOFIE`, and the input gradient is produced by Clad
  (`clad::gradient(...)`) at ROOT-interpreter time.

## What this is (and isn't) for

The point is not "framework X beats framework Y" in absolute terms.

The motivation for the SOFIE+Clad path inside ROOT is that it has **no
external dependencies outside ROOT** (no PyTorch, no ONNX Runtime, no JAX
runtime to ship around) and **works from both C++ and Python**, which makes
it well-suited to combined-fit frameworks built around RooFit. The
analytical gradient produced by Clad falls out of the same toolchain.

PyTorch and JAX are included as reference points, not as competitors that
SOFIE+Clad has to beat: RooFit already exposes a batched-evaluation
interface, so callbacks into batched JAX or PyTorch are perfectly viable
in principle — and useful when the same NN gets evaluated at many
parameter points within a single NLL evaluation, where the per-call Python
overhead amortizes away.

The two regimes shown on the plot are:

* **single sample** (`batch_size = 1`) — closer to per-event evaluation
  inside RooFit / a likelihood scan with one NN call per event.
* **batched** (`batch_size = 1000`) — closer to bulk training/eval, and to
  the multi-point-per-NLL pattern above.

## Model

5 hidden layers × 128 units, ReLU, scalar output, input dimension 20. See
`benchmark_torch.py` and `benchmark_jax.py` for the exact definition (they
must match because the SOFIE benchmark consumes the ONNX exported by the
PyTorch script).

## Layout

```
sofie_pytorch_jax_comparison/
├── README.md
├── benchmark_torch.py        # PyTorch benchmark + ONNX export -> results_torch.json
├── benchmark_jax.py          # JAX benchmark                   -> results_jax.json
├── onnx_to_cpp.C             # SOFIE: model.onnx -> model.hxx
├── benchmark_sofie_clad.C    # SOFIE+Clad benchmark            -> results_sofie_clad.json
├── plot.py                   # reads the three JSON files, writes plot.png
└── run_all.sh                # drives the whole pipeline
```

## Data flow

Each benchmark writes a JSON file with the same shape so the plotting
script doesn't need any framework-specific knowledge:

```json
{
  "entries": [
    {"label": "PyTorch",           "forward_us": 105.4, "grad_us": 407.5},
    {"label": "PyTorch (batched)", "forward_us": 3.2,   "grad_us": 7.7}
  ]
}
```

`plot.py` concatenates `entries` from each input file in the order they are
listed on the command line — so the bar order is set by the order of files
passed to `--inputs`. Adding or removing a framework is a matter of
producing one more JSON file and listing it.

## Running

You need:

* Python with `torch`, `jax`, `numpy`. The plotting script uses ROOT (PyROOT).
* ROOT built with `tmva-sofie=ON` and `roofit_clad=ON`, sourced via
  `thisroot.sh`.

End-to-end:

```bash
. <root_build>/bin/thisroot.sh
./run_all.sh
```

That runs, in order:

1. `python3 benchmark_torch.py --out results_torch.json --onnx model.onnx`
   — trains nothing, just initialises the MLP, benchmarks single + batched,
   and exports `model.onnx`.
2. `root -l -b -q onnx_to_cpp.C` — SOFIE parses `model.onnx` and writes
   `model.hxx`.
3. `root -l -b -q 'benchmark_sofie_clad.C("results_sofie_clad.json")'` —
   includes `model.hxx`, runs Clad on the inference function, sanity-checks
   the gradient against finite differences, then times the forward and
   gradient calls.
4. `python3 benchmark_jax.py --out results_jax.json` — JAX benchmark on a
   freshly-initialised MLP with the same shape.
5. `python3 plot.py --inputs results_torch.json results_jax.json results_sofie_clad.json --out plot.png`.

Each step can also be run in isolation: every script accepts an `--out`
flag (or, for the C++ benchmark, a positional argument), and `plot.py`
takes any list of JSON files via `--inputs`.

## Notes

* The PyTorch and SOFIE benchmarks run on the same numerical model
  (PyTorch initialises it, exports to ONNX, SOFIE reads that ONNX). The JAX
  benchmark uses an independently-initialised MLP of the same shape — the
  weights are different, but the FLOP count is identical and all three
  benchmarks feed an all-ones input, so the active path through ReLU is the
  same.
* The PyTorch and JAX scripts run two passes per invocation: one with
  `batch_size = 1` and `n_runs = 1000` (single-sample regime), one with
  `batch_size = 1000` and `n_runs = 100` (batched regime). The "per sample"
  number is the total wall-time of one timed loop divided by
  `batch_size * n_runs`.
* Each of the three benchmarks repeats the timed loop `--repeats` times
  (default 5 — the C++ benchmark takes this as the second positional arg)
  and reports the **median**. Each result JSON also stores the full
  per-repeat array under `forward_us_repeats` / `grad_us_repeats` so error
  bars or distributions can be added later without re-running.
* All three benchmarks are pinned to a single CPU thread:
  `torch.set_num_threads(1)` for PyTorch, `XLA_FLAGS` +
  `OMP_NUM_THREADS=1` set before `import jax` (must be before — XLA picks
  thread settings up at import time), and the SOFIE+Clad path is
  single-threaded by construction.
* The SOFIE+Clad benchmark only has a single-sample number: the generated
  SOFIE session takes one input vector at a time. That doesn't mean SOFIE
  is "the per-event path" — it just means this benchmark hasn't tried to
  batch it. The framework-level pitch (deps-free, dual-language) is
  separate from any single-sample-vs-batched comparison.
* `plot.png` is the only artefact you should commit/share; the JSON files
  and `model.{onnx,hxx}` are regenerated by `run_all.sh`.
