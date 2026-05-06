#!/usr/bin/env bash
# Run every benchmark in this directory end-to-end and produce plot.png.
#
# Steps:
#   1. PyTorch: train + benchmark + export model.onnx       -> results_torch.json
#   2. ROOT/SOFIE: convert model.onnx to model.hxx
#   3. ROOT/SOFIE+Clad: benchmark using model.hxx           -> results_sofie_clad.json
#   4. JAX: benchmark a freshly-initialised MLP             -> results_jax.json
#   5. Plot all three JSON files into plot.png
#
# Requires:
#   - python3 with torch, jax, numpy
#   - ROOT (with tmva-sofie=ON and roofit_clad=ON), sourced via thisroot.sh

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v root >/dev/null 2>&1; then
    echo "ERROR: 'root' not in PATH. Source <root_build>/bin/thisroot.sh first." >&2
    exit 1
fi

echo "==> [1/5] PyTorch benchmark + ONNX export"
python3 benchmark_torch.py --out results_torch.json --onnx model.onnx

echo "==> [2/5] SOFIE: ONNX -> model.hxx"
root -l -b -q onnx_to_cpp.C

echo "==> [3/5] SOFIE+Clad benchmark"
root -l -b -q 'benchmark_sofie_clad.C("results_sofie_clad.json")'

echo "==> [4/5] JAX benchmark"
python3 benchmark_jax.py --out results_jax.json

echo "==> [5/5] Plot"
python3 plot.py --inputs results_torch.json results_jax.json results_sofie_clad.json --out plot.png

echo "Done. See plot.png."
