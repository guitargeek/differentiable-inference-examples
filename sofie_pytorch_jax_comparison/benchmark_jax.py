import argparse
import json
import os
import statistics
import time

# Pin XLA / BLAS to a single thread BEFORE importing jax. PyTorch is pinned to
# one thread via `torch.set_num_threads(1)` and the SOFIE+Clad benchmark is
# single-threaded by construction; without these env vars XLA happily uses
# every core and the comparison is unfair.
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    "intra_op_parallelism_threads=1"
)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
from jax import random, grad

# -----------------------------
# Config
# -----------------------------
INPUT_DIM = 20
HIDDEN_LAYERS = 5
HIDDEN_DIM = 128
N_SAMPLES = 50000
BATCH_SIZE = 128
DEVICE = "cpu"  # JAX will automatically select cpu/gpu
N_RUNS = 1000

# -----------------------------
# Model definition (pure functions)
# -----------------------------
def init_layer_params(rng, in_dim, out_dim):
    k1, k2 = random.split(rng)
    w = random.normal(k1, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
    b = jnp.zeros((out_dim,))
    return {'w': w, 'b': b}

def init_model(rng, input_dim, hidden_dim, n_layers):
    params = []
    in_dim = input_dim
    for _ in range(n_layers):
        rng, layer_rng = random.split(rng)
        params.append(init_layer_params(layer_rng, in_dim, hidden_dim))
        in_dim = hidden_dim
    # final layer
    rng, layer_rng = random.split(rng)
    params.append(init_layer_params(layer_rng, in_dim, 1))
    return params

def forward(params, x):
    for layer in params[:-1]:
        x = jnp.dot(x, layer['w']) + layer['b']
        x = jax.nn.relu(x)
    # final layer
    layer = params[-1]
    x = jnp.dot(x, layer['w']) + layer['b']
    return x

# -----------------------------
# Synthetic dataset
# -----------------------------
def generate_data(rng, n_samples, input_dim):
    X = random.normal(rng, (n_samples, input_dim))
    y = jnp.sum(X**2, axis=1, keepdims=True) + 0.1 * random.normal(rng, (n_samples,1))
    return X, y

# -----------------------------
# Benchmarking
# -----------------------------
def _time_loop(fn, n_runs):
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = fn().block_until_ready()
    end = time.perf_counter()
    return end - start


def benchmark(params, input_dim, batch_size, n_runs, n_repeats):
    # Match PyTorch / SOFIE: constant ones input.
    x = jnp.ones((batch_size, input_dim))
    x_grad = jnp.ones((batch_size, input_dim))

    n_samples = batch_size * n_runs

    jit_forward = jax.jit(forward)
    grad_forward = jax.jit(grad(lambda p, x: jnp.sum(forward(p, x)), argnums=1))

    # Warmup (also triggers JIT compile).
    _ = jit_forward(params, x).block_until_ready()
    _ = grad_forward(params, x_grad).block_until_ready()

    forward_repeats_us = []
    grad_repeats_us = []
    for _ in range(n_repeats):
        forward_repeats_us.append(
            _time_loop(lambda: jit_forward(params, x), n_runs) / n_samples * 1e6
        )
        grad_repeats_us.append(
            _time_loop(lambda: grad_forward(params, x_grad), n_runs) / n_samples * 1e6
        )

    forward_med = statistics.median(forward_repeats_us)
    grad_med = statistics.median(grad_repeats_us)
    print(
        f"Forward per sample: median {forward_med:.2f} µs "
        f"(min {min(forward_repeats_us):.2f}, max {max(forward_repeats_us):.2f}) over {n_repeats} repeats"
    )
    print(
        f"Grad per sample:    median {grad_med:.2f} µs "
        f"(min {min(grad_repeats_us):.2f}, max {max(grad_repeats_us):.2f}) over {n_repeats} repeats"
    )
    print(f"Grad / Forward ratio: {grad_med / forward_med:.2f}")

    return {
        "forward_us": forward_med,
        "grad_us": grad_med,
        "forward_us_repeats": forward_repeats_us,
        "grad_us_repeats": grad_repeats_us,
    }

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="results_jax.json",
        help="Path to write benchmark results as JSON.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="How many times to repeat the timed loop. Median is reported.",
    )
    args = parser.parse_args()

    rng = random.PRNGKey(0)
    params = init_model(rng, INPUT_DIM, HIDDEN_DIM, HIDDEN_LAYERS)

    print("\n--- JAX Benchmark Results ---")
    single = benchmark(params, INPUT_DIM, 1, 1000, args.repeats)
    print("\n--- JAX Benchmark Results (batched) ---")
    batched = benchmark(params, INPUT_DIM, 1000, 100, args.repeats)

    payload = {
        "entries": [
            {"label": "JAX", **single},
            {"label": "JAX (batched)", **batched},
        ]
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
