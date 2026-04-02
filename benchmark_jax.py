import time
import jax
import jax.numpy as jnp
from jax import random, grad, vmap
from functools import partial

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
def benchmark(params, input_dim, batch_size, n_runs):
    rng = random.PRNGKey(42)
    x = random.normal(rng, (batch_size, input_dim))

    n_samples = batch_size * n_runs

    # -----------------
    # Forward (jit)
    # -----------------
    jit_forward = jax.jit(forward)

    # Warmup
    _ = jit_forward(params, x).block_until_ready()

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = jit_forward(params, x).block_until_ready()
        # _ = forward(params, x).block_until_ready()
    end = time.perf_counter()
    inference_time = (end - start) / n_samples
    print(f"Forward per sample: {inference_time*1e6:.2f} µs")

    # -----------------
    # Gradient w.r.t input
    # -----------------
    grad_forward = jax.jit(grad(lambda p, x: jnp.sum(forward(p, x)), argnums=1))

    x_grad = random.normal(rng, (batch_size, input_dim))
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = grad_forward(params, x_grad).block_until_ready()
    end = time.perf_counter()
    grad_time = (end - start) / n_samples
    print(f"Grad per sample: {grad_time*1e6:.2f} µs")
    print(f"Grad / Forward ratio: {grad_time / inference_time:.2f}")

# -----------------------------
# Main
# -----------------------------
def main():
    rng = random.PRNGKey(0)
    params = init_model(rng, INPUT_DIM, HIDDEN_DIM, HIDDEN_LAYERS)

    X, y = generate_data(rng, N_SAMPLES, INPUT_DIM)

    print("\n--- JAX Benchmark Results ---")
    benchmark(params, INPUT_DIM, 1, 1000)
    print("\n--- JAX Benchmark Results (batched) ---")
    benchmark(params, INPUT_DIM, 1000, 1)

if __name__ == "__main__":
    main()
