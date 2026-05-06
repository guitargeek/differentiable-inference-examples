import argparse
import json
import os
import statistics
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn


# -----------------------------
# Config
# -----------------------------
INPUT_DIM = 20
HIDDEN_LAYERS = 5
HIDDEN_DIM = 128
N_SAMPLES = 50000
BATCH_SIZE = 128
EPOCHS = 10
DEVICE = "cpu"  # change to "cuda" if needed


# -----------------------------
# Model definition
# -----------------------------
class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()

        layers = []
        in_dim = input_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))  # single output

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Benchmarking
# -----------------------------
def benchmark(model, input_dim, batch_size, n_runs, n_repeats):
    model.eval()

    x = torch.ones(batch_size, input_dim)
    x_grad = torch.ones(batch_size, input_dim, requires_grad=True)

    n_samples = batch_size * n_runs

    # Warmup
    for _ in range(10):
        y = model(x_grad)
        y.backward(torch.ones_like(y))
        x_grad.grad.zero_()

    forward_repeats_us = []
    grad_repeats_us = []
    for _ in range(n_repeats):
        # Forward (no grad)
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(x)
        forward_repeats_us.append((time.perf_counter() - start) / n_samples * 1e6)

        # Gradient
        start = time.perf_counter()
        for _ in range(n_runs):
            y = model(x_grad)
            y.backward(torch.ones_like(y))
            x_grad.grad.zero_()
        grad_repeats_us.append((time.perf_counter() - start) / n_samples * 1e6)

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
# ONNX Export
# -----------------------------
def export_onnx(model, input_dim, filename="model.onnx"):
    model.eval()
    dummy_input = torch.ones(1, input_dim).to(DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,  # match PyTorch default
        external_data=False,
    )

    print(f"\nModel exported to {filename}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="results_torch.json",
        help="Path to write benchmark results as JSON.",
    )
    parser.add_argument(
        "--onnx",
        default="model.onnx",
        help="Path to write the ONNX export of the freshly initialised model.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="How many times to repeat the timed loop. Median is reported.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    # Limit to single thread
    torch.set_num_threads(1)

    model = FullyConnectedNet(INPUT_DIM, HIDDEN_DIM, HIDDEN_LAYERS).to(DEVICE)
    export_onnx(model, INPUT_DIM, filename=args.onnx)

    # JIT compile
    model = torch.jit.script(model)
    model = torch.jit.freeze(model)

    print("\n--- PyTorch Benchmark Results ---")
    single = benchmark(model, INPUT_DIM, 1, 1000, args.repeats)
    print("\n--- PyTorch Benchmark Results (batched) ---")
    batched = benchmark(model, INPUT_DIM, 1000, 100, args.repeats)

    payload = {
        "entries": [
            {"label": "PyTorch", **single},
            {"label": "PyTorch (batched)", **batched},
        ]
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
