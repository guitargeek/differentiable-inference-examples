import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.optim as optim


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
def benchmark(model, input_dim, batch_size, n_runs):
    model.eval()

    x = torch.ones(batch_size, input_dim)
    x_grad = torch.ones(batch_size, input_dim, requires_grad=True)

    n_samples = batch_size * n_runs

    # Warmup
    for _ in range(10):
        y = model(x_grad)
        y.backward(torch.ones_like(y))

    # If you want to validate
    # print("Gradient check:")
    # for i, val in enumerate(x_grad.grad[0]):
    #     print(i, val.detach().numpy())
    # x_grad.grad.zero_()

    # Forward (no grad)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)
    end = time.perf_counter()
    inference_time = (end - start) / n_samples
    print(f"Forward per sample: {inference_time*1e6:.2f} µs")

    # Gradient
    start = time.perf_counter()
    for _ in range(n_runs):
        y = model(x_grad)
        y.backward(torch.ones_like(y))
        x_grad.grad.zero_()
    end = time.perf_counter()
    grad_time = (end - start) / n_samples
    print(f"Grad per sample: {grad_time*1e6:.2f} µs")

    print(f"Grad / Forward ratio: {grad_time / inference_time:.2f}")


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
    torch.manual_seed(0)

    model = FullyConnectedNet(INPUT_DIM, HIDDEN_DIM, HIDDEN_LAYERS).to(DEVICE)

    print("\n--- PyTorch Benchmark Results ---")
    benchmark(model, INPUT_DIM, 1, 1000)
    print("\n--- PyTorch Benchmark Results (batched) ---")
    benchmark(model, INPUT_DIM, 1000, 1)
    export_onnx(model, INPUT_DIM)


if __name__ == "__main__":
    main()
