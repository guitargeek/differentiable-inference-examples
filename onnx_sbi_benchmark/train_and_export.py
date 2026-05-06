"""Train an SBI surrogate likelihood ratio with PyTorch and export it to ONNX.

The signal hypothesis is the (analytical) skew-normal distribution

    f(x; mu, alpha) = 2 * phi(x - mu) * Phi(alpha * (x - mu))

modulated by K Gaussian-bump nuisance parameters:

    p(x; mu, alpha, nu) = skewnorm(x; mu, alpha) * (1 + sum_k nu_k * b_k(x))

where b_k(x) = NU_AMP * exp(-0.5*((x_norm - c_k)/sigma)^2) and the bump
centres c_k are evenly spaced over the analysis range. Each nuisance therefore
has a distinct, localised effect on the shape, mimicking the way HEP nuisance
parameters describe independent experimental systematics.

The trained surrogate covers theta = (mu, alpha, nu_1, ..., nu_K). In the
inference script (`benchmark.py`) `mu` and the nuisances are shared across all
channels, while each channel has its own `alpha_i`. This is the realistic
shape of a HEP fit with a shared POI + many shared nuisance parameters and a
small per-channel parameter.

Outputs (written next to this script):
  - surrogate_lhr.onnx   ONNX graph with two input tensors (x and theta) and
                         one scalar output: r(x; theta) = exp(z(x; theta)).
  - surrogate_meta.npz   Prior bounds, n_nuisances, default truths.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import skewnorm

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Problem configuration
# ---------------------------------------------------------------------------
X_MIN, X_MAX = -5.0, 5.0
MU_MIN, MU_MAX = -2.0, 2.0
ALPHA_MIN, ALPHA_MAX = -3.0, 3.0
NU_RANGE = 3.0  # nuisance prior in inference is N(0,1); we sample [-NU_RANGE, NU_RANGE]

# Per-nuisance amplitude on the multiplicative shape perturbation.
# Each bump is bounded by NU_AMP * |nu_k|; total perturbation by
# K * NU_AMP * NU_RANGE if every nuisance peaks simultaneously.
NU_AMP = 0.04
BUMP_SIGMA = 0.35  # bump width in normalised x (i.e. relative to half-range)

MU_TRUE = 0.5
ALPHA_TRUE = 2.0  # default; benchmark.py uses different alpha per channel

N_TRAIN = 400_000
N_EPOCHS = 25
BATCH_SIZE = 1024
LR = 1e-3
HIDDEN = 32
SEED = 42


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LogitNet(nn.Module):
    """MLP returning the classifier logit z(x, theta).

    theta has dimension 2 + K = (mu, alpha, nu_1, ..., nu_K). Inputs are
    normalised to roughly unit cube via the prior ranges.
    """

    def __init__(self, n_nuisances, hidden=HIDDEN):
        super().__init__()
        theta_dim = 2 + n_nuisances
        self.theta_dim = theta_dim
        self.net = nn.Sequential(
            nn.Linear(1 + theta_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        x_scale = 0.5 * (X_MAX - X_MIN)
        mu_scale = 0.5 * (MU_MAX - MU_MIN)
        alpha_scale = 0.5 * (ALPHA_MAX - ALPHA_MIN)
        # Nuisances have unit-Gaussian prior, so scale=1 keeps them in O(1).
        scales = [mu_scale, alpha_scale] + [1.0] * n_nuisances
        self.register_buffer("x_scale", torch.tensor([x_scale], dtype=torch.float32))
        self.register_buffer(
            "theta_scale", torch.tensor(scales, dtype=torch.float32)
        )

    def forward(self, x, theta):
        return self.net(torch.cat([x / self.x_scale, theta / self.theta_scale], dim=-1))


class LikelihoodRatioNet(nn.Module):
    """Wrapper that emits the likelihood ratio r = exp(z) for export."""

    def __init__(self, logit_net):
        super().__init__()
        self.logit_net = logit_net

    def forward(self, x, theta):
        return torch.exp(self.logit_net(x, theta))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def bump_centers(K):
    """K interior points evenly spaced on [-1, 1]."""
    return np.linspace(-1.0, 1.0, K + 2)[1:-1].astype(np.float32)


def shape_factor(x, nu, K):
    """Multiplicative pdf perturbation 1 + sum_k nu_k * b_k(x).

    `x` has shape (n,), `nu` has shape (n, K). Returns shape (n,).
    """
    if K == 0:
        return np.ones_like(x)
    x_norm = x * (2.0 / (X_MAX - X_MIN))  # roughly into [-1, 1]
    cs = bump_centers(K)  # (K,)
    # bumps: shape (n, K)
    bumps = NU_AMP * np.exp(
        -0.5 * ((x_norm[:, None] - cs[None, :]) / BUMP_SIGMA) ** 2
    )
    return 1.0 + (nu * bumps).sum(axis=1)


def sample_target(n, rng, K):
    """Rejection-sample from p(x; mu, alpha, nu) ~ skewnorm(x; mu, alpha) *
    (1 + sum_k nu_k * b_k(x)).
    """
    out_x, out_th = [], []
    # Conservative envelope bound for the shape factor.
    w_envelope = float(1.0 + max(K, 1) * NU_AMP * NU_RANGE)
    while sum(len(a) for a in out_x) < n:
        m = max(int(1.5 * (n - sum(len(a) for a in out_x))), 1024)
        mu = rng.uniform(MU_MIN, MU_MAX, m).astype(np.float32)
        alpha = rng.uniform(ALPHA_MIN, ALPHA_MAX, m).astype(np.float32)
        nu = (rng.standard_normal((m, K)).astype(np.float32)
              if K > 0 else np.zeros((m, 0), dtype=np.float32))
        if K > 0:
            nu = np.clip(nu, -NU_RANGE, NU_RANGE)
        x = skewnorm.rvs(a=alpha, loc=mu, scale=1.0, random_state=rng).astype(np.float32)
        w = shape_factor(x, nu, K)
        # Only accept where the perturbed pdf is positive; clip to envelope.
        u = rng.uniform(0.0, 1.0, m).astype(np.float32)
        accept = (
            (w > 0.0)
            & (u < (w / w_envelope))
            & (x >= X_MIN) & (x <= X_MAX)
        )
        x_a = x[accept]
        if K > 0:
            theta_a = np.column_stack([mu[accept], alpha[accept], nu[accept]])
        else:
            theta_a = np.column_stack([mu[accept], alpha[accept]])
        out_x.append(x_a)
        out_th.append(theta_a)
    x_all = np.concatenate(out_x)[:n]
    th_all = np.concatenate(out_th)[:n]
    return x_all, th_all


def sample_reference(n, rng, K):
    """x ~ U(X_MIN, X_MAX), theta drawn from same priors as target."""
    x = rng.uniform(X_MIN, X_MAX, size=n).astype(np.float32)
    mu = rng.uniform(MU_MIN, MU_MAX, size=n).astype(np.float32)
    alpha = rng.uniform(ALPHA_MIN, ALPHA_MAX, size=n).astype(np.float32)
    if K > 0:
        nu = rng.standard_normal(size=(n, K)).astype(np.float32)
        nu = np.clip(nu, -NU_RANGE, NU_RANGE)
        theta = np.column_stack([mu, alpha, nu])
    else:
        theta = np.column_stack([mu, alpha])
    return x, theta


def build_dataset(rng, K):
    x_t, th_t = sample_target(N_TRAIN, rng, K)
    x_r, th_r = sample_reference(len(x_t), rng, K)
    x = np.concatenate([x_t, x_r])[:, None]
    theta = np.concatenate([th_t, th_r], axis=0)
    y = np.concatenate(
        [np.ones(len(x_t), dtype=np.float32), np.zeros(len(x_r), dtype=np.float32)]
    )[:, None]
    perm = rng.permutation(len(y))
    return x[perm], theta[perm], y[perm]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, x, theta, y):
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    x_t = torch.from_numpy(x).to(device)
    th_t = torch.from_numpy(theta).to(device)
    y_t = torch.from_numpy(y).to(device)
    n = len(y_t)

    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n)
        running, nb = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            z = model(x_t[idx], th_t[idx])
            loss = loss_fn(z, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        print(f"  epoch {epoch + 1:2d}/{N_EPOCHS}  loss = {running / nb:.5f}")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(lr_model, theta_dim, onnx_path):
    lr_model.eval()
    example = (
        torch.zeros(1, 1, dtype=torch.float32),
        torch.zeros(1, theta_dim, dtype=torch.float32),
    )
    exported = torch.export.export(lr_model, example)
    torch.onnx.export(
        exported, args=(), f=str(onnx_path), external_data=False, dynamo=True
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=HIDDEN,
                        help="Width of the two MLP hidden layers")
    parser.add_argument("--n-nuisances", type=int, default=4,
                        help="Number of shared nuisance parameters that the "
                             "surrogate accepts (0 disables nuisances).")
    args = parser.parse_args()

    K = args.n_nuisances
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print(f"Generating training data (N={N_TRAIN}, K_nuisances={K}) ...")
    x, theta, y = build_dataset(rng, K)
    print(f"  total training rows = {len(y)}")

    print(f"Training classifier (hidden={args.hidden}, theta_dim={2 + K}) ...")
    logit_net = LogitNet(n_nuisances=K, hidden=args.hidden)
    train(logit_net, x, theta, y)

    onnx_path = HERE / "surrogate_lhr.onnx"
    print(f"Exporting -> {onnx_path}")
    lr_model = LikelihoodRatioNet(logit_net)
    export_onnx(lr_model, theta_dim=2 + K, onnx_path=onnx_path)

    # Sanity check at the truth
    with torch.no_grad():
        x_chk = torch.tensor([[0.0]], dtype=torch.float32)
        th_chk = torch.zeros(1, 2 + K, dtype=torch.float32)
        th_chk[0, 0] = MU_TRUE
        th_chk[0, 1] = ALPHA_TRUE
        # nuisances at zero (their truth)
        r = lr_model(x_chk, th_chk).item()
    print(f"  r(x=0; mu={MU_TRUE}, alpha={ALPHA_TRUE}, nu=0) = {r:.5f}")

    meta_path = HERE / "surrogate_meta.npz"
    np.savez(
        meta_path,
        mu_true=np.float64(MU_TRUE),
        alpha_true=np.float64(ALPHA_TRUE),
        x_min=np.float64(X_MIN), x_max=np.float64(X_MAX),
        mu_min=np.float64(MU_MIN), mu_max=np.float64(MU_MAX),
        alpha_min=np.float64(ALPHA_MIN), alpha_max=np.float64(ALPHA_MAX),
        n_nuisances=np.int64(K),
        nu_range=np.float64(NU_RANGE),
    )
    print(f"Saved metadata -> {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
