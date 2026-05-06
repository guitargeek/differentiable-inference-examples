"""Train a HistFactory-style morph surrogate with PyTorch and export it to ONNX.

The HistFactory truth model
---------------------------

For each channel `c` and each bin `b`, the expected yield is

    y_{c,b}(theta) = (B_b + mu * S_b) * (1 + sum_k nu_k * h_k(b)
                                            + sum_m alpha_{c,m} * g_m(b))

where
    B_b           : flat background (constant across bins)
    S_b           : Gaussian signal (peaked, evaluated at the bin centre)
    h_k(b), g_m(b): localised Gaussian "bumps" centred at K (resp. M) different
                    bin positions; h_k are shared nuisances, g_m are
                    per-channel nuisances
    mu             : POI (signal strength)
    nu_k           : K shared nuisances
    alpha_{c,m}    : M per-channel nuisances (only the channel index `c`
                    distinguishes them; the surrogate uses the same M-vector
                    everywhere because all channels share the same NN)

Given a "nominal" reference point with mu=mu_ref=1, nu=0, alpha=0, the
HistFactory morph factor is

    r(b, theta) = y_{c,b}(theta) / y_{c,b}^nom(b)
                = (B_b + mu*S_b)/(B_b + S_b) * (1 + sum_k nu_k h_k(b)
                                                  + sum_m alpha_m g_m(b))

The neural network is trained to regress `log r(b, theta)` as a function of
(bin_pos, theta), and exported as a graph that emits `r = exp(log_r)` -- a
strictly positive scalar. On the ROOT side a single `RooONNXFunc` per channel
multiplies into the binned-likelihood `shape`, with the (rescaled) observable
fed in as `bin_pos`; the binned-NLL evaluator already sweeps `x` over bin
centres, so no per-bin replicas of the network are needed. The corresponding
RooHistFunc holds the nominal shape `y^nom(b) = B_b + S_b`, so the per-bin
yield is reconstructed as

    y(b, theta) = nominal(b) * r(b, theta).

Outputs (written next to this script)
-------------------------------------
  - surrogate_morph.onnx : ONNX graph with two input tensors
                            (`bin_pos` shape [1, 1], `theta` shape [1, 1+K+M])
                            and one scalar output `r(b, theta)`.
  - surrogate_meta.npz   : configuration + nominal templates B_b, S_b, bump
                           positions, MU_TRUE, etc. needed by `benchmark.py`.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Problem configuration
# ---------------------------------------------------------------------------
# Observable range; bin positions are normalised into [-1, 1] for the network.
X_MIN, X_MAX = -5.0, 5.0

# Background level (flat) and signal Gaussian shape parameters (in x).
B_LEVEL = 50.0
S_AMP = 30.0
S_MU = 0.0
S_SIGMA = 1.2

# POI bounds and reference (nominal) value. mu_ref=1.0 means that the nominal
# template is exactly the unit-strength signal+background combination.
MU_MIN, MU_MAX = 0.0, 3.0
MU_REF = 1.0
MU_TRUE = 1.5

# Per-bump amplitude on the multiplicative shape perturbation. Each bump is
# bounded by NU_AMP * |nu_k|. The *base* amplitude is rescaled by
# 1/sqrt(K+M+1) so that the typical-case perturbation amplitude stays
# roughly K-invariant ("many small effects" scaling). Without this rescaling
# the morph factor `1 + sum ...` goes negative for high K, which kills the
# log target during training.
BASE_NU_AMP = 0.10
BUMP_SIGMA = 0.30  # bump width in normalised bin position
NU_RANGE = 3.0     # nuisances are sampled in [-NU_RANGE, NU_RANGE]

# Training defaults
N_TRAIN = 200_000
N_EPOCHS = 30
BATCH_SIZE = 1024
LR = 1e-3
HIDDEN = 32
SEED = 42


# ---------------------------------------------------------------------------
# Truth model (numpy)
# ---------------------------------------------------------------------------
def bin_centers_x(n_bins):
    """Return the x-coordinate of the centre of each of `n_bins` bins."""
    edges = np.linspace(X_MIN, X_MAX, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def bin_positions_norm(n_bins):
    """Return bin centres mapped into [-1, 1] for use as NN input."""
    return bin_centers_x(n_bins) * (2.0 / (X_MAX - X_MIN))


def signal_template(n_bins):
    """S_b : Gaussian-shaped signal template, evaluated at bin centres."""
    xc = bin_centers_x(n_bins)
    return (S_AMP * np.exp(-0.5 * ((xc - S_MU) / S_SIGMA) ** 2)).astype(np.float64)


def background_template(n_bins):
    """B_b : flat background template."""
    return np.full(n_bins, B_LEVEL, dtype=np.float64)


def bump_centers(n_bumps):
    """`n_bumps` interior centres evenly spaced over [-1, 1]."""
    if n_bumps == 0:
        return np.zeros(0, dtype=np.float64)
    return np.linspace(-1.0, 1.0, n_bumps + 2)[1:-1].astype(np.float64)


def bump_values(bin_pos_norm, centers, nu_amp):
    """Localised Gaussian bumps, shape (..., K) given bin_pos_norm shape (...,).

    Returns nu_amp * exp(-0.5 ((bin_pos - c)/BUMP_SIGMA)**2).
    """
    if centers.size == 0:
        return np.zeros(bin_pos_norm.shape + (0,), dtype=np.float64)
    diff = bin_pos_norm[..., None] - centers[None, :]
    return nu_amp * np.exp(-0.5 * (diff / BUMP_SIGMA) ** 2)


def expected_yield(bin_pos_norm, mu, nu, alpha, K, M,
                   shared_centers, channel_centers, nu_amp):
    """Compute y(b, theta) = (B_b + mu S_b) * (1 + sum_k nu_k h_k + sum_m alpha_m g_m).

    All inputs are broadcast-compatible numpy arrays. `bin_pos_norm` provides
    the position; B_b and S_b are evaluated from the position via the same
    parametric templates used to build the bin-wise truth.
    """
    # Recover x from the normalised bin position (so that signal/background
    # templates are continuous in bin_pos rather than tied to a fixed binning).
    x = bin_pos_norm * 0.5 * (X_MAX - X_MIN)
    B = np.full_like(x, B_LEVEL)
    S = S_AMP * np.exp(-0.5 * ((x - S_MU) / S_SIGMA) ** 2)
    base = B + mu * S

    factor = np.ones_like(x)
    if K > 0:
        h = bump_values(bin_pos_norm, shared_centers, nu_amp)  # (..., K)
        factor = factor + (nu * h).sum(axis=-1)
    if M > 0:
        g = bump_values(bin_pos_norm, channel_centers, nu_amp)  # (..., M)
        factor = factor + (alpha * g).sum(axis=-1)
    return base * factor


def nominal_yield_from_pos(bin_pos_norm):
    """y^nom(b) = (B_b + mu_ref * S_b)."""
    x = bin_pos_norm * 0.5 * (X_MAX - X_MIN)
    B = np.full_like(x, B_LEVEL)
    S = S_AMP * np.exp(-0.5 * ((x - S_MU) / S_SIGMA) ** 2)
    return B + MU_REF * S


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LogMorphNet(nn.Module):
    """MLP that regresses `log r(b, theta)` -- one scalar output."""

    def __init__(self, K, M, hidden=HIDDEN):
        super().__init__()
        theta_dim = 1 + K + M  # mu + K shared nuisances + M per-channel nuisances
        self.theta_dim = theta_dim
        self.K = K
        self.M = M
        self.net = nn.Sequential(
            nn.Linear(1 + theta_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Inputs are roughly in [-1, 1] already (bin_pos) or O(1) (mu/nu/alpha).
        # A small constant scaling for mu keeps it in O(1) too.
        mu_scale = 0.5 * (MU_MAX - MU_MIN)
        scales = [mu_scale] + [1.0] * K + [1.0] * M
        self.register_buffer("bin_scale", torch.tensor([1.0], dtype=torch.float32))
        self.register_buffer("theta_scale", torch.tensor(scales, dtype=torch.float32))

    def forward(self, bin_pos, theta):
        return self.net(
            torch.cat([bin_pos / self.bin_scale, theta / self.theta_scale], dim=-1)
        )


class MorphFactorNet(nn.Module):
    """Wrapper exposing r = exp(log_r) for ONNX export."""

    def __init__(self, log_net):
        super().__init__()
        self.log_net = log_net

    def forward(self, bin_pos, theta):
        return torch.exp(self.log_net(bin_pos, theta))


# ---------------------------------------------------------------------------
# Data generation: random (bin_pos, theta) -> log r
# ---------------------------------------------------------------------------
def sample_training(n, K, M, nu_amp, rng):
    """Draw `n` samples of (bin_pos_norm, mu, nu, alpha) and compute log r.

    Samples where the morph factor would be non-positive (the linear
    `1 + sum nu*h` form is not strictly positive) are filtered out so the
    log target stays finite. We oversample upfront to compensate.
    """
    # Oversample to compensate for the rejection step. With sqrt-scaling of
    # the per-nuisance amplitude (see BASE_NU_AMP comment), the rejection
    # rate is small at all K, but we still pad by 50% to be safe.
    over = int(n * 1.5)
    bin_pos = rng.uniform(-1.0, 1.0, size=over).astype(np.float64)
    mu = rng.uniform(MU_MIN, MU_MAX, size=over).astype(np.float64)
    nu = (rng.standard_normal(size=(over, K)).clip(-NU_RANGE, NU_RANGE)
          if K > 0 else np.zeros((over, 0)))
    alpha = (rng.standard_normal(size=(over, M)).clip(-NU_RANGE, NU_RANGE)
             if M > 0 else np.zeros((over, 0)))

    shared_c = bump_centers(K)
    channel_c = bump_centers(M)
    y = expected_yield(
        bin_pos, mu, nu, alpha, K, M, shared_c, channel_c, nu_amp
    )
    y_nom = nominal_yield_from_pos(bin_pos)
    ratio = y / y_nom
    keep = (ratio > 0.01) & np.isfinite(ratio)
    n_keep = int(keep.sum())
    if n_keep < n:
        raise RuntimeError(
            f"Only {n_keep}/{over} training samples have positive morph "
            f"factor (need {n}). Reduce BASE_NU_AMP or NU_RANGE."
        )
    keep_idx = np.flatnonzero(keep)[:n]

    bin_pos = bin_pos[keep_idx]
    mu = mu[keep_idx]
    if K > 0: nu = nu[keep_idx]
    if M > 0: alpha = alpha[keep_idx]
    log_r = np.log(ratio[keep_idx])

    bin_pos_in = bin_pos[:, None].astype(np.float32)
    if K + M > 0:
        theta_in = np.concatenate(
            [mu[:, None]] + ([nu] if K > 0 else []) + ([alpha] if M > 0 else []),
            axis=1,
        ).astype(np.float32)
    else:
        theta_in = mu[:, None].astype(np.float32)
    return bin_pos_in, theta_in, log_r[:, None].astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, bin_pos, theta, log_r):
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    bp_t = torch.from_numpy(bin_pos).to(device)
    th_t = torch.from_numpy(theta).to(device)
    y_t = torch.from_numpy(log_r).to(device)
    n = len(y_t)

    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n)
        running, nb = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            pred = model(bp_t[idx], th_t[idx])
            loss = loss_fn(pred, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        print(f"  epoch {epoch + 1:2d}/{N_EPOCHS}  MSE = {running / nb:.6f}")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(morph_model, theta_dim, onnx_path):
    morph_model.eval()
    example = (
        torch.zeros(1, 1, dtype=torch.float32),
        torch.zeros(1, theta_dim, dtype=torch.float32),
    )
    exported = torch.export.export(morph_model, example)
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
    parser.add_argument("--n-shared", type=int, default=2,
                        help="Number of shared nuisances K (h_k bumps)")
    parser.add_argument("--n-per-channel", type=int, default=1,
                        help="Number of per-channel nuisances M (g_m bumps)")
    parser.add_argument("--n-train", type=int, default=N_TRAIN,
                        help="Number of training samples")
    args = parser.parse_args()

    K = args.n_shared
    M = args.n_per_channel
    # Per-nuisance amplitude scaling: keep typical-case perturbation roughly
    # K-invariant, so the morph factor stays positive even at high K.
    nu_amp = BASE_NU_AMP / np.sqrt(K + M + 1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print(
        f"Generating training data (N={args.n_train}, K={K} shared, M={M} per-channel, "
        f"effective nu_amp={nu_amp:.4f})..."
    )
    bin_pos, theta, log_r = sample_training(args.n_train, K, M, nu_amp, rng)
    print(f"  total training rows = {len(log_r)}")
    print(f"  log r range : [{log_r.min():+.3f}, {log_r.max():+.3f}]")

    theta_dim = 1 + K + M
    print(f"Training MLP (hidden={args.hidden}, theta_dim={theta_dim})...")
    log_net = LogMorphNet(K=K, M=M, hidden=args.hidden)
    train(log_net, bin_pos, theta, log_r)

    morph_model = MorphFactorNet(log_net)

    onnx_path = HERE / "surrogate_morph.onnx"
    print(f"Exporting -> {onnx_path}")
    export_onnx(morph_model, theta_dim=theta_dim, onnx_path=onnx_path)

    # Sanity check at the truth: r at theta=(MU_TRUE, 0, 0) for bin_pos=0.
    with torch.no_grad():
        bp = torch.tensor([[0.0]], dtype=torch.float32)
        th = torch.zeros(1, theta_dim, dtype=torch.float32)
        th[0, 0] = MU_TRUE
        r_pred = morph_model(bp, th).item()
    shared_c = bump_centers(K)
    channel_c = bump_centers(M)
    r_true = expected_yield(
        np.array([0.0]), np.array([MU_TRUE]), np.zeros((1, K)), np.zeros((1, M)),
        K, M, shared_c, channel_c, nu_amp,
    )[0] / nominal_yield_from_pos(np.array([0.0]))[0]
    print(f"  r(b=0, mu={MU_TRUE}, nu=0, alpha=0) : NN = {r_pred:.5f}  truth = {r_true:.5f}")

    meta_path = HERE / "surrogate_meta.npz"
    np.savez(
        meta_path,
        x_min=np.float64(X_MIN), x_max=np.float64(X_MAX),
        b_level=np.float64(B_LEVEL),
        s_amp=np.float64(S_AMP), s_mu=np.float64(S_MU), s_sigma=np.float64(S_SIGMA),
        mu_min=np.float64(MU_MIN), mu_max=np.float64(MU_MAX),
        mu_ref=np.float64(MU_REF), mu_true=np.float64(MU_TRUE),
        nu_amp=np.float64(nu_amp), bump_sigma=np.float64(BUMP_SIGMA),
        nu_range=np.float64(NU_RANGE),
        n_shared=np.int64(K), n_per_channel=np.int64(M),
        shared_centers=shared_c.astype(np.float64),
        channel_centers=channel_c.astype(np.float64),
    )
    print(f"Saved metadata -> {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
