import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Benchmark data (µs per sample)
# -----------------------------

frameworks = ["PyTorch", "PyTorch (batched)", "JAX", "JAX (batched)", "SOFIE+Clad (1 thread)"]
forward_times = [85.89, 1.22, 42.12, 1.52, 20.7978]
grad_times    = [662.16, 11.88, 151.07, 126.15, 72.0978]

x = np.arange(len(frameworks))
width = 0.35

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x - width/2, forward_times, width, label='Forward', color="#4C72B0")
rects2 = ax.bar(x + width/2, grad_times, width, label='Gradient', color="#55A868")

# Labels and title
ax.set_ylabel('Time per sample [µs]')
ax.set_title('Forward vs Gradient Time per Sample (CPU) - MLP with ~70k parameters')
ax.set_xticks(x)
ax.set_xticklabels(frameworks, rotation=30, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bars with values
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
# plt.show()
plt.savefig("plot.png")
