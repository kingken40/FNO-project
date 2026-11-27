"""
Plots example starting waves from the band-only test dataset.

Shows one example from each frequency band:
control (17-24), middle (25-31), and high (32-48).
"""

import numpy as np
import matplotlib.pyplot as plt

# Load the small band-only dataset
data = np.load(
    r"band_test_small\frequency_extension_dataset.npz"
)

# 256 spatial points from 0 to 1
x = np.linspace(0, 1, 256)

# Take the first starting wave from each band
control = data["X_test_band_control"][0, 0, :]
middle = data["X_test_band_middle"][0, 0, :]
high = data["X_test_band_high"][0, 0, :]

# Create three stacked graphs
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(x, control)
axes[0].set_title("Control Band (Modes 17-24 Only)")

axes[1].plot(x, middle)
axes[1].set_title("Middle Band (Modes 25-31 Only)")

axes[2].plot(x, high)
axes[2].set_title("High-Frequency Band (Modes 32-48 Only)")
axes[2].set_xlabel("x")

for ax in axes:
    ax.set_ylabel("u₀(x)")

plt.tight_layout()
plt.savefig("band_only_examples.png", dpi=200)
plt.show()
