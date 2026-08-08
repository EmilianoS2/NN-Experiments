import sympy
import numpy as np
import matplotlib.pyplot as plt

# NOTE: sequential order, NOT the shuffled split — panel 3 depends on it
primes = np.array(list(sympy.primerange(2, 110000)))
gaps = np.diff(primes)        # gaps[i] = primes[i+1] - primes[i]
p = primes[:-1]               # the prime each gap starts from

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)

# 1 — the trend exists, the scatter swamps it -----------------------------------
ax = axes[0]
ax.scatter(p, gaps, s=2, color=BLUE, alpha=0.12, edgecolors="none", label="actual gap")
ax.plot(p, np.log(p), color=ORANGE, linewidth=2, label="ln(p) — PNT average")
ax.set_title("1. Gap vs. how far along you are", color=INK, fontsize=11, pad=10, loc="left")
ax.set_xlabel("prime p", color=MUTED, fontsize=9)
ax.set_ylabel("gap to next prime", color=MUTED, fontsize=9)
ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, markerscale=4)

# 2 — the distribution: mode at 6, long right tail ------------------------------
ax = axes[1]
vals, cnts = np.unique(gaps, return_counts=True)
ax.bar(vals, cnts / cnts.sum(), width=1.6, color=BLUE)
ax.axvline(gaps.mean(), color=ORANGE, linewidth=2)
ax.text(gaps.mean() + 1.5, 0.15, f"mean {gaps.mean():.1f}", color=ORANGE, fontsize=9)
ax.text(6, 0.152, "mode 6", color=INK, fontsize=9, ha="center")
ax.set_title("2. How often each gap size occurs", color=INK, fontsize=11, pad=10, loc="left")
ax.set_xlabel("gap size", color=MUTED, fontsize=9)
ax.set_ylabel("share of all gaps", color=MUTED, fontsize=9)
ax.set_xlim(0, 60)

# 3 — the money shot: does this gap tell you the next one? ----------------------
ax = axes[2]
lim = 40
h, xe, ye = np.histogram2d(gaps[:-1], gaps[1:], bins=[np.arange(0, lim + 2, 2)] * 2)
ax.imshow(h.T / h.sum(), origin="lower", cmap="Blues", aspect="auto",
          extent=[0, lim, 0, lim], interpolation="nearest")
r = np.corrcoef(gaps[:-1], gaps[1:])[0, 1]
ax.set_title("3. This gap vs. the next gap", color=INK, fontsize=11, pad=10, loc="left")
ax.set_xlabel("current gap", color=MUTED, fontsize=9)
ax.set_ylabel("next gap", color=MUTED, fontsize=9)
ax.text(0.97, 0.95, f"correlation r = {r:.3f}", transform=ax.transAxes,
        ha="right", va="top", color=INK, fontsize=10)
ax.grid(False)

fig.tight_layout()
fig.savefig("gaps.png", dpi=140, facecolor=SURFACE)
print(f"mean gap {gaps.mean():.2f}   max gap {gaps.max()}   lag-1 correlation {r:.4f}")
