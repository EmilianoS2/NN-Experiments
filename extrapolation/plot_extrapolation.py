# extrapolation/plot_extrapolation.py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from steps_data import X_train, y_train, TRAIN_MAX, truth, nx, ny, dy

# Palette (validated categorical pair: blue/orange, all checks pass) =======
SURFACE, INK, SECOND, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#f0efec"
RELU, TANH = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})

# Train one MLP per activation ============================================
X_t = torch.tensor(nx(X_train), dtype=torch.float32).reshape(-1, 1)
y_t = torch.tensor(ny(y_train), dtype=torch.float32).reshape(-1, 1)

def train(act):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1, 64), act(), nn.Linear(64, 64), act(), nn.Linear(64, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(2000):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X_t), y_t)
        loss.backward()
        opt.step()
    model.eval()

    def predict(x_raw):
        with torch.no_grad():
            t = torch.tensor(nx(x_raw), dtype=torch.float32).reshape(-1, 1)
            return dy(model(t).numpy().ravel())
    return predict

grid = np.linspace(0, 100_000, 2000)
y_true = truth(grid)
y_relu = train(nn.ReLU)(grid)
y_tanh = train(nn.Tanh)(grid)

# Plot ====================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
)

def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.axvspan(0, TRAIN_MAX, color=WASH, zorder=0)
    ax.axvline(TRAIN_MAX, color=AXIS, lw=1.2, ls=(0, (4, 3)), zorder=1)

for ax in (ax1, ax2):
    style(ax)

# -- top: predictions vs truth
ax1.plot(grid, y_true, color=INK,  lw=2, ls=(0, (5, 3)), label="ground truth  (steps = seconds / 3)", zorder=5)
ax1.plot(grid, y_relu, color=RELU, lw=2, label="MLP · ReLU", zorder=4)
ax1.plot(grid, y_tanh, color=TANH, lw=2, label="MLP · tanh", zorder=4)

ax1.set_ylabel("steps taken", color=SECOND, fontsize=10)
ax1.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=SECOND)
ax1.set_title(
    "A network extrapolates according to its activation function, not your data",
    color=INK, fontsize=13, fontweight="bold", loc="left", pad=14,
)

# direct labels at the right edge (offset to avoid collision)
for y, color, name, dy_pt in [(y_true[-1], INK, "truth", 9),
                              (y_relu[-1], RELU, "ReLU", -11),
                              (y_tanh[-1], TANH, "tanh", 0)]:
    ax1.annotate(f"{name}  {y:,.0f}", xy=(grid[-1], y), xytext=(6, dy_pt),
                 textcoords="offset points", color=color, fontsize=9,
                 fontweight="bold", va="center")

lo, hi = ax1.get_ylim()
band = lo + 0.03 * (hi - lo)
ax1.text(TRAIN_MAX * 0.5, band, "trained here", color=MUTED, fontsize=9, ha="center")
ax1.text(TRAIN_MAX * 1.08, band, "never seen  →", color=MUTED, fontsize=9, ha="left")

# -- bottom: absolute error
ax2.plot(grid, np.maximum(np.abs(y_relu - y_true), 1e-2), color=RELU, lw=2, zorder=4)
ax2.plot(grid, np.maximum(np.abs(y_tanh - y_true), 1e-2), color=TANH, lw=2, zorder=4)
ax2.set_yscale("log")
ax2.set_ylabel("absolute error, steps  (log)", color=SECOND, fontsize=10)
ax2.set_xlabel("seconds elapsed", color=SECOND, fontsize=10)
ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))

for y, color, name in [(np.abs(y_relu - y_true)[-1], RELU, "ReLU"),
                       (np.abs(y_tanh - y_true)[-1], TANH, "tanh")]:
    ax2.annotate(f"{name}  {y:,.0f}", xy=(grid[-1], y), xytext=(6, 0),
                 textcoords="offset points", color=color, fontsize=9,
                 fontweight="bold", va="center")

fig.subplots_adjust(right=0.86, hspace=0.12)
fig.savefig("extrapolation.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
print("wrote extrapolation/extrapolation.png")
