"""
run_fractional_ell.py
=====================
Study fractional OAM charges ℓ ∈ {0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
at the low-noise point (η=0.9, γ=0.05).

The optimizer already treats ℓ as continuous during training.  This script
simply initialises ℓ at non-integer values and lets it converge freely,
WITHOUT snapping to the nearest integer post-training.

This tests:
    1. Whether P_err varies smoothly with ℓ (or shows discrete jumps)
    2. Whether fractional ℓ values outperform the nearest integer
    3. The shape of the P_err(ℓ) curve — does it suggest an optimal ℓ*?

Run:
    conda activate noon-sim
    cd ~/OAM-research26
    python run_fractional_ell.py

Output:
    results/fractional_ell_results.csv   — QFI, P_err, r*, theta* vs ℓ_init
    results/fractional_ell_table.txt     — formatted table for LaTeX
    results/figures/fractional_ell_curve.pdf  — P_err vs ℓ plot
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oam_gkp.lattice   import oam_lattice
from oam_gkp.states    import GKPStatePrep, DEFAULT_CUTOFF, DEFAULT_EPSILON
from oam_gkp.circuit   import SensingCircuit
from oam_gkp.loss      import CombinedLoss
from oam_gkp.optimizer import Optimizer

tf.keras.backend.set_floatx("float64")
os.makedirs("results/figures", exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────
ETA        = 0.9
GAMMA      = 0.05
N_STEPS    = 500
LAM        = 10.0
ELL_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Nature-style plot params
plt.rcParams.update({
    "font.family"       : "serif",
    "font.serif"        : ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset"  : "stix",
    "font.size"         : 8,
    "axes.linewidth"    : 0.8,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "xtick.direction"   : "out",
    "ytick.direction"   : "out",
    "savefig.dpi"       : 300,
    "pdf.fonttype"      : 42,
})

NAT = {"sq": "#2166AC", "ell": "#D6604D", "ref": "#B2182B"}

# ─── Run each ℓ value ─────────────────────────────────────────────────────────
results = []

for ell_init in ELL_VALUES:
    print(f"\n{'='*55}")
    print(f"  ℓ_init = {ell_init:.1f}")
    print(f"{'='*55}")

    lattice    = oam_lattice(ell_init)
    state_prep = GKPStatePrep(lattice, cutoff=DEFAULT_CUTOFF,
                               epsilon=DEFAULT_EPSILON)
    circuit    = SensingCircuit(state_prep, eta=ETA, gamma=GAMMA)
    loss_fn    = CombinedLoss(circuit, lam=LAM)
    opt        = Optimizer(circuit, loss_fn, lr=5e-3, log_interval=100)

    opt.train(n_steps=N_STEPS, verbose=True)

    _, info    = loss_fn()
    ell_final  = float(lattice.ell_var.numpy())
    theta_deg  = float(np.degrees(lattice.theta.numpy()))
    r_final    = float(lattice.r.numpy())

    row = {
        "ell_init"  : ell_init,
        "ell_final" : ell_final,
        "theta_deg" : theta_deg,
        "r_star"    : r_final,
        "qfi"       : info["qfi"],
        "perr"      : info["P_err"],
        "capacity"  : info["qfi"] * (-np.log(info["P_err"] + 1e-30)),
    }
    results.append(row)
    print(f"\n  ℓ_init={ell_init:.1f} → ℓ_final={ell_final:.3f}  "
          f"θ={theta_deg:.1f}°  r*={r_final:.3f}  "
          f"QFI={info['qfi']:.4f}  P_err={info['P_err']:.3e}  "
          f"C={row['capacity']:.2f}")

# ─── Save CSV ─────────────────────────────────────────────────────────────────
import csv
csv_path = "results/fractional_ell_results.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print(f"\nResults saved to {csv_path}")

# ─── Print LaTeX table ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("LaTeX table rows (copy into paper):")
print("="*70)
print(r"\toprule")
print(r"$\ell_\mathrm{init}$ & $\ell^*$ & $\theta^*$ & $r^*$ & "
      r"$\mathcal{F}_Q$ & $P_\mathrm{err}$ & $\mathcal{C}$ \\")
print(r"\midrule")
for r in results:
    int_mark = "" if abs(r["ell_final"] - round(r["ell_final"])) > 0.05 else \
               r"$^\dagger$"
    print(f"  ${r['ell_init']:.1f}$ & ${r['ell_final']:.3f}{int_mark}$ & "
          f"${r['theta_deg']:.1f}^\\circ$ & ${r['r_star']:.3f}$ & "
          f"${r['qfi']:.4f}$ & ${r['perr']:.2e}$ & ${r['capacity']:.1f}$ \\\\")
print(r"\bottomrule")
print(r"$^\dagger$ converged to integer value.")

# ─── Plot P_err vs ell_init ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 2.2),
                          gridspec_kw={"wspace": 0.40})

ell_vals  = [r["ell_init"]  for r in results]
perr_vals = [r["perr"]      for r in results]
cap_vals  = [r["capacity"]  for r in results]
qfi_vals  = [r["qfi"]       for r in results]

# Left — P_err vs ell
ax = axes[0]
ax.semilogy(ell_vals, perr_vals, "o-", color=NAT["ell"],
            ms=5, lw=1.3, mec="white", mew=0.6, label=r"$P_\mathrm{err}(\ell)$")
ax.axhline(1e-3, color=NAT["ref"], lw=0.8, ls="--", alpha=0.8)
ax.text(max(ell_vals)*0.97, 1.2e-3, r"$P_\mathrm{th}$",
        color=NAT["ref"], fontsize=6.5, ha="right")
# Mark integer ell values
for ell_int in [0, 1, 2]:
    res_int = next((r for r in results if r["ell_init"] == float(ell_int)), None)
    if res_int:
        ax.plot(ell_int, res_int["perr"], "o", color=NAT["sq"],
                ms=7, mec="white", mew=0.6, zorder=6)
ax.set_xlabel(r"OAM charge $\ell$ (initial value)")
ax.set_ylabel(r"Logical error rate $P_\mathrm{err}$")
ax.set_title(r"$P_\mathrm{err}$ vs $\ell$"
             f"\n(η={ETA}, γ={GAMMA})", fontsize=8, pad=4)
ax.text(0.04, 0.96, "a", transform=ax.transAxes,
        fontsize=9, fontweight="bold", va="top")
ax.set_xlim(-0.15, max(ell_vals)+0.2)

# Right — Metrological capacity vs ell
ax = axes[1]
ax.plot(ell_vals, cap_vals, "s-", color=NAT["ell"],
        ms=5, lw=1.3, mec="white", mew=0.6, label=r"$\mathcal{C}(\ell)$")
for ell_int in [0, 1, 2]:
    res_int = next((r for r in results if r["ell_init"] == float(ell_int)), None)
    if res_int:
        ax.plot(ell_int, res_int["capacity"], "s", color=NAT["sq"],
                ms=7, mec="white", mew=0.6, zorder=6,
                label="Integer ℓ" if ell_int == 0 else "")
ax.set_xlabel(r"OAM charge $\ell$ (initial value)")
ax.set_ylabel(r"Metrological capacity $\mathcal{C} = \mathcal{F}_Q \cdot (-\ln P_\mathrm{err})$")
ax.set_title(r"Capacity $\mathcal{C}$ vs $\ell$"
             f"\n(η={ETA}, γ={GAMMA})", fontsize=8, pad=4)
ax.text(0.04, 0.96, "b", transform=ax.transAxes,
        fontsize=9, fontweight="bold", va="top")
ax.set_xlim(-0.15, max(ell_vals)+0.2)
ax.legend(frameon=False, fontsize=6.5,
          handles=[
              plt.Line2D([0],[0], color=NAT["ell"], marker="s", ms=4, lw=1.2,
                         mec="white", label=r"Fractional $\ell$"),
              plt.Line2D([0],[0], color=NAT["sq"],  marker="s", ms=6, lw=0,
                         mec="white", label=r"Integer $\ell$"),
          ])

fig.savefig("results/figures/fractional_ell_curve.pdf",
            bbox_inches="tight", pad_inches=0.02)
fig.savefig("results/figures/fractional_ell_curve.png",
            dpi=150, bbox_inches="tight", pad_inches=0.02)
plt.close()
print("\nFigure saved: results/figures/fractional_ell_curve.pdf")
print("\nDone. Paste the LaTeX table rows above into the paper.")
