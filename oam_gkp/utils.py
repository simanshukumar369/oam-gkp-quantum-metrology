"""
oam_gkp/utils.py
================
Visualisation and post-training analysis utilities.

All plots are saved as high-quality PDF (300 dpi) by default.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf


# ─── Colour palette ───────────────────────────────────────────────────────────

COLORS = {
    "square"     : "#2196F3",   # blue
    "hexagonal"  : "#4CAF50",   # green
    "oam_ell1"   : "#FF9800",   # orange
    "oam_ell2"   : "#E91E63",   # pink
    "oam_ell3"   : "#9C27B0",   # purple
    "oam_ell4"   : "#795548",   # brown
    "heisenberg" : "#F44336",   # red
    "sql"        : "#607D8B",   # grey
}


# ─── Helper to save figures consistently ──────────────────────────────────────

def _save_figure(save_path: str | None, dpi: int = 300) -> None:
    """Internal helper to save figures as PDF at high resolution."""
    if save_path:
        # Force PDF extension if user forgot
        if not save_path.lower().endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved PDF (300 dpi): {save_path}")
    plt.close()


# ─── 1. Wigner function plot ──────────────────────────────────────────────────

def plot_wigner(
    state_prep,
    title: str = "Twisted GKP Wigner Function",
    save_path: str | None = None,
    n_pts: int = 100,
    extent: float = 7.0,
) -> None:
    """Plot Wigner function with GKP lattice cell overlay."""
    from .states import GKPStatePrep  # avoid circular import

    Q, P, W = state_prep.wigner(
        q_range=(-extent, extent),
        p_range=(-extent, extent),
        n_pts=n_pts,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(abs(W.min()), abs(W.max()))
    im = ax.pcolormesh(Q, P, W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    plt.colorbar(im, ax=ax, label="W(q, p)")

    # Overlay lattice cell
    lattice = state_prep.lattice
    u1, u2  = [v.numpy() for v in lattice.vectors]
    origin  = np.array([0.0, 0.0])
    for n in range(-3, 4):
        for m in range(-3, 4):
            corner = origin + n * u1 + m * u2
            rect   = patches.FancyArrow(
                corner[0], corner[1], u1[0], u1[1],
                width=0.03, head_width=0.15, head_length=0.15,
                color="gold", alpha=0.6, length_includes_head=True
            )
            ax.add_patch(rect)

    theta_deg = np.degrees(float(lattice.theta.numpy()))
    r_val     = float(lattice.r.numpy())
    ell_val   = float(lattice.ell_var.numpy())

    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_title(f"{title}\n"
                 f"ℓ={ell_val:.2f}, r={r_val:.2f}, θ={theta_deg:.1f}°")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")

    plt.tight_layout()
    _save_figure(save_path, dpi=300)


# ─── 2. Training convergence ──────────────────────────────────────────────────

def plot_training_history(
    history: list[dict],
    title: str = "Training Convergence",
    save_path: str | None = None,
) -> None:
    """Plot QFI, P_err, gradient norm, and LR vs training step."""
    steps    = [h["step"]      for h in history]
    qfi      = [h["qfi"]       for h in history]
    p_err    = [h["P_err"]     for h in history]
    grad_n   = [h["grad_norm"] for h in history]
    lr_vals  = [h["lr"]        for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, qfi, color=COLORS["oam_ell1"], lw=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("QFI  $\\mathcal{F}_Q$")
    ax.set_title("Quantum Fisher Information")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(steps, p_err, color=COLORS["square"], lw=1.5)
    ax.axhline(1e-3, ls="--", color="red", alpha=0.7, label="$P_{\\rm thresh}=10^{-3}$")
    ax.set_xlabel("Step")
    ax.set_ylabel("$P_{\\rm err}$")
    ax.set_title("Logical Error Rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogy(steps, np.clip(grad_n, 1e-8, None), color=COLORS["hexagonal"], lw=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("$\\|\\nabla \\mathcal{L}\\|$")
    ax.set_title("Gradient Norm")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, lr_vals, color=COLORS["oam_ell2"], lw=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning rate")
    ax.set_title("Cosine Annealing LR")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(save_path, dpi=300)


# ─── 3. Pareto frontier ───────────────────────────────────────────────────────

def plot_pareto_frontier(
    results_by_geometry: dict[str, list[dict]],
    P_thresh: float = 1e-3,
    save_path: str | None = None,
) -> None:
    """Plot QFI–P_err Pareto frontier for multiple lattice geometries."""
    fig, ax = plt.subplots(figsize=(7, 5))

    color_map = {
        "square"    : COLORS["square"],
        "hexagonal" : COLORS["hexagonal"],
        "oam_ell1"  : COLORS["oam_ell1"],
        "oam_ell2"  : COLORS["oam_ell2"],
        "oam_ell3"  : COLORS["oam_ell3"],
        "oam_ell4"  : COLORS["oam_ell4"],
    }

    for name, res in results_by_geometry.items():
        qfi_vals  = [r["qfi"]   for r in res]
        perr_vals = [r["P_err"] for r in res]
        col = color_map.get(name, "#333333")
        ax.scatter(perr_vals, qfi_vals, s=30, color=col, alpha=0.7, label=name)
        # Connect Pareto-optimal points
        pairs = sorted(zip(perr_vals, qfi_vals), key=lambda x: x[0])
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                color=col, lw=1.2, alpha=0.5)

    ax.axvline(P_thresh, ls="--", color="red", alpha=0.7,
               label=f"$P_{{\\rm thresh}} = {P_thresh}$")
    ax.set_xlabel("Logical error rate $P_{\\rm err}$")
    ax.set_ylabel("Quantum Fisher Information $\\mathcal{F}_Q$")
    ax.set_title("Sensitivity–Fault-Tolerance Pareto Frontier")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(save_path, dpi=300)


# ─── 4. Phase diagram ─────────────────────────────────────────────────────────

def plot_phase_diagram(
    eta_grid: np.ndarray,
    gamma_grid: np.ndarray,
    best_geometry: np.ndarray,
    geometry_names: list[str],
    save_path: str | None = None,
) -> None:
    """Noise phase diagram showing which geometry achieves highest QFI."""
    cmap   = plt.cm.get_cmap("tab10", len(geometry_names))
    ETA, GAMMA = np.meshgrid(eta_grid, gamma_grid, indexing="ij")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    img = ax.pcolormesh(GAMMA, ETA, best_geometry,
                        cmap=cmap, vmin=-0.5, vmax=len(geometry_names) - 0.5,
                        shading="auto")

    cbar = plt.colorbar(img, ax=ax, ticks=range(len(geometry_names)))
    cbar.set_ticklabels(geometry_names)
    cbar.set_label("Optimal geometry")

    ax.set_xlabel("Dephasing rate $\\gamma$")
    ax.set_ylabel("Loss $\\eta$ (transmissivity)")
    ax.set_title("Optimal GKP Geometry vs Noise\n(Paper Fig. 1)")
    plt.tight_layout()

    _save_figure(save_path, dpi=300)


# ─── 5. Summary table ─────────────────────────────────────────────────────────

def print_results_table(results_by_geometry: dict[str, list[dict]]) -> None:
    """Print a formatted table of peak QFI and P_err for each geometry."""
    header = f"{'Geometry':<14} {'QFI (peak)':<14} {'P_err at peak':<16} {'ell*':<6} {'r*':<6} {'θ* (°)':<10}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for name, res in results_by_geometry.items():
        best = max(res, key=lambda x: x["qfi"])
        print(f"  {name:<12} {best['qfi']:<14.4f} {best['P_err']:<16.2e} "
              f"{best['ell_int']:<6} {best['r']:<6.3f} {best['theta_deg']:<10.2f}")
    print("─" * len(header) + "\n")
