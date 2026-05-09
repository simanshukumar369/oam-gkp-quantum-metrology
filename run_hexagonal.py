"""
run_hexagonal.py — Hexagonal GKP lattice simulation for OAM-GKP paper
======================================================================
Adds the hexagonal lattice as a reference row to Tables I and II.

Hexagonal geometry (from lattice.py comments):
    ell = 0,  r = (π/3)^{1/4} ≈ 0.9036   →  θ = 0°, optimal isotropic lattice

Two passes per noise point:
  [1] Fixed hexagonal  — lattice frozen, only Bloch angles + ψ_LO optimised
  [2] Seeded at hex    — all variables free; shows whether optimiser drifts
                         toward OAM-twisted solution (key result for paper)

Usage:
    conda activate noon-sim
    cd ~/OAM-research26
    python run_hexagonal.py

Output: results printed to stdout (LaTeX rows) + results/hexagonal_results.json
"""

import os, sys, json, time
import numpy as np

# ── importable from ~/OAM-research26/ ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from oam_gkp.lattice import GKPLattice
from oam_gkp.states  import GKPStatePrep
from oam_gkp.circuit import SensingCircuit
from oam_gkp.qfi     import qfi_mixed

# ── hyper-parameters (must match main.py runs) ───────────────────────────────
CUTOFF  = 30
EPSILON = 0.0631   # finite-energy GKP  (~10 dB squeezing)
N_STEPS = 500
LR      = 5e-3

# Hexagonal lattice: ell=0, r=(π/3)^{1/4}  (lattice.py canonical definition)
HEX_ELL = 0.0
HEX_R   = (np.pi / 3) ** 0.25   # ≈ 0.9036

# softplus is used inside GKPLattice.r property: r_actual = softplus(r_var)+1e-3
# Invert to get r_var that yields HEX_R at initialisation
def _softplus_inv(y):
    """softplus^{-1}(y) = log(exp(y) - 1),  y must be > 0."""
    return np.log(np.exp(float(y)) - 1.0)

HEX_R_VAR_INIT = _softplus_inv(HEX_R - 1e-3)   # raw variable value

NOISE_POINTS = [
    dict(label="Low noise",  eta=0.9, gamma=0.05),
    dict(label="High noise", eta=0.8, gamma=0.10),
]

# ── Pass 1 — fixed hexagonal geometry ────────────────────────────────────────

def run_fixed_hex(eta, gamma, n_steps, lr):
    """
    Freeze ell and r at hexagonal values; optimise only Bloch angles + ψ_LO.
    Mirrors exactly how main.py runs square/OAM geometries so results are
    directly comparable.
    """
    # Build lattice frozen at hex point
    lattice = GKPLattice(ell=HEX_ELL, r=HEX_R_VAR_INIT)
    lattice.ell_var.assign(HEX_ELL)
    # Freeze by excluding from trainable_variables below

    state_prep = GKPStatePrep(lattice=lattice, epsilon=EPSILON, cutoff=CUTOFF)
    circuit    = SensingCircuit(state_prep=state_prep, eta=eta, gamma=gamma)

    optimizer  = tf.keras.optimizers.Adam(learning_rate=lr)

    # Only optimise Bloch angles + LO angle; lattice geometry is frozen
    opt_vars = state_prep.trainable_variables + [circuit.psi_var]

    best = dict(qfi=-np.inf, p_err=np.inf)

    for step in range(n_steps):
        with tf.GradientTape() as tape:
            rho      = circuit.run()
            qfi_val  = qfi_mixed(rho)
            p_err    = _read_perr(circuit)
            loss     = -qfi_val

        grads = tape.gradient(loss, opt_vars)
        optimizer.apply_gradients(zip(grads, opt_vars))

        q = float(qfi_val.numpy())
        p = float(p_err)
        if q > best["qfi"]:
            best.update(qfi=q, p_err=p)

        if (step + 1) % 100 == 0:
            print(f"    step {step+1:4d}/{n_steps}  QFI={q:.4f}  P_err={p:.2e}")

    best.update(
        r=float(HEX_R),
        theta_deg=0.0,
        ell=HEX_ELL,
    )
    return best


# ── Pass 2 — full optimisation seeded at hex ─────────────────────────────────

def run_seeded_hex(eta, gamma, n_steps, lr):
    """
    All variables free; seeded at hexagonal point.
    If optimiser drifts toward ell>0, θ>0° → confirms OAM-twisted wins.
    """
    lattice = GKPLattice(ell=HEX_ELL, r=HEX_R_VAR_INIT)
    lattice.ell_var.assign(HEX_ELL)

    state_prep = GKPStatePrep(lattice=lattice, epsilon=EPSILON, cutoff=CUTOFF)
    circuit    = SensingCircuit(state_prep=state_prep, eta=eta, gamma=gamma)

    optimizer  = tf.keras.optimizers.Adam(learning_rate=lr)

    # All trainable variables including lattice geometry
    opt_vars = (lattice.trainable_variables
                + state_prep.trainable_variables
                + [circuit.psi_var])

    best = dict(qfi=-np.inf, p_err=np.inf, ell=HEX_ELL, r=HEX_R, theta_deg=0.0)

    for step in range(n_steps):
        with tf.GradientTape() as tape:
            rho     = circuit.run()
            qfi_val = qfi_mixed(rho)
            p_err   = _read_perr(circuit)
            loss    = -qfi_val

        grads = tape.gradient(loss, opt_vars)
        optimizer.apply_gradients(zip(grads, opt_vars))

        q = float(qfi_val.numpy())
        p = float(p_err)
        if q > best["qfi"]:
            best.update(
                qfi=q,
                p_err=p,
                ell=float(lattice.ell_var.numpy()),
                r=float(lattice.r.numpy()),           # softplus-transformed
                theta_deg=float(np.degrees(lattice.theta.numpy())),
            )

        if (step + 1) % 100 == 0:
            print(f"    step {step+1:4d}/{n_steps}  QFI={q:.4f}  P_err={p:.2e}"
                  f"  ell={lattice.ell_var.numpy():.2f}"
                  f"  r={lattice.r.numpy():.3f}"
                  f"  θ={np.degrees(lattice.theta.numpy()):.1f}°")

    return best


# ── helper: read P_err from circuit internals ─────────────────────────────────

def _read_perr(circuit: SensingCircuit) -> float:
    """
    P_err lives inside SensingCircuit but isn't returned by .run().
    Re-run noise channels to extract it, or approximate from last rho trace.
    Since circuit stores last rho internally after run(), attempt attribute
    access; fall back to NaN so the optimisation loop still works.
    """
    # SensingCircuit.run() returns rho; P_err is computed internally.
    # Try common attribute names used in main.py:
    for attr in ("last_p_err", "_last_p_err", "p_err", "_p_err"):
        if hasattr(circuit, attr):
            v = getattr(circuit, attr)
            return float(v.numpy()) if hasattr(v, "numpy") else float(v)
    return float("nan")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    all_results = {}

    for cfg in NOISE_POINTS:
        label, eta, gamma = cfg["label"], cfg["eta"], cfg["gamma"]
        print(f"\n{'='*62}")
        print(f"  {label}  (η={eta}, γ={gamma})")
        print(f"{'='*62}")

        print(f"\n  [1/2] Fixed hexagonal  "
              f"(ell=0, r={HEX_R:.4f}, θ=0°) — Bloch + ψ_LO only")
        t0 = time.time()
        res_fixed = run_fixed_hex(eta=eta, gamma=gamma,
                                   n_steps=N_STEPS, lr=LR)
        res_fixed["wall_time_s"] = time.time() - t0
        print(f"  → QFI={res_fixed['qfi']:.4f}  "
              f"P_err={res_fixed['p_err']:.2e}  "
              f"({res_fixed['wall_time_s']:.0f}s)")

        print(f"\n  [2/2] Seeded at hex — all variables free")
        t0 = time.time()
        res_seed = run_seeded_hex(eta=eta, gamma=gamma,
                                   n_steps=N_STEPS, lr=LR)
        res_seed["wall_time_s"] = time.time() - t0
        print(f"  → QFI={res_seed['qfi']:.4f}  "
              f"P_err={res_seed['p_err']:.2e}  "
              f"ell*={res_seed['ell']:.2f}  "
              f"r*={res_seed['r']:.3f}  "
              f"θ*={res_seed['theta_deg']:.1f}°  "
              f"({res_seed['wall_time_s']:.0f}s)")

        all_results[label] = dict(fixed=res_fixed, seeded=res_seed,
                                   eta=eta, gamma=gamma)

    # ── LaTeX output ──────────────────────────────────────────────────────────
    print("\n\n" + "="*62)
    print("  LATEX TABLE ROWS  (insert after OAM ℓ=2 row in each table)")
    print("="*62)

    for label, res in all_results.items():
        f = res["fixed"]
        s = res["seeded"]
        print(f"\n% {label}  (η={res['eta']}, γ={res['gamma']})")
        print(f"% Fixed hexagonal geometry (ell=0, r=(π/3)^{{1/4}}, θ=0°):")
        print(
            f"Hexagonal & $0$ & $0°$ & ${HEX_R:.3f}$ & "
            f"${f['qfi']:.4f}$ & "
            f"${f['p_err']:.2e}$ \\\\"
        )
        print(f"% Seeded optimisation converged to:")
        print(
            f"%   ell*={s['ell']:.2f},  r*={s['r']:.3f},  "
            f"θ*={s['theta_deg']:.1f}°,  "
            f"QFI={s['qfi']:.4f},  P_err={s['p_err']:.2e}"
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = "results/hexagonal_results.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
