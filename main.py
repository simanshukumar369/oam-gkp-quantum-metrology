"""
main.py
=======
Main entry point for OAM-GKP sensing optimisation.

Usage
-----
    python main.py --mode single   --eta 0.9 --gamma 0.05 --ell 2
    python main.py --mode pareto   --eta 0.8 --gamma 0.10
    python main.py --mode diagram
    python main.py --mode verify

Modes
-----
single  : optimise a single (eta, gamma, ell) configuration and plot results.
pareto  : sweep lambda for all three lattice geometries and plot Pareto frontier.
diagram : sweep (eta, gamma) grid and produce noise phase diagram.
verify  : run sanity checks (symplecticity, QFI scaling, gradient flow).
"""

import argparse
import os
import numpy as np
import tensorflow as tf

from oam_gkp.lattice    import square_lattice, hexagonal_lattice, oam_lattice
from oam_gkp.states     import GKPStatePrep, DEFAULT_CUTOFF, DEFAULT_EPSILON
from oam_gkp.circuit    import SensingCircuit
from oam_gkp.loss       import CombinedLoss, pareto_sweep
from oam_gkp.optimizer  import Optimizer
from oam_gkp.qfi        import qfi_pure, qfi_mixed
from oam_gkp.utils      import (plot_training_history, plot_wigner,
                                 plot_pareto_frontier, print_results_table)

# ─── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Single optimisation run ──────────────────────────────────────────────────

def run_single(eta: float, gamma: float, ell: float, n_steps: int = 2000,
               lam: float = 10.0) -> None:
    print(f"\n{'='*60}")
    print(f"  Single optimisation  eta={eta}  gamma={gamma}  ell={ell}")
    print(f"{'='*60}")

    lattice    = oam_lattice(ell)
    state_prep = GKPStatePrep(lattice, cutoff=DEFAULT_CUTOFF,
                               epsilon=DEFAULT_EPSILON)
    circuit    = SensingCircuit(state_prep, eta=eta, gamma=gamma)
    loss_fn    = CombinedLoss(circuit, lam=lam)
    opt        = Optimizer(circuit, loss_fn, lr=5e-3, log_interval=100)

    history = opt.train(n_steps=n_steps, verbose=True)

    # Post-training: project ell to nearest integer
    ell_int = lattice.discrete_ell()
    print(f"\n  Optimised ell = {ell_int}  "
          f"(theta = {np.degrees(float(lattice.theta.numpy())):.2f} deg)")

    # Validate with MC error rate
    _, info_analytic = loss_fn()
    loss_fn_mc = CombinedLoss(circuit, lam=lam, use_analytic_perr=False)
    _, info_mc = loss_fn_mc()
    print(f"  Analytic P_err = {info_analytic['P_err']:.4e}")
    print(f"  MC       P_err = {info_mc['P_err']:.4e}")

    # ====================== PLOTTING (High-quality PDF) ======================
    base_name = f"training_eta{eta:.2f}_gamma{gamma:.3f}_ell{ell:.1f}"

    plot_training_history(
        history,
        title=f"Training: eta={eta}, gamma={gamma}, ell={ell}",
        save_path=os.path.join(OUTPUT_DIR, f"{base_name}.pdf")
    )

    plot_wigner(
        state_prep,
        title=f"Optimised GKP state  eta={eta}, gamma={gamma}",
        save_path=os.path.join(OUTPUT_DIR, f"wigner_eta{eta:.2f}_gamma{gamma:.3f}_ell{ell:.1f}.pdf")
    )


# ─── Pareto frontier sweep ────────────────────────────────────────────────────

def run_pareto(eta: float, gamma: float, n_steps: int = 500) -> None:
    print(f"\n{'='*60}")
    print(f"  Pareto sweep  eta={eta}  gamma={gamma}")
    print(f"{'='*60}")

    lam_values = np.logspace(-1, 3, 15)   # 15 points from 0.1 to 1000

    def make_circuit(ell_init: float):
        def factory():
            lat  = oam_lattice(ell_init)
            sp   = GKPStatePrep(lat, cutoff=DEFAULT_CUTOFF, epsilon=DEFAULT_EPSILON)
            circ = SensingCircuit(sp, eta=eta, gamma=gamma)
            return circ
        return factory

    geometries = {
        "square"    : lambda: SensingCircuit(
                            GKPStatePrep(square_lattice()), eta=eta, gamma=gamma),
        "hexagonal" : lambda: SensingCircuit(
                            GKPStatePrep(hexagonal_lattice()), eta=eta, gamma=gamma),
        "oam_ell1"  : make_circuit(1.0),
        "oam_ell2"  : make_circuit(2.0),
    }

    results_by_geometry = {}
    for name, factory in geometries.items():
        print(f"\n  Geometry: {name}")
        results_by_geometry[name] = pareto_sweep(
            factory, lam_values, n_steps=n_steps, verbose=True
        )

    print_results_table(results_by_geometry)

    plot_pareto_frontier(
        results_by_geometry,
        save_path=os.path.join(OUTPUT_DIR, f"pareto_eta{eta:.2f}_gamma{gamma:.3f}.pdf")
    )


# ─── Sanity checks (verify mode) ─────────────────────────────────────────────

def run_verify() -> None:
    print("\n" + "="*60)
    print("  Verification / sanity checks")
    print("="*60)

    # 1. Symplecticity
    print("\n[1] Symplecticity check for all lattice types  (expect 2π ≈ 6.2832)")
    for name, lat in [("square", square_lattice()),
                       ("hexagonal", hexagonal_lattice()),
                       ("oam_ell1",  oam_lattice(1.0)),
                       ("oam_ell2",  oam_lattice(2.0))]:
        val = lat.verify_symplectic()
        ok  = abs(val - 2.0 * np.pi) < 1e-6
        print(f"  {name:<12}  u1^T Omega u2 = {val:.8f}  {'OK' if ok else 'FAIL'}")

    # 2. QFI scaling
    print("\n[2] QFI test for Fock states and superpositions")
    for n_fock in [1, 2, 5, 10]:
        cutoff = 30
        ket = np.zeros(cutoff, dtype=complex)
        ket[n_fock] = 1.0
        ket_tf = tf.constant(ket, dtype=tf.complex128)
        qfi_val = float(qfi_pure(ket_tf).numpy())

        ket2 = np.zeros(cutoff, dtype=complex)
        ket2[n_fock] = 1.0 / np.sqrt(2)
        ket2[min(n_fock + 1, cutoff-1)] = 1.0 / np.sqrt(2)
        ket2_tf = tf.constant(ket2, dtype=tf.complex128)
        qfi_super = float(qfi_pure(ket2_tf).numpy())

        print(f"  |{n_fock}> QFI = {qfi_val:.4f} (expect 0.0)   "
              f"  superposition QFI = {qfi_super:.4f} (expect 1.0)")

    # 3. Gradient flow check
    print("\n[3] Gradient flow check (d(QFI)/d(theta))")
    lat = oam_lattice(1.0)
    sp  = GKPStatePrep(lat, cutoff=20, epsilon=0.1)
    with tf.GradientTape() as tape:
        ket  = sp.prepare()
        qfi_v = qfi_pure(ket)
    grads = tape.gradient(qfi_v, lat.trainable_variables)
    for v, g in zip(lat.trainable_variables, grads):
        has_grad = g is not None and not tf.reduce_all(g == 0).numpy()
        print(f"  d(QFI)/d({v.name}) = {float(g.numpy()):.6f}  "
              f"{'OK — non-zero' if has_grad else 'WARNING — zero'}")

    print("\n  All checks complete.\n")


# ─── Argument parsing ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OAM-GKP quantum sensing optimiser")
    parser.add_argument("--mode",    choices=["single","pareto","diagram","verify"],
                        default="verify")
    parser.add_argument("--eta",     type=float, default=0.9,
                        help="photon-loss transmissivity [0,1]")
    parser.add_argument("--gamma",   type=float, default=0.05,
                        help="dephasing rate")
    parser.add_argument("--ell",     type=float, default=2.0,
                        help="OAM charge (for single mode)")
    parser.add_argument("--n_steps", type=int,   default=2000,
                        help="optimisation steps")
    parser.add_argument("--lam",     type=float, default=10.0,
                        help="Lagrange multiplier lambda")
    args = parser.parse_args()

    # Use float64 globally for better precision with symplectic backend
    tf.keras.backend.set_floatx("float64")

    if args.mode == "single":
        run_single(args.eta, args.gamma, args.ell, args.n_steps, args.lam)
    elif args.mode == "pareto":
        run_pareto(args.eta, args.gamma, args.n_steps)
    elif args.mode == "verify":
        run_verify()
    else:
        print("diagram mode: implement run_diagram() from utils.plot_phase_diagram()")


if __name__ == "__main__":
    main()
