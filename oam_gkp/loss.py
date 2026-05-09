"""
oam_gkp/loss.py
===============
Combined sensitivity + fault-tolerance loss function  (Paper Eq. 11):

    L(xi) = -F_Q(phi; xi) + lambda * [P_err(xi) - P_thresh]_+

where:
    F_Q      — quantum Fisher information (maximised)
    P_err    — logical error rate under one GKP correction cycle
    P_thresh — fault-tolerance threshold (default 1e-3)
    lambda   — Lagrange multiplier (swept for Pareto frontier)

The logical error rate P_err is estimated analytically from the effective
displacement spread and the GKP correction radius.  A Monte Carlo estimator
is also provided for validation.
"""

import numpy as np
import tensorflow as tf

from .circuit import SensingCircuit
from .noise   import effective_spread


# ─── GKP logical error rate  (analytic approximation) ────────────────────────

def gkp_logical_error_rate(
    sigma_q: float,
    sigma_p: float,
    r: tf.Tensor,
    a: float = float(np.sqrt(2.0 * np.pi)),
) -> tf.Tensor:
    """Estimate GKP logical error rate from effective displacement spreads.

    The GKP code corrects displacements in the q-direction up to +/- d_q/2
    and in the p-direction up to +/- d_p/2, where:
        d_q = a * r  (half-integer lattice spacing in q)
        d_p = a / r  (half-integer lattice spacing in p)

    Under Gaussian displacement noise with standard deviations (sigma_q, sigma_p),
    the probability of an uncorrectable error is approximately:

        P_err ~  2 * Q(d_q / (2 sigma_q))
               + 2 * Q(d_p / (2 sigma_p))

    where Q(x) = erfc(x / sqrt(2)) / 2 is the Q-function (tail probability).

    This is the leading-order approximation; higher-order lattice images are
    neglected (valid when d >> sigma).

    Args:
        sigma_q, sigma_p : effective displacement spreads (from noise.py).
        r                : aspect ratio (TF tensor, differentiable).
        a                : base lattice spacing sqrt(2 pi).

    Returns:
        P_err : scalar tf.float64.
    """
    r_f    = tf.cast(r, tf.float64)
    d_q    = a * r_f           # correction diameter in q
    d_p    = a / (r_f + 1e-8)  # correction diameter in p

    # Q-function via erfc:  Q(x) = erfc(x/sqrt(2)) / 2
    def Q_func(d, sigma):
        sigma_f = tf.cast(sigma + 1e-12, tf.float64)
        x = d / (2.0 * sigma_f * tf.cast(tf.sqrt(2.0), tf.float64))
        return 0.5 * tf.math.erfc(x)

    P_q = 2.0 * Q_func(d_q, sigma_q)
    P_p = 2.0 * Q_func(d_p, sigma_p)
    P_err = P_q + P_p - P_q * P_p     # inclusion-exclusion (independent axes)
    return tf.cast(P_err, tf.float64)


# ─── Combined loss  (Paper Eq. 11) ────────────────────────────────────────────

class CombinedLoss:
    """Evaluates L(xi) = -QFI + lambda * [P_err - P_thresh]_+

    Parameters
    ----------
    circuit    : SensingCircuit instance.
    lam        : Lagrange multiplier lambda (sweep for Pareto frontier).
    P_thresh   : fault-tolerance threshold (default 1e-3).
    use_analytic_perr : if True, use the analytic GKP error rate estimate;
                        if False, run a Monte Carlo simulation (slower).
    """

    def __init__(
        self,
        circuit: SensingCircuit,
        lam: float = 10.0,
        P_thresh: float = 1e-3,
        use_analytic_perr: bool = True,
    ):
        self.circuit      = circuit
        self.lam          = tf.constant(lam,      dtype=tf.float64)
        self.P_thresh     = tf.constant(P_thresh, dtype=tf.float64)
        # use_analytic is mutable — optimizer switches it in multi-fidelity mode
        self.use_analytic = use_analytic_perr

    def __call__(self) -> tuple[tf.Tensor, dict]:
        """Compute the combined loss and return a diagnostics dict.

        Returns
        -------
        loss    : scalar tf.float64.
        info    : dict with keys: qfi, P_err, qfi_term, penalty_term.
        """
        # ── QFI term ─────────────────────────────────────────────────────────
        qfi = self.circuit.qfi()

        # ── Logical error rate term ───────────────────────────────────────────
        if self.use_analytic:
            P_err = self._analytic_perr()
        else:
            P_err = self._mc_perr()

        # ── Penalty  [P_err - P_thresh]_+ ────────────────────────────────────
        penalty = tf.nn.relu(P_err - self.P_thresh)

        # ── Combined loss ─────────────────────────────────────────────────────
        qfi_term     = -qfi
        penalty_term = self.lam * penalty
        loss = qfi_term + penalty_term

        info = {
            "qfi":          float(qfi.numpy()),
            "P_err":        float(P_err.numpy()),
            "qfi_term":     float(qfi_term.numpy()),
            "penalty_term": float(penalty_term.numpy()),
            "loss":         float(loss.numpy()),
        }
        return loss, info

    def _analytic_perr(self) -> tf.Tensor:
        """Analytic GKP logical error rate from effective spreads."""
        lattice = self.circuit.state_prep.lattice
        theta_v = float(lattice.theta.numpy())
        r_v     = lattice.r                       # differentiable TF tensor

        eta_v   = float(self.circuit.eta.numpy())
        gamma_v = float(self.circuit.gamma.numpy())

        sigma_q, sigma_p = effective_spread(eta_v, gamma_v, theta_v)
        return gkp_logical_error_rate(sigma_q, sigma_p, r_v)

    def _mc_perr(self, n_trials: int = 2000) -> tf.Tensor:
        """Monte Carlo estimate of GKP logical error rate.

        Samples n_trials random displacement errors from the effective
        Gaussian distribution and counts fraction exceeding the GKP
        correction radius.

        Note: this is not differentiable w.r.t. lattice parameters.
        Use only for validation after training.
        """
        lattice = self.circuit.state_prep.lattice
        theta_v = float(lattice.theta.numpy())
        r_v     = float(lattice.r.numpy())
        eta_v   = float(self.circuit.eta.numpy())
        gamma_v = float(self.circuit.gamma.numpy())

        sigma_q, sigma_p = effective_spread(eta_v, gamma_v, theta_v)
        a  = float(np.sqrt(2.0 * np.pi))
        d_q, d_p = a * r_v, a / (r_v + 1e-8)

        dq = np.random.normal(0, sigma_q, n_trials)
        dp = np.random.normal(0, sigma_p, n_trials)

        error = (np.abs(dq) > d_q / 2) | (np.abs(dp) > d_p / 2)
        P_err_mc = float(np.mean(error))
        return tf.constant(P_err_mc, dtype=tf.float64)


# ─── Pareto frontier sweep ────────────────────────────────────────────────────

def pareto_sweep(
    circuit_factory,           # callable: () -> SensingCircuit (fresh circuit)
    lam_values: np.ndarray,
    n_steps: int = 500,
    lr: float = 5e-3,
    verbose: bool = True,
) -> list[dict]:
    """Sweep lambda to map the QFI–P_err Pareto frontier.

    For each lambda in lam_values:
        1. Instantiate a fresh circuit.
        2. Optimise with CombinedLoss for n_steps steps.
        3. Record (QFI, P_err, xi*) at convergence.

    Returns
    -------
    results : list of dicts, one per lambda value, with keys:
              lambda, qfi, P_err, ell, r, theta_deg, epsilon.
    """
    from .optimizer import Optimizer

    results = []
    for i, lam in enumerate(lam_values):
        if verbose:
            print(f"[Pareto] lambda = {lam:.2e}  ({i+1}/{len(lam_values)})")

        circuit = circuit_factory()
        loss_fn = CombinedLoss(circuit, lam=float(lam))
        opt     = Optimizer(circuit, loss_fn, lr=lr)
        opt.train(n_steps=n_steps, verbose=False)

        _, info = loss_fn()
        lattice = circuit.state_prep.lattice
        results.append({
            "lambda":    float(lam),
            "qfi":       info["qfi"],
            "P_err":     info["P_err"],
            "ell":       float(lattice.ell_var.numpy()),
            "ell_int":   lattice.discrete_ell(),
            "r":         float(lattice.r.numpy()),
            "theta_deg": float(np.degrees(lattice.theta.numpy())),
            "epsilon":   float(circuit.state_prep.epsilon.numpy()),
        })

        if verbose:
            print(f"         QFI={info['qfi']:.4f}  "
                  f"P_err={info['P_err']:.2e}")

    return results
