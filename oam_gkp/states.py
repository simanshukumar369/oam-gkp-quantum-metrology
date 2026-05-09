"""
oam_gkp/states.py
=================
Twisted GKP state preparation — two-stage architecture.

Why two stages?
---------------
SF's TFBackend does NOT implement prepare_gkp (removed in SF >= 0.20).
The Fock backend DOES implement it.  We therefore split the circuit:

    Stage 1  (Fock backend, numpy, non-differentiable through GKP params)
        GKP(state, epsilon)  →  base_ket   [square-lattice GKP codeword]

    Stage 2  (pure TF matrix ops, FULLY differentiable)
        Sgate(ln r)          →  apply squeezing matrix  S(s)  in Fock space
        Rgate(theta)         →  apply rotation  diag(exp(i n theta))

Gradients flow through theta (ell) and r automatically via TF autograd.
epsilon and Bloch angles are re-optimised by re-running Stage 1 whenever
their cached values change — cheap, since the Fock engine is fast for
single-mode cutoff ~30-50.

Fock-space operators
--------------------
Rgate(theta) :  U_{mn} = exp(i n theta) delta_{mn}   [diagonal, exact]
Sgate(s)     :  S = expm(s * H),  H = (a_dag^2 - a^2) / 2
                computed via tf.linalg.expm, exact and differentiable through s.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops

from .lattice import GKPLattice


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_CUTOFF   = 30       # fast for testing; use 50 for paper results
DEFAULT_EPSILON  = 0.0631   # ~10 dB squeezing
DEFAULT_AMPL_CUT = 1e-4


# ─── Fock-space operator builders ────────────────────────────────────────────

def annihilation_matrix(cutoff: int) -> tf.Tensor:
    """Annihilation operator a in Fock space.

    a_{m,n} = sqrt(n) * delta_{m, n-1}
    Shape: (cutoff, cutoff), complex128.
    """
    vals = tf.cast(
        tf.sqrt(tf.cast(tf.range(1, cutoff), tf.float64)),
        tf.complex128
    )
    return tf.linalg.diag(vals, k=-1)


def rgate_matrix(theta: tf.Tensor, cutoff: int) -> tf.Tensor:
    """Fock-space matrix of Rgate(theta) = exp(i theta n_hat).

    U_{mn} = exp(i n theta) delta_{mn}  — diagonal, exact.
    Fully differentiable w.r.t. theta.
    """
    theta_c = tf.cast(theta, tf.complex128)
    n       = tf.cast(tf.range(cutoff), tf.complex128)
    phases  = tf.exp(1j * theta_c * n)
    return tf.linalg.diag(phases)


def sgate_matrix(s: tf.Tensor, cutoff: int) -> tf.Tensor:
    """Fock-space matrix of Sgate(s) via matrix exponential.

    S(s) = expm(s * H),   H = (a_dag^2 - a^2) / 2

    SF convention: Sgate(s) squeezes the q quadrature by e^{-s}.
    tf.linalg.expm is differentiable through s.

    Args:
        s      : squeezing parameter (= ln r for aspect ratio r), float64 scalar.
        cutoff : Fock dimension.

    Returns:
        S_mat : shape (cutoff, cutoff), complex128.
    """
    s_c  = tf.cast(s, tf.complex128)
    a    = annihilation_matrix(cutoff)
    adag = tf.linalg.adjoint(a)
    H    = (tf.linalg.matmul(adag, adag) - tf.linalg.matmul(a, a)) / 2.0
    return tf.linalg.expm(s_c * H)


# ─── Stage 1: GKP base state via Fock backend ────────────────────────────────

def _prepare_gkp_fock(
    bloch_theta: float,
    bloch_phi:   float,
    epsilon:     float,
    cutoff:      int,
    ampl_cutoff: float = DEFAULT_AMPL_CUT,
) -> np.ndarray:
    """Prepare the square-lattice GKP logical state using the SF Fock backend.

    SF's GKP op signature:
        ops.GKP(state=[bloch_theta, bloch_phi], epsilon=..., ampl_cutoff=...)
    where state encodes |psi_L> = cos(t/2)|0_L> + exp(i*phi)*sin(t/2)|1_L>.

    Returns ket as numpy complex128 array of shape (cutoff,).
    """
    eng  = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    prog = sf.Program(1)

    with prog.context as q:
        ops.GKP(
            state       = [float(bloch_theta), float(bloch_phi)],
            epsilon     = float(epsilon),
            ampl_cutoff = ampl_cutoff,
        ) | q[0]

    result = eng.run(prog)
    ket    = result.state.ket().astype(np.complex128)

    # Renormalise after Fock truncation
    norm = np.linalg.norm(ket)
    return ket / norm if norm > 1e-10 else ket


# ─── Main class ──────────────────────────────────────────────────────────────

class GKPStatePrep:
    """Prepare a twisted finite-energy GKP state.

    Stage 1  — Fock backend: GKP(bloch_theta, bloch_phi, epsilon) → base_ket
    Stage 2  — TF matrix ops: base_ket → S(ln r) → R(theta) → twisted_ket

    Parameters
    ----------
    lattice     : GKPLattice (trainable ell_var, r_var).
    bloch_theta : Bloch polar angle for logical qubit.
    bloch_phi   : Bloch azimuthal angle.
    epsilon     : finite-energy envelope width (> 0).
    cutoff      : Fock space dimension.
    """

    def __init__(
        self,
        lattice:     GKPLattice,
        bloch_theta: float = 0.0,
        bloch_phi:   float = 0.0,
        epsilon:     float = DEFAULT_EPSILON,
        cutoff:      int   = DEFAULT_CUTOFF,
        ampl_cutoff: float = DEFAULT_AMPL_CUT,
    ):
        self.lattice     = lattice
        self.cutoff      = cutoff
        self.ampl_cutoff = ampl_cutoff

        # ── Trainable variables ───────────────────────────────────────────────

        self.bloch_theta_var = tf.Variable(
            float(bloch_theta), trainable=True,
            dtype=tf.float64, name="bloch_theta"
        )
        self.bloch_phi_var = tf.Variable(
            float(bloch_phi), trainable=True,
            dtype=tf.float64, name="bloch_phi"
        )

        # epsilon > 0 enforced via softplus
        _raw = float(np.log(max(np.exp(float(epsilon)) - 1.0, 1e-8)))
        self._epsilon_raw = tf.Variable(
            _raw, trainable=True, dtype=tf.float64, name="epsilon_raw"
        )

        # ── Stage 1 cache ─────────────────────────────────────────────────────
        self._base_ket_np:  np.ndarray | None = None
        self._cache_key:    tuple | None      = None

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def epsilon(self) -> tf.Tensor:
        return tf.nn.softplus(self._epsilon_raw) + 1e-4

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return (self.lattice.trainable_variables
                + [self.bloch_theta_var, self.bloch_phi_var, self._epsilon_raw])

    # ── Stage 1: cached Fock-backend GKP prep ────────────────────────────────

    def _get_base_ket(self) -> tf.Tensor:
        """Return base GKP ket as a TF constant (re-runs Fock backend on change)."""
        bt  = round(float(self.bloch_theta_var.numpy()), 4)
        bp  = round(float(self.bloch_phi_var.numpy()),   4)
        eps = round(float(self.epsilon.numpy()),         4)
        key = (bt, bp, eps, self.cutoff)

        if key != self._cache_key:
            self._base_ket_np = _prepare_gkp_fock(
                bt, bp, eps, self.cutoff, self.ampl_cutoff
            )
            self._cache_key = key

        return tf.constant(self._base_ket_np, dtype=tf.complex128)

    # ── Stage 2: differentiable Sgate + Rgate ────────────────────────────────

    def prepare(self) -> tf.Tensor:
        """Compute the twisted GKP ket.

        Differentiable w.r.t. lattice.ell_var (→ theta) and lattice.r_var (→ r → s).

        Circuit:  base_ket  →  S(ln r)  →  R(theta)  →  output

        Returns: shape (cutoff,), complex128.
        """
        base_ket = self._get_base_ket()                    # constant (cutoff,)

        s     = tf.math.log(self.lattice.r + 1e-8)        # float64 scalar
        theta = self.lattice.theta                         # float64 scalar

        S_mat = sgate_matrix(s,     self.cutoff)           # (cutoff, cutoff)
        R_mat = rgate_matrix(theta, self.cutoff)           # (cutoff, cutoff)

        ket = tf.linalg.matvec(S_mat, base_ket)
        ket = tf.linalg.matvec(R_mat, ket)

        # Renormalise (expm truncation error)
        norm = tf.cast(tf.sqrt(tf.reduce_sum(tf.abs(ket) ** 2)), tf.complex128)
        return ket / (norm + tf.cast(1e-12, tf.complex128))

    def density_matrix(self) -> tf.Tensor:
        """rho = |psi><psi|, shape (cutoff, cutoff), complex128."""
        ket = self.prepare()
        return tf.einsum("i,j->ij", ket, tf.math.conj(ket))

    # ── Wigner function (via Fock backend, for plotting only) ─────────────────

    def wigner(
        self,
        q_range: tuple = (-6.0, 6.0),
        p_range: tuple = (-6.0, 6.0),
        n_pts:   int   = 80,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute W(q,p) for the current twisted state."""
        ket_np = self.prepare().numpy()

        eng  = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff})
        prog = sf.Program(1)
        with prog.context as q_reg:
            ops.Ket(ket_np) | q_reg[0]
        result = eng.run(prog)

        q_arr = np.linspace(*q_range, n_pts)
        p_arr = np.linspace(*p_range, n_pts)
        Q, P  = np.meshgrid(q_arr, p_arr)
        W     = result.state.wigner(0, q_arr, p_arr)
        return Q, P, W
