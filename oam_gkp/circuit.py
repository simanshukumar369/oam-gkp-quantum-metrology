"""
oam_gkp/circuit.py
==================
Full parameterised sensing circuit  (Paper Eq. 10):

    rho(phi; xi) = E_gamma ∘ E_eta [ R(phi) rho_GKP(xi) R^dag(phi) ]

where:
    rho_GKP(xi)  — twisted GKP state with trainable params xi = (theta, r, eps, L)
    R(phi)       — phase-encoding rotation exp(-i phi n_hat)
    E_eta        — photon-loss channel (transmissivity eta)
    E_gamma      — dephasing channel (rate gamma)

The circuit is fully differentiable: all parameters in xi flow gradients
through the Strawberry Fields TF backend.

Homodyne LO angle psi is also trainable (Paper §3.5, Eq. 12).
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops

from .states import GKPStatePrep
from .noise  import apply_dephasing
from .qfi    import qfi_mixed, cfi_homodyne, normalised_qfi, number_operator


# ─── Sensing circuit ──────────────────────────────────────────────────────────

class SensingCircuit:
    """Full differentiable quantum sensing circuit for phase estimation.

    Parameters
    ----------
    state_prep : GKPStatePrep instance (holds all trainable state parameters).
    eta        : photon-loss transmissivity in [0, 1].  Can be tf.Variable
                 for meta-learning, or a fixed float.
    gamma      : dephasing rate.  Same flexibility.
    phi        : encoded phase (the parameter to estimate). Fixed during
                 QFI evaluation; scanned for homodyne calibration.
    psi        : homodyne LO angle (trainable).
    """

    def __init__(
        self,
        state_prep: GKPStatePrep,
        eta:   float = 0.9,
        gamma: float = 0.05,
        phi:   float = np.pi / 4,
        psi:   float = 0.0,
    ):
        self.state_prep = state_prep
        self.cutoff     = state_prep.cutoff

        # Noise parameters (fixed during optimisation; swept for phase diagram)
        self.eta   = tf.constant(eta,   dtype=tf.float64)
        self.gamma = tf.constant(gamma, dtype=tf.float64)
        self.phi   = tf.constant(phi,   dtype=tf.float64)

        # Trainable LO angle (Paper §3.5)
        self.psi_var = tf.Variable(float(psi), trainable=True,
                                   dtype=tf.float64, name="psi_lo")

        # SF engine (reused across forward passes)
        self._engine = sf.Engine(
            backend="tf",
            backend_options={"cutoff_dim": self.cutoff}
        )

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return self.state_prep.trainable_variables + [self.psi_var]

    # ── Forward pass ─────────────────────────────────────────────────────────

    def run(self) -> tf.Tensor:
        """Execute circuit and return the noisy density matrix rho(phi; xi).

        Steps:
            1. Prepare twisted GKP state  |psi_GKP(xi)>
            2. Apply phase encoding       R(phi) = exp(-i phi n_hat)
            3. Apply photon-loss channel  E_eta  (via SF LossChannel)
            4. Apply dephasing channel    E_gamma (via Fock-space Kraus map)

        Returns
        -------
        rho : density matrix, shape (cutoff, cutoff), complex128.
        """
        ket = self.state_prep.prepare()    # shape (cutoff,), complex128
        rho = tf.einsum("i,j->ij", ket, tf.math.conj(ket))   # pure state DM

        # Step 2 — phase encoding  R(phi) rho R^dag(phi) in Fock space
        rho = self._apply_phase_encoding(rho, self.phi)

        # Step 3 — photon-loss channel via SF (acts on the ket then reforms rho)
        rho = self._apply_loss_fock(rho, self.eta)

        # Step 4 — dephasing in Fock space (Kraus map, Paper Eq. 9)
        rho = apply_dephasing(rho, self.gamma)

        return rho

    # ── Cached QFI and CFI ────────────────────────────────────────────────────

    def qfi(self) -> tf.Tensor:
        """QFI of the output state for phase phi (Paper Eq. 3)."""
        rho = self.run()
        return qfi_mixed(rho)

    def cfi(self) -> tf.Tensor:
        """CFI of adaptive homodyne measurement (Paper Eq. 12)."""
        rho = self.run()
        return cfi_homodyne(rho, self.phi, self.psi_var)

    def mean_photon_number(self) -> tf.Tensor:
        """Mean photon number of the output state."""
        rho    = self.run()
        n_hat  = number_operator(self.cutoff)
        n_real = tf.math.real(tf.linalg.trace(
            tf.linalg.matmul(rho, n_hat)
        ))
        return tf.cast(n_real, tf.float64)

    def normalised_qfi(self) -> tf.Tensor:
        """QFI / (4 n_bar^2) — fraction of Heisenberg limit."""
        q  = self.qfi()
        nb = self.mean_photon_number()
        return normalised_qfi(q, nb)

    # ── Noise channel helpers in Fock space ───────────────────────────────────

    @staticmethod
    def _apply_phase_encoding(rho: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        """Apply R(phi) = exp(-i phi n_hat) to rho in Fock space.

        rho_out_{mn} = rho_{mn} * exp(-i phi (m - n))
        """
        cutoff = rho.shape[0]
        n = tf.cast(tf.range(cutoff), tf.complex128)
        m = tf.cast(tf.range(cutoff), tf.complex128)
        N, M = tf.meshgrid(n, m, indexing="ij")
        phi_c = tf.cast(phi, tf.complex128)
        phase_factor = tf.exp(-1j * phi_c * (N - M))
        return rho * phase_factor

    @staticmethod
    def _apply_loss_fock(rho: tf.Tensor, eta: tf.Tensor) -> tf.Tensor:
        """Apply photon-loss channel to rho in Fock space.

        Implements the Kraus sum:
            rho_out = sum_k K_k rho K_k^dag
        where K_k are the beam-splitter Kraus operators:
            <m|K_k|n> = sqrt(C(n,k)) eta^{(n-k)/2} (1-eta)^{k/2} delta_{m, n-k}

        This is the exact matrix representation of the loss channel in the
        truncated Fock space.

        Args:
            rho : density matrix, shape (cutoff, cutoff), complex128.
            eta : transmissivity, scalar float64.

        Returns:
            rho_out : shape (cutoff, cutoff), complex128.
        """
        cutoff = rho.shape[0]
        eta_c  = tf.cast(eta, tf.complex128)
        one_m_eta = tf.cast(1.0 - eta, tf.complex128)

        rho_out = tf.zeros_like(rho)
        for k in range(cutoff):
            K = _loss_kraus(k, cutoff, eta_c, one_m_eta)
            rho_out = rho_out + tf.linalg.matmul(
                K, tf.linalg.matmul(rho, tf.linalg.adjoint(K))
            )
        return rho_out


# ─── Kraus operator for loss channel ─────────────────────────────────────────

def _loss_kraus(
    k: int,
    cutoff: int,
    eta: tf.Tensor,
    one_m_eta: tf.Tensor,
) -> tf.Tensor:
    """Construct the k-th Kraus operator for the beam-splitter loss channel.

    K_k[m, n] = sqrt(C(n, k)) * eta^{(n-k)/2} * (1-eta)^{k/2} * delta_{m, n-k}

    Shape: (cutoff, cutoff), complex128.
    """
    rows, cols = [], []
    vals = []

    for n in range(k, cutoff):
        m   = n - k
        # Binomial coefficient C(n, k)
        binom = _log_binom(n, k)
        # log amplitude = 0.5 * log C(n,k) + (n-k)/2 * log(eta) + k/2 * log(1-eta)
        log_amp = (0.5 * binom
                   + 0.5 * tf.cast(n - k, tf.complex128) * tf.math.log(eta + 1e-30)
                   + 0.5 * tf.cast(k, tf.complex128) * tf.math.log(one_m_eta + 1e-30))
        rows.append(m)
        cols.append(n)
        vals.append(tf.exp(log_amp))

    K = tf.zeros((cutoff, cutoff), dtype=tf.complex128)
    if not vals:
        return K

    # Scatter into dense matrix
    indices = tf.constant(list(zip(rows, cols)), dtype=tf.int64)
    vals_t  = tf.stack(vals)
    K       = tf.tensor_scatter_nd_update(K, indices, vals_t)
    return K


def _log_binom(n: int, k: int) -> tf.Tensor:
    """Log of binomial coefficient C(n, k) via log-gamma."""
    import math
    val = (math.lgamma(n + 1)
           - math.lgamma(k + 1)
           - math.lgamma(n - k + 1))
    return tf.cast(val, tf.complex128)
