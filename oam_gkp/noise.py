"""
oam_gkp/noise.py
================
Photon-loss and dephasing channel implementations.

Paper Eqs. (8) and (9):
    Loss channel    — Wigner convolution with symmetric Gaussian spread
                      sigma^2 = (1 - eta) / (2 eta) per quadrature.
    Dephasing       — Diffusion along the p-quadrature with variance gamma.

In Strawberry Fields the loss channel is applied per mode in the circuit.
The dephasing channel is implemented as a ThermalLossChannel or via a
custom Kraus map applied to the density matrix in Fock space.

Note on rotated-frame dephasing (Paper §3.2):
    In the OAM-rotated frame with angle theta, the dephasing Gaussian
    spreads along the direction (sin theta, cos theta)^T of the original
    (q, p) frame.  The twisted GKP lattice aligns its larger correction
    radius with this direction, reducing the effective logical error rate.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops


# ─── Loss channel (Paper Eq. 8) ──────────────────────────────────────────────

def apply_loss_channel(
    prog: sf.Program,
    q_reg,
    eta: tf.Tensor,
) -> None:
    """Apply photon-loss channel LossChannel(eta) inside an SF program context.

    The channel acts on the Wigner function as a symmetric Gaussian
    convolution (Paper Eq. 8):
        W_out(r) = (1 / pi (1-eta)) * integral exp(-|r - sqrt(eta)*s|^2 /
                   (1-eta)) W_in(s) d^2 s

    Args:
        prog  : active sf.Program (used for context management only).
        q_reg : qumode register entry, e.g. q[0].
        eta   : transmissivity in [0, 1].  eta=1 means no loss.
    """
    ops.LossChannel(eta) | q_reg


def apply_thermal_loss(
    prog: sf.Program,
    q_reg,
    eta: tf.Tensor,
    nbar: float = 0.0,
) -> None:
    """Apply thermal loss channel ThermalLossChannel(eta, nbar).

    Combines beam-splitter loss with thermal noise of mean photon number nbar.
    For nbar=0 this reduces to the pure loss channel above.

    Args:
        eta  : transmissivity in [0, 1].
        nbar : mean thermal photon number of the environment.
    """
    ops.ThermalLossChannel(eta, nbar) | q_reg


# ─── Dephasing channel (Paper Eq. 9) ─────────────────────────────────────────

def dephasing_kraus(
    gamma: tf.Tensor,
    cutoff: int,
) -> tf.Tensor:
    """Construct the Kraus operator sum for the dephasing channel.

    The dephasing channel in Fock space acts as:
        rho_out_{mn} = rho_in_{mn} * exp(-gamma/2 * (m - n)^2)

    This is equivalent to a random phase kick phi ~ N(0, gamma), i.e.
    a Gaussian diffusion along the p-quadrature in the Wigner picture
    (Paper Eq. 9).

    Args:
        gamma  : dephasing rate (variance of phase kick), shape ().
        cutoff : Fock-space dimension.

    Returns:
        factor : TF tensor of shape (cutoff, cutoff), real-valued,
                 giving the element-wise multiplicative factor on rho.
    """
    gamma = tf.cast(gamma, tf.float64)
    n = tf.cast(tf.range(cutoff), tf.float64)
    m = tf.cast(tf.range(cutoff), tf.float64)
    N, M = tf.meshgrid(n, m, indexing="ij")
    factor = tf.exp(-gamma / 2.0 * tf.square(N - M))
    return factor


def apply_dephasing(
    rho: tf.Tensor,
    gamma: tf.Tensor,
) -> tf.Tensor:
    """Apply dephasing channel to a density matrix in Fock space.

    Args:
        rho   : density matrix, shape (cutoff, cutoff), complex128.
        gamma : dephasing rate, shape ().

    Returns:
        rho_out : dephased density matrix, shape (cutoff, cutoff), complex128.
    """
    cutoff = rho.shape[0]
    factor = tf.cast(dephasing_kraus(gamma, cutoff), tf.complex128)
    return rho * factor


# ─── Displacement spread in rotated frame ────────────────────────────────────

def effective_spread(
    eta: float,
    gamma: float,
    theta: float,
) -> tuple[float, float]:
    """Compute the effective noise spread in the OAM-rotated GKP frame.

    The photon-loss channel induces symmetric spread:
        sigma_loss^2 = (1 - eta) / (2 * eta)  per quadrature.

    The dephasing channel induces asymmetric spread in the original frame
    (along p only).  In the rotated frame (tilted by theta), the effective
    spread along the u1-direction (q-like) and u2-direction (p-like) of
    the twisted lattice are:

        sigma_q_eff^2 = sigma_loss^2 + gamma * sin^2(theta)
        sigma_p_eff^2 = sigma_loss^2 + gamma * cos^2(theta)

    A lattice with larger correction radius along u2 (r > 1) benefits from
    this asymmetry when sigma_p_eff > sigma_q_eff, i.e. when theta < pi/4.

    Args:
        eta   : transmissivity.
        gamma : dephasing rate.
        theta : OAM rotation angle in radians.

    Returns:
        (sigma_q_eff, sigma_p_eff) : effective displacement spreads.
    """
    sigma_loss_sq = (1.0 - eta) / (2.0 * eta + 1e-12)
    sigma_q = np.sqrt(sigma_loss_sq + gamma * np.sin(theta) ** 2)
    sigma_p = np.sqrt(sigma_loss_sq + gamma * np.cos(theta) ** 2)
    return sigma_q, sigma_p


def optimal_aspect_ratio(
    eta: float,
    gamma: float,
    theta: float,
) -> float:
    """Compute the analytically optimal aspect ratio r* given noise parameters.

    The GKP code corrects displacements up to +/- d_j / 2, where d_j is the
    lattice spacing in the j-th direction.  For our twisted lattice:
        d_q = a * r,   d_p = a / r.

    The logical error rate is minimised when the correction radii are
    proportional to the effective spreads:
        d_q / d_p  = sigma_q_eff / sigma_p_eff
        (a*r) / (a/r) = r^2  =>  r* = sqrt(sigma_q_eff / sigma_p_eff)

    Args:
        eta, gamma, theta : as in effective_spread().

    Returns:
        r_star : optimal aspect ratio.
    """
    sigma_q, sigma_p = effective_spread(eta, gamma, theta)
    if sigma_p < 1e-12:
        return 1.0
    r_star = (sigma_q / sigma_p) ** 0.25   # r^2 = sigma_q/sigma_p => r = (...)^{1/2}
    return float(r_star)
