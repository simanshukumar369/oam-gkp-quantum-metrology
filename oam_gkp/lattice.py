"""
oam_gkp/lattice.py
==================
OAM-to-lattice mapping and twisted GKP lattice construction.

All equations reference the paper:
  "Noise-Adaptive Quantum Phase Estimation via OAM-Encoded GKP States"
  Simanshu Kumar, 2026.

Paper Eq. (4)  : theta_ell = ell * pi / ell_max
Paper Eq. (6)  : twisted lattice vectors u1(theta, r), u2(theta, r)
Paper Eq. (7)  : symplecticity check  u1^T Omega u2 = 2
"""

import numpy as np
import tensorflow as tf

# ─── Constants ───────────────────────────────────────────────────────────────

# Standard GKP lattice spacing  a = sqrt(2 pi)
_A = tf.cast(tf.sqrt(2.0 * np.pi), tf.float64)

# Symplectic form  Omega = [[0, 1], [-1, 0]]
_OMEGA = tf.constant([[0.0, 1.0], [-1.0, 0.0]], dtype=tf.float64)


# ─── OAM-to-rotation mapping  (Paper Eq. 4) ──────────────────────────────────

def oam_to_angle(ell: tf.Tensor, ell_max: float = 4.0) -> tf.Tensor:
    """Convert OAM charge ell to lattice rotation angle theta.

    Implements Paper Eq. (4):
        theta_ell = ell * pi / ell_max

    Args:
        ell      : OAM charge, shape (). Treated as continuous during
                   gradient descent; round to nearest integer post-training.
        ell_max  : Maximum OAM charge supported by the optical system
                   (set by mode-field bandwidth).  Default 4 covers
                   ell in {0, 1, 2, 3, 4}.

    Returns:
        theta : Rotation angle in [0, pi), shape ().
    """
    ell   = tf.cast(ell, tf.float64)
    theta = ell * np.pi / ell_max
    return theta


def oam_to_frft_order(ell: tf.Tensor, ell_max: float = 4.0) -> tf.Tensor:
    """Fractional Fourier order alpha corresponding to OAM charge ell.

    alpha = 2 * theta / pi  (inverse of  theta = alpha * pi / 2).

    Args:
        ell, ell_max : as in oam_to_angle().

    Returns:
        alpha : fractional Fourier order in [0, 2), shape ().
    """
    theta = oam_to_angle(ell, ell_max)
    return 2.0 * theta / np.pi


# ─── Rotation matrix helper ───────────────────────────────────────────────────

def rotation_matrix(theta: tf.Tensor) -> tf.Tensor:
    """2x2 SO(2) rotation matrix for angle theta.

    R(theta) = [[cos theta, -sin theta],
                [sin theta,  cos theta]]

    Args:
        theta : rotation angle, shape ().

    Returns:
        R : shape (2, 2), dtype float64.
    """
    theta = tf.cast(theta, tf.float64)
    c = tf.cos(theta)
    s = tf.sin(theta)
    return tf.reshape(tf.stack([c, -s, s, c]), (2, 2))


# ─── Twisted lattice vectors  (Paper Eq. 6) ───────────────────────────────────

def twisted_lattice(
    theta: tf.Tensor,
    r: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute OAM-twisted GKP lattice vectors.

    Implements Paper Eq. (6):
        u1(theta, r) = R(theta) @ [a*r, 0]^T
        u2(theta, r) = R(theta) @ [0, a/r]^T

    where a = sqrt(2*pi) is the standard GKP lattice spacing.

    The symplecticity condition  u1^T Omega u2 = 2  (Paper Eq. 7) is
    preserved for all (theta, r) — see symplecticity_check() below.

    Args:
        theta : lattice rotation angle, shape ().
        r     : aspect ratio > 0, shape ().
                r = 1 gives a square (or rotated-square) lattice.
                r = (pi/3)^{1/4} gives a hexagonal-like lattice.

    Returns:
        u1 : first lattice vector,  shape (2,).
        u2 : second lattice vector, shape (2,).
    """
    theta = tf.cast(theta, tf.float64)
    r     = tf.cast(r, tf.float64)

    R  = rotation_matrix(theta)
    u1 = tf.linalg.matvec(R, tf.stack([_A * r,      tf.zeros((), dtype=tf.float64)]))
    u2 = tf.linalg.matvec(R, tf.stack([tf.zeros((), dtype=tf.float64), _A / r     ]))
    return u1, u2


def symplecticity_check(u1: tf.Tensor, u2: tf.Tensor) -> tf.Tensor:
    """Verify the GKP symplecticity condition u1^T Omega u2 == 2*pi.

    With the convention [q, p] = i (hbar = 1), the GKP stabilizers
    commute iff  u1^T Omega u2 = 2*pi  (a multiple of 2*pi).

    For the standard lattice:
        u1 = (sqrt(2*pi), 0),  u2 = (0, sqrt(2*pi))
        u1^T Omega u2 = sqrt(2*pi) * sqrt(2*pi) = 2*pi  ✓

    Returns the scalar  u1^T Omega u2  (should equal 2*pi ≈ 6.2832).
    """
    return tf.tensordot(u1, tf.linalg.matvec(_OMEGA, u2), axes=1)


# ─── Lattice geometry factory ─────────────────────────────────────────────────

class GKPLattice:
    """Container for a twisted GKP lattice defined by (ell, r, ell_max).

    The three canonical geometries from the paper:
        square     : ell=0, r=1
        hexagonal  : ell=0, r=(pi/3)^{1/4}  [optimal displacement correction]
        OAM-twisted: ell>0, r trainable

    Parameters
    ----------
    ell      : OAM charge (float for differentiability; integer post-training).
    r        : aspect ratio (float, > 0).
    ell_max  : bandwidth cap (default 4).
    """

    def __init__(
        self,
        ell: float = 0.0,
        r: float = 1.0,
        ell_max: float = 4.0,
    ):
        self.ell_max = ell_max

        # Trainable variables (tf.Variable for gradient tracking)
        self.ell_var = tf.Variable(float(ell), trainable=True, dtype=tf.float64,
                                   name="ell")
        self.r_var   = tf.Variable(float(r),   trainable=True, dtype=tf.float64,
                                   name="r")

    # ── Derived quantities ────────────────────────────────────────────────────

    @property
    def theta(self) -> tf.Tensor:
        """Rotation angle from OAM charge (Paper Eq. 4)."""
        return oam_to_angle(self.ell_var, self.ell_max)

    @property
    def r(self) -> tf.Tensor:
        """Aspect ratio (positive-enforced via softplus)."""
        return tf.nn.softplus(self.r_var) + 1e-3   # always > 0

    @property
    def vectors(self) -> tuple[tf.Tensor, tf.Tensor]:
        """Twisted lattice vectors u1, u2 (Paper Eq. 6)."""
        return twisted_lattice(self.theta, self.r)

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return [self.ell_var, self.r_var]

    # ── Squeezing parameter corresponding to aspect ratio ─────────────────────

    @property
    def squeezing_db(self) -> tf.Tensor:
        """Squeezing level in dB associated with aspect ratio r.

        The square GKP lattice at spacing a = sqrt(2*pi) requires squeezing
        s = a^2 / (2 * hbar).  For aspect ratio r, the q-quadrature is
        squeezed by log(r) (nats) = 10 * log10(r^2 / 1) dB.
        """
        return 10.0 * tf.math.log(self.r ** 2) / tf.math.log(10.0)

    # ── Nearest-integer projection for post-training ──────────────────────────

    def discrete_ell(self) -> int:
        """Round continuous ell to nearest integer (post-optimization)."""
        return int(tf.round(self.ell_var).numpy())

    # ── Symplecticity verification ────────────────────────────────────────────

    def verify_symplectic(self) -> float:
        """Return u1^T Omega u2; should be 2*pi ≈ 6.2832 for all (theta, r)."""
        u1, u2 = self.vectors
        val = symplecticity_check(u1, u2)
        return float(val.numpy())

    def __repr__(self) -> str:
        ell_v = float(self.ell_var.numpy())
        r_v   = float(self.r.numpy())
        th_v  = float(self.theta.numpy())
        return (f"GKPLattice(ell={ell_v:.3f}, r={r_v:.3f}, "
                f"theta={np.degrees(th_v):.2f} deg, "
                f"symplectic={self.verify_symplectic():.6f})")


# ─── Preset geometries ────────────────────────────────────────────────────────

def square_lattice() -> GKPLattice:
    """Standard square GKP lattice (ell=0, r=1)."""
    return GKPLattice(ell=0.0, r=1.0)


def hexagonal_lattice() -> GKPLattice:
    """Hexagonal GKP lattice — optimal for symmetric displacement noise.

    The hexagonal lattice has aspect ratio r = (pi/3)^{1/4} and rotation
    theta = pi/6, which is obtained from ell_max adjusted appropriately.
    Here we set theta directly via ell so that ell=1 maps to pi/6.
    """
    # We want theta = pi/6, so ell/ell_max = 1/6  =>  use ell_max=6, ell=1
    lat = GKPLattice(ell=1.0, r=(np.pi / 3) ** 0.25, ell_max=6.0)
    return lat


def oam_lattice(ell: float, ell_max: float = 4.0) -> GKPLattice:
    """OAM-twisted GKP lattice of charge ell."""
    return GKPLattice(ell=ell, r=1.0, ell_max=ell_max)
