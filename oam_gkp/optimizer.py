"""
oam_gkp/optimizer.py
====================
Adam optimiser with gradient clipping, cosine annealing LR schedule,
spectral gradient regularisation, and multi-fidelity P_err estimation.

Techniques used (Paper §4.3):
    • Adam optimiser with gradient clipping at global norm 1.0
    • Cosine annealing learning rate schedule over n_steps
    • Pattern-weighted loss annealing for gradient stability
    • Gradient norm logging for diagnosing vanishing/exploding gradients
    • Multi-fidelity optimisation: analytic P_err for first (1-mc_fraction)
      steps, Monte Carlo P_err for the final mc_fraction of steps.
      This tightens the 10-25% analytic uncertainty near convergence
      at minimal extra cost (MC only runs when near the optimum).

Training loop
    The loop calls circuit.run() inside a tf.GradientTape, computes
    loss, and updates all trainable variables in a single Adam step.
"""

import numpy as np
import tensorflow as tf
import time

from .circuit import SensingCircuit
from .loss    import CombinedLoss


# ─── Learning-rate schedule ───────────────────────────────────────────────────

def cosine_lr(step: int, n_steps: int, lr_init: float, lr_min: float = 1e-5) -> float:
    """Cosine annealing schedule.

    lr(t) = lr_min + 0.5*(lr_init - lr_min)*(1 + cos(pi * t / T))
    """
    frac = step / max(n_steps - 1, 1)
    return lr_min + 0.5 * (lr_init - lr_min) * (1.0 + np.cos(np.pi * frac))


# ─── Pattern-weighted loss annealing ─────────────────────────────────────────

def loss_weight(step: int, n_steps: int, warmup: int = 50) -> float:
    """Linear warmup over first `warmup` steps, then constant 1.0.

    Prevents large early gradients from destabilising the GKP state
    preparation (observed in NOON-state context; carried over here).
    """
    if step < warmup:
        return (step + 1) / warmup
    return 1.0


# ─── Optimizer class ──────────────────────────────────────────────────────────

class Optimizer:
    """Gradient-based optimizer for the OAM-GKP sensing circuit.

    Parameters
    ----------
    circuit      : SensingCircuit instance.
    loss_fn      : CombinedLoss instance.
    lr           : initial learning rate (default 5e-3).
    clip_norm    : global gradient clipping threshold (default 1.0).
    n_steps      : total training steps.
    log_interval : print diagnostics every this many steps.
    """

    def __init__(
        self,
        circuit: SensingCircuit,
        loss_fn: CombinedLoss,
        lr: float = 5e-3,
        clip_norm: float = 1.0,
        log_interval: int = 100,
    ):
        self.circuit      = circuit
        self.loss_fn      = loss_fn
        self.lr_init      = lr
        self.clip_norm    = clip_norm
        self.log_interval = log_interval

        # Adam optimiser; lr will be updated per step
        self._opt = tf.optimizers.Adam(learning_rate=lr)

        # Multi-fidelity: fraction of steps that use MC P_err
        # 0.0 = always analytic (fast), 1.0 = always MC (slow)
        # Default: last 10% of steps switch to MC for tighter bounds
        self._mc_fraction = 0.10

        # History for plotting / diagnostics
        self.history: list[dict] = []

    # ── Single gradient step ──────────────────────────────────────────────────

    def _step(self, step: int, n_steps: int) -> dict:
        """Perform one Adam update and return diagnostics."""
        # Cosine annealing LR
        lr_now = cosine_lr(step, n_steps, self.lr_init)
        self._opt.learning_rate.assign(lr_now)

        # Pattern-weighted warmup
        w = loss_weight(step, n_steps)

        # Multi-fidelity: switch to MC P_err in final mc_fraction of steps
        use_mc = (step >= int((1.0 - self._mc_fraction) * n_steps))
        if use_mc and hasattr(self.loss_fn, '_use_mc_override'):
            self.loss_fn._use_mc_override = True

        with tf.GradientTape() as tape:
            if use_mc:
                # Temporarily use MC estimator for tighter P_err bound
                orig = self.loss_fn.use_analytic
                self.loss_fn.use_analytic = False
                loss, info = self.loss_fn()
                self.loss_fn.use_analytic = orig
            else:
                loss, info = self.loss_fn()
            loss_w = tf.cast(w, tf.float64) * loss

        vars_ = self.circuit.trainable_variables
        grads = tape.gradient(loss_w, vars_)

        # Gradient clipping (Paper §4.3)
        grads_clipped, global_norm = tf.clip_by_global_norm(grads, self.clip_norm)

        # Filter out None gradients (can occur if a variable is unused)
        pairs = [(g, v) for g, v in zip(grads_clipped, vars_) if g is not None]
        if pairs:
            self._opt.apply_gradients(pairs)

        # Aspect ratio positivity: project r_var if needed
        lattice = self.circuit.state_prep.lattice
        if lattice.r_var.numpy() < -10.0:
            lattice.r_var.assign(-10.0)

        info["step"]       = step
        info["lr"]         = lr_now
        info["grad_norm"]  = float(global_norm.numpy())
        info["weight"]     = w
        return info

    # ── Full training loop ────────────────────────────────────────────────────

    def train(
        self,
        n_steps: int = 2000,
        verbose: bool = True,
    ) -> list[dict]:
        """Run the full training loop.

        Args:
            n_steps : number of gradient steps.
            verbose : if True, print progress every log_interval steps.

        Returns
        -------
        history : list of per-step diagnostic dicts.
        """
        t0 = time.time()
        self.history = []

        for step in range(n_steps):
            info = self._step(step, n_steps)
            self.history.append(info)

            if verbose and (step % self.log_interval == 0 or step == n_steps - 1):
                elapsed = time.time() - t0
                lattice = self.circuit.state_prep.lattice
                print(
                    f"  step {step:4d}/{n_steps} | "
                    f"loss={info['loss']:+.4f} | "
                    f"QFI={info['qfi']:.4f} | "
                    f"P_err={info['P_err']:.2e} | "
                    f"ell={float(lattice.ell_var.numpy()):.3f} | "
                    f"r={float(lattice.r.numpy()):.3f} | "
                    f"|grad|={info['grad_norm']:.3f} | "
                    f"lr={info['lr']:.2e} | "
                    f"t={elapsed:.1f}s"
                )

        if verbose:
            print(f"\n  Training complete in {time.time()-t0:.1f}s")
            self._print_summary()

        return self.history

    def _print_summary(self):
        """Print final parameter values after training."""
        state  = self.circuit.state_prep
        lattice = state.lattice
        print("\n  ── Final parameters ───────────────────────────────")
        print(f"    ell          = {float(lattice.ell_var.numpy()):.4f}  "
              f"(integer: {lattice.discrete_ell()})")
        print(f"    r            = {float(lattice.r.numpy()):.4f}")
        print(f"    theta        = {np.degrees(float(lattice.theta.numpy())):.2f} deg")
        print(f"    epsilon      = {float(state.epsilon.numpy()):.4f}")
        print(f"    bloch_theta  = {float(state.bloch_theta_var.numpy()):.4f} rad")
        print(f"    bloch_phi    = {float(state.bloch_phi_var.numpy()):.4f} rad")
        print(f"    psi_LO       = {float(self.circuit.psi_var.numpy()):.4f} rad")
        print(f"    symplectic   = {lattice.verify_symplectic():.8f}  (should be 2)")
        print(f"  ───────────────────────────────────────────────────")

    # ── Convergence check ─────────────────────────────────────────────────────

    def is_converged(self, window: int = 50, tol: float = 1e-4) -> bool:
        """Check if QFI has plateaued over the last `window` steps."""
        if len(self.history) < window:
            return False
        recent = [h["qfi"] for h in self.history[-window:]]
        return float(np.std(recent)) < tol

    # ── Checkpoint save/load ──────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        """Save all trainable variable values to a numpy .npz file."""
        data = {}
        for v in self.circuit.trainable_variables:
            data[v.name.replace(":", "_")] = v.numpy()
        np.savez(path, **data)
        print(f"  Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load trainable variable values from a .npz checkpoint."""
        data = np.load(path)
        for v in self.circuit.trainable_variables:
            key = v.name.replace(":", "_")
            if key in data:
                v.assign(data[key])
        print(f"  Checkpoint loaded from {path}")
