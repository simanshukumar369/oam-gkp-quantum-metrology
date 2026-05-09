"""
patch_perr.py — compute P_err for hexagonal results already saved in JSON
=========================================================================
Loads results/hexagonal_results.json, computes P_err analytically using
the converged r* values and effective_spread() from noise.py, then
re-prints the final LaTeX table rows with correct P_err values.

Run from ~/OAM-research26/:
    conda activate noon-sim
    python patch_perr.py
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from oam_gkp.noise import effective_spread
from oam_gkp.loss  import gkp_logical_error_rate

HEX_ELL = 0.0
HEX_R   = (np.pi / 3) ** 0.25   # ≈ 1.0127

JSON_PATH = "results/hexagonal_results.json"

with open(JSON_PATH) as f:
    data = json.load(f)

print(f"\n{'='*62}")
print(f"  P_err computation for hexagonal runs")
print(f"{'='*62}\n")

results_out = {}

for label, res in data.items():
    eta   = res["eta"]
    gamma = res["gamma"]

    print(f"  {label}  (η={eta}, γ={gamma})")

    for key, tag in [("fixed", "Fixed hex"), ("seeded", "Seeded opt")]:
        r_val     = res[key]["r"]       # converged r* (or HEX_R for fixed)
        theta_deg = res[key]["theta_deg"]
        theta_rad = np.radians(theta_deg)

        # effective_spread uses the lattice rotation angle to compute
        # anisotropic sigma_q, sigma_p under loss + dephasing (noise.py L.149)
        sigma_q, sigma_p = effective_spread(eta, gamma, theta=theta_rad)

        r_tf  = tf.constant(r_val, dtype=tf.float64)
        p_err = gkp_logical_error_rate(sigma_q, sigma_p, r_tf)
        p_err_val = float(p_err.numpy())

        res[key]["p_err"] = p_err_val
        print(f"    [{tag}]  r={r_val:.4f}  θ={theta_deg:.1f}°  "
              f"P_err={p_err_val:.2e}")

    results_out[label] = res
    print()

# ── Save updated JSON ─────────────────────────────────────────────────────────
with open(JSON_PATH, "w") as f:
    json.dump(results_out, f, indent=2)
print(f"Updated JSON saved → {JSON_PATH}\n")

# ── Print LaTeX rows ──────────────────────────────────────────────────────────
print("="*62)
print("  LATEX TABLE ROWS")
print("="*62)

for label, res in results_out.items():
    f = res["fixed"]
    s = res["seeded"]
    qfi_f   = res["fixed"]["qfi"]
    qfi_s   = res["seeded"]["qfi"]
    perr_f  = res["fixed"]["p_err"]
    perr_s  = res["seeded"]["p_err"]
    r_f     = res["fixed"]["r"]
    r_s     = res["seeded"]["r"]
    th_s    = res["seeded"]["theta_deg"]
    ell_s   = res["seeded"]["ell"]

    print(f"\n% {label}  (η={res['eta']}, γ={res['gamma']})")
    print(f"% Fixed hexagonal (ell=0, r=(π/3)^{{1/4}}≈{HEX_R:.3f}, θ=0°):")
    print(
        f"Hexagonal & $0$ & $0\\degree$ & ${r_f:.3f}$ & "
        f"${qfi_f:.4f}$ & ${perr_f:.2e}$ \\\\"
    )
    print(f"% Seeded opt converged to ell*={ell_s:.2f}, r*={r_s:.3f}, "
          f"θ*={th_s:.1f}°:")
    print(
        f"% Hex-seeded opt & ${ell_s:.2f}$ & ${th_s:.1f}\\degree$ & "
        f"${r_s:.3f}$ & ${qfi_s:.4f}$ & ${perr_s:.2e}$ \\\\"
    )
