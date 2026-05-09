"""
oam_gkp — OAM-Encoded GKP States for Fault-Tolerant Quantum Phase Estimation
==============================================================================
Author : Simanshu Kumar
Affil  : Department of Physics, Kumaun University Nainital
Date   : 2026

Module layout
-------------
  lattice.py   — OAM-to-lattice mapping  (Eq. 4, 6 of the paper)
  states.py    — Twisted GKP state preparation via Strawberry Fields
  circuit.py   — Full parameterised sensing circuit  (Eq. 10)
  qfi.py       — Quantum Fisher information utilities  (Eqs. 2-3)
  noise.py     — Photon-loss and dephasing channel helpers  (Eqs. 8-9)
  loss.py      — Combined sensitivity + fault-tolerance loss  (Eq. 11)
  optimizer.py — Adam training loop with gradient clipping
  utils.py     — Visualisation and analysis helpers
"""

__version__ = "0.1.0"
__author__  = "Simanshu Kumar"
