#!/usr/bin/env python3
"""
Convergence analysis for the CutFEM solver after the Nitsche consistency-sign fix.
Runs TRI03 and TRI06 meshes with fixed beta/zeta and reports L2 error + rate.
"""
import sys
import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from _header import EQUILIPY_ROOT

from GradShafranovSolver import GradShafranovSolver
from Mesh import Mesh
from Tokamak import Tokamak
from InitialPlasmaBoundary import InitialPlasmaBoundary
from InitialPSIGuess import InitialGuess
from PlasmaCurrent import CurrentModel

# ── parameters shared across all runs ─────────────────────────────────────────
PLASMA = dict(R0=6.0, epsilon=0.32, kappa=1.7, delta=0.33)

# After the consistency-sign fix, Nitsche only needs a moderate penalty.
# beta ~ 10-100 is the correct order for dimensionless p=1 problems.
# Ghost penalty zeta is a dimensionless prefactor (code multiplies by h^(2p-1)).
BETA = 100.0
ZETA = 1.0

# Mesh series: (element_tag, mesh_names, expected_rate)
MESH_SERIES = [
    ('TRI03', ['TRI03_LINEAR_0.5', 'TRI03_LINEAR_0.1', 'TRI03_LINEAR_0.06', 'TRI03_LINEAR_0.02'], 2),
    ('TRI06', ['TRI06_LINEAR_0.5', 'TRI06_LINEAR_0.1', 'TRI06_LINEAR_0.06', 'TRI06_LINEAR_0.02'], 3),
]

TOK_MESH = 'TRI03-MEGAFINE-LINEAR-REDUCED'


def run_one(mesh_name):
    """Run the solver on a single mesh and return (h, L2_error, relL2)."""
    eq = GradShafranovSolver()

    # switches
    eq.FIXED_BOUNDARY    = True
    eq.GhostStabilization = True
    eq.PARALLEL          = False
    eq.RunTests          = False

    # output: minimal (suppress files)
    eq.plotelemsClas  = False
    eq.plotPSI        = False
    eq.out_proparams  = False
    eq.out_boundaries = False
    eq.out_elemsClas  = False
    eq.out_plasmaLS   = False
    eq.out_plasmaBC   = False
    eq.out_plasmaapprox = False
    eq.out_ghostfaces = False
    eq.out_quadratures = False
    eq.out_elemsys    = False
    eq.out_pickle     = False

    # numerics
    eq.dim               = 2
    eq.QuadratureOrder2D = 8
    eq.QuadratureOrder1D = 6
    eq.ext_maxiter       = 1      # fixed boundary needs only 1 external iteration
    eq.ext_tol           = 1e-3
    eq.int_maxiter       = 1      # linear model: PSI_INDEPENDENT → 1 internal iteration
    eq.int_tol           = 1e-10
    eq.tol_saddle        = 0.1
    eq.beta              = BETA
    eq.zeta              = ZETA
    eq.Nconstrainedges   = -1
    eq.R0_axis           = 6.0
    eq.Z0_axis           = 1.0
    eq.R0_saddle         = 5.0
    eq.Z0_saddle         = -3.5
    eq.opti_maxiter      = 50
    eq.opti_tol          = 1e-6

    eq.InitialiseParameters()
    eq.InitialisePickleLists()

    eq.MESH   = Mesh(mesh_name)
    eq.TOKAMAK = Tokamak(WALL_MESH=Mesh(TOK_MESH))

    eq.initialPHI = InitialPlasmaBoundary(EQUILIBRIUM=eq, GEOMETRY='LINEAR', **PLASMA)
    eq.initialPSI = InitialGuess(EQUILIBRIUM=eq, PSI_GUESS='LINEAR', NOISE=False, A=0.0, **PLASMA)
    eq.PlasmaCurrent = CurrentModel(EQUILIBRIUM=eq, MODEL='LINEAR', **PLASMA)

    eq.DomainDiscretisation(INITIALISATION=True)
    eq.InitialisePSI()

    case = f"CVG-{mesh_name}"
    eq.EQUILI(case)

    h   = eq.MESH.meanLength
    L2  = eq.ErrorL2norm
    rL2 = eq.RelErrorL2norm
    return h, L2, rL2


def convergence_table(tag, mesh_names, expected_rate):
    print(f"\n{'='*70}")
    print(f"  {tag}   (expected rate ~ h^{expected_rate})  beta={BETA}  zeta={ZETA}")
    print(f"{'='*70}")
    print(f"{'Mesh':<30} {'h':>8} {'L2 error':>14} {'relL2':>12} {'rate':>8}")
    print('-'*74)

    prev_h, prev_L2 = None, None
    for mesh in mesh_names:
        try:
            h, L2, rL2 = run_one(mesh)
            if prev_h is not None and prev_L2 is not None and L2 > 0 and prev_L2 > 0:
                rate = np.log(prev_L2 / L2) / np.log(prev_h / h)
                print(f"{mesh:<30} {h:>8.4f} {L2:>14.6e} {rL2:>12.6e} {rate:>8.2f}")
            else:
                print(f"{mesh:<30} {h:>8.4f} {L2:>14.6e} {rL2:>12.6e} {'---':>8}")
            prev_h, prev_L2 = h, L2
        except Exception as e:
            print(f"{mesh:<30}  ERROR: {e}")
    print()


if __name__ == '__main__':
    for tag, meshes, rate in MESH_SERIES:
        convergence_table(tag, meshes, rate)
