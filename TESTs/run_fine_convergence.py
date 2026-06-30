#!/usr/bin/env python3
"""
Fine mesh convergence study for TRI03 and TRI06.
Uses all available mesh levels including h=0.02.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from _header import EQUILIPY_ROOT
from GradShafranovSolver import GradShafranovSolver
from Mesh import Mesh
from Tokamak import Tokamak
from InitialPlasmaBoundary import InitialPlasmaBoundary
from InitialPSIGuess import InitialGuess
from PlasmaCurrent import CurrentModel
import numpy as np

PLASMA_INIT = dict(R0=6.0, epsilon=0.32, kappa=1.7, delta=0.33)
SOLVER_CONFIG = {
    'FIXED_BOUNDARY': True, 'RunTests': False, 'PARALLEL': False,
    'QuadratureOrder2D': 8, 'QuadratureOrder1D': 6,
    'ext_maxiter': 1, 'ext_tol': 1.0e-3,
    'int_maxiter': 50, 'int_tol': 1.0e-10,
    'tol_saddle': 0.1, 'Nconstrainedges': -1,
    'R0_axis': 6.0, 'Z0_axis': 1.0,
    'R0_saddle': 5.0, 'Z0_saddle': -3.5,
    'opti_maxiter': 50, 'opti_tol': 1.0e-6, 'dim': 2,
}


def run_single(mesh_name, beta, zeta):
    try:
        eq = GradShafranovSolver()
        for k, v in SOLVER_CONFIG.items():
            setattr(eq, k, v)
        eq.GhostStabilization = (zeta > 0)
        eq.zeta = zeta
        eq.beta = beta
        for attr in ['plotelemsClas','plotPSI','out_proparams','out_boundaries',
                     'out_elemsClas','out_plasmaLS','out_plasmaBC','out_plasmaapprox',
                     'out_ghostfaces','out_quadratures','out_elemsys','out_pickle']:
            setattr(eq, attr, False)
        eq.InitialiseParameters()
        eq.InitialisePickleLists()
        eq.MESH = Mesh(mesh_name, readfiles=True)
        tok = Mesh('TRI03-MEGAFINE-LINEAR-REDUCED', readfiles=True)
        eq.TOKAMAK = Tokamak(WALL_MESH=tok)
        eq.initialPHI = InitialPlasmaBoundary(EQUILIBRIUM=eq, GEOMETRY='LINEAR', **PLASMA_INIT)
        eq.initialPSI = InitialGuess(EQUILIBRIUM=eq, PSI_GUESS='LINEAR', NOISE=False, **PLASMA_INIT, A=0.0)
        eq.PlasmaCurrent = CurrentModel(EQUILIBRIUM=eq, MODEL='LINEAR', **PLASMA_INIT)
        eq.DomainDiscretisation(INITIALISATION=True)
        eq.InitialisePSI()
        eq.EQUILIPY(f'fine_{mesh_name}_b{int(beta)}_z{int(zeta)}')
        return dict(h=eq.MESH.meanLength, relL2=eq.RelErrorL2norm, L2=eq.ErrorL2norm,
                    Ne=eq.MESH.Ne, Nn=eq.MESH.Nn)
    except Exception as e:
        import traceback
        print(f"  ERROR {mesh_name}: {e}")
        return None


def study(elem_type, levels, beta, zeta):
    suffix = '_LINEAR_'
    print(f"\n{'='*65}")
    print(f"  {elem_type}  beta={beta}  zeta={zeta}")
    print(f"  {'level':<8} {'h':>8} {'Ne':>8} {'relL2':>12} {'rate':>8}")
    print(f"  {'-'*55}")
    prev = None
    for lv in levels:
        mname = f"{elem_type}{suffix}{lv}"
        res = run_single(mname, beta, zeta)
        if res is None:
            print(f"  {lv:<8} FAILED")
            continue
        rate = np.nan
        if prev:
            rate = np.log(prev['relL2']/res['relL2']) / np.log(prev['h']/res['h'])
        rate_s = f"{rate:.2f}" if not np.isnan(rate) else "  ---"
        print(f"  {lv:<8} {res['h']:>8.4f} {res['Ne']:>8d} {res['relL2']:>12.4e} {rate_s:>8}", flush=True)
        prev = res


if __name__ == '__main__':
    LEVELS = ['1.0', '0.5', '0.1', '0.06', '0.02']
    BETA = 100.0

    for elem_type in ['TRI03', 'TRI06']:
        for zeta in [0.0, 1.0]:
            study(elem_type, LEVELS, BETA, zeta)
