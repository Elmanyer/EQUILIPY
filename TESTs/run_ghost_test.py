#!/usr/bin/env python3
"""
Ghost stabilization diagnostic: compare convergence rates with zeta=0 vs zeta>0.

Runs TRI03 and TRI06 across mesh levels with:
  - zeta=0   (ghost OFF) -> should show optimal rates if ghost is the problem
  - zeta=1   (ghost ON, low) -> reference
  - zeta=100 (ghost ON, moderate)

Expected:
  TRI03 optimal: O(h^2), TRI06 optimal: O(h^3)
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
    'FIXED_BOUNDARY': True,
    'RunTests': False,
    'PARALLEL': False,
    'QuadratureOrder2D': 8,
    'QuadratureOrder1D': 6,
    'ext_maxiter': 1,      # single outer iteration (fixed boundary)
    'ext_tol': 1.0e-3,
    'int_maxiter': 50,
    'int_tol': 1.0e-10,
    'tol_saddle': 0.1,
    'Nconstrainedges': -1,
    'R0_axis': 6.0, 'Z0_axis': 1.0,
    'R0_saddle': 5.0, 'Z0_saddle': -3.5,
    'opti_maxiter': 50, 'opti_tol': 1.0e-6,
    'dim': 2,
}


def run(mesh_name, beta, zeta):
    try:
        mesh_path = EQUILIPY_ROOT / 'MESHES' / mesh_name
        if not mesh_path.exists():
            return None, f'mesh not found: {mesh_path}'

        eq = GradShafranovSolver()
        for k, v in SOLVER_CONFIG.items():
            setattr(eq, k, v)
        eq.GhostStabilization = (zeta > 0)
        eq.zeta = zeta
        eq.beta = beta

        # disable all output
        for attr in ['plotelemsClas','plotPSI','out_proparams','out_boundaries',
                     'out_elemsClas','out_plasmaLS','out_plasmaBC','out_plasmaapprox',
                     'out_ghostfaces','out_quadratures','out_elemsys','out_pickle']:
            setattr(eq, attr, False)

        eq.InitialiseParameters()
        eq.InitialisePickleLists()

        eq.MESH = Mesh(mesh_name, readfiles=True)
        tok_mesh = Mesh('TRI03-MEGAFINE-LINEAR-REDUCED', readfiles=True)
        eq.TOKAMAK = Tokamak(WALL_MESH=tok_mesh)

        eq.initialPHI = InitialPlasmaBoundary(EQUILIBRIUM=eq, GEOMETRY='LINEAR', **PLASMA_INIT)
        eq.initialPSI  = InitialGuess(EQUILIBRIUM=eq, PSI_GUESS='LINEAR', NOISE=False, **PLASMA_INIT, A=0.0)
        eq.PlasmaCurrent = CurrentModel(EQUILIBRIUM=eq, MODEL='LINEAR', **PLASMA_INIT)

        eq.DomainDiscretisation(INITIALISATION=True)
        eq.InitialisePSI()
        eq.EQUILI(f'ghost_test_{mesh_name}_b{int(beta)}_z{int(zeta)}')

        return {
            'h': eq.MESH.meanLength,
            'L2': eq.ErrorL2norm,
            'relL2': eq.RelErrorL2norm,
            'Ne': eq.MESH.Ne,
        }, None
    except Exception as e:
        import traceback
        return None, traceback.format_exc()


def convergence_study(elem_type, levels, beta, zeta_list):
    suffix = '_REC_' if elem_type.startswith('QUA') else '_LINEAR_'
    print(f"\n{'='*72}")
    print(f"  {elem_type}   beta={beta}")
    print(f"  {'mesh':<12} {'h':>8}  ", end='')
    for z in zeta_list:
        print(f"  zeta={z:<6}  rate", end='')
    print()
    print('-'*72)

    prev = {z: None for z in zeta_list}

    for lv in levels:
        mesh_name = f"{elem_type}{suffix}{lv}"
        print(f"  {lv:<12} ", end='', flush=True)
        h_val = None
        for z in zeta_list:
            res, err = run(mesh_name, beta, z)
            if res is None:
                print(f"  ERR({str(err)[:20]})  ---", end='')
                continue
            h_val = res['h']
            e = res['relL2']
            rate = np.nan
            if prev[z] is not None:
                h0, e0 = prev[z]
                if h0 > h_val and e0 > 0 and e > 0:
                    rate = np.log(e0/e) / np.log(h0/h_val)
            prev[z] = (h_val, e)
            rate_str = f"{rate:5.2f}" if not np.isnan(rate) else "  ---"
            print(f"  {e:.3e}  {rate_str}", end='', flush=True)
        if h_val:
            print(f"  (h={h_val:.4f})")
        else:
            print()


if __name__ == '__main__':
    LEVELS_TRI = ['1.0', '0.5', '0.1', '0.06']
    BETA = 100.0
    ZETA_LIST = [0.0, 1.0, 100.0]

    convergence_study('TRI03', LEVELS_TRI, BETA, ZETA_LIST)
    convergence_study('TRI06', LEVELS_TRI, BETA, ZETA_LIST)
