#!/usr/bin/env python3
"""
Point-wise diagnostic: check PSI_h vs PSI_exact at key points across mesh levels.

Goal: determine whether the solver converges to PSI_exact or to a different function.
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
from scipy.sparse import lil_matrix

PLASMA_INIT = dict(R0=6.0, epsilon=0.32, kappa=1.7, delta=0.33)

SOLVER_CONFIG = {
    'FIXED_BOUNDARY': True, 'RunTests': False, 'PARALLEL': False,
    'QuadratureOrder2D': 8, 'QuadratureOrder1D': 6,
    'ext_maxiter': 1, 'ext_tol': 1.0e-3,
    'int_maxiter': 50, 'int_tol': 1.0e-10,
    'tol_saddle': 0.1, 'Nconstrainedges': -1,
    'R0_axis': 6.0, 'Z0_axis': 1.0, 'R0_saddle': 5.0, 'Z0_saddle': -3.5,
    'opti_maxiter': 50, 'opti_tol': 1.0e-6, 'dim': 2,
}


def build_solver(mesh_name, beta=100.0, zeta=1.0):
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
    tok_mesh = Mesh('TRI03-MEGAFINE-LINEAR-REDUCED', readfiles=True)
    eq.TOKAMAK = Tokamak(WALL_MESH=tok_mesh)
    eq.initialPHI = InitialPlasmaBoundary(EQUILIBRIUM=eq, GEOMETRY='LINEAR', **PLASMA_INIT)
    eq.initialPSI  = InitialGuess(EQUILIBRIUM=eq, PSI_GUESS='LINEAR', NOISE=False, **PLASMA_INIT, A=0.0)
    eq.PlasmaCurrent = CurrentModel(EQUILIBRIUM=eq, MODEL='LINEAR', **PLASMA_INIT)
    eq.DomainDiscretisation(INITIALISATION=True)
    eq.InitialisePSI()
    return eq


def run_and_diagnose(mesh_name, beta=100.0, zeta=1.0):
    eq = build_solver(mesh_name, beta, zeta)
    eq.EQUILI(f'ptwise_{mesh_name}')

    # After solve: PSI_NORM[:,1] holds the solution
    R0 = PLASMA_INIT['R0']

    # 1. Evaluate PSI_h vs PSI_exact at nodes
    PSI_h = eq.PSI_NORM[:, 1]
    PSI_exact_nodes = np.array([eq.PlasmaCurrent.PSIanalytical(eq.MESH.X[i,:])
                                  for i in range(eq.MESH.Nn)])

    # 2. Find node closest to magnetic axis (R*=1, Z*=0 = R=R0, Z=0)
    R_axis, Z_axis = 6.0, 1.0  # physical approximate O-point
    dist = np.sqrt((eq.MESH.X[:,0]-R_axis)**2 + (eq.MESH.X[:,1]-Z_axis)**2)
    iaxis = np.argmin(dist)

    PSI_h_axis = PSI_h[iaxis]
    PSI_ex_axis = PSI_exact_nodes[iaxis]

    # 3. Plasma node errors
    plasma_nodes = set()
    for ielem in eq.MESH.PlasmaElems:
        ELEMENT = eq.MESH.Elements[ielem]
        for n in ELEMENT.Te:
            plasma_nodes.add(n)

    plasma_nodes = list(plasma_nodes)
    err_plasma = PSI_h[plasma_nodes] - PSI_exact_nodes[plasma_nodes]
    rms_err = np.sqrt(np.mean(err_plasma**2))
    rms_exact = np.sqrt(np.mean(PSI_exact_nodes[plasma_nodes]**2))

    # 4. Check sign: is PSI_h and PSI_exact same sign at axis?
    sign_match = np.sign(PSI_h_axis) == np.sign(PSI_ex_axis)

    return {
        'h': eq.MESH.meanLength,
        'relL2': eq.RelErrorL2norm,
        'PSI_h_axis': PSI_h_axis,
        'PSI_ex_axis': PSI_ex_axis,
        'axis_err': PSI_h_axis - PSI_ex_axis,
        'axis_rel_err': abs((PSI_h_axis - PSI_ex_axis)/PSI_ex_axis) if PSI_ex_axis != 0 else np.inf,
        'rms_plasma_err': rms_err / rms_exact,  # relative
        'sign_match': sign_match,
        'Ne': eq.MESH.Ne,
    }


def main():
    print("=" * 70)
    print("Point-wise convergence diagnostic")
    print("=" * 70)

    LEVELS = ['1.0', '0.5', '0.1', '0.06']
    BETA = 100.0
    ZETA = 1.0

    for elem_type in ['TRI03', 'TRI06']:
        suffix = '_LINEAR_'
        print(f"\n{elem_type}  beta={BETA}  zeta={ZETA}")
        print(f"{'h':>8}  {'relL2':>12}  {'PSI_h axis':>12}  {'PSI_ex axis':>12}  {'axis_err':>12}  {'axis_rel%':>10}  {'sign':>6}")
        print("-" * 85)

        prev_h = None
        prev_axis_err = None

        for lv in LEVELS:
            mesh_name = f"{elem_type}{suffix}{lv}"
            try:
                res = run_and_diagnose(mesh_name, BETA, ZETA)
                rate = np.nan
                if prev_h and prev_axis_err and prev_axis_err != 0 and res['axis_err'] != 0:
                    rate = np.log(abs(prev_axis_err)/abs(res['axis_err'])) / np.log(prev_h/res['h'])
                sign_str = "✓" if res['sign_match'] else "✗"
                rate_str = f"{rate:.2f}" if not np.isnan(rate) else "---"
                print(f"  {res['h']:>6.4f}  {res['relL2']:>12.4e}  {res['PSI_h_axis']:>12.6f}  {res['PSI_ex_axis']:>12.6f}  {res['axis_err']:>12.4e}  {res['axis_rel_err']:>9.2e} {sign_str}  (rate={rate_str})")
                prev_h = res['h']
                prev_axis_err = res['axis_err']
            except Exception as e:
                import traceback
                print(f"  {lv}: ERROR: {e}")
                traceback.print_exc()


if __name__ == '__main__':
    main()
