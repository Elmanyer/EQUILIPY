#!/usr/bin/env python3
"""
Residual diagnostic: check consistency of assembled GS system with PSI_exact.
Decompose residual by contribution type to identify the O(h) source.
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
from Element import Element
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

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


def build_solver(mesh_name, beta=100.0, zeta=0.0):
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
    return eq


def assemble_only(eq):
    """Run just the assembly phase without solving."""
    eq.InitialisePSI_B()
    # Set initial PSIe for source term evaluation
    R0 = PLASMA_INIT['R0']
    PSI_exact_nodes = np.array([eq.PlasmaCurrent.PSIanalytical(eq.MESH.X[i,:])
                                  for i in range(eq.MESH.Nn)])
    for ELEMENT in eq.MESH.Elements:
        ELEMENT.PSIe = PSI_exact_nodes[ELEMENT.Te]
    eq.AssembleGlobalSystem()
    return PSI_exact_nodes


def compute_residual_breakdown(eq, PSI_exact_nodes):
    """Compute residual of PSI_exact in assembled system, broken down by contribution type."""
    Nn = eq.MESH.Nn
    LHS_csr = eq.LHS.tocsr()
    RHS = eq.RHS.flatten()

    residual_full = LHS_csr @ PSI_exact_nodes - RHS
    norm_LHS = np.max(np.abs(LHS_csr.data)) if len(LHS_csr.data) > 0 else 1.0
    norm_PSI = np.linalg.norm(PSI_exact_nodes)

    # Global residual metrics
    print(f"\n  ||r||_inf = {np.max(np.abs(residual_full)):.4e}")
    print(f"  ||r||_2   = {np.linalg.norm(residual_full):.4e}")
    print(f"  ||LHS||_inf = {norm_LHS:.4e}")
    print(f"  ||PSI_ex||_2 = {norm_PSI:.4e}")
    print(f"  Rel residual (inf): {np.max(np.abs(residual_full)) / (norm_LHS * norm_PSI):.4e}")

    # DOF classification
    plasma_nodes = set()
    vacuum_nodes = set()
    cut_nodes = set()
    for ielem in eq.MESH.PlasmaElems:
        for n in eq.MESH.Elements[ielem].Te:
            plasma_nodes.add(n)
    for ielem in eq.MESH.VacuumElems:
        for n in eq.MESH.Elements[ielem].Te:
            vacuum_nodes.add(n)
    for ielem in eq.MESH.PlasmaBoundElems:
        for n in eq.MESH.Elements[ielem].Te:
            cut_nodes.add(n)

    plasma_only = list(plasma_nodes - cut_nodes)
    vacuum_only = list(vacuum_nodes - cut_nodes)
    cut_only = list(cut_nodes)
    outer_wall = set(eq.MESH.BoundaryNodes)

    pure_plasma = [n for n in plasma_only if n not in outer_wall]
    pure_vacuum = [n for n in vacuum_only if n not in outer_wall]

    def rel_rms(nodes):
        if not nodes:
            return np.nan
        r = residual_full[nodes]
        return np.linalg.norm(r) / (norm_LHS * np.linalg.norm(PSI_exact_nodes[nodes]))

    print(f"\n  Rel residual by node type (rel to LHS * PSI_ex):")
    print(f"    Interior plasma  ({len(pure_plasma):5d} nodes): {rel_rms(pure_plasma):.4e}")
    print(f"    Interior vacuum  ({len(pure_vacuum):5d} nodes): {rel_rms(pure_vacuum):.4e}")
    print(f"    Cut element      ({len(cut_only):5d} nodes): {rel_rms(cut_only):.4e}")
    print(f"    Outer wall       ({len(outer_wall):5d} nodes): {rel_rms(list(outer_wall)):.4e}")

    return {
        'rel_res_plasma': rel_rms(pure_plasma),
        'rel_res_vacuum': rel_rms(pure_vacuum),
        'rel_res_cut': rel_rms(cut_only),
        'h': eq.MESH.meanLength,
    }


def main():
    LEVELS = ['1.0', '0.5', '0.1', '0.06']
    BETA = 100.0
    ZETA = 0.0

    print(f"{'='*60}")
    print(f"Residual consistency diagnostic  beta={BETA} zeta={ZETA}")
    print(f"{'='*60}")
    print(f"\n{'h':>8} {'plasma_rel':>12} {'vacuum_rel':>12} {'cut_rel':>12}")
    print("-"*50)

    results = []
    for lv in LEVELS:
        mesh_name = f"TRI03_LINEAR_{lv}"
        print(f"\n  [{mesh_name}]", flush=True)
        eq = build_solver(mesh_name, BETA, ZETA)
        PSI_exact = assemble_only(eq)
        res = compute_residual_breakdown(eq, PSI_exact)
        results.append(res)

    print(f"\n{'='*60}")
    print("SUMMARY (convergence of residuals with h):")
    print(f"{'h':>8} {'plasma_rel':>12} {'vacuum_rel':>12} {'cut_rel':>12}")
    print("-"*50)
    prev = None
    for r in results:
        if prev:
            hp = prev['h']
            hr = r['h']
            rp_p = np.log(prev['rel_res_plasma']/r['rel_res_plasma']) / np.log(hp/hr) if r['rel_res_plasma'] > 0 else np.nan
            rp_v = np.log(prev['rel_res_vacuum']/r['rel_res_vacuum']) / np.log(hp/hr) if r['rel_res_vacuum'] > 0 else np.nan
            rp_c = np.log(prev['rel_res_cut']/r['rel_res_cut']) / np.log(hp/hr) if r['rel_res_cut'] > 0 else np.nan
            print(f"  {r['h']:>6.4f} {r['rel_res_plasma']:>12.4e} rate={rp_p:.2f}  {r['rel_res_vacuum']:>12.4e} rate={rp_v:.2f}  {r['rel_res_cut']:>12.4e} rate={rp_c:.2f}")
        else:
            print(f"  {r['h']:>6.4f} {r['rel_res_plasma']:>12.4e} ----  {r['rel_res_vacuum']:>12.4e} ----  {r['rel_res_cut']:>12.4e} ----")
        prev = r


if __name__ == '__main__':
    main()
