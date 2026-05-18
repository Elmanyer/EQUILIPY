#!/usr/bin/env python3
"""
Parameter Optimization Sweep for EQUILIPY CutFEM Solver

This script performs:
1. Parameter optimization (beta, zeta) for EACH mesh level
2. Convergence analysis using mesh-specific optimal parameters
3. Systematic sweep launcher with clear configuration
"""

import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from _header import EQUILIPY_ROOT

from GradShafranovSolver import GradShafranovSolver
from Mesh import Mesh
from Tokamak import Tokamak
from InitialPlasmaBoundary import InitialPlasmaBoundary
from InitialPSIGuess import InitialGuess
from PlasmaCurrent import CurrentModel


# ============================================================================
# CONFIGURATION - Modify these to control the sweep
# ============================================================================

# Mesh specifications: (element_type, mesh_levels)
# TRI meshes use the _LINEAR_ naming convention: e.g. TRI03_LINEAR_1.0
# QUA meshes use the _REC_ naming convention:    e.g. QUA04_REC_1.0
MESH_SPECS = [
    ('TRI03', ['1.0', '0.5', '0.1', '0.06', '0.02']),
    ('TRI06', ['1.0', '0.5', '0.1', '0.06', '0.02']),
    ('QUA04', ['1.0', '0.5', '0.1', '0.06', '0.02']),
    ('QUA09', ['1.0', '0.5', '0.1', '0.06', '0.02']),
]

# Parameter ranges for optimization (swept for EACH mesh)
# Sensitivity sweep results (LINEAR test case, FIXED_BOUNDARY=True):
#   TRI03: beta=10-100, zeta=0      → O(h²) rate ✓
#   TRI06: beta=10-100, zeta=0      → O(h³) rate ✓
#   QUA04: beta=100, zeta=0-1       → O(h²) rate ✓  (large zeta over-penalizes p=1)
#   QUA09: beta=100, zeta=100       → O(h³) rate ✓  (structured mesh needs strong ghost)
BETA_SWEEP_TRI  = [1e1, 5e1, 1e2, 5e2]
ZETA_SWEEP_TRI  = [0.0, 5e-1, 1e0, 5e0]
BETA_SWEEP_QUA4 = [1e1, 5e1, 1e2, 5e2]
ZETA_SWEEP_QUA4 = [0.0, 5e-1, 1e0, 5e0]
BETA_SWEEP_QUA9 = [1e2, 5e2, 1e3]
ZETA_SWEEP_QUA9 = [1e1, 5e1, 1e2, 5e2]
# Default sweeps
BETA_SWEEP = BETA_SWEEP_TRI
ZETA_SWEEP = ZETA_SWEEP_TRI

# Standard solver settings (shared across all runs)
# QuadratureOrder2D: TRI supports up to 8; QUA supports up to 5 only.
# Override per element family in run_solver().
SOLVER_CONFIG = {
    'FIXED_BOUNDARY': True,
    'RunTests': False,
    'PARALLEL': False,
    'QuadratureOrder2D': 8,   # overridden to 5 for QUA elements
    'QuadratureOrder1D': 6,
    'ext_maxiter': 1,          # 1 outer iteration for FIXED_BOUNDARY (deterministic)
    'ext_tol': 1.0e-3,
    'int_maxiter': 50,
    'int_tol': 1.0e-10,
    'tol_saddle': 0.1,
    'Nconstrainedges': -1,
    'R0_axis': 6.0,
    'Z0_axis': 1.0,
    'R0_saddle': 5.0,
    'Z0_saddle': -3.5,
    'opti_maxiter': 50,
    'opti_tol': 1.0e-6,
}

# Initial plasma conditions (from benchmark)
PLASMA_INIT = {
    'R0': 6.0,
    'epsilon': 0.32,
    'kappa': 1.7,
    'delta': 0.33,
}


# ============================================================================
# Helpers
# ============================================================================

def _mesh_suffix(elem_type):
    """Folder naming convention differs between TRI and QUA families."""
    return '_REC_' if elem_type.startswith('QUA') else '_LINEAR_'


def _param_str(val):
    """Compact label for a parameter value used in case/directory names."""
    if val == 0:
        return "0"
    e = int(np.floor(np.log10(abs(val))))
    m = int(round(val / 10**e))
    return f"{m}e{e}"


# ============================================================================
# Core Simulation Function
# ============================================================================

def run_solver(mesh_name, case_name, zeta=1.0, beta=1e4, ghost_enabled=True):
    """
    Execute a single EQUILIPY simulation via the full EQUILI solver loop.

    The solver writes a pickle of the completed simulation to its output
    directory. The path to that directory is returned in the result dict so
    the caller can decide whether to keep or discard it.

    Args:
        mesh_name: Full mesh name (e.g. 'TRI03_LINEAR_1.0', 'QUA04_REC_0.5')
        case_name: Unique case identifier (used to name the output directory)
        zeta: Ghost penalty parameter
        beta: Nitsche penalty parameter
        ghost_enabled: Enable ghost stabilization

    Returns:
        dict with results or error info, including 'outputdir'
    """
    try:
        mesh_path = EQUILIPY_ROOT / 'MESHES' / mesh_name
        if not mesh_path.exists():
            return {'success': False, 'error': f'Mesh not found: {mesh_path}',
                    'outputdir': None}

        # Initialize solver
        eq = GradShafranovSolver()

        # Configure solver parameters
        for key, val in SOLVER_CONFIG.items():
            setattr(eq, key, val)
        # QUA elements support GaussQuadrature up to order 5 only (TRI supports up to 8)
        eq.QuadratureOrder2D = 5 if mesh_name.upper().startswith('QUA') else 8
        eq.GhostStabilization = ghost_enabled
        eq.zeta = zeta
        eq.beta = beta
        eq.dim = 2

        # Output: disable all file output, enable pickle only
        eq.plotelemsClas = False
        eq.plotPSI = False
        eq.out_proparams = False
        eq.out_boundaries = False
        eq.out_elemsClas = False
        eq.out_plasmaLS = False
        eq.out_plasmaBC = False
        eq.out_plasmaapprox = False
        eq.out_ghostfaces = False
        eq.out_quadratures = False
        eq.out_elemsys = False
        eq.out_pickle = True

        eq.InitialiseParameters()
        eq.InitialisePickleLists()

        # Load mesh
        eq.MESH = Mesh(mesh_name, readfiles=True)

        # Tokamak wall
        tok_mesh = Mesh('TRI03-MEGAFINE-LINEAR-REDUCED', readfiles=True)
        eq.TOKAMAK = Tokamak(WALL_MESH=tok_mesh)

        # Initialize plasma
        eq.initialPHI = InitialPlasmaBoundary(
            EQUILIBRIUM=eq,
            GEOMETRY='LINEAR',
            **PLASMA_INIT
        )
        eq.initialPSI = InitialGuess(
            EQUILIBRIUM=eq,
            PSI_GUESS='LINEAR',
            NOISE=False,
            **PLASMA_INIT,
            A=0.0
        )
        eq.PlasmaCurrent = CurrentModel(
            EQUILIBRIUM=eq,
            MODEL='LINEAR',
            **PLASMA_INIT
        )

        # Prepare domain, then hand control to EQUILI which handles
        # output initialisation, the full solver loop, and pickle writing.
        eq.DomainDiscretisation(INITIALISATION=True)
        eq.InitialisePSI()
        eq.EQUILI(case_name)

        return {
            'success': True,
            'h': eq.MESH.meanLength,
            'L2_error': eq.ErrorL2norm,
            'relL2_error': eq.RelErrorL2norm,
            'n_elements': eq.MESH.Ne,
            'n_nodes': eq.MESH.Nn,
            'n_cut_elems': len(eq.MESH.PlasmaBoundElems),
            'n_ghost_faces': len(eq.MESH.GhostFaces) if eq.MESH.GhostFaces else 0,
            'outputdir': eq.outputdir,
        }

    except Exception as e:
        import traceback
        return {'success': False, 'error': str(e), 'traceback': traceback.format_exc(),
                'outputdir': None}


# ============================================================================
# Parameter Optimization
# ============================================================================

def optimize_on_mesh(mesh_name, beta_values=None, zeta_values=None, verbose=True):
    """
    Run parameter optimization (beta, zeta) on a single mesh.

    For each (beta, zeta) pair the solver is run with a unique case name so
    each run writes its own pickle file.  After the sweep, all output
    directories except the one belonging to the best run are deleted, leaving
    only the pickle of the lowest-L2-error configuration.

    Args:
        mesh_name: Mesh to optimize on
        beta_values: List of beta values to sweep
        zeta_values: List of zeta values to sweep
        verbose: Print progress

    Returns:
        dict with optimal params and sweep results
    """
    mn = mesh_name.upper()
    if mn.startswith('QUA09'):
        default_beta = BETA_SWEEP_QUA9
        default_zeta = ZETA_SWEEP_QUA9
    elif mn.startswith('QUA'):
        default_beta = BETA_SWEEP_QUA4
        default_zeta = ZETA_SWEEP_QUA4
    else:
        default_beta = BETA_SWEEP_TRI
        default_zeta = ZETA_SWEEP_TRI
    beta_values = beta_values or default_beta
    zeta_values = zeta_values or default_zeta

    if verbose:
        print(f"\n{'='*70}")
        print(f"Optimizing: {mesh_name}")
        print(f"{'='*70}")
        print(f"{'beta':<12} {'zeta':<10} {'relL2':<15} {'status':<10}")
        print("-" * 55)

    best_result = None
    best_params = None
    best_error = float('inf')
    best_outputdir = None
    sweep_data = []

    for beta in beta_values:
        for zeta in zeta_values:
            case_name = f"TS-FIXED-CutFEM-b{_param_str(beta)}-z{_param_str(zeta)}"
            result = run_solver(mesh_name, case_name,
                                zeta=zeta, beta=beta, ghost_enabled=True)

            if result['success']:
                error = result['relL2_error']
                status = "OK"

                sweep_data.append({
                    'beta': beta,
                    'zeta': zeta,
                    'relL2_error': error,
                    'outputdir': result['outputdir'],
                })

                if error < best_error:
                    best_error = error
                    best_params = {'beta': beta, 'zeta': zeta}
                    best_result = result
                    best_outputdir = result['outputdir']
                    status = "BEST"
            else:
                error = np.inf
                status = "FAIL"
                sweep_data.append({
                    'beta': beta,
                    'zeta': zeta,
                    'relL2_error': error,
                    'outputdir': result.get('outputdir'),
                })

            if verbose:
                print(f"{beta:<12.0e} {zeta:<10.2f} {error:<15.6e} {status:<10}")

    # Keep only the pickle of the best run; delete all other output directories.
    for entry in sweep_data:
        od = entry.get('outputdir')
        if od and od != best_outputdir and os.path.isdir(od):
            shutil.rmtree(od)

    if verbose and best_params:
        print(f"\n{'='*70}")
        print(f"OPTIMAL: beta={best_params['beta']:.2e}, zeta={best_params['zeta']:.4f}")
        print(f"         relL2={best_error:.6e}")
        if best_outputdir:
            pickle_name = os.path.basename(best_outputdir) + '.pickle'
            print(f"         pickle: {best_outputdir}/{pickle_name}")
        print(f"{'='*70}")

    return {
        'mesh_name': mesh_name,
        'optimal_params': best_params,
        'best_error': best_error,
        'best_result': best_result,
        'sweep_data': sweep_data,
    }


# ============================================================================
# Analysis on All Mesh Levels
# ============================================================================

def analyze_all_levels(elem_type, mesh_levels, verbose=True):
    """
    For each mesh level:
    1. Optimize parameters on that mesh
    2. Compute convergence metrics

    Args:
        elem_type: Element type (TRI03, TRI06, QUA04, QUA09, etc.)
        mesh_levels: List of mesh refinement levels
        verbose: Print progress

    Returns:
        list of result dicts, each with optimized params for its mesh level
    """
    suffix = _mesh_suffix(elem_type)
    results = []
    previous_result = None

    for level in mesh_levels:
        mesh_name = f"{elem_type}{suffix}{level}"

        # Step 1: Optimize parameters on THIS mesh
        opt_result = optimize_on_mesh(mesh_name, verbose=verbose)

        if not opt_result['optimal_params']:
            if verbose:
                print(f"ERROR: Could not optimize {mesh_name}")
            continue

        best_result = opt_result['best_result']

        # Step 2: Compute convergence rate if we have previous result
        conv_rate = np.nan
        if previous_result is not None:
            h_prev = previous_result['h']
            h_curr = best_result['h']
            err_prev = previous_result['L2_error']
            err_curr = best_result['L2_error']

            if h_prev > h_curr and err_prev > 0 and err_curr > 0:
                conv_rate = np.log(err_prev / err_curr) / np.log(h_prev / h_curr)

        # Step 3: Store result
        result_entry = {
            'elem_type': elem_type,
            'mesh_level': level,
            'h': best_result['h'],
            'L2_error': best_result['L2_error'],
            'relL2_error': best_result['relL2_error'],
            'n_elements': best_result['n_elements'],
            'n_cut_elems': best_result['n_cut_elems'],
            'n_ghost_faces': best_result['n_ghost_faces'],
            'beta': opt_result['optimal_params']['beta'],
            'zeta': opt_result['optimal_params']['zeta'],
            'conv_rate': conv_rate,
        }
        results.append(result_entry)
        previous_result = best_result

        if verbose:
            rate_str = f"{conv_rate:.2f}" if not np.isnan(conv_rate) else "---"
            print(f"\n  Result: h={best_result['h']:.4f}, L2={best_result['L2_error']:.6e}, "
                  f"rate={rate_str}\n")

    return results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("="*70)
    print("EQUILIPY Parameter Optimization Sweep")
    print("(Optimize per mesh level)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    all_results = {}
    summary_lines = []

    # For each element type
    for elem_type, mesh_levels in MESH_SPECS:
        print(f"\n{'='*70}")
        print(f"Processing: {elem_type}")
        print(f"{'='*70}")

        # Optimize on EACH mesh level individually
        results = analyze_all_levels(elem_type, mesh_levels, verbose=True)

        if results:
            all_results[elem_type] = results

            # Print summary for this element type
            print(f"\n{'='*70}")
            print(f"Summary for {elem_type}")
            print(f"{'='*70}")
            print(f"{'Mesh':<10} {'h':<12} {'L2 error':<14} {'relL2':<14} {'rate':<8} {'beta':<12} {'zeta':<10}")
            print("-" * 85)

            for r in results:
                rate_str = f"{r['conv_rate']:.2f}" if not np.isnan(r['conv_rate']) else "---"
                print(f"{r['mesh_level']:<10} {r['h']:<12.4f} {r['L2_error']:<14.6e} "
                      f"{r['relL2_error']:<14.6e} {rate_str:<8} {r['beta']:<12.2e} {r['zeta']:<10.4f}")

            # Compute average convergence rate
            rates = [r['conv_rate'] for r in results if not np.isnan(r['conv_rate'])]
            if rates:
                avg_rate = np.mean(rates)
                print(f"\nAverage convergence rate: {avg_rate:.2f}")
                summary_lines.append(f"  {elem_type}: {avg_rate:.2f} (expected: 2.0 for P1, 3.0 for P2)")

    # Phase: Final Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    for line in summary_lines:
        print(line)

    # Save detailed results
    summary_file = str(EQUILIPY_ROOT / 'TESTs' / 'optimization_results.txt')
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"EQUILIPY Parameter Optimization Results\n")
        f.write(f"(Per-Mesh-Level Optimization)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")

        for elem_type, results in all_results.items():
            f.write(f"\n{elem_type}\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Mesh':<10} {'h':<12} {'L2 error':<14} {'relL2':<14} {'rate':<8} {'beta':<12} {'zeta':<10}\n")
            f.write("-" * 85 + "\n")

            for r in results:
                rate_str = f"{r['conv_rate']:.2f}" if not np.isnan(r['conv_rate']) else "---"
                f.write(f"{r['mesh_level']:<10} {r['h']:<12.4f} {r['L2_error']:<14.6e} "
                        f"{r['relL2_error']:<14.6e} {rate_str:<8} {r['beta']:<12.2e} {r['zeta']:<10.4f}\n")

            rates = [r['conv_rate'] for r in results if not np.isnan(r['conv_rate'])]
            if rates:
                avg_rate = np.mean(rates)
                f.write(f"Average convergence rate: {avg_rate:.2f}\n")

    print(f"\nDetailed results saved to: {summary_file}")

    print("\n" + "="*70)
    print("Optimization complete!")
    print("="*70)

    return all_results


if __name__ == '__main__':
    results = main()
