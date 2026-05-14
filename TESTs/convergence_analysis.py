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
MESH_SPECS = [
    ('TRI03', ['1.0', '0.5', '0.1', '0.06', '0.02']),
    ('TRI06', ['1.0', '0.5', '0.1', '0.06', '0.02']),
    ('TRI10', ['1.0', '0.5', '0.1', '0.06']),
]

# Parameter ranges for optimization (swept for EACH mesh)
BETA_SWEEP = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
ZETA_SWEEP = [1e2, 1e3, 1e4, 1e5, 1e6]

# Standard solver settings (shared across all runs)
SOLVER_CONFIG = {
    'FIXED_BOUNDARY': True,
    'RunTests': False,
    'PARALLEL': False,
    'QuadratureOrder2D': 8,
    'QuadratureOrder1D': 6,
    'ext_maxiter': 5,
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
# Core Simulation Function
# ============================================================================

def run_solver(mesh_name, zeta=1.0, beta=1e4, ghost_enabled=True):
    """
    Execute a single EQUILIPY simulation.

    Args:
        mesh_name: Full mesh name (e.g., 'TRI03_LINEAR_1.0')
        zeta: Ghost penalty parameter
        beta: Plasma beta constraint
        ghost_enabled: Enable ghost stabilization

    Returns:
        dict with results or error info
    """
    try:
        mesh_path = BASE_DIR / 'MESHES' / mesh_name
        if not mesh_path.exists():
            return {'success': False, 'error': f'Mesh not found: {mesh_path}'}

        # Initialize solver
        eq = GradShafranovSolver()

        # Configure solver
        for key, val in SOLVER_CONFIG.items():
            setattr(eq, key, val)
        eq.GhostStabilization = ghost_enabled
        eq.zeta = zeta
        eq.beta = beta
        eq.dim = 2

        # Disable all output/plotting
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

        # Run simulation
        eq.DomainDiscretisation(INITIALISATION=True)
        eq.InitialisePSI()
        eq.InitialisePSI_B()
        eq.AssembleGlobalSystem()
        eq.SolveSystem()
        eq.NormalisePSI()
        eq.UpdatePSI_NORM()
        eq.UpdateElementalPSI()
        eq.ComputeEuclierrorField()
        eq.ComputeL2errorPlasma()

        return {
            'success': True,
            'h': eq.MESH.meanLength,
            'L2_error': eq.ErrorL2norm,
            'relL2_error': eq.RelErrorL2norm,
            'n_elements': eq.MESH.Ne,
            'n_nodes': eq.MESH.Nn,
            'n_cut_elems': len(eq.MESH.PlasmaBoundElems),
            'n_ghost_faces': len(eq.MESH.GhostFaces) if eq.MESH.GhostFaces else 0,
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================================
# Parameter Optimization
# ============================================================================

def optimize_on_mesh(mesh_name, beta_values=None, zeta_values=None, verbose=True):
    """
    Run parameter optimization (beta, zeta) on a single mesh.

    Args:
        mesh_name: Mesh to optimize on
        beta_values: List of beta values to sweep
        zeta_values: List of zeta values to sweep
        verbose: Print progress

    Returns:
        dict with optimal params and sweep results
    """
    beta_values = beta_values or BETA_SWEEP
    zeta_values = zeta_values or ZETA_SWEEP

    if verbose:
        print(f"\n{'='*70}")
        print(f"Optimizing: {mesh_name}")
        print(f"{'='*70}")
        print(f"{'beta':<12} {'zeta':<10} {'relL2':<15} {'status':<10}")
        print("-" * 55)

    best_result = None
    best_params = None
    best_error = float('inf')
    sweep_data = []

    for beta in beta_values:
        for zeta in zeta_values:
            result = run_solver(mesh_name, zeta=zeta, beta=beta, ghost_enabled=True)

            if result['success']:
                error = result['relL2_error']
                status = "OK"

                sweep_data.append({
                    'beta': beta,
                    'zeta': zeta,
                    'relL2_error': error,
                })

                if error < best_error:
                    best_error = error
                    best_params = {'beta': beta, 'zeta': zeta}
                    best_result = result
                    status = "BEST"
            else:
                error = np.inf
                status = "FAIL"

            if verbose:
                print(f"{beta:<12.0e} {zeta:<10.2f} {error:<15.6e} {status:<10}")

    if verbose and best_params:
        print(f"\n{'='*70}")
        print(f"OPTIMAL: beta={best_params['beta']:.2e}, zeta={best_params['zeta']:.4f}")
        print(f"         relL2={best_error:.6e}")
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
        elem_type: Element type (TRI03, TRI06, etc.)
        mesh_levels: List of mesh refinement levels
        verbose: Print progress

    Returns:
        list of result dicts, each with optimized params for its mesh level
    """
    results = []
    previous_result = None

    for i, level in enumerate(mesh_levels):
        mesh_name = f"{elem_type}_LINEAR_{level}"

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
    summary_file = 'optimization_results.txt'
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
