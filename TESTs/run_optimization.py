#!/usr/bin/env python3
"""
(beta, zeta) parameter search for the EQUILIPY CutFEM Grad-Shafranov solver
(fixed-boundary LINEAR problem), on STRUCTURED meshes.

Ghost-penalty (zeta) stabilization improves the solution quality and restores the
optimal convergence rate for STRUCTURED cut meshes; on unstructured meshes it is
irrelevant (the cut pattern is already irregular). The search is therefore run only
on the structured CutFEM families:
    - triangles     : TRI03_REC_STRUC, TRI06_REC_STRUC, TRI10_REC_STRUC
    - quadrilaterals: QUA04_REC,       QUA09_REC,       QUA16_REC
For each family it sweeps a small (beta, zeta) grid on one representative mesh,
reports the relative L2 error of each combination and flags the best. Every
simulation is saved as a pickle (RESULTS/FIXED_BOUNDARY/...) and the table is
written to optimization_results.txt.

Usage:
    python run_optimization.py            # representative mesh level 0.1
    python run_optimization.py 0.05       # use a different representative level
"""
import sys
from datetime import datetime
import numpy as np

from _header import EQUILIPY_ROOT
from _runner import run_case

# Structured CutFEM families (all element orders).
STRUCT_FAMILIES = [
    'TRI03_REC_STRUC', 'TRI06_REC_STRUC', 'TRI10_REC_STRUC',
    'QUA04_REC',       'QUA09_REC',       'QUA16_REC',
]

# Search grids. zeta > 0 enables ghost stabilization; zeta = 0 is the no-ghost baseline.
BETA_GRID = [10.0, 100.0, 1000.0]
ZETA_GRID = [0.0, 1.0, 10.0, 100.0]

# Representative mesh level for the search (asymptotic regime, inexpensive).
OPT_LEVEL = '0.1'

OUTFILE = EQUILIPY_ROOT / 'TESTs' / 'optimization_results.txt'

COLHEAD = f"{'beta':>10}{'zeta':>8}{'relL2':>16}{'status':>9}"


def search(family, level, lines):
    """Sweep the (beta, zeta) grid for one family; append rows; return best tuple."""
    header = f"\n{family}_{level}   (beta, zeta) search"
    print(header); print(COLHEAD); print('-' * 43)
    lines += [header, COLHEAD, '-' * 43]

    rows, best = [], None
    for beta in BETA_GRID:
        for zeta in ZETA_GRID:
            res = run_case(family, level, beta, zeta, case_prefix='OPT')
            err = res['relL2_error'] if res['success'] else np.inf
            rows.append((beta, zeta, err))
            if res['success'] and (best is None or err < best[2]):
                best = (beta, zeta, err)

    for beta, zeta, err in rows:
        if not np.isfinite(err):
            flag = 'FAIL'
        elif best and (beta, zeta) == best[:2]:
            flag = 'BEST'
        else:
            flag = 'ok'
        row = f"{beta:>10.0e}{zeta:>8.1f}{err:>16.6e}{flag:>9}"
        print(row, flush=True); lines.append(row)

    if best:
        msg = f"OPTIMAL {family}: beta={best[0]:g}, zeta={best[1]:g}  (relL2={best[2]:.6e})"
        print(msg); lines += ['', msg]
    return best


def main():
    level = sys.argv[1] if len(sys.argv) > 1 else OPT_LEVEL
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    title = f"EQUILIPY (beta, zeta) search - structured CutFEM (fixed boundary, LINEAR)   {stamp}"
    print('=' * 60); print(title); print('=' * 60)
    lines = ['=' * 60, title, f"representative mesh level: {level}",
             f"families: {', '.join(STRUCT_FAMILIES)}",
             f"beta grid: {BETA_GRID}", f"zeta grid: {ZETA_GRID}", '=' * 60]

    for family in STRUCT_FAMILIES:
        search(family, level, lines)

    OUTFILE.write_text('\n'.join(lines) + '\n')
    print(f"\nResults table written to: {OUTFILE}")


if __name__ == '__main__':
    main()
