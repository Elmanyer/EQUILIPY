#!/usr/bin/env python3
"""
(beta, zeta) parameter search for the EQUILIPY CutFEM Grad-Shafranov solver
(fixed-boundary LINEAR problem).

Sweeps a small (beta, zeta) grid for the TRI03 and TRI06 families on one
representative mesh, reports the relative L2 error of each combination and flags
the best. Every simulation is saved as a pickle (RESULTS/FIXED_BOUNDARY/...) and
the table is written to optimization_results.txt.

The search is intentionally small: with the corrected solver the optimum is flat
(beta ~ 100, zeta ~ 0) and neighbouring values differ only marginally.

Usage:
    python run_optimization.py            # representative mesh level 0.1
    python run_optimization.py 0.06       # use a different representative level
"""
import sys
from datetime import datetime
import numpy as np

from _header import EQUILIPY_ROOT
from _runner import run_case

# Small grids centred on the known-good optimum (beta=100, zeta=0).
BETA_GRID = [10.0, 100.0, 1000.0]
ZETA_GRID = [0.0, 1.0]

# Representative mesh level for the search (asymptotic regime, inexpensive).
OPT_LEVEL = '0.1'

OUTFILE = EQUILIPY_ROOT / 'TESTs' / 'optimization_results.txt'

COLHEAD = f"{'beta':>10}{'zeta':>8}{'relL2':>16}{'status':>9}"


def search(elem_type, level, lines):
    """Sweep the (beta, zeta) grid for one family; append rows; return best tuple."""
    header = f"\n{elem_type}_LINEAR_{level}   (beta, zeta) search"
    print(header); print(COLHEAD); print('-' * 43)
    lines += [header, COLHEAD, '-' * 43]

    rows, best = [], None
    for beta in BETA_GRID:
        for zeta in ZETA_GRID:
            res = run_case(elem_type, level, beta, zeta, case_prefix='OPT')
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
        msg = f"OPTIMAL {elem_type}: beta={best[0]:g}, zeta={best[1]:g}  (relL2={best[2]:.6e})"
        print(msg); lines += ['', msg]
    return best


def main():
    level = sys.argv[1] if len(sys.argv) > 1 else OPT_LEVEL
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    title = f"EQUILIPY (beta, zeta) parameter search (fixed boundary, LINEAR)   {stamp}"
    print('=' * 60); print(title); print('=' * 60)
    lines = ['=' * 60, title, f"representative mesh level: {level}",
             f"beta grid: {BETA_GRID}", f"zeta grid: {ZETA_GRID}", '=' * 60]

    for elem_type in ('TRI03', 'TRI06'):
        search(elem_type, level, lines)

    OUTFILE.write_text('\n'.join(lines) + '\n')
    print(f"\nResults table written to: {OUTFILE}")


if __name__ == '__main__':
    main()
