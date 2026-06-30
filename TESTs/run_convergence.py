#!/usr/bin/env python3
"""
Fixed-boundary convergence study for the EQUILIPY CutFEM Grad-Shafranov solver.

Runs the LINEAR analytical test case on the TRI03_LINEAR_<h> and TRI06_LINEAR_<h>
mesh families at the validated optimal (beta, zeta) parameters, computes the L2
error and convergence rate at each refinement, saves every simulation as a pickle
(RESULTS/FIXED_BOUNDARY/...) and writes a results table to convergence_results.txt.

Usage:
    python run_convergence.py                 # all refinement levels
    python run_convergence.py 1.0 0.5 0.1     # a subset of levels (faster)
"""
import sys
from datetime import datetime
import numpy as np

from _header import EQUILIPY_ROOT
from _runner import run_case, conv_rate, MESH_LEVELS

# Optimal parameters per family (refresh/confirm with run_optimization.py).
# beta=100, zeta=0 gives optimal CutFEM rates for the corrected fixed-boundary solver.
OPTIMAL = {
    'TRI03': dict(beta=100.0, zeta=0.0),   # P1 -> O(h^2)
    'TRI06': dict(beta=100.0, zeta=0.0),   # P2 -> O(h^3)
}
EXPECTED_RATE = {'TRI03': 2, 'TRI06': 3}

OUTFILE = EQUILIPY_ROOT / 'TESTs' / 'convergence_results.txt'

COLHEAD = f"{'level':<8}{'h':>10}{'Ne':>9}{'L2 error':>15}{'relL2':>15}{'rate':>8}"


def study(elem_type, levels, lines):
    """Run the convergence sweep for one element family; append rows to `lines`."""
    beta = OPTIMAL[elem_type]['beta']
    zeta = OPTIMAL[elem_type]['zeta']
    header = (f"\n{elem_type}_LINEAR   beta={beta:g}  zeta={zeta:g}  "
              f"(expected rate ~ h^{EXPECTED_RATE[elem_type]})")
    print(header); print(COLHEAD); print('-' * 65)
    lines += [header, COLHEAD, '-' * 65]

    prev_h, prev_L2 = None, None
    for lv in levels:
        res = run_case(elem_type, lv, beta, zeta, case_prefix='CONV')
        if not res['success']:
            row = f"{lv:<8}  FAILED: {res['error']}"
        else:
            rate = conv_rate(prev_h, prev_L2, res['h'], res['L2_error'])
            rate_s = f"{rate:.2f}" if not np.isnan(rate) else '---'
            row = (f"{lv:<8}{res['h']:>10.4f}{res['Ne']:>9d}"
                   f"{res['L2_error']:>15.6e}{res['relL2_error']:>15.6e}{rate_s:>8}")
            prev_h, prev_L2 = res['h'], res['L2_error']
        print(row, flush=True)
        lines.append(row)


def main():
    levels = sys.argv[1:] or MESH_LEVELS
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    title = f"EQUILIPY fixed-boundary convergence study (LINEAR)   {stamp}"
    print('=' * 65); print(title); print('=' * 65)
    lines = ['=' * 65, title, f"levels: {', '.join(levels)}", '=' * 65]

    for elem_type in ('TRI03', 'TRI06'):
        study(elem_type, levels, lines)

    OUTFILE.write_text('\n'.join(lines) + '\n')
    print(f"\nResults table written to: {OUTFILE}")


if __name__ == '__main__':
    main()
