#!/usr/bin/env python3
"""
Fixed-boundary convergence study for the EQUILIPY Grad-Shafranov solver (LINEAR case).

Runs three solver/mesh groups over all refinement levels, computing the L2 error and
convergence rate at each refinement, saving every simulation as a pickle
(RESULTS/FIXED_BOUNDARY/...) and writing a results table to convergence_results.txt:

    1. FEM (body-fitted)      : *_FEM meshes. No cut elements, so ghost stabilization
                                is irrelevant (zeta = 0).
    2. CutFEM structured      : TRI*_REC_STRUC and QUA*_REC meshes. Ghost stabilization
                                (zeta > 0) restores the optimal rate; refresh the zeta
                                values below from run_optimization.py.
    3. CutFEM unstructured    : TRI*_REC_UNSTR meshes. Ghost stabilization is irrelevant
                                on unstructured cut patterns, so zeta = 0.

All element orders are covered: TRI03/QUA04 -> O(h^2), TRI06/QUA09 -> O(h^3),
TRI10/QUA16 -> O(h^4).

Usage:
    python run_convergence.py                 # all groups, all refinement levels
    python run_convergence.py 1.0 0.5 0.1     # a subset of levels (faster)
"""
import sys
from datetime import datetime
import numpy as np

from _header import EQUILIPY_ROOT
from _runner import run_case, conv_rate, expected_rate, MESH_LEVELS

# Solver/mesh groups (all element orders per group).
GROUPS = {
    'FEM (body-fitted)': [
        'TRI03_FEM', 'TRI06_FEM', 'TRI10_FEM',
        'QUA04_FEM', 'QUA09_FEM', 'QUA16_FEM',
    ],
    'CutFEM structured': [
        'TRI03_REC_STRUC', 'TRI06_REC_STRUC', 'TRI10_REC_STRUC',
        'QUA04_REC',       'QUA09_REC',       'QUA16_REC',
    ],
    'CutFEM unstructured': [
        'TRI03_REC_UNSTR', 'TRI06_REC_UNSTR', 'TRI10_REC_UNSTR',
    ],
}

# Nitsche penalty (used by CutFEM; irrelevant to FEM which has no cut elements).
BETA = 100.0

# Ghost-penalty parameter for STRUCTURED CutFEM, per element order (2/3/4).
# Refresh these from run_optimization.py.
STRUCT_ZETA = {2: 1.0, 3: 10.0, 4: 100.0}


def case_params(family):
    """(beta, zeta) for a family. Ghost (zeta>0) only for structured CutFEM meshes."""
    if '_FEM' in family:
        return BETA, 0.0                                   # body-fitted: no cut elements
    if family.endswith('_REC_UNSTR'):
        return BETA, 0.0                                   # unstructured: ghost irrelevant
    return BETA, STRUCT_ZETA[expected_rate(family)]        # structured CutFEM: ghost helps


OUTFILE = EQUILIPY_ROOT / 'TESTs' / 'convergence_results.txt'

COLHEAD = f"{'level':<8}{'h':>10}{'Ne':>9}{'L2 error':>15}{'relL2':>15}{'rate':>8}"


def study(family, levels, lines):
    """Run the convergence sweep for one mesh family; append rows to `lines`."""
    beta, zeta = case_params(family)
    header = (f"\n{family}   beta={beta:g}  zeta={zeta:g}  "
              f"(expected rate ~ h^{expected_rate(family)})")
    print(header); print(COLHEAD); print('-' * 65)
    lines += [header, COLHEAD, '-' * 65]

    prev_h, prev_L2 = None, None
    for lv in levels:
        res = run_case(family, lv, beta, zeta, case_prefix='CONV')
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

    for group_name, families in GROUPS.items():
        banner = f"\n########## {group_name} ##########"
        print(banner); lines.append(banner)
        for family in families:
            study(family, levels, lines)

    OUTFILE.write_text('\n'.join(lines) + '\n')
    print(f"\nResults table written to: {OUTFILE}")


if __name__ == '__main__':
    main()
