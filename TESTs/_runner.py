"""
Shared execution core for the EQUILIPY fixed-boundary LINEAR run scripts
(run_convergence.py and run_optimization.py).

Exposes a single entry point, run_case(), that builds and runs one full EQUILIPY
fixed-boundary Grad-Shafranov simulation on a <FAMILY>_<h> mesh and returns the
L2 error metrics together with the output (pickle) directory. Keeping this logic in
one place lets the two run scripts stay short and consistent.

A "family" is the mesh name without the refinement suffix, e.g.:
    - FEM (body-fitted)         : TRI03_FEM, TRI06_FEM, TRI10_FEM, QUA04_FEM, QUA09_FEM, QUA16_FEM
    - CutFEM structured         : TRI03_REC_STRUC, TRI06_REC_STRUC, TRI10_REC_STRUC, QUA04_REC, QUA09_REC, QUA16_REC
    - CutFEM unstructured (tri) : TRI03_REC_UNSTR, TRI06_REC_UNSTR, TRI10_REC_UNSTR
The full mesh name is f"{family}_{level}" (e.g. "QUA16_REC_0.1").
"""
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from _header import EQUILIPY_ROOT
from GradShafranovSolver import GradShafranovSolver
from Mesh import Mesh
from Tokamak import Tokamak
from InitialPlasmaBoundary import InitialPlasmaBoundary
from InitialPSIGuess import InitialGuess
from PlasmaCurrent import CurrentModel

# Plasma geometry (ITER-like benchmark with a LINEAR analytical solution).
PLASMA_INIT = dict(R0=6.0, epsilon=0.32, kappa=1.7, delta=0.33)

# First-wall mesh used as the (fixed) computational-domain boundary.
TOK_MESH = 'TRI03_ITER_FIRSTWALL'

# Mesh refinement levels (the h-label embedded in the <FAMILY>_<h> mesh name).
MESH_LEVELS = ['1.0', '0.5', '0.1', '0.05', '0.02']

# Output switches to silence. GradShafranovSolver.__init__ does not chain to the
# EquilipyOutput mixin __init__, so these must be set explicitly on every instance
# (out_pickle is then turned on per-run inside run_case).
OUTPUT_SWITCHES = [
    'plotelemsClas', 'plotPSI', 'out_proparams', 'out_boundaries', 'out_elemsClas',
    'out_plasmaLS', 'out_plasmaBC', 'out_plasmaapprox', 'out_ghostfaces',
    'out_quadratures', 'out_elemsys', 'out_PSIcrit', 'out_pickle',
]

# Base solver configuration for the FIXED_BOUNDARY LINEAR problem (see CLAUDE.md
# "Common solver config"). QuadratureOrder2D is overridden per element type in
# run_case (QUA supports orders 1-5 only; TRI uses 8).
SOLVER_CONFIG = {
    'FIXED_BOUNDARY':    True,
    'RunTests':          False,
    'PARALLEL':          False,
    'QuadratureOrder2D': 8,        # overridden to 5 for QUA families
    'QuadratureOrder1D': 6,
    'ext_maxiter':       1,        # FIXED_BOUNDARY: a single outer iteration is exact
    'ext_tol':           1.0e-3,
    'int_maxiter':       50,       # LINEAR is PSI-independent -> converges in 1 inner step
    'int_tol':           1.0e-10,
    'tol_saddle':        0.1,
    'Nconstrainedges':   -1,
    'R0_axis':           6.0,
    'Z0_axis':           1.0,
    'R0_saddle':         5.0,
    'Z0_saddle':         -3.5,
    'opti_maxiter':      50,
    'opti_tol':          1.0e-6,
    'dim':               2,
}


def element_base(family):
    """Element tag (e.g. 'TRI06', 'QUA16') from a mesh family (e.g. 'TRI06_REC_STRUC')."""
    return family.split('_')[0]


def quadrature_order_2D(family):
    """Surface quadrature order for a family: QUA supports orders 1-5 only, TRI uses 8."""
    return 5 if element_base(family).startswith('QUA') else 8


def expected_rate(family):
    """Expected asymptotic L2 convergence rate O(h^(p+1)) from the element order p."""
    return {'03': 2, '06': 3, '10': 4,       # TRI03/06/10
            '04': 2, '09': 3, '16': 4}[element_base(family)[3:]]


def param_tag(val):
    """Compact label for a parameter value, used to build unique case/pickle names."""
    if val == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(val))))
    mant = int(round(val / 10 ** exp))
    return f"{mant}e{exp}"


def run_case(family, level, beta, zeta, case_prefix, save_pickle=True):
    """
    Run one fixed-boundary LINEAR EQUILIPY simulation.

    Args:
        family      : mesh family, e.g. 'TRI06_REC_STRUC', 'QUA16_REC', 'TRI10_FEM'.
        level       : mesh refinement label, e.g. '0.1' (selects f"{family}_{level}").
        beta        : Nitsche penalty parameter.
        zeta        : ghost-penalty parameter; ghost stabilization is enabled iff zeta > 0.
        case_prefix : prefix for the output case name (and therefore the pickle directory).
        save_pickle : if True, EQUILIPY writes the simulation pickle under
                      RESULTS/FIXED_BOUNDARY/<case>-<mesh>/.

    Returns:
        dict: always has 'success' and 'mesh_name'. On success also h, L2_error,
        relL2_error, Ne, Nn, n_cut, outputdir. On failure, 'error' and 'traceback'.
    """
    mesh_name = f"{family}_{level}"
    try:
        eq = GradShafranovSolver()
        for key, val in SOLVER_CONFIG.items():
            setattr(eq, key, val)
        eq.QuadratureOrder2D = quadrature_order_2D(family)   # per element type
        for switch in OUTPUT_SWITCHES:         # silence all file/plot output ...
            setattr(eq, switch, False)
        eq.GhostStabilization = (zeta > 0)
        eq.beta = beta
        eq.zeta = zeta
        eq.out_pickle = save_pickle            # ... except the simulation pickle

        eq.InitialiseParameters()
        eq.InitialisePickleLists()         # requires out_pickle to be set first

        eq.MESH = Mesh(mesh_name, readfiles=True)
        eq.TOKAMAK = Tokamak(WALL_MESH=Mesh(TOK_MESH, readfiles=True))

        eq.initialPHI = InitialPlasmaBoundary(EQUILIBRIUM=eq, GEOMETRY='LINEAR', **PLASMA_INIT)
        eq.initialPSI = InitialGuess(EQUILIBRIUM=eq, PSI_GUESS='LINEAR', NOISE=False, A=0.0, **PLASMA_INIT)
        eq.PlasmaCurrent = CurrentModel(EQUILIBRIUM=eq, MODEL='LINEAR', **PLASMA_INIT)

        eq.DomainDiscretisation(INITIALISATION=True)
        eq.InitialisePSI()

        case_name = f"{case_prefix}-{family}-b{param_tag(beta)}-z{param_tag(zeta)}"
        eq.EQUILIPY(case_name)

        return dict(success=True, mesh_name=mesh_name, h=eq.MESH.meanLength,
                    L2_error=eq.ErrorL2norm, relL2_error=eq.RelErrorL2norm,
                    Ne=eq.MESH.Ne, Nn=eq.MESH.Nn,
                    n_cut=len(eq.MESH.PlasmaBoundElems), outputdir=eq.outputdir)
    except Exception as exc:
        import traceback
        return dict(success=False, mesh_name=mesh_name, error=str(exc),
                    traceback=traceback.format_exc(), outputdir=None)


def conv_rate(h_prev, err_prev, h_cur, err_cur):
    """Local convergence rate between two refinements; nan if it cannot be computed."""
    if h_prev is None or err_prev is None or err_prev <= 0 or err_cur <= 0 or h_prev <= h_cur:
        return np.nan
    return np.log(err_prev / err_cur) / np.log(h_prev / h_cur)
