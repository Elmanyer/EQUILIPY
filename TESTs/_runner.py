"""
Shared execution core for the EQUILIPY fixed-boundary LINEAR run scripts
(run_convergence.py and run_optimization.py).

Exposes a single entry point, run_case(), that builds and runs one full EQUILIPY
fixed-boundary Grad-Shafranov simulation on a TRIxx_LINEAR_<h> mesh and returns the
L2 error metrics together with the output (pickle) directory. Keeping this logic in
one place lets the two run scripts stay short and consistent.
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
TOK_MESH = 'TRI03-MEGAFINE-LINEAR-REDUCED'

# Mesh refinement levels: the h-label embedded in the TRIxx_LINEAR_<h> mesh name.
MESH_LEVELS = ['1.0', '0.5', '0.1', '0.06', '0.02']

# Output switches to silence. GradShafranovSolver.__init__ does not chain to the
# EquilipyOutput mixin __init__, so these must be set explicitly on every instance
# (out_pickle is then turned on per-run inside run_case).
OUTPUT_SWITCHES = [
    'plotelemsClas', 'plotPSI', 'out_proparams', 'out_boundaries', 'out_elemsClas',
    'out_plasmaLS', 'out_plasmaBC', 'out_plasmaapprox', 'out_ghostfaces',
    'out_quadratures', 'out_elemsys', 'out_PSIcrit', 'out_pickle',
]

# Validated solver configuration for the FIXED_BOUNDARY LINEAR problem (see CLAUDE.md
# "Common solver config"). TRI elements only, so QuadratureOrder2D=8 throughout.
SOLVER_CONFIG = {
    'FIXED_BOUNDARY':    True,
    'RunTests':          False,
    'PARALLEL':          False,
    'QuadratureOrder2D': 8,
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


def param_tag(val):
    """Compact label for a parameter value, used to build unique case/pickle names."""
    if val == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(val))))
    mant = int(round(val / 10 ** exp))
    return f"{mant}e{exp}"


def run_case(elem_type, level, beta, zeta, case_prefix, save_pickle=True):
    """
    Run one fixed-boundary LINEAR EQUILIPY simulation.

    Args:
        elem_type   : 'TRI03' or 'TRI06'.
        level       : mesh refinement label, e.g. '0.1' (selects TRIxx_LINEAR_<level>).
        beta        : Nitsche penalty parameter.
        zeta        : ghost-penalty parameter; ghost stabilization is enabled iff zeta > 0.
        case_prefix : prefix for the output case name (and therefore the pickle directory).
        save_pickle : if True, EQUILIPY writes the simulation pickle under
                      RESULTS/FIXED_BOUNDARY/<case>-<mesh>/.

    Returns:
        dict: always has 'success' and 'mesh_name'. On success also h, L2_error,
        relL2_error, Ne, Nn, n_cut, outputdir. On failure, 'error' and 'traceback'.
    """
    mesh_name = f"{elem_type}_LINEAR_{level}"
    try:
        eq = GradShafranovSolver()
        for key, val in SOLVER_CONFIG.items():
            setattr(eq, key, val)
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

        case_name = f"{case_prefix}-{elem_type}-b{param_tag(beta)}-z{param_tag(zeta)}"
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
