import sys
sys.path.append('../src/')

from GradShafranovSolver import *

MESH = 'QUA09-ULTRAFINE-LINEAR-REDUCED'

##############################################################

## CREATE GRAD-SHAFRANOV PROBLEM 
Equilibrium = GradShafranovSolver(MESH)

#### DECLARE SWITCHS:
## FIXED/FREE-BOUNDARY PROBLEM
Equilibrium.FIXED_BOUNDARY = True
## GHOST PENALTY STABILISATION
Equilibrium.GhostStabilization = False
## OUTPUT PLOTS IN RUNTIME
Equilibrium.plotelemsClas = False      # OUTPUT SWITCH FOR ELEMENTS CLASSIFICATION PLOTS AT EACH ITERATION
Equilibrium.plotPSI = False            # OUTPUT SWITCH FOR PSI SOLUTION PLOTS AT EACH ITERATION
## OUTPUT FILES
Equilibrium.out_proparams = True       # OUTPUT SWITCH FOR SIMULATION PARAMETERS 
Equilibrium.out_elemsClas = True       # OUTPUT SWITCH FOR CLASSIFICATION OF MESH ELEMENTS
Equilibrium.out_plasmaLS = True        # OUTPUT SWITCH FOR PLASMA BOUNDARY LEVEL-SET FIELD VALUES
Equilibrium.out_plasmaBC = True        # OUTPUT SWITCH FOR PLASMA BOUNDARY CONDITION VALUES 
Equilibrium.out_plasmaapprox = True    # OUTPUT SWITCH FOR PLASMA BOUNDARY APPROXIMATION DATA 
Equilibrium.out_ghostfaces = True      # OUTPUT SWITCH FOR GHOST STABILISATION FACES DATA 
Equilibrium.out_elemsys = False        # OUTPUT SWITCH FOR ELEMENTAL MATRICES
## OUTPUT PICKLING
Equilibrium.out_pickle = True          # OUTPUT SWITCH FOR SIMULATION DATA PYTHON PICKLE

# DEFINE NUMERICAL_TREATMENT PARAMETERS  
Equilibrium.QuadratureOrder2D = 5   # ORDER OF NUMERICAL INTEGRATION QUADRATURES                    
Equilibrium.ext_maxiter = 10           # EXTERNAL LOOP (PHI_B) MAXIMUM ITERATIONS
Equilibrium.ext_tol = 1.0e-3        # EXTERNAL LOOP (PHI_B) CONVERGENCE TOLERANCE
Equilibrium.int_maxiter = 10           # INTERNAL LOOP (PHI_NORM) MAXIMUM ITERATIONS
Equilibrium.int_tol = 1.0e-4        # INTERNAL LOOP (PHI_NORM) CONVERGENCE TOLERANCE
Equilibrium.beta = 1.0e6            # NITSCHE'S METHOD PENALTY PARAMETER
Equilibrium.zeta = 1.0e-2           # GHOST PENALTY PARAMETER
Equilibrium.alpha = 1.0             # AIKITEN'S METHOD RELAXATION PARAMETER
Equilibrium.R0_axis = 6.0           # MAGNETIC AXIS OPTIMIZATION ROUTINE INITIAL GUESS R COORDINATE
Equilibrium.Z0_axis = 0.0           # MAGNETIC AXIS OPTIMIZATION ROUTINE INITIAL GUESS Z COORDINATE
Equilibrium.R0_saddle = 5.0           # ACTIVE SADDLE POINT OPTIMIZATION ROUTINE INITIAL GUESS R COORDINATE
Equilibrium.Z0_saddle = 3.0           # ACTIVE SADDLE POINT OPTIMIZATION ROUTINE INITIAL GUESS Z COORDINATE
Equilibrium.opti_maxiter = 50         # CRITICAL POINTS OPTIMIZATION ALGORITHM MAXIMAL ITERATIONS NUMBER
Equilibrium.opti_tol = 1.0e-6       # CRITICAL POINTS OPTIMIZATION ALGORITHM SOLUTION TOLERANCE

### DEFINE INITIAL PLASMA BOUNDARY
Equilibrium.initialPHI = InitialPlasmaBoundary(PROBLEM = Equilibrium,   
                                           GEOMETRY = 'LINEAR', # PREDEFINED MODEL
                                           R0 = 6.0,            # MEAN RADIUS          
                                           epsilon = 0.32,      # INVERSE ASPECT RATIO
                                           kappa = 1.7,         # ELONGATION
                                           delta = 0.33)        # TRIANGULARITY

### DEFINE INITIAL GUESS FOR PLASMA MAGNETIC FLUX
Equilibrium.initialPSI = InitialGuess(PROBLEM = Equilibrium,
                                  PSI_GUESS = 'LINEAR', # PREDEFINED MODEL
                                  NOISE = True,         # WHITE NOISE 
                                  R0 = 6.0,             # MEAN RADIUS          
                                  epsilon = 0.32,       # INVERSE ASPECT RATIO
                                  kappa = 1.7,          # ELONGATION
                                  delta = 0.33,         # TRIANGULARITY
                                  A = 2.0)              # NOISE AMPLITUDE

### DEFINE PLASMA CURRENT MODEL
Equilibrium.PlasmaCurrent = CurrentModel(PROBLEM = Equilibrium,
                                     MODEL = 'LINEAR',  # PREDEFINED MODEL
                                     R0 = 6.0,          # MEAN RADIUS          
                                     epsilon = 0.32,    # INVERSE ASPECT RATIO
                                     kappa = 1.7,       # ELONGATION
                                     delta = 0.33)      # TRIANGULARITY

## NAME SIMULATION CASE 
CASE = "TS-FIXED-LINEAR-FEM"
## COMPUTE PLASMA EQUILIBRIUM
Equilibrium.EQUILI(CASE)