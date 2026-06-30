from _header import EQUILIPY_ROOT

from GradShafranovSolver import *

############# EQUILIBRIUM PROBLEM PARAMETERS ##############

## CREATE GRAD-SHAFRANOV PROBLEM
Equilibrium = GradShafranovSolver()

## DECLARE SWITCHS:
##### GHOST PENALTY STABILISATION
Equilibrium.FIXED_BOUNDARY = True
Equilibrium.GhostStabilization = False
Equilibrium.PARALLEL = False

##### OUTPUT PLOTS IN RUNTIME
Equilibrium.plotelemsClas = False      # OUTPUT SWITCH FOR ELEMENTS CLASSIFICATION PLOTS AT EACH ITERATION
Equilibrium.plotPSI = False            # OUTPUT SWITCH FOR PSI SOLUTION PLOTS AT EACH ITERATION
##### OUTPUT FILES
Equilibrium.out_proparams = True       # OUTPUT SWITCH FOR SIMULATION PARAMETERS 
Equilibrium.out_boundaries = False
Equilibrium.out_elemsClas = True       # OUTPUT SWITCH FOR CLASSIFICATION OF MESH ELEMENTS
Equilibrium.out_plasmaLS = True        # OUTPUT SWITCH FOR PLASMA BOUNDARY LEVEL-SET FIELD VALUES
Equilibrium.out_plasmaBC = True        # OUTPUT SWITCH FOR PLASMA BOUNDARY CONDITION VALUES 
Equilibrium.out_plasmaapprox = True    # OUTPUT SWITCH FOR PLASMA BOUNDARY APPROXIMATION DATA 
Equilibrium.out_ghostfaces = False     # OUTPUT SWITCH FOR GHOST STABILISATION FACES DATA 
Equilibrium.out_quadratures = False
Equilibrium.out_elemsys = False        # OUTPUT SWITCH FOR ELEMENTAL MATRICES
##### OUTPUT PICKLING
Equilibrium.out_pickle = True          # OUTPUT SWITCH FOR SIMULATION DATA PYTHON PICKLE

# DEFINE NUMERICAL_TREATMENT PARAMETERS  
Equilibrium.dim = 2                    # PROBLEM SPATIAL DIMENSION
Equilibrium.QuadratureOrder2D = 8      # ORDER OF SURFACE NUMERICAL INTEGRATION QUADRATURES 
Equilibrium.QuadratureOrder1D = 5      # ORDER OF LENGTH NUMERICAL INTEGRATION QUADRATURES 
Equilibrium.ext_maxiter = 5            # EXTERNAL LOOP (PHI_B) MAXIMUM ITERATIONS
Equilibrium.ext_tol = 1.0e-3           # EXTERNAL LOOP (PHI_B) CONVERGENCE TOLERANCE
Equilibrium.int_maxiter = 10           # INTERNAL LOOP (PHI_NORM) MAXIMUM ITERATIONS
Equilibrium.int_tol = 1.0e-4           # INTERNAL LOOP (PHI_NORM) CONVERGENCE TOLERANCE
Equilibrium.tol_saddle = 0.1           # TOLERANCE FOR DISTANCE BETWEEN CONSECUTIVE ITERATION SADDLE POINTS (LETS PLASMA REGION CHANGE)
Equilibrium.beta = 1.0e6               # NITSCHE'S METHOD PENALTY PARAMETER
Equilibrium.Nconstrainedges = -1       # NUMBER OF PLAMA BOUNDARY APPROXIMATION EDGES ON WHICH CONSTRAIN BC
Equilibrium.zeta = 1.0e-2              # GHOST PENALTY PARAMETER 
Equilibrium.R0_axis = 6.0              # MAGNETIC AXIS OPTIMIZATION ROUTINE INITIAL GUESS R COORDINATE
Equilibrium.Z0_axis = 1.0              # MAGNETIC AXIS OPTIMIZATION ROUTINE INITIAL GUESS Z COORDINATE
Equilibrium.R0_saddle = 5.0            # ACTIVE SADDLE POINT OPTIMIZATION ROUTINE INITIAL GUESS R COORDINATE
Equilibrium.Z0_saddle = -3.5           # ACTIVE SADDLE POINT OPTIMIZATION ROUTINE INITIAL GUESS Z COORDINATE
Equilibrium.opti_maxiter = 50          # CRITICAL POINTS OPTIMIZATION ALGORITHM MAXIMAL ITERATIONS NUMBER
Equilibrium.opti_tol = 1.0e-6          # CRITICAL POINTS OPTIMIZATION ALGORITHM SOLUTION TOLERANCE

Equilibrium.InitialiseParameters()
Equilibrium.InitialisePickleLists()

################ COMPUTATIONAL DOMAIN MESH ###############

###### LINEAR TRIANGULAR ELEMENT MESH
#MESH = 'TRI03_FEM_1.0'
#MESH = 'TRI03_FEM_0.5'
#MESH = 'TRI03_FEM_0.1'
#MESH = 'TRI03_FEM_0.05'
#MESH = 'TRI03_FEM_0.02'

###### QUADRATIC TRIANGULAR ELEMENT MESH
#MESH = 'TRI06_FEM_1.0'
#MESH = 'TRI06_FEM_0.5'
MESH = 'TRI06_FEM_0.1'
#MESH = 'TRI06_FEM_0.05'
#MESH = 'TRI06_FEM_0.02'

###### LINEAR QUADRILATERAL ELEMENT MESH
#MESH = 'QUA04_FEM_1.0'
#MESH = 'QUA04_FEM_0.5'
#MESH = 'QUA04_FEM_0.1'
#MESH = 'QUA04_FEM_0.05'
#MESH = 'QUA04_FEM_0.02'

###### QUADRATIC QUADRILATERAL ELEMENT MESH
#MESH = 'QUA09_FEM_1.0'
#MESH = 'QUA09_FEM_0.5'
#MESH = 'QUA09_FEM_0.1'
#MESH = 'QUA09_FEM_0.05'
#MESH = 'QUA09_FEM_0.02'

Equilibrium.MESH = Mesh(MESH)

###################### TOKAMAK ###########################

TOKmesh = Mesh('TRI03-MEGAFINE-LINEAR-REDUCED')
Equilibrium.TOKAMAK = Tokamak(WALL_MESH = TOKmesh)

############### INITIAL PLASMA BOUNDARY ##################

Equilibrium.initialPHI = InitialPlasmaBoundary(EQUILIBRIUM = Equilibrium,   
                                           GEOMETRY = 'LINEAR', # PREDEFINED MODEL
                                           R0 = 6.0,            # MEAN RADIUS          
                                           epsilon = 0.32,      # INVERSE ASPECT RATIO
                                           kappa = 1.7,         # ELONGATION
                                           delta = 0.33)        # TRIANGULARITY

######### INITIAL GUESS FOR PLASMA MAGNETIC FLUX #########

Equilibrium.initialPSI = InitialGuess(EQUILIBRIUM = Equilibrium,
                                  PSI_GUESS = 'LINEAR', # PREDEFINED MODEL
                                  NOISE = True,         # WHITE NOISE 
                                  R0 = 6.0,             # MEAN RADIUS          
                                  epsilon = 0.32,       # INVERSE ASPECT RATIO
                                  kappa = 1.7,          # ELONGATION
                                  delta = 0.33,         # TRIANGULARITY
                                  A = 2.0)              # NOISE AMPLITUDE

################# PLASMA CURRENT MODEL ##################

Equilibrium.PlasmaCurrent = CurrentModel(EQUILIBRIUM = Equilibrium,
                                     MODEL = 'LINEAR',  # PREDEFINED MODEL
                                     R0 = 6.0,          # MEAN RADIUS          
                                     epsilon = 0.32,    # INVERSE ASPECT RATIO
                                     kappa = 1.7,       # ELONGATION
                                     delta = 0.33)      # TRIANGULARITY

################# INITIALISE MESH DATA ##################

Equilibrium.DomainDiscretisation(INITIALISATION=True)
Equilibrium.InitialisePSI()

################### LAUNCH SIMULATION ###################

CASE = "TS-FIXED-LINEAR-FEM"

Equilibrium.EQUILIPY(CASE)