import numpy as np
from math import ceil
from Element import *
from Mesh import *

class EquilipyInitialisation:
    
    def InitialiseParameters(self):
        
        # OVERRIDE CRITICAL POINT OPTIMIZATION OUTPUT WHEN FIXED-BOUNDARY PROBLEM
        if self.FIXED_BOUNDARY:
            self.out_PSIcrit = False
        else:
            self.out_PSIcrit = True
        
        # OVERRIDE GHOST FACES OUTPUT WHEN GHOST STABILIZATION IS OFF
        if not self.GhostStabilization:
            self.out_ghostfaces = False
            
        # COMPUTE 1D NUMERICAL QUADRATURE ORDER
        self.QuadratureOrder1D = ceil(0.5*(self.QuadratureOrder2D+1))
            
        # INITIALISE CRITICAL POINTS ARRAY
        self.Xcrit = np.zeros([2,2,3])  # [(iterations n, n+1), (extremum, saddle point), (R_crit,Z_crit,elem_crit)]
        self.Xcrit[0,0,:-1] = np.array([self.R0_axis,self.Z0_axis])
        self.Xcrit[0,1,:-1] = np.array([self.R0_saddle,self.Z0_saddle])
        
        # INITIALISE AITKEN'S RELAXATION LAMBDAS
        self.lambdaPSI = np.zeros([2])
        self.lambdaPSI[0] = self.lambda0           # INITIAL LAMBDA PARAMETER
        self.lambdamax = 0.95
        self.lambdamin = 0.0
        return
    
    
    def InitialisePickleLists(self):
        # INITIALISE FULL SIMULATION DATA LISTS
        if self.out_pickle:
            self.PlasmaLS_sim = list()
            self.MeshElements_sim = list()
            self.PlasmaNodes_sim = list()
            self.VacuumNodes_sim = list()
            self.PlasmaBoundApprox_sim = list()
            self.PlasmaBoundGhostFaces_sim = list()
            self.PlasmaUpdateIt_sim = list()
            self.PSI_sim = list()
            self.PSI_NORM_sim = list()
            self.PSI_B_sim = list()
            self.Residu_sim = list()
            self.PSIIt_sim = list()
            if not self.FIXED_BOUNDARY:
                self.PSIcrit_sim = list()
        return
    
    
    def InitialisePlasmaLevelSet(self):
        """
        Computes the initial level-set function values describing the plasma boundary. Negative values represent inside the plasma region.
        """ 
        
        self.PlasmaLS = np.zeros([self.MESH.Nn,2])
        self.PlasmaLSstar = np.zeros([self.MESH.Nn,2])
        self.PlasmaLS[:,0] = self.initialPHI.PHI0
        self.PlasmaLS[:,1] = self.PlasmaLS[:,0]
        return 
    
    
    def InitialiseMESH(self,mesh_name):
        """
        Initializes all mesh related elements and preprocess mesh data for simulation:
            - Initializes some simulation parameters.
            - Initializes python pickle lists for direct output.
            - Initializes the level-set function for plasma and vacuum vessel boundaries.
            - Initializes the elements in the computational domain.
            - Classifies elements and writes their classification.
            - Approximates the plasma boundary interface.
            - Finds ghost faces if necessary
            - Computes elemental numerical integration quadratures.
        """
        
        print("INITIALISE MESH...")
        path_to_mesh = self.pwd + '/MESHES/' + mesh_name
        self.MESH = Mesh(path_to_mesh)
        
        print("     -> READ MESH FILES...", end="")
        self.MESH.ReadMeshFile()
        self.MESH.ReadFixFile()
        self.dim = self.MESH.dim
        print('Done!')
        return
    
    
    def Ini(self):
        
        print('INITIALISE ELEMENTAL DISCRETISATION...')
        
        # INITIALISE LEVEL-SET FUNCTION
        print("     -> INITIALISE LEVEL-SET...", end="")
        self.InitialisePlasmaLevelSet()
        print('Done!')
        
        # INITIALISE ELEMENTS 
        print("     -> INITIALISE ELEMENTS...")
        self.MESH.InitialiseElements(self.PlasmaLS)
        if type(self.PlasmaCurrent) != type(None) and self.PlasmaCurrent.DIMENSIONLESS:
            self.MESH.DimensionlessCoordinates(self.PlasmaCurrent.R0)
        print('     Done!')
        
        # CLASSIFY ELEMENTS   
        print("     -> CLASSIFY ELEMENTS...", end="")
        self.MESH.IdentifyNearestNeighbors()
        self.MESH.IdentifyBoundaries()
        self.MESH.ClassifyElements()
        print("Done!")

        # COMPUTE PLASMA BOUNDARY APPROXIMATION
        print("     -> APPROXIMATE PLASMA BOUNDARY INTERFACE...", end="")
        self.MESH.ComputePlasmaBoundaryApproximation()
        print("Done!")
        
        # IDENTIFY GHOST FACES 
        if self.GhostStabilization:
            print("     -> IDENTIFY GHOST FACES...", end="")
            self.MESH.ComputePlasmaBoundaryGhostFaces()
            print("Done!")
        
        # COMPUTE NUMERICAL INTEGRATION QUADRATURES
        print('     -> COMPUTE NUMERICAL INTEGRATION QUADRATURES...', end="")
        self.MESH.ComputeIntegrationQuadratures(self.QuadratureOrder2D,self.QuadratureOrder1D)
        if self.GhostStabilization:
            self.MESH.ComputeGhostFacesQuadratures(self.QuadratureOrder1D)
        print('Done!')
        
        # COMPUTE NUMBER OF NODES ON PLASMA BOUNDARY APPROXIMATION
        self.MESH.NnPB = self.MESH.ComputePlasmaBoundaryNumberNodes()
        
        print('Done!')
        return  
    
    
    
    def InitialisePSI(self):  
        """
        Initializes the PSI vectors used for storing iterative solutions during the simulation and computes the initial guess.

        This function:
            - Computes the number of nodes on boundary approximations for the plasma boundary and vacuum vessel first wall.
            - Initializes PSI solution arrays.
            - Computes an initial guess for the normalized PSI values and assigns them to the corresponding elements.
            - Computes initial vacuum vessel first wall PSI values and stores them for the first iteration.
            - Assigns boundary constraint values for both the plasma and vacuum vessel boundaries.
        """
        print('INITIALISE PSI...')
        
        ####### INITIALISE PSI VECTORS
        print('     -> INITIALISE PSI ARRAYS...', end="")
        # INITIALISE ITERATIVE UPDATED ARRAYS
        self.PSI = np.zeros([self.MESH.Nn],dtype=float)            # SOLUTION FROM SOLVING CutFEM SYSTEM OF EQUATIONS (INTERNAL LOOP)       
        self.PSI_NORMstar = np.zeros([self.MESH.Nn,2],dtype=float) # UNRELAXED NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        self.PSI_NORM = np.zeros([self.MESH.Nn,2],dtype=float)     # RELAXED NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        self.PSI_CONV = np.zeros([self.MESH.Nn],dtype=float)       # CONVERGED SOLUTION FIELD
        print('Done!')
        
        ####### COMPUTE INITIAL GUESS AND STORE IT IN ARRAY FOR N=0
        # COMPUTE INITIAL GUESS
        print('     -> COMPUTE INITIAL GUESS FOR PSI_NORM...', end="")
        self.PSI_NORM[:,0] = self.initialPSI.PSI0
        self.PSI_NORM[:,1] = self.PSI_NORM[:,0]
        self.PSI_0 = self.initialPSI.PSI0_0
        self.PSI_X = self.initialPSI.PSI0_X
        # ASSIGN VALUES TO EACH ELEMENT
        self.UpdateElementalPSI()
        print('Done!')  
        
        print('Done!') 
        return
    
        
    def InitialisePSI_B(self):
        
        print('INITIALISE PSI_B...')
        
        ####### INITIALISE PSI BOUNDARY VECTOR
        self.PSI_B = np.zeros([self.MESH.Nnbound,2],dtype=float)   # VACUUM VESSEL FIRST WALL PSI VALUES (EXTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)    
        
        ####### COMPUTE INITIAL VACUUM VESSEL BOUNDARY VALUES PSI_B AND STORE THEM IN ARRAY FOR N=0
        print('     -> COMPUTE INITIAL VACUUM VESSEL BOUNDARY VALUES PSI_B...', end="")
        # COMPUTE INITIAL TOTAL PLASMA CURRENT CORRECTION FACTOR
        #self.ComputeTotalPlasmaCurrentNormalization()
        self.PSI_B[:,0] = self.ComputeBoundaryPSI()
        self.PSI_B[:,1] = self.PSI_B[:,0]
        print('Done!')
        
        ####### ASSIGN CONSTRAINT VALUES ON PLASMA BOUNDARY
        print('     -> ASSIGN INITIAL BOUNDARY VALUES...', end="")
        # ASSIGN PLASMA BOUNDARY VALUES
        self.PSI_X = self.PSIseparatrix   # INITIAL CONSTRAINT VALUE ON SEPARATRIX
        self.UpdatePlasmaBoundaryValues()
        self.UpdateVacuumVesselBoundaryValues()
        print('Done!')  
        
        print('Done!')  
        return
    
    
    