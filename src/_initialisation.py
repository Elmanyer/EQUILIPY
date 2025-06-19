import numpy as np
from math import ceil
from Element import *

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
        self.Xcrit[0,0,:-1] = np.array([self.EXTR_R0,self.EXTR_Z0])
        self.Xcrit[0,1,:-1] = np.array([self.SADD_R0,self.SADD_Z0])
        
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
        
        self.PlasmaLS = np.zeros([self.Mesh.Nn,2])
        self.PlasmaLSstar = np.zeros([self.Mesh.Nn,2])
        self.PlasmaLS[:,0] = self.initialPHI.PHI0
        self.PlasmaLS[:,1] = self.PlasmaLS[:,0]
        return 
    
    
    def InitialiseElements(self):
        """ 
        Function initialising attribute ELEMENTS which is a list of all elements in the mesh. 
        """
        self.Mesh.Elements = [Element(index = e,
                                    ElType = self.Mesh.ElType,
                                    ElOrder = self.Mesh.ElOrder,
                                    Xe = self.Mesh.X[self.Mesh.T[e,:],:],
                                    Te = self.Mesh.T[e,:],
                                    PlasmaLSe = self.PlasmaLS[self.Mesh.T[e,:],1]) for e in range(self.Mesh.Ne)]
        
        # COMPUTE MESH MEAN SIZE
        self.Mesh.meanArea, self.Mesh.meanLength = self.ComputeMeshElementsMeanSize()
        print("         · MESH ELEMENTS MEAN AREA = " + str(self.Mesh.meanArea) + " m^2")
        print("         · MESH ELEMENTS MEAN LENGTH = " + str(self.Mesh.meanLength) + " m")
        print("         · RECOMMENDED NITSCHE'S PENALTY PARAMETER VALUE    beta ~ C·" + str(self.Mesh.ElOrder**2/self.Mesh.meanLength))
        return
    
    
    def DimensionlessCoordinates(self): 
        for ELEMENT in self.Mesh.Elements:
            ELEMENT.Xe /= self.PlasmaCurrent.R0
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
        
        ####### INITIALISE PSI VECTORS
        print('     -> INITIALISE PSI ARRAYS...', end="")
        # INITIALISE ITERATIVE UPDATED ARRAYS
        self.PSI = np.zeros([self.Mesh.Nn],dtype=float)            # SOLUTION FROM SOLVING CutFEM SYSTEM OF EQUATIONS (INTERNAL LOOP)       
        self.PSI_NORMstar = np.zeros([self.Mesh.Nn,2],dtype=float) # UNRELAXED NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        self.PSI_NORM = np.zeros([self.Mesh.Nn,2],dtype=float)     # RELAXED NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        self.PSI_CONV = np.zeros([self.Mesh.Nn],dtype=float)       # CONVERGED SOLUTION FIELD
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
        return
    
        
    def InitialisePSI_B(self):
        ####### INITIALISE PSI BOUNDARY VECTOR
        self.PSI_B = np.zeros([self.Mesh.Nnbound,2],dtype=float)   # VACUUM VESSEL FIRST WALL PSI VALUES (EXTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)    
        
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
        return
    
    
    def InitialiseMESH(self):
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
        
        # MESH INPUT FILES
        self.mesh_folder = self.pwd + '/MESHES/' + MESH
        
        print("READ MESH FILES...")
        self.ReadMeshFile()
        self.ReadFixFile()
        print('Done!')
        
        
        print("     -> INITIALISE SIMULATION PARAMETERS...", end="")
        self.InitialiseParameters()
        self.InitialisePickleLists()
        print('Done!')
        
        # INITIALISE LEVEL-SET FUNCTION
        print("     -> INITIALISE LEVEL-SET...", end="")
        self.InitialisePlasmaLevelSet()
        print('Done!')
        
        # INITIALISE ELEMENTS 
        print("     -> INITIALISE ELEMENTS...")
        self.InitialiseElements()
        if type(self.PlasmaCurrent) != type(None) and self.PlasmaCurrent.DIMENSIONLESS:
            self.DimensionlessCoordinates()
        print('     Done!')
        
        # CLASSIFY ELEMENTS   
        print("     -> CLASSIFY ELEMENTS...", end="")
        self.IdentifyNearestNeighbors()
        self.IdentifyBoundaries()
        self.ClassifyElements()
        print("Done!")

        # COMPUTE PLASMA BOUNDARY APPROXIMATION
        print("     -> APPROXIMATE PLASMA BOUNDARY INTERFACE...", end="")
        self.ComputePlasmaBoundaryApproximation()
        print("Done!")
        
        # IDENTIFY GHOST FACES 
        if self.GhostStabilization:
            print("     -> IDENTIFY GHOST FACES...", end="")
            self.ComputePlasmaBoundaryGhostFaces()
            print("Done!")
        
        # COMPUTE NUMERICAL INTEGRATION QUADRATURES
        print('     -> COMPUTE NUMERICAL INTEGRATION QUADRATURES...', end="")
        self.ComputeIntegrationQuadratures()
        print('Done!')
        
        # COMPUTE NUMBER OF NODES ON PLASMA BOUNDARY APPROXIMATION
        self.Mesh.NnPB = self.ComputePlasmaBoundaryNumberNodes()
        
        print('Done!')
        return  