# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Author: Pau Manyer Fuertes
# Email: pau.manyer@bsc.es
# Date: July 2025
# Institution: Barcelona Supercomputing Center (BSC)
# Department: Computer Applications in Science and Engineering (CASE)
# Research Group: Nuclear Fusion 


# This script contains the Python object defining a plasma equilibrium problem, 
# modeled using the Grad-Shafranov PDE for an axisymmetrical system such as a tokamak. 
 
import os
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from ShapeFunctions import *
from Element import *
from Mesh import *
from Tokamak import *
from Magnet import *
from Greens import *
from _initialisation import *
from _critical import *
from _update import *
from _Bfield import *
from _L2error import *
from _output import *
from _plot import *
from InitialPlasmaBoundary import *
from InitialPSIGuess import *
from PlasmaCurrent import *
#from mpi4py import MPI

class GradShafranovSolver(EquilipyInitialisation,
                          EquilipyCritical,
                          EquilipyUpdate,
                          EquilipyBfield,
                          EquilipyL2error,
                          EquilipyOutput,
                          EquilipyPlotting):
    
    mu0 = 12.566370E-7           # H m-1    Magnetic permeability

    def __init__(self):
    
        ################# DEFINE ATTRIBUTES ########################   
        
        # PROBLEM CASE PARAMETERS 
        self.CASE = None
        self.FIXED_BOUNDARY = None          # PLASMA BOUNDARY FIXED BEHAVIOUR: True  or  False 
        self.GhostStabilization = False     # GHOST STABILIZATION SWITCH
        self.PARALLEL = False               # PARALLEL SIMULATION BASED ON MPI RUN (NEEDS TO RUN ON .py file)
        self.dim = None                     # DIMENSION
        
        # SIMULATION OBJECTS
        self.MESH = None                    # MESH
        self.PlasmaCurrent = None           # GRAD-SHAFRANOV PLASMA CURRENT MODEL Jphi
        self.initialPHI = None              # INITIAL/FIXED PLASMA BOUNDARY LEVEL-SET
        self.initialPSI = None              # INITIAL PSI GUESS
        self.TOKAMAK = None                 # TOKAMAK
        
        # NUMERICAL TREATMENT PARAMETERS
        self.QuadratureOrder2D = None       # NUMERICAL INTEGRATION QUADRATURE ORDER (2D)
        self.QuadratureOrder1D = None       # NUMERICAL INTEGRATION QUADRATURE ORDER (1D)
        self.it = 0                         # TOTAL NUMBER OF ITERATIONS COUNTER
        self.int_tol = None                 # INTERNAL LOOP STRUCTURE CONVERGENCE TOLERANCE
        self.ext_tol = None                 # EXTERNAL LOOP STRUCTURE CONVERGENCE TOLERANCE
        self.int_maxiter = None             # INTERNAL LOOP STRUCTURE MAXIMUM ITERATIONS NUMBER
        self.ext_maxiter = None             # EXTERNAL LOOP STRUCTURE MAXIMUM ITERATIONS NUMBER
        self.ext_cvg = None                 # EXTERNAL LOOP STRUCTURE CONVERGENCE FLAG
        self.int_cvg = None                 # INTERNAL LOOP STRUCTURE CONVERGENCE FLAG
        self.ext_it = None                  # EXTERNAL LOOP STRUCTURE ITERATIONS NUMBER
        self.int_it = None                  # INTERNAL LOOP STRUCTURE ITERATIONS NUMBER
        self.SADDLE_dist = None             # DISTANCE BETWEEN CONSECUTIVE SADDLE POINTS
        self.tol_saddle = None              # TOLERANCE FOR DISTANCE BETWEEN CONSECUTIVE ITERATION SADDLE POINTS (LETS PLASMA REGION CHANGE)
        #### BOUNDARY CONSTRAINTS
        self.beta = None                    # NITSCHE'S METHOD PENALTY TERM
        self.Nconstrainedges = None         # NUMBER OF PLAMA BOUNDARY APPROXIMATION EDGES ON WHICH CONSTRAIN BC
        #### STABILIZATION
        self.zeta = None                    # GHOST PENALTY PARAMETER
        #### OPTIMIZATION OF CRITICAL POINTS
        self.R0_axis = None                 # MAGNETIC AXIS OPTIMIZATION INITIAL GUESS R COORDINATE
        self.Z0_axis = None                 # MAGNETIC AXIS OPTIMIZATION INITIAL GUESS Z COORDINATE
        self.R0_saddle = None               # SADDLE POINT OPTIMIZATION INITIAL GUESS R COORDINATE
        self.Z0_saddle = None               # SADDLE POINT OPTIMIZATION INITIAL GUESS Z COORDINATE
        self.opti_maxiter = None            # NONLINEAR OPTIMIZATION METHOD MAXIMAL NUMBER OF ITERATION
        self.opti_tol = None                # NONLINEAR OPTIMIZATION METHOD TOLERANCE
        
        # ARRAYS
        self.LHS = None                     # GLOBAL SYSTEM LEFT-HAND-SIDE MATRIX
        self.RHS = None                     # GLOBAL SYSTEM RIGHT-HAND-SIDE VECTOR
        self.PlasmaLS = None                # PLASMA REGION GEOMETRY LEVEL-SET FUNCTION NODAL VALUES 
        self.PSI = None                     # PSI SOLUTION FIELD OBTAINED BY SOLVING CutFEM SYSTEM
        self.Xcrit = None                   # COORDINATES MATRIX FOR CRITICAL PSI POINTS
        self.PSI_0 = None                   # PSI VALUE AT MAGNETIC AXIS MINIMA
        self.PSI_X = None                   # PSI VALUE AT SADDLE POINT (PLASMA SEPARATRIX)
        self.PSI_NORM = None                # NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PSI_B = None                   # COMPUTATIONAL DOMAIN'S BOUNDARY PSI VALUES (EXTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PSI_CONV = None                # CONVERGED NORMALISED PSI SOLUTION FIELD 
        self.int_residu = None              # INTERNAL LOOP RESIDU
        self.ext_residu = None              # EXTERNAL LOOP RESIDU
        
        self.Brzfield = None                # MAGNETIC (R,Z) COMPONENTS FIELD AT INTEGRATION NODES
        
        self.PSIseparatrix = 0.0            # PSI VALUE ON SEPARATRIX
        self.PSI_NORMseparatrix = 1.0       # PSI_NORM VALUE ON SEPARATRIX
        
        ###############################
        # WORKING DIRECTORY
        pwd = os.getcwd()
        self.pwd = pwd[:-6]    # -6 CORRESPONDS TO 6 CHARACTERS IN '/TESTs'
        print('Working directory: ' + self.pwd)
        
        # INITIALISE PARALLEL PROCESSING COMMUNICATOR AND ARRAYS
        #if self.PARALLEL:
            #self.comm = MPI.COMM_WORLD
        return
    
    
    def print_all_attributes(self):
        """
        Display all attributes of the object and their corresponding values.

        This method iterates over all attributes of the instance and prints
        them in the format: attribute_name: value.
        """
        for attribute, value in vars(self).items():
            print(f"{attribute}: {value}")
        return
    
    
    ##################################################################################################
    ######################################### BOUNDARY PSI_B #########################################
    ##################################################################################################
    
    def ComputeBoundaryPSI(self):
        """
        Compute the boundary values of the poloidal flux function (PSI_B) on the vacuum vessel first wall approximation.
        Such values are obtained by summing the contributions from the external magnets, COILS and SOLENOIDS, and the 
        contribution from the plasma current itself using the Green's function.

        Output:
            PSI_B (ndarray): Array of boundary values for the poloidal flux function (PSI_B) 
                            defined over the nodes of the vacuum vessel's boundary.
        """
        # INITIALISE BOUNDARY VALUES ARRAY
        PSI_B = np.zeros([self.MESH.Nnbound])    
        
        # FOR FIXED PLASMA BOUNDARY PROBLEM, THE PSI BOUNDARY VALUES PSI_B ARE EQUAL TO THE ANALYTICAL SOLUTION
        if self.FIXED_BOUNDARY:
            for inode, node in enumerate(self.MESH.BoundaryNodes):
                # ISOLATE BOUNDARY NODE COORDINATES
                Xbound = self.MESH.X[node,:]
                # COMPUTE PSI BOUNDARY VALUES
                PSI_B[inode] = self.PlasmaCurrent.PSIanalytical(Xbound)

        # FOR THE FREE BOUNDARY PROBLEM, THE PSI BOUNDARY VALUES ARE COMPUTED BY PROJECTING THE MAGNETIC CONFINEMENT EFFECT USING THE GREENS FUNCTION FORMALISM
        else:  
            for inode, node in enumerate(self.MESH.BoundaryNodes):
                # ISOLATE BOUNDARY NODE COORDINATES
                Xbound = self.MESH.X[node,:]
                
                ##### COMPUTE PSI BOUNDARY VALUES
                # CONTRIBUTION FROM EXTERNAL MAGNETS
                PSI_B[inode] += self.mu0 * self.TOKAMAK.Psi(Xbound)
                            
                # CONTRIBUTION FROM PLASMA CURRENT  ->>  INTEGRATE OVER PLASMA REGION
                #   1. INTEGRATE IN PLASMA ELEMENTS
                for ielem in self.MESH.PlasmaElems:
                    # ISOLATE ELEMENT OBJECT
                    ELEMENT = self.MESH.Elements[ielem]
                    # INTERPOLATE ELEMENTAL PSI ON PHYSICAL GAUSS NODES
                    PSIg = ELEMENT.Ng @ ELEMENT.PSIe
                    # LOOP OVER GAUSS NODES
                    for ig in range(ELEMENT.ng):
                        PSI_B[inode] += self.mu0 * GreensFunction(Xbound, ELEMENT.Xg[ig,:])*self.PlasmaCurrent.Jphi(ELEMENT.Xg[ig,:],
                                            PSIg[ig])*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                                    
                #   2. INTEGRATE IN CUT ELEMENTS, OVER SUBELEMENT IN PLASMA REGION
                for ielem in self.MESH.PlasmaBoundElems:
                    # ISOLATE ELEMENT OBJECT
                    ELEMENT = self.MESH.Elements[ielem]
                    # INTEGRATE ON SUBELEMENT INSIDE PLASMA REGION
                    for SUBELEM in ELEMENT.SubElements:
                        if SUBELEM.Dom < 0:  # IN PLASMA REGION
                            # INTERPOLATE ELEMENTAL PSI ON PHYSICAL GAUSS NODES
                            PSIg = SUBELEM.Ng @ ELEMENT.PSIe
                            # LOOP OVER GAUSS NODES
                            for ig in range(SUBELEM.ng):
                                PSI_B[inode] += self.mu0 * GreensFunction(Xbound, SUBELEM.Xg[ig,:])*self.PlasmaCurrent.Jphi(SUBELEM.Xg[ig,:],
                                                    PSIg[ig])*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]  
        return PSI_B
    
    
    ##################################################################################################
    #################################### DOMAIN DISCRETISATION #######################################
    ##################################################################################################

    def DomainDiscretisation(self,INITIALISATION = False):
        """
        Performs the full domain discretization process for the simulation, including both 
        initialization steps and runtime classification of elements and boundary approximations.

        Tasks:
            - (If INITIALISATION=True)
                - Initializes the plasma level-set function.
                - Initializes finite elements with level-set data.
                - Converts mesh coordinates to dimensionless form (if enabled).
                - Identifies nearest neighbors and mesh boundaries.
                - Computes standard 2D quadrature rules for volume integration.
            - (Always)
                - Classifies mesh elements based on the level-set function.
                - Approximates the plasma boundary using cut elements.
                - Computes adapted quadratures over cut elements and interfaces.
                - Assigns Dirichlet constraints along the approximated plasma boundary.
                - Identifies ghost faces and computes associated quadrature rules (if enabled).
        """
        
        print('PERFORM DOMAIN DISCRETISATION...')
        
        ####################### INITIALISATION TASKS ########################
        
        # THE FOLLOWING COMMANDS ARE ONLY EXECUTED WHEN INITIALISING THE SIMULATION
        if INITIALISATION:
            print('INITIALISATION TASKS...')
        
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
            
            # IDENTIFY ELEMENTS MESH RELATION   
            print("     -> IDENTIFY ELEMENTS MESH RELATION...", end="")
            self.MESH.IdentifyBoundaries()
            self.MESH.IdentifyNearestNeighbors()
            print("Done!")
            
            # COMPUTE STANDARD 2D QUADRATURE ENTITIES FOR ALL ELEMENTS 
            print('     -> COMPUTE STANDARD NUMERICAL INTEGRATION QUADRATURES...', end="")
            self.MESH.ComputeStandardQuadratures(self.QuadratureOrder2D)
            print("Done!")
                
            # DEFINE STANDARD SURFACE QUADRATURE NUMBER OF INTEGRATION NODES
            self.nge = self.MESH.Elements[0].ng
            
            print("Done!")
            
        #####################################################################
        ################# RUN-TIME DOMAIN DISCRETISATION ####################
        
        # CLASSIFY ELEMENTS   
        print("     -> CLASSIFY ELEMENTS...", end="")
        self.PlasmaLS = self.MESH.ClassifyElements(self.PlasmaLS)
        print("Done!")

        # COMPUTE PLASMA BOUNDARY APPROXIMATION
        print("     -> APPROXIMATE PLASMA BOUNDARY INTERFACE...", end="")
        self.MESH.ObtainPlasmaBoundaryElementalPath()
        self.MESH.ObtainPlasmaBoundaryActiveElements(numelements = self.Nconstrainedges)
        self.MESH.ComputePlasmaBoundaryApproximation()
        print("Done!")
        
        # COMPUTE ADAPTED QUADRATURE ENTITIES FOR CUT ELEMENTS
        print('     -> COMPUTE PLASMA BOUNDARY APPROXIMATION QUADRATURES...', end="")
        self.MESH.ComputeAdaptedQuadratures(self.QuadratureOrder2D,self.QuadratureOrder1D)
        print("Done!")
        
        # FIXED CONSTRAINTS PSI_P ON PLASMA BOUNDARY
        print('     -> ASSIGN PLASMA BOUNDARY CONSTRAINT VALUES...', end="") 
        self.FixElementalPSI_P()     
        print('Done!')          
            
        # CUT ELEMENTS GHOST FACES 
        if self.GhostStabilization:
            # IDENTIFY GHOST FACES
            print("     -> IDENTIFY GHOST FACES...", end="")
            self.MESH.ComputePlasmaBoundaryGhostFaces()
            print("Done!")
            # COMPUTE QUADRATURES FOR GHOST FACES ON PLASMA BOUNDARY ELEMENTS
            print("     -> COMPUTE GHOST FACES QUADRATURES...", end="")
            self.MESH.ComputeGhostFacesQuadratures(self.QuadratureOrder1D)
            print("Done!")
      
        print('Done!')
        return
        
    
    ##################################################################################################
    ########################################## INTEGRATION ###########################################
    ##################################################################################################
    
    def IntegratePlasmaDomain(self,fun,PSI_INDEPENDENT=False):
        """ 
        Function that integrates function fun over the plasma region. 
        """ 
        
        integral = 0
        if not PSI_INDEPENDENT:
            # INTEGRATE OVER PLASMA ELEMENTS
            for ielem in self.MESH.PlasmaElems:
                # ISOLATE ELEMENT
                ELEMENT = self.MESH.Elements[ielem]
                # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                PSIg = ELEMENT.Ng @ ELEMENT.PSIe
                # LOOP OVER GAUSS NODES
                for ig in range(ELEMENT.ng):
                    integral += fun(ELEMENT.Xg[ig,:],PSIg[ig])*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                        
            # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
            for ielem in self.MESH.PlasmaBoundElems:
                # ISOLATE ELEMENT
                ELEMENT = self.MESH.Elements[ielem]
                # LOOP OVER SUBELEMENTS
                for SUBELEM in ELEMENT.SubElements:
                    # INTEGRATE IN SUBDOMAIN INSIDE PLASMA REGION
                    if SUBELEM.Dom < 0:
                        # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                        PSIg = SUBELEM.Ng @ ELEMENT.PSIe
                        # LOOP OVER GAUSS NODES
                        for ig in range(SUBELEM.ng):
                            integral += fun(SUBELEM.Xg[ig,:],PSIg[ig])*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]
        else:
            # INTEGRATE OVER PLASMA ELEMENTS
            for ielem in self.MESH.PlasmaElems:
                # ISOLATE ELEMENT
                ELEMENT = self.MESH.Elements[ielem]
                # LOOP OVER GAUSS NODES
                for ig in range(ELEMENT.ng):
                    integral += fun(ELEMENT.Xg[ig,:])*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                        
            # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
            for ielem in self.MESH.PlasmaBoundElems:
                # ISOLATE ELEMENT
                ELEMENT = self.MESH.Elements[ielem]
                # LOOP OVER SUBELEMENTS
                for SUBELEM in ELEMENT.SubElements:
                    # INTEGRATE IN SUBDOMAIN INSIDE PLASMA REGION
                    if SUBELEM.Dom < 0:
                        # LOOP OVER GAUSS NODES
                        for ig in range(SUBELEM.ng):
                            integral += fun(SUBELEM.Xg[ig,:])*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]
                             
        return integral
    
    
    def IntegrateGhostStabilizationTerms(self):
        """
        Integrates ghost stabilization terms across internal ghost faces between elements to
        stabilize the numerical solution in high-order finite element methods (e.g., CutFEM).

        Process:
            - Iterates over all ghost faces defined in the mesh (`self.MESH.GhostFaces`).
            - For each ghost face, retrieves adjacent elements and the associated face information.
            - Constructs the local stabilization matrix (`LHSe`) and RHS vector (`RHSe`) across both elements.
            - Computes ghost penalty contributions using gradient jumps across shared faces.
            - Applies Dirichlet boundary conditions if any boundary nodes are present.
            - Assembles the local contributions into the global system matrices `self.LHS` and `self.RHS`.

        Stabilization:
            - Uses gradient jump terms weighted by a user-defined penalty parameter `zeta`.
            - Alternative option (commented) allows stabilization based on solution jumps.
        """
        if self.out_elemsys:
            self.file_elemsys.write('GHOST_FACES\n')
            
        for ghostface in self.MESH.GhostFaces:
            # ISOLATE ADJACENT ELEMENTS
            ELEMENT0 = self.MESH.Elements[ghostface[1][0]]
            ELEMENT1 = self.MESH.Elements[ghostface[2][0]]
            # ISOLATE COMMON EDGE 
            FACE0 = ELEMENT0.GhostFaces[ghostface[1][2]]
            FACE1 = ELEMENT1.GhostFaces[ghostface[2][2]]
            # DEFINE ELEMENTAL MATRIX
            LHSe = np.zeros([ELEMENT0.n+ELEMENT1.n,ELEMENT0.n+ELEMENT1.n])
            RHSe = np.zeros([ELEMENT0.n+ELEMENT1.n])
            
            # COMPUTE ADEQUATE GHOST PENALTY TERM
            penalty = self.zeta*max(ELEMENT0.length,ELEMENT1.length)  #**(1-2*self.MESH.ElOrder)
            #penalty = self.zeta
            
            # LOOP OVER GAUSS INTEGRATION NODES
            for ig in range(FACE0.ng):  
                # SHAPE FUNCTIONS NORMAL GRADIENT IN PHYSICAL SPACE
                n_dot_Ngrad0 = FACE0.NormalVec@FACE0.invJg[ig,:,:]@np.array([FACE0.dNdxig[ig,:],FACE0.dNdetag[ig,:]])
                n_dot_Ngrad1 = FACE1.NormalVec@FACE1.invJg[ig,:,:]@np.array([FACE1.dNdxig[ig,:],FACE1.dNdetag[ig,:]])
                n_dot_Ngrad = np.concatenate((n_dot_Ngrad0,n_dot_Ngrad1), axis=0)
                    
                # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM    
                for i in range(ELEMENT0.n+ELEMENT1.n):  # ROWS ELEMENTAL MATRIX
                    for j in range(ELEMENT0.n+ELEMENT1.n):  # COLUMNS ELEMENTAL MATRIX
                        ### GHOST PENALTY TERM  (GRADIENT JUMP) [ jump(nabla(N_i))*jump(nabla(N_j)) *(Jacobiano) ]  
                        LHSe[i,j] += penalty*n_dot_Ngrad[i]*n_dot_Ngrad[j] * FACE0.detJg1D[ig] * FACE0.Wg[ig]
                        ### GHOST PENALTY TERM  (SOLUTION JUMP) [ jump(N_i)*jump(N_j)]
                        #LHSe[i,j] += penalty*FACE0.Ng[ig,i]*FACE0.Ng[ig,j] * FACE0.detJg1D[ig] * FACE0.Wg[ig] 
                        
            # PRESCRIBE BC
            if not type(ELEMENT0.Teboun) == type(None) or not type(ELEMENT1.Teboun) == type(None):
                if not type(ELEMENT0.Teboun) == type(None) and type(ELEMENT1.Teboun) == type(None):
                    Tbounghost = ELEMENT0.Teboun[0].copy()
                    PSI_Bghost = ELEMENT0.PSI_Be.copy()
                elif type(ELEMENT0.Teboun) == type(None) and not type(ELEMENT1.Teboun) == type(None):
                    Tbounghost = ELEMENT1.Teboun[0].copy()
                    PSI_Bghost = ELEMENT1.PSI_Be.copy()
                elif not type(ELEMENT0.Teboun) == type(None) and not type(ELEMENT1.Teboun) == type(None):
                    Tbounghost = np.concatenate((ELEMENT0.Teboun[0],[ELEMENT0.n+index for index in ELEMENT1.Teboun[0]]), axis=0)
                    PSI_Bghost = np.concatenate((ELEMENT0.PSI_Be, ELEMENT1.PSI_Be), axis=0)
                
                for ibounode in Tbounghost:
                    adiag = LHSe[ibounode,ibounode]
                    # PASS MATRIX COLUMN TO RIGHT-HAND-SIDE
                    RHSe -= LHSe[:,ibounode]*PSI_Bghost[ibounode]
                    # NULLIFY BOUNDARY NODE ROW
                    LHSe[ibounode,:] = 0
                    # NULLIFY BOUNDARY NODE COLUMN
                    LHSe[:,ibounode] = 0
                    # PRESCRIBE BOUNDARY CONDITION ON BOUNDARY NODE
                    if abs(adiag) > 0:
                        LHSe[ibounode,ibounode] = adiag
                        RHSe[ibounode] = adiag*PSI_Bghost[ibounode]
                    else:
                        LHSe[ibounode,ibounode] = 1
                        RHSe[ibounode] = PSI_Bghost[ibounode]
                                                     
            # ASSEMBLE ELEMENTAL CONTRIBUTIONS INTO GLOBAL SYSTEM
            Tghost = np.concatenate((ELEMENT0.Te,ELEMENT1.Te), axis=0)
            for i in range(ELEMENT0.n+ELEMENT1.n):   # ROWS ELEMENTAL MATRIX
                for j in range(ELEMENT0.n+ELEMENT1.n):   # COLUMNS ELEMENTAL MATRIX
                    self.LHS[Tghost[i],Tghost[j]] += LHSe[i,j]     
                self.RHS[Tghost[i]] += RHSe[i]
                    
            if self.out_elemsys:
                self.file_elemsys.write("ghost face {:d} common to elements {:d} {:d}\n".format(0,ELEMENT0.index,ELEMENT1.Dom))
                self.file_elemsys.write('elmat\n')
                np.savetxt(self.file_elemsys,LHSe,delimiter=',',fmt='%e')
                                   
        return
    
    
    def AssembleGlobalSystem(self):
        """
        Assembles the global system of equations (LHS matrix and RHS vector) for the finite element
        formulation, accounting for standard, cut, and interface elements using domain-specific 
        quadrature rules and stabilization techniques.

        Tasks:
            - Initializes global sparse system matrices: self.LHS (global stiffness) and self.RHS (global load).
            - Iterates over:
                1. Non-cut elements (standard quadrature).
                2. Cut-elements (subelement tessellation with adapted quadrature).
                3. Interface edges in cut-elements (1D quadrature over the interface).
                4. Internal ghost faces (optional ghost penalty stabilization).

        Sets:
            - self.LHS : Global stiffness matrix (sparse).
            - self.RHS : Global right-hand side vector.
        """
        
        # INITIALISE GLOBAL SYSTEM MATRICES
        self.LHS = lil_matrix((self.MESH.Nn,self.MESH.Nn))  # FOR SPARSE MATRIX, USE LIST-OF-LIST FORMAT 
        self.RHS = np.zeros([self.MESH.Nn,1])
        
        # OPEN ELEMENTAL MATRICES OUTPUT FILE
        if self.out_elemsys:
            self.file_elemsys.write('NON_CUT_ELEMENTS\n')
        
        # INTEGRATE OVER THE SURFACE OF ELEMENTS WHICH ARE NOT CUT BY ANY INTERFACE (STANDARD QUADRATURES)
        print("     Integrate over non-cut elements...", end="")
        
        for ielem in self.MESH.NonCutElems: 
            # ISOLATE ELEMENT 
            ELEMENT = self.MESH.Elements[ielem]  
            # COMPUTE SOURCE TERM (PLASMA CURRENT)  mu0*R*Jphi  IN PLASMA REGION NODES
            SourceTermg = np.zeros([ELEMENT.ng])
            if ELEMENT.Dom < 0:
                # MAP PSI VALUES FROM ELEMENT NODES TO GAUSS NODES
                PSIg = ELEMENT.Ng @ ELEMENT.PSIe
                for ig in range(ELEMENT.ng):
                    SourceTermg[ig] = self.PlasmaCurrent.SourceTerm(ELEMENT.Xg[ig,:],PSIg[ig])
                    
            # COMPUTE ELEMENTAL MATRICES
            LHSe, RHSe = ELEMENT.IntegrateElementalDomainTerms(SourceTermg)
                
            # PRESCRIBE BC:
            if not type(ELEMENT.Teboun) == type(None):
                LHSe, RHSe = ELEMENT.PrescribeDirichletBC(LHSe,RHSe)
            
            if self.out_elemsys:
                self.file_elemsys.write("elem {:d} {:d}\n".format(ELEMENT.index+1,ELEMENT.Dom))
                self.file_elemsys.write('elmat\n')
                for irow in range(ELEMENT.n):
                    values = " ".join("{:.6e}".format(val) for val in LHSe[irow,:])
                    self.file_elemsys.write("{}\n".format(values))
                self.file_elemsys.write('elrhs\n')
                values = " ".join("{:.6e}".format(val) for val in RHSe)
                self.file_elemsys.write("{}\n".format(values))
            
            # ASSEMBLE ELEMENTAL CONTRIBUTIONS INTO GLOBAL SYSTEM
            for i in range(ELEMENT.n):   # ROWS ELEMENTAL MATRIX
                for j in range(ELEMENT.n):   # COLUMNS ELEMENTAL MATRIX
                    self.LHS[ELEMENT.Te[i],ELEMENT.Te[j]] += LHSe[i,j]
                self.RHS[ELEMENT.Te[i]] += RHSe[i]
                
        print("Done!")
        
        if self.out_elemsys:
            self.file_elemsys.write('END_NON_CUT_ELEMENTS\n')
            self.file_elemsys.write('CUT_ELEMENTS_SURFACE\n')
        
        # INTEGRATE OVER THE SURFACES OF SUBELEMENTS IN ELEMENTS CUT BY INTERFACES (ADAPTED QUADRATURES)
        print("     Integrate over cut-elements subelements...", end="")
        
        for ielem in self.MESH.PlasmaBoundElems:
            # ISOLATE ELEMENT 
            ELEMENT = self.MESH.Elements[ielem]
            # NOW, EACH INTERFACE ELEMENT IS DIVIDED INTO SUBELEMENTS ACCORDING TO THE POSITION OF THE APPROXIMATED INTERFACE ->> TESSELLATION
            # ON EACH SUBELEMENT THE WEAK FORM IS INTEGRATED USING ADAPTED NUMERICAL INTEGRATION QUADRATURES
            ####### COMPUTE DOMAIN TERMS
            # LOOP OVER SUBELEMENTS 
            for SUBELEM in ELEMENT.SubElements:  
                # COMPUTE SOURCE TERM (PLASMA CURRENT)  mu0*R*Jphi  IN PLASMA REGION NODES
                SourceTermg = np.zeros([SUBELEM.ng])
                if SUBELEM.Dom < 0:
                    # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PSIg = SUBELEM.Ng @ ELEMENT.PSIe
                    for ig in range(SUBELEM.ng):
                        SourceTermg[ig] = self.PlasmaCurrent.SourceTerm(SUBELEM.Xg[ig,:],PSIg[ig])
                        
                # COMPUTE ELEMENTAL MATRICES
                LHSe, RHSe = SUBELEM.IntegrateElementalDomainTerms(SourceTermg)
                    
                # PRESCRIBE BC:
                if not type(ELEMENT.Teboun) == type(None):
                    LHSe, RHSe = ELEMENT.PrescribeDirichletBC(LHSe,RHSe)
                
                if self.out_elemsys:
                    self.file_elemsys.write("elem {:d} {:d} subelem {:d} {:d}\n".format(ELEMENT.index+1,ELEMENT.Dom,SUBELEM.index+1,SUBELEM.Dom))
                    self.file_elemsys.write('elmat\n')
                    for irow in range(ELEMENT.n):
                        values = " ".join("{:.6e}".format(val) for val in LHSe[irow,:])
                        self.file_elemsys.write("{}\n".format(values))
                    self.file_elemsys.write('elrhs\n')
                    values = " ".join("{:.6e}".format(val) for val in RHSe)
                    self.file_elemsys.write("{}\n".format(values))
                
                # ASSEMBLE ELEMENTAL CONTRIBUTIONS INTO GLOBAL SYSTEM
                for i in range(SUBELEM.n):   # ROWS ELEMENTAL MATRIX
                    for j in range(SUBELEM.n):   # COLUMNS ELEMENTAL MATRIX
                        self.LHS[SUBELEM.Te[i],SUBELEM.Te[j]] += LHSe[i,j]
                    self.RHS[SUBELEM.Te[i]] += RHSe[i]
                
        print("Done!")
        
        if self.out_elemsys:
            self.file_elemsys.write('END_CUT_ELEMENTS_SURFACE\n')
            self.file_elemsys.write('CUT_ELEMENTS_INTERFACE\n')
        
        # INTEGRATE OVER THE CUT EDGES IN ELEMENTS CUT BY INTERFACES (ADAPTED QUADRATURES)
        print("     Integrate along cut-elements interface edges...", end="")
        
        for ielem in self.MESH.PlasmaBoundActiveElems:
            # ISOLATE ELEMENT 
            ELEMENT = self.MESH.Elements[ielem]
            # COMPUTE ELEMENTAL MATRICES
            LHSe,RHSe = ELEMENT.IntegrateElementalInterfaceTerms(self.beta)
                
            # PRESCRIBE BC:
            if not type(ELEMENT.Teboun) == type(None):
                LHSe, RHSe = ELEMENT.PrescribeDirichletBC(LHSe,RHSe)
                
            if self.out_elemsys:
                self.file_elemsys.write("elem {:d} {:d}\n".format(ELEMENT.index+1,ELEMENT.Dom))
                self.file_elemsys.write('elmat\n')
                for irow in range(ELEMENT.n):
                    values = " ".join("{:.6e}".format(val) for val in LHSe[irow,:])
                    self.file_elemsys.write("{}\n".format(values))
                self.file_elemsys.write('elrhs\n')
                values = " ".join("{:.6e}".format(val) for val in RHSe)
                self.file_elemsys.write("{}\n".format(values))
            
            # ASSEMBLE ELEMENTAL CONTRIBUTIONS INTO GLOBAL SYSTEM
            for i in range(len(ELEMENT.Te)):   # ROWS ELEMENTAL MATRIX
                for j in range(len(ELEMENT.Te)):   # COLUMNS ELEMENTAL MATRIX
                    self.LHS[ELEMENT.Te[i],ELEMENT.Te[j]] += LHSe[i,j]
                self.RHS[ELEMENT.Te[i]] += RHSe[i]
        
        print("Done!")
        
        if self.out_elemsys:
            self.file_elemsys.write('END_CUT_ELEMENTS_INTERFACE\n')
      
        # INTEGRATE GHOST PENALTY TERM OVER CUT ELEMENTS INTERNAL CUT EDGES
        if self.GhostStabilization:
            print("     Integrate ghost penalty term along cut elements internal ghost faces...", end="")
            self.IntegrateGhostStabilizationTerms()
            print("Done!") 
            
        # WRITE GLOBAL SYSTEM MATRICES
        if self.out_elemsys:
            self.file_globalsys.write('RHS_VECTOR\n')
            for inode in range(self.MESH.Nn):
                self.file_globalsys.write("{:d} {:f}\n".format(inode+1, self.RHS[inode,0]))
            self.file_globalsys.write('END_RHS_VECTOR\n')
                
            self.file_globalsys.write('LHS_MATRIX\n')
            inode = 0
            for irow in range(self.MESH.Nn):
                for jcol in range(self.MESH.Nn):
                    if self.LHS[irow,jcol] != 0:
                        inode += 1
                        self.file_globalsys.write("{:d} {:d} {:d} {:f}\n".format(inode, irow+1, jcol+1, self.LHS[irow,jcol]))
            self.file_globalsys.write('END_LHS_MATRIX\n')
        
        print("Done!")   
        return
    
    
    ##################################################################################################
    ############################################ SOLVER ##############################################
    ##################################################################################################
    
    def SolveSystem(self):
        """
        Solves the global linear system of equations for the scalar field variable PSI.

        Sets:
            - self.PSI : Solution vector (nodal values) of the scalar field, reshaped as a column vector.
        """
        self.PSI = spsolve(self.LHS.tocsr(), self.RHS).reshape([self.MESH.Nn,1])
        return

    
    ##################################################################################################
    ######################################## MAIN ALGORITHM ##########################################
    ##################################################################################################
    
    def EQUILI(self,CASE):
        """
        Runs the main equilibrium solver loop for the Grad-Shafranov problem.

        Input:
            - CASE (str): Identifier name for the simulation case used for output directory setup.

        Workflow:
            - Prepares output directories and initializes all necessary fields.
            - Writes initial condition data (PSI, PSI_B, plasma region, boundary conditions).
            - Executes a two-level iterative process:
                * External loop updates boundary values.
                * Internal loop solves the Grad-Shafranov equation until convergence.
            - Within each internal iteration:
                1. Assembles and solves the global system for PSI.
                2. Computes and writes critical PSI values (if free-boundary).
                3. Normalizes PSI and writes the normalized field.
                4. Checks convergence of PSI_NORM.
                5. Updates plasma region geometry (if required).
                6. Updates normalized fields and computes plasma current normalization.
            - After each external iteration:
                * Updates and writes boundary values (PSI_B).
                * Checks convergence of PSI_B and writes residuals.

        Final Steps:
            - Plots final solution (if enabled).
            - If fixed-boundary with a custom current model, computes and writes L2 error.
            - Saves results and closes output files.

        Notes:
            - Convergence of the internal loop is based on the PSI_NORM field.
            - Convergence of the external loop is based on changes in PSI_B.
            - The equilibrium solution is assumed converged when both criteria are satisfied or the max iterations are reached.

        Sets:
            - self.PSI : Final scalar field solution (nodal values).
            - self.PSI_B : Final boundary values used in computation.
            - self.PSI_NORM : Normalized scalar field.
            - Output files and plots (as per configuration).
        """
        
        print("PREPARE OUTPUT DIRECTORY...",end='')
        # INITIALISE SIMULATION CASE NAME
        self.CASE = CASE
        self.InitialiseOutput()
        # OPEN OUTPUT FILES
        self.openOUTPUTfiles()   
        print('Done!')
        
        # INITIALISE PSI UNKNOWNS
        self.it = 0
        self.ext_it = 0
        self.int_it = 0
        self.InitialisePSI_B()          
        
        # WRITE INITIAL SIMULATION DATA
        print("WRITE INITIAL SIMULATION DATA...",end='')
        self.writeboundaries()
        if self.GhostStabilization:
            self.writeNeighbours()
        self.writePlasmaBoundaryData()
        self.writeQuadratures()
        self.writePSI()
        self.writePSI_B()
        self.writePlasmaBC()
        print('Done!')
        
        if self.plotPSI:
            self.PlotSolutionPSI()  # PLOT INITIAL SOLUTION

        # START DOBLE LOOP STRUCTURE
        print('START ITERATION...')
        self.ext_cvg = False
        self.ext_it = 0
        
        #######################################################
        ################## EXTERNAL LOOP ######################
        #######################################################
        while (self.ext_cvg == False and self.ext_it < self.ext_maxiter):
            self.ext_it += 1
            self.int_cvg = False
            self.int_it = 0
            #######################################################
            ################## INTERNAL LOOP ######################
            #######################################################
            while (self.int_cvg == False and self.int_it < self.int_maxiter):
                self.int_it += 1
                self.it += 1
                print('OUTER ITERATION = '+str(self.ext_it)+' , INNER ITERATION = '+str(self.int_it))
                print('     Total current = ', self.IntegratePlasmaDomain(self.PlasmaCurrent.Jphi))
                
                if self.plotelemsClas:
                    self.PlotClassifiedElements(GHOSTFACES=self.GhostStabilization)
                    
                # INNER LOOP ALGORITHM: SOLVING GRAD-SHAFRANOV BVP
                self.AssembleGlobalSystem()                 # 1. ASSEMBLE SYSTEM
                self.SolveSystem()                          # 2. SOLVE SYSTEM  ->> PSI
                if not self.FIXED_BOUNDARY:
                    self.ComputeCriticalPSI()               # 3. COMPUTE CRITICAL VALUES   PSI_0 AND PSI_X
                    self.writePSIcrit()                     #       -> WRITE CRITICAL POINTS
                self.NormalisePSI()                         # 4. NORMALISE PSI RESPECT TO CRITICAL VALUES  ->> PSI_NORM[:,1] 
                self.writePSI()                             #       -> WRITE NEW SOLUTION PSI_NORM[:,1]         
                if self.plotPSI:
                    self.PlotSolutionPSI()                  #       -> PLOT SOLUTION AND NORMALISED SOLUTION
                self.CheckConvergence('PSI_NORM')           # 5. CHECK CONVERGENCE OF PSI_NORM FIELD
                self.writeresidu("INTERNAL")                #       -> WRITE INTERNAL LOOP RESIDU
                
                                                            # 6. UPDATE PLASMA REGION IF:  
                if not self.FIXED_BOUNDARY:                 #                                       - FREE-BOUNDARY PROBLEM
                    self.SADDLE_dist = np.linalg.norm(self.Xcrit[1,1,:-1]-self.Xcrit[0,1,:-1])    # - DISTANCE BETWEEN SADDLE POINTS < tol_saddle
                    if self.SADDLE_dist > self.tol_saddle:
                        self.ComputePSILevelSet()           #       -> COMPUTE NEW PLASMA BOUNDARY LEVEL-SET
                        self.UpdateElementalPlasmaLevSet()  #       -> UPDATE ELEMENTAL PLASMA LEVEL-SET VALUES 
                        self.DomainDiscretisation()         #       -> DISCRETISE DOMAIN ACCORDING TO NEW PLASMA REGION
                        self.writePlasmaBoundaryData()      #       -> WRITE NEW PLASMA REGION DATA
                    else:
                        print("Plasma region unchanged: distance between consecutive saddle points = ", self.SADDLE_dist)
                        print(" ")   
                
                self.UpdatePSI_NORM()                       # 7. UPDATE PSI_NORM ARRAY
                self.UpdateElementalPSI()                   # 8. UPDATE PSI_NORM VALUES IN CORRESPONDING ELEMENTS 
                self.PlasmaCurrent.Normalise()              # 9 .COMPUTE PLASMA CURRENT NORMALISATION FACTOR ACCORDING TO NEW PLASMA REGION
                
                #######################################################
                ################ END INTERNAL LOOP ####################
                #######################################################
                
            if not self.FIXED_BOUNDARY:
                print('COMPUTE COMPUTATIONAL BOUNDARY VALUES PSI_B...', end="")
                self.PSI_B[:,1] = self.ComputeBoundaryPSI()     # COMPUTE COMPUTATIONAL BOUNDARY VALUES PSI_B WITH INTERNALLY CONVERGED PSI_NORM
                self.writePSI_B()                               #       -> WRITE NEW BOUNDARY CONDITIONS PSI_B
                print('Done!')
            
                print('UPDATE COMPUTATIONAL DOMAIN BOUNDARY VALUES...', end="")
                self.UpdateElementalPSI_B()                     # UPDATE BOUNDARY CONDITIONS PSI_B ON BOUNDARY ELEMENTS
                print('Done!')
            
            self.CheckConvergence('PSI_B')                  # CHECK CONVERGENCE OF COMPUTATIONAL BOUNDARY PSI VALUES  (PSI_B)
            self.writeresidu("EXTERNAL")                    # WRITE EXTERNAL LOOP RESIDU 
            self.UpdatePSI_B()                              # UPDATE PSI_B VALUES 
            
            #######################################################
            ################ END EXTERNAL LOOP ####################
            #######################################################
            
        print('SOLUTION CONVERGED')
        if self.plotPSI:
            self.PlotSolutionPSI()
        
        if self.FIXED_BOUNDARY and self.PlasmaCurrent.CURRENT_MODEL != self.PlasmaCurrent.JARDIN_CURRENT:
            self.ComputeErrorField()
            self.ComputeL2errorPlasma()
            self.writeerror()
        
        self.closeOUTPUTfiles()
        self.writeSimulationPickle()
        return
    
    
    
    
    
    
    
    
    