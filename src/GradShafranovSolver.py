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
# Date: October 2024
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
from mpi4py import MPI

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
        self.it_plasma = None               # ITERATION AFTER WHICH THE PLASMA REGION CAN BE UPDATED
        #### BOUNDARY CONSTRAINTS
        self.beta = None                    # NITSCHE'S METHOD PENALTY TERM
        #### STABILIZATION
        self.PSIrelax = False               # PSI SOLUTION AITKEN RELAXATION SWITCH
        self.lambdaPSI = None               # PHI LEVEL-SET AITKEN RELAXATION PARAMETER 
        self.lambdamin = None               # PHI LEVEL-SET AITKEN RELAXATION PARAMETER MINIMAL VALUE
        self.lambdamax = None               # PHI LEVEL-SET AITKEN RELAXATION PARAMETER MAXIMAL VALUE 
        self.lambda0 = None                 # INITIAL AIKTEN'S SCHEME RELAXATION CONSTANT  (alpha0 = 1 - lambda0)
        self.PHIrelax = False               # PHI LEVEL-SET AITKEN RELAXATION SWITCH
        self.alphaPHI = None                # PHI LEVEL-SET AITKEN RELAXATION INITIAL PARAMETER
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
        self.PlasmaLS = None                # PLASMA REGION GEOMETRY LEVEL-SET FUNCTION NODAL VALUES AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PlasmaLSstar = None            # UNRELAXED PLASMA REGION GEOMETRY LEVEL-SET FUNCTION NODAL VALUES
        self.PSI = None                     # PSI SOLUTION FIELD OBTAINED BY SOLVING CutFEM SYSTEM
        self.Xcrit = None                   # COORDINATES MATRIX FOR CRITICAL PSI POINTS
        self.PSI_0 = None                   # PSI VALUE AT MAGNETIC AXIS MINIMA
        self.PSI_X = None                   # PSI VALUE AT SADDLE POINT (PLASMA SEPARATRIX)
        self.PSI_NORMstar = None            # UNRELAXED NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PSI_NORM = None                # RELAXED NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PSI_B = None                   # VACUUM VESSEL WALL PSI VALUES (EXTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PSI_CONV = None                # CONVERGED NORMALISED PSI SOLUTION FIELD 
        self.int_residu = None              # INTERNAL LOOP RESIDU
        self.ext_residu = None              # EXTERNAL LOOP RESIDU
        
        self.Brzfield = None                # MAGNETIC (R,Z) COMPONENTS FIELD AT INTEGRATION NODES
        
        self.PSIseparatrix = 1.0
        
        ###############################
        # WORKING DIRECTORY
        pwd = os.getcwd()
        self.pwd = pwd[:-6]
        print('Working directory: ' + self.pwd)
        
        # INITIALISE PARALLEL PROCESSING COMMUNICATOR AND ARRAYS
        if self.PARALLEL:
            self.comm = MPI.COMM_WORLD
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
                for MAGNET in self.TOKAMAK.MAGNETS: 
                    PSI_B[inode] += self.mu0 * MAGNET.Psi(Xbound)
                            
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
    ####################################### AITKEN RELAXATION ########################################
    ##################################################################################################

    def AitkenRelaxation(self,RELAXATION = False):
        if RELAXATION:
            if self.it == 1: 
                alpha = 1 - self.lambdaPSI[0]
            else:
                residual0 = self.PSI_NORM[:,0] - self.PSI_NORMstar[:,0]
                residual1 = self.PSI_NORM[:,1] - self.PSI_NORMstar[:,1]
                self.lambdaPSI[1] = self.lambdaPSI[0] + (self.lambdaPSI[0] - 1)*(residual0-residual1)@residual1/np.linalg.norm(residual0-residual1)
                alpha = 1 - max(min(self.lambdaPSI[1], self.lambdamax), self.lambdamin)
                # UPDATE lambda
                self.lambdaPSI[0] = self.lambdaPSI[1]
                
            print("AITKEN'S RELAXATION PARAMETER (alpha) = ", alpha)
            newPSI = (1-alpha)*self.PSI_NORM[:,1] + alpha*self.PSI_NORMstar[:,1]
        else:
            newPSI = self.PSI_NORMstar[:,1]
            
        # UPDATE ARRAYS
        self.PSI_NORMstar[:,0] = self.PSI_NORMstar[:,1]
        self.PSI_NORM[:,0] = self.PSI_NORM[:,1]
        self.PSI_NORM[:,1] = newPSI
        return
        
    
    ##################################################################################################
    ########################################## INTEGRATION ###########################################
    ##################################################################################################
    
    def IntegratePlasmaDomain(self,fun,PSIdependent=True):
        """ Function that integrates function fun over the plasma domain. """ 
        
        integral = 0
        if PSIdependent:
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
        Assembles the global matrices (Left-Hand Side and Right-Hand Side) derived from the discretized linear system of equations using the Galerkin approximation.

        The assembly process involves:
            1. Non-cut elements: Integration over elements not cut by any interface (using standard quadrature).
            2. Cut elements: Integration over subelements in elements cut by interfaces, using adapted quadratures.
            3. Boundary elements: For computational domain boundary elements, integration over the vacuum vessel boundary (if applicable).
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
                self.file_elemsys.write("elem {:d} {:d}\n".format(ELEMENT.index,ELEMENT.Dom))
                self.file_elemsys.write('elmat\n')
                np.savetxt(self.file_elemsys,LHSe,delimiter=',',fmt='%e')
                self.file_elemsys.write('elrhs\n')
                np.savetxt(self.file_elemsys,RHSe,fmt='%e')
            
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
                    self.file_elemsys.write("elem {:d} {:d} subelem {:d} {:d}\n".format(ELEMENT.index,ELEMENT.Dom,SUBELEM.index,SUBELEM.Dom))
                    self.file_elemsys.write('elmat\n')
                    np.savetxt(self.file_elemsys,LHSe,delimiter=',',fmt='%e')
                    self.file_elemsys.write('elrhs\n')
                    np.savetxt(self.file_elemsys,RHSe,fmt='%e')
                
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
        
        for ielem in self.MESH.PlasmaBoundElems:
            # ISOLATE ELEMENT 
            ELEMENT = self.MESH.Elements[ielem]
            # COMPUTE ELEMENTAL MATRICES
            LHSe,RHSe = ELEMENT.IntegrateElementalInterfaceTerms(self.beta)
                
            # PRESCRIBE BC:
            if not type(ELEMENT.Teboun) == type(None):
                LHSe, RHSe = ELEMENT.PrescribeDirichletBC(LHSe,RHSe)
                
            if self.out_elemsys:
                self.file_elemsys.write("elem {:d} {:d}\n".format(ELEMENT.index,ELEMENT.Dom))
                self.file_elemsys.write('elmat\n')
                np.savetxt(self.file_elemsys,LHSe,delimiter=',',fmt='%e')
                self.file_elemsys.write('elrhs\n')
                np.savetxt(self.file_elemsys,RHSe,fmt='%e')
            
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
            for irow in range(self.MESH.Nn):
                for jcol in range(self.MESH.Nn):
                    if self.LHS[irow,jcol] != 0:
                        self.file_globalsys.write("{:d} {:d} {:f}\n".format(irow+1, jcol+1, self.LHS[irow,jcol]))
            self.file_globalsys.write('END_LHS_MATRIX\n')
        
        print("Done!")   
        return
    
    
    ##################################################################################################
    ############################################ SOLVER ##############################################
    ##################################################################################################
    
    def SolveSystem(self):
        self.PSI = spsolve(self.LHS.tocsr(), self.RHS).reshape([self.MESH.Nn,1])
        return

    
    ##################################################################################################
    ######################################## MAIN ALGORITHM ##########################################
    ##################################################################################################
    
    def EQUILI(self,CASE):
        """
        Main subroutine for solving the Grad-Shafranov boundary value problem (BVP) using the CutFEM method.

        This function orchestrates the entire iterative process for solving the plasma equilibrium problem. It involves:
            1. Reading input files, including mesh and boundary data.
            2. Initializing parameters and setting up directories for output.
            3. Running an outer loop (external loop) that controls the convergence of the overall solution.
            4. Running an inner loop (internal loop) that solves the system for the plasma current and updates the mesh and solution iteratively.
            5. Evaluating and checking convergence criteria at each step.
            6. Writing results at each iteration, including solution values, critical points, and residuals.

        The function also handles:
            - Copying simulation files,
            - Plotting the solution when requested,
            - Checking convergence for both the PSI field and vacuum vessel first wall values,
            - Updating the plasma boundary and mesh classification, and
            - Computing and normalizing critical plasma quantities.

        The solution process continues until convergence criteria for both the internal and external loops are satisfied.
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
        self.writePlasmaBoundaryData()
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
                    
                # INNER LOOP ALGORITHM: SOLVING GRAD-SHAFRANOV BVP WITH CutFEM
                self.AssembleGlobalSystem()                 # 0. ASSEMBLE CUTFEM SYSTEM
                self.SolveSystem()                          # 1. SOLVE CutFEM SYSTEM  ->> PSI
                if not self.FIXED_BOUNDARY:
                    self.ComputeCriticalPSI()               # 2. COMPUTE CRITICAL VALUES   PSI_0 AND PSI_X
                    self.writePSIcrit()                     #    WRITE CRITICAL POINTS
                self.NormalisePSI()                         # 3. NORMALISE PSI RESPECT TO CRITICAL VALUES  ->> PSI_NORM 
                self.AitkenRelaxation(RELAXATION = self.PSIrelax)
                
                self.writePSI()                             #    WRITE SOLUTION             
                if self.plotPSI:
                    self.PlotSolutionPSI()                  #    PLOT SOLUTION AND NORMALISED SOLUTION
                self.CheckConvergence('PSI_NORM')           # 4. CHECK CONVERGENCE OF PSI_NORM FIELD
                self.writeresidu("INTERNAL")                #    WRITE INTERNAL LOOP RESIDU
                
                self.UpdatePlasmaRegion(RELAXATION = self.PHIrelax)                   # 5. UPDATE MESH ELEMENTS CLASSIFACTION RESPECT TO NEW PLASMA BOUNDARY LEVEL-SET
                
                self.UpdateElementalPSI()                   # 7. UPDATE PSI_NORM VALUES IN CORRESPONDING ELEMENTS (ELEMENT.PSIe = PSI_NORM[ELEMENT.Te,0])
                self.UpdatePlasmaBoundaryValues()           # 8. UPDATE ELEMENTAL CONSTRAINT VALUES PSIgseg FOR PLASMA/VACUUM INTERFACE
                self.PlasmaCurrent.Normalise()
                
                #######################################################
                ################ END INTERNAL LOOP ####################
                #######################################################
                
            #self.ComputeTotalPlasmaCurrentNormalization()
            print('COMPUTE VACUUM VESSEL FIRST WALL VALUES PSI_B...', end="")
            self.PSI_B[:,1] = self.ComputeBoundaryPSI()     # COMPUTE VACUUM VESSEL FIRST WALL VALUES PSI_B WITH INTERNALLY CONVERGED PSI_NORM
            self.UpdateVacuumVesselBoundaryValues()
            self.writePSI_B()
            print('Done!')
            
            self.CheckConvergence('PSI_B')            # CHECK CONVERGENCE OF VACUUM VESSEL FIEST WALL PSI VALUES  (PSI_B)
            self.writeresidu("EXTERNAL")              # WRITE EXTERNAL LOOP RESIDU 
            self.UpdatePSI_B()                   # UPDATE PSI_NORM AND PSI_B VALUES 
            
            #######################################################
            ################ END EXTERNAL LOOP ####################
            #######################################################
            
        print('SOLUTION CONVERGED')
        if self.plotPSI:
            self.PlotSolutionPSI()
        
        if self.FIXED_BOUNDARY and self.PlasmaCurrent.CURRENT_MODEL != self.PlasmaCurrent.JARDIN_CURRENT:
            self.ErrorL2norm, self.RelErrorL2norm, self.ErrorL2normPlasmaBound, self.RelErrorL2normPlasmaBound = self.ComputeL2errorPlasma()
            self.ErrorL2normINT, self.RelErrorL2normINT = self.ComputeL2errorInterface()
            self.InterfGradJumpErrorL2norm, self.JumpError, self.JumpRelError = self.ComputeL2errorInterfaceJump()
            self.writeerror()
        
        self.closeOUTPUTfiles()
        self.writeSimulationPickle()
        return
    
    
    
    
    
    
    
    
    