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
 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.path as mpath
import os
import shutil
from random import random
from scipy.interpolate import griddata
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from GaussQuadrature import *
from ShapeFunctions import *
from Element import *
from Magnet import *
from Greens import *

class GradShafranovFEMCutFEM:
    
    # PHYSICAL CONSTANTS
    epsilon0 = 8.8542E-12        # F m-1    Magnetic permitivity 
    mu0 = 12.566370E-7           # H m-1    Magnetic permeability

    def __init__(self,MESH,CASE):
        # WORKING DIRECTORY
        pwd = os.getcwd()
        self.pwd = pwd[:pwd.rfind("EQUILIPY_FEMCutFEM")+18]
        
        # INPUT FILES:
        self.mesh_folder = self.pwd + '/MESHES/' + MESH
        self.MESH = MESH[MESH.rfind("TS-FEMCUTFEM-")+13:]
        self.case_file = self.pwd + '/CASES/' + CASE
        self.CASE = CASE[CASE.rfind('/')+1:]
        
        # OUTPUT FILES
        self.outputdir = self.pwd + '/../RESULTS_FEMCutFEM/' + self.CASE + '-' + self.MESH
        self.PARAMS_file = None             # OUTPUT FILE CONTAINING THE SIMULATION PARAMETERS 
        self.PSI_file = None                # OUTPUT FILE CONTAINING THE PSI FIELD VALUES OBTAINED BY SOLVING THE CutFEM SYSTEM
        self.PSIcrit_file = None            # OUTPUT FILE CONTAINING THE CRITICAL PSI VALUES
        self.PSI_NORM_file = None           # OUTPUT FILE CONTAINING THE PSI_NORM FIELD VALUES (AFTER NORMALISATION OF PSI FIELD)
        self.PSI_B_file = None              # OUTPUT FILE CONTAINING THE PSI_B BOUNDARY VALUES
        self.RESIDU_file = None             # OUTPUT FILE CONTAINING THE RESIDU FOR EACH ITERATION
        self.ElementsClassi_file = None     # OUTPUT FILE CONTAINING THE CLASSIFICATION OF MESH ELEMENTS
        self.PlasmaLevSetVals_file = None   # OUTPUT FILE CONTAINING THE PLASMA BOUNDARY LEVEL-SET FIELD VALUES
        self.L2error_file = None            # OUTPUT FILE CONTAINING THE ERROR FIELD AND THE L2 ERROR NORM FOR THE CONVERGED SOLUTION 
        self.ELMAT_file = None              # OUTPUT FILE CONTAINING THE ELEMENTAL MATRICES FOR EACH ITERATION
        
        # SWITCHS PARAMETERS
        self.GhostPenalty = False
        self.PARAMS_output = False            # OUTPUT SWITCH FOR SIMULATION PARAMETERS 
        self.PSI_output = False               # OUTPUT SWITCH FOR PSI FIELD VALUES OBTAINED BY SOLVING THE CutFEM SYSTEM
        self.PSIcrit_output = False           # OUTPUT SWITCH FOR CRITICAL PSI VALUES
        self.PSI_NORM_output = False          # OUTPUT SWITCH FOR THE PSI_NORM FIELD VALUES (AFTER NORMALISATION OF PSI FIELD)
        self.PSI_B_output = False             # OUTPUT SWITCH FOR PSI_B BOUNDARY VALUES
        self.RESIDU_output = False            # OUTPUT SWITCH FOR RESIDU FOR EACH ITERATION
        self.ElementsClassi_output = False    # OUTPUT SWITCH FOR CLASSIFICATION OF MESH ELEMENTS
        self.PlasmaLevSetVals_output = False  # OUTPUT SWITCH FOR PLASMA BOUNDARY LEVEL-SET FIELD VALUES
        self.L2error_output = False           # OUTPUT SWITCH FOR ERROR FIELD AND THE L2 ERROR NORM FOR THE CONVERGED SOLUTION 
        self.ELMAT_output = False             # OUTPUT SWITCH FOR ELEMENTAL MATRICES
        self.plotElemsClassi_output = False   # OUTPUT SWITCH FOR ELEMENTS CLASSIFICATION PLOTS AT EACH ITERATION
        self.plotPSI_output = False           # OUTPUT SWITCH FOR PSI SOLUTION PLOTS AT EACH ITERATION
        
        # PROBLEM CASE PARAMETERS
        self.PLASMA_BOUNDARY = None         # PLASMA BOUNDARY BEHAVIOUR: self.FIXED_BOUNDARY  or  self.FREE_BOUNDARY 
        self.PLASMA_CURRENT = None          # PLASMA CURRENT MODELISATION: self.LINEAR_CURRENT, self.NONLINEAR_CURRENT or self.PROFILES_CURRENT
        self.TOTAL_CURRENT = None           # TOTAL CURRENT IN PLASMA
        
        # ELEMENTAL CLASSIFICATION
        self.PlasmaElems = None             # LIST OF ELEMENTS (INDEXES) INSIDE PLASMA REGION
        self.VacuumElems = None             # LIST OF ELEMENTS (INDEXES) OUTSIDE PLASMA REGION (VACUUM REGION)
        self.PlasmaBoundElems = None        # LIST OF CUT ELEMENT'S INDEXES, CONTAINING INTERFACE BETWEEN PLASMA AND VACUUM
        self.VacVessWallElems = None        # LIST OF CUT (OR NOT) ELEMENT'S INDEXES, CONTAINING VACUUM VESSEL FIRST WALL (OR COMPUTATIONAL DOMAIN'S BOUNDARY)
        self.NonCutElems = None             # LIST OF ALL NON CUT ELEMENTS
        self.Elements = None                # ARRAY CONTAINING ALL ELEMENTS IN MESH (PYTHON OBJECTS)
        
        # ARRAYS
        self.PlasmaBoundLevSet = None       # PLASMA REGION GEOMETRY LEVEL-SET FUNCTION NODAL VALUES
        self.PSI = None                     # PSI SOLUTION FIELD OBTAINED BY SOLVING CutFEM SYSTEM
        self.Xcrit = None                   # COORDINATES MATRIX FOR CRITICAL PSI POINTS
        self.PSI_0 = None                   # PSI VALUE AT MAGNETIC AXIS MINIMA
        self.PSI_X = None                   # PSI VALUE AT SADDLE POINT (PLASMA SEPARATRIX)
        self.PSI_NORM = None                # NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PSI_B = None                   # VACUUM VESSEL WALL PSI VALUES (EXTERNAL LOOP) AT ITERATION N (COLUMN 0) AND N+1 (COLUMN 1) 
        self.PSI_CONV = None                # CONVERGED NORMALISED PSI SOLUTION FIELD 
        self.residu_INT = None              # INTERNAL LOOP RESIDU
        self.residu_EXT = None              # EXTERNAL LOOP RESIDU
        
        # VACCUM VESSEL FIRST WALL GEOMETRY
        self.epsilon = None                 # VACUUM VESSEL INVERSE ASPECT RATIO
        self.kappa = None                   # VACUUM VESSEL ELONGATION
        self.delta = None                   # VACUUM VESSEL TRIANGULARITY
        self.R0 = None                      # VACUUM VESSEL MEAN RADIUS

        # PARAMETRISED INITIAL PLASMA EQUILIBRIUM GUESS
        self.CONTROL_POINTS = None          # NUMBER OF CONTROL POINTS
        self.R_SADDLE = None                # R COORDINATE OF ACTIVE SADDLE POINT
        self.Z_SADDLE = None                # Z COORDINATE OF ACTIVE SADDLE POINT
        self.R_RIGHTMOST = None             # R COORDINATE OF POINT ON THE RIGHT
        self.Z_RIGHTMOST = None             # Z COORDINATE OF POINT ON THE RIGHT
        self.R_LEFTMOST = None              # R COORDINATE OF POINT ON THE LEFT
        self.Z_LEFTMOST = None              # Z COORDINATE OF POINT ON THE LEFT
        self.R_TOP = None                   # R COORDINATE OF POINT ON TOP
        self.Z_TOP = None                   # Z COORDINATE OF POINT ON TOP
                
        ###### FOR FREE-BOUNDARY PROBLEM
        # CONFINING MAGNETS
        self.Ncoils = None                  # TOTAL NUMBER OF COILS
        self.COILS = None                   # ARRAY OF COIL OBJECTS
        self.Nsolenoids = None              # TOTAL NUMBER OF SOLENOIDS
        self.SOLENOIDS = None               # ARRAY OF SOLENOID OBJECTS
        
        # PRESSURE AND TOROIDAL FIELD PROFILES
        self.B0 = None                      # TOROIDAL FIELD MAGNITUDE ON MAGNETIC AXIS
        self.q0 = None                      # TOKAMAK SAFETY FACTOR
        self.P0 = None                      # PRESSURE PROFILE FACTOR
        self.n_p = None                     # EXPONENT FOR PRESSURE PROFILE p_hat FUNCTION
        self.G0 = None                      # TOROIDAL FIELD FACTOR
        self.n_g = None                     # EXPONENT FOR TOROIDAL FIELD PROFILE g_hat FUNCTION
        
        ########################
        # COMPUTATIONAL MESH
        self.ElTypeALYA = None              # TYPE OF ELEMENTS CONSTITUTING THE MESH, USING ALYA NOTATION
        self.ElType = None                  # TYPE OF ELEMENTS CONSTITUTING THE MESH: 1: TRIANGLES,  2: QUADRILATERALS
        self.ElOrder = None                 # ORDER OF MESH ELEMENTS: 1: LINEAR,   2: QUADRATIC,   3: CUBIC
        self.X = None                       # MESH NODAL COORDINATES MATRIX
        self.T = None                       # MESH ELEMENTS CONNECTIVITY MATRIX 
        self.Nn = None                      # TOTAL NUMBER OF MESH NODES
        self.Ne = None                      # TOTAL NUMBER OF MESH ELEMENTS
        self.n = None                       # NUMBER OF NODES PER ELEMENT
        self.numedges = None                # NUMBER OF EDGES PER ELEMENT (= 3 IF TRIANGULAR; = 4 IF QUADRILATERAL)
        self.nedge = None                   # NUMBER OF NODES ON ELEMENTAL EDGE
        self.dim = None                     # SPACE DIMENSION
        self.Tbound = None                  # MESH BOUNDARIES CONNECTIVITY MATRIX  (LAST COLUMN YIELDS THE ELEMENT INDEX OF THE CORRESPONDING BOUNDARY EDGE)
        self.Nbound = None                  # NUMBER OF COMPUTATIONAL DOMAIN'S BOUNDARIES (NUMBER OF ELEMENTAL EDGES)
        self.Nnbound = None                 # NUMBER OF NODES ON COMPUTATIONAL DOMAIN'S BOUNDARY
        self.BoundaryNodes = None           # LIST OF NODES (GLOBAL INDEXES) ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        self.DOFNodes = None                # LIST OF NODES (GLOBAL INDEXES) CORRESPONDING TO UNKNOW DEGREES OF FREEDOM IN THE CUTFEM SYSTEM
        self.NnDOF = None                   # NUMBER OF UNKNOWN DEGREES OF FREEDOM NODES
        self.PlasmaNodes = None             # LIST OF NODES (GLOBAL INDEXES) INSIDE THE PLASMA DOMAIN
        self.VacuumNodes = None             # LIST OF NODES (GLOBAL INDEXES) IN THE VACUUM REGION
        self.NnPB = None                    # NUMBER OF NODES ON PLASMA BOUNDARY APPROXIMATION
        self.Rmax = None                    # COMPUTATIONAL MESH MAXIMAL X (R) COORDINATE
        self.Rmin = None                    # COMPUTATIONAL MESH MINIMAL X (R) COORDINATE
        self.Zmax = None                    # COMPUTATIONAL MESH MAXIMAL Y (Z) COORDINATE
        self.Zmin = None                    # COMPUTATIONAL MESH MINIMAL Y (Z) COORDINATE
        
        # NUMERICAL TREATMENT PARAMETERS
        self.LHS = None                     # GLOBAL CutFEM SYSTEM LEFT-HAND-SIDE MATRIX
        self.RHS = None                     # GLOBAL CutFEM SYSTEM RIGHT-HAND-SIDE VECTOR
        self.LHSred = None                  # REDUCED (IMPOSED BC) GLOBAL CutFEM SYSTEM LEFT-HAND-SIDE MATRIX
        self.RHSred = None                  # REDUCED (IMPOSED BC) GLOBAL CutFEM SYSTEM RIGHT-HAND-SIDE VECTOR
        self.QuadratureOrder = None         # NUMERICAL INTEGRATION QUADRATURE ORDER
        self.SoleOrder = 2                  # SOLENOID ELEMENT (BAR) ORDER
        
        self.PlasmaBoundGhostFaces = None   # LIST OF PLASMA BOUNDARY GHOST FACES
        self.PlasmaBoundGhostElems = None   # LIST OF ELEMENTS CONTAINING PLASMA BOUNDARY FACES
        
        #### DOBLE WHILE LOOP STRUCTURE PARAMETERS
        self.INT_TOL = None                 # INTERNAL LOOP STRUCTURE CONVERGENCE TOLERANCE
        self.EXT_TOL = None                 # EXTERNAL LOOP STRUCTURE CONVERGENCE TOLERANCE
        self.INT_ITER = None                # INTERNAL LOOP STRUCTURE MAXIMUM ITERATIONS NUMBER
        self.EXT_ITER = None                # EXTERNAL LOOP STRUCTURE MAXIMUM ITERATIONS NUMBER
        self.converg_EXT = None             # EXTERNAL LOOP STRUCTURE CONVERGENCE FLAG
        self.converg_INT = None             # INTERNAL LOOP STRUCTURE CONVERGENCE FLAG
        self.it_EXT = None                  # EXTERNAL LOOP STRUCTURE ITERATIONS NUMBER
        self.it_INT = None                  # INTERNAL LOOP STRUCTURE ITERATIONS NUMBER
        self.it = 0                         # TOTAL NUMBER OF ITERATIONS COUNTER
        self.alpha = None                   # AIKTEN'S SCHEME RELAXATION CONSTANT
        self.gamma = None                   # PLASMA TOTAL CURRENT CORRECTION FACTOR
        #### BOUNDARY CONSTRAINTS
        self.beta = None                    # NITSCHE'S METHOD PENALTY TERM
        #### STABILIZATION
        self.zeta = None                    # GHOST PENALTY PARAMETER
        #### OPTIMIZATION OF CRITICAL POINTS
        self.EXTR_R0 = None                 # MAGNETIC AXIS OPTIMIZATION INITIAL GUESS R COORDINATE
        self.EXTR_Z0 = None                 # MAGNETIC AXIS OPTIMIZATION INITIAL GUESS Z COORDINATE
        self.SADD_R0 = None                 # SADDLE POINT OPTIMIZATION INITIAL GUESS R COORDINATE
        self.SADD_Z0 = None                 # SADDLE POINT OPTIMIZATION INITIAL GUESS Z COORDINATE
        
        # EQUILIPY PARAMETRISATION WITH FLAGS
        #### PLASMA BOUNDARY BEHAVIOUR FLAGS
        self.FIXED_BOUNDARY = 0
        self.FREE_BOUNDARY = 1
        #### PLASMA MODEL PARAMETERS
        self.LINEAR_CURRENT = 0
        self.NONLINEAR_CURRENT = 1
        self.ZHENG_CURRENT = 2
        self.PROFILES_CURRENT = 3
        #### SEGMENTS' SET FLAGS
        self.INTERFAPPROX = 0
        self.GHOSTFACES = 1
        
        # PLASMA MODELS COEFFICIENTS
        self.coeffsLINEAR = None                       # TOKAMAK FIRST WALL LEVEL-0 CONTOUR COEFFICIENTS (LINEAR PLASMA MODEL CASE SOLUTION)
        self.coeffsNONLINEAR = [1.15*np.pi, 1.15, -0.5]  # [Kr, Kz, R0] 
        self.coeffsZHENG = None
        
        self.L2error = None
        
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
    
    def ALYA2Py(self):
        """ 
        Translate ALYA elemental type to EQUILIPY elemental ElType and ElOrder.
        """
        match self.ElTypeALYA:
            case 2:
                self.ElType = 0
                self.ElOrder = 1
            case 3:
                self.ElType = 0
                self.ElOrder = 2
            case 4:
                self.ElType = 0
                self.ElOrder = 3
            case 10:
                self.ElType = 1
                self.ElOrder = 1
            case 11:
                self.ElType = 1
                self.ElOrder = 2
            case 16:
                self.ElType = 1
                self.ElOrder = 3
            case 12:
                self.ElType = 2
                self.ElOrder = 1
            case 14:
                self.ElType = 2
                self.ElOrder = 2
            case 15:
                self.ElType = 2
                self.ElOrder = 3
        return
    
    
    ##################################################################################################
    ############################### READ INPUT DATA FILES ############################################
    ##################################################################################################
    
    def ReadMesh(self):
        """ 
        Read input mesh data files, .dom.dat and .geo.dat, and build mesh simulation attributes. 
        """
        
        print("     -> READ MESH DATA FILES...",end='')
        # READ DOM FILE .dom.dat
        MeshDataFile = self.mesh_folder +'/'+ 'TS-FEMCUTFEM-' + self.MESH +'.dom.dat'
        self.Nn = 0   # number of nodes
        self.Ne = 0   # number of elements
        file = open(MeshDataFile, 'r') 
        for line in file:
            l = line.split('=')
            if l[0] == '  NODAL_POINTS':  # read number of nodes
                self.Nn = int(l[1])
            elif l[0] == '  ELEMENTS':  # read number of elements
                self.Ne = int(l[1])
            elif l[0] == '  SPACE_DIMENSIONS':  # read space dimensions 
                self.dim = int(l[1])
            elif l[0] == '  TYPES_OF_ELEMENTS':
                self.ElTypeALYA = int(l[1])
            elif l[0] == '  BOUNDARIES':  # read number of boundaries
                self.Nbound = int(l[1])
        file.close()
        
        self.ALYA2Py()
        
        # NUMBER OF NODES PER ELEMENT
        self.n, self.nedge = ElementalNumberOfNodes(self.ElType, self.ElOrder)
        if self.ElType == 1:
            self.numedges = 3
        elif self.ElType == 2:
            self.numedges = 4
        
        # READ MESH FILE .geo.dat
        MeshFile = self.mesh_folder +'/'+ 'TS-FEMCUTFEM-' + self.MESH +'.geo.dat'
        self.T = np.zeros([self.Ne,self.n], dtype = int)
        self.X = np.zeros([self.Nn,self.dim], dtype = float)
        self.Tbound = np.zeros([self.Nbound,self.nedge+1], dtype = int)   # LAST COLUMN YIELDS THE ELEMENT INDEX OF THE CORRESPONDING BOUNDARY EDGE 
        file = open(MeshFile, 'r') 
        i = -1
        j = -1
        k = -1
        for line in file:
            # first we format the line read in order to remove all the '\n'  
            l = line.split(' ')
            l = [m for m in l if m != '']
            for e, el in enumerate(l):
                if el == '\n':
                    l.remove('\n') 
                elif el[-1:]=='\n':
                    l[e]=el[:-1]
            # WE IDENTIFY WHEN THE CONNECTIVITY MATRIX DATA STARTS
            if l[0] == 'ELEMENTS':
                i=0
                continue
            # WE IDENTIFY WHEN THE CONNECTIVITY MATRIX DATA ENDS
            elif l[0] == 'END_ELEMENTS':
                i=-1
                continue
            # WE IDENTIFY WHEN THE NODAL COORDINATES DATA STARTS
            elif l[0] == 'COORDINATES':
                j=0
                continue
            # WE IDENTIFY WHEN THE NODAL COORDINATES DATA ENDS
            elif l[0] == 'END_COORDINATES':
                j=-1
                continue
            # WE IDENTIFY WHEN THE COMPUTATIONAL DOMAIN'S BOUNDARY DATA STARTS
            elif l[0] == 'BOUNDARIES,':
                k=0
                continue
            # WE IDENTIFY WHEN THE COMPUTATIONAL DOMAIN'S BOUNDARY DATA ENDS
            elif l[0] == 'END_BOUNDARIES':
                k=-1
                continue
            if i>=0:
                for m in range(self.n):
                    self.T[i,m] = int(l[m+1])
                i += 1
            if j>=0:
                for m in range(self.dim):
                    self.X[j,m] = float(l[m+1])
                j += 1
            if k>=0:
                for m in range(self.nedge+1):
                    self.Tbound[k,m] = int(l[m+1])
                k += 1
        file.close()
        # PYTHON INDEXES START AT 0 AND NOT AT 1. THUS, THE CONNECTIVITY MATRIX INDEXES MUST BE ADAPTED
        self.T = self.T - 1
        self.Tbound = self.Tbound - 1
        
        print('Done!')
        return
    
    def ReadEQUILIdata(self):
        """ 
        Reads problem data from input file equ.dat.      
        """
        
        #############################################
        # INTER-CODE FUNCTIONS TO READ INPUT PARAMETERS BY BLOCKS 
        
        def BlockProblemParameters(self,line):
            if line[0] == 'PLASB:':          # READ PLASMA BOUNDARY CONDITION (FIXED OR FREE)
                if line[1] == 'FIXED':
                    self.PLASMA_BOUNDARY =  self.FIXED_BOUNDARY
                elif line[1] == 'FREED':
                    self.PLASMA_BOUNDARY = self.FREE_BOUNDARY
            elif line[0] == 'PLASC:':         # READ MODEL FOR PLASMA CURRENT (LINEAR, NONLINEAR, ZHENG OR DEFINED USING PROFILES FOR PRESSURE AND TOROIDAL FIELD)
                if line[1] == 'LINEA':
                    self.PLASMA_CURRENT = self.LINEAR_CURRENT
                elif line[1] == 'NONLI':
                    self.PLASMA_CURRENT = self.NONLINEAR_CURRENT
                elif line[1] == 'ZHENG':
                    self.PLASMA_CURRENT = self.ZHENG_CURRENT
                elif line[1] == 'PROFI':
                    self.PLASMA_CURRENT = self.PROFILES_CURRENT
                else:
                    self.PLASMA_CURRENT = l[1]
            elif line[0] == 'TOTAL_CURRENT:':        # READ TOTAL PLASMA CURRENT
                self.TOTAL_CURRENT = float(line[1])
            return
        
        def BlockFirstWall(self,line):
            if line[0] == 'R0TOK:':          # READ TOKAMAK FIRST WALL MAJOR RADIUS 
                self.R0 = float(line[1])
            elif line[0] == 'EPSILON:':      # READ TOKAMAK FIRST WALL INVERSE ASPECT RATIO
                self.epsilon = float(line[1])
            elif line[0] == 'KAPPA:':        # READ TOKAMAK FIRST WALL ELONGATION 
                self.kappa = float(line[1])
            elif line[0] == 'DELTA:':        # READ TOKAMAK FIRST WALL TRIANGULARITY 
                self.delta = float(line[1])
            return
        
        def BlockF4E(self,line):
            # READ PLASMA SHAPE CONTROL POINTS
            if line[0] == 'CONTROL_POINTS:':    # READ INITIAL PLASMA REGION NUMBER OF CONTROL POINTS
                self.CONTROL_POINTS = int(line[1])
            elif line[0] == 'R_SADDLE:':    # READ INITIAL PLASMA REGION SADDLE POINT R COORDINATE
                self.R_SADDLE = float(line[1])
            elif line[0] == 'Z_SADDLE:':    # READ INITIAL PLASMA REGION SADDLE POINT Z COORDINATE
                self.Z_SADDLE = float(line[1])
            elif line[0] == 'R_RIGHTMOST:':    # READ INITIAL PLASMA REGION RIGHT POINT R COORDINATE
                self.R_RIGHTMOST = float(line[1])
            elif line[0] == 'Z_RIGHTMOST:':    # READ INITIAL PLASMA REGION RIGHT POINT Z COORDINATE 
                self.Z_RIGHTMOST = float(line[1])
            elif line[0] == 'R_LEFTMOST:':    # READ INITIAL PLASMA REGION LEFT POINT R COORDINATE 
                self.R_LEFTMOST = float(line[1])
            elif line[0] == 'Z_LEFTMOST:':    # READ INITIAL PLASMA REGION LEFT POINT Z COORDINATE 
                self.Z_LEFTMOST = float(line[1])
            elif line[0] == 'R_TOPP:':    # READ INITIAL PLASMA REGION TOP POINT R COORDINATE 
                self.R_TOP = float(line[1])
            elif line[0] == 'Z_TOPP:':    # READ INITIAL PLASMA REGION TOP POINT Z COORDINATE 
                self.Z_TOP = float(line[1])
            return
        
        def BlockExternalMagnets(self,line,i,j):
            if line[0] == 'N_COILS:':    # READ TOTAL NUMBER COILS 
                self.Ncoils = int(line[1])
                self.COILS = [Coil(index = icoil, 
                                   dim=self.dim, 
                                   X=np.zeros([self.dim]), 
                                   I=None) for icoil in range(self.Ncoils)] 
            elif line[0] == 'Rposi:' and i<self.Ncoils:    # READ i-th COIL X POSITION
                self.COILS[i].X[0] = float(line[1])
            elif line[0] == 'Zposi:' and i<self.Ncoils:    # READ i-th COIL Y POSITION
                self.COILS[i].X[1] = float(line[1])
            elif line[0] == 'Inten:' and i<self.Ncoils:    # READ i-th COIL INTENSITY
                self.COILS[i].I = float(line[1])
                i += 1
            # READ SOLENOID PARAMETERS:
            elif line[0] == 'N_SOLENOIDS:':    # READ TOTAL NUMBER OF SOLENOIDS
                self.Nsolenoids = int(line[1])
                self.SOLENOIDS = [Solenoid(index = isole, 
                                           dim=self.dim, 
                                           Xe=np.zeros([2,self.dim]), 
                                           I=None,
                                           Nturns = None) for isole in range(self.Nsolenoids)] 
            elif line[0] == 'Rposi:' and j<self.Nsolenoids:    # READ j-th SOLENOID X POSITION
                self.SOLENOIDS[j].Xe[0,0] = float(line[1])
                self.SOLENOIDS[j].Xe[1,0] = float(line[1])
            elif line[0] == 'Zlowe:' and j<self.Nsolenoids:     # READ j-th SOLENOID Y POSITION
                self.SOLENOIDS[j].Xe[0,1] = float(line[1])
            elif line[0] == 'Zuppe:' and j<self.Nsolenoids:      # READ j-th SOLENOID Y POSITION
                self.SOLENOIDS[j].Xe[1,1] = float(line[1])
            elif line[0] == 'Nturn:' and j<self.Nsolenoids:      # READ j-th SOLENOID NUMBER OF TURNS
                self.SOLENOIDS[j].Nturns = int(line[1])
            elif line[0] == 'Inten:' and j<self.Nsolenoids:    # READ j-th SOLENOID INTENSITY
                self.SOLENOIDS[j].I = float(line[1])
                j += 1
            return i, j
        
        def BlockProfiles(self,line):
            if line[0] == 'B0_equ:':    # READ TOROIDAL FIELD MAGNITUDE ON MAGNETIC AXIS
                self.B0 = float(line[1])
            elif line[0] == 'q0_equ:':    # READ TOKAMAK SAFETY FACTOR 
                self.q0 = float(line[1])
            elif line[0] == 'np_equ:':    # READ EXPONENT FOR PRESSURE PROFILE p_hat FUNCTION 
                self.n_p = float(line[1])
            elif line[0] == 'g0_equ:':    # READ TOROIDAL FIELD PROFILE FACTOR
                self.G0 = float(line[1])
            elif line[0] == 'ng_equ:':    # READ EXPONENT FOR TOROIDAL FIELD PROFILE g_hat FUNCTION
                self.n_g = float(line[1])
            return
        
        def BlockNumericalTreatement(self,line):
            if line[0] == 'QUADRATURE_ORDER:':   # READ NUMERICAL INTEGRATION QUADRATURE ORDER
                self.QuadratureOrder = int(line[1])
            elif line[0] == 'EXT_ITER:':         # READ MAXIMAL NUMBER OF ITERATION FOR EXTERNAL LOOP
                self.EXT_ITER = int(line[1])
            elif line[0] == 'EXT_TOL:':          # READ TOLERANCE FOR EXTERNAL LOOP
                self.EXT_TOL = float(line[1])
            elif line[0] == 'INT_ITER:':         # READ MAXIMAL NUMBER OF ITERATION FOR INTERNAL LOOP
                self.INT_ITER = int(line[1])
            elif line[0] == 'INT_TOL:':          # READ TOLERANCE FOR INTERNAL LOOP
                self.INT_TOL = float(line[1])
            elif line[0] == 'BETA_equ:':         # READ NITSCHE'S METHOD PENALTY PARAMETER 
                self.beta = float(line[1])
            elif line[0] == 'ZETA_equ:':         # READ GHOST PENALTY PARAMETER 
                self.zeta = float(line[1])
            elif line[0] == 'RELAXATION:':       # READ AITKEN'S METHOD RELAXATION PARAMETER
                self.alpha = float(line[1])
            elif line[0] == 'EXTR_R0:':	         # READ MAGNETIC AXIS OPTIMIZATION ROUTINE INITIAL GUESS R COORDINATE
                self.EXTR_R0 = float(line[1])
            elif line[0] == 'EXTR_Z0:':          # READ MAGNETIC AXIS OPTIMIZATION ROUTINE INITIAL GUESS Z COORDINATE
                self.EXTR_Z0 = float(line[1])
            elif line[0] == 'SADD_R0:':          # READ ACTIVE SADDLE POINT OPTIMIZATION ROUTINE INITIAL GUESS R COORDINATE
                self.SADD_R0 = float(line[1])
            elif line[0] == 'SADD_Z0:':          # READ ACTIVE SADDLE POINT OPTIMIZATION ROUTINE INITIAL GUESS Z COORDINATE
                self.SADD_Z0 = float(line[1])
            return
        
        ################################################
                
        print("     -> READ EQUILI DATA FILE...",end='')
        # READ EQU FILE .equ.dat
        EQUILIDataFile = self.case_file +'.equ.dat'
        file = open(EQUILIDataFile, 'r') 
        i = 0; j = 0
        for line in file:
            l = line.split(' ')
            l = [m for m in l if m != '']
            for e, el in enumerate(l):
                if el == '\n':
                    l.remove('\n') 
                elif el[-1:]=='\n':
                    l[e]=el[:-1]
                    
            if l:  # LINE NOT EMPTY
                # READ PROBLEM PARAMETERS
                BlockProblemParameters(self,l)
                # READ TOKAMAK FIRST WALL GEOMETRY PARAMETERS
                BlockFirstWall(self,l)
                # READ CONTROL POINTS COORDINATES FOR F4E PLASMA SHAPE
                if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:
                    BlockF4E(self,l)
                # READ PARAMETERS FOR PRESSURE AND TOROIDAL FIELD PROFILES
                if self.PLASMA_CURRENT == self.PROFILES_CURRENT:
                    BlockProfiles(self,l)
                # READ COIL PARAMETERS
                if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:
                    i,j = BlockExternalMagnets(self,l,i,j)
                # READ NUMERICAL TREATMENT PARAMETERS
                BlockNumericalTreatement(self,l)
        
        print('Done!')  
        return
    
    def ReadFixdata(self):
        """
        Read fix set data from input file .fix.dat. 
        """
        print("     -> READ FIX DATA FILE...",end='')
        # READ EQU FILE .equ.dat
        FixDataFile = self.mesh_folder +'/'+ 'TS-FEMCUTFEM-' + self.MESH +'.fix.dat'
        file = open(FixDataFile, 'r') 
        self.BoundaryIden = np.zeros([self.Nbound],dtype=int)
        for line in file:
            l = line.split(' ')
            l = [m for m in l if m != '']
            for e, el in enumerate(l):
                if el == '\n':
                    l.remove('\n') 
                elif el[-1:]=='\n':
                    l[e]=el[:-1]
            
            if l[0] == "ON_BOUNDARIES" or l[0] == "END_ON_BOUNDARIES":
                pass
            else:
                self.BoundaryIden[int(l[0])-1] = int(l[1])
                
        # DEFINE THE DIFFERENT SETS OF BOUNDARY NODES
        self.BoundaryNodesSets = [set(),set()]
        for iboun in range(self.Nbound):
            for node in self.Tbound[iboun,:-1]:
                self.BoundaryNodesSets[self.BoundaryIden[iboun]-1].add(node)
        # CONVERT BOUNDARY NODES SET INTO ARRAY
        self.BoundaryNodesSets[0] = np.array(sorted(self.BoundaryNodesSets[0]))
        self.BoundaryNodesSets[1] = np.array(sorted(self.BoundaryNodesSets[1]))
        print('Done!')
        return
    
    
    ##################################################################################################
    ############################# INITIAL GUESS AND SOLUTION CASE ####################################
    ##################################################################################################
    
    def ComputeLinearSolutionCoefficients(self):
        """ 
        Computes the coeffients for the magnetic flux in the linear source term case, that is for 
                GRAD-SHAFRANOV EQ:  DELTA*(PSI) = R^2   (plasma current is linear such that Jphi = R/mu0)
        for which the exact solution is 
                PSI = R^4/8 + D1 + D2*R^2 + D3*(R^4-4*R^2*Z^2)
            This function returns coefficients D1, D2, D3
                
        Geometrical dimensionless parameters: 
                - epsilon: magnetic confinement cross-section inverse aspect ratio
                - kappa: magnetic confinement cross-section elongation
                - delta: magnetic confinement cross-section triangularity 
        """
                
        A = np.array([[1, (1+self.epsilon)**2, (1+self.epsilon)**4], 
                    [1, (1-self.epsilon)**2, (1-self.epsilon)**4],
                    [1, (1-self.delta*self.epsilon)**2, (1-self.delta*self.epsilon)**4-4*(1-self.delta*self.epsilon)**2*self.kappa**2*self.epsilon**2]])
        b = -(1/8)*np.array([[(1+self.epsilon)**4], [(1-self.epsilon)**4], [(1-self.delta*self.epsilon)**4]])
        
        coeffs = np.linalg.solve(A,b)
        return coeffs.T[0].tolist() 
    
    def ComputeZhengSolutionCoefficients(self):
        """ Computes the coefficients for the Grad-Shafranov equation analytical solution proposed in ZHENG paper. """
        Ri = self.R0*(1-self.epsilon)  # PLASMA SHAPE EQUATORIAL INNERMOST POINT R COORDINATE
        Ro = self.R0*(1+self.epsilon)  # PLASMA SHAPE EQUATORIAL OUTERMOST POINT R COORDINATE
        a = (Ro-Ri)/2                  # PLASMA MINOR RADIUS
        Rt = self.R0 - self.delta*a    # PLASMA SHAPE HIGHEST POINT R COORDINATE
        Zt = self.kappa*a              # PLASMA SHAPE HIGHEST POINT Z COORDINATE
        
        coeffs = np.zeros([6])
        
        # SET THE COEFFICIENT A2 TO 0 FOR SIMPLICITY
        coeffs[5] = 0
        # COMPUTE COEFFICIENT A1 BY IMPOSING A CONSTANT TOTAL TOROIDAL PLASMA CURRENT Ip
        #                   Jphi = (A1*R**2 - A2)/ R*mu0 
        # IF A2 = 0, WE HAVE THEN       Jphi = A1* (R/mu0)   THAT IS WHAT WE NEED TO INTEGRATE
        # HENCE,   A1 = Ip/integral(Jphi)
        def fun(X,PSI):
            return X[0]/self.mu0
        
        #self.coeffsZHENG[4] = self.TOTAL_CURRENT/self.PlasmaDomainIntegral(fun)
        
        coeffs[4] = -0.1
        
        # FOR COEFFICIENTS C1, C2, C3 AND C4, WE SOLVE A LINEAR SYSTEM OF EQUATIONS BASED ON THE PLASMA SHAPE GEOMETRY
        A = np.array([[1,Ri**2,Ri**4,np.log(Ri)*Ri**2],
                      [1,Ro**2,Ro**4,np.log(Ro)*Ro**2],
                      [1,Rt**2,(Rt**2-4*Zt**2)*Rt**2,np.log(Rt)*Rt**2-Zt**2],
                      [0,2,4*(Rt**2-2*Zt**2),2*np.log(Rt)+1]])
        
        b = np.array([[-(coeffs[4]*Ri**4)/8],
                      [-(coeffs[4]*Ro**4)/8],
                      [-(coeffs[4]*Rt**4)/8+(coeffs[5]*Zt**2)/2],
                      [-(coeffs[4]*Rt**2)/2]])
        
        coeffs_red = np.linalg.solve(A,b)
        coeffs[:4] = coeffs_red.T[0].tolist()
        return coeffs
    
    def PSIAnalyticalSolution(self,X,MODEL):
        """
        Compute the analytical solution for PSI at point X based on the specified model.

        Input:
            - X (array-like): Spatial coordinates [X1, X2].
            - MODEL (str): The model to use for the computation. Options include:
                  -> LINEAR_CURRENT: Linear current model.
                  -> NONLINEAR_CURRENT: Nonlinear current model.
                  -> ZHENG_CURRENT: Zheng's model.
                  -> FAKE: A mock model for testing purposes.

        Output:
            float: The computed analytical solution for PSI.
        """
        match MODEL:
            case self.LINEAR_CURRENT:
                # DIMENSIONALESS COORDINATES
                Xstar = X/self.R0
                # ANALYTICAL SOLUTION
                PSIexact = (Xstar[0]**4)/8 + self.coeffsLINEAR[0] + self.coeffsLINEAR[1]*Xstar[0]**2 + self.coeffsLINEAR[2]*(Xstar[0]**4-4*Xstar[0]**2*Xstar[1]**2)
                
            case self.NONLINEAR_CURRENT:
                # DIMENSIONALESS COORDINATES
                Xstar = X/self.R0 
                # ANALYTICAL SOLUTION
                PSIexact = np.sin(self.coeffsNONLINEAR[0]*(Xstar[0]+self.coeffsNONLINEAR[2]))*np.cos(self.coeffsNONLINEAR[1]*Xstar[1])  
                
            case self.ZHENG_CURRENT:
                # ANALYTICAL SOLUTION
                PSIexact = self.coeffsZHENG[0]+self.coeffsZHENG[1]*X[0]**2+self.coeffsZHENG[2]*(X[0]**4-4*X[0]**2*X[1]**2)+self.coeffsZHENG[3]*(np.log(X[0])
                                    *X[0]**2-X[1]**2)+(self.coeffsZHENG[4]*X[0]**4)/8 - (self.coeffsZHENG[5]*X[1]**2)/2
                
            case "FAKE":
                PSIexact = np.sin(2*X[0])+0.1*X[1]**3
        
        return PSIexact
    
    ##################################################################################################
    ###################################### PLASMA CURRENT ############################################
    ##################################################################################################
    
    def Jphi(self,X,PSI):
        """
        Compute the toroidal plasma current density (J_phi) based on the specified plasma current model.

        Input:
            - X (array-like): Spatial coordinates [X1, X2].
            - PSI (float): Poloidal flux function value at the given coordinates.

        Output:
            Jphi (float): The computed toroidal plasma current density (J_phi).
        """
        match self.PLASMA_CURRENT:
            case self.LINEAR_CURRENT:
                # COMPUTE LINEAR MODEL PLASMA CURRENT
                Jphi = X[0]/self.mu0
            
            case self.NONLINEAR_CURRENT: 
                # COMPUTE NONLINEAR MODEL PLASMA CURRENT
                Kr, Kz, r0 = self.coeffsNONLINEAR
                Jphi = -((Kr**2+Kz**2)*PSI+(Kr/X[0])*np.cos(Kr*(X[0]+r0))*np.cos(Kz*X[1])+X[0]*(np.sin(Kr*(X[0]+r0))**2*np.cos(Kz*X[1])**2
                            -PSI**2+np.exp(-np.sin(Kr*(X[0]+r0))*np.cos(Kz*X[1]))-np.exp(-PSI)))/(X[0]*self.mu0)
            
            case self.ZHENG_CURRENT:
                # COMPUTE PLASMA CURRENT MODEL BASED ON ZHENG PAPER
                Jphi = (self.coeffsZHENG[4]*X[0]**2 - self.coeffsZHENG[5])/ (X[0]*self.mu0)
        
            case self.PROFILES_CURRENT:
                ## OPTION WITH GAMMA APPLIED TO funG AND WITHOUT denom
                Jphi = -X[0] * self.dPdPSI(PSI) - 0.5*self.dG2dPSI(PSI)/ (X[0]*self.mu0)
            
        return Jphi
    
    ######## PLASMA PRESSURE MODELING
    
    def dPdPSI(self,PSI):
        """
        Compute the derivative of the plasma pressure profile with respect to PSI.

        Input:
            PSI (float): Poloidal flux function value.

        Output:
            dp (float): The computed derivative of the plasma pressure profile (dP/dPSI).
        """ 
        dp = self.P0*self.n_p*(PSI**(self.n_p-1))
        return dp
    
    ######## TOROIDAL FUNCTION MODELING
    
    def dG2dPSI(self,PSI):
        # FUNCTION MODELING TOROIDAL FIELD FUNCTION DERIVATIVE IN PLASMA REGION
        dg = (self.G0**2)*self.n_g*(PSI**(self.n_g-1))
        return dg
    
    
    def SourceTerm(self,X,PSI):
        """
        Compute the source term for the plasma current based on the specified plasma current model.

        Input:
            - X (array-like): Spatial coordinates [X1, X2].
            - PSI (float): Poloidal flux function value.

        Output:
            source (float): The computed source term for the plasma current.
        """
        match self.PLASMA_CURRENT:
            case self.LINEAR_CURRENT:
                Xstar = X/self.R0
                # COMPUTE LINEAR MODEL PLASMA CURRENT
                source = Xstar[0]**2
            
            case self.NONLINEAR_CURRENT: 
                Xstar = X/self.R0
                # COMPUTE NONLINEAR MODEL PLASMA CURRENT
                Kr, Kz, r0 = self.coeffsNONLINEAR
                source = -((Kr**2+Kz**2)*PSI+(Kr/Xstar[0])*np.cos(Kr*(Xstar[0]+r0))*np.cos(Kz*Xstar[1])+Xstar[0]*(np.sin(Kr*(Xstar[0]+r0))**2*np.cos(Kz*Xstar[1])**2
                            -PSI**2+np.exp(-np.sin(Kr*(Xstar[0]+r0))*np.cos(Kz*Xstar[1]))-np.exp(-PSI)))
                
            case self.ZHENG_CURRENT:
                # COMPUTE PLASMA CURRENT MODEL BASED ON ZHENG PAPER
                source = self.coeffsZHENG[4]*X[0]**2 - self.coeffsZHENG[5]
                
            case self.PROFILES_CURRENT:
                source = -self.mu0*X[0]**2*self.dPdPSI(PSI) - 0.5*self.dG2dPSI(PSI)
                
        if self.PLASMA_CURRENT == 'FAKE':
            source = -4*np.sin(2*X[0]) + 0.6*X[1]

        return source
        
        
    
    ##################################################################################################
    ###################################### LEVEL-SET DESCRIPTION #####################################
    ##################################################################################################
    
    def F4E_PlasmaBoundLevSet(self):
        """ # IN ORDER TO FIND THE CURVE PARAMETRIZING THE PLASMA REGION BOUNDARY, WE LOOK FOR THE COEFFICIENTS DEFINING
        # A 3rd ORDER HAMILTONIAN FROM WHICH WE WILL TAKE THE 0-LEVEL CURVE AS PLASMA REGION BOUNDARY. THAT IS
        #
        # H(x,y) = A00 + A10x + A01y + A20x**2 + A11xy + A02y**2 + A30x**3 + A21x**2y + A12xy**2 + A03y**3
        # 
        # HENCE, WE NEED TO IMPOSE CONSTRAINTS ON THE HAMILTONIAN FUNCTION IN ORDER TO SOLVE A SYSTEM OF EQUATIONS 
        # (LINEAR OR NONLINEAR). THE RESULTING SYSTEM WILL READ AS   Ax = b.
        # IN ORDER TO SIMPLIFY SUCH PROBLEM, WE ASSUME THAT:
        #   - ORIGIN (0,0) ON 0-LEVEL CURVE ---> A00 = 0
        #   - SADDLE POINT AT (0,0) ---> A10 = A01 = 0 
        # EVEN IF THAT IS NOT THE CASE IN THE PHYSICAL PLASMA REGION, WE ONLY NEED TO TRANSLATE THE REFERENCE FRAME 
        # RESPECT TO THE REAL SADDLE POINT LOCATION P0 IN ORDER TO WORK WITH EQUIVALENT PROBLEMS.
        # FURTHERMORE, WE CAN NORMALISE RESPECT TO A20 WITHOUT LOSS OF GENERALITY. THEREFORE, WE DEPART FROM 
        #
        # H(x,y) = x**2 + A11xy + A02y**2 + A30x**3 + A21x**2y + A12xy**2 + A03y**3
        # 
        # AS MENTIONED EARLIER, THE PROFILE WILL CORRESPOND TO THE 0-LEVEL CURVE, WHICH MEANS WE MUST OBTAIN THE 
        # COEFFICIENTS FOR 
        #
        # A11xy + A02y**2 + A30x**3 + A21x**2y + A12xy**2 + A03y**3 = -x**2
        #
        # WE NEED HENCE TO IMPOSE 6 CONSTRAINTS IN ORDER TO DETERMINE THE REMAINING COEFFICIENTS
        
        # For this method we constraint the curve to:
        # - go through points P1, P2 and P3 (CONTROL POINTS)
        # - have vertical tangents at points P1 and P2
        # - have a 90º angle at saddle point
        
        # where the control points are defined as:
        #      - P0: SADDLE POINT
        #      - P1: RIGHTMOST POINT
        #      - P2: LEFTMOST POINT
        #      - P3: TOP POINT
        
        # Input: - P0: SADDLE POINT COORDINATES
        #        - P1: RIGHTMOST POINT COORDINATES
        #        - P2: LEFTMOST POINT COORDINATES
        #        - P3: TOP POINT COORDINATES
        #        - X: NODAL COORDINATES MATRIX
        # """
        
        # THE FOLLOWING FUNCTIONS TRANSLATE THE CONSTRAINTS ON THE PROBLEM INTO EQUATIONS FOR THE FINAL SYSTEM OF EQUATIONS TO SOLVE
        def Point_on_curve(P):
            # Function returning the row coefficients in the system Ax=b corresponding to the equation 
            # obtained when constraining the curve to pass through point P. Such equation corresponds 
            # basically to   H(P) = 0.
            x, y = P
            Arow = [x*y, y**2, x**3, x**2*y, x*y**2, y**3]
            brow = -x**2
            return Arow, brow

        def VerticalTangent(P):
            # Function returning the row coefficients in the system Ax=b corresponding to the equation
            # obtained when constraining the curve to have a vertical tangent at point P. Such equation  
            # corresponds basically to   dH/dy(P) = 0.
            x, y = P
            Arow = [x, 2*y, 0, x**2, 2*x*y, 3*y**2]
            brow = 0
            return Arow, brow

        def HorizontalTangent(P):
            # Function returning the row coefficients in the system Ax=b corresponding to the equation
            # obtained when constraining the curve to have a horizontal tangent at point P. Such equation  
            # corresponds basically to   dH/dx(P) = 0.
            x, y = P
            Arow = [y, 0, 3*x**2, 2*x*y, y**2, 0]
            brow = -2*x
            return Arow, brow

        def RightAngle_SaddlePoint(A,b):
            # Function imposing a 90º angle at the closed surface saddle point at (0,0), which can be shown 
            # is equivalent to fixing  A02 = -1
            # Hence, what we need to do is take the second column of matrix A, corresponding to the A02 factors,
            # multiply them by -1 and pass them to the system's RHS, vector b. Then, we will reduce the system size.
            
            bred = np.delete(b+A[:,1].reshape((6,1)),5,0)     # pass second column to RHS and delete last row
            A = np.delete(A,1,1)    # delete second column 
            Ared = np.delete(A,5,0)    # delete last row
            return Ared, bred
        
        # BUILD CONTROL POINTS
        P0 = np.array([self.R_SADDLE, self.Z_SADDLE])
        P1 = np.array([self.R_RIGHTMOST, self.Z_RIGHTMOST])
        P2 = np.array([self.R_LEFTMOST, self.Z_LEFTMOST])
        P3 = np.array([self.R_TOP, self.Z_TOP])
        
        # 1. RESCALE POINT COORDINATES SO THAT THE SADDLE POINT IS LOCATED AT ORIGIN (0,0)
        P1star = P1-P0
        P2star = P2-P0
        P3star = P3-P0

        # 2. COMPUTE HAMILTONIAN COEFFICIENTS
        # Build system matrices
        A = np.zeros([6,6])
        b = np.zeros([6,1])

        # Constraints on point P1 = (a1,b1)
        Arow11, brow11 = Point_on_curve(P1star)
        Arow12, brow12 = VerticalTangent(P1star)
        A[0,:] = Arow11
        b[0] = brow11
        A[1,:] = Arow12
        b[1] = brow12

        # Constraints on point P2 = (a2,b2)
        Arow21, brow21 = Point_on_curve(P2star)
        Arow22, brow22 = VerticalTangent(P2star)
        A[2,:] = Arow21
        b[2] = brow21
        A[3,:] = Arow22
        b[3] = brow22
        
        # Constraints on point P3 = (a3,b3)
        Arow31, brow31 = Point_on_curve(P3star)
        A[4,:] = Arow31
        b[4] = brow31

        # 90º on saddle point (0,0)
        Ared, bred = RightAngle_SaddlePoint(A,b)   # Now A = [5x5] and  b = [5x1]
        
        # Solve system of equations and obtain Hamiltonian coefficients
        Q, R = np.linalg.qr(Ared)
        y = np.dot(Q.T, bred)
        coeffs_red = np.linalg.solve(R, y)  # Hamiltonian coefficients  [5x1]
        
        coeffs = np.insert(coeffs_red,1,-1,0)        # insert second coefficient A02 = -1
        
        # 2. OBTAIN LEVEL-SET VALUES ON MESH NODES
        # HAMILTONIAN  ->>  Z(x,y) = H(x,y) = x**2 + A11xy + A02y**2 + A30x**3 + A21x**2y + A12xy**2 + A03y**3
        Xstar = self.X[:,0]-P0[0]
        Ystar = self.X[:,1]-P0[1]
        LS = np.zeros([self.Nn])
        for i in range(self.Nn):
            LS[i] = Xstar[i]**2+coeffs[0]*Xstar[i]*Ystar[i]+coeffs[1]*Ystar[i]**2+coeffs[2]*Xstar[i]**3+coeffs[3]*Xstar[i]**2*Ystar[i]+coeffs[4]*Xstar[i]*Ystar[i]**2+coeffs[5]*Ystar[i]**3

        # MODIFY HAMILTONIAN VALUES SO THAT OUTSIDE THE PLASMA REGION THE LEVEL-SET IS POSITIVE  
        for i in range(self.Nn):
            if self.X[i,0] < P2[0] or self.X[i,1] < P0[1]:
                LS[i] = np.abs(LS[i])
        return LS
    
    
    def ComputePSILevelSet(self,PSI):
        
        # OBTAIN POINTS CONFORMING THE NEW PLASMA DOMAIN BOUNDARY
        fig, ax = plt.subplots(figsize=(6, 8))
        cs = ax.tricontour(self.X[:,0],self.X[:,1], PSI, levels=[0])

        paths = list()

        # CHECK IF CONTOUR SETS CONTAINS SADDLE POINT OR COMPUTATIONAL BOUNDARY POINTS (CLOSE ENOUGH) 
        for item in cs.collections:
            for path in item.get_paths():
                path_dict = dict()
                path_dict['coords'] = path.vertices
                path_dict['saddlepoint'] = False
                path_dict['compbound'] = False
                for point in path.vertices:
                    # COMPUTE DISTANCE TO SADDLE POINT
                    dist_saddle = np.linalg.norm(point-self.Xcrit[1,1,0:2])
                    # COMPUTE DISTANCE TO COMPUTATIONAL BOUNDARY NODES
                    dist_bound = np.sqrt((self.X[self.BoundaryNodes,0]-point[0])**2+(self.X[self.BoundaryNodes,1]-point[1])**2)
                    # CHECK IF CONTOUR CONTAINS SADDLE POINT
                    if  dist_saddle < 0.1:
                        path_dict['saddlepoint'] = True
                        # CHECK IF CONTOUR CONTAINS COMPUTATIONAL DOMAIN BOUNDARY POINTS
                    elif np.any(dist_bound <= 0.1):
                        path_dict['compbound'] = True
                paths.append(path_dict)

        # DIFFERENT PROCEDURES:
        # 1. DISCARD SETS WHICH DO NOT CONTAIN THE SADDLE POINT
        paths_temp = list()
        for path in paths:
            if path['saddlepoint']:
                paths_temp.append(path)
        paths = paths_temp.copy()        

        # IF THERE ARE MORE THAN 1 CONTOUR SET CONTAINING THE SADDLE POINT, REMOVE THE SETS CONTAINING COMPUTATIONAL BOUNDARY POINTS
        if len(paths) > 1:
            paths_temp = list()
            for path in paths:
                if not path['compbound']:
                    paths_temp.append(path)
            paths = paths_temp.copy()
            # TAKE THE REMAINING SET AS THE NEW PLASMA BOUNDARY SET
            if len(paths) == 1:
                plasmaboundary = paths[0]['coords']
                
        # IF A SINGLE CONTOUR REMAINS, CHECK WHETHER IT CONTAINS COMPUTATIONAL BOUNDARIES 
        else:
            # IF THE REMAINING SET CONTAINS BOTH SADDLE POINT AND COMPUTATIONAL BOUNDARY POINTS
            if paths[0]['compbound']:       
                plasmaboundary = list()
                oncontour = False
                firstpass = True
                secondpass = False
                counter = 0
                for point in paths[0]['coords']:
                    if np.linalg.norm(point-self.Xcrit[1,1,0:2]) < 0.3 and firstpass:
                        oncontour = True 
                        firstpass = False
                        plasmaboundary.append(point)
                    elif oncontour:
                        plasmaboundary.append(point)
                        counter += 1
                    if counter > 50:
                        secondpass = True
                    if np.linalg.norm(point-self.Xcrit[1,1,0:2]) < 0.3 and secondpass: 
                        oncontour = False 
                                
                plasmaboundary.append(plasmaboundary[0])
                plasmaboundary = np.array(plasmaboundary)
            # IF THE REMAINING SET DOES NOT CONTAIN ANY COMPUTATIONAL BOUNDARY POINT, TAKE IT AS THE NEW PLASMA BOUNDARY SET 
            else: 
                plasmaboundary = paths[0]['coords']

        fig.clear()
        plt.close(fig)

        # Create a Path object for the new plasma domain
        polygon_path = mpath.Path(plasmaboundary)
        # Check if the mesh points are inside the new plasma domain
        inside = polygon_path.contains_points(self.X)

        # 1. INVERT SIGN DEPENDING ON SOLUTION PLASMA REGION SIGN
        if self.PSI_0 > 0: # WHEN THE OBTAINED SOLUTION IS POSITIVE INSIDE THE PLASMA
            PSILevSet = -PSI.copy()
        else: # WHEN THE OBTAINED SOLUTION IS NEGATIVE INSIDE THE PLASMA
            PSILevSet = PSI.copy()

        # 2. DISCARD POINTS OUTSIDE THE PLASMA REGION
        for inode in range(self.Nn):
            if not inside[inode]:
                PSILevSet[inode] = np.abs(PSILevSet[inode])
    
        return PSILevSet
    
    ##################################################################################################
    ################################# VACUUM VESSEL BOUNDARY PSI_B ###################################
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
        PSI_B = np.zeros([self.Nnbound])    
        
        # FOR FIXED PLASMA BOUNDARY PROBLEM, THE PSI BOUNDARY VALUES PSI_B ARE EQUAL TO THE ANALYTICAL SOLUTION
        if self.PLASMA_BOUNDARY == self.FIXED_BOUNDARY:
            for inode, node in enumerate(self.BoundaryNodes):
                # ISOLATE BOUNDARY NODE COORDINATES
                Xbound = self.X[node,:]
                # COMPUTE PSI BOUNDARY VALUES
                PSI_B[inode] = self.PSIAnalyticalSolution(Xbound,self.PLASMA_CURRENT)

        # FOR THE FREE BOUNDARY PROBLEM, THE PSI BOUNDARY VALUES ARE COMPUTED BY PROJECTING THE MAGNETIC CONFINEMENT EFFECT USING THE GREENS FUNCTION FORMALISM
        elif self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:  
            for inode, node in enumerate(self.BoundaryNodes):
                # ISOLATE BOUNDARY NODE COORDINATES
                Xbound = self.X[node,:]
                
                ##### COMPUTE PSI BOUNDARY VALUES
                # CONTRIBUTION FROM EXTERNAL COILS CURRENT 
                for COIL in self.COILS: 
                    PSI_B[inode] += self.mu0 * COIL.Psi(Xbound)
                
                # CONTRIBUTION FROM EXTERNAL SOLENOIDS CURRENT   
                for SOLENOID in self.SOLENOIDS:
                    PSI_B[inode] += self.mu0 * SOLENOID.Psi(Xbound)
                            
                # CONTRIBUTION FROM PLASMA CURRENT  ->>  INTEGRATE OVER PLASMA REGION
                #   1. INTEGRATE IN PLASMA ELEMENTS
                for ielem in self.PlasmaElems:
                    # ISOLATE ELEMENT OBJECT
                    ELEMENT = self.Elements[ielem]
                    # INTERPOLATE ELEMENTAL PSI ON PHYSICAL GAUSS NODES
                    PSIg = ELEMENT.Ng @ ELEMENT.PSIe
                    # LOOP OVER GAUSS NODES
                    for ig in range(ELEMENT.ng):
                        for l in range(ELEMENT.n):
                            PSI_B[inode] += self.mu0 * GreensFunction(Xbound, ELEMENT.Xg[ig,:])*self.Jphi(ELEMENT.Xg[ig,:],
                                                        PSIg[ig])*ELEMENT.Ng[ig,l]*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]*self.gamma
                                    
                #   2. INTEGRATE IN CUT ELEMENTS, OVER SUBELEMENT IN PLASMA REGION
                for ielem in self.PlasmaBoundElems:
                    # ISOLATE ELEMENT OBJECT
                    ELEMENT = self.Elements[ielem]
                    # INTEGRATE ON SUBELEMENT INSIDE PLASMA REGION
                    for SUBELEM in ELEMENT.SubElements:
                        if SUBELEM.Dom < 0:  # IN PLASMA REGION
                            # INTERPOLATE ELEMENTAL PSI ON PHYSICAL GAUSS NODES
                            PSIg = SUBELEM.Ng @ ELEMENT.PSIe
                            # LOOP OVER GAUSS NODES
                            for ig in range(SUBELEM.ng):
                                for l in range(SUBELEM.n):
                                    PSI_B[inode] += self.mu0 * GreensFunction(Xbound, SUBELEM.Xg[ig,:])*self.Jphi(SUBELEM.Xg[ig,:],
                                                        PSIg[ig])*SUBELEM.Ng[ig,l]*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]*self.gamma   
        return PSI_B
    
    ##################################################################################################
    ###################################### ELEMENTS DEFINITION #######################################
    ##################################################################################################
    
    def IdentifyNearestNeighbors(self):
        """
        Finds the nearest neighbours for each element in mesh.
        """
        edgemap = {}  # Dictionary to map edges to elements

        # Loop over all elements to populate the edgemap dictionary
        for ELEM in self.Elements:
            # Initiate elemental nearest neigbhors attribute
            ELEM.neighbours = -np.ones([self.numedges],dtype=int)
            # Get the edges of the element (as sorted tuples for uniqueness)
            edges = list()
            for iedge in range(ELEM.numedges):
                edges.append(tuple(sorted([ELEM.Te[iedge],ELEM.Te[int((iedge+1)%ELEM.numedges)]])))
            
            for local_edge_idx, edge in enumerate(edges):
                if edge not in edgemap:
                    edgemap[edge] = []
                edgemap[edge].append((ELEM.index, local_edge_idx))
        
        # Loop over the edge dictionary to find neighbours
        for edge, elements in edgemap.items():
            if len(elements) == 2:  # Shared edge between two elements
                (elem1, edge1), (elem2, edge2) = elements
                self.Elements[elem1].neighbours[edge1] = elem2
                self.Elements[elem2].neighbours[edge2] = elem1
        return
    
    def IdentifyVacuumVesselElements(self):
        """
        Identifies elements on the vacuum vessel first wall. 
        """
        # GLOBAL INDEXES OF ELEMENTS CONTAINING THE VACUUM VESSEL FIRST WALL
        self.VacVessWallElems = np.unique(self.Tbound[:,-1]) 
        # ASSIGN ELEMENTAL DOMAIN FLAG
        for ielem in self.VacVessWallElems:
            self.Elements[ielem].Dom = 2  
  
        return
    
    
    def ClassifyElements(self):
        """ 
        Function that separates the elements inside vacuum vessel domain into 3 groups: 
                - PlasmaElems: elements inside the plasma region 
                - PlasmaBoundElems: (cut) elements containing the plasma region boundary 
                - VacuumElems: elements outside the plasma region which are not the vacuum vessel boundary
        """
                
        """ FOR HIGH ORDER ELEMENTS (QUADRATIC, CUBIC...), ELEMENTS LYING ON GEOMETRY BOUNDARIES OR INTERFACES MAY BE CLASSIFIED AS SUCH DUE TO
        THE LEVEL-SET SIGN ON NODES WHICH ARE NOT VERTICES OF THE ELEMENT ('HIGH ORDER' NODES). IN CASES WHERE ON SUCH NODES POSSES DIFFERENT SIGN, THIS MAY LEAD TO AN INTERFACE
        WHICH CUTS TWICE THE SAME ELEMENTAL EDGE, MEANING THE INTERFACE ENTERS AND LEAVES THROUGH THE SAME SEGMENT. THE PROBLEM WITH THAT IS THE SUBROUTINE 
        RESPONSIBLE FOR APPROXIMATING THE INTERFACE INSIDE ELEMENTS ONLY SEES THE LEVEL-SET VALUES ON THE VERTICES, BECAUSE IT DOES ONLY BOTHER ON WHETHER THE 
        ELEMENTAL EDGE IS CUT OR NOT. 
        IN LIGHT OF SUCH OCURRENCES, THE CLASSIFICATION OF ELEMENTS BASED ON LEVEL-SET SIGNS WILL BE IMPLEMENTED SUCH THAT ONLY THE VALUES ON THE VERTICES ARE
        TAKEN INTO ACCOUNT. THAT WAY, THIS CASES ARE ELUDED. ON THE OTHER HAND, WE NEED TO DETECT ALSO SUCH CASES IN ORDER TO MODIFY THE VALUES OF THE MESH 
        LEVEL-SET VALUES AND ALSO ON THE ELEMENTAL VALUES. """
        
        self.PlasmaElems = np.zeros([self.Ne], dtype=int)        # GLOBAL INDEXES OF ELEMENTS INSIDE PLASMA REGION
        self.VacuumElems = np.zeros([self.Ne], dtype=int)        # GLOBAL INDEXES OF ELEMENTS OUTSIDE PLASMA REGION (VACUUM REGION)
        self.PlasmaBoundElems = np.zeros([self.Ne], dtype=int)   # GLOBAL INDEXES OF ELEMENTS CONTAINING PLASMA BOUNDARY
        
        kplasm = 0
        kvacuu = 0
        kint = 0
        
        def CheckElementalVerticesLevelSetSigns(LSe):
            region = None
            DiffHighOrderNodes = []
            # CHECK SIGN OF LEVEL SET ON ELEMENTAL VERTICES
            for i in range(self.numedges-1):
                # FIND ELEMENTS LYING ON THE INTERFACE (LEVEL-SET VERTICES VALUES EQUAL TO 0 OR WITH DIFFERENT SIGN)
                if LSe[i] == 0:  # if node is on Level-Set 0 contour
                    region = 0
                    break
                elif np.sign(LSe[i]) !=  np.sign(LSe[i+1]):  # if the sign between vertices values change -> INTERFACE ELEMENT
                    region = 0
                    break
                # FIND ELEMENTS LYING INSIDE A SPECIFIC REGION (LEVEL-SET VERTICES VALUES WITH SAME SIGN)
                else:
                    if i+2 == self.numedges:   # if all vertices values have the same sign
                        # LOCATE ON WHICH REGION LIES THE ELEMENT
                        if np.sign(LSe[i+1]) > 0:   # all vertices values with positive sign -> EXTERIOR REGION ELEMENT
                            region = +1
                        else:   # all vertices values with negative sign -> INTERIOR REGION ELEMENT 
                            region = -1
                            
                        # CHECK LEVEL-SET SIGN ON ELEMENTAL 'HIGH ORDER' NODES
                        #for i in range(self.numedges,self.n-self.numedges):  # LOOP OVER NODES WHICH ARE NOT ON VERTICES
                        for i in range(self.numedges,self.n):
                            if np.sign(LSe[i]) != np.sign(LSe[0]):
                                DiffHighOrderNodes.append(i)
                
            return region, DiffHighOrderNodes
            
        for ielem in range(self.Ne):
            regionplasma, DHONplasma = CheckElementalVerticesLevelSetSigns(self.Elements[ielem].LSe)    
            if regionplasma < 0:   # ALL PLASMA LEVEL-SET NODAL VALUES NEGATIVE -> INSIDE PLASMA DOMAIN 
                self.PlasmaElems[kplasm] = ielem
                self.Elements[ielem].Dom = -1
                kplasm += 1
            elif regionplasma == 0:  # DIFFERENT SIGN IN PLASMA LEVEL-SET NODAL VALUES -> PLASMA/VACUUM INTERFACE ELEMENT
                self.PlasmaBoundElems[kint] = ielem
                self.Elements[ielem].Dom = 0
                kint += 1
            elif regionplasma > 0: # ALL PLASMA LEVEL-SET NODAL VALUES POSITIVE -> OUTSIDE PLASMA DOMAIN
                if self.Elements[ielem].Dom == 2:  # ALREADY CLASSIFIED AS VACUUM VESSEL ELEMENT
                    continue
                else:  # IF NOT VACUUM VESSEL ELEMENTS -> VACUUM ELEMENTS
                    self.VacuumElems[kvacuu] = ielem
                    self.Elements[ielem].Dom = +1
                    kvacuu += 1
                  
            # IF THERE EXISTS 'HIGH-ORDER' NODES WITH DIFFERENT PLASMA LEVEL-SET SIGN
            if DHONplasma:
                self.OLDplasmaBoundLevSet = self.PlasmaBoundLevSet.copy()
                for inode in DHONplasma:  # LOOP OVER LOCAL INDICES 
                    self.Elements[ielem].LSe[inode] *= -1 
                    self.PlasmaBoundLevSet[self.Elements[ielem].Te[inode]] *= -1       
        
        # DELETE REST OF UNUSED MEMORY
        self.PlasmaElems = self.PlasmaElems[:kplasm]
        self.VacuumElems = self.VacuumElems[:kvacuu]
        self.PlasmaBoundElems = self.PlasmaBoundElems[:kint]
        
        # GATHER NON-CUT ELEMENTS  
        self.NonCutElems = np.concatenate((self.PlasmaElems, self.VacuumElems, self.VacVessWallElems), axis=0)
        
        # CLASSIFY NODES ACCORDING TO NEW ELEMENT CLASSIFICATION
        self.ClassifyNodes()
        return
    
    def ObtainClassification(self):
        """ 
        Function which produces an array where the different values code for the groups in which elements are classified. 
        """
        Classification = np.zeros([self.Ne],dtype=int)
        for ielem in self.PlasmaElems:
            Classification[ielem] = -1
        for ielem in self.PlasmaBoundElems:
            Classification[ielem] = 0
        for ielem in self.VacuumElems:
            Classification[ielem] = +1
        for ielem in self.VacVessWallElems:
            Classification[ielem] = +2
            
        return Classification
    
    
    def ClassifyNodes(self):
        self.PlasmaNodes = set()
        self.VacuumNodes = set()
        for ielem in self.PlasmaElems:
            for node in self.T[ielem,:]:
                self.PlasmaNodes.add(node) 
        for ielem in self.VacuumElems:
            for node in self.T[ielem,:]:
                self.VacuumNodes.add(node)    
        for ielem in self.PlasmaBoundElems:
            for node in self.T[ielem,:]:
                if self.PlasmaBoundLevSet[node] < 0:
                    self.PlasmaNodes.add(node)
                else:
                    self.VacuumNodes.add(node)
        for ielem in self.VacVessWallElems:
            for node in self.T[ielem,:]:
                self.VacuumNodes.add(node)
        for node in self.BoundaryNodes:
            self.VacuumNodes.remove(node)   
        
        self.PlasmaNodes = np.array(sorted(list(self.PlasmaNodes)))
        self.VacuumNodes = np.array(sorted(list(self.VacuumNodes)))
        return
    
    def ComputePlasmaBoundaryNumberNodes(self):
        """
        Computes the total number of nodes located on the plasma boundary approximation
        """  
        nnodes = 0
        for ielem in self.PlasmaBoundElems:
            for SEGMENT in self.Elements[ielem].InterfApprox.Segments:
                nnodes += SEGMENT.ng
        return nnodes
    
    
    def IdentifyPlasmaBoundaryGhostFaces(self):
        """
        Identifies the elemental ghost faces on which the ghost penalty term needs to be integrated, for elements containing the plasma
        boundary.
        """
        
        # RESET ELEMENTAL GHOST FACES
        for ielem in np.concatenate((self.PlasmaBoundElems,self.PlasmaElems),axis=0):
            self.Elements[ielem].GhostFaces = None
            
        GhostFaces_dict = dict()    # [(CUT_EDGE_NODAL_GLOBAL_INDEXES): {(ELEMENT_INDEX_1, EDGE_INDEX_1), (ELEMENT_INDEX_2, EDGE_INDEX_2)}]
        
        for ielem in self.PlasmaBoundElems:
            ELEM = self.Elements[ielem]
            for iedge, neighbour in enumerate(ELEM.neighbours):
                if neighbour >= 0 and self.Elements[neighbour].Dom <= 0:
                    # IDENTIFY THE CORRESPONDING FACE IN THE NEIGHBOUR ELEMENT
                    NEIGHBOUR = self.Elements[neighbour]
                    neighbour_edge = list(NEIGHBOUR.neighbours).index(ELEM.index)
                    # OBTAIN GLOBAL INDICES OF GHOST FACE NODES
                    nodes = np.zeros([ELEM.nedge],dtype=int)
                    nodes[0] = iedge
                    nodes[1] = (iedge+1)%ELEM.numedges
                    for knode in range(self.ElOrder-1):
                        nodes[2+knode] = self.numedges + iedge*(self.ElOrder-1)+knode
                    if tuple(sorted(ELEM.Te[nodes])) not in GhostFaces_dict:
                        GhostFaces_dict[tuple(sorted(ELEM.Te[nodes]))] = set()
                    GhostFaces_dict[tuple(sorted(ELEM.Te[nodes]))].add((ELEM.index,iedge))
                    GhostFaces_dict[tuple(sorted(ELEM.Te[nodes]))].add((neighbour,neighbour_edge))
                    
        XIe = ReferenceElementCoordinates(self.ElType,self.ElOrder)
        GhostFaces = list()
        GhostElems = set()

        for elems in GhostFaces_dict.values():
            # ISOLATE ADJACENT ELEMENTS
            (ielem1, iedge1), (ielem2,iedge2) = elems
            ELEM1 = self.Elements[ielem1]
            ELEM2 = self.Elements[ielem2]
            # DECLARE NEW GHOST FACES ELEMENTAL ATTRIBUTE
            if ELEM1.GhostFaces == None:  
                ELEM1.GhostFaces = list()
            if ELEM2.GhostFaces == None:  
                ELEM2.GhostFaces = list()
                
            # ADD GHOST FACE TO ELEMENT 1
            nodes1 = np.zeros([ELEM1.nedge],dtype=int)
            nodes1[0] = iedge1
            nodes1[1] = (iedge1+1)%ELEM1.numedges
            for knode in range(ELEM1.ElOrder-1):
                nodes1[2+knode] = ELEM1.numedges + iedge1*(ELEM1.ElOrder-1)+knode
            ELEM1.GhostFaces.append(Segment(index = iedge1,
                                            ElOrder = ELEM1.ElOrder,
                                            Tseg = nodes1,
                                            Xseg = ELEM1.Xe[nodes1,:],
                                            XIseg = XIe[nodes1,:]))
            
            # ADD GHOST FACE TO ELEMENT 2
            nodes2 = np.zeros([ELEM2.nedge],dtype=int)
            nodes2[0] = iedge2
            nodes2[1] = (iedge2+1)%ELEM2.numedges
            for knode in range(ELEM2.ElOrder-1):
                nodes2[2+knode] = ELEM2.numedges + iedge2*(ELEM2.ElOrder-1)+knode
            ELEM2.GhostFaces.append(Segment(index = iedge2,
                                            ElOrder = ELEM2.ElOrder,
                                            Tseg = nodes2,
                                            Xseg = ELEM2.Xe[nodes2,:],
                                            XIseg = XIe[nodes2,:]))
            
            # CORRECT SECOND ADJACENT ELEMENT GHOST FACE TO MATCH NODES -> PERMUTATION
            permutation = [list(ELEM2.Te[nodes2]).index(x) for x in ELEM1.Te[nodes1]]
            ELEM2.GhostFaces[-1].Xseg = ELEM2.GhostFaces[-1].Xseg[permutation,:]
            ELEM2.GhostFaces[-1].XIseg = ELEM2.GhostFaces[-1].XIseg[permutation,:]

            GhostFaces.append((list(ELEM1.Te[nodes1]),(ielem1,iedge1,len(ELEM1.GhostFaces)-1),(ielem2,iedge2,len(ELEM2.GhostFaces)-1), permutation))
            GhostElems.add(ielem1)
            GhostElems.add(ielem2)
            
        return GhostFaces, list(GhostElems)
    
    
    ##################################################################################################
    ############################### SOLUTION NORMALISATION ###########################################
    ##################################################################################################
    
    # SEARCH ELEMENT CONTAINING POINT IN MESH
    def SearchElement(self,X,searchelements):
        """
        Identify the element within a specified list searchelements and contains a given point X.

        Input:
            - X (array-like): Coordinates of the point to locate, specified as [x, y].
            - searchelements (list of int): List of element indices to search within.

        Output:
            elem (int or None): Index of the element containing the point X. 
                                Returns None if no element contains the point.
        """

        if self.ElType == 1: # FOR TRIANGULAR ELEMENTS
            for elem in searchelements:
                Xe = self.Elements[elem].Xe
                # Calculate the cross products (c1, c2, c3) for the point relative to each edge of the triangle
                c1 = (Xe[1,0]-Xe[0,0])*(X[1]-Xe[0,1])-(Xe[1,1]-Xe[0,1])*(X[0]-Xe[0,0])
                c2 = (Xe[2,0]-Xe[1,0])*(X[1]-Xe[1,1])-(Xe[2,1]-Xe[1,1])*(X[0]-Xe[1,0])
                c3 = (Xe[0,0]-Xe[2,0])*(X[1]-Xe[2,1])-(Xe[0,1]-Xe[2,1])*(X[0]-Xe[2,0])
                if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0): # INSIDE TRIANGLE
                    return elem
        elif self.ElType == 2: # FOR QUADRILATERAL ELEMENTS
            for elem in searchelements:
                Xe = self.Elements[elem].Xe
                # This algorithm counts how many times a ray starting from the point intersects the edges of the quadrilateral. 
                # If the count is odd, the point is inside; otherwise, it is outside.
                inside = False
                for i in range(4):
                    if ((Xe[i,1] > X[1]) != (Xe[(i+1)%4,1]>X[1])) and (X[0]<(Xe[(i+1)%4,0]-Xe[i,0])*(X[1]-Xe[i,1])/(Xe[(i+1)%4,1]-Xe[i,1])+Xe[i,0]):
                        inside = not inside
                if inside:
                    return elem
                
    def funcPSI(self,X):
        """ Interpolates PSI value at point X. """
        elem = self.SearchElement(X,range(self.Ne))
        psi = self.Elements[elem].ElementalInterpolationPHYSICAL(X,self.PSI[self.Elements[elem].Te])
        return psi
    
    
    def gradPSI(self,X):
        """ Interpolates PSI gradient at point X. """
        elem = self.SearchElement(X,range(self.Ne))
        gradpsi = self.Elements[elem].GRADElementalInterpolationPHYSICAL(X,self.PSI[self.Elements[elem].Te])
        return gradpsi
    
    
    def ComputeCriticalPSI(self,PSI):
        """
        Compute the critical values of the magnetic flux function (PSI).

        Input:
            PSI (array-like): Poloidal magnetic flux function values at the computational domain nodes.

        The following attributes are updated:
                - self.PSI_0 (float): Value of PSI at the magnetic axis (local extremum).
                - self.PSI_X (float): Value of PSI at the separatrix (saddle point), if applicable.
                - self.Xcrit (array-like): Coordinates and element indices of critical points:
                    - self.Xcrit[1,0,:] -> Magnetic axis ([R, Z, element index]).
                    - self.Xcrit[1,1,:] -> Saddle point ([R, Z, element index]).
        """
        # INTERPOLATION OF GRAD(PSI)
        def gradPSI(X,Rfine,Zfine,gradPSIfine):
            dPSIdr = griddata((Rfine.flatten(),Zfine.flatten()), gradPSIfine[0].flatten(), (X[0],X[1]), method='cubic')
            dPSIdz = griddata((Rfine.flatten(),Zfine.flatten()), gradPSIfine[1].flatten(), (X[0],X[1]), method='cubic')
            GRAD = np.array([dPSIdr,dPSIdz])
            return GRAD
        
        # EVALUATE HESSIAN MATRIX ENTRIES
        def EvaluateHESSIAN(X,gradPSIfine,Rfine,Zfine,dr,dz):
            # compute second derivatives on fine mesh
            dgradPSIdrfine = np.gradient(gradPSIfine[0],dr,dz)
            dgradPSIdzfine = np.gradient(gradPSIfine[1],dr,dz)
            # interpolate HESSIAN components on point 
            dPSIdrdr = griddata((Rfine.flatten(),Zfine.flatten()), dgradPSIdrfine[0].flatten(), (X[0],X[1]), method='cubic')
            dPSIdzdr = griddata((Rfine.flatten(),Zfine.flatten()), dgradPSIdrfine[1].flatten(), (X[0],X[1]), method='cubic')
            dPSIdzdz = griddata((Rfine.flatten(),Zfine.flatten()), dgradPSIdzfine[1].flatten(), (X[0],X[1]), method='cubic')
            if dPSIdrdr*dPSIdzdz-dPSIdzdr**2 > 0:
                return "LOCAL EXTREMUM"
            else:
                return "SADDLE POINT"
            
        # 1. INTERPOLATE PSI VALUES ON A FINER STRUCTURED MESH USING PSI ON NODES
        # DEFINE FINER STRUCTURED MESH
        Mr = 75
        Mz = 105
        rfine = np.linspace(self.Rmin, self.Rmax, Mr)
        zfine = np.linspace(self.Zmin, self.Zmax, Mz)
        # INTERPOLATE PSI VALUES
        Rfine, Zfine = np.meshgrid(rfine,zfine)
        PSIfine = griddata((self.X[:,0],self.X[:,1]), PSI.T[0], (Rfine, Zfine), method='cubic')
        # CORRECT VALUES AT INTRPOLATION POINTS OUTSIDE OF COMPUTATIONAL DOMAIN
        for ir in range(Mr):
            for iz in range(Mz):
                if np.isnan(PSIfine[iz,ir]):
                    PSIfine[iz,ir] = 0
        
        # 2. DEFINE GRAD(PSI) WITH FINER MESH VALUES USING FINITE DIFFERENCES
        dr = (self.Rmax-self.Rmin)/Mr
        dz = (self.Zmax-self.Zmin)/Mz
        gradPSIfine = np.gradient(PSIfine,dr,dz)
        
        # FIND SOLUTION OF  GRAD(PSI) = 0   NEAR MAGNETIC AXIS AND SADDLE POINT 
        if self.it == 1:
            X0_extr = np.array([self.EXTR_R0,self.EXTR_Z0],dtype=float)
            X0_saddle = np.array([self.SADD_R0,self.SADD_Z0],dtype=float)
        else:
            # TAKE PREVIOUS SOLUTION AS INITIAL GUESS
            X0_extr = self.Xcrit[0,0,:-1]
            X0_saddle = self.Xcrit[0,1,:-1]
            
        # 3. LOOK FOR LOCAL EXTREMUM
        sol = optimize.root(gradPSI, X0_extr, args=(Rfine,Zfine,gradPSIfine))
        if sol.success == True:
            self.Xcrit[1,0,:-1] = sol.x
            # 4. CHECK HESSIAN LOCAL EXTREMUM
            # LOCAL EXTREMUM
            nature = EvaluateHESSIAN(self.Xcrit[1,0,:-1], gradPSIfine, Rfine, Zfine, dr, dz)
            if nature != "LOCAL EXTREMUM":
                print("ERROR IN LOCAL EXTREMUM HESSIAN")
            # 5. INTERPOLATE VALUE OF PSI AT LOCAL EXTREMUM
            # LOOK FOR ELEMENT CONTAINING LOCAL EXTREMUM
            elem = self.SearchElement(self.Xcrit[1,0,:-1],self.PlasmaElems)
            self.Xcrit[1,0,-1] = elem
        else:
            if self.it == 1:
                print("LOCAL EXTREMUM NOT FOUND. TAKING SOLUTION AT INITIAL GUESS")
                elem = self.SearchElement(X0_extr,self.PlasmaElems)
                self.Xcrit[1,0,:-1] = X0_extr
                self.Xcrit[1,0,-1] = elem
            else:
                print("LOCAL EXTREMUM NOT FOUND. TAKING PREVIOUS SOLUTION")
                self.Xcrit[1,0,:] = self.Xcrit[0,0,:]
            
        # INTERPOLATE PSI VALUE ON CRITICAL POINT
        self.PSI_0 = self.Elements[int(self.Xcrit[1,0,-1])].ElementalInterpolationPHYSICAL(self.Xcrit[1,0,:-1],PSI[self.Elements[int(self.Xcrit[1,0,-1])].Te]) 
        print('LOCAL EXTREMUM AT ',self.Xcrit[1,0,:-1],' (ELEMENT ', int(self.Xcrit[1,0,-1]),') WITH VALUE PSI_0 = ',self.PSI_0)
            
        if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:
            # 3. LOOK FOR SADDLE POINT
            sol = optimize.root(gradPSI, X0_saddle, args=(Rfine,Zfine,gradPSIfine))
            if sol.success == True:
                self.Xcrit[1,1,:-1] = sol.x 
                # 4. CHECK HESSIAN SADDLE POINT
                nature = EvaluateHESSIAN(self.Xcrit[1,1,:-1], gradPSIfine, Rfine, Zfine, dr, dz)
                if nature != "SADDLE POINT":
                    print("ERROR IN SADDLE POINT HESSIAN")
                # 5. INTERPOLATE VALUE OF PSI AT SADDLE POINT
                # LOOK FOR ELEMENT CONTAINING SADDLE POINT
                elem = self.SearchElement(self.Xcrit[1,1,:-1],np.concatenate((self.VacuumElems,self.PlasmaBoundElems,self.PlasmaElems),axis=0))
                self.Xcrit[1,1,-1] = elem
            else:
                if self.it == 1:
                    print("SADDLE POINT NOT FOUND. TAKING SOLUTION AT INITIAL GUESS")
                    elem = self.SearchElement(self.Xcrit[0,1,:-1],self.PlasmaBoundElems)
                    self.Xcrit[1,1,:-1] = self.Xcrit[0,1,:-1]
                    self.Xcrit[1,1,-1] = elem
                else:
                    print("SADDLE POINT NOT FOUND. TAKING PREVIOUS SOLUTION")
                    self.Xcrit[1,1,:] = self.Xcrit[0,1,:]
                
            # INTERPOLATE PSI VALUE ON CRITICAL POINT
            self.PSI_X = self.Elements[int(self.Xcrit[1,1,-1])].ElementalInterpolationPHYSICAL(self.Xcrit[1,1,:-1],PSI[self.Elements[int(self.Xcrit[1,1,-1])].Te]) 
            print('SADDLE POINT AT ',self.Xcrit[1,1,:-1],' (ELEMENT ', int(self.Xcrit[1,1,-1]),') WITH VALUE PSI_X = ',self.PSI_X)
        
        else:
            self.Xcrit[1,1,:-1] = [self.Rmin,self.Zmin]
            self.PSI_X = 0
            
        return 

    
    def NormalisePSI(self):
        """
        Normalize the magnetic flux function (PSI) based on critical PSI values (PSI_0 and PSI_X).
        """
        if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY or self.PLASMA_CURRENT == self.PROFILES_CURRENT:
            for i in range(self.Nn):
                self.PSI_NORM[i,1] = (self.PSI[i]-self.PSI_X)/np.abs(self.PSI_0-self.PSI_X)
                #self.PSI_NORM[i,1] = (self.PSI_X-self.PSI[i])/(self.PSI_X-self.PSI_0)
        else: 
            for i in range(self.Nn):
                self.PSI_NORM[i,1] = self.PSI[i]
        return 
    
    
    def ComputeTotalPlasmaCurrent(self):
        """ Function that computes de total toroidal current carried by the plasma """  
        return self.PlasmaDomainIntegral(self.Jphi)
    
    
    def ComputeTotalPlasmaCurrentNormalization(self):
        """
        Compute and apply a correction factor to ensure the total plasma current in the computational domain matches the specified input parameter `TOTAL_CURRENT`.
        """
        if self.PLASMA_CURRENT == self.PROFILES_CURRENT:
            # COMPUTE TOTAL PLASMA CURRENT    
            Tcurrent = self.ComputeTotalPlasmaCurrent()
            print('Total plasma current computed = ', Tcurrent)    
            # COMPUTE TOTAL PLASMA CURRENT CORRECTION FACTOR            
            self.gamma = self.TOTAL_CURRENT/Tcurrent
            print("Total plasma current normalization factor = ", self.gamma)
            # COMPUTED NORMALISED TOTAL PLASMA CURRENT
            print("Normalised total plasma current = ", Tcurrent*self.gamma)
        
        return
    
    
    ##################################################################################################
    ###################### CONVERGENCE VALIDATION and VARIABLES UPDATING #############################
    ##################################################################################################
    
    def CheckConvergence(self,VALUES):
        """
        Function to evaluate convergence criteria during iterative computation. 
        Based on the type of value being checked (`PSI_NORM` or `PSI_B`), it calculates 
        the L2 norm of the residual between consecutive iterations and determines if the solution 
        has converged to the desired tolerance.

        Input:
            VALUES (str) Specifies the variable type to check for convergence:
                - "PSI_NORM" : Normalized magnetic flux (used for internal convergence).
                - "PSI_B"    : Boundary flux (used for external convergence).
        """
        if VALUES == "PSI_NORM":
            # FOR THE LINEAR AND ZHENG MODELS (FIXED BOUNDARY) THE SOURCE TERM DOESN'T DEPEND ON PSI, THEREFORE A SINGLE INTERNAL ITERATION IS ENOUGH
            if self.PLASMA_CURRENT == self.LINEAR_CURRENT or self.PLASMA_CURRENT == self.ZHENG_CURRENT:
                self.converg_INT = True  # STOP INTERNAL WHILE LOOP 
                self.residu_INT = 0
            else:
                # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
                if np.linalg.norm(self.PSI_NORM[:,1]) > 0:
                    L2residu = np.linalg.norm(self.PSI_NORM[:,1] - self.PSI_NORM[:,0])/np.linalg.norm(self.PSI_NORM[:,1])
                else: 
                    L2residu = np.linalg.norm(self.PSI_NORM[:,1] - self.PSI_NORM[:,0])
                if L2residu < self.INT_TOL:
                    self.converg_INT = True   # STOP INTERNAL WHILE LOOP 
                else:
                    self.converg_INT = False
                    
                self.residu_INT = L2residu
                print("Internal iteration = ",self.it_INT,", PSI_NORM residu = ", L2residu)
                print(" ")
            
        elif VALUES == "PSI_B":
            # FOR FIXED BOUNDARY PROBLEM, THE BOUNDARY VALUES ARE ALWAYS THE SAME, THEREFORE A SINGLE EXTERNAL ITERATION IS NEEDED
            if self.PLASMA_BOUNDARY == self.FIXED_BOUNDARY:
                self.converg_EXT = True  # STOP EXTERNAL WHILE LOOP 
                self.residu_EXT = 0
            else:
                # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
                if np.linalg.norm(self.PSI_B[:,1]) > 0:
                    L2residu = np.linalg.norm(self.PSI_B[:,1] - self.PSI_B[:,0])/np.linalg.norm(self.PSI_B[:,1])
                else: 
                    L2residu = np.linalg.norm(self.PSI_B[:,1] - self.PSI_B[:,0])
                if L2residu < self.EXT_TOL:
                    self.converg_EXT = True   # STOP EXTERNAL WHILE LOOP 
                else:
                    self.converg_EXT = False
                    
                self.residu_EXT = L2residu
                print("External iteration = ",self.it_EXT,", PSI_B residu = ", L2residu)
                print(" ")
        return 
    
    def UpdatePSI(self,VALUES):
        """
        Updates the PSI arrays.

        Input:
            VALUES (str) 
                - 'PSI_NORM' : Updates the normalized PSI values.
                - 'PSI_B'    : Updates the boundary PSI values, or stores the converged values if external convergence is reached.
        """
        if VALUES == 'PSI_NORM':
            self.PSI_NORM[:,0] = self.PSI_NORM[:,1]
            self.Xcrit[0,:,:] = self.Xcrit[1,:,:]
        
        elif VALUES == 'PSI_B':
            if self.converg_EXT == False:
                self.PSI_B[:,0] = self.PSI_B[:,1]
                self.PSI_NORM[:,0] = self.PSI_NORM[:,1]
            elif self.converg_EXT == True:
                self.PSI_CONV = self.PSI_NORM[:,1]
        return
    
    def UpdateElementalPSI(self):
        """ 
        Function to update the elemental PSI values, respect to PSI_NORM.
        """
        for ELEMENT in self.Elements:
            ELEMENT.PSIe = self.PSI_NORM[ELEMENT.Te,0]  # TAKE VALUES OF ITERATION N
        return
    
    def UpdatePlasmaBoundaryValues(self):
        """
        Updates the plasma boundary PSI values constraints (PSIgseg) on the interface approximation segments integration points.
        """
        
        for ielem in self.PlasmaBoundElems:
            for SEGMENT in self.Elements[ielem].InterfApprox.Segments:
                # INITIALISE BOUNDARY VALUES
                SEGMENT.PSIgseg = np.zeros([SEGMENT.ng])
                # FOR EACH INTEGRATION POINT ON THE PLASMA/VACUUM INTERFACE APPROXIMATION SEGMENT
                for ignode in range(SEGMENT.ng):
                    # FIXED BOUNDARY PROBLEM -> ANALYTICAL SOLUTION PLASMA BOUNDARY VALUES 
                    if self.PLASMA_BOUNDARY == self.FIXED_BOUNDARY:
                        SEGMENT.PSIgseg[ignode] = self.PSIAnalyticalSolution(SEGMENT.Xg[ignode,:],self.PLASMA_CURRENT)
                    # FREE BOUNDARY PROBLEM -> PLASMA BOUNDARY VALUES = SEPARATRIX VALUE
                    else:
                        SEGMENT.PSIgseg[ignode] = self.PSI_X
        return
    
    def UpdateVacuumVesselBoundaryValues(self):
        
        
        
        return
    
    
    def UpdateElementalPlasmaLevSet(self):
        for ELEMENT in self.Elements:
            ELEMENT.LSe = self.PlasmaBoundLevSet[self.T[ELEMENT.index,:]]
        return
    
    
    ##################################################################################################
    ############################### OPERATIONS OVER GROUPS ###########################################
    ##################################################################################################
    
    def PlasmaDomainIntegral(self,fun):
        """ 
        Integrates function fun over plasma region surface, such that
                fun = fun(X,PSI)        
        """
        
        integral = 0
        # INTEGRATE OVER PLASMA ELEMENTS
        for ielem in self.PlasmaElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[ielem]
            # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
            PSIg = ELEMENT.Ng @ ELEMENT.PSIe
            # LOOP OVER ELEMENTAL NODES
            for i in range(ELEMENT.n):
                 # LOOP OVER GAUSS NODES
                for ig in range(ELEMENT.ng):
                    integral += fun(ELEMENT.Xg[ig,:],PSIg[ig])*ELEMENT.Ng[ig,i]*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                    
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for ielem in self.PlasmaBoundElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[ielem]
            # LOOP OVER SUBELEMENTS
            for SUBELEM in ELEMENT.SubElements:
                # INTEGRATE IN SUBDOMAIN INSIDE PLASMA REGION
                if SUBELEM.Dom < 0:
                    # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PSIg = SUBELEM.Ng @ ELEMENT.PSIe
                    # LOOP OVER GAUSS NODES
                    for ig in range(SUBELEM.ng):
                        # LOOP OVER ELEMENTAL NODES
                        for i in range(SUBELEM.n):
                            integral += fun(SUBELEM.Xg[ig,:],PSIg[ig])*SUBELEM.Ng[ig,i]*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]            
        return integral
        
    
    ##################### INITIALISATION 
    
    def InitialiseParameters(self):
        
        # TOKAMAK'S 1rst WALL GEOMETRY COEFFICIENTS, USED ALSO FOR LINEAR PLASMA MODEL ANALYTICAL SOLUTION (INITIAL GUESS)
        self.coeffsLINEAR = self.ComputeLinearSolutionCoefficients()
        # ZHENG ANALYTICAL SOLUTION COEFFICIENTS
        self.coeffsZHENG = self.ComputeZhengSolutionCoefficients()
        
        if self.PLASMA_CURRENT == self.PROFILES_CURRENT:
            # COMPUTE PRESSURE PROFILE FACTOR
            self.P0=self.B0*((self.kappa**2)+1)/(self.mu0*(self.R0**2)*self.q0*self.kappa)
            
        # OBTAIN BOUNDARY NODES
        self.BoundaryNodes = set()     # GLOBAL INDEXES OF NODES ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        for i in range(self.nedge):
            for node in self.Tbound[:,i]:
                self.BoundaryNodes.add(node)
        # CONVERT BOUNDARY NODES SET INTO ARRAY
        self.BoundaryNodes = list(sorted(self.BoundaryNodes))
        self.Nnbound = len(self.BoundaryNodes)
        
        # OBTAIN DOF NODES
        self.DOFNodes =  [x for x in list(range(self.Nn)) if x not in set(self.BoundaryNodes)]
        self.NnDOF = len(self.DOFNodes)
        
        # OBTAIN COMPUTATIONAL MESH LIMITS
        self.Rmax = np.max(self.X[:,0])
        self.Rmin = np.min(self.X[:,0])
        self.Zmax = np.max(self.X[:,1])
        self.Zmin = np.min(self.X[:,1])
            
        # INITIALISE CRITICAL POINTS ARRAY
        self.Xcrit = np.zeros([2,2,3])  # [(iterations n, n+1), (extremum, saddle point), (R_crit,Z_crit,elem_crit)]
        return
    
    
    def InitialisePlasmaLevelSet(self):
        """
        Computes the initial level-set function values describing the plasma boundary. Negative values represent inside the plasma region.
        """
            
        # FOR THE FIXED BOUNDARY CASE, WE AIM AT VALIDATING THE CORRECT INTEGRATION OF THE GRAD-SHAFRANOV'S EQUATION USING CUTFEM BY INTEGRATING 
        # A VERSION OF THE EQUATION FOR WHICH WE POSSES AN ANALYTICAL SOLUTION. IN ORDER TO DO SO, WE WANT TO DEFINE A FIXED PLASMA BOUNDARY 
        # SMALLER THAN THE VACUUM VESSEL (COMPUTATIONAL DOMAIN), SO THAT BOUNDARY CONDITIONS (ANALYTICAL SOLUTION) CAN WEAKLY BE IMPOSED ON THE 
        # BOUNDARY AND THUS CHECK IF THE CUTFEM INTEGRATION SCHEME WORKS CORRECTLY.  
        
        if self.PLASMA_BOUNDARY == self.FIXED_BOUNDARY:  
            #self.PlasmaBoundLevSet = (-1)*np.ones([self.Nn],dtype=float)
            
            self.PlasmaBoundLevSet = np.zeros([self.Nn],dtype=float)
            for inode in range(self.Nn):
                self.PlasmaBoundLevSet[inode] = (-1)*self.PSIAnalyticalSolution(self.X[inode,:],self.ZHENG_CURRENT)
                
        # FOR THE FREE BOUNDARY CASE, THE INITIAL PLASMA REGION SHAPE IS DESCRIBED USING THE F4E SHAPE CONTROL POINTS
        elif self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:    
            self.PlasmaBoundLevSet = self.F4E_PlasmaBoundLevSet()
            self.Xcrit[0,1,:-1] = np.array([self.R_SADDLE,self.Z_SADDLE])
        return 
    
    def InitialiseElements(self):
        """ 
        Function initialising attribute ELEMENTS which is a list of all elements in the mesh. 
        """
        self.Elements = [Element(index = e,
                                 ElType = self.ElType,
                                 ElOrder = self.ElOrder,
                                 Xe = self.X[self.T[e,:],:],
                                 Te = self.T[e,:],
                                 PlasmaLSe = self.PlasmaBoundLevSet[self.T[e,:]]) for e in range(self.Ne)]
        return
    
    
    def InitialGuess(self):
        """ 
        This function computes the problem's initial guess for PSI_NORM. 
        
        Output: 
            PSI0: PSI_NORM initial values
        """
        PSI0 = np.zeros([self.Nn])
        if self.PLASMA_CURRENT == self.PROFILES_CURRENT: 
            for i in range(self.Nn):
                PSI0[i] = self.PSIAnalyticalSolution(self.X[i,:],self.LINEAR_CURRENT)*(-0.5)
                #PSI0[i] = self.PSIAnalyticalSolution(self.X[i,:],self.ZHENG_CURRENT)*0.5
        else:     
            for i in range(self.Nn):
                PSI0[i] = self.PSIAnalyticalSolution(self.X[i,:],self.PLASMA_CURRENT)*2*random()
        return PSI0
    
    
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
        
        ####### COMPUTE NUMBER OF NODES ON PLASMA BOUNDARY APPROXIMATION
        self.NnPB = self.ComputePlasmaBoundaryNumberNodes()
        
        ####### INITIALISE PSI VECTORS
        print('         -> INITIALISE PSI ARRAYS...', end="")
        self.PSI = np.zeros([self.Nn])            # SOLUTION FROM SOLVING CutFEM SYSTEM OF EQUATIONS (INTERNAL LOOP)       
        self.PSI_NORM = np.zeros([self.Nn,2])     # NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        self.PSI_B = np.zeros([self.Nnbound,2])   # VACUUM VESSEL FIRST WALL PSI VALUES (EXTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)    
        self.PSI_CONV = np.zeros([self.Nn])       # CONVERGED SOLUTION FIELD
        print('Done!')
        
        ####### COMPUTE INITIAL GUESS AND STORE IT IN ARRAY FOR N=0
        # COMPUTE INITIAL GUESS
        print('         -> COMPUTE INITIAL GUESS FOR PSI_NORM...', end="")
        self.PSI_NORM[:,0] = self.InitialGuess()  
        # ASSIGN VALUES TO EACH ELEMENT
        self.UpdateElementalPSI()
        print('Done!')   
        
        ####### COMPUTE INITIAL VACUUM VESSEL BOUNDARY VALUES PSI_B AND STORE THEM IN ARRAY FOR N=0
        print('         -> COMPUTE INITIAL VACUUM VESSEL BOUNDARY VALUES PSI_B...', end="")
        # COMPUTE INITIAL TOTAL PLASMA CURRENT CORRECTION FACTOR
        self.ComputeTotalPlasmaCurrentNormalization()
        self.PSI_B[:,0] = self.ComputeBoundaryPSI()
        print('Done!')
        
        ####### ASSIGN CONSTRAINT VALUES ON PLASMA BOUNDARY
        print('         -> ASSIGN INITIAL BOUNDARY VALUES...', end="")
        # ASSIGN PLASMA BOUNDARY VALUES
        self.PSI_X = 0   # INITIAL CONSTRAINT VALUE ON SEPARATRIX
        self.UpdatePlasmaBoundaryValues()
        print('Done!')    
        return
    
    
    def Initialization(self):
        """
        Initializes all necessary elements for the simulation:
            - Initializes the level-set function for plasma and vacuum vessel boundaries.
            - Initializes the elements in the computational domain.
            - Classifies elements and writes their classification.
            - Computes the active nodes in the system.
            - Approximates the vacuum vessel first wall and plasma/vacuum interface.
            - Computes numerical integration quadratures for the problem.
            - Initializes PSI unknowns and computes initial guesses.
            - Writes the initial PSI, normalized PSI, and vacuum vessel boundary PSI values.
        """
        
        self.InitialiseParameters()
        
        # INITIALISE LEVEL-SET FUNCTION
        print("     -> INITIALISE LEVEL-SET...", end="")
        self.InitialisePlasmaLevelSet()
        self.writePlasmaBoundaryLS()
        print('Done!')
        
        # INITIALISE ELEMENTS 
        print("     -> INITIALISE ELEMENTS...", end="")
        self.InitialiseElements()
        print('Done!')
        
        # CLASSIFY ELEMENTS   
        print("     -> CLASSIFY ELEMENTS...", end="")
        self.IdentifyNearestNeighbors()
        self.IdentifyVacuumVesselElements()
        self.ClassifyElements()
        self.writeElementsClassification()
        print("Done!")

        # COMPUTE PLASMA BOUNDARY APPROXIMATION
        print("     -> APPROXIMATE PLASMA BOUNDARY INTERFACE...", end="")
        self.ComputePlasmaBoundaryApproximation()
        print("Done!")
        
        # IDENTIFY GHOST FACES 
        if self.GhostPenalty:
            print("     -> IDENTIFY GHOST FACES...", end="")
            self.ComputePlasmaBoundaryGhostFaces()
            print("Done!")
        
        # COMPUTE NUMERICAL INTEGRATION QUADRATURES
        print('     -> COMPUTE NUMERICAL INTEGRATION QUADRATURES...', end="")
        self.ComputeIntegrationQuadratures()
        print('Done!')
        
        # INITIALISE PSI UNKNOWNS
        print("     -> COMPUTE INITIAL GUESS...")
        self.InitialisePSI()
        self.writePSI()
        self.writePSI_NORM()
        self.writePSI_B()
        print('     Done!')
        return  
        
    
    ##################### PLASMA BOUNDARY APPROXIMATION #########################
    
    def ComputePlasmaBoundaryApproximation(self):
        """ 
        Computes the elemental cutting segments conforming to the plasma boundary approximation.
        Computes normal vectors for each segment.

        The function double checks the orthogonality of the normal vectors. 
        """
        for inter, ielem in enumerate(self.PlasmaBoundElems):
            # APPROXIMATE PLASMA/VACUUM INTERACE GEOMETRY CUTTING ELEMENT 
            self.Elements[ielem].InterfaceApproximation(inter)
            # COMPUTE OUTWARDS NORMAL VECTOR
            self.Elements[ielem].InterfaceNormal()
        # CHECK NORMAL VECTORS
        self.CheckNormalVectors(self.INTERFAPPROX)
        return
    
    def ComputePlasmaBoundaryGhostFaces(self):
        # COMPUTE PLASMA BOUNDARY GHOST FACES
        self.PlasmaBoundGhostFaces, self.PlasmaBoundGhostElems = self.IdentifyPlasmaBoundaryGhostFaces()
        # COMPUTE ELEMENTAL GHOST FACES NORMAL VECTORS
        for ielem in self.PlasmaBoundGhostElems:
            self.Elements[ielem].GhostFacesNormals()
        # CHECK NORMAL VECTORS
        self.CheckNormalVectors(self.GHOSTFACES)
        return
    
    def CheckNormalVectors(self,SET):
        """
        Checks the orthogonality of the normal vectors to the segments constituting the set SET.

        This function verifies if the normal vectors at the boundaries (either plasma or vacuum vessel) are orthogonal to 
        the corresponding interface segments. It checks the dot product between the segment direction vector and the 
        normal vector, raising an exception if the dot product is not close to zero (indicating non-orthogonality).
        """
            
        for ielem in self.PlasmaBoundElems:
            if SET == self.INTERFAPPROX:
                SEGMENTS = self.Elements[ielem].InterfApprox.Segments
            elif SET == self.GHOSTFACES:
                SEGMENTS = self.Elements[ielem].GhostFaces
                
            for SEGMENT in SEGMENTS:
                dir = np.array([SEGMENT.Xseg[1,0]-SEGMENT.Xseg[0,0], SEGMENT.Xseg[1,1]-SEGMENT.Xseg[0,1]]) 
                scalarprod = np.dot(dir,SEGMENT.NormalVec)
                if scalarprod > 1e-10: 
                    raise Exception('Dot product equals',scalarprod, 'for mesh element', ielem, ": Normal vector not perpendicular")
        return

    
    ##################### COMPUTE NUMERICAL INTEGRATION QUADRATURES FOR EACH ELEMENT GROUP 
    
    def ComputeIntegrationQuadratures(self):
        """
        Computes the numerical integration quadratures for different types of elements and boundaries.

        The function computes quadrature entities for the following cases:
            1. Standard 2D quadratures for non-cut elements.
            2. Adapted quadratures for cut elements.
            3. Boundary quadratures for elements on the computational domain's boundary (vacuum vessel).
            4. Quadratures for solenoids in the case of a free-boundary plasma problem.
        """
        
        # COMPUTE STANDARD 2D QUADRATURE ENTITIES FOR NON-CUT ELEMENTS 
        for ielem in self.NonCutElems:
            self.Elements[ielem].ComputeStandardQuadrature2D(self.QuadratureOrder)
            
        # COMPUTE ADAPTED QUADRATURE ENTITIES FOR INTERFACE ELEMENTS
        for ielem in self.PlasmaBoundElems:
            self.Elements[ielem].ComputeAdaptedQuadratures(self.QuadratureOrder)
            
        # COMPUTE QUADRATURES FOR GHOST FACES ON PLASMA BOUNDARY ELEMENTS
        if self.GhostPenalty:
            for ielem in self.PlasmaBoundGhostElems:
                self.Elements[ielem].ComputeGhostFacesQuadratures(self.QuadratureOrder)
                
        return
    
    #################### UPDATE EMBEDED METHOD ##############################
    
    def UpdateElements(self):
        """
        If necessary, the level-set function is updated according to the new normalised solution's 0-level contour.
        If the new saddle point is close enough to the old one, the function exits early, assuming the plasma region is already well-defined.
        
        On the contrary, it updates the following:
            1. Plasma boundary level-set function values.
            2. Plasma region classification.
            3. Plasma boundary approximation and normal vectors.
            4. Numerical integration quadratures for the plasma and vacuum elements.
            5. Updates nodes on the plasma boundary approximation.
        """
                
        if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:
            # IN CASE WHERE THE NEW SADDLE POINT (N+1) CORRESPONDS (CLOSE TO) TO THE OLD SADDLE POINT, THEN THAT MEANS THAT THE PLASMA REGION
            # IS ALREADY WELL DEFINED BY THE OLD LEVEL-SET 
            
            if np.linalg.norm(self.Xcrit[1,1,:-1]-self.Xcrit[0,1,:-1]) < 0.5:
                return
            
            else:
                ###### UPDATE PLASMA REGION LEVEL-SET FUNCTION VALUES ACCORDING TO SOLUTION OBTAINED
                # . RECALL THAT PLASMA REGION IS DEFINED BY NEGATIVE VALUES OF LEVEL-SET -> NEED TO INVERT SIGN
                # . CLOSED GEOMETRY DEFINED BY 0-LEVEL CONTOUR BENEATH ACTIVE SADDLE POINT (DIVERTOR REGION) NEEDS TO BE
                #   DISCARTED BECAUSE THE LEVEL-SET DESCRIBES ONLY THE PLASMA REGION GEOMETRY -> NEED TO POST-PROCESS CUTFEM
                #   SOLUTION IN ORDER TO TAKE ITS 0-LEVEL CONTOUR ENCLOSING ONLY THE PLASMA REGION.  
                
                self.PlasmaBoundLevSet = self.ComputePSILevelSet(self.PSI_NORM[:,1])
                
                ###### RECOMPUTE ALL PLASMA BOUNDARY ELEMENTS ATTRIBUTES
                # UPDATE PLASMA REGION LEVEL-SET ELEMENTAL VALUES     
                self.UpdateElementalPlasmaLevSet()
                # CLASSIFY ELEMENTS ACCORDING TO NEW LEVEL-SET
                self.ClassifyElements()
                # RECOMPUTE PLASMA BOUNDARY APPROXIMATION and NORMAL VECTORS
                self.ComputePlasmaBoundaryApproximation()
                # REIDENTIFY PLASMA BOUNDARY GHOST FACES
                if self.GhostPenalty:
                    self.ComputePlasmaBoundaryGhostFaces()
                
                ###### RECOMPUTE NUMERICAL INTEGRATION QUADRATURES
                # COMPUTE STANDARD QUADRATURE ENTITIES FOR NON-CUT ELEMENTS
                for ielem in np.concatenate((self.PlasmaElems, self.VacuumElems), axis = 0):
                    self.Elements[ielem].ComputeStandardQuadrature2D(self.QuadratureOrder)
                # COMPUTE ADAPTED QUADRATURE ENTITIES FOR INTERFACE ELEMENTS
                for ielem in self.PlasmaBoundElems:
                    self.Elements[ielem].ComputeAdaptedQuadratures(self.QuadratureOrder)
                # COMPUTE PLASMA BOUNDARY GHOST FACES QUADRATURES
                if self.GhostPenalty:
                    for ielem in self.PlasmaBoundGhostElems: 
                        self.Elements[ielem].ComputeGhostFacesQuadratures(self.QuadratureOrder)
                    
                # RECOMPUTE NUMBER OF NODES ON PLASMA BOUNDARY APPROXIMATION 
                self.NnPB = self.ComputePlasmaBoundaryNumberNodes()
                    
                return
            
    #################### L2 ERROR COMPUTATION ############################
    
    def ComputeL2error(self):
        """
        Computes the L2 error of the PSI field by integrating the squared difference between the analytical solution and the 
        computed solution over the plasma region.

        Output:
            L2error (float): The computed L2 error value, which measures the difference between the analytical and numerical PSI solutions.
        """
        L2error = 0
        # INTEGRATE OVER PLASMA ELEMENTS
        for elem in self.PlasmaElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[elem]
            # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
            PSIg = ELEMENT.Ng @ ELEMENT.PSIe
            # LOOP OVER GAUSS NODES
            for ig in range(ELEMENT.ng):
                # LOOP OVER ELEMENTAL NODES
                for i in range(ELEMENT.n):
                    L2error += (PSIg[ig]-self.PSIAnalyticalSolution(ELEMENT.Xg[ig,:],self.PLASMA_CURRENT))**2*ELEMENT.Ng[ig,i]*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                    
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for elem in self.PlasmaBoundElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Elements[elem]
            # LOOP OVER SUBELEMENTS
            for SUBELEM in ELEMENT.SubElements:
                # INTEGRATE IN SUBDOMAIN INSIDE PLASMA REGION
                if SUBELEM.Dom < 0:
                    # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PSIg = SUBELEM.Ng @ ELEMENT.PSIe
                    # LOOP OVER GAUSS NODES
                    for ig in range(SUBELEM.ng):
                        # LOOP OVER ELEMENTAL NODES
                        for i in range(SUBELEM.n):
                            L2error += (PSIg[ig]-self.PSIAnalyticalSolution(SUBELEM.Xg[ig,:],self.PLASMA_CURRENT))**2*SUBELEM.Ng[ig,i]*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]                  
        return L2error
    
    
    ##################################################################################################
    ################################ CutFEM GLOBAL SYSTEM ############################################
    ##################################################################################################
    
    def IntegrateGhostPenaltyTerms(self):
            
        for ghostface in self.PlasmaBoundGhostFaces:
            # ISOLATE COMMON CUT EDGE FROM ADJACENT ELEMENTS
            Tface = ghostface[0]
            FACE0 = self.Elements[ghostface[1][0]].GhostFaces[ghostface[1][2]]
            FACE1 = self.Elements[ghostface[2][0]].GhostFaces[ghostface[2][2]]
            # DEFINE ELEMENTAL MATRICES
            LHSe = np.zeros([len(Tface),len(Tface)])
            
            # LOOP OVER GAUSS INTEGRATION NODES
            for ig in range(FACE0.ng):  
                # SHAPE FUNCTIONS GRADIENT IN PHYSICAL SPACE
                n_dot_Ngrad0 = FACE0.NormalVec@np.array([FACE0.dNdxig[ig,:],FACE0.dNdetag[ig,:]])
                n_dot_Ngrad1 = FACE1.NormalVec@np.array([FACE1.dNdxig[ig,:],FACE1.dNdetag[ig,:]])
                R = FACE0.Xg[ig,0]
                if self.PLASMA_CURRENT == self.LINEAR_CURRENT or self.PLASMA_CURRENT == self.NONLINEAR_CURRENT:   # DIMENSIONLESS SOLUTION CASE 
                    n_dot_Ngrad0 *= self.R0
                    n_dot_Ngrad1 *= self.R0
                    R /= self.R0
                # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM
                for i in range(len(Tface)):  # ROWS ELEMENTAL MATRIX
                    for j in range(len(Tface)):  # COLUMNS ELEMENTAL MATRIX
                        # COMPUTE LHS MATRIX TERMS
                        ### GHOST PENALTY TERM  (GRADIENT JUMP) [ jump(nabla(N_i))*jump(nabla(N_j)) *(Jacobiano) ]  
                        LHSe[i,j] += self.zeta*(n_dot_Ngrad1[i]+n_dot_Ngrad0[i])*(n_dot_Ngrad1[j]+n_dot_Ngrad0[j]) * FACE0.detJg[ig] * FACE0.Wg[ig]
                        ### GHOST PENALTY TERM  (SOLUTION JUMP) [ jump(N_i)*jump(N_j)]
                        #LHSe[i,j] += self.zeta*(FACE0.Ng[ig,i]-FACE1.Ng[ig,i])*(FACE0.Ng[ig,j]-FACE1.Ng[ig,j])    
                               
                                                 
            # ASSEMBLE ELEMENTAL CONTRIBUTIONS INTO GLOBAL SYSTEM
            for i in range(len(Tface)):   # ROWS ELEMENTAL MATRIX
                for j in range(len(Tface)):   # COLUMNS ELEMENTAL MATRIX
                    self.LHS[Tface[i],Tface[j]] += LHSe[i,j]                          
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
        self.LHS = lil_matrix((self.Nn,self.Nn))  # FOR SPARSE MATRIX, USE LIST-OF-LIST FORMAT 
        self.RHS = np.zeros([self.Nn,1])
        
        # OPEN ELEMENTAL MATRICES OUTPUT FILE
        if self.ELMAT_output:
            self.ELMAT_file.write('NON_CUT_ELEMENTS\n')
        
        # INTEGRATE OVER THE SURFACE OF ELEMENTS WHICH ARE NOT CUT BY ANY INTERFACE (STANDARD QUADRATURES)
        print("     Integrate over non-cut elements...", end="")
        
        for ielem in self.NonCutElems: 
            # ISOLATE ELEMENT 
            ELEMENT = self.Elements[ielem]  
            # COMPUTE SOURCE TERM (PLASMA CURRENT)  mu0*R*Jphi  IN PLASMA REGION NODES
            SourceTermg = np.zeros([ELEMENT.ng])
            if ELEMENT.Dom < 0:
                # MAP PSI VALUES FROM ELEMENT NODES TO GAUSS NODES
                PSIg = ELEMENT.Ng @ ELEMENT.PSIe
                for ig in range(ELEMENT.ng):
                    SourceTermg[ig] = self.SourceTerm(ELEMENT.Xg[ig,:],PSIg[ig])
                    
            # COMPUTE ELEMENTAL MATRICES
            if self.PLASMA_CURRENT == self.LINEAR_CURRENT or self.PLASMA_CURRENT == self.NONLINEAR_CURRENT:  # DIMENSIONLESS SOLUTION CASE
                LHSe, RHSe = ELEMENT.IntegrateElementalDomainTerms(SourceTermg,self.R0)
            else:
                LHSe, RHSe = ELEMENT.IntegrateElementalDomainTerms(SourceTermg)
            
            if self.ELMAT_output:
                self.ELMAT_file.write("elem {:d} {:d}\n".format(ELEMENT.index,ELEMENT.Dom))
                self.ELMAT_file.write('elmat\n')
                np.savetxt(self.ELMAT_file,LHSe,delimiter=',',fmt='%e')
                self.ELMAT_file.write('elrhs\n')
                np.savetxt(self.ELMAT_file,RHSe,fmt='%e')
            
            # ASSEMBLE ELEMENTAL CONTRIBUTIONS INTO GLOBAL SYSTEM
            for i in range(ELEMENT.n):   # ROWS ELEMENTAL MATRIX
                for j in range(ELEMENT.n):   # COLUMNS ELEMENTAL MATRIX
                    self.LHS[ELEMENT.Te[i],ELEMENT.Te[j]] += LHSe[i,j]
                self.RHS[ELEMENT.Te[i]] += RHSe[i]
                
        print("Done!")
        
        if self.ELMAT_output:
            self.ELMAT_file.write('END_NON_CUT_ELEMENTS\n')
            self.ELMAT_file.write('CUT_ELEMENTS_SURFACE\n')
        
        # INTEGRATE OVER THE SURFACES OF SUBELEMENTS IN ELEMENTS CUT BY INTERFACES (ADAPTED QUADRATURES)
        print("     Integrate over cut-elements subelements...", end="")
        
        for ielem in self.PlasmaBoundElems:
            # ISOLATE ELEMENT 
            ELEMENT = self.Elements[ielem]
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
                        SourceTermg[ig] = self.SourceTerm(SUBELEM.Xg[ig,:],PSIg[ig])
                        
                # COMPUTE ELEMENTAL MATRICES
                if self.PLASMA_CURRENT == self.LINEAR_CURRENT or self.PLASMA_CURRENT == self.NONLINEAR_CURRENT:  # DIMENSIONLESS SOLUTION CASE
                    LHSe, RHSe = SUBELEM.IntegrateElementalDomainTerms(SourceTermg,self.R0)
                else:
                    LHSe, RHSe = SUBELEM.IntegrateElementalDomainTerms(SourceTermg)
                
                if self.ELMAT_output:
                    self.ELMAT_file.write("elem {:d} {:d} subelem {:d} {:d}\n".format(ELEMENT.index,ELEMENT.Dom,SUBELEM.index,SUBELEM.Dom))
                    self.ELMAT_file.write('elmat\n')
                    np.savetxt(self.ELMAT_file,LHSe,delimiter=',',fmt='%e')
                    self.ELMAT_file.write('elrhs\n')
                    np.savetxt(self.ELMAT_file,RHSe,fmt='%e')
                
                # ASSEMBLE ELEMENTAL CONTRIBUTIONS INTO GLOBAL SYSTEM
                for i in range(len(SUBELEM.Te)):   # ROWS ELEMENTAL MATRIX
                    for j in range(len(SUBELEM.Te)):   # COLUMNS ELEMENTAL MATRIX
                        self.LHS[SUBELEM.Te[i],SUBELEM.Te[j]] += LHSe[i,j]
                    self.RHS[SUBELEM.Te[i]] += RHSe[i]
                
        print("Done!")
        
        if self.ELMAT_output:
            self.ELMAT_file.write('END_CUT_ELEMENTS_SURFACE\n')
            self.ELMAT_file.write('CUT_ELEMENTS_INTERFACE\n')
        
        # INTEGRATE OVER THE CUT EDGES IN ELEMENTS CUT BY INTERFACES (ADAPTED QUADRATURES)
        print("     Integrate along cut-elements interface edges...", end="")
        
        for ielem in self.PlasmaBoundElems:
            # ISOLATE ELEMENT 
            ELEMENT = self.Elements[ielem]
            # COMPUTE ELEMENTAL MATRICES
            if self.PLASMA_CURRENT == self.LINEAR_CURRENT or self.PLASMA_CURRENT == self.NONLINEAR_CURRENT:  # DIMENSIONLESS SOLUTION CASE
                LHSe,RHSe = ELEMENT.IntegrateElementalInterfaceTerms(self.beta,self.R0)
            else: 
                LHSe,RHSe = ELEMENT.IntegrateElementalInterfaceTerms(self.beta)
                
            if self.ELMAT_output:
                self.ELMAT_file.write("elem {:d} {:d}\n".format(ELEMENT.index,ELEMENT.Dom))
                self.ELMAT_file.write('elmat\n')
                np.savetxt(self.ELMAT_file,LHSe,delimiter=',',fmt='%e')
                self.ELMAT_file.write('elrhs\n')
                np.savetxt(self.ELMAT_file,RHSe,fmt='%e')
            
            # ASSEMBLE ELEMENTAL CONTRIBUTIONS INTO GLOBAL SYSTEM
            for i in range(len(ELEMENT.Te)):   # ROWS ELEMENTAL MATRIX
                for j in range(len(ELEMENT.Te)):   # COLUMNS ELEMENTAL MATRIX
                    self.LHS[ELEMENT.Te[i],ELEMENT.Te[j]] += LHSe[i,j]
                self.RHS[ELEMENT.Te[i]] += RHSe[i]
        
        print("Done!")
        
        if self.ELMAT_output:
            self.ELMAT_file.write('END_CUT_ELEMENTS_INTERFACE\n')
      
        # INTEGRATE GHOST PENALTY TERM OVER CUT ELEMENTS INTERNAL CUT EDGES
        if self.GhostPenalty:
            print("     Integrate ghost penalty term along cut elements internal ghost faces...", end="")
            self.IntegrateGhostPenaltyTerms()
            print("Done!") 
        
        print("Done!")   
        return
    
    
    def ImposeStrongBC(self):
        """
        Imposes strong boundary conditions on Vacuum Vessel first wall (= computational domain's boundary) by substitution.

        This function modifies the Right-Hand Side (RHS) vector and reduces the Left-Hand Side (LHS) matrix based on 
        the PSI_B boundary conditions. 

        Steps:
            1. Modifies the RHS by subtracting the contribution of the plasma boundary nodes.
            2. Reduces the LHS matrix and RHS vector to exclude the plasma boundary nodes and returns the reduced system.
        """
        
        RHS_temp = self.RHS.copy()
        # 1. STRONGLY IMPOSE BC BY PASSING LHS COLUMNS MULTIPLIES BY PSI_B VALUE TO RHS
        for inode, node in enumerate(self.BoundaryNodes):
            RHS_temp -= self.PSI_B[inode,0]*self.LHS[:,node]
            
        # 2. REDUCE GLOBAL SYSTEM
        self.LHSred = self.LHS[np.ix_(self.DOFNodes,self.DOFNodes)]
        self.RHSred = RHS_temp[self.DOFNodes,:]
        
        # 3. CONVERT LHSred MATRIX TO CSR FORMAT FOR BETTER OPERATIVITY
        self.LHSred = self.LHSred.tocsr()
        return 
    
    
    def SolveSystem(self):
        """
        Solves the reduced linear system of equations to obtain the solution PSI at iteration N+1.
        """
        
        # SOLVE REDUCED SYSTEM 
        self.PSI = np.zeros([self.Nn,1])
        unknowns = spsolve(self.LHSred, self.RHSred)
        # STORE DOF VALUES
        for inode, node in enumerate(self.DOFNodes):
            self.PSI[node] = unknowns[inode]
        # STORE BOUNDARY VALUES
        for inode, node in enumerate(self.BoundaryNodes):
            self.PSI[node] = self.PSI_B[inode,0]
        
        return
    
    
    ##################################################################################################
    ######################################## MAGNETIC FIELD B ########################################
    ##################################################################################################
    
    def Br(self,X):
        """
        Total radial magnetic at point X such that    Br = -1/R dpsi/dZ
        """
        elem = self.SearchElement(X,range(self.Ne))
        return self.Elements[elem].Br(X)
    
    def Bz(self,X):
        """
        Total vertical magnetic at point X such that    Bz = (1/R) dpsi/dR
        """
        elem = self.SearchElement(X,range(self.Ne))
        return self.Elements[elem].Bz(X)
    
    def Bpol(self,X):
        """
        Toroidal magnetic field
        """
        elem = self.SearchElement(X,range(self.Ne))
        Brz = self.Elements[elem].Brz(X)
        return np.sqrt(Brz[0] * Brz[0] + Brz[1] * Brz[1])
    
    def Btor(self,X):
        """
        Toroidal magnetic field
        """
        
        return
    
    def Btot(self,X):
        """
        Total magnetic field
        """
        
        return
    
    
    def Br_field(self):
        """
        Total radial magnetic field such that    Br = (-1/R) dpsi/dZ
        """
        Br = np.zeros([self.Nn])
        counter = np.zeros([self.Nn])
        for ELEM in self.Elements:
            # COMPUTE ELEMENTAL CONTRIBUTIONS 
            Bre = ELEM.Bre()
            # ASSEMBLE ELEMENTAL CONTRIBUTIONS TO TOTAL FIELD
            for inode in range(ELEM.n):
                Br[ELEM.Te[inode]] += Bre[inode]
                counter[ELEM.Te[inode]] += 1
        # COMPUTE MEAN VALUE 
        Br /= counter
        return Br
    
    def Bz_field(self):
        """
        Total vertical magnetic field such that    Bz = (1/R) dpsi/dR
        """
        Bz = np.zeros([self.Nn])
        counter = np.zeros([self.Nn])
        for ELEM in self.Elements:
            # COMPUTE ELEMENTAL CONTRIBUTIONS 
            Bze = ELEM.Bze()
            # ASSEMBLE ELEMENTAL CONTRIBUTIONS TO TOTAL FIELD
            for inode in range(ELEM.n):
                Bz[ELEM.Te[inode]] += Bze[inode]
                counter[ELEM.Te[inode]] += 1
        # COMPUTE MEAN VALUE 
        Bz /= counter
        return Bz
    
    def Brz_field(self):
        """
        Magnetic vector field such that    (Br, Bz) = ((-1/R) dpsi/dZ, (1/R) dpsi/dR)
        """
        Brz = np.zeros([self.Nn,2])
        counter = np.zeros([self.Nn])
        for ELEM in self.Elements:
            # COMPUTE ELEMENTAL CONTRIBUTIONS 
            Brze = ELEM.Brze()
            # ASSEMBLE ELEMENTAL CONTRIBUTIONS TO TOTAL FIELD
            for inode in range(ELEM.n):
                Brz[ELEM.Te[inode],:] += Brze[inode,:]
                counter[ELEM.Te[inode]] += 1
        # COMPUTE MEAN VALUE 
        for i in range(2):
            Brz[:,i] /= counter
        return Brz
    
    def ComputeMagnetsBfield(self,regular_grid=False,**kwargs):
        if regular_grid:
            # Define regular grid
            Nr = 50
            Nz = 70
            grid_r, grid_z= np.meshgrid(np.linspace(kwargs['rmin'], kwargs['rmax'], Nr),np.linspace(kwargs['zmin'], kwargs['zmax'], Nz))
            Br = np.zeros([Nz,Nr])
            Bz = np.zeros([Nz,Nr])
            for ir in range(Nr):
                for iz in range(Nz):
                    # SUM COILS CONTRIBUTIONS
                    for COIL in self.COILS:
                        Br[iz,ir] += COIL.Br(np.array([grid_r[iz,ir],grid_z[iz,ir]]))
                        Bz[iz,ir] += COIL.Bz(np.array([grid_r[iz,ir],grid_z[iz,ir]]))
                    # SUM SOLENOIDS CONTRIBUTIONS
                    for SOLENOID in self.SOLENOIDS:
                        Br[iz,ir] += SOLENOID.Br(np.array([grid_r[iz,ir],grid_z[iz,ir]]))
                        Bz[iz,ir] += SOLENOID.Bz(np.array([grid_r[iz,ir],grid_z[iz,ir]]))
            return grid_r, grid_z, Br, Bz
        else:
            Br = np.zeros([self.Nn])
            Bz = np.zeros([self.Nn])
            for inode in range(self.Nn):
                # SUM COILS CONTRIBUTIONS
                for COIL in self.COILS:
                    Br[inode] += COIL.Br(self.X[inode,:])
                    Br[inode] += COIL.Br(self.X[inode,:])
                # SUM SOLENOIDS CONTRIBUTIONS
                for SOLENOID in self.SOLENOIDS:
                    Br[inode] += SOLENOID.Br(self.X[inode,:])
                    Br[inode] += SOLENOID.Br(self.X[inode,:])
            return Br, Bz
    
    
    def ComputGradientPSIfield(self):
        
        PSIgrad = np.zeros([3*self.Ne,self.dim])
        Xgrad = np.zeros([3*self.Ne,self.dim])
        XIgrad = np.array([[1/3,1/3],
                        [1/3,2/3],
                        [2/3,1/3]])
        k = 0
        for ELEM in self.Elements:
            for XI in XIgrad:
                # COMPUTE ELEMENTAL PHYSICAL POINTS WHERE GRADIENT IS EVALUATED
                X = ELEM.Mapping(XI.reshape([1,2]))
                Xgrad[k,:] = X
                # INTERPOLATE PSI GRADIENT
                PSIgrad[k,:] += ELEM.GRADElementalInterpolationREFERENCE(XI,self.PSI[ELEM.Te])
                k += 1

        # COMPUTE NORM
        modPSIgrad = np.sqrt(PSIgrad[:,0]**2+PSIgrad[:,1]**2)
        return PSIgrad, modPSIgrad
    
    ##################################################################################################
    ############################################# OUTPUT #############################################
    ##################################################################################################
    
    def openOUTPUTfiles(self):
        """
        Open files for selected output. 
        """
        if self.PSI_output:
            self.PSI_file = open(self.outputdir+'/UNKNO.dat', 'w')
            self.PSI_file.write('UNKNOWN_PSIpol_FIELD\n')
            
        if self.PSI_NORM_output:
            self.PSI_NORM_file = open(self.outputdir+'/PSIpol.dat', 'w')
            self.PSI_NORM_file.write('PSIpol_FIELD\n')
        
        if self.PSIcrit_output:
            self.PSIcrit_file = open(self.outputdir+'/PSIcrit.dat', 'w')
            self.PSIcrit_file.write('PSIcrit_VALUES\n')
        
        if self.PSI_B_output:
            self.PSI_B_file = open(self.outputdir+'/PSIpol_B.dat', 'w')
            self.PSI_B_file.write('PSIpol_B_VALUES\n')
        
        if self.RESIDU_output:
            self.RESIDU_file = open(self.outputdir+'/Residu.dat', 'w')
            self.RESIDU_file.write('RESIDU_VALUES\n')
        
        if self.ElementsClassi_output:
            self.ElementsClassi_file = open(self.outputdir+'/MeshElementsClassification.dat', 'w')
            self.ElementsClassi_file.write('MESH_ELEMENTS_CLASSIFICATION\n')
        
        if self.PlasmaLevSetVals_output:
            self.PlasmaLevSetVals_file = open(self.outputdir+'/PlasmaBoundLS.dat', 'w')
            self.PlasmaLevSetVals_file.write('PLASMA_BOUNDARY_LEVEL_SET\n')
        
        if self.ELMAT_output:
            self.ELMAT_file = open(self.outputdir+'/ElementalMatrices.dat', 'w')
            self.ELMAT_file.write('ELEMENTAL_MATRICES_FILE\n')
        
        return
    
    def closeOUTPUTfiles(self):
        """
        Close files for selected output.
        """
        if self.PSI_output:
            self.PSI_file.write('END_UNKNOWN_PSIpol_FIELD')
            self.PSI_file.close()
        
        if self.PSI_NORM_output:
            self.PSI_NORM_file.write('END_PSIpol_FIELD')
            self.PSI_NORM_file.close()
        
        if self.PSIcrit_output:
            self.PSIcrit_file.write('END_PSIcrit_VALUES')
            self.PSIcrit_file.close()
        
        if self.PSI_B_output:
            self.PSI_B_file.write('END_PSIpol_B_VALUES')
            self.PSI_B_file.close()
        
        if self.RESIDU_output:
            self.RESIDU_file.write('END_RESIDU_VALUES')
            self.RESIDU_file.close()
        
        if self.ElementsClassi_output:
            self.ElementsClassi_file.write('END_MESH_ELEMENTS_CLASSIFICATION')
            self.ElementsClassi_file.close()
        
        if self.PlasmaLevSetVals_output:
            self.PlasmaLevSetVals_file.write('END_PLASMA_BOUNDARY_LEVEL_SET')
            self.PlasmaLevSetVals_file.close()
        
        if self.ELMAT_output:
            self.ELMAT_file.write('END_ELEMENTAL_MATRICES_FILE\n')
            self.ELMAT_file.close()
            
        return
    
    def copysimfiles(self):
        """
        Copies the simulation files (DOM.DAT, GEO.DAT, and EQU.DAT) to the output directory for the given case and mesh.

        This function handles the copying of essential simulation data files from the mesh folder and case file location
        to the output directory. The files copied include the mesh domain data (`dom.dat`), geometry data (`geo.dat`), and 
        equilibrium data (`equ.dat`).

        Steps:
            1. Copies the mesh domain file (`dom.dat`) from the mesh folder to the output directory.
            2. Copies the geometry file (`geo.dat`) from the mesh folder to the output directory.
            3. Copies the equilibrium data file (`equ.dat`) from the case file to the output directory.
        """
        
        # COPY DOM.DAT FILE
        MeshDataFile = self.mesh_folder +'/'+ 'TS-FEMCUTFEM-' + self.MESH +'.dom.dat'
        shutil.copy2(MeshDataFile,self.outputdir+'/'+self.CASE+'-'+self.MESH+'.dom.dat')
        # COPY GEO.DAT FILE
        MeshFile = self.mesh_folder +'/'+ 'TS-FEMCUTFEM-' + self.MESH +'.geo.dat'
        shutil.copy2(MeshFile,self.outputdir+'/'+self.CASE+'-'+self.MESH+'.geo.dat')
        # COPY EQU.DAT FILE
        EQUILIDataFile = self.case_file +'.equ.dat'
        shutil.copy2(EQUILIDataFile,self.outputdir+'/'+self.CASE+'-'+self.MESH+'.equ.dat')
        
        return
    
    def writeparams(self):
        """
        Write simulation parameters in output file.
        """
        if self.PARAMS_output:
            self.PARAMS_file = open(self.outputdir+'/PARAMETERS.dat', 'w')
            self.PARAMS_file.write('SIMULATION_PARAMTERS_FILE\n')
            self.PARAMS_file.write('\n')
            
            self.PARAMS_file.write('MESH_PARAMETERS\n')
            self.PARAMS_file.write("    NPOIN = {:d}\n".format(self.Nn))
            self.PARAMS_file.write("    NELEM = {:d}\n".format(self.Ne))
            self.PARAMS_file.write("    ELEM = {:d}\n".format(self.ElTypeALYA))
            self.PARAMS_file.write("    NBOUN = {:d}\n".format(self.Nbound))
            self.PARAMS_file.write('END_MESH_PARAMETERS\n')
            self.PARAMS_file.write('\n')
            
            self.PARAMS_file.write('PROBLEM_TYPE_PARAMETERS\n')
            self.PARAMS_file.write("    PLASMA_BOUNDARY_equ = {:d}\n".format(self.PLASMA_BOUNDARY))
            self.PARAMS_file.write("    PLASMA_CURRENT_equ = {:d}\n".format(self.PLASMA_CURRENT))
            self.PARAMS_file.write("    TOTAL_PLASMA_CURRENT = {:f}\n".format(self.TOTAL_CURRENT))
            self.PARAMS_file.write('END_PROBLEM_TYPE_PARAMETERS\n')
            self.PARAMS_file.write('\n')
            
            self.PARAMS_file.write('VACUUM_VESSEL_FIRST_WALL_GEOMETRY_PARAMETERS\n')
            self.PARAMS_file.write("    R0_equ = {:f}\n".format(self.R0))
            self.PARAMS_file.write("    EPSILON_equ = {:f}\n".format(self.epsilon))
            self.PARAMS_file.write("    KAPPA_equ = {:f}\n".format(self.kappa))
            self.PARAMS_file.write("    DELTA_equ = {:f}\n".format(self.delta))
            self.PARAMS_file.write('END_VACUUM_VESSEL_FIRST_WALL_GEOMETRY_PARAMETERS\n')
            self.PARAMS_file.write('\n')
            
            if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:
                self.PARAMS_file.write('PLASMA_REGION_GEOMETRY_PARAMETERS\n')
                self.PARAMS_file.write("    CONTROL_POINTS_equ = {:d}\n".format(self.CONTROL_POINTS))
                self.PARAMS_file.write("    R_SADDLE_equ = {:f}\n".format(self.R_SADDLE))
                self.PARAMS_file.write("    Z_SADDLE_equ = {:f}\n".format(self.Z_SADDLE))
                self.PARAMS_file.write("    R_RIGHTMOST_equ = {:f}\n".format(self.R_RIGHTMOST))
                self.PARAMS_file.write("    Z_RIGHTMOST_equ = {:f}\n".format(self.Z_RIGHTMOST))
                self.PARAMS_file.write("    R_LEFTMOST_equ = {:f}\n".format(self.R_LEFTMOST))
                self.PARAMS_file.write("    Z_LEFTMOST_equ = {:f}\n".format(self.Z_LEFTMOST))
                self.PARAMS_file.write("    R_TOP_equ = {:f}\n".format(self.R_TOP))
                self.PARAMS_file.write("    Z_TOP_equ = {:f}\n".format(self.Z_TOP))
                self.PARAMS_file.write('END_PLASMA_REGION_GEOMETRY_PARAMETERS\n')
                self.PARAMS_file.write('\n')
            
            if self.PLASMA_CURRENT == self.PROFILES_CURRENT:
                self.PARAMS_file.write('PLASMA_CURRENT_MODEL_PARAMETERS\n')
                self.PARAMS_file.write("    B0_equ = {:f}\n".format(self.B0))
                self.PARAMS_file.write("    q0_equ = {:f}\n".format(self.q0))
                self.PARAMS_file.write("    n_p_equ = {:f}\n".format(self.n_p))
                self.PARAMS_file.write("    g0_equ = {:f}\n".format(self.G0))
                self.PARAMS_file.write("    n_g_equ = {:f}\n".format(self.n_g))
                self.PARAMS_file.write('END_PLASMA_CURRENT_MODEL_PARAMETERS\n')
                self.PARAMS_file.write('\n')
            
            if self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:
                self.PARAMS_file.write('EXTERNAL_COILS_PARAMETERS\n')
                self.PARAMS_file.write("    N_COILS_equ = {:d}\n".format(self.Ncoils))
                for COIL in self.COILS:
                    self.PARAMS_file.write("    Rposi = {:f}\n".format(COIL.X[0]))
                    self.PARAMS_file.write("    Zposi = {:f}\n".format(COIL.X[1]))
                    self.PARAMS_file.write("    Inten = {:f}\n".format(COIL.I))
                    self.PARAMS_file.write('\n')
                self.PARAMS_file.write('END_EXTERNAL_COILS_PARAMETERS\n')
                self.PARAMS_file.write('\n')
                
                self.PARAMS_file.write('EXTERNAL_SOLENOIDS_PARAMETERS\n')
                self.PARAMS_file.write("    N_SOLENOIDS_equ = {:d}\n".format(self.Nsolenoids))
                for SOLENOID in self.SOLENOIDS:
                    self.PARAMS_file.write("    Rlow = {:f}\n".format(SOLENOID.Xe[0,0]))
                    self.PARAMS_file.write("    Zlow = {:f}\n".format(SOLENOID.Xe[0,1]))
                    self.PARAMS_file.write("    Rup = {:f}\n".format(SOLENOID.Xe[1,0]))
                    self.PARAMS_file.write("    Zup = {:f}\n".format(SOLENOID.Xe[1,1]))
                    self.PARAMS_file.write("    Inten = {:f}\n".format(SOLENOID.I))
                    self.PARAMS_file.write("    Nturns = {:f}\n".format(SOLENOID.Nturns))
                    self.PARAMS_file.write('\n')
                self.PARAMS_file.write('END_EXTERNAL_SOLENOIDS_PARAMETERS\n')
                self.PARAMS_file.write('\n')
            
            self.PARAMS_file.write('NUMERICAL_TREATMENT_PARAMETERS\n')
            self.PARAMS_file.write("    QUADRATURE_ORDER_equ = {:d}\n".format(self.QuadratureOrder))
            self.PARAMS_file.write("    MAX_EXT_IT_equ = {:d}\n".format(self.EXT_ITER))
            self.PARAMS_file.write("    EXT_TOL_equ = {:e}\n".format(self.EXT_TOL))
            self.PARAMS_file.write("    MAX_INT_IT_equ = {:d}\n".format(self.INT_ITER))
            self.PARAMS_file.write("    INT_TOL_equ = {:e}\n".format(self.INT_TOL))
            self.PARAMS_file.write("    Beta_equ = {:f}\n".format(self.beta))
            self.PARAMS_file.write('END_NUMERICAL_TREATMENT_PARAMETERS\n')
            self.PARAMS_file.write('\n')
            
            self.PARAMS_file.write('END_SIMULATION_PARAMTERS_FILE\n')
            self.PARAMS_file.close()
        return
    
    def writePSI(self):
        if self.PSI_output:
            self.PSI_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            for inode in range(self.Nn):
                self.PSI_file.write("{:d} {:e}\n".format(inode+1,float(self.PSI[inode])))
            self.PSI_file.write('END_ITERATION\n')
        return
    
    def writePSI_NORM(self):
        if self.PSI_NORM_output:
            self.PSI_NORM_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            for inode in range(self.Nn):
                self.PSI_NORM_file.write("{:d} {:e}\n".format(inode+1,self.PSI_NORM[inode,0]))
            self.PSI_NORM_file.write('END_ITERATION\n')
        return
    
    def writePSI_B(self):
        if self.PSI_B_output:
            self.PSI_B_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            for inode in range(self.Nnbound):
                self.PSI_B_file.write("{:d} {:e}\n".format(inode+1,self.PSI_B[inode,0]))
            self.PSI_B_file.write('END_ITERATION\n')
        return
    
    def writeresidu(self,which_loop):
        if self.RESIDU_output:
            if which_loop == "INTERNAL":
                if self.it_INT == 1:
                    self.RESIDU_file.write("INTERNAL_LOOP_STRUCTURE\n")
                self.RESIDU_file.write("  INTERNAL_ITERATION = {:d} \n".format(self.it_INT))
                self.RESIDU_file.write("      INTERNAL_RESIDU = {:f} \n".format(self.residu_INT))
                
            elif which_loop == "EXTERNAL":
                self.RESIDU_file.write("END_INTERNAL_LOOP_STRUCTURE\n")
                self.RESIDU_file.write("EXTERNAL_ITERATION = {:d} \n".format(self.it_EXT))
                self.RESIDU_file.write("  EXTERNAL_RESIDU = {:f} \n".format(self.residu_EXT))
        
        return
    
    def writePSIcrit(self):
        if self.PSIcrit_output:
            self.PSIcrit_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            self.PSIcrit_file.write("{:f}  {:f}  {:f}  {:f}\n".format(self.Xcrit[1,0,-1],self.Xcrit[1,0,0],self.Xcrit[1,0,1],self.PSI_0[0]))
            self.PSIcrit_file.write("{:f}  {:f}  {:f}  {:f}\n".format(self.Xcrit[1,1,-1],self.Xcrit[1,1,0],self.Xcrit[1,1,1],self.PSI_X[0]))
            self.PSIcrit_file.write('END_ITERATION\n')
        return
    
    def writeElementsClassification(self):
        if self.ElementsClassi_output:
            self.ElementsClassi_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            MeshClassi = self.ObtainClassification()
            for ielem in range(self.Ne):
                self.ElementsClassi_file.write("{:d} {:d}\n".format(ielem+1,MeshClassi[ielem]))
            self.ElementsClassi_file.write('END_ITERATION\n')
        return
    
    def writePlasmaBoundaryLS(self):
        if self.PlasmaLevSetVals_output:
            self.PlasmaLevSetVals_file.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            for inode in range(self.Nn):
                self.PlasmaLevSetVals_file.write("{:d} {:e}\n".format(inode+1,self.PlasmaBoundLevSet[inode]))
            self.PlasmaLevSetVals_file.write('END_ITERATION\n')
        return
    
    def writeconvergedsolution(self):
        if self.L2error_output:
            self.L2error_file = open(self.outputdir+'/PSIconverged.dat', 'w')
            self.L2error_file.write('PSI_CONVERGED_FIELD\n')
            for inode in range(self.Nn):
                self.L2error_file.write("{:d} {:e}\n".format(inode+1,self.PSI_CONV[inode])) 
            self.L2error_file.write('END_PSI_CONVERGED_FIELD\n')
            
            AnaliticalNorm = np.zeros([self.Nn])
            error = np.zeros([self.Nn])
            for inode in range(self.Nn):
                AnaliticalNorm[inode] = self.PSIAnalyticalSolution(self.X[inode,:],self.PLASMA_CURRENT)
                error[inode] = abs(AnaliticalNorm[inode]-self.PSI_CONV[inode])
                if error[inode] == 0:
                    error[inode] = 1e-15
                    
            self.L2error_file.write('ERROR_FIELD\n')
            for inode in range(self.Nn):
                self.L2error_file.write("{:d} {:e}\n".format(inode+1,error[inode])) 
            self.L2error_file.write('END_ERROR_FIELD\n')
            
            self.L2error_file.write("L2ERROR = {:e}".format(self.L2error))
    
            self.L2error_file.close()
        return
    
    
    ##################################################################################################
    ######################################## MAIN ALGORITHM ##########################################
    ##################################################################################################
    
    def EQUILI(self):
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
        
        # READ INPUT FILES
        print("READ INPUT FILES...")
        self.ReadMesh()
        self.ReadFixdata()
        self.ReadEQUILIdata()
        print('Done!')
        
        # OUTPUT RESULTS FOLDER
        print("PREPARE OUTPUT DIRECTORY...",end='')
        # Check if the directory exists
        if not os.path.exists(self.outputdir):
            # Create the directory
            os.makedirs(self.outputdir)
        # COPY SIMULATION FILES
        self.copysimfiles()
        # WRITE SIMULATION PARAMETERS FILE (IF ON)
        self.writeparams() 
        # OPEN OUTPUT FILES
        self.openOUTPUTfiles()    
        print('Done!')
        
        # INITIALIZATION
        print("INITIALIZATION...")
        self.it = 0
        self.it_EXT = 0
        self.it_INT = 0
        self.Initialization()
        print('Done!')

        if self.plotPSI_output:
            self.PlotSolutionPSI()  # PLOT INITIAL SOLUTION

        # START DOBLE LOOP STRUCTURE
        print('START ITERATION...')
        self.converg_EXT = False
        self.it_EXT = 0
        
        #######################################################
        ################## EXTERNAL LOOP ######################
        #######################################################
        while (self.converg_EXT == False and self.it_EXT < self.EXT_ITER):
            self.it_EXT += 1
            self.converg_INT = False
            self.it_INT = 0
            #######################################################
            ################## INTERNAL LOOP ######################
            #######################################################
            while (self.converg_INT == False and self.it_INT < self.INT_ITER):
                self.it_INT += 1
                self.it += 1
                print('OUTER ITERATION = '+str(self.it_EXT)+' , INNER ITERATION = '+str(self.it_INT))
                
                # WRITE ITERATION CONFIGURATION
                self.writePlasmaBoundaryLS()
                self.writeElementsClassification()
                if self.plotElemsClassi_output:
                    self.PlotClassifiedElements(GHOSTFACES=self.GhostPenalty)
                    
                # INNER LOOP ALGORITHM: SOLVING GRAD-SHAFRANOV BVP WITH CutFEM
                self.AssembleGlobalSystem()                 # 0. ASSEMBLE CUTFEM SYSTEM
                self.ImposeStrongBC()
                self.SolveSystem()                          # 1. SOLVE CutFEM SYSTEM  ->> PSI
                self.writePSI()                             #    WRITE SOLUTION 
                self.ComputeCriticalPSI(self.PSI)           # 2. COMPUTE CRITICAL VALUES   PSI_0 AND PSI_X
                self.writePSIcrit()                         #    WRITE CRITICAL POINTS
                self.NormalisePSI()                         # 3. NORMALISE PSI RESPECT TO CRITICAL VALUES  ->> PSI_NORM 
                self.writePSI_NORM()                        #    WRITE NORMALISED SOLUTION
                if self.plotPSI_output:
                    self.PlotSolutionPSI()                  #    PLOT SOLUTION AND NORMALISED SOLUTION
                self.CheckConvergence('PSI_NORM')           # 4. CHECK CONVERGENCE OF PSI_NORM FIELD
                self.writeresidu("INTERNAL")                #    WRITE INTERNAL LOOP RESIDU
                self.UpdateElements()                       # 5. UPDATE MESH ELEMENTS CLASSIFACTION RESPECT TO NEW PLASMA BOUNDARY LEVEL-SET
                self.UpdatePSI('PSI_NORM')                  # 6. UPDATE PSI_NORM VALUES (PSI_NORM[:,0] = PSI_NORM[:,1])
                self.UpdateElementalPSI()                   # 7. UPDATE PSI_NORM VALUES IN CORRESPONDING ELEMENTS (ELEMENT.PSIe = PSI_NORM[ELEMENT.Te,0])
                self.UpdatePlasmaBoundaryValues()           # 8. UPDATE ELEMENTAL CONSTRAINT VALUES PSIgseg FOR PLASMA/VACUUM INTERFACE
                
                #######################################################
                ################ END INTERNAL LOOP ####################
                #######################################################
            
            self.ComputeTotalPlasmaCurrentNormalization()
            print('COMPUTE VACUUM VESSEL FIRST WALL VALUES PSI_B...', end="")
            self.PSI_B[:,1] = self.ComputeBoundaryPSI()     # COMPUTE VACUUM VESSEL FIRST WALL VALUES PSI_B WITH INTERNALLY CONVERGED PSI_NORM
            print('Done!')
            
            self.CheckConvergence('PSI_B')            # CHECK CONVERGENCE OF VACUUM VESSEL FIEST WALL PSI VALUES  (PSI_B)
            self.writeresidu("EXTERNAL")              # WRITE EXTERNAL LOOP RESIDU 
            self.UpdatePSI('PSI_B')                   # UPDATE PSI_NORM AND PSI_B VALUES
            #self.UpdateElementalPSIg(self.VACVESbound)   # UPDATE ELEMENTAL CONSTRAINT VALUES PSIgseg FOR VACUUM VESSEL FIRST WALL INTERFACE 
            
            #######################################################
            ################ END EXTERNAL LOOP ####################
            #######################################################
            
        print('SOLUTION CONVERGED')
        self.PlotSolutionPSI()
        
        if self.PLASMA_BOUNDARY == self.FIXED_BOUNDARY and self.PLASMA_CURRENT != self.PROFILES_CURRENT:
            self.L2error = self.ComputeL2error()
            self.writeconvergedsolution()
        
        self.closeOUTPUTfiles()
        return
    
    ##################################################################################################
    ############################### RENDERING AND REPRESENTATION #####################################
    ##################################################################################################
    
    def PlotFIELD(self,FIELD,plotnodes):
        
        fig, axs = plt.subplots(1, 1, figsize=(5,5))
        axs.set_xlim(self.Rmin,self.Rmax)
        axs.set_ylim(self.Zmin,self.Zmax)
        a = axs.tricontourf(self.X[plotnodes,0],self.X[plotnodes,1], FIELD[plotnodes], levels=30)
        axs.tricontour(self.X[plotnodes,0],self.X[plotnodes,1], FIELD[plotnodes], levels=[0], colors = 'black')
        axs.tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        plt.colorbar(a, ax=axs)
        plt.show()
        
        return

    
    def PlotSolutionPSI(self):
        """ FUNCTION WHICH PLOTS THE FIELD VALUES FOR PSI, OBTAINED FROM SOLVING THE CUTFEM SYSTEM, 
        AND PSI_NORM IF NORMALISED. """
        
        def subplotfield(self,ax,field):
            a = ax.tricontourf(self.X[:,0],self.X[:,1], field, levels=50)
            ax.tricontour(self.X[:,0],self.X[:,1], field, levels=[0], colors = 'black')
            ax.tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
            ax.set_xlim(self.Rmin, self.Rmax)
            ax.set_ylim(self.Zmin, self.Zmax)
            plt.colorbar(a, ax=ax)
            return
        
        if self.PLASMA_CURRENT == self.PROFILES_CURRENT:
            psi_sol = " normalised solution PSI_NORM"
        else:
            psi_sol = " solution PSI"
        
        if self.it == 0:  # INITIAL GUESS PLOT
            fig, axs = plt.subplots(1, 1, figsize=(6,5))
            subplotfield(self,axs,self.PSI_NORM[:,0])
            axs.set_title('Initial PSI guess')
            plt.show(block=False)
            plt.pause(0.8)
            
        elif self.converg_EXT:  # CONVERGED SOLUTION PLOT
            fig, axs = plt.subplots(1, 1, figsize=(6,5))
            subplotfield(self,axs,self.PSI_CONV)
            axs.set_title('Converged'+psi_sol)
            plt.show()
            
        elif self.PLASMA_CURRENT == self.PROFILES_CURRENT:  # ITERATION SOLUTION FOR PROFILES PLASMA CURRENT (PLOT PSI and PSI_NORM)
            fig, axs = plt.subplots(1, 2, figsize=(11,5))
            # LEFT PLOT: PSI at iteration N+1 WITHOUT NORMALISATION (SOLUTION OBTAINED BY SOLVING CUTFEM SYSTEM)
            subplotfield(self,axs[0],self.PSI[:,0])
            axs[0].set_title('Poloidal magnetic flux PSI')
            # RIGHT PLOT: NORMALISED PSI at iteration N+1
            subplotfield(self,axs[1],self.PSI_NORM[:,1])
            axs[1].set_title('Normalised poloidal magnetic flux PSI_NORM')
            axs[1].yaxis.set_visible(False)
            ## PLOT LOCATION OF CRITICAL POINTS
            for i in range(2):
                # LOCAL EXTREMUM
                axs[i].scatter(self.Xcrit[1,0,0],self.Xcrit[1,0,1],marker = 'x',color='red', s = 40, linewidths = 2)
                # SADDLE POINT
                axs[i].scatter(self.Xcrit[1,1,0],self.Xcrit[1,1,1],marker = 'x',color='green', s = 40, linewidths = 2)
            plt.suptitle("Iteration n = "+str(self.it))
            plt.show(block=False)
            plt.pause(0.8)
                
        else:  # ITERATION SOLUTION FOR ANALYTICAL PLASMA CURRENT CASES (PLOT PSI)
            fig, axs = plt.subplots(1, 1, figsize=(6,5))
            subplotfield(self,axs,self.PSI[:,0])
            axs.set_title('Poloidal magnetic flux PSI')
            axs.set_title("Iteration n = "+str(self.it)+ psi_sol)
            plt.show(block=False)
            plt.pause(0.8)
            
        return
    
    
    def PlotMagneticField(self,streamplot=True):
        
        if streamplot:
            R, Z, Br, Bz = self.ComputeMagnetsBfield(regular_grid=True)
            # Poloidal field magnitude
            Bp = np.sqrt(Br**2 + Br**2)
            plt.contourf(R, Z, np.log(Bp), 50)
            plt.streamplot(R, Z, Br, Bz)
            plt.show()
        
        return
    
    def InspectElement(self,element_index,BOUNDARY,PSI,TESSELLATION,GHOSTFACES,NORMALS,QUADRATURE):
        
        ELEM = self.Elements[element_index]
        Xmin = np.min(ELEM.Xe[:,0])-0.1
        Xmax = np.max(ELEM.Xe[:,0])+0.1
        Ymin = np.min(ELEM.Xe[:,1])-0.1
        Ymax = np.max(ELEM.Xe[:,1])+0.1
            
        color = self.ElementColor(ELEM.Dom)
        colorlist = ['#009E73','#D55E00','#CC79A7','#56B4E9']

        fig, axs = plt.subplots(1, 2, figsize=(10,6))
        axs[0].set_xlim(self.Rmin-0.25,self.Rmax+0.25)
        axs[0].set_ylim(self.Zmin-0.25,self.Zmax+0.25)
        if PSI:
            axs[0].tricontourf(self.X[:,0],self.X[:,1], self.PSI_NORM[:,1], levels=30, cmap='plasma')
            axs[0].tricontour(self.X[:,0],self.X[:,1], self.PSI_NORM[:,1], levels=[0], colors = 'black')
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        # PLOT ELEMENT EDGES
        for iedge in range(ELEM.numedges):
            axs[0].plot([ELEM.Xe[iedge,0],ELEM.Xe[int((iedge+1)%ELEM.numedges),0]],[ELEM.Xe[iedge,1],ELEM.Xe[int((iedge+1)%ELEM.numedges),1]], color=color, linewidth=3)

        axs[1].set_xlim(Xmin,Xmax)
        axs[1].set_ylim(Ymin,Ymax)
        axs[1].set_aspect('equal')
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red',linewidths=2)
        # PLOT ELEMENT EDGES
        for iedge in range(ELEM.numedges):
            axs[1].plot([ELEM.Xe[iedge,0],ELEM.Xe[int((iedge+1)%ELEM.numedges),0]],[ELEM.Xe[iedge,1],ELEM.Xe[int((iedge+1)%ELEM.numedges),1]], color=color, linewidth=8)
        for inode in range(ELEM.n):
            if ELEM.LSe[inode] < 0:
                cl = 'blue'
            else:
                cl = 'red'
            axs[1].scatter(ELEM.Xe[inode,0],ELEM.Xe[inode,1],s=120,color=cl,zorder=5)
        if TESSELLATION and (ELEM.Dom == 0 or ELEM.Dom == 2):
            for isub, SUBELEM in enumerate(ELEM.SubElements):
                # PLOT SUBELEMENT EDGES
                for i in range(SUBELEM.numedges):
                    axs[1].plot([SUBELEM.Xe[i,0], SUBELEM.Xe[(i+1)%SUBELEM.numedges,0]], [SUBELEM.Xe[i,1], SUBELEM.Xe[(i+1)%SUBELEM.numedges,1]], color=colorlist[isub], linewidth=3.5)
                axs[1].scatter(SUBELEM.Xe[:,0],SUBELEM.Xe[:,1], marker='o', s=60, color=colorlist[isub], zorder=5)
        if BOUNDARY:
            axs[1].scatter(ELEM.InterfApprox.Xint[:,0],ELEM.InterfApprox.Xint[:,1],marker='o',color='red',s=100, zorder=5)
            for SEGMENT in ELEM.InterfApprox.Segments:
                axs[1].scatter(SEGMENT.Xseg[:,0],SEGMENT.Xseg[:,1],marker='o',color='green',s=30, zorder=5)
        if GHOSTFACES:
            for FACE in ELEM.GhostFaces:
                axs[1].plot(FACE.Xseg[:2,0],FACE.Xseg[:2,1],color=colorlist[-1],linestyle='dashed',linewidth=3)
        if NORMALS:
            if BOUNDARY:
                for SEGMENT in ELEM.InterfApprox.Segments:
                    # PLOT NORMAL VECTORS
                    Xsegmean = np.mean(SEGMENT.Xseg, axis=0)
                    dl = 10
                    axs[1].arrow(Xsegmean[0],Xsegmean[1],SEGMENT.NormalVec[0]/dl,SEGMENT.NormalVec[1]/dl,width=0.01)
            if GHOSTFACES:
                for FACE in ELEM.GhostFaces:
                    # PLOT NORMAL VECTORS
                    Xsegmean = np.mean(FACE.Xseg, axis=0)
                    dl = 10
                    axs[1].arrow(Xsegmean[0],Xsegmean[1],FACE.NormalVec[0]/dl,FACE.NormalVec[1]/dl,width=0.01)
        if QUADRATURE:
            if ELEM.Dom == -1 or ELEM.Dom == 1 or ELEM.Dom == 3:
                # PLOT STANDARD QUADRATURE INTEGRATION POINTS
                axs[1].scatter(ELEM.Xg[:,0],ELEM.Xg[:,1],marker='x',c='black')
            elif ELEM.Dom == 2:
                # PLOT STANDARD QUADRATURE INTEGRATION POINTS 
                axs[1].scatter(ELEM.Xg[:,0],ELEM.Xg[:,1],marker='x',c='black', zorder=5)
                # PLOT VACUUM VESSEL INTEGRATION POINTS ON COMPUTATIONAL BOUNDARY EDGES
                for SEGMENT in ELEM.InterfApprox.Segments:
                    axs[1].scatter(SEGMENT.Xg[:,0],SEGMENT.Xg[:,1],marker='x',color='grey',s=50, zorder = 5)
            else:
                if TESSELLATION:
                    for isub, SUBELEM in enumerate(ELEM.SubElements):
                        # PLOT QUADRATURE SUBELEMENTAL INTEGRATION POINTS
                        axs[1].scatter(SUBELEM.Xg[:,0],SUBELEM.Xg[:,1],marker='x',c=colorlist[isub], zorder=3)
                if BOUNDARY:
                    # PLOT PLASMA BOUNDARY INTEGRATION POINTS
                    for SEGMENT in ELEM.InterfApprox.Segments:
                        axs[1].scatter(SEGMENT.Xg[:,0],SEGMENT.Xg[:,1],marker='x',color='grey',s=50, zorder=5)
                if GHOSTFACES:
                    # PLOT CUT EDGES QUADRATURES 
                    for FACE in ELEM.GhostFaces:
                        axs[1].scatter(FACE.Xg[:,0],FACE.Xg[:,1],marker='x',color='k',s=50, zorder=6)
        return
    
    
    def InspectGhostFace(self,BOUNDARY,index):
        if BOUNDARY == self.PLASMAbound:
            ghostface = self.PlasmaBoundGhostFaces[index]
        elif BOUNDARY == self.VACVESbound:
            ghostface == self.VacVessWallGhostFaces[index]
    
        # ISOLATE ELEMENTS
        ELEMS = [self.Elements[ghostface[1][0]],self.Elements[ghostface[2][0]]]
        FACES = [ELEMS[0].CutEdges[ghostface[1][1]],ELEMS[1].CutEdges[ghostface[2][1]]]
        
        color = self.ElementColor(ELEMS[0].Dom)
        colorlist = ['#009E73','#D55E00','#CC79A7','#56B4E9']
        
        Rmin = min((min(ELEMS[0].Xe[:,0]),min(ELEMS[1].Xe[:,0])))
        Rmax = max((max(ELEMS[0].Xe[:,0]),max(ELEMS[1].Xe[:,0])))
        Zmin = min((min(ELEMS[0].Xe[:,1]),min(ELEMS[1].Xe[:,1])))
        Zmax = max((max(ELEMS[0].Xe[:,1]),max(ELEMS[1].Xe[:,1])))
        
        fig, axs = plt.subplots(1, 1, figsize=(6,6))
        axs.set_xlim(Rmin,Rmax)
        axs.set_ylim(Zmin,Zmax)
        # PLOT ELEMENTAL EDGES:
        for ELEM in ELEMS:
            for iedge in range(ELEM.numedges):
                axs[1].plot([ELEM.Xe[iedge,0],ELEM.Xe[int((iedge+1)%ELEM.numedges),0]],[ELEM.Xe[iedge,1],ELEM.Xe[int((iedge+1)%ELEM.numedges),1]], color=color, linewidth=8)
            for inode in range(ELEM.n):
                if ELEM.LSe[inode] < 0:
                    cl = 'blue'
                else:
                    cl = 'red'
                axs[1].scatter(ELEM.Xe[inode,0],ELEM.Xe[inode,1],s=120,color=cl,zorder=5)
                
        # PLOT CUT EDGES
        for iedge, FACE in enumerate(FACES):
            axs.plot(FACE.Xseg[:2,0],FACE.Xseg[:2,1],color='#D55E00',linestyle='dashed',linewidth=3)
            
        for inode in range(FACES[0].n):
            axs.text(FACES[0].Xseg[inode,0]+0.03,FACES[0].Xseg[inode,1],str(inode),fontsize=12, color=colorlist[0])
        for inode in range(FACES[1].n):
            axs.text(FACES[1].Xseg[inode,0]-0.03,FACES[1].Xseg[inode,1],str(inode),fontsize=12, color=colorlist[1])
            
        for iedge, FACE in enumerate(FACES):
            # PLOT NORMAL VECTORS
            Xsegmean = np.mean(FACE.Xseg, axis=0)
            dl = 10
            axs.arrow(Xsegmean[0],Xsegmean[1],FACE.NormalVec[0]/dl,FACE.NormalVec[1]/dl,width=0.01, color=colorlist[iedge])
            # PLOT CUT EDGES QUADRATURES 
            for FACE in FACES:
                axs.scatter(FACE.Xg[:,0],FACE.Xg[:,1],marker='x',color='k',s=80, zorder=6)
                
        for inode in range(FACES[0].ng):
            axs.text(FACES[0].Xg[inode,0]+0.03,FACES[0].Xg[inode,1],str(inode),fontsize=12, color=colorlist[0])
        for inode in range(FACES[1].ng):
            axs.text(FACES[1].Xg[inode,0]-0.03,FACES[1].Xg[inode,1],str(inode),fontsize=12, color=colorlist[1])
            
        return
    
    
    def PlotLevelSetEvolution(self,Zlow,Rleft):
        
        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        axs[0].set_xlim(self.Rmin,self.Rmax)
        axs[0].set_ylim(self.Zmin,self.Zmax)
        a = axs[0].tricontourf(self.X[:,0],self.X[:,1], self.PSI_NORM[:,1], levels=30)
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PSI_NORM[:,1], levels=[0], colors = 'black')
        plt.colorbar(a, ax=axs[0])

        axs[1].set_xlim(self.Rmin,self.Rmax)
        axs[1].set_ylim(self.Zmin,self.Zmax)
        a = axs[1].tricontourf(self.X[:,0],self.X[:,1], np.sign(self.PlasmaBoundLevSet), levels=30)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PSI_NORM[:,1], levels=[0], colors = 'black',linewidths = 3)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red',linestyles = 'dashed')
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet_ALL[:,self.it-1], levels=[0], colors = 'orange',linestyles = 'dashed')
        axs[1].plot([self.Rmin,self.Rmax],[Zlow,Zlow],color = 'green')
        axs[1].plot([Rleft,Rleft],[self.Zmin,self.Zmax],color = 'green')

        plt.show()
        
        return
    
    
    def PlotMesh(self):
        plt.figure(figsize=(7,10))
        plt.ylim(np.min(self.X[:,1]),np.max(self.X[:,1]))
        plt.xlim(np.min(self.X[:,0]),np.max(self.X[:,0]))
        # Plot nodes
        plt.plot(self.X[:,0],self.X[:,1],'.')
        # Plot element edges
        for e in range(self.Ne):
            for i in range(self.numedges):
                plt.plot([self.X[self.T[e,i],0], self.X[self.T[e,int((i+1)%self.n)],0]], 
                        [self.X[self.T[e,i],1], self.X[self.T[e,int((i+1)%self.n)],1]], color='black', linewidth=1)
        plt.show()
        return
    
    def PlotClassifiedElements(self,GHOSTFACES,**kwargs):
        plt.figure(figsize=(5,6))
        if not kwargs:
            plt.ylim(self.Zmin-0.25,self.Zmax+0.25)
            plt.xlim(self.Rmin-0.25,self.Rmax+0.25)
        else: 
            plt.ylim(kwargs['zmin'],kwargs['zmax'])
            plt.xlim(kwargs['rmin'],kwargs['rmax'])
        
        # PLOT PLASMA REGION ELEMENTS
        for elem in self.PlasmaElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numedges+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numedges,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'red')
        # PLOT VACCUM ELEMENTS
        for elem in self.VacuumElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numedges+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numedges,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'gray')
        # PLOT PLASMA BOUNDARY ELEMENTS
        for elem in self.PlasmaBoundElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numedges+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numedges,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'gold')
        # PLOT VACUUM VESSEL FIRST WALL ELEMENTS
        for elem in self.VacVessWallElems:
            ELEMENT = self.Elements[elem]
            Xe = np.zeros([ELEMENT.numedges+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.numedges,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'cyan')
             
        # PLOT PLASMA BOUNDARY  
        plt.tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors='green',linewidths=3)
                
        # PLOT GHOSTFACES 
        if GHOSTFACES:
            for ghostface in self.PlasmaBoundGhostFaces:
                plt.plot(self.X[ghostface[0][:2],0],self.X[ghostface[0][:2],1],linewidth=2,color='#56B4E9')
            
        #colorlist = ['#009E73','#D55E00','#CC79A7','#56B4E9']
        plt.show()
        return
        
    
    def PlotNormalVectors(self):
        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        
        axs[0].set_xlim(self.Rmin-0.5,self.Rmax+0.5)
        axs[0].set_ylim(self.Zmin-0.5,self.Zmax+0.5)
        axs[1].set_xlim(6,6.4)
        if self.PLASMA_BOUNDARY == self.FIXED_BOUNDARY:
            axs[1].set_ylim(2.5,3.5)
        elif self.PLASMA_BOUNDARY == self.FREE_BOUNDARY:
            axs[1].set_ylim(2.2,2.6)

        for i in range(2):
            # PLOT PLASMA/VACUUM INTERFACE
            axs[i].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors='green',linewidths=6)
            # PLOT NORMAL VECTORS
            for ielem in self.PlasmaBoundElems:
                ELEMENT = self.Elements[ielem]
                if i == 0:
                    dl = 5
                else:
                    dl = 10
                for j in range(ELEMENT.n):
                    plt.plot([ELEMENT.Xe[j,0], ELEMENT.Xe[int((j+1)%ELEMENT.n),0]], 
                            [ELEMENT.Xe[j,1], ELEMENT.Xe[int((j+1)%ELEMENT.n),1]], color='k', linewidth=1)
                for SEGMENT in ELEMENT.InterfApprox.Segments:
                    # PLOT INTERFACE APPROXIMATIONS
                    axs[0].plot(SEGMENT.Xseg[:,0],SEGMENT.Xseg[:,1], linestyle='-', color = 'red', linewidth = 2)
                    axs[1].plot(SEGMENT.Xseg[:,0],SEGMENT.Xseg[:,1], linestyle='-', marker='o',color = 'red', linewidth = 2)
                    # PLOT NORMAL VECTORS
                    Xsegmean = np.mean(SEGMENT.Xseg, axis=0)
                    axs[i].arrow(Xsegmean[0],Xsegmean[1],SEGMENT.NormalVec[0]/dl,SEGMENT.NormalVec[1]/dl,width=0.01)
                
        axs[1].set_aspect('equal')
        plt.show()
        return
    
    
    def PlotInterfaceValues(self):
        """ Function which plots the values PSIgseg at the interface edges, for both the plasma/vacuum interface and the vacuum vessel first wall. """

        # IMPOSED BOUNDARY VALUES
        ### VACUUM VESSEL FIRST WALL
        PSI_Bg = self.PSI_B[:,1]
        PSI_B = self.PSI[self.BoundaryNodes]
            
        ### PLASMA BOUNDARY
        X_Dg = np.zeros([self.NnPB,self.dim])
        PSI_Dg = np.zeros([self.NnPB])
        PSI_D = np.zeros([self.NnPB])
        k = 0
        for ielem in self.PlasmaBoundElems:
            ELEMENT = self.Elements[ielem]
            for SEGMENT in ELEMENT.InterfApprox.Segments:
                for inode in range(SEGMENT.ng):
                    X_Dg[k,:] = SEGMENT.Xg[inode,:]
                    PSI_Dg[k] = SEGMENT.PSIgseg[inode]
                    PSI_D[k] = ELEMENT.ElementalInterpolationPHYSICAL(X_Dg[k,:],self.PSI[ELEMENT.Te])
                    k += 1
            
        fig, axs = plt.subplots(1, 2, figsize=(14,7))
        ### UPPER ROW SUBPLOTS 
        # LEFT SUBPLOT: CONSTRAINT VALUES ON PSI
        axs[0].set_aspect('equal')
        axs[0].set_ylim(self.Zmin-0.5,self.Zmax+0.5)
        axs[0].set_xlim(self.Rmin-0.5,self.Rmax+0.5)
        cmap = plt.get_cmap('jet')
        
        norm = plt.Normalize(np.min([PSI_Bg.min(),PSI_Dg.min()]),np.max([PSI_Bg.max(),PSI_Dg.max()]))
        linecolors_Dg = cmap(norm(PSI_Dg))
        linecolors_Bg = cmap(norm(PSI_Bg))
        axs[0].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_Dg)
        axs[0].scatter(self.X[self.BoundaryNodes,0],self.X[self.BoundaryNodes,1],color = linecolors_Bg)

        # RIGHT SUBPLOT: RESULTING VALUES ON CUTFEM SYSTEM 
        axs[1].set_aspect('equal')
        axs[1].set_ylim(self.Zmin-0.5,self.Zmax+0.5)
        axs[1].set_xlim(self.Rmin-0.5,self.Rmax+0.5)
        linecolors_D = cmap(norm(PSI_D))
        linecolors_B = cmap(norm(PSI_B))
        axs[1].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_D)
        axs[1].scatter(self.X[self.BoundaryNodes,0],self.X[self.BoundaryNodes,1],color = linecolors_B)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[1])

        plt.show()
        return
    
    
    def PlotPlasmaBoundaryConstraints(self):
        
        # COLLECT PSIgseg DATA ON PLASMA/VACUUM INTERFACE
        X_Dg = np.zeros([len(self.PlasmaBoundElems)*self.Ng1D,self.dim])
        PSI_Dexact = np.zeros([len(self.PlasmaBoundElems)*self.Ng1D])
        PSI_Dg = np.zeros([len(self.PlasmaBoundElems)*self.Ng1D])
        X_D = np.zeros([len(self.PlasmaBoundElems)*self.n,self.dim])
        PSI_D = np.zeros([len(self.PlasmaBoundElems)*self.n])
        error = np.zeros([len(self.PlasmaBoundElems)*self.n])
        k = 0
        l = 0
        for ielem in self.PlasmaBoundElems:
            for SEGMENT in self.Elements[ielem].InterfApprox.Segments:
                for inode in range(SEGMENT.ng):
                    X_Dg[k,:] = SEGMENT.Xg[inode,:]
                    if self.PLASMA_CURRENT != self.PROFILES_CURRENT:
                        PSI_Dexact[k] = self.PSIAnalyticalSolution(X_Dg[k,:],self.PLASMA_CURRENT)
                    else:
                        PSI_Dexact[k] = SEGMENT.PSIgseg[inode]
                    PSI_Dg[k] = SEGMENT.PSIgseg[inode]
                    k += 1
            for jnode in range(self.Elements[ielem].n):
                X_D[l,:] = self.Elements[ielem].Xe[jnode,:]
                PSI_Dexact_node = self.PSIAnalyticalSolution(X_D[l,:],self.PLASMA_CURRENT)
                PSI_D[l] = self.PSI[self.Elements[ielem].Te[jnode]]
                error[l] = np.abs(PSI_D[l]-PSI_Dexact_node)
                l += 1
            
        fig, axs = plt.subplots(1, 4, figsize=(18,6)) 
        # LEFT SUBPLOT: ANALYTICAL VALUES
        axs[0].set_aspect('equal')
        axs[0].set_ylim(self.Zmin-0.5,self.Zmax+0.5)
        axs[0].set_xlim(self.Rmin-0.5,self.Rmax+0.5)
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(PSI_Dexact.min(),PSI_Dexact.max())
        linecolors_Dexact = cmap(norm(PSI_Dexact))
        axs[0].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_Dexact)
        
        # CENTER SUBPLOT: CONSTRAINT VALUES ON PSI
        axs[1].set_aspect('equal')
        axs[1].set_ylim(self.Zmin-0.5,self.Zmax+0.5)
        axs[1].set_xlim(self.Rmin-0.5,self.Rmax+0.5)
        #norm = plt.Normalize(PSI_Dg.min(),PSI_Dg.max())
        linecolors_Dg = cmap(norm(PSI_Dg))
        axs[1].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_Dg)

        # RIGHT SUBPLOT: RESULTING VALUES ON CUTFEM SYSTEM 
        axs[2].set_aspect('equal')
        axs[2].set_ylim(self.Zmin-0.5,self.Zmax+0.5)
        axs[2].set_xlim(self.Rmin-0.5,self.Rmax+0.5)
        linecolors_D = cmap(norm(PSI_D))
        axs[2].scatter(X_D[:,0],X_D[:,1],color = linecolors_D)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[2])
        
        axs[3].set_aspect('equal')
        axs[3].set_ylim(self.Zmin-0.5,self.Zmax+0.5)
        axs[3].set_xlim(self.Rmin-0.5,self.Rmax+0.5)
        norm = plt.Normalize(np.log(error).min(),np.log(error).max())
        linecolors_error = cmap(norm(np.log(error)))
        axs[3].scatter(X_D[:,0],X_D[:,1],color = linecolors_error)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[3])

        plt.show()
        
        return 
    
    
    def PlotIntegrationQuadratures(self):
        
        plt.figure(figsize=(9,11))
        plt.ylim(self.Zmin-0.25,self.Zmax+0.25)
        plt.xlim(self.Rmin-0.25,self.Rmax+0.25)

        # PLOT NODES
        plt.plot(self.X[:,0],self.X[:,1],'.',color='black')
        Tmesh = self.T +1
        # PLOT PLASMA REGION ELEMENTS
        for elem in self.PlasmaElems:
            ELEMENT = self.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='red', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='red')
        # PLOT VACCUM ELEMENTS
        for elem in self.VacuumElems:
            ELEMENT = self.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='gray', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='gray')
        # PLOT EXTERIOR ELEMENTS IF EXISTING
        for elem in self.ExteriorElems:
            ELEMENT = self.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='black', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='black')
            
        # PLOT PLASMA BOUNDARY ELEMENTS
        for elem in self.PlasmaBoundElems:
            ELEMENT = self.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='gold', linewidth=1)
            # PLOT SUBELEMENT EDGES AND INTEGRATION POINTS
            for SUBELEM in ELEMENT.SubElements:
                # PLOT SUBELEMENT EDGES
                for i in range(self.n):
                    plt.plot([SUBELEM.Xe[i,0], SUBELEM.Xe[(i+1)%SUBELEM.n,0]], [SUBELEM.Xe[i,1], SUBELEM.Xe[(i+1)%SUBELEM.n,1]], color='gold', linewidth=1)
                # PLOT QUADRATURE INTEGRATION POINTS
                plt.scatter(SUBELEM.Xg[:,0],SUBELEM.Xg[:,1],marker='x',c='gold')
            # PLOT INTERFACE  APPROXIMATION AND INTEGRATION POINTS
            for SEGMENT in ELEMENT.InterfApprox.Segments:
                # PLOT INTERFACE APPROXIMATION
                plt.plot(SEGMENT.Xseg[:,0], SEGMENT.Xseg[:,1], color='green', linewidth=1)
                # PLOT INTERFACE QUADRATURE
                plt.scatter(SEGMENT.Xg[:,0],SEGMENT.Xg[:,1],marker='o',c='green')
                
        # PLOT VACUUM VESSEL FIRST WALL ELEMENTS
        for elem in self.VacVessWallElems:
            ELEMENT = self.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='darkturquoise', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='darkturquoise')

        plt.show()
        return

    
    def PlotError(self,plotnodes):
        
        AnaliticalNorm = np.zeros([self.Nn])
        for inode in range(self.Nn):
            AnaliticalNorm[inode] = self.PSIAnalyticalSolution(self.X[inode,:],self.PLASMA_CURRENT)
           
        error = np.zeros([len(self.PlasmaNodes)])
        for i, inode in enumerate(self.PlasmaNodes):
            error[i] = abs(AnaliticalNorm[inode]-self.PSI_CONV[inode])
            if error[i] == 0:
                error[i] = 1e-15
            
        print(np.linalg.norm(error))
            
        fig, axs = plt.subplots(1, 3, figsize=(16,5))
        axs[0].set_xlim(self.Rmin,self.Rmax)
        axs[0].set_ylim(self.Zmin,self.Zmax)
        a = axs[0].tricontourf(self.X[plotnodes,0],self.X[plotnodes,1], AnaliticalNorm[plotnodes], levels=30)
        axs[0].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs[0].tricontour(self.X[plotnodes,0],self.X[plotnodes,1], AnaliticalNorm[plotnodes], levels=[0], colors = 'black')
        plt.colorbar(a, ax=axs[0])

        axs[1].set_xlim(self.Rmin,self.Rmax)
        axs[1].set_ylim(self.Zmin,self.Zmax)
        a = axs[1].tricontourf(self.X[plotnodes,0],self.X[plotnodes,1], self.PSI_CONV[plotnodes], levels=30)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PSI_CONV, levels=[0], colors = 'black')
        plt.colorbar(a, ax=axs[1])

        axs[2].set_xlim(self.Rmin,self.Rmax)
        axs[2].set_ylim(self.Zmin,self.Zmax)
        a = axs[2].tricontourf(self.X[self.PlasmaNodes,0],self.X[self.PlasmaNodes,1], np.log(error), levels=30)
        axs[2].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red')
        plt.colorbar(a, ax=axs[2])

        plt.show()
        
        return
    
    @staticmethod
    def ElementColor(dom):
        if dom == -1:
            color = 'red'
        elif dom == 0:
            color = 'gold'
        elif dom == 1:
            color = 'grey'
        elif dom == 2:
            color = 'cyan'
        elif dom == 3:
            color = 'black'
        return color
    
    def PlotREFERENCE_PHYSICALelement(self,element_index,TESSELLATION,BOUNDARY,NORMALS,QUADRATURE):
        ELEM = self.Elements[element_index]
        Xmin = np.min(ELEM.Xe[:,0])-0.1
        Xmax = np.max(ELEM.Xe[:,0])+0.1
        Ymin = np.min(ELEM.Xe[:,1])-0.1
        Ymax = np.max(ELEM.Xe[:,1])+0.1
        if ELEM.ElType == 1:
            numedges = 3
        elif ELEM.ElType == 2:
            numedges = 4
            
        color = self.ElementColor(ELEM.Dom)
        colorlist = ['#009E73','#D55E00','#CC79A7','#56B4E9']

        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        XIe = ReferenceElementCoordinates(ELEM.ElType,ELEM.ElOrder)
        XImin = np.min(XIe[:,0])-0.4
        XImax = np.max(XIe[:,0])+0.25
        ETAmin = np.min(XIe[:,1])-0.4
        ETAmax = np.max(XIe[:,1])+0.25
        axs[0].set_xlim(XImin,XImax)
        axs[0].set_ylim(ETAmin,ETAmax)
        axs[0].tricontour(XIe[:,0],XIe[:,1], ELEM.LSe, levels=[0], colors = 'red',linewidths=2)
        # PLOT ELEMENT EDGES
        for iedge in range(ELEM.numedges):
            axs[0].plot([XIe[iedge,0],XIe[int((iedge+1)%ELEM.numedges),0]],[XIe[iedge,1],XIe[int((iedge+1)%ELEM.numedges),1]], color=color, linewidth=8)
        for inode in range(ELEM.n):
            if ELEM.LSe[inode] < 0:
                cl = 'blue'
            else:
                cl = 'red'
            axs[0].scatter(XIe[inode,0],XIe[inode,1],s=120,color=cl,zorder=5)

        if TESSELLATION and (ELEM.Dom == 0 or ELEM.Dom == 2):
            for isub, SUBELEM in enumerate(ELEM.SubElements):
                # PLOT SUBELEMENT EDGES
                for i in range(SUBELEM.numedges):
                    axs[0].plot([SUBELEM.XIe[i,0], SUBELEM.XIe[(i+1)%SUBELEM.numedges,0]], [SUBELEM.XIe[i,1], SUBELEM.XIe[(i+1)%SUBELEM.numedges,1]], color=colorlist[isub], linewidth=3.5)
                axs[0].scatter(SUBELEM.XIe[:,0],SUBELEM.XIe[:,1], marker='o', s=60, color=colorlist[isub], zorder=5)
        if BOUNDARY:
            axs[0].scatter(ELEM.InterfApprox.XIint[:,0],ELEM.InterfApprox.XIint[:,1],marker='o',color='red',s=100, zorder=5)
            for SEGMENT in ELEM.InterfApprox.Segments:
                axs[0].scatter(SEGMENT.XIseg[:,0],SEGMENT.XIseg[:,1],marker='o',color='green',s=30, zorder=5)
        if NORMALS:
            for SEGMENT in ELEM.InterfApprox.Segments:
                # PLOT NORMAL VECTORS
                Xsegmean = np.mean(SEGMENT.Xseg, axis=0)
                dl = 10
                #axs[0].arrow(Xsegmean[0],Xsegmean[1],SEGMENT.NormalVec[0]/dl,SEGMENT.NormalVec[1]/dl,width=0.01)
        if QUADRATURE:
            if ELEM.Dom == -1 or ELEM.Dom == 1 or ELEM.Dom == 3:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[0].scatter(ELEM.XIg[:,0],ELEM.XIg[:,1],marker='x',c='black')
            elif ELEM.Dom == 2:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[0].scatter(ELEM.XIg[:,0],ELEM.XIg[:,1],marker='x',c='black', zorder=5)
                # PLOT INTERFACE INTEGRATION POINTS
                for SEGMENT in ELEM.InterfApprox.Segments:
                    axs[0].scatter(SEGMENT.XIg[:,0],SEGMENT.XIg[:,1],marker='x',color='grey',s=50, zorder = 5)
                        
                        
        axs[1].set_xlim(Xmin,Xmax)
        axs[1].set_ylim(Ymin,Ymax)
        axs[1].tricontour(self.X[:,0],self.X[:,1], self.PlasmaBoundLevSet, levels=[0], colors = 'red',linewidths=2)
        # PLOT ELEMENT EDGES
        for iedge in range(ELEM.numedges):
            axs[1].plot([ELEM.Xe[iedge,0],ELEM.Xe[int((iedge+1)%ELEM.numedges),0]],[ELEM.Xe[iedge,1],ELEM.Xe[int((iedge+1)%ELEM.numedges),1]], color=color, linewidth=8)
        for inode in range(ELEM.n):
            if ELEM.LSe[inode] < 0:
                cl = 'blue'
            else:
                cl = 'red'
            axs[1].scatter(ELEM.Xe[inode,0],ELEM.Xe[inode,1],s=120,color=cl,zorder=5)
        if TESSELLATION and (ELEM.Dom == 0 or ELEM.Dom == 2):
            for isub, SUBELEM in enumerate(ELEM.SubElements):
                # PLOT SUBELEMENT EDGES
                for i in range(SUBELEM.numedges):
                    axs[1].plot([SUBELEM.Xe[i,0], SUBELEM.Xe[(i+1)%SUBELEM.numedges,0]], [SUBELEM.Xe[i,1], SUBELEM.Xe[(i+1)%SUBELEM.numedges,1]], color=colorlist[isub], linewidth=3.5)
                axs[1].scatter(SUBELEM.Xe[:,0],SUBELEM.Xe[:,1], marker='o', s=60, color=colorlist[isub], zorder=5)
        if BOUNDARY:
            axs[1].scatter(ELEM.InterfApprox.Xint[:,0],ELEM.InterfApprox.Xint[:,1],marker='o',color='red',s=100, zorder=5)
            for SEGMENT in ELEM.InterfApprox.Segments:
                axs[1].scatter(SEGMENT.Xseg[:,0],SEGMENT.Xseg[:,1],marker='o',color='green',s=30, zorder=5)
        if NORMALS:
            for SEGMENT in ELEM.InterfApprox.Segments:
                # PLOT NORMAL VECTORS
                Xsegmean = np.mean(SEGMENT.Xseg, axis=0)
                dl = 10
                axs[1].arrow(Xsegmean[0],Xsegmean[1],SEGMENT.NormalVec[0]/dl,SEGMENT.NormalVec[1]/dl,width=0.01)
        if QUADRATURE:
            if ELEM.Dom == -1 or ELEM.Dom == 1 or ELEM.Dom == 3:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[1].scatter(ELEM.Xg[:,0],ELEM.Xg[:,1],marker='x',c='black')
            elif ELEM.Dom == 2:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[1].scatter(ELEM.Xg[:,0],ELEM.Xg[:,1],marker='x',c='black', zorder=5)
                # PLOT INTERFACE INTEGRATION POINTS
                for SEGMENT in ELEM.InterfApprox.Segments:
                    axs[1].scatter(SEGMENT.Xg[:,0],SEGMENT.Xg[:,1],marker='x',color='grey',s=50, zorder = 5)
                
        return
    