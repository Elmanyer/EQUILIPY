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


import numpy as np
from math import ceil
from Element import *
from Mesh import *
from Magnet import *

class EquilipyInitialisation:
    
    def InitialiseParameters(self):
        """
        Initialize simulation parameters based on current object settings.

        - Disables critical point output if fixed-boundary problem is set.
        - Disables ghost face output if ghost stabilization is turned off.
        - Computes default 1D quadrature order if not already specified,
        based on 2D quadrature order.
        """
        
        print("INITIALISE SIMULATION PARAMETERS...", end="")
        # OVERRIDE CRITICAL POINT OPTIMIZATION OUTPUT WHEN FIXED-BOUNDARY PROBLEM
        if self.FIXED_BOUNDARY:
            self.out_PSIcrit = False
        else:
            self.out_PSIcrit = True
        
        # OVERRIDE GHOST FACES OUTPUT WHEN GHOST STABILIZATION IS OFF
        if not self.GhostStabilization:
            self.out_ghostfaces = False
            
        # COMPUTE 1D NUMERICAL QUADRATURE ORDER (IF NOT SPECIFIED)
        if type(self.QuadratureOrder1D) == type(None):
            self.QuadratureOrder1D = ceil(0.5*(self.QuadratureOrder2D+1))
        
        print('Done!')
        return
    
    
    def InitialisePickleLists(self):
        """
        Initialize lists to store simulation data for pickling.
        """
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
        Initialises the solver's plasma boundary level-set function values. Negative values represent inside the plasma region.
        """ 
        self.PlasmaLS = self.initialPHI.PHI0
        return 
    
    
    def InitialisePSI(self):  
        """
        Initialize PSI vectors and compute the initial guess for the simulation.

        Tasks:
            - Initializes PSI solution arrays.
            - Assigns initial guess to PSI arrays.
            - Assigns initial PSI to the corresponding elements.
            - Assigns plasma boundary constraint values.
        """
        print('INITIALISE PSI...')
        
        ####### INITIALISE PSI VECTORS
        print('     -> INITIALISE PSI ARRAYS...', end="")
        # INITIALISE ITERATIVE UPDATED ARRAYS
        self.PSI = np.zeros([self.MESH.Nn],dtype=float)            # SOLUTION FROM SOLVING CutFEM SYSTEM OF EQUATIONS (INTERNAL LOOP)       
        self.PSI_NORM = np.zeros([self.MESH.Nn,2],dtype=float)     # NORMALISED PSI SOLUTION FIELD (INTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)
        self.PSI_CONV = np.zeros([self.MESH.Nn],dtype=float)       # CONVERGED SOLUTION FIELD
        print('Done!')
        
        ####### COMPUTE INITIAL GUESS AND STORE IT IN ARRAY FOR N=0
        # COMPUTE INITIAL GUESS
        print('     -> COMPUTE INITIAL GUESS FOR PSI_NORM...', end="")
        self.PSI_NORM[:,0] = self.initialPSI.PSI0
        self.PSI_NORM[:,1] = self.PSI_NORM[:,0]
        # INITIALISE CRITICAL POINTS ARRAY
        self.Xcrit = np.zeros([2,2,3])  # [(iterations n, n+1), (extremum, saddle point), (R_crit,Z_crit,elem_crit)]
        if not self.initialPSI.Opoint0:
            self.Xcrit[0,0,:-1] = np.zeros([2])
        else:
            self.Xcrit[0,0,:-1] = self.initialPSI.Opoint0[0][0]
        if not self.initialPSI.Xpoint0:
            self.Xcrit[0,1,:-1] = np.zeros([2])
        else:
            self.Xcrit[0,1,:-1] = self.initialPSI.Xpoint0[0][0]
        self.PSI_0 = self.initialPSI.PSI0_0
        self.PSI_X = self.initialPSI.PSI0_X 
        # ASSIGN VALUES TO EACH ELEMENT
        self.UpdateElementalPSI()
        print('Done!')  
        
        print('Done!') 
        return
    
        
    def InitialisePSI_B(self):
        """
        Initialize and compute the PSI_B vector for computational domain's boundary PSI values.

        Tasks:
            - Initializes PSI_B array.
            - Computes initial computational domain's boundary PSI values and assigns them to the PSI_B array.
            - Updates the computational domain's boundary values based on the computed PSI_B.
        """
        
        print('INITIALISE PSI_B...')
        
        ####### INITIALISE PSI BOUNDARY VECTOR
        self.PSI_B = np.zeros([self.MESH.Nnbound,2],dtype=float)   # COMPUTATIONAL DOMAIN BOUNDARY PSI VALUES (EXTERNAL LOOP) AT ITERATIONS N AND N+1 (COLUMN 0 -> ITERATION N ; COLUMN 1 -> ITERATION N+1)    
        
        ####### COMPUTE INITIAL COMPUTATIONAL DOMAIN BOUNDARY VALUES PSI_B AND STORE THEM IN ARRAY FOR N=0
        print('     -> COMPUTE INITIAL COMPUTATIONAL DOMAIN BOUNDARY VALUES PSI_B...', end="")
        # COMPUTE INITIAL TOTAL PLASMA CURRENT CORRECTION FACTOR
        #self.ComputeTotalPlasmaCurrentNormalization()
        self.PSI_B[:,0] = self.ComputeBoundaryPSI()
        self.PSI_B[:,1] = self.PSI_B[:,0]
        print('Done!')
        
        print('     -> ASSIGN INITIAL COMPUTATIONAL DOMAIN BOUNDARY VALUES...', end="")
        self.UpdateElementalPSI_B()
        print('Done!')
        
        print('Done!')  
        return
    
    
    