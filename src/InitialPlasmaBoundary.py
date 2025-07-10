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



from weakref import proxy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from AnalyticalSolutions import *
from functools import partial
import _plot as eqplot

class InitialPlasmaBoundary:
    
    """
    Represents the initial plasma boundary configuration using a level-set function
    defined over the computational and extended domains.

    Purpose:
        - Constructs and evaluates an analytical or parametrized level-set function
          describing the initial guess for the plasma boundary shape.

    Supports:
        - LINEAR: Analytically defined plasma shape based on aspect ratio, elongation, and triangularity.
        - ZHENG: Parametrized model from Zheng's solution.
        - F4E: Hamiltonian-based parametrization used in EUROfusion designs.
        - OTHER: Custom user-provided level-set function.
    """
    
    def __init__(self,EQUILIBRIUM,GEOMETRY,**kwargs):
        """
        Initializes the plasma boundary using a predefined geometry model or a custom function.

        Input:
            - EQUILIBRIUM : Object containing the mesh and problem configuration.
            - GEOMETRY (str): Identifier string specifying the geometry model:
                * 'LINEAR' — Analytic linear model with elongation and triangularity.
                * 'ZHENG' — Parametrization from Zheng's analytic solution.
                * 'F4E' — EUROfusion Hamiltonian-based parametrization.
                * 'OTHER' — User-specified level-set function via keyword argument.
            - kwargs: Additional parameters required by the specific geometry model:
                * For 'LINEAR' and 'ZHENG': R0, epsilon, kappa, delta
                * For 'F4E': Xsaddle, Xleft, Xright, Xtop
                * For 'OTHER': PHI0 (callable level-set function)

        Sets:
            - self.PHI0fun : Chosen level-set function defining the initial boundary.
            - self.PHI0 : Level-set field over the computational mesh.
            - self.PHI0rec : Level-set field over an extended rectangular mesh.
            - self.Xrec : Coordinates of the rectangular mesh used for visualization or debugging.
            - self.coeffs : Coefficients for the selected model, if applicable.
        """

        # IMPORT PROBLEM DATA
        self.eq = proxy(EQUILIBRIUM)
        # INITIAL BOUNDARY PREDEFINED MODELS
        self.INITIAL_GEOMETRY = None
        self.LINEAR_SOLUTION = 0
        self.ZHENG_SOLUTION = 1
        self.F4E_HAMILTONIAN = 2
        self.OTHER_PARAMETRIATION = 3
        
        # GENERAL ATTRIBUTES
        self.PHI0fun = None         # INITIAL PLASMA BOUNDARY LEVEL-SET FUNCTION
        self.PHI0 = None            # INITIAL PLASMA BOUNDARY LEVEL-SET FIELD (COMPUTATIONAL DOMAIN)
        self.Xrec = None            # EXTENDED RECTANGULAR MESH
        self.PHI0rec = None         # INITIAL PLASMA BOUNDARY LEVEL-SET FIELD
        
        ##### PRE-DEFINED INITIAL PLASMA BOUNDARY GEOMETRIES
        match GEOMETRY:
            case 'LINEAR':
                # GEOMETRY PARAMETERS
                self.INITIAL_GEOMETRY = self.LINEAR_SOLUTION
                self.R0 = kwargs['R0']                    # MEAN RADIUS
                self.epsilon = kwargs['epsilon']          # INVERSE ASPECT RATIO
                self.kappa = kwargs['kappa']              # ELONGATION
                self.delta = kwargs['delta']              # TRIANGULARITY
                self.coeffs = ComputeLinearSolutionCoefficients(self.epsilon,self.kappa,self.delta)
                # GEOMETRY LEVEL-SET FUNCTION
                self.PHI0fun = partial(PSIanalyticalLINEAR, R0=self.R0, coeffs=self.coeffs)
            
            case 'ZHENG':
                # GEOMETRY PARAMETERS
                self.INITIAL_GEOMETRY = self.ZHENG_SOLUTION
                self.R0 = kwargs['R0']                     # MEAN RADIUS
                self.epsilon = kwargs['epsilon']           # INVERSE ASPECT RATIO
                self.kappa = kwargs['kappa']               # ELONGATION
                self.delta = kwargs['delta']               # TRIANGULARITY
                self.coeffs = ComputeZhengSolutionCoefficients(self.R0,self.epsilon,self.kappa,self.delta)
                # GEOMETRY LEVEL-SET FUNCTION
                self.PHI0fun = self.PHIzheng
                
            case 'F4E':
                # GEOMETRY PARAMETERS
                self.INITIAL_GEOMETRY = self.F4E_HAMILTONIAN
                self.X_SADDLE = kwargs['Xsaddle']          # ACTIVE SADDLE POINT
                self.X_RIGHT = kwargs['Xright']            # POINT ON THE RIGHT
                self.X_LEFT = kwargs['Xleft']              # POINT ON THE LEFT
                self.X_TOP = kwargs['Xtop']                # POINT ON TOP
                self.coeffs = ComputeF4EPlasmaLScoeffs(self.X_SADDLE, self.X_RIGHT, self.X_LEFT, self.X_TOP)
                # GEOMETRY LEVEL-SET FUNCTION
                self.PHI0fun = partial(F4EPlasmaLS, coeffs=self.coeffs, X_SADDLE=self.X_SADDLE, X_LEFT=self.X_LEFT)
                
            case 'OTHER':
                self.INITIAL_GEOMETRY = self.OTHER_PARAMETRIATION      
                self.PHI0fun = kwargs['PHI0']
                
        # COMPUTE PLASMA BOUNDARY LEVEL-SET VALUES ON COMPUTATIONAL DOMAIN
        self.PHI0 = self.ComputeField(self.eq.MESH.X)
        # COMPUTE PLASMA BOUNDARY LEVEL-SET VALUES ON EXTENDED RECTANGULAR MESH
        ### PREPARE RECTANGULAR MESH
        Nr = 50
        Nz = 60
        self.Xrec = np.zeros([Nr*Nz,2])
        inode = 0
        for r in np.linspace(self.eq.MESH.Rmin-0.1,self.eq.MESH.Rmax+0.1,Nr):
            for z in np.linspace(self.eq.MESH.Zmin-0.1,self.eq.MESH.Zmax+0.1,Nz):
                self.Xrec[inode,:] = [r,z]
                inode += 1
        self.PHI0rec = self.ComputeField(self.Xrec)
        return
    
##################################################################################################
######################################## ZHENG MODEL LS ##########################################
##################################################################################################

    def PHIzheng(self,X):
        return -PSIanalyticalZHENG(X,self.coeffs)
    
    
##################################################################################################
########################################## REPRESENTATION ########################################
##################################################################################################
    
    def ComputeField(self,X):
        """
        Compute the initial level-set field values at given points.

        Input:
            - X (ndarray): Array of points of shape (N, 2), where each row is [R, Z].

        Output:
            PHI0 (ndarray): Array of computed field values at each input point.
        """
        PHI0 = np.zeros([np.shape(X)[0]])
        for inode in range(np.shape(X)[0]):
            PHI0[inode] = self.PHI0fun(X[inode,:])
        return PHI0
    
##################################################################################################
########################################## REPRESENTATION ########################################
##################################################################################################
    
    def Plot(self):

        #### FIGURE
        # PLOT PHI LEVEL-SET BACKGROUND VALUES 
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_aspect('equal')
        ax.set_xlim(np.min(self.Xrec[:,0])-eqplot.padx, np.max(self.Xrec[:,0])+eqplot.padx)
        ax.set_ylim(np.min(self.Xrec[:,1])-eqplot.pady, np.max(self.Xrec[:,1])+eqplot.pady)
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        ax.set_title('Initial plasma domain')
        
        # Plot low-opacity background (outside plasma region)
        ax.tricontourf(self.Xrec[:,0],self.Xrec[:,1],self.PHI0rec,cmap = eqplot.phicmap, levels=eqplot.Nphilevels, alpha=0.5)
        # Plot level-set inside computational domain
        contourf = ax.tricontourf(self.Xrec[:,0],self.Xrec[:,1],self.PHI0rec,cmap = eqplot.phicmap, levels=eqplot.Nphilevels)
        
        patch = PathPatch(self.eq.MESH.boundary_path, transform=ax.transData)
        for coll in contourf.collections:
            coll.set_clip_path(patch)
        
        # PLOT LEVEL-SET CONTOURS
        ax.tricontour(self.Xrec[:,0],self.Xrec[:,1],self.PHI0rec,
                      levels=eqplot.Nphilevels,
                      colors='black', 
                      linewidths=1)
        # PLOT INITIAL PLASMA BOUNDARY
        ax.tricontour(self.Xrec[:,0],self.Xrec[:,1],self.PHI0rec,
                      levels=[0],
                      colors=eqplot.plasmabouncolor, 
                      linewidths=eqplot.plasmabounlinewidth)
        
        # PLOT MESH BOUNDARY
        self.eq.MESH.PlotBoundary(ax = ax)
        # PLOT TOKAMAK FIRST WALL
        self.eq.TOKAMAK.PlotFirstWall(ax = ax)
        return