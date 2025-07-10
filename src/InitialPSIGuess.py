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
from random import random
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from AnalyticalSolutions import *
from functools import partial
import _plot as eqplot

class InitialGuess:
    
    """
    Encapsulates the construction of an initial guess for the poloidal magnetic flux (Ψ)
    field over a computational domain. Multiple predefined analytical and semi-analytical
    models are supported, optionally with white noise and normalization.

    Purpose:
        - Provides a flexible mechanism to define Ψ₀ (initial magnetic flux surface).
        - Enables initialization for equilibrium solvers or iterative methods.

    Supported Models:
        - LINEAR        : Analytical solution based on aspect ratio, elongation, triangularity.
        - ZHENG         : Semi-analytical model based on Zheng's formulation.
        - NONLINEAR     : Custom nonlinear analytical profile.
        - F4E           : EUROfusion-based Hamiltonian level-set profile.
        - FOCUS         : Localized flux peak at specified point.
        - OTHER         : User-defined callable function Ψ₀.
    """

    
    def __init__(self,EQUILIBRIUM,PSI_GUESS,NORMALISE=False,NOISE=False,**kwargs):
        """
        Initialize InitialGuess instance with a selected model.

        Parameters:
            EQUILIBRIUM : object
                Equilibrium problem proxy with mesh and geometry data.
            PSI_GUESS : str
                Name of initial guess model ('LINEAR', 'ZHENG', 'NONLINEAR', 'F4E', 'FOCUS', 'OTHER').
            NORMALISE : bool, optional
                If True, normalizes Ψ to standard range.
            NOISE : bool, optional
                If True, adds white noise to Ψ.
            **kwargs : dict
                Model-specific parameters, e.g. R0, epsilon, kappa, delta, A, X0, etc.
        """

        # IMPORT PROBLEM DATA
        self.eq = proxy(EQUILIBRIUM)
        # PSI INITIAL GUESS PREDEFINED MODELS
        self.INITIAL_GUESS = None
        self.LINEAR_SOLUTION = 0
        self.ZHENG_SOLUTION = 1
        self.NONLINEAR_SOLUTION = 2
        self.F4E_HAMILTONIAN = 3
        self.FOCUS_PSI = 4
        self.OTHER_GUESS = 5
        
        # GENERAL ATTRIBUTES
        self.PSI0fun = None         # INITIAL GUESS FUNCTION
        self.PSI0 = None            # INITIAL GUESS FIELD (COMPUTATIONAL DOMAIN)
        self.Opoint0 = None         # INITIAL GUESS O-POINTS (LOCAL EXTREMA)
        self.Xpoint0 = None         # INITIAL GUESS X-POINTS (SADDLE POINTS)
        self.PSI0_0 = None          # INITIAL GUESS VALUE AT O-POINT
        self.PSI0_X = None          # INITIAL GUESS VALUE AT X-POINT
        self.X0 = None              # CRITICAL POINTS INITIAL GUESSES
        self.NOISE = NOISE          # WHITE NOISE SWITCH
        self.NORMALISE = NORMALISE  # NORMALISATION SWITCH
        
        # DEFINE THE MODEL
        match PSI_GUESS:
            case 'LINEAR':
                # INITIAL GUESS PARAMETERS
                self.INITIAL_GUESS = self.LINEAR_SOLUTION
                self.R0 = kwargs['R0']                    # MEAN RADIUS
                self.epsilon = kwargs['epsilon']          # INVERSE ASPECT RATIO
                self.kappa = kwargs['kappa']              # ELONGATION
                self.delta = kwargs['delta']              # TRIANGULARITY
                self.coeffs = ComputeLinearSolutionCoefficients(self.epsilon,self.kappa,self.delta)
                # INITIAL GUESS
                if NOISE:
                    self.A = kwargs['A']
                    self.PSI0fun = self.PSIlinearNOISE
                else:
                    self.PSI0fun = partial(PSIanalyticalLINEAR, R0=self.R0, coeffs=self.coeffs)

            case 'ZHENG':
                # INITIAL GUESS PARAMETERS
                self.INITIAL_GUESS = self.ZHENG_SOLUTION
                self.R0 = kwargs['R0']                     # MEAN RADIUS
                self.epsilon = kwargs['epsilon']           # INVERSE ASPECT RATIO
                self.kappa = kwargs['kappa']               # ELONGATION
                self.delta = kwargs['delta']               # TRIANGULARITY
                self.coeffs = ComputeZhengSolutionCoefficients(self.R0,self.epsilon,self.kappa,self.delta)
                # INITIAL GUESS
                if NOISE:
                    self.A = kwargs['A']
                    self.PSI0fun = self.PSIzhengNOISE
                else:
                    self.PSI0fun = partial(PSIanalyticalZHENG, coeffs=self.coeffs)

            case 'NONLINEAR':
                # INITIAL GUESS PARAMETERS
                self.INITIAL_GUESS = self.NONLINEAR_SOLUTION
                self.R0 = kwargs['R0']
                self.coeffs = [1.15*np.pi,  # [Kr, 
                               1.15,        #  Kz,
                               -0.5]        #  R0] 
                # INITIAL GUESS
                if NOISE:
                    self.A = kwargs['A']
                    self.PSI0fun = self.PSInonlinearNOISE
                else:
                    self.PSI0fun = partial(PSIanalyticalNONLINEAR, R0=self.R0, coeffs=self.coeffs)
                    
            case 'F4E':
                # GEOMETRY PARAMETERS
                self.INITIAL_GUESS = self.F4E_HAMILTONIAN
                self.X_SADDLE = kwargs['Xsaddle']          # ACTIVE SADDLE POINT
                self.X_RIGHT = kwargs['Xright']            # POINT ON THE RIGHT
                self.X_LEFT = kwargs['Xleft']              # POINT ON THE LEFT
                self.X_TOP = kwargs['Xtop']                # POINT ON TOP
                self.coeffs = ComputeF4EPlasmaLScoeffs(self.X_SADDLE, self.X_RIGHT, self.X_LEFT, self.X_TOP)
                # GEOMETRY LEVEL-SET FUNCTION
                self.PSI0fun = partial(F4EPlasmaLS, coeffs=self.coeffs, X_SADDLE=self.X_SADDLE, X_LEFT=self.X_LEFT)
                
            case 'FOCUS':
                self.INITIAL_GUESS = self.FOCUS_PSI
                self.R0 = kwargs['R0']
                self.Z0 = kwargs['Z0']
                self.radius = kwargs['radius']
                self.PSI0fun = self.PSIfocus

            case 'OTHER':
                self.INITIAL_GUESS = self.OTHER_GUESS
                self.PSI0fun = kwargs['PSI0fun']
                
        if 'X0' in kwargs:
            self.X0 = kwargs['X0']
                
        # COMPUTE INITIAL GUESS ON COMPUTATIONAL DOMAIN    
        self.PSI0 = self.ComputeField(self.eq.MESH.X,NORMALISE,self.X0)
        return
    
    
    def ComputeField(self,X,NORMALISE=False,X0=None):
        """
        Evaluate the initial guess function Ψ0fun at given points.

        Input:
            - X (ndarray): Array of points [[R,z], ...] where Ψ0 is evaluated.
            - NORMALISE (bool, optional): If True, normalize the computed field values.
            - X0 (optional): Critical points or parameters used for normalization (if any).

        Returns:
            - ndarray: Computed Ψ0 values at each point in X.
        """
        PSI0 = np.zeros([np.shape(X)[0]])
        for inode in range(np.shape(X)[0]):
            PSI0[inode] = self.PSI0fun(X[inode,:])
        if NORMALISE:
            PSI0 = self.NormalisePSI(PSI0,X0)
        return PSI0
    
    def NormalisePSI(self,PSI,X0=None):
        self.Opoint0, self.Xpoint0 = self.eq.FindCritical(PSI,X0)
        if not self.Opoint0:
            raise ValueError("No O-points found!")
        else:
            self.PSI0_0 = self.Opoint0[0][1]
        if not self.Xpoint0:
            self.PSI0_X = 0.0
        else:
            self.PSI0_X = self.Xpoint0[0][1]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        PSI = (PSI - self.PSI0_0) / (self.PSI0_X - self.PSI0_0)
        return PSI
    
    
##################################################################################################
######################################## LINEAR MODEL ############################################
##################################################################################################

    def PSIlinearNOISE(self,X):
        return PSIanalyticalLINEAR(X,self.R0,self.coeffs)*self.A*random()

##################################################################################################
######################################## ZHENG MODEL #############################################
##################################################################################################

    def PSIzhengNOISE(self,X):
        return PSIanalyticalZHENG(X,self.coeffs)*self.A*random()

##################################################################################################
##################################### NONLINEAR MODEL ############################################
##################################################################################################
    
    def PSInonlinearNOISE(self,X):
        return PSIanalyticalNONLINEAR(X,self.R0,self.coeffs)*self.A*random()
    
##################################################################################################
################################### EXPONENTIAL MODEL ############################################
##################################################################################################
    
    def PSIfocus(self,X):
        return np.exp(-((X[0] - self.R0) ** 2 + (X[1] - self.Z0) ** 2) / self.radius**2)
    
##################################################################################################
###################################### REPRESENTATION ############################################
################################################################################################## 
    
    def Plot(self):
        # DEFINE SEPARATRIX DEPENDING ON NORMALISATION
        if self.NORMALISE:
            klevel = 1
        else:
            klevel = 0

        #### FIGURE
        # PLOT INITIAL PSI GUESS BACKGROUND VALUES
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_xlim(self.eq.MESH.Rmin-eqplot.padx,self.eq.MESH.Rmax+eqplot.padx)
        ax.set_ylim(self.eq.MESH.Zmin-eqplot.pady,self.eq.MESH.Zmax+eqplot.pady)
        ax.set_aspect('equal')
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        ax.set_title('Initial poloidal magnetic flux guess')
        
        # PLOT INITIAL PSI GUESS
        contourf = ax.tricontourf(self.eq.MESH.X[:,0],self.eq.MESH.X[:,1], self.PSI0, levels = eqplot.Npsilevels, 
                                  cmap=eqplot.plasmacmap)
        contour = ax.tricontour(self.eq.MESH.X[:,0],self.eq.MESH.X[:,1], self.PSI0, levels = eqplot.Npsilevels, 
                                colors='black', 
                                linewidths=1)
        contour0 = ax.tricontour(self.eq.MESH.X[:,0],self.eq.MESH.X[:,1], self.PSI0, levels=[klevel], 
                                 colors=eqplot.plasmabouncolor, 
                                 linewidths=eqplot.plasmabounlinewidth)
        
        patch = PathPatch(self.eq.MESH.boundary_path, transform=ax.transData)
        for cont in [contourf,contour,contour0]:
            for coll in cont.collections:
                coll.set_clip_path(patch)
            
        # PLOT MESH BOUNDARY
        self.eq.MESH.PlotBoundary(ax = ax)
        # PLOT TOKAMAK FIRST WALL
        self.eq.TOKAMAK.PlotFirstWall(ax = ax)
        
        # PLOT COLORBAR
        plt.colorbar(contourf, ax=ax)
        return

