from weakref import proxy
import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from AnalyticalSolutions import *
from functools import partial

class InitialGuess:
    
    def __init__(self,PROBLEM,PSI_GUESS,NORMALISE=False,NOISE=False,**kwargs):
        # IMPORT PROBLEM DATA
        self.problem = proxy(PROBLEM)
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
        self.PSI0 = self.ComputeField(self.problem.X,NORMALISE,self.X0)
        return
    
    
    def ComputeField(self,X,NORMALISE=False,X0=None):
        PSI0 = np.zeros([np.shape(X)[0]])
        for inode in range(np.shape(X)[0]):
            PSI0[inode] = self.PSI0fun(X[inode,:])
        if NORMALISE:
            PSI0 = self.NormalisePSI(PSI0,X0)
        return PSI0
    
    def NormalisePSI(self,PSI,X0=None):
        self.Opoint, self.Xpoint = self.problem.FindCritical(PSI,X0)
        if not self.Opoint:
            raise ValueError("No O-points found!")
        else:
            self.PSI0_0 = self.Opoint[0][1]
        if not self.Xpoint:
            self.PSI0_X = 0.0
        else:
            self.PSI0_X = self.Xpoint[0][1]

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
        ax.set_aspect('equal')
        contourf = ax.tricontourf(self.problem.X[:,0],self.problem.X[:,1],self.PSI0,levels=30)
        contour = ax.tricontour(self.problem.X[:,0],self.problem.X[:,1],self.PSI0,levels=30,colors='black', linewidths=1)
        contour0 = ax.tricontour(self.problem.X[:,0],self.problem.X[:,1],self.PSI0,levels=[klevel],colors='black', linewidths=3)
        # Define computational domain's boundary path
        compboundary = np.zeros([len(self.problem.BoundaryVertices)+1,2])
        compboundary[:-1,:] = self.problem.X[self.problem.BoundaryVertices,:]
        # Close path
        compboundary[-1,:] = compboundary[0,:]
        clip_path = Path(compboundary)
        patch = PathPatch(clip_path, transform=ax.transData)
        for cont in [contourf,contour,contour0]:
            for coll in cont.collections:
                coll.set_clip_path(patch)
            
        # PLOT MESH BOUNDARY
        for iboun in range(self.problem.Nbound):
            ax.plot(self.problem.X[self.problem.Tbound[iboun,:2],0],self.problem.X[self.problem.Tbound[iboun,:2],1],linewidth = 4, color = 'grey')
        # PLOT COLORBAR
        plt.colorbar(contourf, ax=ax)
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        ax.set_title('Initial poloidal magnetic flux guess')
        return

