from weakref import proxy
import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from AnalyticalSolutions import *
from functools import partial

class InitialGuess:
    
    def __init__(self,PROBLEM,PSI_GUESS,NOISE=False,**kwargs):
        # IMPORT PROBLEM DATA
        self.problem = proxy(PROBLEM)
        # PSI INITIAL GUESS PREDEFINED MODELS
        self.INITIAL_GUESS = None
        self.LINEAR_SOLUTION = 0
        self.ZHENG_SOLUTION = 1
        self.NONLINEAR_SOLUTION = 2
        self.OTHER_GUESS = 3
        
        self.NOISE = NOISE
        
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
                    self.PSI0 = self.PSIlinearNOISE
                else:
                    self.PSI0 = partial(PSIanalyticalLINEAR, R0=self.R0, coeffs=self.coeffs)

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
                    self.PSI0 = self.PSIzhengNOISE
                else:
                    self.PSI0 = partial(PSIanalyticalZHENG, coeffs=self.coeffs)

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
                    self.PSI0 = self.PSInonlinearNOISE
                else:
                    self.PSI0 = partial(PSIanalyticalNONLINEAR, R0=self.R0, coeffs=self.coeffs)

            case 'OTHER':
                self.INITIAL_GUESS = self.OTHER_GUESS
                self.PSI0 = kwargs['PSI0']
                
        return
    
    
    def ComputeField(self,X):
        
        PSI0 = np.zeros([np.shape(X)[0]])
        for inode in range(np.shape(X)[0]):
            PSI0[inode] = self.PSI0(X[inode,:])
        return PSI0
    
    
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
###################################### REPRESENTATION ############################################
################################################################################################## 
    
    def Plot(self):
        
        # COMPUTE INITIAL PSI GUESS FIELD
        PSI0 = self.ComputeField(self.problem.X)
        #### FIGURE
        # PLOT INITIAL PSI GUESS BACKGROUND VALUES
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_aspect('equal')
        contourf = ax.tricontourf(self.problem.X[:,0],self.problem.X[:,1],PSI0,levels=30)
        contour = ax.tricontour(self.problem.X[:,0],self.problem.X[:,1],PSI0,levels=30,colors='black', linewidths=1)
        # Define computational domain's boundary path
        compboundary = np.zeros([len(self.problem.BoundaryVertices)+1,2])
        compboundary[:-1,:] = self.problem.X[self.problem.BoundaryVertices,:]
        # Close path
        compboundary[-1,:] = compboundary[0,:]
        clip_path = Path(compboundary)
        patch = PathPatch(clip_path, transform=ax.transData)
        for cont in [contourf,contour]:
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

