
from weakref import proxy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from AnalyticalSolutions import *
from functools import partial

class InitialPlasmaBoundary:
    
    plasmacmap = plt.get_cmap('jet')
    #plasmacmap = plt.get_cmap('winter_r')
    plasmabouncolor = 'green'
    vacvesswallcolor = 'gray'
    magneticaxiscolor = 'red'
    
    def __init__(self,EQUILIBRIUM,GEOMETRY,**kwargs):
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
        # Plot low-opacity background (outside plasma region)
        ax.tricontourf(self.Xrec[:,0],self.Xrec[:,1],self.PHI0rec,levels=30, alpha=0.5)
        # Plot level-set inside computational domain
        contourf = ax.tricontourf(self.Xrec[:,0],self.Xrec[:,1],self.PHI0rec,levels=30)
        
        patch = PathPatch(self.eq.MESH.boundary_path, transform=ax.transData)
        for coll in contourf.collections:
            coll.set_clip_path(patch)
        
        # PLOT MESH BOUNDARY
        for iboun in range(self.eq.MESH.Nbound):
            ax.plot(self.eq.MESH.X[self.eq.MESH.Tbound[iboun,:2],0],self.eq.MESH.X[self.eq.MESH.Tbound[iboun,:2],1],linewidth = 4, color = self.vacvesswallcolor)
        # PLOT LEVEL-SET CONTOURS
        ax.tricontour(self.Xrec[:,0],self.Xrec[:,1],self.PHI0rec,levels=30,colors='black', linewidths=1)
        # PLOT INITIAL PLASMA BOUNDARY
        ax.tricontour(self.Xrec[:,0],self.Xrec[:,1],self.PHI0rec,levels=[0],colors='red', linewidths=3)
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        ax.set_title('Initial plasma domain')
        return