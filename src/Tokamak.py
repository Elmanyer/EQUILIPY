import matplotlib.pyplot as plt
from Magnet import *
import _plot as eqplot

class Tokamak:
    
    def __init__(self,WALL_MESH,MAGNETS=None):
        
        # TOKAMAK EXTERNAL MAGNETS
        self.MAGNETS = MAGNETS                          # ARRAY OF EXTERNAL MAGNETS
        
        # DEFINE TOKAMAK WALL DATA FROM PROBLEM MESH (MESH BOUNDARY == TOKAMAK FIRST WALL)
        self.Nn = len(WALL_MESH.BoundaryVertices)
        self.nodes = WALL_MESH.BoundaryVertices      # FIRST WALL VERTICES NODES (GLOBAL INDEXES)
        self.Xwall = WALL_MESH.X[self.nodes,:]       # FIRST WALL VERTICES COORDINATES MATRIX
        self.wall_path = WALL_MESH.boundary_path     # FIRST WALL PATH (FOR PATCHING)
                
        # OBTAIN COMPUTATIONAL MESH LIMITS
        self.Rmax = np.max(self.Xwall[:,0])
        self.Rmin = np.min(self.Xwall[:,0])
        self.Zmax = np.max(self.Xwall[:,1])
        self.Zmin = np.min(self.Xwall[:,1])
        return
    
    
    def Psi(self,X):
        psi = 0
        for magnet in self.MAGNETS:
            psi += magnet.Psi(X)
        return psi
    
    
    def ComputeField(self,X):
        n = np.shape(X)[0]
        psifield = np.zeros([n])
        for inode in range(n):
            psifield[inode] = self.Psi(X[inode,:])
        return psifield
    
    
    ##################################################################################################
    ######################################### REPRESENTATION #########################################
    ##################################################################################################
    
    def PlotFirstWall(self,ax=None):
        # GENERATE FIGURE IF NON EXISTENT
        if type(ax) == type(None):
            fig, ax = plt.subplots(1, 1, figsize=(5,6))
            ax.set_aspect('equal')
            ax.set_xlabel('R (in m)')
            ax.set_ylabel('Z (in m)')
            ax.set_title('Tokamak first wall')
            ax.set_xlim(np.min(self.Xwall[:,0])-eqplot.padx,np.max(self.Xwall[:,0])+eqplot.padx)
            ax.set_ylim(np.min(self.Xwall[:,1])-eqplot.pady,np.max(self.Xwall[:,1])+eqplot.pady)
        # PLOT TOKAMAK FIRST WALL
        for inode in range(self.Nn):
            ax.plot([self.Xwall[inode,0],self.Xwall[int((inode+1)%self.Nn),0]],
                    [self.Xwall[inode,1],self.Xwall[int((inode+1)%self.Nn),1]],
                    linewidth = eqplot.firstwalllinewidth, 
                    color = eqplot.firstwallcolor)
        return
    
    def PlotMagnets(self,ax=None):
        # GENERATE FIGURE IF NON EXISTENT
        if type(ax) == type(None):
            fig, ax = plt.subplots(1, 1, figsize=(5,6))
            ax.set_aspect('equal')
            ax.set_xlabel('R (in m)')
            ax.set_ylabel('Z (in m)')
            ax.set_title('Tokamak magnets')
        # PLOT MAGNETS
        if not type(self.MAGNETS) == type(None):
            for magnet in self.MAGNETS:
                magnet.Plot(ax)
        return
    
    
    def Plot(self):
        # GENERATE FIGURE
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_aspect('equal')
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        ax.set_title('Tokamak geometry')
        
        # PLOT TOKAMAK FIRST WALL
        self.PlotFirstWall(ax=ax)
        # PLOT EXTERNAL MAGNETS
        self.PlotMagnets(ax=ax)
        plt.show()
        return
    