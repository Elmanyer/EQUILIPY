import matplotlib.pyplot as plt
from Magnet import *

class Tokamak:
    
    vacvesswallcolor = 'gray'
    magnetcolor = '#56B4E9'
    
    def __init__(self,EQUILIBRIUM,MAGNETS=None):
        
        # TOKAMAK EXTERNAL MAGNETS
        self.MAGNETS = MAGNETS                          # ARRAY OF EXTERNAL MAGNETS
        
        # DEFINE TOKAMAK WALL DATA FROM PROBLEM MESH (MESH BOUNDARY == TOKAMAK FIRST WALL)
        self.Nn = len(EQUILIBRIUM.MESH.BoundaryVertices)
        self.nodes = EQUILIBRIUM.MESH.BoundaryVertices      # FIRST WALL VERTICES NODES (GLOBAL INDEXES)
        self.Xwall = EQUILIBRIUM.MESH.X[self.nodes,:]       # FIRST WALL VERTICES COORDINATES MATRIX
        self.wall_path = EQUILIBRIUM.MESH.boundary_path     # FIRST WALL PATH (FOR PATCHING)
                
        return
    
    
    def Psi(self,X):
        
        psi = 0
        for magnet in self.MAGNETS:
            psi += magnet.Psi(X)
        
        return psi
    
    
    def Plot(self):
        
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_aspect('equal')
        
        # PLOT TOKAMAK FIRST WALL
        for inode in range(self.Nn):
            ax.plot([self.Xwall[inode,0],self.Xwall[int((inode+1)%self.Nn),0]],[self.Xwall[inode,1],self.Xwall[int((inode+1)%self.Nn),1]],linewidth = 5, color = self.vacvesswallcolor)
        
        # PLOT EXTERNAL MAGNETS
        if not type(self.MAGNETS) == type(None):
            for magnet in self.MAGNETS:
                magnet.Plot(ax)
                
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        ax.set_title('Tokamak geometry')
        plt.show()
        return
    