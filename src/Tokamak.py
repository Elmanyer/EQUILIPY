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



import matplotlib.pyplot as plt
from Magnet import *
import _plot as eqplot

class Tokamak:
    
    """
    Represents a Tokamak device with external magnets and a first wall boundary.
    """
    
    def __init__(self,WALL_MESH,MAGNETS=None):
        """
        Initialize the Tokamak object using the wall mesh and optional external magnets.

        Input:
            - WALL_MESH (Mesh): Mesh object representing the Tokamak wall with attributes
                               BoundaryVertices (list or array), X (coordinates array),
                               and boundary_path.
            - MAGNETS (list or None): Optional list of external magnet objects.
        """
        
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
        """
        Compute the magnetic flux function Psi at a given point by summing contributions from all magnets.

        Input:
            X (array-like): Coordinate [R, Z] at which to evaluate Psi.

        Ouput:
            float: Magnetic flux Psi at point X.
        """
        psi = 0
        for magnet in self.MAGNETS:
            psi += magnet.Psi(X)
        return psi
    
    
    def ComputeField(self,X):
        """
        Compute the magnetic flux Psi field over multiple points.

        Input:
            X (ndarray): Array of points with shape (N, 2), where each row is [R, Z].

        Ouput:
            psifield (ndarray): Array of Psi values at each input point.
        """
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
    