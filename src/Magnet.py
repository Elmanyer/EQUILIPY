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



from Greens import *

class Coil:
    """
    Class representing a tokamak's external coil (confinement magnet).
    """
    
    def __init__(self,index,dim,X,I):
        """
        Constructor to initialize the Coil object with the provided attributes.

        Input:
            - index (int): The index of the coil in the global system.
            - dim (int): The spatial dimension of the coil coordinates.
            - X (numpy.ndarray): A 1D array representing the position coordinates of the coil in physical space.
            - I (float): The current carried by the coil.
        """
        
        self.index = index      # COIL INDEX
        self.dim = dim          # SPATIAL DIMENSION
        self.X = X              # COIL POSITION COORDINATES
        self.I = I              # COIL CURRENT
        return
    
    
    def Br(self,X):
        """
        Calculate radial magnetic field Br at X=(R,Z)
        """
        return GreensBr(self.X,X) * self.I

    def Bz(self,X):
        """
        Calculate vertical magnetic field Bz at X=(R,Z)
        """
        return GreensBz(self.X,X) * self.I

    def Psi(self,X):
        """
        Calculate poloidal flux psi at X=(R,Z) due to coil
        """
        return GreensFunction(self.X,X) * self.I
    
    
    
class Solenoid:
    """
    Class representing a tokamak's external solenoid (confinement magnet).
    """
    
    def __init__(self,index,dim,Xe,I,Nturns):
        """
        Constructor to initialize the Solenoid object with the provided attributes.

        Input:
            - index (int): The index of the solenoid in the global system.
            - dim (int): The spatial dimension of the solenoid coordinates.
            - X (numpy.ndarray): Solenoid nodal coordinates matrix.
            - I (float): The current carried by the solenoid.
        """
        
        self.index = index      # SOLENOID INDEX
        self.dim = dim          # SPATIAL DIMENSION
        self.Xe = Xe            # SOLENOID POSITION COORDINATES MATRIX
        self.I = I              # SOLENOID CURRENT
        self.Nturns = Nturns    # SOLENOID NUMBER OF TURNS
        
        # NUMERICAL INTEGRATION QUADRATURE
        self.ng = None          # NUMBER OF GAUSS INTEGRATION NODES FOR STANDARD 1D QUADRATURE
        self.XIg = None         # GAUSS INTEGRATION NODES (REFERENCE SPACE)
        self.Xg = None          # GAUSS INTEGRATION NODES (PHYSICAL SPACE)
        self.Wg = None          # GAUSS INTEGRATION WEIGTHS 
        self.Ng = None          # REFERENCE SHAPE FUNCTIONS EVALUATED AT GAUSS INTEGRATION NODES 
        self.dNdxig = None      # REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT GAUSS INTEGRATION NODES
        self.detJg = None       # DETERMINANT OF JACOBIAN OF TRANSFORMATION FROM 1D REFERENCE ELEMENT TO 2D PHYSICAL SOLENOID
        return
    
    def Solenoid_coils(self):
        """
        Calculate the position of the individual coils constituting the solenoid.
        """
        Xcoils = np.zeros([self.Nturns,self.dim])
        Xcoils[0,:] = self.Xe[0,:]
        Xcoils[-1,:] = self.Xe[1,:]
        dr = (self.Xe[1,0]-self.Xe[0,0])/(self.Nturns-1)
        dz = (self.Xe[1,1]-self.Xe[0,1])/(self.Nturns-1)
        for icoil in range(1,self.Nturns):
            Xcoils[icoil,:] = [self.Xe[0,0]+dr*icoil, 
                               self.Xe[0,1]+dz*icoil]
        return Xcoils
    
    def Psi(self,X):
        """
        Calculate poloidal flux psi at (R,Z) due to solenoid
        """
        Psi_sole = 0.0
        Xcoils = self.Solenoid_coils()
        for icoil in range(self.Nturns):
            Psi_sole += GreensFunction(Xcoils[icoil,:],X) * self.I
        return Psi_sole

    def Br(self,X):
        """
        Calculate radial magnetic field Br at (R,Z) due to solenoid
        """
        Br_sole = 0.0
        Xcoils = self.Solenoid_coils()
        for icoil in range(self.Nturns):
            Br_sole += GreensBr(Xcoils[icoil,:],X) * self.I
        return Br_sole

    def Bz(self,X):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to solenoid
        """
        Bz_sole = 0.0
        Xcoils = self.Solenoid_coils()
        for icoil in range(self.Nturns):
            Bz_sole += GreensBz(Xcoils[icoil,:],X) * self.I
        return Bz_sole

