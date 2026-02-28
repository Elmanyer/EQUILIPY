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
from Magnet import *

class EquilipyBfield:
    
    def Br(self,X):
        """
        Total radial magnetic at point X such that    Br = -1/R dpsi/dZ
        """
        elem = self.SearchElement(X,range(self.MESH.Ne))
        return self.MESH.Elements[elem].Br(X)
    
    def Bz(self,X):
        """
        Total vertical magnetic at point X such that    Bz = (1/R) dpsi/dR
        """
        elem = self.SearchElement(X,range(self.MESH.Ne))
        return self.MESH.Elements[elem].Bz(X)
    
    def Bpol(self,X):
        """
        Toroidal magnetic field
        """
        elem = self.SearchElement(X,range(self.MESH.Ne))
        Brz = self.MESH.Elements[elem].Brz(X)
        return np.sqrt(Brz[0] * Brz[0] + Brz[1] * Brz[1])
    
    def Btor(self,X):
        """
        Toroidal magnetic field
        """
        
        return
    
    def Btot(self,X):
        """
        Total magnetic field
        """
        
        return
    
    
    def ComputeBrField(self):
        """
        Total radial magnetic field such that    Br = (-1/R) dpsi/dZ
        """
        self.ComputePlasmaBoundStandardQuadratures()
        Br = np.zeros([self.MESH.Ne*self.nge])
        for ielem, ELEMENT in enumerate(self.MESH.Elements):
            Br[ielem*self.nge:(ielem+1)*self.nge] = ELEMENT.Brg()
        return Br
    
    def ComputeBzField(self):
        """
        Total vertical magnetic field such that    Bz = (1/R) dpsi/dR
        """
        self.ComputePlasmaBoundStandardQuadratures()
        Bz = np.zeros([self.MESH.Ne*self.nge])
        for ielem, ELEMENT in enumerate(self.MESH.Elements):
            Bz[ielem*self.nge:(ielem+1)*self.nge] = ELEMENT.Bzg()
        return Bz
    
    def ComputeBrzField(self):
        """
        Magnetic vector field such that    (Br, Bz) = ((-1/R) dpsi/dZ, (1/R) dpsi/dR)
        """
        self.ComputePlasmaBoundStandardQuadratures()
        self.Brzfield = np.zeros([self.MESH.Ne*self.nge,self.MESH.dim])
        for ielem, ELEMENT in enumerate(self.MESH.Elements):
            self.Brzfield[ielem*self.nge:(ielem+1)*self.nge,:] = ELEMENT.Brzg()
        return 
    
    
    def ComputeMagnetsBfield(self,regular_grid=False,**kwargs):
        if regular_grid:
            # Define regular grid
            Nr = 50
            Nz = 70
            grid_r, grid_z= np.meshgrid(np.linspace(kwargs['rmin'], kwargs['rmax'], Nr),np.linspace(kwargs['zmin'], kwargs['zmax'], Nz))
            Br = np.zeros([Nz,Nr])
            Bz = np.zeros([Nz,Nr])
            for ir in range(Nr):
                for iz in range(Nz):
                    # SUM COILS CONTRIBUTIONS
                    for COIL in self.COILS:
                        Br[iz,ir] += COIL.Br(np.array([grid_r[iz,ir],grid_z[iz,ir]]))
                        Bz[iz,ir] += COIL.Bz(np.array([grid_r[iz,ir],grid_z[iz,ir]]))
                    # SUM SOLENOIDS CONTRIBUTIONS
                    for SOLENOID in self.SOLENOIDS:
                        Br[iz,ir] += SOLENOID.Br(np.array([grid_r[iz,ir],grid_z[iz,ir]]))
                        Bz[iz,ir] += SOLENOID.Bz(np.array([grid_r[iz,ir],grid_z[iz,ir]]))
            return grid_r, grid_z, Br, Bz
        else:
            Br = np.zeros([self.MESH.Nn])
            Bz = np.zeros([self.MESH.Nn])
            for inode in range(self.MESH.Nn):
                # SUM COILS CONTRIBUTIONS
                for COIL in self.COILS:
                    Br[inode] += COIL.Br(self.MESH.X[inode,:])
                    Br[inode] += COIL.Br(self.MESH.X[inode,:])
                # SUM SOLENOIDS CONTRIBUTIONS
                for SOLENOID in self.SOLENOIDS:
                    Br[inode] += SOLENOID.Br(self.MESH.X[inode,:])
                    Br[inode] += SOLENOID.Br(self.MESH.X[inode,:])
            return Br, Bz
    
    
    def IntegrateBpolPlasmaDomain(self):
        integral = 0
        
        # INTEGRATE OVER PLASMA ELEMENTS
        for ielem in self.MESH.PlasmaElems:
            # ISOLATE ELEMENT
            ELEMENT = self.MESH.Elements[ielem]
            # COMPUTE MAGNETIC FIELD AT INTEGRATION NODES
            Brzg = ELEMENT.Brzg()
            # LOOP OVER GAUSS NODES
            for ig in range(ELEMENT.ng):
                integral += (Brzg[ig,0]**2 + Brzg[ig,1]**2)*ELEMENT.Xg[ig,0]*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                    
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for ielem in self.MESH.PlasmaBoundElems:
            # ISOLATE ELEMENT
            ELEMENT = self.MESH.Elements[ielem]
            # LOOP OVER SUBELEMENTS
            for SUBELEM in ELEMENT.SubElements:
                # INTEGRATE IN SUBDOMAIN INSIDE PLASMA REGION
                if SUBELEM.Dom < 0:
                    # COMPUTE MAGNETIC FIELD AT INTEGRATION NODES
                    Brzg = SUBELEM.Brzg()
                    # LOOP OVER GAUSS NODES
                    for ig in range(SUBELEM.ng):
                        integral += (Brzg[ig,0]**2 + Brzg[ig,1]**2)*SUBELEM.Xg[ig,0]*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]
                        
        return integral