import numpy as np
from Magnet import *

class EquilipyBfield:
    
    def Br(self,X):
        """
        Total radial magnetic at point X such that    Br = -1/R dpsi/dZ
        """
        elem = self.SearchElement(X,range(self.Mesh.Ne))
        return self.Mesh.Elements[elem].Br(X)
    
    def Bz(self,X):
        """
        Total vertical magnetic at point X such that    Bz = (1/R) dpsi/dR
        """
        elem = self.SearchElement(X,range(self.Mesh.Ne))
        return self.Mesh.Elements[elem].Bz(X)
    
    def Bpol(self,X):
        """
        Toroidal magnetic field
        """
        elem = self.SearchElement(X,range(self.Mesh.Ne))
        Brz = self.Mesh.Elements[elem].Brz(X)
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
        Br = np.zeros([self.Mesh.Ne*self.nge])
        for ielem, ELEMENT in enumerate(self.Mesh.Elements):
            Br[ielem*self.nge:(ielem+1)*self.nge] = ELEMENT.Brg()
        return Br
    
    def ComputeBzField(self):
        """
        Total vertical magnetic field such that    Bz = (1/R) dpsi/dR
        """
        self.ComputePlasmaBoundStandardQuadratures()
        Bz = np.zeros([self.Mesh.Ne*self.nge])
        for ielem, ELEMENT in enumerate(self.Mesh.Elements):
            Bz[ielem*self.nge:(ielem+1)*self.nge] = ELEMENT.Bzg()
        return Bz
    
    def ComputeBrzField(self):
        """
        Magnetic vector field such that    (Br, Bz) = ((-1/R) dpsi/dZ, (1/R) dpsi/dR)
        """
        self.ComputePlasmaBoundStandardQuadratures()
        self.Brzfield = np.zeros([self.Mesh.Ne*self.nge,self.Mesh.dim])
        for ielem, ELEMENT in enumerate(self.Mesh.Elements):
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
            Br = np.zeros([self.Mesh.Nn])
            Bz = np.zeros([self.Mesh.Nn])
            for inode in range(self.Mesh.Nn):
                # SUM COILS CONTRIBUTIONS
                for COIL in self.COILS:
                    Br[inode] += COIL.Br(self.Mesh.X[inode,:])
                    Br[inode] += COIL.Br(self.Mesh.X[inode,:])
                # SUM SOLENOIDS CONTRIBUTIONS
                for SOLENOID in self.SOLENOIDS:
                    Br[inode] += SOLENOID.Br(self.Mesh.X[inode,:])
                    Br[inode] += SOLENOID.Br(self.Mesh.X[inode,:])
            return Br, Bz
    
    
    def IntegrateBpolPlasmaDomain(self):
        integral = 0
        
        # INTEGRATE OVER PLASMA ELEMENTS
        for ielem in self.Mesh.PlasmaElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Mesh.Elements[ielem]
            # COMPUTE MAGNETIC FIELD AT INTEGRATION NODES
            Brzg = ELEMENT.Brzg()
            # LOOP OVER GAUSS NODES
            for ig in range(ELEMENT.ng):
                integral += (Brzg[ig,0]**2 + Brzg[ig,1]**2)*ELEMENT.Xg[ig,0]*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                    
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for ielem in self.Mesh.PlasmaBoundElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Mesh.Elements[ielem]
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