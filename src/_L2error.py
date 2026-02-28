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
from Element import *

class EquilipyL2error:
    
    """
    A class to compute errors between the numerical and analytical solutions of the magnetic flux function (PSI) in equilibrium simulations.
    """

    def __init__(self):
        """
        Constructor to initialize error tracking attributes.
        """
        # INITIATE ERROR ARRAYS
        self.PSIexact = None                    # ANALYTICAL SOLUTION FIELD
        self.PSIerror = None                    # ABSOLUTE PSI FIELD ERROR 
        self.PSIrelerror = None                 # ABSOLUTE RELATIVE PSI FIELD ERROR 
        self.ErrorEuclinorm = None              # EUCLIDEAN NORM ERROR
        self.RelErrorEuclinorm = None           # EUCLIDEAN NORM RELATIVE ERROR
        self.ErrorL2norm = None                 # L2 INTEGRAL NORM ERROR 
        self.RelErrorL2norm = None              # L2 INTEGRAL NORM RELATIVE ERROR
        
        super().__init__()
        return
    
    
    def ComputeErrorField(self):
        """
        Computes the error between the numerical and analytical PSI solutions.

        Computes:
            self.PSIexact          : Analytical PSI at each node.
            self.PSIerror          : Absolute pointwise error.
            self.PSIrelerror       : Relative pointwise error.
            self.ErrorEuclinorm    : Euclidean norm of absolute error.
            self.RelErrorEuclinorm : Euclidean norm of relative error.
        """
        # COMPUTE ERROR FIELDS
        self.PSIexact = np.zeros([self.MESH.Nn])
        self.PSIerror = np.zeros([self.MESH.Nn])
        self.PSIrelerror = np.zeros([self.MESH.Nn])
        for inode in range(self.MESH.Nn):
            self.PSIexact[inode] = self.PlasmaCurrent.PSIanalytical(self.MESH.X[inode,:])
            self.PSIerror[inode] = abs(self.PSIexact[inode]-self.PSI_NORM[inode,1])
            self.PSIrelerror[inode] = self.PSIerror[inode]/abs(self.PSIexact[inode])
            if self.PSIerror[inode] < 1e-16:
                self.PSIerror[inode] = 1e-16
                self.PSIrelerror[inode] = 1e-16
    
        # COMPUTE EUCLIDEAN NORM ERRORS
        self.ErrorEuclinorm = np.linalg.norm(self.PSIerror)
        self.RelErrorEuclinorm = np.linalg.norm(self.PSIrelerror)
        return
    
    
    def ComputeL2errorPlasma(self):
        """
        Computes the L2 integral norm error of the PSI field by integrating the squared difference between the analytical solution and the 
        computed solution over the plasma region.
        
        Computes:
            self.ErrorL2norm        : L2 INTEGRAL NORM ERROR 
            self.RelErrorL2norm     : L2 INTEGRAL NORM RELATIVE ERROR
        """
        ErrorL2norm = 0
        PSIexactL2norm = 0
        # INTEGRATE OVER PLASMA ELEMENTS
        for elem in self.MESH.PlasmaElems:
            # ISOLATE ELEMENT
            ELEMENT = self.MESH.Elements[elem]
            # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
            PSIg = ELEMENT.Ng @ ELEMENT.PSIe
            # LOOP OVER GAUSS NODES
            for ig in range(ELEMENT.ng):
                ErrorL2norm += (PSIg[ig]-self.PlasmaCurrent.PSIanalytical(ELEMENT.Xg[ig,:],NORMALISED=True))**2*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                PSIexactL2norm += self.PlasmaCurrent.PSIanalytical(ELEMENT.Xg[ig,:],NORMALISED=True)**2*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                    
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for elem in self.MESH.PlasmaBoundElems:
            # ISOLATE ELEMENT
            ELEMENT = self.MESH.Elements[elem]
            # LOOP OVER SUBELEMENTS
            for SUBELEM in ELEMENT.SubElements:
                # INTEGRATE IN SUBDOMAIN INSIDE PLASMA REGION
                if SUBELEM.Dom < 0:
                    # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PSIg = SUBELEM.Ng @ ELEMENT.PSIe
                    # LOOP OVER GAUSS NODES
                    for ig in range(SUBELEM.ng):
                        ErrorL2norm += (PSIg[ig]-self.PlasmaCurrent.PSIanalytical(SUBELEM.Xg[ig,:],NORMALISED=True))**2*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]
                        PSIexactL2norm += self.PlasmaCurrent.PSIanalytical(SUBELEM.Xg[ig,:],NORMALISED=True)**2*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]                  
        
        self.RelErrorL2norm = np.sqrt(ErrorL2norm/PSIexactL2norm)
        self.ErrorL2norm = np.sqrt(ErrorL2norm)
        return
    
    
    def ComputeL2errorDomain(self):
        """
        Computes the L2 integral norm error of the PSI field by integrating the squared difference between the analytical solution and the 
        computed solution over the full computational domain.
        """
        ErrorL2norm = 0
        PSIexactL2norm = 0
        # INTEGRATE OVER ALL ELEMENTS
        for ELEMENT in self.MESH.Elements:
            # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
            PSIg = ELEMENT.Ng @ ELEMENT.PSIe
            # LOOP OVER GAUSS NODES
            for ig in range(ELEMENT.ng):
                ErrorL2norm += (PSIg[ig]-self.PlasmaCurrent.PSIanalytical(ELEMENT.Xg[ig,:],NORMALISED=True))**2 *ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                PSIexactL2norm += self.PlasmaCurrent.PSIanalytical(ELEMENT.Xg[ig,:],NORMALISED=True)**2 *ELEMENT.detJg[ig]*ELEMENT.Wg[ig]                  
        
        return np.sqrt(ErrorL2norm), np.sqrt(ErrorL2norm/PSIexactL2norm)
    
    
    def ComputeL2errorInterface(self):
        
        ErrorL2norm = 0
        PSIexactL2norm = 0
        # INTEGRATE OVER CUT ELEMENTS' INTERFACE 
        for elem in self.MESH.PlasmaBoundElems:
            # ISOLATE ELEMENTAL INTERFACE APPROXIMATION
            INTAPPROX = self.MESH.Elements[elem].InterfApprox
            # COMPUTE SOLUTION PSI VALUES ON INTERFACE
            PSIg = INTAPPROX.Ng@self.MESH.Elements[elem].PSIe
            # LOOP OVER GAUSS NODES
            for ig in range(INTAPPROX.ng):
                # COMPUTE L2 ERROR
                ErrorL2norm += (PSIg[ig]-self.PlasmaCurrent.PSIanalytical(INTAPPROX.Xg[ig,:]))**2 *INTAPPROX.detJg1D[ig]*INTAPPROX.Wg[ig]
                PSIexactL2norm += self.PlasmaCurrent.PSIanalytical(INTAPPROX.Xg[ig,:])**2 *INTAPPROX.detJg1D[ig]*INTAPPROX.Wg[ig]
        
        if ErrorL2norm == 0:        
            return 0, 0
        else:
            return np.sqrt(ErrorL2norm), np.sqrt(ErrorL2norm/PSIexactL2norm)
    
    
    def ComputeL2errorInterfaceJump(self):
        
        JumpError = np.zeros([self.MESH.NnPB])
        JumpRelError = np.zeros([self.MESH.NnPB])
        ErrorL2norm = 0
        dn = 1e-4
        knode = 0
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for elem in self.MESH.PlasmaBoundElems:
            ELEMENT = self.MESH.Elements[elem]
            # ISOLATE ELEMENTAL INTERFACE APPROXIMATION
            INTAPPROX = ELEMENT.InterfApprox
            # MAP PSI VALUES
            PSIg = INTAPPROX.Ng@ELEMENT.PSIe
            # LOOP OVER GAUSS NODES
            for ig in range(INTAPPROX.ng):
                # OBTAIN GAUSS POINTS SHIFTED IN THE NORMAL DIRECTIONS LEFT AND RIGHT FROM THE ORIGINAL INTERFACE GAUSS NODE
                XIgplus = INTAPPROX.XIg[ig,:] + dn*INTAPPROX.NormalVecREF[ig]
                XIgminus = INTAPPROX.XIg[ig,:] - dn*INTAPPROX.NormalVecREF[ig]
                # EVALUATE GRADIENTS
                Ngplus, dNdxigplus, dNdetagplus = EvaluateReferenceShapeFunctions(XIgplus.reshape((1,2)), ELEMENT.ElType, ELEMENT.ElOrder)
                Ngminus, dNdxigminus, dNdetagminus = EvaluateReferenceShapeFunctions(XIgminus.reshape((1,2)), ELEMENT.ElType, ELEMENT.ElOrder)
                # EVALUATE JACOBIAN
                invJgplus, detJgplus = Jacobian(ELEMENT.Xe,dNdxigplus[0],dNdetagplus[0])
                invJgminus, detJgminus = Jacobian(ELEMENT.Xe,dNdxigminus[0],dNdetagminus[0])
                # COMPUTE PSI VALUES
                PSIgplus = Ngplus@ELEMENT.PSIe
                PSIgminus = Ngminus@ELEMENT.PSIe
                # COMPUTE PHYSICAL GRADIENT
                Ngradplus = invJgplus@np.array([dNdxigplus[0],dNdetagplus[0]])
                Ngradminus = invJgminus@np.array([dNdxigminus[0],dNdetagminus[0]])
                # COMPUTE GRADIENT DIFFERENCE
                diffgrad = 0
                grad = 0
                for inode in range(ELEMENT.n):
                    diffgrad += (Ngradplus[:,inode]*PSIgplus - Ngradminus[:,inode]*PSIgminus)@INTAPPROX.NormalVec[ig]
                    grad += INTAPPROX.NormalVec[ig]@np.array([INTAPPROX.dNdxig[ig,inode],INTAPPROX.dNdetag[ig,inode]])*PSIg[ig]
                JumpError[knode] = diffgrad
                JumpRelError[knode] = diffgrad/abs(grad)
                knode += 1
                # COMPUTE L2 ERROR
                ErrorL2norm += diffgrad**2*INTAPPROX.detJg1D[ig]*INTAPPROX.Wg[ig]
        
        return np.sqrt(ErrorL2norm), JumpError, JumpRelError