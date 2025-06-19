import numpy as np
from Element import *

class EquilipyL2error:

    def __init__(self):
        
        # INITIATE ERROR ARRAYS
        self.PSIerror = None
        self.PSIrelerror = None
        self.ErrorL2norm = None
        self.RelErrorL2norm = None
        self.ErrorL2normPlasmaBound = None
        self.RelErrorL2normPlasmaBound = None 
        self.ErrorL2normINT = None
        self.RelErrorL2normINT = None
        
        super().__init__()
        return
    
    
    def ComputeL2errorPlasma(self):
        """
        Computes the L2 error of the PSI field by integrating the squared difference between the analytical solution and the 
        computed solution over the plasma region.

        Output:
            L2error (float): The computed L2 error value, which measures the difference between the analytical and numerical PSI solutions.
        """
        ErrorL2norm = 0
        PSIexactL2norm = 0
        # INTEGRATE OVER PLASMA ELEMENTS
        for elem in self.Mesh.PlasmaElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Mesh.Elements[elem]
            # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
            PSIg = ELEMENT.Ng @ ELEMENT.PSIe
            # LOOP OVER GAUSS NODES
            for ig in range(ELEMENT.ng):
                ErrorL2norm += (PSIg[ig]-self.PlasmaCurrent.PSIanalytical(ELEMENT.Xg[ig,:],NORMALISED=True))**2*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                PSIexactL2norm += self.PlasmaCurrent.PSIanalytical(ELEMENT.Xg[ig,:],NORMALISED=True)**2*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                    
        ErrorL2normPlasmaBound = 0
        PSIexactL2normPlasmaBound = 0
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for elem in self.Mesh.PlasmaBoundElems:
            # ISOLATE ELEMENT
            ELEMENT = self.Mesh.Elements[elem]
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
                        ErrorL2normPlasmaBound += (PSIg[ig]-self.PlasmaCurrent.PSIanalytical(SUBELEM.Xg[ig,:],NORMALISED=True))**2*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]
                        PSIexactL2normPlasmaBound += self.PlasmaCurrent.PSIanalytical(SUBELEM.Xg[ig,:],NORMALISED=True)**2*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]                  
        
        if ErrorL2normPlasmaBound == 0:
            return np.sqrt(ErrorL2norm), np.sqrt(ErrorL2norm/PSIexactL2norm), 0,0
        else:
            return np.sqrt(ErrorL2norm), np.sqrt(ErrorL2norm/PSIexactL2norm), np.sqrt(ErrorL2normPlasmaBound), np.sqrt(ErrorL2normPlasmaBound/PSIexactL2normPlasmaBound)
    
    
    def ComputeL2error(self):
        """
        Computes the L2 error of the PSI field by integrating the squared difference between the analytical solution and the 
        computed solution over the plasma region.

        Output:
            L2error (float): The computed L2 error value, which measures the difference between the analytical and numerical PSI solutions.
        """
        # COMPUTE STANDARD QUADRATURES FOR PLASMA BOUNDARY ELEMENTS IF NOT ALREADY DONE
        self.ComputePlasmaBoundStandardQuadratures()
        
        ErrorL2norm = 0
        PSIexactL2norm = 0
        # INTEGRATE OVER ALL ELEMENTS
        for ELEMENT in self.Mesh.Elements:
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
        for elem in self.Mesh.PlasmaBoundElems:
            # ISOLATE ELEMENTAL INTERFACE APPROXIMATION
            INTAPPROX = self.Mesh.Elements[elem].InterfApprox
            # COMPUTE SOLUTION PSI VALUES ON INTERFACE
            PSIg = INTAPPROX.Ng@self.Mesh.Elements[elem].PSIe
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
        
        JumpError = np.zeros([self.Mesh.NnPB])
        JumpRelError = np.zeros([self.Mesh.NnPB])
        ErrorL2norm = 0
        dn = 1e-4
        knode = 0
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for elem in self.Mesh.PlasmaBoundElems:
            ELEMENT = self.Mesh.Elements[elem]
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