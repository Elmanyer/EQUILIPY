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


# This script contains the definition for class ELEMENT, an object which 
# embodies the cell elements constituing the mesh. For each ELEMENT object,
# coordinates, shape functions, numerical integration quadratures... data is 
# stored and defines the object. Several elemental methods are also defined 
# inside this class.


from GaussQuadrature import *
from ShapeFunctions import *
from scipy import optimize
from itertools import chain
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from Segment import *
from InterfaceApprox import *

class Element:
    
    ##################################################################################################
    ################################ ELEMENT INITIALISATION ##########################################
    ##################################################################################################
    
    def __init__(self,index,ElType,ElOrder,Xe,Te,PlasmaLSe,interfedge=-1):
        """ 
        Initializes an element object with the specified properties, including its type, order, nodal coordinates, 
        and level-set values for the plasma and vacuum vessel regions. 

        The constructor also calculates the number of nodes and edges based on the element type and order, 
        and sets up necessary attributes for quadrature integration and interface handling.

        Input:
            - index (int): Global index of the element in the computational mesh.
            - ElType (int): Element type identifier:
                        - 0: Segment (1D element)
                        - 1: Triangle (2D element)
                        - 2: Quadrilateral (2D element)
            - ElOrder (int): Element order:
                        - 1: Linear element
                        - 2: Quadratic element
            - Xe (numpy.ndarray): Elemental nodal coordinates in physical space.
            - Te (numpy.ndarray): Element connectivity matrix.
            - PlasmaLSe (numpy.ndarray): Level-set values for the plasma region at each nodal point.
            - VacVessLSe (numpy.ndarray): Level-set values for the vacuum vessel first wall region at each nodal point.
        """
        
        self.index = index                                              # GLOBAL INDEX ON COMPUTATIONAL MESH
        self.ElType = ElType                                            # ELEMENT TYPE -> 0: SEGMENT ;  1: TRIANGLE  ; 2: QUADRILATERAL
        self.ElOrder = ElOrder                                          # ELEMENT ORDER -> 1: LINEAR ELEMENT  ;  2: QUADRATIC
        self.numedges = ElementalNumberOfEdges(ElType)                  # ELEMENTAL NUMBER OF EDGES
        self.n, self.nedge = ElementalNumberOfNodes(ElType, ElOrder)    # NUMBER OF NODES PER ELEMENT, PER ELEMENTAL EDGE
        self.Xe = Xe                                                    # ELEMENTAL NODAL MATRIX (PHYSICAL COORDINATES)
        self.dim = len(Xe[0,:])                                         # SPATIAL DIMENSION
        self.Te = Te                                                    # ELEMENTAL CONNECTIVITIES
        self.LSe = PlasmaLSe                                            # ELEMENTAL NODAL PLASMA REGION LEVEL-SET VALUES
        self.PSIe = np.zeros([self.n])                                  # ELEMENTAL NODAL PSI VALUES
        self.PSI_Be = np.zeros([self.n]) 
        self.Dom = None                                                 # DOMAIN WHERE THE ELEMENT LIES (-1: "PLASMA"; 0: "PLASMA INTERFACE"; +1: "VACUUM" ; +2: FIRST WALL ; +3: "EXTERIOR")
        self.neighbours = None                                          # GLOBAL INDEXES OF NEAREST NEIGHBOURS ELEMENTS CORRESPONDING TO EACH ELEMENTAL FACE (LOCAL INDEX ORDERING FOR FACES)
        self.Teboun = None
        
        # INTEGRATION QUADRATURES ENTITIES
        self.ng = None              # NUMBER OF GAUSS INTEGRATION NODES IN STANDARD 2D GAUSS QUADRATURE
        self.XIg = None             # GAUSS INTEGRATION NODES 
        self.Wg = None              # GAUSS INTEGRATION WEIGTHS
        self.Ng = None              # REFERENCE SHAPE FUNCTIONS EVALUATED AT GAUSS INTEGRATION NODES 
        self.dNdxig = None          # REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT GAUSS INTEGRATION NODES
        self.dNdetag = None         # REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO ETA EVALUATED AT GAUSS INTEGRATION NODES
        self.Xg = None              # PHYSICAL GAUSS INTEGRATION NODES MAPPED FROM 2D REFERENCE ELEMENT
        self.invJg = None           # INVERSE MATRIX OF JACOBIAN OF TRANSFORMATION FROM 2D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT, EVALUATED AT GAUSS INTEGRATION NODES
        self.detJg = None           # MATRIX DETERMINANT OF JACOBIAN OF TRANSFORMATION FROM 2D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT, EVALUATED AT GAUSS INTEGRATION NODES 
        
        ### ATTRIBUTES FOR CUT ELEMENTS
        self.InterfApprox = None    # PLASMA/VACUUM INTERFACE APPROXIMATION ELEMENTAL OBJECT
        self.GhostFaces = None      # LIST OF SEGMENT OBJECTS CORRESPONDING TO ELEMENTAL EDGES WHICH ARE INTEGRATED AS GHOST PENALTY TERMS
        self.Nesub = None           # NUMBER OF SUBELEMENTS GENERATED IN TESSELLATION
        self.SubElements = None
        self.interfedge = interfedge  # LOCAL INDEX OF EDGE CORRESPONDING TO INTERFACE CUT (interfedge = -1  IF NONCUT ELEMENT)
                
        self.area, self.length = self.ComputeArea_Length()
        return
    
    
    def print_all_attributes(self):
        """
        Display all attributes of the object and their corresponding values.

        This method iterates over all attributes of the instance and prints
        them in the format: attribute_name: value.
        """
        for attribute, value in vars(self).items():
            print(f"{attribute}: {value}")
        return
    
    ##################################################################################################
    #################################### ELEMENTAL ATTRIBUTES ########################################
    ##################################################################################################
    
    def ComputeArea(self):
        match self.ElType:
            case 1:
                # REGULAR TRIANGLE
                if self.interfedge == -1:  
                    area = compute_triangle_area(self.Xe)
                # CUT ELEMENT SUBTRIANGLE
                else:
                    Xepoly = np.zeros([3+self.ElOrder-1,2])
                    ipoint = 0
                    for iedge in range(self.nedge):
                        inode = iedge
                        jnode = int((iedge+1)%self.nedge)
                        if iedge == 0:
                            Xepoly[ipoint,:] = self.Xe[inode,:]
                            Xepoly[ipoint+1,:] = self.Xe[jnode,:]
                            ipoint += 2
                        elif iedge == self.interfedge:
                            inodeHO = self.nedge+(self.ElOrder-1)*inode
                            Xepoly[ipoint:ipoint+(self.ElOrder-1),:] = self.Xe[inodeHO:inodeHO+(self.ElOrder-1),:]
                            ipoint += self.ElOrder-1
                        else:
                            Xepoly[ipoint,:] = self.Xe[inode,:]
                            ipoint += 1
                            
                    area = polygon_area(Xepoly,self.ElOrder,self.interfedge)

            case 2:
                area = compute_quadrilateral_area(self.Xe)
                
        return area
    
    def ComputeArea_Length(self):  
        area = self.ComputeArea()
        match self.ElType:
            case 1:
                length = np.sqrt(4*area/np.sqrt(3)) 
            case 2:
                length = np.sqrt(area)
        return area, length
    
    
    def isinside(self,X):
        inside = False
        if self.ElType == 1: # FOR TRIANGULAR ELEMENTS
            # Calculate the cross products (c1, c2, c3) for the point relative to each edge of the triangle
            c1 = (self.Xe[1,0]-self.Xe[0,0])*(X[1]-self.Xe[0,1])-(self.Xe[1,1]-self.Xe[0,1])*(X[0]-self.Xe[0,0])
            c2 = (self.Xe[2,0]-self.Xe[1,0])*(X[1]-self.Xe[1,1])-(self.Xe[2,1]-self.Xe[1,1])*(X[0]-self.Xe[1,0])
            c3 = (self.Xe[0,0]-self.Xe[2,0])*(X[1]-self.Xe[2,1])-(self.Xe[0,1]-self.Xe[2,1])*(X[0]-self.Xe[2,0])
            if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0): # INSIDE TRIANGLE
                inside = True
        elif self.ElType == 2: # FOR QUADRILATERAL ELEMENTS
            # This algorithm counts how many times a ray starting from the point intersects the edges of the quadrilateral. 
            # If the count is odd, the point is inside; otherwise, it is outside.
            for i in range(4):
                if ((self.Xe[i,1] > X[1]) != (self.Xe[(i+1)%4,1]>X[1])) and (X[0]<(self.Xe[(i+1)%4,0]-self.Xe[i,0])*(X[1]-self.Xe[i,1])/(self.Xe[(i+1)%4,1]-self.Xe[i,1])+self.Xe[i,0]):
                    inside = not inside
        return inside
    
    
    def CheckElementalVerticesLevelSetSigns(self):
        
        region = None
        DiffHighOrderNodes = []
        # CHECK SIGN OF LEVEL SET ON ELEMENTAL VERTICES
        for i in range(self.numedges-1):
            # FIND ELEMENTS LYING ON THE INTERFACE (LEVEL-SET VERTICES VALUES EQUAL TO 0 OR WITH DIFFERENT SIGN)
            if self.LSe[i] == 0:  # if node is on Level-Set 0 contour
                region = 0
                break
            elif np.sign(self.LSe[i]) !=  np.sign(self.LSe[i+1]):  # if the sign between vertices values change -> INTERFACE ELEMENT
                region = 0
                break
            # FIND ELEMENTS LYING INSIDE A SPECIFIC REGION (LEVEL-SET VERTICES VALUES WITH SAME SIGN)
            else:
                if i+2 == self.numedges:   # if all vertices values have the same sign
                    # LOCATE ON WHICH REGION LIES THE ELEMENT
                    if np.sign(self.LSe[i+1]) > 0:   # all vertices values with positive sign -> EXTERIOR REGION ELEMENT
                        region = +1
                    else:   # all vertices values with negative sign -> INTERIOR REGION ELEMENT 
                        region = -1
                        
                    # CHECK LEVEL-SET SIGN ON ELEMENTAL 'HIGH ORDER' NODES
                    #for i in range(self.numedges,self.n-self.numedges):  # LOOP OVER NODES WHICH ARE NOT ON VERTICES
                    for i in range(self.numedges,self.n):
                        if np.sign(self.LSe[i]) != np.sign(self.LSe[0]):
                            DiffHighOrderNodes.append(i)
            
        return region, DiffHighOrderNodes
    
    
    ##################################################################################################
    #################################### ELEMENTAL MAPPING ###########################################
    ##################################################################################################
    
    
    def Mapping(self,Xi):
        """ 
        This function implements the mapping corresponding to the transformation from natural to physical coordinates. 
        That is, given a point in the reference element with coordinates Xi, this function returns the coordinates X of the corresponding point mapped
        in the physical element with nodal coordinates Xe. 
        In order to do that, we solve the nonlinear system implicitly araising from the original isoparametric equations. 
        
        Input: 
            - Xg: coordinates of point in reference space for which to compute the coordinate in physical space.
            - Xe: nodal coordinates of physical element.
        Output: 
             X: coodinates of mapped point in reference element.
        """
        X = np.zeros([self.dim])
        for i in range(self.n):
            Nig, foo, foo = ShapeFunctionsReference(Xi, self.ElType, self.ElOrder, i+1)
            X += Nig*self.Xe[i,:]
        return X
    
    def InverseMapping(self, X):
        """ 
        This function implements the inverse mapping corresponding to the transformation from natural to physical coordinates (thus, for the inverse transformation
        we go from physical to natural coordinates). That is, given a point in physical space with coordinates X in the element with nodal coordinates Xe, 
        this function returns the point mapped in the reference element with natural coordinates Xi. 
        In order to do that, we solve the nonlinear system implicitly araising from the original isoparametric equations. 
        
        Input: 
            X: physical coordinates of point for which compute the corresponding point in the reference space.
        Output: 
            Xg: coodinates of mapped point in reference element.
        """
        
        # DEFINE THE NONLINEAR SYSTEM 
        def fun(Xi, X, Xe):
            f = np.array([-X[0],-X[1]])
            for i in range(self.n):
                Nig, foo, foo = ShapeFunctionsReference(Xi, self.ElType, self.ElOrder, i+1)
                f[0] += Nig*Xe[i,0]
                f[1] += Nig*Xe[i,1]
            return f
        # SOLVE NONLINEAR SYSTEM
        Xi0 = np.array([1/2, 1/2])  # INITIAL GUESS FOR ROOT SOLVER
        sol = optimize.root(fun, Xi0, args=(X,self.Xe))
        Xi = sol.x
        return Xi
    
    def ElementalInterpolationREFERENCE(self,XI,Fe):
        """ 
        Interpolate field F on REFERENCE element with nodal values Fe on point XI using elemental shape functions. 
        """
        F = 0
        for i in range(self.n):
            N, foo, foo = ShapeFunctionsReference(XI, self.ElType, self.ElOrder, i+1)
            F += N*Fe[i]
        return F
    
    def GRADElementalInterpolationREFERENCE(self,XI,Fe):
        """ 
        Interpolate gradient field dF on REFERENCE element with nodal values Fe on point XI using elemental shape functions derivatives. 
        """
        dF = np.zeros([self.dim])
        for i in range(self.n):
            foo, dNdxi, dNdeta = ShapeFunctionsReference(XI, self.ElType, self.ElOrder, i+1)
            dF += np.array([dNdxi,dNdeta])*Fe[i]
        return dF
    
    def ElementalInterpolationPHYSICAL(self,X,Fe):
        """ 
        Interpolate field F with nodal values Fe on point X using elemental shape functions. 
        """
        XI = self.InverseMapping(X)
        return self.ElementalInterpolationREFERENCE(XI,Fe)
    
    def GRADElementalInterpolationPHYSICAL(self,X,Fe):
        """ 
        Interpolate gradient field dF with nodal values Fe on point X using elemental shape functions derivatives. 
        """
        XI = self.InverseMapping(X)
        return self.GRADElementalInterpolationREFERENCE(XI,Fe)
    
    
    ##################################################################################################
    ##################################### MAGNETIC FIELD B ###########################################
    ##################################################################################################
    
    def Br(self,X):
        """
        Total radial magnetic field at point X such that    Br = -1/R dpsi/dZ
        """
        # MAP PHYSICAL POINT TO REFERENCE ELEMENT
        XI = self.InverseMapping(X)
        # COMPUTE REFERENCE SHAPE FUNCTIONS DERIVATIVES AT MAPPED POINT
        foo, dNdxi, dNdeta = EvaluateReferenceShapeFunctions(XI.reshape((1,2)), self.ElType, self.ElOrder)
        # EVALUATE JACOBIAN OF TRANSFORMATION AT POINT
        invJ, foo = Jacobian(self.Xe,dNdxi[0],dNdeta[0])
        # OBTAIN GRADIENT IN PHYSICAL SPACE 
        Ngrad = invJ@np.array([dNdxi[0], dNdeta[0]])
        # COMPUTE RADIAL MAGNETIC COMPONENT
        Br = - Ngrad[1,:]@self.PSIe/X[0]
        return Br
    
    def Brg(self):
        """
        Total radial magnetic field at integration nodes such that    Br = -1/R dpsi/dZ
        """
        Brg = np.zeros([self.ng])
        # LOOP OVER INTEGRATION NODES
        for ig in range(self.ng):
            # COMPUTE GRADIENT IN PHYSICAL SPACE
            Ngrad = self.invJg[ig,:,:]@np.array([self.dNdxig[ig,:],self.dNdetag[ig,:]])
            # COMPUTE RADIAL MAGNETIC COMPONENT
            Brg[ig] = - Ngrad[1,:]@self.PSIe/self.Xg[ig,0]
        return Brg
    
    
    def Bz(self,X):
        """
        Total vertical magnetic field at point X such that    Bz = 1/R dpsi/dR
        """
        # MAP PHYSICAL POINT TO REFERENCE ELEMENT
        XI = self.InverseMapping(X)
        # COMPUTE REFERENCE SHAPE FUNCTIONS DERIVATIVES AT MAPPED POINT
        foo, dNdxi, dNdeta = EvaluateReferenceShapeFunctions(XI.reshape((1,2)), self.ElType, self.ElOrder)
        # EVALUATE JACOBIAN OF TRANSFORMATION AT POINT
        invJ, foo = Jacobian(self.Xe,dNdxi[0],dNdeta[0])
        # OBTAIN GRADIENT IN PHYSICAL SPACE 
        Ngrad = invJ@np.array([dNdxi[0], dNdeta[0]])
        # COMPUTE VERTICAL MAGNETIC COMPONENT
        Bz = Ngrad[0,:]@self.PSIe/X[0]
        return Bz
    
    def Bzg(self):
        """
        Total radial magnetic field at integration nodes such that     Bz = 1/R dpsi/dR
        """
        Bzg = np.zeros([self.ng])
        # LOOP OVER INTEGRATION NODES
        for ig in range(self.ng):
            # COMPUTE GRADIENT IN PHYSICAL SPACE
            Ngrad = self.invJg[ig,:,:]@np.array([self.dNdxig[ig,:],self.dNdetag[ig,:]])
            # COMPUTE VERTICAL MAGNETIC COMPONENT
            Bzg[ig] = Ngrad[0,:]@self.PSIe/self.Xg[ig,0]
        return Bzg
    
    def Brz(self,X):
        """
        Total magnetic field vector at point X such that    (Br, Bz) = (-1/R dpsi/dZ, 1/R dpsi/dR)
        """
        # MAP PHYSICAL POINT TO REFERENCE ELEMENT
        XI = self.InverseMapping(X)
        # COMPUTE REFERENCE SHAPE FUNCTIONS DERIVATIVES AT MAPPED POINT
        foo, dNdxi, dNdeta = EvaluateReferenceShapeFunctions(XI.reshape((1,2)), self.ElType, self.ElOrder)
        # EVALUATE JACOBIAN OF TRANSFORMATION AT POINT
        invJ, foo = Jacobian(self.Xe,dNdxi[0],dNdeta[0])
        # OBTAIN GRADIENT IN PHYSICAL SPACE  
        Ngrad = invJ@np.array([dNdxi[0], dNdeta[0]])
        # COMPUTE MAGNETIC VECTOR
        Brz = Ngrad[[1,0],:]@self.PSIe/X[0]
        Brz[0] *= -1
        return Brz
    
    def Brzg(self):
        """
        Total radial magnetic field at integration nodes such that    Br = -1/R dpsi/dZ
        """
        Brzg = np.zeros([self.ng,self.dim])
        # LOOP OVER INTEGRATION NODES
        for ig in range(self.ng):
            # COMPUTE GRADIENT IN PHYSICAL SPACE
            Ngrad = self.invJg[ig,:,:]@np.array([self.dNdxig[ig,:],self.dNdetag[ig,:]])
            # COMPUTE MAGNETIC VECTOR
            Brzg[ig,:] = Ngrad[[1,0],:]@self.PSIe/self.Xg[ig,0]
            Brzg[0] *= -1
        return Brzg
    
    ##################################################################################################
    ######################### CUT ELEMENTS INTERFACE APPROXIMATION ###################################
    ##################################################################################################
    
    def PHI(self,X):
        """ ISOPARAMETRIC INTERPOLATION OF LEVEL-SET FUNCTION PHI EVALUATED AT POINT X"""
        N, foo, foo = EvaluateReferenceShapeFunctions(X, self.ElType, self.ElOrder)
        return N@self.LSe
    
    
    def InterfaceApproximation(self,interface_index):
        """
        Approximates the interface between plasma and vacuum regions by computing the intersection points 
        of the plasma/vacuum boundary with the edges and interior of the element.

        The function performs the following steps:
            1. Reads the level-set nodal values
            2. Computes the coordinates of the reference element.
            3. Identifies the intersection points of the interface with the edges of the REFERENCE element.
            4. Uses interpolation to approximate the interface inside the REFERENCE element, including high-order interior nodes.
            5. Maps the interface approximation back to PHYSICAL space using shape functions.
            6. Associates elemental connectivity to interface segments.
            7. Generates segment objects for each segment of the interface and computes high-order segment nodes.

        Input:
            interface_index (int): The index of the interface to be approximated.
        """    
        
        # OBTAIN REFERENCE ELEMENT COORDINATES
        XIe = ReferenceElementCoordinates(self.ElType,self.ElOrder)

        # FIND POINTS ON INTERFACE USING ELEMENTAL INTERPOLATION
        #### INTERSECTION WITH EDGES
        XIintEND = np.zeros([2,2])
        ElIntNodes = np.zeros([2,self.nedge],dtype=int)
        k = 0
        for iedge in range(self.numedges):  # Loop over elemental edges
            # Check for sign change along the edge
            inode = iedge
            jnode = (iedge + 1) % self.numedges
            if self.LSe[inode] * self.LSe[jnode] < 0:
                # FIND HIGH-ORDER NODES BETWEEN VERTICES
                #edge_index = get_edge_index(self.ElType,inode,jnode)
                ElIntNodes[k,:2] = [inode, jnode]
                for knode in range(self.ElOrder-1):
                    ElIntNodes[k,2+knode] = self.numedges + iedge*(self.ElOrder-1) + knode
                    
                if abs(XIe[jnode,0]-XIe[inode,0]) < 1e-6: # VERTICAL EDGE
                    #### DEFINE CONSTRAINT PHI FUNCTION
                    xi = XIe[inode,0]
                    def PHIedge(eta):
                        X = np.array([xi,eta[0]],dtype=float).reshape((1,2))
                        return self.PHI(X)
                    #### FIND INTERSECTION POINT:
                    Eta0 = 1/2  # INITIAL GUESS FOR ROOT SOLVER
                    sol = optimize.root(PHIedge, Eta0)
                    XIintEND[k,:] = [xi, sol.x[0]]
                else:
                    def edgeconstraint(xi):
                        # FUNCTION DEFINING THE CONSTRAINT ON THE ELEMENTAL EDGE
                        m = (XIe[jnode,1]-XIe[inode,1])/(XIe[jnode,0]-XIe[inode,0])
                        eta = m*(xi-XIe[inode,0])+XIe[inode,1]
                        return eta
                    def PHIedge(xi):
                        X = np.array([xi,edgeconstraint(xi)]).reshape((1,2))
                        return self.PHI(X)
                    #### FIND INTERSECTION POINT:
                    Xi0 = 1/2  # INITIAL GUESS FOR ROOT SOLVER
                    sol = optimize.root(PHIedge, Xi0)
                    XIintEND[k,:] = [sol.x[0], edgeconstraint(sol.x[0])]
                k += 1
                    
        if self.ElOrder == 1: # LINEAR ELEMENT INTERFACE APPROXIMATION -> LINEAR APPROXIMATION 
            XIint = XIintEND 
        else:
            #### HIGH-ORDER INTERFACE NODES
            # IN THIS CASE, WITH USE THE REGULARITY OF THE REFERENCE TRIANGLE TO FIND THE NODES
            # LYING ON THE INTERFACE INSIDE THE ELEMENT. SIMILARLY TO THE INTERSECTION NODES ON THE
            # ELEMENTAL EDGES, EACH INTERIOR NODE CAN BE FOUND BY IMPOSING TWO CONDITIONS:
            #    - PHI = 0
            #    - NODE ON LINE DIVIDING THE INTERFACE ARC

            def fun(X):
                F = np.zeros([X.shape[0]])
                # SEPARATE GUESS VECTOR INTO INDIVIDUAL NODAL COORDINATES
                XHO = X.reshape((self.ElOrder-1,self.dim)) 
                # PHI = 0 ON NODES
                for inode in range(self.ElOrder-1):
                    F[inode] = self.PHI(XHO[inode,:].reshape((1,2)))
                # EQUAL DISTANCES BETWEEN INTERFACE NODES
                if self.ElOrder == 2:
                    F[-1] = np.linalg.norm(XIintEND[0,:]-X)-np.linalg.norm(XIintEND[1,:]-X)
                if self.ElOrder == 3:
                    #### FIRST INTERVAL
                    F[self.ElOrder-1] = np.linalg.norm(XIintEND[0,:]-XHO[0,:])-np.linalg.norm(XHO[0,:]-XHO[1,:])
                    #### LAST INTERVAL
                    F[-1] = np.linalg.norm(XIintEND[1,:]-XHO[-1,:])-np.linalg.norm(XHO[-1,:]-XHO[-2,:])
                #### INTERIOR INTERVALS
                if self.ElOrder > 3:
                    for intv in range(self.ElOrder-3):
                        F[self.ElOrder+intv] = np.linalg.norm(XHO[intv+1,:]-XHO[intv+2,:]) - np.linalg.norm(XHO[intv+2,:]-XHO[intv+3,:])
                return F

            # PREPARE INITIAL GUESS
            X0 = np.zeros([(self.ElOrder-1)*2])
            for inode in range(1,self.ElOrder):
                X0[2*(inode-1):2*inode] = XIintEND[0,:] + np.array([(XIintEND[1,0]-XIintEND[0,0]),(XIintEND[1,1]-XIintEND[0,1])])*inode/self.ElOrder
            X0 = X0.reshape((1,(self.ElOrder-1)*2))
            # COMPUTE HIGH-ORDER INTERFACE NODES COORDINATES
            sol = optimize.root(fun, X0)
            # STORE SOLUTION NODES
            XIintINT = np.zeros([self.ElOrder-1,2])
            for inode in range(self.ElOrder-1):
                XIintINT[inode,:] = np.reshape(sol.x, (self.ElOrder-1,2))[inode,:]

            ##### STORE INTERFACE APPROXIMATION DATA IN INTERFACE OBJECT 
            ## CONCATENATE INTERFACE NODES
            XIint = np.concatenate((XIintEND,XIintINT),axis=0)
        
        
        ## MAP BACK TO PHYSICAL SPACE
        # EVALUATE REFERENCE SHAPE FUNCTIONS AT POINTS TO MAP (INTERFACE NODES)
        Nint, foo, foo = EvaluateReferenceShapeFunctions(XIint, self.ElType, self.ElOrder)
        # COMPUTE SCALAR PRODUCT
        Xint = Nint@self.Xe
        
        ## ASSOCIATE ELEMENTAL CONNECTIVITY TO INTERFACE SEGMENTS
        lnods = [0,np.arange(2,self.ElOrder+1),1]
        Tint = list(chain.from_iterable([x] if not isinstance(x, np.ndarray) else x for x in lnods))
        
        # GENERATE ELEMENTAL PLASMA/VACUUM INTERFACE APPROXIMATION OBJECT
        self.InterfApprox = InterfaceApprox(index = interface_index,
                                            n = self.nedge,
                                            Xint = Xint,
                                            XIint = XIint,
                                            Tint = Tint,
                                            ElIntNodes = ElIntNodes)     
        return 
    
    
    ##################################################################################################
    ##################################### INTERFACE NORMALS ##########################################
    ##################################################################################################
    
    def InterfaceNormals(self):
        """ 
        This function computes the interface normal vector pointing outwards at the Gaussian integration nodes. 
        """
        # COMPUTE THE NORMAL VECTOR FOR EACH SEGMENT CONFORMING THE INTERFACE APPROXIMATION
        self.InterfApprox.NormalVec = list()
        self.InterfApprox.NormalVecREF = list()
        for ig in range(self.InterfApprox.ng):
            #### PREPARE NORMAL VECTOR IN PHYSICAL SPACE
            Ngrad = self.InterfApprox.invJg[ig,:,:]@np.array([self.InterfApprox.dNdxig[ig,:],self.InterfApprox.dNdetag[ig,:]])
            dphidr, dphidz = Ngrad@self.LSe
            ntest_rz = np.array([dphidr,dphidz])
            ntest_rz = ntest_rz/np.linalg.norm(ntest_rz)
            #### PERFORM THE TEST IN REFERENCE SPACE
            # COMPUTE DERIVATIVES OF INTERPOLATED PHI 
            dphidxi = self.InterfApprox.dNdxig[ig,:]@self.LSe
            dphideta = self.InterfApprox.dNdetag[ig,:]@self.LSe
            # PREPARE TEST NORMAL VECTOR 
            ntest_xieta = np.array([dphidxi, dphideta])
            ntest_xieta = ntest_xieta/np.linalg.norm(ntest_xieta)    # normalize
            # PREPARE TEST POINT              
            XItest = self.InterfApprox.XIg[ig,:] + 0.5*ntest_xieta   # point on which to test the Level-Set 
            # INTERPOLATE LEVEL-SET ON TEST POINT
            LStest = self.ElementalInterpolationREFERENCE(XItest,self.LSe)
            # CHECK SIGN OF LEVEL-SET 
            if LStest > 0:  # TEST POINT OUTSIDE PLASMA REGION
                self.InterfApprox.NormalVec.append(ntest_rz)
                self.InterfApprox.NormalVecREF.append(ntest_xieta)
            else:   # TEST POINT INSIDE PLASMA REGION --> NEED TO TAKE THE OPPOSITE NORMAL VECTOR
                self.InterfApprox.NormalVec.append(-1*ntest_rz)
                self.InterfApprox.NormalVecREF.append(-1*ntest_xieta)
        return 
    
    
    def CheckInterfaceNormals(self):
        
        for ig, vec in enumerate(self.InterfApprox.NormalVec):
            # CHECK UNIT LENGTH
            if np.abs(np.linalg.norm(vec)-1) > 1e-6:
                raise Exception('Normal vector norm equals',np.linalg.norm(vec), 'for mesh element', self.index, ": Normal vector not unitary")
            # CHECK ORTHOGONALITY
            Ngrad = self.InterfApprox.invJg[ig,:,:]@np.array([self.InterfApprox.dNdxig[ig,:],self.InterfApprox.dNdetag[ig,:]])
            dphidr, dphidz = Ngrad@self.LSe
            tangvec = np.array([-dphidz, dphidr]) 
            scalarprod = np.dot(tangvec,vec)
            if scalarprod > 1e-10: 
                raise Exception('Dot product equals',scalarprod, 'for mesh element', self.index, ": Normal vector not perpendicular")
        return
     
    
    def GhostFacesNormals(self):
        
        for FACE in self.GhostFaces:
            #### PREPARE TEST NORMAL VECTOR IN PHYSICAL SPACE
            dx = FACE.Xseg[1,0] - FACE.Xseg[0,0]
            dy = FACE.Xseg[1,1] - FACE.Xseg[0,1]
            ntest = np.array([-dy, dx]) 
            ntest = ntest/np.linalg.norm(ntest) 
            Xsegmean = np.mean(FACE.Xseg,axis=0)
            dl = min((max(self.Xe[:self.numedges,0])-min(self.Xe[:self.numedges,0])),(max(self.Xe[:self.numedges,1])-min(self.Xe[:self.numedges,1])))
            dl *= 0.1
            Xtest = Xsegmean + dl*ntest 
            
            #### TEST IF POINT Xtest LIES INSIDE TRIANGLE ELEMENT
            # Create a Path object for element
            polygon_path = mpath.Path(np.concatenate((self.Xe[:self.numedges,:],self.Xe[0,:].reshape(1,self.dim)),axis=0))
            # Check if Xtest is inside the element
            inside = polygon_path.contains_points(Xtest.reshape(1,self.dim))
                
            if not inside:  # TEST POINT OUTSIDE ELEMENT
                FACE.NormalVec = ntest
            else:   # TEST POINT INSIDE ELEMENT --> NEED TO TAKE THE OPPOSITE NORMAL VECTOR
                FACE.NormalVec = -1*ntest
                
        return
    
    ##################################################################################################
    ################################ ELEMENTAL TESSELLATION ##########################################
    ##################################################################################################
        
    @staticmethod
    def HO_TRI_interf(XeLIN,ElOrder,XintHO,interfedge):
        """
        Generates a high-order triangular element from a linear one with nodal vertices coordinates XeLIN, incorporating high-order 
        nodes on the edges and interior, and adapting if necessary one of the edges to the interface high-order approximation.

        This function performs the following steps:
            1. Extends the input linear (low-order) element coordinates with high-order nodes on the edges.
            2. Adds interface high-order nodes if necessary on the edge indicated by `interfedge`. 
            3. For triangular elements with an order of 3 or higher, adds an interior high-order node at 
                the centroid of the element.

        Input: 
            - XeLIN (numpy.ndarray): An array of shape (n, 2) containing the coordinates of the linear (low-order) element nodes.
            - ElOrder (int): The order of the element, determining the number of high-order nodes to be added.
            - XintHO (numpy.ndarray): An array containing the high-order interface nodes (interface points) to be inserted along 
                the specified edge.
            - interfedge (int): The edge index where the interface high-order nodes should be inserted.

        Output: 
            XeHO (numpy.ndarray): An array containing the coordinates of the high-order element nodes, including those on 
                the edges and interior.
        """
        nedge = len(XeLIN[:,0])
        XeHO = XeLIN.copy()
        # MAKE IT HIGH-ORDER:
        for iedge in range(nedge):
            # EDGE HIGH-ORDER NODES
            if interfedge == iedge:
                XeHO = np.concatenate((XeHO,XintHO[2:,:]), axis=0)
            else:
                inode = iedge
                jnode = (iedge+1)%nedge
                for k in range(1,ElOrder):
                    HOnode = np.array([XeLIN[inode,0]+((XeLIN[jnode,0]-XeLIN[inode,0])/ElOrder)*k,XeLIN[inode,1]+((XeLIN[jnode,1]-XeLIN[inode,1])/ElOrder)*k])
                    XeHO = np.concatenate((XeHO,HOnode.reshape((1,2))), axis=0)
        # INTERIOR HIGH-ORDER NODES:
        if ElOrder == 3:
            HOnode = np.array([np.mean(XeHO[:,0]),np.mean(XeHO[:,1])])
            XeHO = np.concatenate((XeHO,HOnode.reshape((1,2))), axis=0)
        return XeHO

    @staticmethod
    def HO_QUA_interf(XeLIN,ElOrder,XintHO,interfedge):
        """
        Generates a high-order quadrilateral element from a linear one with nodal vertices coordinates XeLIN, incorporating high-order 
        nodes on the edges and interior, and adapting if necessary one of the edges to the interface high-order approximation.

        This function performs the following steps:
            1. Extends the input linear (low-order) element coordinates with high-order nodes on the edges.
            2. Adds interface high-order nodes if necessary on the edge indicated by `interfedge`. 
            3. For quadrilateral elements of order 2, adds an interior high-order node at the centroid of the element.
            3. For quadrilateral elements of order 3, adds an interior high-order nodes.

        Input: 
            - XeLIN (numpy.ndarray): An array of shape (n, 2) containing the coordinates of the linear (low-order) element nodes.
            - ElOrder (int): The order of the element, determining the number of high-order nodes to be added.
            - XintHO (numpy.ndarray): An array containing the high-order interface nodes (interface points) to be inserted along 
                the specified edge.
            - interfedge (int): The edge index where the interface high-order nodes should be inserted.

        Output: 
            XeHO (numpy.ndarray): An array containing the coordinates of the high-order element nodes, including those on 
                the edges and interior.
        """
        nedge = len(XeLIN[:,0])
        XeHO = XeLIN.copy()
        for iedge in range(nedge):
            # EDGE HIGH-ORDER NODES
            if interfedge == iedge:
                XeHO = np.concatenate((XeHO,XintHO[2:,:]), axis=0)
            else:
                inode = iedge
                jnode = (iedge+1)%nedge
                for k in range(1,ElOrder):
                    HOnode = np.array([XeLIN[inode,0]+((XeLIN[jnode,0]-XeLIN[inode,0])/ElOrder)*k,XeLIN[inode,1]+((XeLIN[jnode,1]-XeLIN[inode,1])/ElOrder)*k])
                    XeHO = np.concatenate((XeHO,HOnode.reshape((1,2))), axis=0)
        # INTERIOR HIGH-ORDER NODES:
        if ElOrder == 2:
            HOnode = np.array([np.mean(XeHO[:,0]),np.mean(XeHO[:,1])])
            XeHO = np.concatenate((XeHO,HOnode.reshape((1,2))), axis=0)
        elif ElOrder == 3:
            for k in range(1,ElOrder):
                dx = (XeHO[12-k,0]-XeHO[5+k,0])/ElOrder
                dy = (XeHO[12-k,1]-XeHO[5+k,1])/ElOrder
                for j in range(1,ElOrder):
                    if k == 1:
                        HOnode = XeHO[11,:] - np.array([dx*j,dy*j])
                    elif k == 2:
                        HOnode = XeHO[7,:] + np.array([dx*j,dy*j])
                    XeHO = np.concatenate((XeHO,HOnode.reshape((1,2))), axis=0)
        return XeHO


    def ReferenceElementTessellation(self):
        """ 
        This function performs the TESSELLATION of a HIGH-ORDER REFERENCE ELEMENT with interface nodal coordinates XIeintHO
        
        Output: XIeTESSHO: High-order subelemental nodal coordinates matrix for each child element generated in the tessellation,
                            such that:
                                        XIeTESSHO = [[[ xi00, eta00 ],
                                                        [ xi01, eta01 ],      NODAL COORDINATE MATRIX
                                                            ....    ],         FOR SUBELEMENT 0
                                                        [ xi0n, eta0n ]],
                                                        
                                                        [[ xi10, eta10 ],
                                                        [ xi11, eta11 ],      NODAL COORDINATE MATRIX
                                                            ....    ],         FOR SUBELEMENT 1
                                                        [ xi1n, eta1n ]],
                                                        
                                                            ....    ]
        """
        # FIRST WE NEED TO DETERMINE WHICH IS THE VERTEX COMMON TO BOTH EDGES INTERSECTING WITH THE INTERFACE
        # AND ORGANISE THE NODAL MATRIX ACCORDINGLY SO THAT
        #       - THE FIRST ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH IS SHARED BY BOTH EDGES INTERSECTING THE INTERFACE 
        #       - THE SECOND ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH DEFINES THE EDGE ON WHICH THE FIRST INTERSECTION POINT IS LOCATED
        #       - THE THIRD ROW CORRESPONDS TO THE VERTEX COORDINATES WHICH DEFINES THE EDGE ON WHICH THE SECOND INTERSECTION POINT IS LOCATED
        # HOWEVER, WHEN LOOKING FOR THE APPROXIMATION OF THE PHYSICAL INTERFACE THIS PROCESS IS ALREADY DONE, THEREFORE WE CAN SKIP IT. 
        # IF INPUT Xemod IS PROVIDED, THE TESSELLATION IS DONE ACCORDINGLY TO ADAPTED NODAL MATRIX Xemod WHICH IS ASSUMED TO HAS THE PREVIOUSLY DESCRIBED STRUCTURE.
        # IF NOT, THE COMMON NODE IS DETERMINED (THIS IS THE CASE FOR INSTANCE WHEN THE REFERENCE ELEMENT IS TESSELLATED).

        XIeLIN = ReferenceElementCoordinates(self.ElType,1)
        edgenodes = self.InterfApprox.ElIntNodes[:,:2]

        if self.ElType == 1:  # TRIANGULAR ELEMENT
            Nesub = 3
            SubElType = 1
            distance = np.zeros([2])
            edgenode = np.zeros([2],dtype=int)
            commonnode = (set(edgenodes[0,:])&set(edgenodes[1,:])).pop() # COMMON NODE TO INTERSECTED EDGES
            # LOOK FOR NODE ON EDGE WHERE INTERSECTION POINT LIES BUT OTHER THAN COMMON NODE AND COMPUTE DISTANCE
            for i in range(2):
                edgenodeset = set(edgenodes[i,:])
                edgenodeset.remove(commonnode)
                edgenode[i] = edgenodeset.pop()
                distance[i] = np.linalg.norm(self.InterfApprox.XIint[i,:]-XIeLIN[edgenode[i],:])
            
            XIeTESSLIN = list()
            interfedge = [1,1,-1]
            XIeTESSLIN.append(np.concatenate((XIeLIN[int(commonnode),:].reshape((1,2)),self.InterfApprox.XIint[:2,:]),axis=0))
            if distance[0] < distance[1]:
                XIeTESSLIN.append(np.concatenate((XIeLIN[edgenode[1],:].reshape((1,2)),self.InterfApprox.XIint[:2,:]),axis=0))
                XIeTESSLIN.append(np.concatenate((XIeLIN[[edgenode[0],edgenode[1]],:],self.InterfApprox.XIint[0,:].reshape((1,2))),axis=0))
            else:
                XIeTESSLIN.append(np.concatenate((XIeLIN[edgenode[0],:].reshape((1,2)),self.InterfApprox.XIint[:2,:]),axis=0))
                XIeTESSLIN.append(np.concatenate((XIeLIN[[edgenode[1],edgenode[0]],:],self.InterfApprox.XIint[1,:].reshape((1,2))),axis=0))
            
            # TURN LINEAR SUBELEMENTS INTO HIGH-ORDER SUBELEMENTS
            XIeTESSHO = list()
            for isub in range(Nesub):
                XIeHO = self.HO_TRI_interf(XIeTESSLIN[isub],self.ElOrder,self.InterfApprox.XIint,interfedge[isub])
                XIeTESSHO.append(XIeHO)
                
            
        elif self.ElType == 2:  # QUADRILATERAL ELEMENT
            # LOOK FOR TESSELLATION CONFIGURATION BY USING SIGN OF prod(LSe)
                    #  -> IF prod(LSe) > 0, THEN CUT SPLITS PARENT QUADRILATERAL ELEMENT INTO 2 CHILD QUADRILATERAL ELEMENTS
                    #  -> IF prod(LSe) < 0, THEN CUT SPLITS PARENT QUADRILATERAL ELEMENT INTO PENTAGON AND TRIANGLE -> PENTAGON IS SUBDIVIDED INTO 3 SUBTRIANGLES
            
            if np.prod(self.LSe[:self.numedges]) > 0:  # 2 SUBQUADRILATERALS
                Nesub = 2
                SubElType = 2
            
                interfedge = [1,1]
                XIeTESSLIN = list()
                XIeTESSLIN.append(np.concatenate((XIeLIN[edgenodes[0,0],:].reshape((1,2)),self.InterfApprox.XIint[:2,:],XIeLIN[edgenodes[1,1],:].reshape((1,2))),axis=0))
                XIeTESSLIN.append(np.concatenate((XIeLIN[edgenodes[0,1],:].reshape((1,2)),self.InterfApprox.XIint[:2,:],XIeLIN[edgenodes[1,0],:].reshape((1,2))),axis=0))
                
                # TURN LINEAR TRIANGULAR SUBELEMENTS INTO HIGH-ORDER TRIANGULAR SUBELEMENTS
                XIeTESSHO = list()
                for isub in range(Nesub):
                    XIeHO = self.HO_QUA_interf(XIeTESSLIN[isub],self.ElOrder,self.InterfApprox.XIint,interfedge[isub])
                    XIeTESSHO.append(XIeHO)
                
            else:  # 4 SUBTRIANGLES
                Nesub = 4
                SubElType = 1
                # LOOK FOR COMMON NODE
                edgenode = np.zeros([2],dtype=int)
                distance = np.zeros([2])
                commonnode = (set(edgenodes[0,:])&set(edgenodes[1,:])).pop()
                # LOOK FOR NODE ON EDGE WHERE INTERSECTION POINT LIES BUT OTHER THAN COMMON NODE
                for i in range(2):
                    edgenodeset = set(edgenodes[i,:])
                    edgenodeset.remove(commonnode)
                    edgenode[i] = edgenodeset.pop()
                # LOOK FOR OPPOSITE NODE
                for i in range(4):  # LOOP OVER VERTEX
                    if np.isin(edgenodes, i).any():  # CHECK IF VERTEX IS PART OF THE EDGES ON WHICH THE INTERSECTION POINTS LIE
                        pass
                    else:
                        oppositenode = i
                        
                XIeTESSLIN = list()
                interfedge = [1,1,-1,-1]
                XIeTESSLIN.append(np.concatenate((XIeLIN[int(commonnode),:].reshape((1,2)),self.InterfApprox.XIint[:2,:]),axis=0))
                XIeTESSLIN.append(np.concatenate((XIeLIN[oppositenode,:].reshape((1,2)),self.InterfApprox.XIint[:2,:]),axis=0))
                XIeTESSLIN.append(np.concatenate((self.InterfApprox.XIint[0,:].reshape((1,2)),XIeLIN[[edgenode[0],oppositenode],:]),axis=0))
                XIeTESSLIN.append(np.concatenate((self.InterfApprox.XIint[1,:].reshape((1,2)),XIeLIN[[edgenode[1],oppositenode],:]),axis=0))
                
                # TURN LINEAR TRIANGULAR SUBELEMENTS INTO HIGH-ORDER TRIANGULAR SUBELEMENTS
                XIeTESSHO = list()
                for isub in range(Nesub):
                    XIeHO = self.HO_TRI_interf(XIeTESSLIN[isub],self.ElOrder,self.InterfApprox.XIint,interfedge[isub])
                    XIeTESSHO.append(XIeHO)
                
        return Nesub, SubElType, XIeTESSHO, interfedge
    
        
    ##################################################################################################
    ############################### ELEMENTAL NUMERICAL QUADRATURES ##################################
    ##################################################################################################
        
    def ComputeStandardQuadrature2D(self,NumQuadOrder2D):
        """
        Computes the numerical integration quadratures for 2D elements that are not cut by any interface.
        This function applies the standard FEM integration methodology using reference shape functions 
        evaluated at standard Gauss integration nodes. It is designed for elements where no interface cuts 
        through, and the traditional FEM approach is used for integration.

        Input:
            NumQuadOrder (int): The order of the numerical integration quadrature to be used.

        This function performs the following tasks:
            1. Computes the standard quadrature on the reference space in 2D.
            2. Evaluates reference shape functions on the standard reference quadrature using Gauss nodes.
            3. Precomputes the necessary integration entities, including:
                - Jacobian inverse matrix for the transformation between reference and physical 2D spaces.
                - Jacobian determinant for the transformation.
                - Standard physical Gauss integration nodes mapped from the reference element.
        """
        
        # COMPUTE THE STANDARD QUADRATURE ON THE REFERENCE SPACE IN 2D
        #### REFERENCE ELEMENT QUADRATURE TO INTEGRATE SURFACES 
        self.XIg, self.Wg, self.ng = GaussQuadrature(self.ElType,NumQuadOrder2D)
        
        # EVALUATE THE REFERENCE SHAPE FUNCTIONS ON THE STANDARD REFERENCE QUADRATURE ->> STANDARD FEM APPROACH
        # EVALUATE REFERENCE SHAPE FUNCTIONS 
        self.Ng, self.dNdxig, self.dNdetag = EvaluateReferenceShapeFunctions(self.XIg, self.ElType, self.ElOrder)
        
        # PRECOMPUTE THE NECESSARY INTEGRATION ENTITIES EVALUATED AT THE STANDARD GAUSS INTEGRATION NODES ->> STANDARD FEM APPROACH
        # WE COMPUTE THUS:
        #       - THE JACOBIAN OF THE TRANSFORMATION BETWEEN REFERENCE AND PHYSICAL 2D SPACES INVERSE MATRIX 
        #       - THE JACOBIAN OF THE TRANSFORMATION BETWEEN REFERENCE AND PHYSICAL 2D SPACES MATRIX DETERMINANT
        #       - THE STANDARD PHYSICAL GAUSS INTEGRATION NODES MAPPED FROM THE REFERENCE ELEMENT
          
        # COMPUTE MAPPED GAUSS NODES
        self.Xg = self.Ng @ self.Xe       
        # COMPUTE JACOBIAN INVERSE AND DETERMINANT
        self.invJg = np.zeros([self.ng,self.dim,self.dim])
        self.detJg = np.zeros([self.ng])
        for ig in range(self.ng):
            self.invJg[ig,:,:], self.detJg[ig] = Jacobian(self.Xe,self.dNdxig[ig,:],self.dNdetag[ig,:])
            self.detJg[ig] = abs(self.detJg[ig])
        
        # CHECK NUMERICAL QUADRATURE
        self.CheckQuadrature2D()
        return    
    
    
    def ComputeAdaptedQuadratures2D(self,NumQuadOrder2D):
        """ 
        Computes the numerical integration quadratures for both 2D and 1D elements that are cut by an interface. 
        This function uses an adapted quadrature approach, modifying the standard FEM quadrature method to accommodate 
        interface interactions within the element.

        Input:
            NumQuadOrder (int): The order of the numerical integration quadrature to be used for both the 2D and 1D elements.

        This function performs the following tasks:
            1. Tessellates the reference element to account for elemental subelements.
            2. Maps the tessellated subelements to the physical space.
            3. Determines the level-set values for different domains (e.g., plasma, vacuum).
            4. Generates subelement objects, assigning region flags and interpolating level-set values within subelements.
            5. Computes integration quadrature for each subelement using adapted quadratures (2D).
            6. Computes the quadrature for the elemental interface approximation (1D), mapping to physical elements.
        """
        
        ######### ADAPTED QUADRATURE TO INTEGRATE OVER ELEMENTAL SUBELEMENTS (2D)
        # TESSELLATE REFERENCE ELEMENT
        self.Nesub, SubElType, XIeTESSHO, interfedge = self.ReferenceElementTessellation()
        # MAP TESSELLATION TO PHYSICAL SPACE
        XeTESSHO = list()
        for isub in range(self.Nesub):
            # EVALUATE ELEMENTAL REFERENCE SHAPE FUNCTIONS AT SUBELEMENTAL NODAL COORDINATES 
            N2D, foo, foo = EvaluateReferenceShapeFunctions(XIeTESSHO[isub], self.ElType, self.ElOrder)
            # MAP SUBELEMENTAL NODAL COORDINATES TO PHYSICAL SPACE
            XeTESSHO.append(N2D @ self.Xe)
        
        # GENERATE SUBELEMENTAL OBJECTS
        self.SubElements = [Element(index = isubel, 
                                    ElType = SubElType, 
                                    ElOrder = self.ElOrder,
                                    Xe = XeTESSHO[isubel],
                                    Te = self.Te,
                                    PlasmaLSe = None,
                                    interfedge = interfedge[isubel]) for isubel in range(self.Nesub)]
        
        for isub, SUBELEM in enumerate(self.SubElements):
            #### ASSIGN REFERENCE SPACE TESSELLATION
            SUBELEM.XIe = XIeTESSHO[isub]
            #### ASSIGN A REGION FLAG TO EACH SUBELEMENT
            # INTERPOLATE VALUE OF LEVEL-SET FUNCTION INSIDE SUBELEMENT
            LSesub = self.ElementalInterpolationREFERENCE(np.mean(SUBELEM.XIe,axis=0),self.LSe)
            if LSesub < 0: 
                SUBELEM.Dom = -1
            else:
                SUBELEM.Dom = 1
                    
        # COMPUTE INTEGRATION QUADRATURE FOR EACH SUBELEMENT
        for SUBELEM in self.SubElements:
            # STANDARD REFERENCE ELEMENT QUADRATURE (2D)
            XIg2Dstand, SUBELEM.Wg, SUBELEM.ng = GaussQuadrature(SUBELEM.ElType,NumQuadOrder2D)
            # EVALUATE SUBELEMENTAL REFERENCE SHAPE FUNCTIONS 
            Nstand2D, dNdxistand2D, dNdetastand2D = EvaluateReferenceShapeFunctions(XIg2Dstand, SUBELEM.ElType, SUBELEM.ElOrder)
            # MAP 2D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE SUBELEMENTS  ->> ADAPTED 2D QUADRATURE FOR SUBELEMENTS
            SUBELEM.XIg = Nstand2D @ SUBELEM.XIe
            # EVALUATE ELEMENTAL REFERENCE SHAPE FUNCTIONS ON ADAPTED REFERENCE QUADRATURE
            SUBELEM.Ng, SUBELEM.dNdxig, SUBELEM.dNdetag = EvaluateReferenceShapeFunctions(SUBELEM.XIg, self.ElType, self.ElOrder)
            # MAPP ADAPTED REFERENCE QUADRATURE ON PHYSICAL ELEMENT
            SUBELEM.Xg = SUBELEM.Ng @ self.Xe
            
            # EVALUATE INTEGRATION ENTITIES (JACOBIAN INVERSE MATRIX AND DETERMINANT) ON ADAPTED QUADRATURES NODES
            SUBELEM.invJg = np.zeros([SUBELEM.ng,SUBELEM.dim,SUBELEM.dim])
            SUBELEM.detJg = np.zeros([SUBELEM.ng])
            for ig in range(SUBELEM.ng):
                SUBELEM.invJg[ig,:,:], SUBELEM.detJg[ig] = Jacobian(SUBELEM.Xe,dNdxistand2D[ig,:],dNdetastand2D[ig,:])
                SUBELEM.detJg[ig] = abs(SUBELEM.detJg[ig])
        
            # CHECK NUMERICAL QUADRATURE
            SUBELEM.CheckQuadrature2D(elemindex = self.index)
        return
    
    
    def ComputeAdaptedQuadrature1D(self,NumQuadOrder1D):
        
        ######### ADAPTED QUADRATURE TO INTEGRATE OVER ELEMENTAL INTERFACE APPROXIMATION (1D)
        #### STANDARD REFERENCE ELEMENT QUADRATURE TO INTEGRATE LINES (1D)
        XIg1Dstand, self.InterfApprox.Wg, self.InterfApprox.ng = GaussQuadrature(0,NumQuadOrder1D)
        #### QUADRATURE TO INTEGRATE LINES (1D)
        N1D, dNdxi1D, foo = EvaluateReferenceShapeFunctions(XIg1Dstand, 0, self.ElOrder)
                
        # MAP 1D REFERENCE STANDARD GAUSS INTEGRATION NODES ON THE REFERENCE INTERFACE ->> ADAPTED 1D QUADRATURE FOR INTERFACE
        self.InterfApprox.XIg = N1D @ self.InterfApprox.XIint
        # EVALUATE 2D REFERENCE SHAPE FUNCTION ON INTERFACE ADAPTED QUADRATURE
        self.InterfApprox.Ng, self.InterfApprox.dNdxig, self.InterfApprox.dNdetag = EvaluateReferenceShapeFunctions(self.InterfApprox.XIg, self.ElType, self.ElOrder)
        # MAP REFERENCE INTERFACE ADAPTED QUADRATURE ON PHYSICAL ELEMENT 
        self.InterfApprox.Xg = N1D @ self.InterfApprox.Xint
        # EVALUATE INTEGRATION ENTITIES (JACOBIAN INVERSE MATRIX AND DETERMINANT) ON ADAPTED QUADRATURES NODES
        self.InterfApprox.invJg = np.zeros([self.InterfApprox.ng,self.dim,self.dim])
        self.InterfApprox.detJg = np.zeros([self.InterfApprox.ng])
        self.InterfApprox.detJg1D = np.zeros([self.InterfApprox.ng])
        for ig in range(self.InterfApprox.ng):
            self.InterfApprox.invJg[ig,:,:], self.InterfApprox.detJg[ig] = Jacobian(self.Xe,self.InterfApprox.dNdxig[ig,:],self.InterfApprox.dNdetag[ig,:])
            self.InterfApprox.detJg1D[ig] = Jacobian1D(self.InterfApprox.Xint,dNdxi1D[ig,:])
            self.InterfApprox.detJg[ig] = abs(self.InterfApprox.detJg[ig])
            
        # CHECK NUMERICAL QUADRATURE
        self.CheckQuadrature1D()
            
        # COMPUTE OUTWARDS NORMAL VECTORS ON INTEGRATION NODES
        self.InterfaceNormals()
        
        # CHECK ORTHOGONALITY OF NORMAL VECTORS
        self.CheckInterfaceNormals()
        return
    
    
    def CheckQuadrature2D(self, elemindex = -1, tol=1e-4):
        # CHECK NUMERICAL INTEGRATION QUADRATURE BY INTEGRATING AREA
        error = False
        integral = 0
        for ig in range(self.ng):
            for inode in range(self.n):
                integral += self.Ng[ig,inode]*self.detJg[ig]*self.Wg[ig] 
        if abs(integral - self.area) > tol:
            error = True
        
        try:     
            if error:
                if elemindex == -1:
                    raise ValueError('Element '+ str(self.index)+': surface integration quadrature is not accurate.')
                else:
                    raise ValueError('Element '+ str(elemindex)+', subelem '+str(self.index)+': surface integration quadrature is not accurate.')
        except ValueError as e:
            print("Warning: ", e)
        return 
    
    
    def CheckQuadrature1D(self, tol=1e-2):
        # CHECK CUT ELEMENTS NUMERICAL INTEGRATION QUADRATURE BY INTEGRATING ARC LENGTH
        error = False
        
        # PIECE-WISE LINEAR APPROXIMATION
        length = 0
        for inode in range(self.ElOrder):
            lengthi = np.linalg.norm(self.InterfApprox.Xint[self.InterfApprox.Tint[inode+1],:]-self.InterfApprox.Xint[self.InterfApprox.Tint[inode],:])
            length += lengthi
        
        integral = 0
        for ig in range(self.InterfApprox.ng):
            for inode in range(self.n):
                integral += self.InterfApprox.Ng[ig,inode]*self.InterfApprox.detJg1D[ig]*self.InterfApprox.Wg[ig]
                
        if abs(length - integral) > tol:
            error = True
                
        try:     
            if error:
                raise ValueError('Element '+ str(self.index)+': arc length integration quadrature is not accurate.')
        except ValueError as e:
            print("Warning: ", e)
            
        return
    
    
    def ComputeGhostFacesQuadratures(self,NumQuadOrder1D):
        
        ######### ADAPTED QUADRATURE TO INTEGRATE OVER ELEMENTAL GHOST FACES
        #### STANDARD REFERENCE ELEMENT QUADRATURE TO INTEGRATE LINES (1D)
        XIg1Dstand, Wg1D, Ng1D = GaussQuadrature(0,NumQuadOrder1D)
        #### QUADRATURE TO INTEGRATE LINES (1D)
        N1D, dNdxi1D, foo = EvaluateReferenceShapeFunctions(XIg1Dstand, 0, self.ElOrder)
                    
        ######### ADAPTED QUADRATURE TO INTERGRATE OVER ELEMENTAL GHOST FACES (1D)
        for FACE in self.GhostFaces:
            FACE.ng = Ng1D
            FACE.Wg = Wg1D
            FACE.detJg = np.zeros([FACE.ng])
            # MAP 1D REFERENCE STANDARD GAUSS INTEGRATION NODES ON ELEMENTAL CUT EDGE ->> ADAPTED 1D QUADRATURE FOR CUT EDGE
            FACE.XIg = N1D @ FACE.XIseg
            # EVALUATE 2D REFERENCE SHAPE FUNCTION ON ELEMENTAL CUT EDGE 
            FACE.Ng, FACE.dNdxig, FACE.dNdetag = EvaluateReferenceShapeFunctions(FACE.XIg, self.ElType, self.ElOrder)
            # DISCARD THE NODAL SHAPE FUNCTIONS WHICH ARE NOT ON THE FACE (ZERO VALUE)
            #FACE.Ng = FACE.Ng[:,FACE.Tseg]
            #FACE.dNdxig = FACE.dNdxig[:,FACE.Tseg]
            #FACE.dNdetag = FACE.dNdetag[:,FACE.Tseg]
            # MAPP REFERENCE INTERFACE ADAPTED QUADRATURE ON PHYSICAL ELEMENT 
            FACE.Xg = N1D @ FACE.Xseg
            # EVALUATE INTEGRATION ENTITIES (JACOBIAN INVERSE MATRIX AND DETERMINANT) ON ADAPTED QUADRATURES NODES
            FACE.invJg = np.zeros([FACE.ng,FACE.dim,FACE.dim])
            FACE.detJg = np.zeros([FACE.ng])
            FACE.detJg1D = np.zeros([FACE.ng])
            for ig in range(FACE.ng):
                FACE.invJg[ig,:,:], FACE.detJg[ig] = Jacobian(self.Xe,FACE.dNdxig[ig,:],FACE.dNdetag[ig,:])
                FACE.detJg[ig] = abs(FACE.detJg[ig])
                FACE.detJg1D[ig] = Jacobian1D(FACE.Xseg,dNdxi1D[ig,:]) 
        return
    
    def PlotInterfaceApproximation(self,LSfunc):
        #### DEFINE SUBSPACES COMPUTATIONAL GRIDS, PHYSICAL AND REFERENCE
        Rmin = min(self.Xe[:,0])
        Rmax = max(self.Xe[:,0])
        Zmin = min(self.Xe[:,1])
        Zmax = max(self.Xe[:,1])
        dR = (Rmax-Rmin)/5
        dZ = (Zmax-Zmin)/5
        Rmin -= dR
        Rmax += dR
        Zmin -= dZ
        Zmax += dZ
        Ximin = -0.2
        Ximax = 1.2
        Etamin = -0.2
        Etamax = 1.2
        Nr = 40
        Nz = 40
        
        #### PHYSICAL SPACE
        # PHYSICAL SUBPLOT GRID
        xgrid = np.linspace(Rmin,Rmax,Nr)
        ygrid = np.linspace(Zmin,Zmax,Nz)
        X = np.zeros([Nr*Nz,2])
        for ix in range(Nr):
            for iy in range(Nz):
                X[ix*Nr+iy,:] = [xgrid[ix],ygrid[iy]]
        # COMPUTE EXACT LEVEL-SET VALUES
        PHIexact = np.zeros([Nr*Nz])
        for i in range(len(X[:,0])):
            PHIexact[i] = LSfunc(X[i,:])
        # COMPUTE INTERPOLATED LEVEL-SET FUNCTION IN PHYSICAL SPACE
        PHIint = np.zeros([Nr*Nz])
        for i in range(len(X[:,0])):
            PHIint[i] = self.ElementalInterpolationPHYSICAL(X[i,:],self.LSe)

        #### REFERENCE SPACE
        # REFERENCE SUBPLOT GRID
        xigrid = np.linspace(Ximin,Ximax,Nr)
        etagrid = np.linspace(Etamin,Etamax,Nz)
        XI = np.zeros([Nr*Nz,2])
        for ix in range(Nr):
            for iy in range(Nz):
                XI[ix*Nr+iy,:] = [xigrid[ix],etagrid[iy]]
        # MAP EXACT LEVEL-SET FUNCTION TO REFERENCE SPACE
        PHIexactREF = np.zeros([Nr*Nz])
        for i in range(len(XI[:,0])):
            PHIexactREF[i] = LSfunc(self.Mapping(XI[i,:]))
        # COMPUTE INTERPOLATED LEVEL-SET FUNCTION IN REFERENCE SPACE
        PHIintREF = np.zeros([Nr*Nz])
        for i in range(len(XI[:,0])):
            PHIintREF[i] = self.ElementalInterpolationREFERENCE(XI[i,:],self.LSe)
        
        # REPRESENTING LEVEL-SET ELEMENTAL APPROXIMATION
        fig, axs = plt.subplots(1, 2, figsize=(12,5))
        #### LEFT PLOT: PHYSICAL SPACE
        # PLOT ELEMENT EDGES
        for iedge in range(self.nedge):
            axs[0].plot([self.Xe[iedge,0],self.Xe[int((iedge+1)%self.nedge),0]],[self.Xe[iedge,1],self.Xe[int((iedge+1)%self.nedge),1]], color='k', linewidth=2)
        # PLOT NODES WITH NEGATIVE OR POSITIVE LEVEL-SET VALUES
        for inode in range(len(self.Xe[:,0])):
            if self.LSe[inode] < 0:
                cl = 'blue'
            else:
                cl = 'red'
            axs[0].scatter(self.Xe[inode,0],self.Xe[inode,1],s=60,color=cl,zorder=7)
        for inode in range(self.n):
            axs[0].text(self.Xe[inode,0]-dR/6,self.Xe[inode,1]+dZ/6,str(inode),fontsize=12)
        # PLOT INTERFACE
        axs[0].tricontour(X[:,0],X[:,1],PHIexact,levels=[0],colors='red', linewidths=3)
        # PLOT INTERPOLATED INTERFACE
        axs[0].tricontour(X[:,0],X[:,1],PHIint,levels=[0],colors='violet', linewidths=3)
        #axs[0].contour(xgrid,ygrid,PHIint,levels=[0],colors='violet', linewidths=3)
        # PLOT INTERSECTION POINTS
        axs[0].scatter(self.InterfApprox.Xint[:,0],self.InterfApprox.Xint[:,1],s=60,marker='o',color='green',zorder=7)
        axs[0].set_ylim([Zmin,Zmax])
        axs[0].set_xlim([Rmin,Rmax])

        #### RIGHT PLOT: REFERENCE SPACE
        XIe = ReferenceElementCoordinates(self.ElType,self.ElOrder)
        # PLOT ELEMENT EDGES
        for iedge in range(self.nedge):
            axs[1].plot([XIe[iedge,0],XIe[int((iedge+1)%self.nedge),0]],[XIe[iedge,1],XIe[int((iedge+1)%self.nedge),1]], color='k', linewidth=2)
        for inode in range(len(XIe[:,0])):
            axs[1].scatter(XIe[inode,0],XIe[inode,1],s=60,color='k',zorder=7)
        # PLOT NODES WITH NEGATIVE OR POSITIVE LEVEL-SET VALUES
        for inode in range(len(XIe[:,0])):
            if self.LSe[inode] < 0:
                cl = 'blue'
            else:
                cl = 'red'
            axs[1].scatter(XIe[inode,0],XIe[inode,1],s=60,color=cl,zorder=7)
        for inode in range(len(XIe[:,0])):
            axs[1].text(XIe[inode,0]+0.03,XIe[inode,1]+0.03,str(inode),fontsize=12)
        # PLOT INTERFACE
        axs[1].tricontour(XI[:,0],XI[:,1],PHIexactREF,levels=[0],colors='red', linewidths=3)
        #axs[1].tricontour(XIe[:,0],XIe[:,1],PHIe,levels=[0],colors='red', linewidths=3)
        # PLOT INTERPOLATED INTERFACE
        axs[1].tricontour(XI[:,0],XI[:,1],PHIintREF,levels=[0],colors='violet', linewidths=3)
        # PLOT INTERSECTION POINTS
        axs[1].scatter(self.InterfApprox.XIint[:,0],self.InterfApprox.XIint[:,1],s=60,marker='o',color='green',zorder=7)
        axs[1].set_ylim([Etamin,Etamax])
        axs[1].set_xlim([Ximin,Ximax])
        plt.show()
        
        return
    
    
    def CheckReferenceQuadratures(self):
        #### DEFINE REFERENCE SUBSPACE COMPUTATIONAL GRID
        Ximin = -0.2
        Ximax = 1.2
        Etamin = -0.2
        Etamax = 1.2
        Nr = 40
        Nz = 40
        XIe = ReferenceElementCoordinates(self.ElType,self.ElOrder)
        
        # REFERENCE SUBPLOT GRID
        xigrid = np.linspace(Ximin,Ximax,Nr)
        etagrid = np.linspace(Etamin,Etamax,Nz)
        XI = np.zeros([Nr*Nz,2])
        for ix in range(Nr):
            for iy in range(Nz):
                XI[ix*Nr+iy,:] = [xigrid[ix],etagrid[iy]]
        # COMPUTE INTERPOLATED LEVEL-SET FUNCTION IN REFERENCE SPACE
        PHIintREF = np.zeros([Nr*Nz])
        for i in range(len(XI[:,0])):
            PHIintREF[i] = self.ElementalInterpolationREFERENCE(XI[i,:],self.LSe)
        
        colorlist = ['orange','gold','grey','cyan']
        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        # LEFT PLOT: PARENT ELEMENT
        axs[0].set_ylim([Etamin,Etamax])
        axs[0].set_xlim([Ximin,Ximax])
        # PLOT ELEMENT EDGES
        for iedge in range(self.nedge):
            axs[0].plot([XIe[iedge,0],XIe[int((iedge+1)%self.nedge),0]],[XIe[iedge,1],XIe[int((iedge+1)%self.nedge),1]], color='k', linewidth=3)
        # PLOT NODES WITH NEGATIVE OR POSITIVE LEVEL-SET VALUES
        for inode in range(len(XIe[:,0])):
            if self.LSe[inode] < 0:
                cl = 'blue'
            else:
                cl = 'red'
            axs[0].scatter(XIe[inode,0],XIe[inode,1],s=60,color=cl,zorder=7)
        for inode in range(len(XIe[:,0])):
            axs[0].text(XIe[inode,0]+0.03,XIe[inode,1]+0.03,str(inode),fontsize=12)
        # PLOT INTERPOLATED INTERFACE
        axs[0].tricontour(XI[:,0],XI[:,1],PHIintREF,levels=[0],colors='violet', linewidths=3)
        # PLOT INTERSECTION POINTS
        axs[0].scatter(self.InterfApprox.XIint[:,0],self.InterfApprox.XIint[:,1],s=60,marker='o',color='green',zorder=7)

        #### RIGHT PLOT: HIGH-ORDER SUBELEMENTS
        axs[1].set_ylim([Etamin,Etamax])
        axs[1].set_xlim([Ximin,Ximax])
        xitext = [-0.05,0,-0.05]
        etatext = [0.05,0.05,-0.05]
        for isub, SUBELEM in enumerate(self.SubElements):
            # PLOT SUBELEMENT EDGES
            for iedge in range(SUBELEM.nedge):
                inode = iedge
                jnode = int((iedge+1)%SUBELEM.nedge)
                if iedge == SUBELEM.interfedge:
                    inodeHO = SUBELEM.nedge+(SUBELEM.ElOrder-1)*inode
                    xcoords = [SUBELEM.XIe[inode,0],SUBELEM.XIe[inodeHO:inodeHO+(SUBELEM.ElOrder-1),0],SUBELEM.XIe[jnode,0]]
                    xcoords = list(chain.from_iterable([x] if not isinstance(x, np.ndarray) else x for x in xcoords))
                    ycoords = [SUBELEM.XIe[inode,1],SUBELEM.XIe[inodeHO:inodeHO+(SUBELEM.ElOrder-1),1],SUBELEM.XIe[jnode,1]]
                    ycoords = list(chain.from_iterable([y] if not isinstance(y, np.ndarray) else y for y in ycoords))
                    axs[1].plot(xcoords,ycoords, color=colorlist[isub], linewidth=3)
                else:
                    axs[1].plot([SUBELEM.XIe[inode,0],SUBELEM.XIe[jnode,0]],[SUBELEM.XIe[inode,1],SUBELEM.XIe[jnode,1]], color=colorlist[isub], linewidth=3)
            axs[1].scatter(SUBELEM.XIe[:,0],SUBELEM.XIe[:,1],s=60,color=colorlist[isub],zorder=7)
            axs[1].scatter(SUBELEM.XIg[:,0],SUBELEM.XIg[:,1],s=60,marker='x',color=colorlist[isub],zorder=7)
            # WRITE NODE NUMBER
            #for inode in range(SUBELEM.n):
            #    axs[1].text(SUBELEM.XIe[inode,0]+xitext[isub],SUBELEM.XIe[inode,1]+etatext[isub],str(inode),fontsize=12, color=colorlist[isub])
        # PLOT REFERENCE INTERFACE APPROXIMATION QUADRATURE
        axs[1].scatter(self.InterfApprox.XIg[:,0],self.InterfApprox.XIg[:,1,],s=60,marker='x',color='green',zorder=7)
        plt.show()
        
        ###### CHECK QUADRATURES
        # SURFACE INTEGRATION QUADARTURES -> COMPUTE ELEMENTAL AREA
        totalarea = 0
        totalintegral = 0  
        for isub, SUBELEM in enumerate(self.SubElements):
            # COMPUTE AREA ARITHMETICALLY
            if SUBELEM.interfedge == -1:  # REGULAR TRIANGLE
                area = compute_triangle_area(SUBELEM.XIe)
            else:
                Xepoly = np.zeros([3+SUBELEM.ElOrder-1,2])
                ipoint = 0
                for iedge in range(SUBELEM.nedge):
                    inode = iedge
                    jnode = int((iedge+1)%SUBELEM.nedge)
                    if iedge == 0:
                        Xepoly[ipoint,:] = SUBELEM.XIe[inode,:]
                        Xepoly[ipoint+1,:] = SUBELEM.XIe[jnode,:]
                        ipoint += 2
                    elif iedge == SUBELEM.interfedge:
                        inodeHO = SUBELEM.nedge+(SUBELEM.ElOrder-1)*inode
                        Xepoly[ipoint:ipoint+(SUBELEM.ElOrder-1),:] = SUBELEM.XIe[inodeHO:inodeHO+(SUBELEM.ElOrder-1),:]
                        ipoint += SUBELEM.ElOrder-1
                    else:
                        Xepoly[ipoint,:] = SUBELEM.XIe[inode,:]
                        ipoint += 1
                area = polygon_area(Xepoly,SUBELEM.ElOrder,SUBELEM.interfedge)
            
            #area = compute_triangle_area(Xesub)
            
            # COMPUTE INTEGRAL
            invJg = np.zeros([SUBELEM.ng,2,2])
            detJg = np.zeros([SUBELEM.ng])
            for ig in range(SUBELEM.ng):
                invJg[ig,:,:], detJg[ig] = Jacobian(SUBELEM.XIe,SUBELEM.dNdxig[ig,:],SUBELEM.dNdetag[ig,:])
        
            integral = 0
            for ig in range(SUBELEM.ng):
                for inode in range(SUBELEM.n):
                    integral += SUBELEM.Ng[ig,inode]*abs(detJg[ig])*SUBELEM.Wg[ig]
                #integral += abs(detJg[ig])*Wg2Dstand[ig]
                    
            print("SUBELEMENT ,", isub," POLYGONAL DECOMPOSITION AREA = ", area)
            totalarea += area
            print("SUBELEMENT ,", isub," INTEGRAL AREA = ", integral)
            totalintegral += abs(integral)
            
        trianglearea = compute_triangle_area(XIe)
        print("ELEMENT AREA = ", trianglearea)
        print("ELEMENTAL POLYGONAL DECOMPOSITION AREA = ", totalarea)
        print("ELEMENTAL INTEGRAL AREA = ", totalintegral)
        
        # LINE INTEGRATION QUADRATURES
        # PIECE-WISE LINEAR APPROXIMATION
        length = 0
        for inode in range(self.ElOrder):
            lengthi = np.linalg.norm(self.InterfApprox.XIint[self.InterfApprox.Tint[inode+1],:]-self.InterfApprox.XIint[self.InterfApprox.Tint[inode],:])
            length += lengthi
        print('PIECE-WISE LINEAR INTERFACE ARC LENGTH APPROXIMATION = ', length)
        
        # OBTAIN 1D STANDARD GAUSS QUADRATURE
        XIg1Dstand, foo, foo = GaussQuadrature(0,self.InterfApprox.ng)
        # EVALUATE SUBELEMENTAL REFERENCE SHAPE FUNCTIONS 
        foo, dNdxi1D, foo = EvaluateReferenceShapeFunctions(XIg1Dstand, 0, self.ElOrder)
        # COMPUTE 1D MAPPING DETERMINANT 
        detJg1D = np.zeros([self.InterfApprox.ng])
        for ig in range(self.InterfApprox.ng):
            detJg1D[ig] = Jacobian1D(self.InterfApprox.XIint,dNdxi1D[ig,:])
        
        integral = 0
        for ig in range(self.InterfApprox.ng):
            for inode in range(self.n):
                integral += self.InterfApprox.Ng[ig,inode]*detJg1D[ig]*self.InterfApprox.Wg[ig]
        print('ISOPARAMETRIC INTERFACE ARC LENGTH APPROXIMATION = ', integral)
        return
    
    
    
    def CheckPhysicalQuadratures(self):
        #### DEFINE SUBSPACES COMPUTATIONAL GRIDS, PHYSICAL AND REFERENCE
        Rmin = min(self.Xe[:,0])
        Rmax = max(self.Xe[:,0])
        Zmin = min(self.Xe[:,1])
        Zmax = max(self.Xe[:,1])
        dR = (Rmax-Rmin)/5
        dZ = (Zmax-Zmin)/5
        Rmin -= dR
        Rmax += dR
        Zmin -= dZ
        Zmax += dZ
        Nr = 40
        Nz = 40
        
        colorlist = ['orange','gold','grey','cyan']

        if self.interfedge == -1:
            fig, axs = plt.subplots(1, 1, figsize=(5,5))
            #### LEFT PLOT: PHYSICAL ELEMENT
            axs.set_ylim([Zmin,Zmax])
            axs.set_xlim([Rmin,Rmax])
            # PLOT ELEMENT EDGES
            for iedge in range(self.nedge):
                axs.plot([self.Xe[iedge,0],self.Xe[int((iedge+1)%self.nedge),0]],[self.Xe[iedge,1],self.Xe[int((iedge+1)%self.nedge),1]], color='k', linewidth=2)
            # PLOT NODES WITH NEGATIVE OR POSITIVE LEVEL-SET VALUES
            for inode in range(self.n):
                if self.LSe[inode] < 0:
                    cl = 'blue'
                else:
                    cl = 'red'
                axs.scatter(self.Xe[inode,0],self.Xe[inode,1],s=60,color=cl,zorder=7)
            for inode in range(self.n):
                axs.text(self.Xe[inode,0]-dR/6,self.Xe[inode,1]+dZ/6,str(inode),fontsize=12)
            # PLOT QUADRATURE
            axs.scatter(self.Xg[:,0],self.Xg[:,1],s=60,marker='x',color=colorlist[0],zorder=7)
            
            # CHECK QUADRATURES
            totalarea = compute_triangle_area(self.Xe) 
            # COMPUTE INTEGRAL
            integral = 0
            for ig in range(self.ng):
                for inode in range(self.n):
                    integral += self.Ng[ig,inode]*abs(self.detJg[ig])*self.Wg[ig]
                    
            print("ELEMENTAL POLYGONAL DECOMPOSITION AREA = ", totalarea)
            print("ELEMENTAL INTEGRAL AREA = ", integral)
            
        else: 
            #### PHYSICAL SPACE
            # PHYSICAL SUBPLOT GRID
            xgrid = np.linspace(Rmin,Rmax,Nr)
            ygrid = np.linspace(Zmin,Zmax,Nz)
            X = np.zeros([Nr*Nz,2])
            for ix in range(Nr):
                for iy in range(Nz):
                    X[ix*Nr+iy,:] = [xgrid[ix],ygrid[iy]]
            # COMPUTE INTERPOLATED LEVEL-SET FUNCTION IN PHYSICAL SPACE
            PHIint = np.zeros([Nr*Nz])
            for i in range(len(X[:,0])):
                PHIint[i] = self.ElementalInterpolationPHYSICAL(X[i,:],self.LSe)

            fig, axs = plt.subplots(1, 2, figsize=(12,5))
            #### LEFT PLOT: PHYSICAL ELEMENT
            axs[0].set_ylim([Zmin,Zmax])
            axs[0].set_xlim([Rmin,Rmax])
            # PLOT ELEMENT EDGES
            for iedge in range(self.nedge):
                axs[0].plot([self.Xe[iedge,0],self.Xe[int((iedge+1)%self.nedge),0]],[self.Xe[iedge,1],self.Xe[int((iedge+1)%self.nedge),1]], color='k', linewidth=2)
            # PLOT NODES WITH NEGATIVE OR POSITIVE LEVEL-SET VALUES
            for inode in range(self.n):
                if self.LSe[inode] < 0:
                    cl = 'blue'
                else:
                    cl = 'red'
                axs[0].scatter(self.Xe[inode,0],self.Xe[inode,1],s=60,color=cl,zorder=7)
            for inode in range(self.n):
                axs[0].text(self.Xe[inode,0]-dR/6,self.Xe[inode,1]+dZ/6,str(inode),fontsize=12)
            # PLOT INTERPOLATED INTERFACE
            axs[0].tricontour(X[:,0],X[:,1],PHIint,levels=[0],colors='violet', linewidths=3)
            # PLOT INTERSECTION POINTS
            axs[0].scatter(self.InterfApprox.Xint[:,0],self.InterfApprox.Xint[:,1],s=60,marker='o',color='green',zorder=7)

            #### RIGHT PLOT: TESSELLATED PHYSICAL ELEMENT
            axs[1].set_ylim([Zmin,Zmax])
            axs[1].set_xlim([Rmin,Rmax])
            xtext = [0.05,-0.02,-0.1]
            ytext = [-0.25,0.08,-0.25]
            for isub, SUBELEM in enumerate(self.SubElements):
                # PLOT SUBELEMENT EDGES
                for iedge in range(SUBELEM.nedge):
                    inode = iedge
                    jnode = int((iedge+1)%SUBELEM.nedge)
                    if iedge == self.interfedge[isub]:
                        inodeHO = SUBELEM.nedge+(SUBELEM.ElOrder-1)*inode
                        xcoords = [SUBELEM.Xe[inode,0],SUBELEM.Xe[inodeHO:inodeHO+(SUBELEM.ElOrder-1),0],SUBELEM.Xe[jnode,0]]
                        xcoords = list(chain.from_iterable([x] if not isinstance(x, np.ndarray) else x for x in xcoords))
                        ycoords = [SUBELEM.Xe[inode,1],SUBELEM.Xe[inodeHO:inodeHO+(SUBELEM.ElOrder-1),1],SUBELEM.Xe[jnode,1]]
                        ycoords = list(chain.from_iterable([y] if not isinstance(y, np.ndarray) else y for y in ycoords))
                        axs[1].plot(xcoords,ycoords, color=colorlist[isub], linewidth=3)
                    else:
                        axs[1].plot([SUBELEM.Xe[inode,0],SUBELEM.Xe[jnode,0]],[SUBELEM.Xe[inode,1],SUBELEM.Xe[jnode,1]], color=colorlist[isub], linewidth=3)
                axs[1].scatter(SUBELEM.Xe[:,0],SUBELEM.Xe[:,1],s=60,color=colorlist[isub],zorder=7)
                axs[1].scatter(SUBELEM.Xg[:,0],SUBELEM.Xg[:,1],s=60,marker='x',color=colorlist[isub],zorder=7)
                # WRITE NODE NUMBER
                #for inode in range(SUBELEM.n):
                #    axs[1].text(SUBELEM.Xe[inode,0]+xtext[isub],SUBELEM.Xe[inode,1]+ytext[isub],str(inode),fontsize=12, color=colorlist[isub])
            # PLOT REFERENCE INTERFACE APPROXIMATION QUADRATURE
            axs[1].scatter(self.InterfApprox.Xg[:,0],self.InterfApprox.Xg[:,1,],s=60,marker='x',color='green',zorder=7)

            # CHECK QUADRATURES
            totalarea = 0
            totalintegral = 0  
            for isub, SUBELEM in enumerate(self.SubElements):
                # COMPUTE AREA ARITHMETICALLY
                if SUBELEM.interfedge == -1:  # REGULAR TRIANGLE
                    area = compute_triangle_area(SUBELEM.Xe)
                else:
                    Xepoly = np.zeros([3+SUBELEM.ElOrder-1,2])
                    ipoint = 0
                    for iedge in range(SUBELEM.nedge):
                        inode = iedge
                        jnode = int((iedge+1)%SUBELEM.nedge)
                        if iedge == 0:
                            Xepoly[ipoint,:] = SUBELEM.Xe[inode,:]
                            Xepoly[ipoint+1,:] = SUBELEM.Xe[jnode,:]
                            ipoint += 2
                        elif iedge == SUBELEM.interfedge:
                            inodeHO = SUBELEM.nedge+(SUBELEM.ElOrder-1)*inode
                            Xepoly[ipoint:ipoint+(SUBELEM.ElOrder-1),:] = SUBELEM.Xe[inodeHO:inodeHO+(SUBELEM.ElOrder-1),:]
                            ipoint += SUBELEM.ElOrder-1
                        else:
                            Xepoly[ipoint,:] = SUBELEM.Xe[inode,:]
                            ipoint += 1
                    area = polygon_area(Xepoly,SUBELEM.ElOrder,SUBELEM.interfedge)
                
                # COMPUTE INTEGRAL
                integral = 0
                for ig in range(SUBELEM.ng):
                    for inode in range(SUBELEM.n):
                        integral += SUBELEM.Ng[ig,inode]*abs(SUBELEM.detJg[ig])*SUBELEM.Wg[ig]
                        
                print("SUBELEMENT ,", isub," POLYGONAL DECOMPOSITION AREA = ", area)
                totalarea += area
                print("SUBELEMENT ,", isub," INTEGRAL AREA = ", integral)
                totalintegral += abs(integral)
                
            trianglearea = compute_triangle_area(self.Xe)
            print("ELEMENT AREA = ", trianglearea)
            print("ELEMENTAL POLYGONAL DECOMPOSITION AREA = ", totalarea)
            print("ELEMENTAL INTEGRAL AREA = ", totalintegral)

            # LINE INTEGRATION QUADRATURES
            # PIECE-WISE LINEAR APPROXIMATION
            length = 0
            for inode in range(self.ElOrder):
                lengthi = np.linalg.norm(self.InterfApprox.Xint[self.InterfApprox.Tint[inode+1],:]-self.InterfApprox.Xint[self.InterfApprox.Tint[inode],:])
                length += lengthi
            print('PIECE-WISE LINEAR INTERFACE ARC LENGTH APPROXIMATION = ', length)

            integral = 0
            for ig in range(self.InterfApprox.ng):
                for inode in range(self.n):
                    integral += self.InterfApprox.Ng[ig,inode]*self.InterfApprox.detJg1D[ig]*self.InterfApprox.Wg[ig]
            print('ISOPARAMETRIC INTERFACE ARC LENGTH APPROXIMATION = ', integral)
        return
    
    ##################################################################################################
    ################################ ELEMENTAL INTEGRATION ###########################################
    ##################################################################################################
    
    def IntegrateElementalDomainTerms(self,SourceTermg):
        """ 
        This function computes the elemental contributions to the global system by integrating the source terms over 
        the elemental domain. It calculates the left-hand side (LHS) matrix and right-hand side (RHS) vector using 
        Gauss integration nodes.

        Input:
            SourceTermg (ndarray): The Grad-Shafranov equation source term evaluated at the physical Gauss integration nodes.
        

        This function computes:
            1. The elemental contributions to the LHS matrix (stiffness term and gradient term).
            2. The elemental contributions to the RHS vector (source term).

        Output:
            - LHSe (ndarray): The elemental left-hand side matrix (stiffness matrix) of the system.
            - RHSe (ndarray): The elemental right-hand side vector of the system.

        The function loops over Gauss integration nodes to compute these contributions and assemble the elemental system.
        """
                    
        LHSe = np.zeros([self.n,self.n])
        RHSe = np.zeros([self.n])
        
        # LOOP OVER GAUSS INTEGRATION NODES
        for ig in range(self.ng):  
            # SHAPE FUNCTIONS GRADIENT IN PHYSICAL SPACE
            Ngrad = self.invJg[ig,:,:]@np.array([self.dNdxig[ig,:],self.dNdetag[ig,:]])
            # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM 
            for i in range(self.n):   # ROWS ELEMENTAL MATRIX
                for j in range(self.n):   # COLUMNS ELEMENTAL MATRIX
                    # COMPUTE LHS MATRIX TERMS
                    ### STIFFNESS TERM  [ nabla(N_i)*nabla(N_j) ]  
                    LHSe[i,j] -= (1/self.Xg[ig,0])*Ngrad[:,j]@Ngrad[:,i]*self.detJg[ig]*self.Wg[ig]
                # COMPUTE RHS VECTOR TERMS [ (source term)*N_i ]
                RHSe[i] += (1/self.Xg[ig,0])*SourceTermg[ig] * self.Ng[ig,i] *self.detJg[ig]*self.Wg[ig]
                
        return LHSe, RHSe
    
    
    def PrescribeDirichletBC(self,elmat,elrhs):
        for Teboun in self.Teboun:
            for ibounode in Teboun:
                adiag = elmat[ibounode,ibounode]
                # PASS MATRIX COLUMN TO RIGHT-HAND-SIDE
                elrhs -= elmat[:,ibounode]*self.PSI_Be[ibounode]
                # NULLIFY BOUNDARY NODE ROW
                elmat[ibounode,:] = 0
                # NULLIFY BOUNDARY NODE COLUMN
                elmat[:,ibounode] = 0
                # PRESCRIBE BOUNDARY CONDITION ON BOUNDARY NODE
                if abs(adiag) > 0:
                    elmat[ibounode,ibounode] = adiag
                    elrhs[ibounode] = adiag*self.PSI_Be[ibounode]
                else:
                    elmat[ibounode,ibounode] = 1
                    elrhs[ibounode] = self.PSI_Be[ibounode]
        return elmat, elrhs
    
    
    def IntegrateElementalInterfaceTerms(self,beta):
        """ 
        This function computes the elemental contributions to the global system from the interface terms, using 
        Nitsche's method. It integrates the interface conditions over the elemental interface approximation segments. 
        It calculates the left-hand side (LHS) matrix and right-hand side (RHS) vector using Gauss integration nodes.

        Input:
            beta (float): The penalty parameter for Nitsche's method, which controls the strength of the penalty term.
        
        This function computes:
            1. The elemental contributions to the LHS matrix (including Dirichlet boundary term, symmetric Nitsche's term, and penalty term).
            2. The elemental contributions to the RHS vector (including symmetric Nitsche's term and penalty term).

        Output: 
            - LHSe (ndarray): The elemental left-hand side matrix (stiffness matrix) of the system, incorporating Nitsche's method.
            - RHSe (ndarray): The elemental right-hand side vector of the system, incorporating Nitsche's method.

        The function loops over interface segments and Gauss integration nodes to compute these contributions and assemble the global system.
        """
        
        LHSe = np.zeros([self.n,self.n])
        RHSe = np.zeros([self.n])
    
        # LOOP OVER GAUSS INTEGRATION NODES
        for ig in range(self.InterfApprox.ng):  
            # SHAPE FUNCTIONS NORMAL GRADIENT IN PHYSICAL SPACE
            n_dot_Ngrad = self.InterfApprox.NormalVec[ig]@self.InterfApprox.invJg[ig,:,:]@np.array([self.InterfApprox.dNdxig[ig,:],self.InterfApprox.dNdetag[ig,:]])
            # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM
            for i in range(self.n):  # ROWS ELEMENTAL MATRIX
                for j in range(self.n):  # COLUMNS ELEMENTAL MATRIX
                    # COMPUTE LHS MATRIX TERMS
                    ### DIRICHLET BOUNDARY TERM  [ N_i*(n dot nabla(N_j)) ]  
                    LHSe[i,j] += (1/self.InterfApprox.Xg[ig,0])*self.InterfApprox.Ng[ig,i] * n_dot_Ngrad[j] * self.InterfApprox.detJg1D[ig] * self.InterfApprox.Wg[ig]
                    ### SYMMETRIC NITSCHE'S METHOD TERM   [ N_j*(n dot nabla(N_i)) ]
                    LHSe[i,j] += (1/self.InterfApprox.Xg[ig,0])*n_dot_Ngrad[i]*self.InterfApprox.Ng[ig,j] * self.InterfApprox.detJg1D[ig] * self.InterfApprox.Wg[ig]
                    ### PENALTY TERM   [ beta * (N_i*N_j) ]
                    LHSe[i,j] += beta * (1/self.length) * self.InterfApprox.Ng[ig,i] * self.InterfApprox.Ng[ig,j] * self.InterfApprox.detJg1D[ig] * self.InterfApprox.Wg[ig]
                    
                # COMPUTE RHS VECTOR TERMS 
                ### SYMMETRIC NITSCHE'S METHOD TERM  [ PSI_D * (n dot nabla(N_i)) ]
                RHSe[i] +=  (1/self.InterfApprox.Xg[ig,0])*self.InterfApprox.PSIg[ig] * n_dot_Ngrad[i] * self.InterfApprox.detJg1D[ig] * self.InterfApprox.Wg[ig]
                ### PENALTY TERM   [ beta * N_i * PSI_D ]
                RHSe[i] +=  beta * (1/self.length) * self.InterfApprox.PSIg[ig] * self.InterfApprox.Ng[ig,i] * self.InterfApprox.detJg1D[ig] * self.InterfApprox.Wg[ig]
        
        return LHSe, RHSe
    
    
    ##################################################################################################
    ################################ ELEMENT CHARACTERISATION ########################################
    ##################################################################################################
        
def ElementalNumberOfEdges(elemType):
    """ 
    This function returns the number of edges for a given element type. The element types are represented by integers:
    - 0: For 1D elements (e.g., line segments)
    - 1: For 2D triangular elements
    - 2: For 2D quadrilateral elements
    
    Input:
        elemType (int): The type of element for which to determine the number of edges. The possible values are:
    
    Output: 
        numedges (int): The number of edges for the given element type. 
    """
    match elemType:
        case 0:
            numedges = 1
        case 1:
            numedges = 3
        case 2:  
            numedges = 4
    return numedges     

    
def ElementalNumberOfNodes(elemType, elemOrder):
    """ 
    This function returns the number of nodes and the number of edges for a given element type and order. 
    The element types are represented by integers:
        - 0: 1D element (line segment)
        - 1: 2D triangular element
        - 2: 2D quadrilateral element
    
    The element order corresponds to the polynomial degree of the elemental shape functions.

    Input:
        - elemType (int): The type of element. Possible values:
                        - 0: 1D element (segment)
                        - 1: 2D triangular element
                        - 2: 2D quadrilateral element
        - elemOrder (int): The order (degree) of the element, determining the number of nodes.

    Output: 
        - n (int): The number of nodes for the given element type and order.
        - nedge (int): The number of edges for the given element order.
    """
    match elemType:
        case 0:
            n = elemOrder +1        
        case 1:
            match elemOrder:
                case 1:
                    n = 3
                case 2: 
                    n = 6
                case 3:
                    n = 10
        case 2:
            match elemOrder:
                case 1:
                    n = 4
                case 2:
                    n = 9
                case 3:
                    n = 16
    nedge = elemOrder + 1
    return n, nedge
    
    
def ReferenceElementCoordinates(elemType,elemOrder):
    """
    Returns nodal coordinates matrix for reference element of type elemType and order elemOrder.
    
    Input:
        - elemType (int): The type of element. Possible values:
                        - 0: 1D element (segment)
                        - 1: 2D triangular element
                        - 2: 2D quadrilateral element
        - elemOrder (int): The order (degree) of the element, determining the number of nodes.

    Ouput:
        Xe (ndarray): reference element nodal coordinates matrix
    """
    
    match elemType:
        case 0:    # LINE (1D ELEMENT)
            match elemOrder:
                case 0:
                    # --1--
                    Xe = np.array([0])
                case 1:
                    # 1---2
                    Xe = np.array([-1,1])      
                case 2:         
                    # 1---3---2
                    Xe = np.array([-1,1,0])
                case 3:         
                    # 1-3-4-2
                    Xe = np.array([-1,1,-1/3,1/3])
    
        case 1:   # TRIANGLE
            match elemOrder:
                case 1:
                    # 2
                    # |\
                    # | \
                    # 3--1
                    Xe = np.array([[1,0],
                                   [0,1],
                                   [0,0]])
                case 2:
                    # 2
                    # |\
                    # 5 4
                    # |  \
                    # 3-6-1
                    Xe = np.array([[1,0],
                                   [0,1],
                                   [0,0],
                                   [1/2,1/2],
                                   [0,1/2],
                                   [1/2,0]])
                case 3:
                    #  2
                    # | \
                    # 6  5 
                    # |   \
                    # 7 10 4
                    # |     \
                    # 3-8--9-1
                    Xe = np.array([[1,0],
                                   [0,1],
                                   [0,0],
                                   [2/3,1/3],
                                   [1/3,2/3],
                                   [0,2/3],
                                   [0,1/3],
                                   [1/3,0],
                                   [2/3,0],
                                   [1/3,1/3]])
                            
        case 2:    # QUADRILATERAL
            match elemOrder:
                case 1: 
                    # 4-----3
                    # |     |
                    # |     |
                    # 1-----2
                    Xe = np.array([[-1,-1],
                                   [1,-1],
                                   [1,1],
                                   [-1,1]])
                case 2:
                    # 4---7---3
                    # |       |
                    # 8   9   6
                    # |       |
                    # 1---5---2
                    Xe = np.array([[-1,-1],
                                   [1,-1],
                                   [1,1],
                                   [-1,1],
                                   [0,-1],
                                   [1,0],
                                   [0,1],
                                   [-1,0],
                                   [0,0]])
                case 3:
                    # 4---10--9---3
                    # |           |
                    # 11  16  15  8
                    # |           |
                    # 12  13  14  7
                    # |           |
                    # 1---5---6---2
                    Xe = np.array([[-1,-1],
                                   [1,-1],
                                   [1,1],
                                   [-1,1],
                                   [-1/3,-1],
                                   [1/3,-1],
                                   [1,-1/3],
                                   [1,1/3],
                                   [1/3,1],
                                   [-1/3,1],
                                   [-1,1/3],
                                   [-1,-1/3],
                                   [-1/3,-1/3],
                                   [1/3,-1/3],
                                   [1/3,1/3],
                                   [-1/3,1/3]])
    return Xe


def get_edge_index(ElType,inode,jnode):
    """
    Determines the edge index from the given vertices.

    Input:
        - elemType (int): The type of element. Possible values:
                    - 0: 1D element (segment)
                    - 1: 2D triangular element
                    - 2: 2D quadrilateral element
        - inode (int): Index of the first vertex of the edge.
        - jnode (int): Index of the second vertex of the edge.

    Output:
        - The index of the edge in the list.
    """
    if ElType == 1:
        element_edges = [(0,1), (1,2), (2,0)]
    elif ElType == 2:
        element_edges = [(0,1), (1,2), (2,3), (3,0)]
    
    return element_edges.index((inode,jnode))


def compute_triangle_area(Xe):
    x1, y1 = Xe[0,:]
    x2, y2 = Xe[1,:]
    x3, y3 = Xe[2,:]
    # Calculate the area using the determinant formula
    area = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    return area

def polygon_area(Xe,ElOrder,interfedge):
    totalarea = 0
    for isub in range(ElOrder):
        Xesub = Xe[0,:].reshape((1,2))
        Xesub = np.concatenate((Xesub,Xe[interfedge+isub:interfedge+isub+2,:]),axis=0)
        totalarea += compute_triangle_area(Xesub)
    return totalarea

def compute_quadrilateral_area(Xe):
    # Split the quadrilateral into two triangles
    triangle1 = Xe[:3,:]
    triangle2 = np.concatenate((Xe[2:,:], np.reshape(Xe[0,:],(1,2))), axis=0)
    # Compute the area of the two triangles and sum them
    area = compute_triangle_area(triangle1) + compute_triangle_area(triangle2)
    return area     
    
