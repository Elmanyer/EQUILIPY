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
from Segment import *

class Element:
    
    ##################################################################################################
    ################################ ELEMENT INITIALISATION ##########################################
    ##################################################################################################
    
    def __init__(self,index,ElType,ElOrder,Xe,Te,PlasmaLSe):
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
        self.Dom = None                                                 # DOMAIN WHERE THE ELEMENT LIES (-1: "PLASMA"; 0: "PLASMA INTERFACE"; +1: "VACUUM" ; +2: FIRST WALL ; +3: "EXTERIOR")
        self.neighbours = None                                          # GLOBAL INDEXES OF NEAREST NEIGHBOURS ELEMENTS CORRESPONDING TO EACH ELEMENTAL FACE (LOCAL INDEX ORDERING FOR FACES)
        
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
        self.interfedge = None
        return
    
    
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
        
        N, foo, foo = EvaluateReferenceShapeFunctions(Xi, self.ElType, self.ElOrder)
        X = N @ self.Xe
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
        Br = 0
        XI = self.InverseMapping(X)
        for i in range(self.n):
            foo, foo, dNdeta = ShapeFunctionsReference(XI, self.ElType, self.ElOrder, i+1)
            Br -= dNdeta*self.PSIe[i]/X[0]
        return Br
    
    def Bre(self):
        """
        Elemental nodes total radial magnetic field such that    Br = -1/R dpsi/dZ
        """
        Bre = np.zeros([self.n])
        XIe = ReferenceElementCoordinates(self.ElType,self.ElOrder)
        foo, foo, dNdeta = EvaluateReferenceShapeFunctions(XIe, self.ElType, self.ElOrder)
        for inode in range(self.n):
            Bre[inode] = -self.PSIe@dNdeta[inode,:]/self.Xe[inode,0]
        return Bre
    
    def Bz(self,X):
        """
        Total vertical magnetic field at point X such that    Bz = 1/R dpsi/dR
        """
        Bz = 0
        XI = self.InverseMapping(X)
        for i in range(self.n):
            foo, dNdxi, foo = ShapeFunctionsReference(XI, self.ElType, self.ElOrder, i+1)
            Bz += dNdxi*self.PSIe[i]/X[0]
        return Bz
    
    def Bze(self):
        """
        Elemental nodes total vertical magnetic field such that    Bz = 1/R dpsi/dR
        """
        Bze = np.zeros([self.n])
        XIe = ReferenceElementCoordinates(self.ElType,self.ElOrder)
        foo, dNdxi, foo = EvaluateReferenceShapeFunctions(XIe, self.ElType, self.ElOrder)
        for inode in range(self.n):
            Bze[inode] = self.PSIe@dNdxi[inode,:]/self.Xe[inode,0]
        return Bze
    
    def Brz(self,X):
        """
        Total magnetic field vector at point X such that    (Br, Bz) = (-1/R dpsi/dZ, 1/R dpsi/dR)
        """
        Brz = np.zeros([2])
        XI = self.InverseMapping(X)
        for i in range(self.n):
            foo, dNdxi, dNdeta = ShapeFunctionsReference(XI, self.ElType, self.ElOrder, i+1)
            Brz[0] -= dNdeta*self.PSIe[i]/X[0]
            Brz[1] += dNdxi*self.PSIe[i]/X[0]
        return Brz
    
    def Brze(self):
        """
        Elemental nodes total magnetic field vector such that    (Br, Bz) = (-1/R dpsi/dZ, 1/R dpsi/dR)
        """
        Brze = np.zeros([self.n,2])
        XIe = ReferenceElementCoordinates(self.ElType,self.ElOrder)
        foo, dNdxi, dNdeta = EvaluateReferenceShapeFunctions(XIe, self.ElType, self.ElOrder)
        for inode in range(self.n):
            Brze[inode,0] = -self.PSIe@dNdeta[inode,:]/self.Xe[inode,0]
            Brze[inode,1] = self.PSIe@dNdxi[inode,:]/self.Xe[inode,0]
        return Brze
    
    ##################################################################################################
    ######################### CUT ELEMENTS INTERFACE APPROXIMATION ###################################
    ##################################################################################################
    
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

        # GENERATE ELEMENTAL PLASMA/VACUUM INTERFACE APPROXIMATION OBJECT
        self.InterfApprox = InterfaceApprox(index = interface_index,
                                            Nsegments = self.ElOrder)
        
        # FIND POINTS ON INTERFACE USING ELEMENTAL INTERPOLATION
        #### INTERSECTION WITH EDGES
        XIintEND = np.zeros([2,2])
        self.InterfApprox.ElIntNodes = np.zeros([2,self.nedge],dtype=int)
        k = 0
        for iedge in range(self.numedges):  # Loop over elemental edges
            # Check for sign change along the edge
            inode = iedge
            jnode = (iedge + 1) % self.numedges
            if self.LSe[inode] * self.LSe[jnode] < 0:
                # FIND HIGH-ORDER NODES BETWEEN VERTICES
                #edge_index = get_edge_index(self.ElType,inode,jnode)
                self.InterfApprox.ElIntNodes[k,:2] = [inode, jnode]
                for knode in range(self.ElOrder-1):
                    self.InterfApprox.ElIntNodes[k,2+knode] = self.numedges + iedge*(self.ElOrder-1) + knode
                    
                if abs(XIe[jnode,0]-XIe[inode,0]) < 1e-6: # VERTICAL EDGE
                    #### DEFINE CONSTRAINT PHI FUNCTION
                    xi = XIe[inode,0]
                    def PHIedge(eta):
                        X = np.array([xi,eta[0]],dtype=float).reshape((1,2))
                        N, foo, foo = EvaluateReferenceShapeFunctions(X, self.ElType, self.ElOrder)
                        return N@self.LSe
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
                        N, foo, foo = EvaluateReferenceShapeFunctions(X, self.ElType, self.ElOrder)
                        return N@self.LSe
                    #### FIND INTERSECTION POINT:
                    Xi0 = 1/2  # INITIAL GUESS FOR ROOT SOLVER
                    sol = optimize.root(PHIedge, Xi0)
                    XIintEND[k,:] = [sol.x[0], edgeconstraint(sol.x[0])]
                k += 1
                    
        if self.ElOrder == 1: # LINEAR ELEMENT INTERFACE APPROXIMATION -> LINEAR APPROXIMATION 
            self.InterfApprox.XIint = XIintEND 
        else:
            #### HIGH-ORDER INTERFACE NODES
            # IN THIS CASE, WITH USE THE REGULARITY OF THE REFERENCE TRIANGLE TO FIND THE NODES
            # LYING ON THE INTERFACE INSIDE THE ELEMENT. SIMILARLY TO THE INTERSECTION NODES ON THE
            # ELEMENTAL EDGES, EACH INTERIOR NODE CAN BE FOUND BY IMPOSING TWO CONDITIONS:
            #    - PHI = 0
            #    - NODE ON LINE DIVIDING THE INTERFACE ARC
            def PHI(X):
                N, foo, foo = EvaluateReferenceShapeFunctions(X, self.ElType, self.ElOrder)
                return N@self.LSe

            def fun(X):
                F = np.zeros([X.shape[0]])
                # SEPARATE GUESS VECTOR INTO INDIVIDUAL NODAL COORDINATES
                XHO = X.reshape((self.ElOrder-1,self.dim)) 
                # PHI = 0 ON NODES
                for inode in range(self.ElOrder-1):
                    F[inode] = PHI(XHO[inode,:].reshape((1,2)))
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
            self.InterfApprox.XIint = np.concatenate((XIintEND,XIintINT),axis=0)
        
        
        ## MAP BACK TO PHYSICAL SPACE
        # EVALUATE REFERENCE SHAPE FUNCTIONS AT POINTS TO MAP (INTERFACE NODES)
        Nint, foo, foo = EvaluateReferenceShapeFunctions(self.InterfApprox.XIint, self.ElType, self.ElOrder)
        # COMPUTE SCALAR PRODUCT
        self.InterfApprox.Xint = Nint@self.Xe
        
        ## ASSOCIATE ELEMENTAL CONNECTIVITY TO INTERFACE SEGMENTS
        lnods = [0,np.arange(2,self.ElOrder+1),1]
        self.InterfApprox.Tint = list(chain.from_iterable([x] if not isinstance(x, np.ndarray) else x for x in lnods))
    
        #### GENERATE SEGMENT OBJECTS AND STORE INTERFACE DATA
        self.InterfApprox.Segments = [Segment(index = iseg,
                                    ElOrder = self.ElOrder,
                                    Tseg = None,
                                    Xseg = self.InterfApprox.Xint[self.InterfApprox.Tint[iseg:iseg+2],:],
                                    XIseg = self.InterfApprox.XIint[self.InterfApprox.Tint[iseg:iseg+2],:]) 
                                    for iseg in range(self.InterfApprox.Nsegments)]  
            
        for SEGMENT in self.InterfApprox.Segments:
            # COMPUTE INNER HIGH-ORDER NODES IN EACH SEGMENT CONFORMING THE INTERFACE APPROXIMATION (BOTH REFERENCE AND PHYSICAL SPACE)
            XsegHO = np.zeros([SEGMENT.n,SEGMENT.dim])
            XIsegHO = np.zeros([SEGMENT.n,SEGMENT.dim])
            XsegHO[:2,:] = SEGMENT.Xseg
            XIsegHO[:2,:] = SEGMENT.XIseg
            dx = (SEGMENT.Xseg[1,0]-SEGMENT.Xseg[0,0])/(SEGMENT.n-1)
            dy = (SEGMENT.Xseg[1,1]-SEGMENT.Xseg[0,1])/(SEGMENT.n-1)
            dxi = (SEGMENT.XIseg[1,0]-SEGMENT.XIseg[0,0])/(SEGMENT.n-1)
            deta = (SEGMENT.XIseg[1,1]-SEGMENT.XIseg[0,1])/(SEGMENT.n-1)
            for iinnernode in range(2,SEGMENT.n):
                XsegHO[iinnernode,:] = [SEGMENT.Xseg[0,0]+(iinnernode-1)*dx,
                                        SEGMENT.Xseg[0,1]+(iinnernode-1)*dy]
                XIsegHO[iinnernode,:] = [SEGMENT.XIseg[0,0]+(iinnernode-1)*dxi,
                                        SEGMENT.XIseg[0,1]+(iinnernode-1)*deta]
            # STORE HIGH-ORDER SEGMENT ELEMENTS CONFORMING HIGH-ORDER INTERFACE APPROXIMATION
            SEGMENT.Xseg = XsegHO
            SEGMENT.XIseg = XIsegHO
               
        return 
    
    
    ##################################################################################################
    ##################################### INTERFACE NORMALS ##########################################
    ##################################################################################################
    
    def InterfaceNormal(self):
        """ 
        This function computes the interface normal vector pointing outwards. 
        """
        # COMPUTE THE NORMAL VECTOR FOR EACH SEGMENT CONFORMING THE INTERFACE APPROXIMATION
        for SEGMENT in self.InterfApprox.Segments:
            #### PREPARE TEST NORMAL VECTOR IN PHYSICAL SPACE
            dx = SEGMENT.Xseg[1,0] - SEGMENT.Xseg[0,0]
            dy = SEGMENT.Xseg[1,1] - SEGMENT.Xseg[0,1]
            ntest_xy = np.array([-dy, dx]) 
            ntest_xy = ntest_xy/np.linalg.norm(ntest_xy) 
            
            #### PERFORM THE TEST IN REFERENCE SPACE
            # PREPARE TEST NORMAL VECTOR IN REFERENCE SPACE
            dxi = SEGMENT.XIseg[1,0] - SEGMENT.XIseg[0,0]
            deta = SEGMENT.XIseg[1,1] - SEGMENT.XIseg[0,1]
            ntest_xieta = np.array([-deta, dxi])                     # test this normal vector
            ntest_xieta = ntest_xieta/np.linalg.norm(ntest_xieta)    # normalize
            XIsegmean = np.mean(SEGMENT.XIseg, axis=0)               # mean point on interface
            XItest = XIsegmean + 0.5*ntest_xieta                     # physical point on which to test the Level-Set 
            # INTERPOLATE LEVEL-SET IN REFERENCE SPACE
            LStest = self.ElementalInterpolationREFERENCE(XItest,self.LSe)
            # CHECK SIGN OF LEVEL-SET 
            if LStest > 0:  # TEST POINT OUTSIDE PLASMA REGION
                SEGMENT.NormalVec = ntest_xy
            else:   # TEST POINT INSIDE PLASMA REGION --> NEED TO TAKE THE OPPOSITE NORMAL VECTOR
                SEGMENT.NormalVec = -1*ntest_xy
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
        
    def ComputeStandardQuadrature2D(self,NumQuadOrder):
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
        self.XIg, self.Wg, self.ng = GaussQuadrature(self.ElType,NumQuadOrder)
        
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
            
        return    
    
    
    def ComputeAdaptedQuadratures(self,NumQuadOrder):
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
        self.Nesub, SubElType, XIeTESSHO, self.interfedge = self.ReferenceElementTessellation()
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
                                    PlasmaLSe = None) for isubel in range(self.Nesub)]
        
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
            XIg2Dstand, SUBELEM.Wg, SUBELEM.ng = GaussQuadrature(SUBELEM.ElType,NumQuadOrder)
            # EVALUATE SUBELEMENTAL REFERENCE SHAPE FUNCTIONS 
            N2Dstand, foo, foo = EvaluateReferenceShapeFunctions(XIg2Dstand, SUBELEM.ElType, SUBELEM.ElOrder)
            # MAP 2D REFERENCE GAUSS INTEGRATION NODES ON THE REFERENCE SUBELEMENTS  ->> ADAPTED 2D QUADRATURE FOR SUBELEMENTS
            SUBELEM.XIg = N2Dstand @ SUBELEM.XIe
            # EVALUATE ELEMENTAL REFERENCE SHAPE FUNCTIONS ON ADAPTED REFERENCE QUADRATURE
            SUBELEM.Ng, SUBELEM.dNdxig, SUBELEM.dNdetag = EvaluateReferenceShapeFunctions(SUBELEM.XIg, self.ElType, self.ElOrder)
            # MAPP ADAPTED REFERENCE QUADRATURE ON PHYSICAL ELEMENT
            SUBELEM.Xg = SUBELEM.Ng @ self.Xe
            
            # EVALUATE INTEGRATION ENTITIES (JACOBIAN INVERSE MATRIX AND DETERMINANT) ON ADAPTED QUADRATURES NODES
            SUBELEM.invJg = np.zeros([SUBELEM.ng,SUBELEM.dim,SUBELEM.dim])
            SUBELEM.detJg = np.zeros([SUBELEM.ng])
            for ig in range(SUBELEM.ng):
                SUBELEM.invJg[ig,:,:], SUBELEM.detJg[ig] = Jacobian(self.Xe,SUBELEM.dNdxig[ig,:],SUBELEM.dNdetag[ig,:])
        
        ######### ADAPTED QUADRATURE TO INTEGRATE OVER ELEMENTAL INTERFACE APPROXIMATION (1D)
        #### STANDARD REFERENCE ELEMENT QUADRATURE TO INTEGRATE LINES (1D)
        XIg1Dstand, Wg1D, Ng1D = GaussQuadrature(0,NumQuadOrder)
        #### QUADRATURE TO INTEGRATE LINES (1D)
        N1D, dNdxi1D, foo = EvaluateReferenceShapeFunctions(XIg1Dstand, 0, self.nedge-1)
                
        for SEGMENT in self.InterfApprox.Segments:
            SEGMENT.ng = Ng1D
            SEGMENT.Wg = Wg1D
            SEGMENT.detJg = np.zeros([SEGMENT.ng])
            # MAP 1D REFERENCE STANDARD GAUSS INTEGRATION NODES ON THE REFERENCE INTERFACE ->> ADAPTED 1D QUADRATURE FOR INTERFACE
            SEGMENT.XIg = N1D @ SEGMENT.XIseg
            # EVALUATE 2D REFERENCE SHAPE FUNCTION ON INTERFACE ADAPTED QUADRATURE
            SEGMENT.Ng, SEGMENT.dNdxig, SEGMENT.dNdetag = EvaluateReferenceShapeFunctions(SEGMENT.XIg, self.ElType, self.ElOrder)
            # MAPP REFERENCE INTERFACE ADAPTED QUADRATURE ON PHYSICAL ELEMENT 
            SEGMENT.Xg = N1D @ SEGMENT.Xseg
            for ig in range(SEGMENT.ng):
                SEGMENT.detJg[ig] = Jacobian1D(SEGMENT.Xseg,dNdxi1D[ig,:]) 
                    
        return 
    
    
    def ComputeGhostFacesQuadratures(self,NumQuadOrder):
        
        ######### ADAPTED QUADRATURE TO INTEGRATE OVER ELEMENTAL GHOST FACES
        #### STANDARD REFERENCE ELEMENT QUADRATURE TO INTEGRATE LINES (1D)
        XIg1Dstand, Wg1D, Ng1D = GaussQuadrature(0,NumQuadOrder)
        #### QUADRATURE TO INTEGRATE LINES (1D)
        N1D, dNdxi1D, foo = EvaluateReferenceShapeFunctions(XIg1Dstand, 0, self.nedge-1)
                    
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
            FACE.Ng = FACE.Ng[:,FACE.Tseg]
            FACE.dNdxig = FACE.dNdxig[:,FACE.Tseg]
            FACE.dNdetag = FACE.dNdetag[:,FACE.Tseg]
            # MAPP REFERENCE INTERFACE ADAPTED QUADRATURE ON PHYSICAL ELEMENT 
            FACE.Xg = N1D @ FACE.Xseg
            for ig in range(FACE.ng):
                FACE.detJg[ig] = Jacobian1D(FACE.Xseg,dNdxi1D[ig,:]) 
        
        return
    
    
    ##################################################################################################
    ################################ ELEMENTAL INTEGRATION ###########################################
    ##################################################################################################
    
    def IntegrateElementalDomainTerms(self,SourceTermg,*args):
        """ 
        This function computes the elemental contributions to the global system by integrating the source terms over 
        the elemental domain. It calculates the left-hand side (LHS) matrix and right-hand side (RHS) vector using 
        Gauss integration nodes.

        Input:
            - SourceTermg (ndarray): The Grad-Shafranov equation source term evaluated at the physical Gauss integration nodes.
        
            - *args (tuple, optional): Additional arguments for specific cases, such as the dimensionless solution case where 
                                `args[0]` might represent a scaling factor (R0).

        This function computes:
            1. The elemental contributions to the LHS matrix (stiffness term and gradient term).
            2. The elemental contributions to the RHS vector (source term).

        Output:
            - LHSe (ndarray): The elemental left-hand side matrix (stiffness matrix) of the system.
            - RHSe (ndarray): The elemental right-hand side vector of the system.

        The function loops over Gauss integration nodes to compute these contributions and assemble the elemental system.
        """
                    
        LHSe = np.zeros([len(self.Te),len(self.Te)])
        RHSe = np.zeros([len(self.Te)])
        
        # LOOP OVER GAUSS INTEGRATION NODES
        for ig in range(self.ng):  
            # SHAPE FUNCTIONS GRADIENT IN PHYSICAL SPACE
            Ngrad = self.invJg[ig,:,:]@np.array([self.dNdxig[ig,:],self.dNdetag[ig,:]])
            R = self.Xg[ig,0]
            if args:   # DIMENSIONLESS SOLUTION CASE  -->> args[0] = R0
                Ngrad *= args[0]
                R /= args[0]
            # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM 
            for i in range(len(self.Te)):   # ROWS ELEMENTAL MATRIX
                for j in range(len(self.Te)):   # COLUMNS ELEMENTAL MATRIX
                    # COMPUTE LHS MATRIX TERMS
                    ### STIFFNESS TERM  [ nabla(N_i)*nabla(N_j) ]  
                    LHSe[i,j] -= Ngrad[:,j]@Ngrad[:,i]*self.detJg[ig]*self.Wg[ig]
                    ### GRADIENT TERM (ASYMMETRIC)  [ (1/R)*N_i*dNdr_j ]  ONLY RESPECT TO R
                    LHSe[i,j] += (1/R)*self.Ng[ig,j]*Ngrad[0,i]*self.detJg[ig]*self.Wg[ig]
                # COMPUTE RHS VECTOR TERMS [ (source term)*N_i ]
                RHSe[i] += SourceTermg[ig] * self.Ng[ig,i] *self.detJg[ig]*self.Wg[ig]
                
        return LHSe, RHSe
    
    
    def IntegrateElementalInterfaceTerms(self,beta,*args):
        """ 
        This function computes the elemental contributions to the global system from the interface terms, using 
        Nitsche's method. It integrates the interface conditions over the elemental interface approximation segments. 
        It calculates the left-hand side (LHS) matrix and right-hand side (RHS) vector using Gauss integration nodes.

        Input:
            - beta (float): The penalty parameter for Nitsche's method, which controls the strength of the penalty term.
        
            - *args (tuple, optional): Additional arguments for specific cases, such as the dimensionless solution case where 
                                `args[0]` might represent a scaling factor (R0).

        This function computes:
            1. The elemental contributions to the LHS matrix (including Dirichlet boundary term, symmetric Nitsche's term, and penalty term).
            2. The elemental contributions to the RHS vector (including symmetric Nitsche's term and penalty term).

        Output: 
            - LHSe (ndarray): The elemental left-hand side matrix (stiffness matrix) of the system, incorporating Nitsche's method.
            - RHSe (ndarray): The elemental right-hand side vector of the system, incorporating Nitsche's method.

        The function loops over interface segments and Gauss integration nodes to compute these contributions and assemble the global system.
        """
        
        LHSe = np.zeros([len(self.Te),len(self.Te)])
        RHSe = np.zeros([len(self.Te)])
    
        # LOOP OVER SEGMENTS CONSTITUTING THE INTERFACE ELEMENTAL APPROXIMATION
        for SEGMENT in self.InterfApprox.Segments:
            # LOOP OVER GAUSS INTEGRATION NODES
            for ig in range(SEGMENT.ng):  
                # SHAPE FUNCTIONS GRADIENT IN PHYSICAL SPACE
                n_dot_Ngrad = SEGMENT.NormalVec@np.array([SEGMENT.dNdxig[ig,:],SEGMENT.dNdetag[ig,:]])
                if args:   # DIMENSIONLESS SOLUTION CASE  -->> args[0] = R0
                    n_dot_Ngrad *= args[0]
                    
                # COMPUTE ELEMENTAL CONTRIBUTIONS AND ASSEMBLE GLOBAL SYSTEM
                for i in range(len(self.Te)):  # ROWS ELEMENTAL MATRIX
                    for j in range(len(self.Te)):  # COLUMNS ELEMENTAL MATRIX
                        # COMPUTE LHS MATRIX TERMS
                        ### DIRICHLET BOUNDARY TERM  [ N_i*(n dot nabla(N_j)) ]  
                        LHSe[i,j] += SEGMENT.Ng[ig,i] * n_dot_Ngrad[j] * SEGMENT.detJg[ig] * SEGMENT.Wg[ig]
                        ### SYMMETRIC NITSCHE'S METHOD TERM   [ N_j*(n dot nabla(N_i)) ]
                        LHSe[i,j] += n_dot_Ngrad[i]*SEGMENT.Ng[ig,j] * SEGMENT.detJg[ig] * SEGMENT.Wg[ig]
                        ### PENALTY TERM   [ beta * (N_i*N_j) ]
                        LHSe[i,j] += beta * SEGMENT.Ng[ig,i] * SEGMENT.Ng[ig,j] * SEGMENT.detJg[ig] * SEGMENT.Wg[ig]
                    # COMPUTE RHS VECTOR TERMS 
                    ### SYMMETRIC NITSCHE'S METHOD TERM  [ PSI_D * (n dot nabla(N_i)) ]
                    RHSe[i] +=  SEGMENT.PSIgseg[ig] * n_dot_Ngrad[i] * SEGMENT.detJg[ig] * SEGMENT.Wg[ig]
                    ### PENALTY TERM   [ beta * N_i * PSI_D ]
                    RHSe[i] +=  beta * SEGMENT.PSIgseg[ig] * SEGMENT.Ng[ig,i] * SEGMENT.detJg[ig] * SEGMENT.Wg[ig]
        
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
    
