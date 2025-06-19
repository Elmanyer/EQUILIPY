import numpy as np
from os.path import basename
from Element import *

class Mesh:
    
    def __init__(self,path_to_folder):
        
        self.name = basename(path_to_folder)
        self.mesh_directory = path_to_folder
        
        # ATTRIBUTES
        self.ElTypeALYA = None              # TYPE OF ELEMENTS CONSTITUTING THE MESH, USING ALYA NOTATION
        self.ElType = None                  # TYPE OF ELEMENTS CONSTITUTING THE MESH: 1: TRIANGLES,  2: QUADRILATERALS
        self.ElOrder = None                 # ORDER OF MESH ELEMENTS: 1: LINEAR,   2: QUADRATIC,   3: CUBIC
        self.X = None                       # MESH NODAL COORDINATES MATRIX
        self.T = None                       # MESH ELEMENTS CONNECTIVITY MATRIX 
        self.Nn = None                      # TOTAL NUMBER OF MESH NODES
        self.Ne = None                      # TOTAL NUMBER OF MESH ELEMENTS
        self.n = None                       # NUMBER OF NODES PER ELEMENT
        self.numedges = None                # NUMBER OF EDGES PER ELEMENT (= 3 IF TRIANGULAR; = 4 IF QUADRILATERAL)
        self.nedge = None                   # NUMBER OF NODES ON ELEMENTAL EDGE
        self.dim = None                     # SPACE DIMENSION
        self.Tbound = None                  # MESH BOUNDARIES CONNECTIVITY MATRIX  (LAST COLUMN YIELDS THE ELEMENT INDEX OF THE CORRESPONDING BOUNDARY EDGE)
        self.Nbound = None                  # NUMBER OF COMPUTATIONAL DOMAIN'S BOUNDARIES (NUMBER OF ELEMENTAL EDGES)
        self.Nnbound = None                 # NUMBER OF NODES ON COMPUTATIONAL DOMAIN'S BOUNDARY
        self.BoundaryNodes = None           # LIST OF NODES (GLOBAL INDEXES) ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        self.BoundaryVertices = None        # LIST OF CONSECUTIVE NODES (GLOBAL INDEXES) WHICH ARE COMPUTATIONAL DOMAIN'S BOUNDARY VERTICES 
        self.DOFNodes = None                # LIST OF NODES (GLOBAL INDEXES) CORRESPONDING TO UNKNOW DEGREES OF FREEDOM IN THE CUTFEM SYSTEM
        self.NnDOF = None                   # NUMBER OF UNKNOWN DEGREES OF FREEDOM NODES
        self.PlasmaNodes = None             # LIST OF NODES (GLOBAL INDEXES) INSIDE THE PLASMA DOMAIN
        self.VacuumNodes = None             # LIST OF NODES (GLOBAL INDEXES) IN THE VACUUM REGION
        self.NnPB = None                    # NUMBER OF NODES ON PLASMA BOUNDARY APPROXIMATION
        self.PlasmaElems = None             # LIST OF ELEMENTS (INDEXES) INSIDE PLASMA REGION
        self.VacuumElems = None             # LIST OF ELEMENTS (INDEXES) OUTSIDE PLASMA REGION (VACUUM REGION)
        self.PlasmaBoundElems = None        # LIST OF CUT ELEMENT'S INDEXES, CONTAINING INTERFACE BETWEEN PLASMA AND VACUUM
        self.FirstWallElems = None          # LIST OF CUT (OR NOT) ELEMENT'S INDEXES, CONTAINING VACUUM VESSEL FIRST WALL (OR COMPUTATIONAL DOMAIN'S BOUNDARY)
        self.NonCutElems = None             # LIST OF ALL NON CUT ELEMENTS
        self.Elements = None                # ARRAY CONTAINING ALL ELEMENTS IN MESH (PYTHON OBJECTS)
        
        self.Rmax = None                    # COMPUTATIONAL MESH MAXIMAL X (R) COORDINATE
        self.Rmin = None                    # COMPUTATIONAL MESH MINIMAL X (R) COORDINATE
        self.Zmax = None                    # COMPUTATIONAL MESH MAXIMAL Y (Z) COORDINATE
        self.Zmin = None                    # COMPUTATIONAL MESH MINIMAL Y (Z) COORDINATE
        self.meanArea = None                # MESH ELEMENTS MEAN AREA
        self.meanLength = None              # MESH ELEMENTS MEAN LENTH
        self.nge = None                     # NUMBER OF INTEGRATION NODES PER ELEMENT (STANDARD SURFACE QUADRATURE)
        
        self.GhostFaces = None              # LIST OF PLASMA BOUNDARY GHOST FACES
        self.GhostElems = None              # LIST OF ELEMENTS CONTAINING PLASMA BOUNDARY FACES
        return
    
    def ALYA2Py(self):
        """ 
        Translate ALYA elemental type to EQUILIPY elemental ElType and ElOrder.
        """
        match self.ElTypeALYA:
            case 2:
                self.ElType = 0
                self.ElOrder = 1
            case 3:
                self.ElType = 0
                self.ElOrder = 2
            case 4:
                self.ElType = 0
                self.ElOrder = 3
            case 10:
                self.ElType = 1
                self.ElOrder = 1
            case 11:
                self.ElType = 1
                self.ElOrder = 2
            case 16:
                self.ElType = 1
                self.ElOrder = 3
            case 12:
                self.ElType = 2
                self.ElOrder = 1
            case 14:
                self.ElType = 2
                self.ElOrder = 2
            case 15:
                self.ElType = 2
                self.ElOrder = 3
        return
    
    
    ##################################################################################################
    ############################### READ INPUT DATA FILES ############################################
    ##################################################################################################
    
    def ReadMeshFile(self):
        """ 
        Read input mesh data files, .dom.dat and .geo.dat, and build mesh simulation attributes. 
        """
        
        print("     -> READ MESH DATA FILES...",end='')
        # READ DOM FILE .dom.dat
        MeshDataFile = self.mesh_folder +'/' + self.MESH +'.dom.dat'
        self.Nn = 0   # number of nodes
        self.Ne = 0   # number of elements
        file = open(MeshDataFile, 'r') 
        for line in file:
            l = line.split('=')
            if l[0] == '  NODAL_POINTS':  # read number of nodes
                self.Nn = int(l[1])
            elif l[0] == '  ELEMENTS':  # read number of elements
                self.Ne = int(l[1])
            elif l[0] == '  SPACE_DIMENSIONS':  # read space dimensions 
                self.dim = int(l[1])
            elif l[0] == '  TYPES_OF_ELEMENTS':
                self.ElTypeALYA = int(l[1])
            elif l[0] == '  BOUNDARIES':  # read number of boundaries
                self.Nbound = int(l[1])
        file.close()
        
        self.ALYA2Py()
        
        # NUMBER OF NODES PER ELEMENT
        self.n, self.nedge = ElementalNumberOfNodes(self.ElType, self.ElOrder)
        if self.ElType == 1:
            self.numedges = 3
        elif self.ElType == 2:
            self.numedges = 4
        
        # READ MESH FILE .geo.dat
        MeshFile = self.mesh_folder +'/'+ self.MESH +'.geo.dat'
        self.T = np.zeros([self.Ne,self.n], dtype = int)
        self.X = np.zeros([self.Nn,self.dim], dtype = float)
        self.Tbound = np.zeros([self.Nbound,self.nedge+1], dtype = int)   # LAST COLUMN YIELDS THE ELEMENT INDEX OF THE CORRESPONDING BOUNDARY EDGE 
        file = open(MeshFile, 'r') 
        i = -1
        j = -1
        k = -1
        for line in file:
            # first we format the line read in order to remove all the '\n'  
            l = line.split(' ')
            l = [m for m in l if m != '']
            for e, el in enumerate(l):
                if el == '\n':
                    l.remove('\n') 
                elif el[-1:]=='\n':
                    l[e]=el[:-1]
            # WE IDENTIFY WHEN THE CONNECTIVITY MATRIX DATA STARTS
            if l[0] == 'ELEMENTS':
                i=0
                continue
            # WE IDENTIFY WHEN THE CONNECTIVITY MATRIX DATA ENDS
            elif l[0] == 'END_ELEMENTS':
                i=-1
                continue
            # WE IDENTIFY WHEN THE NODAL COORDINATES DATA STARTS
            elif l[0] == 'COORDINATES':
                j=0
                continue
            # WE IDENTIFY WHEN THE NODAL COORDINATES DATA ENDS
            elif l[0] == 'END_COORDINATES':
                j=-1
                continue
            # WE IDENTIFY WHEN THE COMPUTATIONAL DOMAIN'S BOUNDARY DATA STARTS
            elif l[0] == 'BOUNDARIES,':
                k=0
                continue
            # WE IDENTIFY WHEN THE COMPUTATIONAL DOMAIN'S BOUNDARY DATA ENDS
            elif l[0] == 'END_BOUNDARIES':
                k=-1
                continue
            if i>=0:
                for m in range(self.n):
                    self.T[i,m] = int(l[m+1])
                i += 1
            if j>=0:
                for m in range(self.dim):
                    self.X[j,m] = float(l[m+1])
                j += 1
            if k>=0:
                for m in range(self.nedge+1):
                    self.Tbound[k,m] = int(l[m+1])
                k += 1
        file.close()
        # PYTHON INDEXES START AT 0 AND NOT AT 1. THUS, THE CONNECTIVITY MATRIX INDEXES MUST BE ADAPTED
        self.T = self.T - 1
        self.Tbound = self.Tbound - 1
        
        # ORGANISE BOUNDARY ATTRIBUTES
        self.BoundaryAttributes()
        print('Done!')
        
        # OBTAIN COMPUTATIONAL MESH LIMITS
        self.Rmax = np.max(self.X[:,0])
        self.Rmin = np.min(self.X[:,0])
        self.Zmax = np.max(self.X[:,1])
        self.Zmin = np.min(self.X[:,1])
        return
    

    def ReadFixFile(self):
        """
        Read fix set data from input file .fix.dat. 
        """
        print("     -> READ FIX DATA FILE...",end='')
        # READ EQU FILE .equ.dat
        FixDataFile = self.mesh_folder +'/' + self.MESH +'.fix.dat'
        file = open(FixDataFile, 'r') 
        self.BoundaryIden = np.zeros([self.Nbound],dtype=int)
        for line in file:
            l = line.split(' ')
            l = [m for m in l if m != '']
            for e, el in enumerate(l):
                if el == '\n':
                    l.remove('\n') 
                elif el[-1:]=='\n':
                    l[e]=el[:-1]
            
            if l[0] == "ON_BOUNDARIES" or l[0] == "END_ON_BOUNDARIES":
                pass
            else:
                self.BoundaryIden[int(l[0])-1] = int(l[1])
                
        # DEFINE THE DIFFERENT SETS OF BOUNDARY NODES
        self.BoundaryNodesSets = [set(),set()]
        for iboun in range(self.Nbound):
            for node in self.Tbound[iboun,:-1]:
                self.BoundaryNodesSets[self.BoundaryIden[iboun]-1].add(node)
        # CONVERT BOUNDARY NODES SET INTO ARRAY
        self.BoundaryNodesSets[0] = np.array(sorted(self.BoundaryNodesSets[0]))
        self.BoundaryNodesSets[1] = np.array(sorted(self.BoundaryNodesSets[1]))
        print('Done!')
        return
    
    
    def BoundaryAttributes(self):
    
        # OBTAIN BOUNDARY NODES
        self.BoundaryNodes = set()     # GLOBAL INDEXES OF NODES ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        for i in range(self.nedge):
            for node in self.Tbound[:,i]:
                self.BoundaryNodes.add(node)
        # CONVERT BOUNDARY NODES SET INTO ARRAY
        self.BoundaryNodes = list(sorted(self.BoundaryNodes))
        self.Nnbound = len(self.BoundaryNodes)
        
        # OBTAIN DOF NODES
        self.DOFNodes =  [x for x in list(range(self.Nn)) if x not in set(self.BoundaryNodes)]
        self.NnDOF = len(self.DOFNodes)
        
        # OBTAIN BOUNDARY ELEMENTS
        self.BoundaryElems = np.unique(self.Tbound[:,-1]) 
        
        # OBTAIN BOUNDARY NODE PATH (CONSECUTIVE BOUNDARY NODAL VERTICES)
        self.BoundaryVertices = np.zeros([self.Nbound],dtype=int)
        for iboun in range(self.Nbound-1):
            if iboun == 0:  # INITIALIZATION
                self.BoundaryVertices[0] = self.Tbound[0,0]
                self.BoundaryVertices[1] = self.Tbound[0,1]
            else:
                # LOOK FOR ADJACENT BOUNDARY
                bounnode = np.where(self.Tbound[:,:2] == self.BoundaryVertices[iboun])
                if self.BoundaryVertices[iboun-1] in self.Tbound[bounnode[0][0],bounnode[1]]:
                    nextbounelem = bounnode[0][1]
                else:
                    nextbounelem = bounnode[0][0]
                self.BoundaryVertices[iboun+1] = self.Tbound[nextbounelem,np.where(self.Tbound[nextbounelem,:2] != self.BoundaryVertices[iboun])[0]]
        return
    
    
    def ComputeMeshElementsMeanSize(self):
        
        # COMPUTE MEAN AREA OF ELEMENT
        meanArea = 0
        meanLength = 0
        for ELEMENT in self.Elements:
            meanArea += ELEMENT.area
            meanLength += ELEMENT.length
        meanArea /= self.Ne
        meanLength /= self.Ne
        
        return meanArea, meanLength
    
    
    ##################################################################################################
    ###################################### ELEMENTS DEFINITION #######################################
    ##################################################################################################
    
    def IdentifyNearestNeighbors(self):
        """
        Finds the nearest neighbours for each element in mesh.
        """
        edgemap = {}  # Dictionary to map edges to elements

        # Loop over all elements to populate the edgemap dictionary
        for ELEMENT in self.Elements:
            # Initiate elemental nearest neigbhors attribute
            ELEMENT.neighbours = -np.ones([self.numedges],dtype=int)
            # Get the edges of the element (as sorted tuples for uniqueness)
            edges = list()
            for iedge in range(ELEMENT.numedges):
                edges.append(tuple(sorted([ELEMENT.Te[iedge],ELEMENT.Te[int((iedge+1)%ELEMENT.numedges)]])))
            
            for local_edge_idx, edge in enumerate(edges):
                if edge not in edgemap:
                    edgemap[edge] = []
                edgemap[edge].append((ELEMENT.index, local_edge_idx))
        
        # Loop over the edge dictionary to find neighbours
        for edge, elements in edgemap.items():
            if len(elements) == 2:  # Shared edge between two elements
                (elem1, edge1), (elem2, edge2) = elements
                self.Elements[elem1].neighbours[edge1] = elem2
                self.Elements[elem2].neighbours[edge2] = elem1
        return
    
    def IdentifyBoundaries(self):
        
        #### ASSIGN BOUNDARY CONNECTIVITIES
        # BOUNDARY ELEMENTS
        for ielem in self.BoundaryElems:
            self.Elements[ielem].Teboun = list()
        # LOOP OVER BOUNDARIES
        for iboun, ibounelem in enumerate(self.Tbound[:,-1]):
            # OBTAIN LOCAL INDEXES
            Teboun = [index for index, global_index in enumerate(self.Elements[ibounelem].Te) 
                      for bound_index in self.Tbound[iboun,:-1] if global_index == bound_index]
            self.Elements[ibounelem].Teboun.append(Teboun)
            
        # ELEMENTS WHICH ARE NOT BOUNDARY ELEMENTS (NONE OF THEIR EDGES IS A BOUNDARY EDGE) BUT POSSES BOUNDARY NODES
        for bounnode in self.BoundaryNodes:
            bounconnec = np.where(self.T == bounnode)
            for iboun, bounelem in enumerate(bounconnec[0]):
                if type(self.Elements[bounelem].Teboun) == type(None):
                    self.Elements[bounelem].Teboun = list()
                    self.Elements[bounelem].Teboun.append([bounconnec[1][iboun]])
                    
        # LIST ALL ELEMENTS WITH BOUNDARY NODES
        self.DirichletElems = list()
        for ELEM in self.Elements:
            if type(ELEM.Teboun) != type(None):
                self.DirichletElems.append(ELEM.index)
            
        # VACUUM VESSEL WALL ELEMENTS (BY DEFAULT)
        self.FirstWallElems = self.BoundaryElems.copy()
        # ASSIGN ELEMENTAL DOMAIN FLAG
        for ielem in self.FirstWallElems:
            self.Elements[ielem].Dom = 2  
  
        return
    
    
    def ClassifyElements(self):
        """ 
        Function that separates the elements inside vacuum vessel domain into 3 groups: 
                - PlasmaElems: elements inside the plasma region 
                - PlasmaBoundElems: (cut) elements containing the plasma region boundary 
                - VacuumElems: elements outside the plasma region which are not the vacuum vessel boundary
        """
                
        """ FOR HIGH ORDER ELEMENTS (QUADRATIC, CUBIC...), ELEMENTS LYING ON GEOMETRY BOUNDARIES OR INTERFACES MAY BE CLASSIFIED AS SUCH DUE TO
        THE LEVEL-SET SIGN ON NODES WHICH ARE NOT VERTICES OF THE ELEMENT ('HIGH ORDER' NODES). IN CASES WHERE ON SUCH NODES POSSES DIFFERENT SIGN, THIS MAY LEAD TO AN INTERFACE
        WHICH CUTS TWICE THE SAME ELEMENTAL EDGE, MEANING THE INTERFACE ENTERS AND LEAVES THROUGH THE SAME SEGMENT. THE PROBLEM WITH THAT IS THE SUBROUTINE 
        RESPONSIBLE FOR APPROXIMATING THE INTERFACE INSIDE ELEMENTS ONLY SEES THE LEVEL-SET VALUES ON THE VERTICES, BECAUSE IT DOES ONLY BOTHER ON WHETHER THE 
        ELEMENTAL EDGE IS CUT OR NOT. 
        IN LIGHT OF SUCH OCURRENCES, THE CLASSIFICATION OF ELEMENTS BASED ON LEVEL-SET SIGNS WILL BE IMPLEMENTED SUCH THAT ONLY THE VALUES ON THE VERTICES ARE
        TAKEN INTO ACCOUNT. THAT WAY, THIS CASES ARE ELUDED. ON THE OTHER HAND, WE NEED TO DETECT ALSO SUCH CASES IN ORDER TO MODIFY THE VALUES OF THE MESH 
        LEVEL-SET VALUES AND ALSO ON THE ELEMENTAL VALUES. """
        
        self.PlasmaElems = np.zeros([self.Ne], dtype=int)        # GLOBAL INDEXES OF ELEMENTS INSIDE PLASMA REGION
        self.VacuumElems = np.zeros([self.Ne], dtype=int)        # GLOBAL INDEXES OF ELEMENTS OUTSIDE PLASMA REGION (VACUUM REGION)
        self.PlasmaBoundElems = np.zeros([self.Ne], dtype=int)   # GLOBAL INDEXES OF ELEMENTS CONTAINING PLASMA BOUNDARY
        
        kplasm = 0
        kvacuu = 0
        kint = 0
        
        def CheckElementalVerticesLevelSetSigns(LSe):
            region = None
            DiffHighOrderNodes = []
            # CHECK SIGN OF LEVEL SET ON ELEMENTAL VERTICES
            for i in range(self.numedges-1):
                # FIND ELEMENTS LYING ON THE INTERFACE (LEVEL-SET VERTICES VALUES EQUAL TO 0 OR WITH DIFFERENT SIGN)
                if LSe[i] == 0:  # if node is on Level-Set 0 contour
                    region = 0
                    break
                elif np.sign(LSe[i]) !=  np.sign(LSe[i+1]):  # if the sign between vertices values change -> INTERFACE ELEMENT
                    region = 0
                    break
                # FIND ELEMENTS LYING INSIDE A SPECIFIC REGION (LEVEL-SET VERTICES VALUES WITH SAME SIGN)
                else:
                    if i+2 == self.numedges:   # if all vertices values have the same sign
                        # LOCATE ON WHICH REGION LIES THE ELEMENT
                        if np.sign(LSe[i+1]) > 0:   # all vertices values with positive sign -> EXTERIOR REGION ELEMENT
                            region = +1
                        else:   # all vertices values with negative sign -> INTERIOR REGION ELEMENT 
                            region = -1
                            
                        # CHECK LEVEL-SET SIGN ON ELEMENTAL 'HIGH ORDER' NODES
                        #for i in range(self.numedges,self.n-self.numedges):  # LOOP OVER NODES WHICH ARE NOT ON VERTICES
                        for i in range(self.numedges,self.n):
                            if np.sign(LSe[i]) != np.sign(LSe[0]):
                                DiffHighOrderNodes.append(i)
                
            return region, DiffHighOrderNodes
            
        for ielem in range(self.Ne):
            regionplasma, DHONplasma = CheckElementalVerticesLevelSetSigns(self.Elements[ielem].LSe)
            if regionplasma < 0:   # ALL PLASMA LEVEL-SET NODAL VALUES NEGATIVE -> INSIDE PLASMA DOMAIN 
                # ALREADY CLASSIFIED AS VACUUM VESSEL ELEMENT (= BOUNDARY ELEMENT)
                if self.Elements[ielem].Dom == 2:  
                    # REMOVE FROM VACUUM VESSEL WALL ELEMENT LIST
                    self.FirstWallElems = self.FirstWallElems[self.FirstWallElems != ielem]
                # REDEFINE CLASSIFICATION
                self.PlasmaElems[kplasm] = ielem
                self.Elements[ielem].Dom = -1
                kplasm += 1
            elif regionplasma == 0:  # DIFFERENT SIGN IN PLASMA LEVEL-SET NODAL VALUES -> PLASMA/VACUUM INTERFACE ELEMENT
                # ALREADY CLASSIFIED AS VACUUM VESSEL ELEMENT (= BOUNDARY ELEMENT)
                if self.Elements[ielem].Dom == 2:  
                    # REMOVE FROM VACUUM VESSEL WALL ELEMENT LIST
                    self.FirstWallElems = self.FirstWallElems[self.FirstWallElems != ielem]
                # REDEFINE CLASSIFICATION
                self.PlasmaBoundElems[kint] = ielem
                self.Elements[ielem].Dom = 0
                kint += 1
            elif regionplasma > 0: # ALL PLASMA LEVEL-SET NODAL VALUES POSITIVE -> OUTSIDE PLASMA DOMAIN
                if self.Elements[ielem].Dom == 2:  # ALREADY CLASSIFIED AS VACUUM VESSEL ELEMENT (= BOUNDARY ELEMENT)
                    continue
                else:
                    # VACUUM ELEMENTS
                    self.VacuumElems[kvacuu] = ielem
                    self.Elements[ielem].Dom = +1
                    kvacuu += 1
                    
            # IF THERE EXISTS 'HIGH-ORDER' NODES WITH DIFFERENT PLASMA LEVEL-SET SIGN
            if DHONplasma:
                for inode in DHONplasma:  # LOOP OVER LOCAL INDICES 
                    self.Elements[ielem].LSe[inode] *= -1 
                    self.PlasmaLS[self.Elements[ielem].Te[inode],1] *= -1       
        
        # DELETE REST OF UNUSED MEMORY
        self.PlasmaElems = self.PlasmaElems[:kplasm]
        self.VacuumElems = self.VacuumElems[:kvacuu]
        self.PlasmaBoundElems = self.PlasmaBoundElems[:kint]
        
        # GATHER NON-CUT ELEMENTS  
        self.NonCutElems = np.concatenate((self.PlasmaElems, self.VacuumElems, self.FirstWallElems), axis=0)
        
        if len(self.NonCutElems) + len(self.PlasmaBoundElems) != self.Ne:
            raise ValueError("Non-cut elements + Cut elements =/= Total number of elements  --> Wrong mesh classification!!")
        
        # CLASSIFY NODES ACCORDING TO NEW ELEMENT CLASSIFICATION
        self.ClassifyNodes()
        return
    
    
    def ObtainClassification(self):
        """ 
        Function which produces an array where the different values code for the groups in which elements are classified. 
        """
        Classification = np.zeros([self.Ne],dtype=int)
        for ielem in self.PlasmaElems:
            Classification[ielem] = -1
        for ielem in self.PlasmaBoundElems:
            Classification[ielem] = 0
        for ielem in self.VacuumElems:
            Classification[ielem] = +1
        for ielem in self.FirstWallElems:
            Classification[ielem] = +2
            
        return Classification
    
    
    def ClassifyNodes(self):
        self.PlasmaNodes = set()
        self.VacuumNodes = set()
        for ielem in self.PlasmaElems:
            for node in self.T[ielem,:]:
                self.PlasmaNodes.add(node) 
        for ielem in self.VacuumElems:
            for node in self.T[ielem,:]:
                self.VacuumNodes.add(node)    
        for ielem in self.PlasmaBoundElems:
            for node in self.T[ielem,:]:
                if self.PlasmaLS[node,1] < 0:
                    self.PlasmaNodes.add(node)
                else:
                    self.VacuumNodes.add(node)
        for ielem in self.FirstWallElems:
            for node in self.T[ielem,:]:
                self.VacuumNodes.add(node)
        for node in self.BoundaryNodes:
            if node in self.VacuumNodes:
                self.VacuumNodes.remove(node)   
        
        self.PlasmaNodes = np.array(sorted(list(self.PlasmaNodes)))
        self.VacuumNodes = np.array(sorted(list(self.VacuumNodes)))
        
        if self.out_pickle:
            self.PlasmaNodes_sim.append(self.PlasmaNodes.copy())
            self.VacuumNodes_sim.append(self.VacuumNodes.copy())
        return
    
    
    def ComputePlasmaBoundaryNumberNodes(self):
        """
        Computes the total number of nodes located on the plasma boundary approximation
        """  
        nnodes = 0
        for ielem in self.PlasmaBoundElems:
            nnodes += self.Elements[ielem].InterfApprox.ng
        return nnodes
    
    
    def IdentifyPlasmaBoundaryGhostFaces(self):
        """
        Identifies the elemental ghost faces on which the ghost penalty term needs to be integrated, for elements containing the plasma
        boundary.
        """
        
        # RESET ELEMENTAL GHOST FACES
        for ielem in np.concatenate((self.PlasmaBoundElems,self.PlasmaElems),axis=0):
            self.Elements[ielem].GhostFaces = None
            
        GhostFaces_dict = dict()    # [(CUT_EDGE_NODAL_GLOBAL_INDEXES): {(ELEMENT_INDEX_1, EDGE_INDEX_1), (ELEMENT_INDEX_2, EDGE_INDEX_2)}]
        
        for ielem in self.PlasmaBoundElems:
            ELEMENT = self.Elements[ielem]
            for iedge, neighbour in enumerate(ELEMENT.neighbours):
                if neighbour >= 0:
                    # IDENTIFY THE CORRESPONDING FACE IN THE NEIGHBOUR ELEMENT
                    NEIGHBOUR = self.Elements[neighbour]
                    neighbour_edge = list(NEIGHBOUR.neighbours).index(ELEMENT.index)
                    # OBTAIN GLOBAL INDICES OF GHOST FACE NODES
                    nodes = np.zeros([ELEMENT.nedge],dtype=int)
                    nodes[0] = iedge
                    nodes[1] = (iedge+1)%ELEMENT.numedges
                    for knode in range(self.ElOrder-1):
                        nodes[2+knode] = self.numedges + iedge*(self.ElOrder-1)+knode
                    if tuple(sorted(ELEMENT.Te[nodes])) not in GhostFaces_dict:
                        GhostFaces_dict[tuple(sorted(ELEMENT.Te[nodes]))] = set()
                    GhostFaces_dict[tuple(sorted(ELEMENT.Te[nodes]))].add((ELEMENT.index,iedge))
                    GhostFaces_dict[tuple(sorted(ELEMENT.Te[nodes]))].add((neighbour,neighbour_edge))
                    
        XIe = ReferenceElementCoordinates(self.ElType,self.ElOrder)
        GhostFaces = list()
        GhostElems = set()

        for elems in GhostFaces_dict.values():
            # ISOLATE ADJACENT ELEMENTS
            (ielem1, iedge1), (ielem2,iedge2) = elems
            ELEM1 = self.Elements[ielem1]
            ELEM2 = self.Elements[ielem2]
            # DECLARE NEW GHOST FACES ELEMENTAL ATTRIBUTE
            if type(ELEM1.GhostFaces) == type(None):  
                ELEM1.GhostFaces = list()
            if type(ELEM2.GhostFaces) == type(None):  
                ELEM2.GhostFaces = list()
                
            # ADD GHOST FACE TO ELEMENT 1
            nodes1 = np.zeros([ELEM1.nedge],dtype=int)
            nodes1[0] = iedge1
            nodes1[1] = (iedge1+1)%ELEM1.numedges
            for knode in range(ELEM1.ElOrder-1):
                nodes1[2+knode] = ELEM1.numedges + iedge1*(ELEM1.ElOrder-1)+knode
            ELEM1.GhostFaces.append(Segment(index = iedge1,
                                            ElOrder = ELEM1.ElOrder,
                                            Tseg = nodes1,
                                            Xseg = ELEM1.Xe[nodes1,:],
                                            XIseg = XIe[nodes1,:]))
            
            # ADD GHOST FACE TO ELEMENT 2
            nodes2 = np.zeros([ELEM2.nedge],dtype=int)
            nodes2[0] = iedge2
            nodes2[1] = (iedge2+1)%ELEM2.numedges
            for knode in range(ELEM2.ElOrder-1):
                nodes2[2+knode] = ELEM2.numedges + iedge2*(ELEM2.ElOrder-1)+knode
            ELEM2.GhostFaces.append(Segment(index = iedge2,
                                            ElOrder = ELEM2.ElOrder,
                                            Tseg = nodes2,
                                            Xseg = ELEM2.Xe[nodes2,:],
                                            XIseg = XIe[nodes2,:]))
            
            # CORRECT SECOND ADJACENT ELEMENT GHOST FACE TO MATCH NODES -> PERMUTATION
            permutation = [list(ELEM2.Te[nodes2]).index(x) for x in ELEM1.Te[nodes1]]
            ELEM2.GhostFaces[-1].Xseg = ELEM2.GhostFaces[-1].Xseg[permutation,:]
            ELEM2.GhostFaces[-1].XIseg = ELEM2.GhostFaces[-1].XIseg[permutation,:]

            GhostFaces.append((list(ELEM1.Te[nodes1]),(ielem1,iedge1,len(ELEM1.GhostFaces)-1),(ielem2,iedge2,len(ELEM2.GhostFaces)-1), permutation))
            GhostElems.add(ielem1)
            GhostElems.add(ielem2)
            
        return GhostFaces, list(GhostElems)
    
    