from _header import EQUILIPY_ROOT
from _logging import EqPrint
import numpy as np
from os.path import basename
from Element import *
from matplotlib.path import Path
import _plot as eqplot

class Mesh:
    
    def __init__(self,mesh_name,readfiles=True):
        
        """
        Mesh object constructor.
        """
        
        self.pwd = str(EQUILIPY_ROOT)
        path_to_folder = self.pwd + '/MESHES/' + mesh_name
        self.name = mesh_name
        self.directory = path_to_folder
        
        if readfiles:
            EqPrint('Mesh folder: ' + path_to_folder)
        
        # INITIALISE ATTRIBUTES
        self.ElTypeALYA = None              # TYPE OF ELEMENTS CONSTITUTING THE MESH, USING ALYA NOTATION
        self.ElType = None                  # TYPE OF ELEMENTS CONSTITUTING THE MESH: 1: TRIANGLES,  2: QUADRILATERALS
        self.ElOrder = None                 # ORDER OF MESH ELEMENTS: 1: LINEAR,   2: QUADRATIC,   3: CUBIC
        self.X = None                       # MESH NODAL COORDINATES MATRIX
        self.T = None                       # MESH ELEMENTS CONNECTIVITY MATRIX 
        self.Nn = None                      # TOTAL NUMBER OF MESH NODES
        self.Ne = None                      # TOTAL NUMBER OF MESH ELEMENTS
        self.Rmax = None                    # COMPUTATIONAL MESH MAXIMAL X (R) COORDINATE
        self.Rmin = None                    # COMPUTATIONAL MESH MINIMAL X (R) COORDINATE
        self.Zmax = None                    # COMPUTATIONAL MESH MAXIMAL Y (Z) COORDINATE
        self.Zmin = None                    # COMPUTATIONAL MESH MINIMAL Y (Z) COORDINATE
        self.meanArea = None                # MESH ELEMENTS MEAN AREA
        self.meanLength = None              # MESH ELEMENTS MEAN LENTH
        self.n = None                       # NUMBER OF NODES PER ELEMENT
        self.numedges = None                # NUMBER OF EDGES PER ELEMENT (= 3 IF TRIANGULAR; = 4 IF QUADRILATERAL)
        self.nedge = None                   # NUMBER OF NODES ON ELEMENTAL EDGE
        self.dim = None                     # SPACE DIMENSION
        self.Tbound = None                  # MESH BOUNDARIES CONNECTIVITY MATRIX  (LAST COLUMN YIELDS THE ELEMENT INDEX OF THE CORRESPONDING BOUNDARY EDGE)
        self.Nbound = None                  # NUMBER OF COMPUTATIONAL DOMAIN'S BOUNDARIES (NUMBER OF ELEMENTAL EDGES)
        self.Nnbound = None                 # NUMBER OF NODES ON COMPUTATIONAL DOMAIN'S BOUNDARY
        self.BoundaryNodes = None           # LIST OF NODES (GLOBAL INDEXES) ON THE COMPUTATIONAL DOMAIN'S BOUNDARY
        self.BoundaryNodesSets = None
        self.BoundaryVertices = None        # LIST OF CONSECUTIVE NODES (GLOBAL INDEXES) WHICH ARE COMPUTATIONAL DOMAIN'S BOUNDARY VERTICES 
        self.boundary_path = None           # COMPUTATIONAL DOMAIN'S PATH (FOR PATCHING)
        self.DOFNodes = None                # LIST OF NODES (GLOBAL INDEXES) CORRESPONDING TO UNKNOW DEGREES OF FREEDOM IN THE CUTFEM SYSTEM
        self.NnDOF = None                   # NUMBER OF UNKNOWN DEGREES OF FREEDOM NODES
        self.PlasmaNodes = None             # LIST OF NODES (GLOBAL INDEXES) INSIDE THE PLASMA DOMAIN
        self.VacuumNodes = None             # LIST OF NODES (GLOBAL INDEXES) IN THE VACUUM REGION
        self.NnPB = None                    # NUMBER OF NODES ON PLASMA BOUNDARY APPROXIMATION
        self.PlasmaElems = None             # LIST OF ELEMENTS (INDEXES) INSIDE PLASMA REGION
        self.VacuumElems = None             # LIST OF ELEMENTS (INDEXES) OUTSIDE PLASMA REGION (VACUUM REGION)
        self.PlasmaBoundElems = None        # LIST OF CUT ELEMENT'S INDEXES, CONTAINING INTERFACE BETWEEN PLASMA AND VACUUM
        self.PlasmaBoundActiveElems = None  # LIST OF CUT ELEMENT'S INDEXES, CONTAINING INTERFACE BETWEEN PLASMA AND VACUUM, ON WHICH BC ARE APPLIED
        self.BoundaryElems = None           # LIST OF CUT (OR NOT) ELEMENT'S INDEXES AT THE COMPUTATIONAL DOMAIN'S BOUNDARY
        self.NonCutElems = None             # LIST OF ALL NON CUT ELEMENTS
        self.DirichletElems = None          # LIST OF ALL ELEMENTS POSSESSING A BOUNDARY NODE (NODE ON WHICH APPLY DIRICHLET BC)
        self.Elements = None                # ARRAY CONTAINING ALL ELEMENTS IN MESH (PYTHON OBJECTS)
        
        self.nge = None                     # NUMBER OF INTEGRATION NODES PER ELEMENT (STANDARD SURFACE QUADRATURE)
        self.Xg = None                      # INTEGRATION NODAL MESH COORDINATES MATRIX 
        
        self.GhostFaces = None              # LIST OF PLASMA BOUNDARY GHOST FACES
        self.GhostElems = None              # LIST OF ELEMENTS CONTAINING PLASMA BOUNDARY FACES
        
        # READ MESH FILES
        if readfiles:
            EqPrint("READ MESH FILES...", end="")
            self.ReadMeshFile()
            self.ReadFixFile()
            print('Done!')
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
        Read input mesh data files, .dom.dat and .geo.dat, and build mesh attributes. 
        """
        
        # READ DOM FILE .dom.dat
        MeshDataFile = self.directory +'/' + self.name +'.dom.dat'
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
        MeshFile = self.directory +'/'+ self.name +'.geo.dat'
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
        # READ EQU FILE .equ.dat
        FixDataFile = self.directory +'/' + self.name +'.fix.dat'
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
        return
    
    
    def BoundaryAttributes(self):
        """
        Identifies and computes key geometric and topological properties related to the 
        computational domain's boundary.

        Tasks:
            - Extracts the global indices of nodes located on the boundary (`BoundaryNodes`).
            - Computes the number of boundary nodes (`Nnbound`) and degrees of freedom (`DOFNodes`).
            - Identifies elements that are adjacent to the domain boundary (`BoundaryElems`).
            - Constructs a continuous, ordered path of nodal indices (`BoundaryVertices`) that trace the boundary.
            - Builds a closed geometric path (`boundary_path`) representing the computational domain's outer boundary,
            using the ordered boundary vertices.
        """
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
                # Find the other vertex on this boundary edge (not the current one)
                edge_verts = self.Tbound[nextbounelem, :2]
                current_vertex = self.BoundaryVertices[iboun]
                if edge_verts[0] == current_vertex:
                    self.BoundaryVertices[iboun+1] = edge_verts[1]
                else:
                    self.BoundaryVertices[iboun+1] = edge_verts[0]
                
        ### COMPUTATIONAL DOMAIN'S BOUNDARY PATH 
        # TAKE BOUNDARY VERTICES
        compboundary = np.zeros([len(self.BoundaryVertices)+1,2])
        compboundary[:-1,:] = self.X[self.BoundaryVertices,:]
        # CLOSE PATH
        compboundary[-1,:] = compboundary[0,:]
        self.boundary_path = Path(compboundary)
        
        return
    
    
    def ComputeMeshElementsMeanSize(self):
        """
        Computes the average geometric size of the mesh elements.
        """
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
    
    def DimensionlessCoordinates(self,R0): 
        """
        Converts the coordinates of all mesh elements to dimensionless form using the reference length R0.

        Input:
            R0 (float): Reference length used to normalize the coordinates (typically the major radius).
        """
        for ELEMENT in self.Elements:
            ELEMENT.Xe /= R0
            ELEMENT.area, ELEMENT.length = ELEMENT.ComputeArea_Length()
        return 
    
    
    def InitialiseElements(self,PlasmaLS):
        """ 
        Function initialising all elements in the mesh. 
        """
        self.Elements = [Element(index = e,
                                    ElType = self.ElType,
                                    ElOrder = self.ElOrder,
                                    Xe = self.X[self.T[e,:],:],
                                    Te = self.T[e,:],
                                    PlasmaLSe = PlasmaLS[self.T[e,:]]) for e in range(self.Ne)]
        
        # COMPUTE MESH MEAN SIZE
        self.meanArea, self.meanLength = self.ComputeMeshElementsMeanSize()
        EqPrint("         · MESH ELEMENTS MEAN AREA = " + str(self.meanArea) + " m^2")
        EqPrint("         · MESH ELEMENTS MEAN LENGTH = " + str(self.meanLength) + " m")
        return
    
    
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
        """
        Identifies and assigns boundary information to mesh elements and nodes.

        Tasks:
            - Initializes the boundary edge connectivity (`Teboun`) for each boundary element.
            - Loops through boundary edges to determine which local nodes in each element are on the boundary.
            - Identifies interior elements that are not full boundary elements but have at least one node on the boundary.
            - Populates a list of all elements that have any boundary nodes (`DirichletElems`).
            - Sets a domain flag (`Dom = 2`) for elements that are classified as boundary elements.
        """
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
            
        # ASSIGN ELEMENTAL DOMAIN FLAG
        for ielem in self.BoundaryElems:
            self.Elements[ielem].Dom = 2  
  
        return
    
    
    def ClassifyElements(self,PlasmaLS):
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
            
        for ielem in range(self.Ne):
            regionplasma, DHONplasma = self.Elements[ielem].CheckElementalVerticesLevelSetSigns()
            if regionplasma < 0:   # ALL PLASMA LEVEL-SET NODAL VALUES NEGATIVE -> INSIDE PLASMA DOMAIN 
                # ALREADY CLASSIFIED AS COMPUTATIONAL BOUNDARY ELEMENT (= BOUNDARY ELEMENT)
                if self.Elements[ielem].Dom == 2:  
                    # REMOVE FROM COMPUTATIONAL BOUNDARY ELEMENT LIST
                    self.BoundaryElems = self.BoundaryElems[self.BoundaryElems != ielem]
                # REDEFINE CLASSIFICATION
                self.PlasmaElems[kplasm] = ielem
                self.Elements[ielem].Dom = -1
                kplasm += 1
            elif regionplasma == 0:  # DIFFERENT SIGN IN PLASMA LEVEL-SET NODAL VALUES -> PLASMA/VACUUM INTERFACE ELEMENT
                # ALREADY CLASSIFIED AS COMPUTATIONAL BOUNDARY ELEMENT (= BOUNDARY ELEMENT)
                if self.Elements[ielem].Dom == 2:  
                    # REMOVE FROM COMPUTATIONAL BOUNDARY ELEMENT LIST
                    self.BoundaryElems = self.BoundaryElems[self.BoundaryElems != ielem]
                # REDEFINE CLASSIFICATION
                self.PlasmaBoundElems[kint] = ielem
                self.Elements[ielem].Dom = 0
                kint += 1
            elif regionplasma > 0: # ALL PLASMA LEVEL-SET NODAL VALUES POSITIVE -> OUTSIDE PLASMA DOMAIN
                if self.Elements[ielem].Dom == 2:  # ALREADY CLASSIFIED AS COMPUTATIONAL BOUNDARY ELEMENT (= BOUNDARY ELEMENT)
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
                    PlasmaLS[self.Elements[ielem].Te[inode]] *= -1       
        
        # DELETE REST OF UNUSED MEMORY
        self.PlasmaElems = self.PlasmaElems[:kplasm]
        self.VacuumElems = self.VacuumElems[:kvacuu]
        self.PlasmaBoundElems = self.PlasmaBoundElems[:kint]
        
        # GATHER NON-CUT ELEMENTS  
        self.NonCutElems = np.concatenate((self.PlasmaElems, self.VacuumElems, self.BoundaryElems), axis=0)
        
        if len(self.NonCutElems) + len(self.PlasmaBoundElems) != self.Ne:
            raise ValueError("Non-cut elements + Cut elements =/= Total number of elements  --> Wrong mesh classification!!")
        
        # CLASSIFY NODES ACCORDING TO NEW ELEMENT CLASSIFICATION
        #self.ClassifyNodes()
        return PlasmaLS
    
    
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
        for ielem in self.BoundaryElems:
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
                if self.PlasmaLS[node] < 0:
                    self.PlasmaNodes.add(node)
                else:
                    self.VacuumNodes.add(node)
        for ielem in self.BoundaryElems:
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
    
    
    ##################################################################################################
    ############################### PLASMA BOUNDARY APPROXIMATION ####################################
    ##################################################################################################
    
    def ObtainPlasmaBoundaryElementalPath(self):
        """
        Constructs an ordered path of plasma boundary elements.

        The resulting `PlasmaBoundElemPath` is stored as a list of element indices representing a connected boundary path.
        """
        self.PlasmaBoundElemPath = list()
        
        if self.PlasmaBoundElems.size != 0:
            # INTIALISE PATH LIST
            self.PlasmaBoundElemPath.append(self.PlasmaBoundElems[0])
            # CONSTRUCT PATH
            for ielem in range(len(self.PlasmaBoundElems)-1):
                # LOOK AT ELEMENT NEIGHBOURS
                for ineigh in self.Elements[self.PlasmaBoundElemPath[ielem]].neighbours:
                    if ineigh == -1:
                        pass
                    else:
                        # IF NEIGHBOUR ELEMENT IS PLASMA BOUNDARY ELEMENT
                        if self.Elements[ineigh].Dom == 0:
                            # IF FIRST ITERATION 
                            if ielem == 0:
                                self.PlasmaBoundElemPath.append(self.Elements[ineigh].index)
                                break
                            else:
                                # CHECK THAT IS NOT THE PREVIOUS ADJACENT ELEMENT
                                if ineigh != self.PlasmaBoundElemPath[ielem-1]:
                                    self.PlasmaBoundElemPath.append(self.Elements[ineigh].index)
                                    break
        return
    
    
    def ObtainPlasmaBoundaryActiveElements(self,numelements = -1):
        """
        Selects and assigns the active plasma boundary elements used for applying plasma boundary conditions.

        Input:
            numelements (int): Number of plasma boundary elements to select as active.
                - If -1 (default), all elements in the plasma boundary element path are used.
                - If a positive integer, selects that many equidistant elements along the plasma boundary path.
        """
        if self.PlasmaBoundElemPath:
            # PLASMA BOUNDARY ACTIVE ELEMS = PLASMA BOUNDARY ELEMS  --> CONSTRAIN BC ON ALL PLASMA BOUNDARY ELEMS
            if numelements == -1:
                self.PlasmaBoundActiveElems = self.PlasmaBoundElemPath
            # SELECT EQUIDISTANT PLASMA BOUNDARY ELEMENTS 
            else:
                indices = np.linspace(0, len(self.PlasmaBoundElemPath) - 1, numelements, dtype=int)
                self.PlasmaBoundActiveElems = np.array(self.PlasmaBoundElemPath)[indices]
        else:
            self.PlasmaBoundActiveElems = list()
        return
    
    
    def ComputePlasmaBoundaryApproximation(self):
        """ 
        Computes the elemental plasma boundary approximation.
        Computes normal vectors for each constraint node.
        """
        for inter, ielem in enumerate(self.PlasmaBoundElems):
            # APPROXIMATE PLASMA/VACUUM INTERACE GEOMETRY CUTTING ELEMENT 
            self.Elements[ielem].InterfaceApproximation(inter)
        return
    
    
    def ComputePlasmaBoundaryNumberNodes(self):
        """
        Computes the total number of nodes located on the plasma boundary approximation
        """  
        nnodes = 0
        for ielem in self.PlasmaBoundActiveElems:
            nnodes += self.Elements[ielem].InterfApprox.ng
        return nnodes
    
    
    ##################################################################################################
    ################################# PLASMA BOUNDARY GHOST FACES ####################################
    ##################################################################################################
    
    def ValidateEdgeNodeOrdering(self):
        """
        Validates that edge node indices are computed consistently for all elements.
        
        For triangular elements, all edges should be enumerated counter-clockwise:
        - Edge 0: vertices [0, 1]
        - Edge 1: vertices [1, 2]
        - Edge 2: vertices [2, 0]
        
        Issues detected here may cause incorrect normal vector computation and assembly.
        """
        degenerate_edges = []
        
        for elem in self.Elements:
            for iedge in range(elem.numedges):
                # Get edge vertices
                v0_local = iedge
                v1_local = (iedge + 1) % elem.numedges
                
                # Check for degenerate edges (zero length)
                edge_vector = elem.Xe[v1_local,:] - elem.Xe[v0_local,:]
                edge_length = np.linalg.norm(edge_vector)
                
                if edge_length < 1e-14:
                    degenerate_edges.append((elem.index, iedge, edge_length))
        
        if degenerate_edges:
            raise ValueError(f"Found {len(degenerate_edges)} degenerate edges in mesh. "
                           f"This indicates severely distorted elements or connectivity issues. "
                           f"First problem: Element {degenerate_edges[0][0]}, Edge {degenerate_edges[0][1]}")
    
    
    def IdentifyPlasmaBoundaryGhostFaces(self):
        """
        Identifies the elemental ghost faces on which the ghost penalty term needs to be integrated, for elements containing the plasma
        boundary.
        """
        
        # RESET ELEMENTAL GHOST FACES
        for ielem in np.concatenate((self.PlasmaBoundElems,self.PlasmaElems),axis=0):
            self.Elements[ielem].GhostFaces = None
        
        # VALIDATE EDGE NODE ORDERING before proceeding
        self.ValidateEdgeNodeOrdering()
            
        GhostFaces_dict = dict()    # [(CUT_EDGE_NODAL_GLOBAL_INDEXES): {(ELEMENT_INDEX_1, EDGE_INDEX_1), (ELEMENT_INDEX_2, EDGE_INDEX_2)}]
        
        for ielem in self.PlasmaBoundElems:
            ELEMENT = self.Elements[ielem]
            for iedge, neighbour in enumerate(ELEMENT.neighbours):
                if neighbour >= 0 and self.Elements[neighbour].Dom < 1:
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
        self.GhostFaces = list()
        self.GhostElems = set()

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
            # Find local connectivities of ghost face nodes in element 1
            nodes1 = np.zeros([ELEM1.nedge],dtype=int)
            nodes1[0] = iedge1
            nodes1[1] = (iedge1+1)%ELEM1.numedges
            for knode in range(ELEM1.ElOrder-1):
                nodes1[2+knode] = ELEM1.numedges + iedge1*(ELEM1.ElOrder-1)+knode
            # Add ghost face to element 1
            ELEM1.GhostFaces.append(Segment(index = iedge1,
                                            ElOrder = ELEM1.ElOrder,
                                            Tseg = nodes1,
                                            Xseg = ELEM1.Xe[nodes1,:],
                                            XIseg = XIe[nodes1,:]))
            
            # ADD GHOST FACE TO ELEMENT 2
            # Find local connectivities of ghost face nodes in element 2
            nodes2 = np.zeros([ELEM2.nedge],dtype=int)
            nodes2[0] = iedge2
            nodes2[1] = (iedge2+1)%ELEM2.numedges
            for knode in range(ELEM2.ElOrder-1):
                nodes2[2+knode] = ELEM2.numedges + iedge2*(ELEM2.ElOrder-1)+knode
            # Add ghost face to element 2
            ELEM2.GhostFaces.append(Segment(index = iedge2,
                                            ElOrder = ELEM2.ElOrder,
                                            Tseg = nodes2,
                                            Xseg = ELEM2.Xe[nodes2,:],
                                            XIseg = XIe[nodes2,:]))
            
            # CORRECT SECOND ADJACENT ELEMENT GHOST FACE TO MATCH NODES -> PERMUTATION
            permutation = [list(ELEM2.Te[nodes2]).index(x) for x in ELEM1.Te[nodes1]]
            ELEM2.GhostFaces[-1].Xseg = ELEM2.GhostFaces[-1].Xseg[permutation,:]
            ELEM2.GhostFaces[-1].XIseg = ELEM2.GhostFaces[-1].XIseg[permutation,:]

            self.GhostFaces.append((list(ELEM1.Te[nodes1]),(ielem1,iedge1,len(ELEM1.GhostFaces)-1),(ielem2,iedge2,len(ELEM2.GhostFaces)-1), permutation))
            self.GhostElems.add(ielem1)
            self.GhostElems.add(ielem2)
            
        self.GhostElems = list(self.GhostElems)
        return 
    
    
    def ComputePlasmaBoundaryGhostFaces(self):
        """
        Computes ghost faces associated with the plasma boundary for stabilization purposes.

        Tasks:
            - Identifies ghost faces on plasma boundary elements.
            - Computes normal vectors for the ghost faces of each ghost element.
            - Validates the computed ghost face normal vectors.
            - CRITICAL: Verifies that normals on adjacent ghost faces are opposite (n1 = -n2).
        """
        # COMPUTE PLASMA BOUNDARY GHOST FACES
        self.IdentifyPlasmaBoundaryGhostFaces()
        # COMPUTE ELEMENTAL GHOST FACES NORMAL VECTORS
        for ielem in self.GhostElems:
            self.Elements[ielem].GhostFacesNormals()
        # CHECK NORMAL VECTORS (unitary and orthogonal)
        self.CheckGhostFacesNormalVectors()
        # CRITICAL: Check that adjacent normals are opposite
        self.ValidateAdjacentGhostFaceNormals()
        return
    
    
    def CheckGhostFacesNormalVectors(self):
        """
        This function verifies if the normal vectors at the plasma boundary ghost faces are unitary and orthogonal to 
        the corresponding interface segments. It checks the dot product between the segment tangent vector and the 
        normal vector, raising an exception if the dot product is not close to zero (indicating non-orthogonality).
        """
        
        for ielem in self.GhostElems:
            for FACE in self.Elements[ielem].GhostFaces:
                # CHECK UNIT LENGTH
                if np.abs(np.linalg.norm(FACE.NormalVec)-1) > 1e-6:
                    raise Exception('Normal vector norm equals',np.linalg.norm(FACE.NormalVec), 'for mesh element', ielem, ": Normal vector not unitary")
                # CHECK ORTHOGONALITY
                tangvec = np.array([FACE.Xseg[1,0]-FACE.Xseg[0,0], FACE.Xseg[1,1]-FACE.Xseg[0,1]]) 
                scalarprod = np.dot(tangvec,FACE.NormalVec)
                if scalarprod > 1e-10: 
                    raise Exception('Dot product equals',scalarprod, 'for mesh element', ielem, ": Normal vector not perpendicular")
        return
    
    
    def ValidateAdjacentGhostFaceNormals(self):
        """
        CRITICAL VALIDATION: Verify that normal vectors on shared ghost faces point in opposite directions (n1 = -n2).
        
        This is required by CutFEM theory for proper ghost penalty stabilization.
        If this check fails, the stabilization will be inconsistent and may cause instability.
        
        Raises:
            - ValueError: If any pair of adjacent ghost faces have inconsistent normals
        """
        if self.GhostFaces is None or len(self.GhostFaces) == 0:
            return
        
        tolerance = 1e-10
        failures = []
        
        for ghost_face_tuple in self.GhostFaces:
            # Extract element and face information
            elem1_idx, edge1_idx, face1_list_idx = ghost_face_tuple[1]
            elem2_idx, edge2_idx, face2_list_idx = ghost_face_tuple[2]
            
            # Get the actual face objects
            face1 = self.Elements[elem1_idx].GhostFaces[face1_list_idx]
            face2 = self.Elements[elem2_idx].GhostFaces[face2_list_idx]
            
            # Check opposition: n1 + n2 should be close to zero
            normal_sum = face1.NormalVec + face2.NormalVec
            normal_sum_norm = np.linalg.norm(normal_sum)
            
            if normal_sum_norm > tolerance:
                failures.append((elem1_idx, elem2_idx, normal_sum_norm, 
                               face1.NormalVec, face2.NormalVec))
        
        if failures:
            error_msg = f"CRITICAL: {len(failures)} ghost face pairs have inconsistent normals:\n"
            for elem1, elem2, norm_sum, n1, n2 in failures[:3]:  # Show first 3 failures
                error_msg += f"  Elements {elem1} & {elem2}: ||n1 + n2|| = {norm_sum:.2e}\n"
                error_msg += f"    n1 = {n1}, n2 = {n2}\n"
            if len(failures) > 3:
                error_msg += f"  ... and {len(failures)-3} more\n"
            raise ValueError(error_msg)

    
    
    ##################################################################################################
    ############################# NUMERICAL INTEGRATION QUADRATURES ##################################
    ##################################################################################################
    
    def ComputeStandardQuadratures(self,QuadOrder2D):
        """
        Computes the STANDARD FEM numerical integration QUADRATURES for ALL MESH ELEMENTS.
        """
        # COMPUTE STANDARD 2D QUADRATURE ENTITIES FOR NON-CUT ELEMENTS 
        for ELEMENT in self.Elements:
            ELEMENT.ComputeStandardQuadrature2D(QuadOrder2D)
            
        # DEFINE STANDARD SURFACE QUADRATURE NUMBER OF INTEGRATION NODES
        self.nge = self.Elements[0].ng
        return
    
    def ComputeAdaptedQuadratures(self,QuadOrder2D,QuadOrder1D):
        """
        Computes the ADAPTED CutFEM numerical integration QUADRATURES for CUT ELEMENTS.
        """
        # COMPUTE ADAPTED QUADRATURE ENTITIES FOR INTERFACE ELEMENTS
        for ielem in self.PlasmaBoundElems:
            self.Elements[ielem].ComputeAdaptedQuadratures2D(QuadOrder2D)
            
        for ielem in self.PlasmaBoundActiveElems:
            self.Elements[ielem].ComputeAdaptedQuadrature1D(QuadOrder1D)
            
        # COMPUTE NUMBER OF NODES ON PLASMA BOUNDARY APPROXIMATION
        self.NnPB = self.ComputePlasmaBoundaryNumberNodes()
        return

    def ComputeGhostFacesQuadratures(self,QuadOrder1D):
        """
        Computes the ELEMENTAL FACES numerical integration QUADRATURES for GHOST ELEMENTS.
        """
        # COMPUTE QUADRATURES FOR GHOST FACES ON PLASMA BOUNDARY ELEMENTS
        for ielem in self.GhostElems:
            self.Elements[ielem].ComputeGhostFacesQuadratures(QuadOrder1D)
        return

    ##################################################################################################
    ############################### INTERFACE GEOMETRIC ERROR ANALYSIS ################################
    ##################################################################################################

    def ComputeInterfaceApproximationError(self, verbose=False):
        """
        Computes comprehensive geometric error metrics for interface approximation.

        The plasma interface is approximated by finding zero-crossings of the level-set function
        on element edges. This function estimates the geometric errors introduced by this discretization.

        Errors measured:
        - Point-to-interface distance: deviation of approximation nodes from exact PHI=0 contour
        - Arc-length error: difference between piecewise-linear approximation and exact arc length
        - Normal vector error: angular difference between exact and approximated normal vectors

        Returns:
            Dictionary with structure:
            {
                'summary': {
                    'total_elements': int,
                    'point_distance_max': float,
                    'point_distance_mean': float,
                    'point_distance_std': float,
                    'arc_length_error_max': float,
                    'arc_length_error_mean': float,
                    'arc_length_error_relative_mean': float,
                    'normal_angle_error_max': float,  # degrees
                    'normal_angle_error_mean': float,
                },
                'per_element': {
                    elem_idx: {
                        'point_distances': np.ndarray,
                        'arc_length_error': float,
                        'arc_length_exact': float,
                        'arc_length_approx': float,
                        'normal_errors': List[float],  # degrees
                        'h': float
                    }
                }
            }
        """
        if len(self.PlasmaBoundElems) == 0:
            if verbose:
                EqPrint("Warning: No plasma boundary elements - skipping interface error computation")
            return None

        errors_data = {
            'summary': {},
            'per_element': {}
        }

        all_point_distances = []
        all_arc_length_errors = []
        all_arc_length_relative_errors = []
        all_normal_errors = []

        # Iterate over all cut elements
        for elem_idx in self.PlasmaBoundElems:
            elem = self.Elements[elem_idx]

            # Skip if element has no interface approximation
            if not hasattr(elem, 'InterfApprox') or elem.InterfApprox is None:
                continue

            # Compute point-to-interface distance errors
            point_dists = self._compute_element_interface_point_errors(elem_idx)

            # Compute arc-length errors
            arc_length_error, arc_length_exact, arc_length_approx = \
                self._compute_element_interface_arc_length_error(elem_idx)

            # Compute normal vector errors
            normal_errors = self._compute_element_interface_normal_error(elem_idx)

            # Store element-specific results
            h = elem.length if hasattr(elem, 'length') else 0.1
            errors_data['per_element'][elem_idx] = {
                'point_distances': point_dists,
                'arc_length_error': arc_length_error,
                'arc_length_exact': arc_length_exact,
                'arc_length_approx': arc_length_approx,
                'normal_errors': normal_errors,
                'h': h
            }

            # Accumulate for summary statistics
            if len(point_dists) > 0:
                all_point_distances.extend(point_dists)

            if arc_length_error is not None:
                all_arc_length_errors.append(arc_length_error)
                if arc_length_exact > 1e-14:
                    all_arc_length_relative_errors.append(arc_length_error / arc_length_exact)

            if normal_errors is not None:
                all_normal_errors.extend(normal_errors)

        # Compute summary statistics
        errors_data['summary']['total_elements'] = len(self.PlasmaBoundElems)

        if len(all_point_distances) > 0:
            all_point_distances = np.array(all_point_distances)
            errors_data['summary']['point_distance_max'] = float(np.max(all_point_distances))
            errors_data['summary']['point_distance_mean'] = float(np.mean(all_point_distances))
            errors_data['summary']['point_distance_std'] = float(np.std(all_point_distances))
        else:
            errors_data['summary']['point_distance_max'] = 0.0
            errors_data['summary']['point_distance_mean'] = 0.0
            errors_data['summary']['point_distance_std'] = 0.0

        if len(all_arc_length_errors) > 0:
            all_arc_length_errors = np.array(all_arc_length_errors)
            errors_data['summary']['arc_length_error_max'] = float(np.max(np.abs(all_arc_length_errors)))
            errors_data['summary']['arc_length_error_mean'] = float(np.mean(np.abs(all_arc_length_errors)))
        else:
            errors_data['summary']['arc_length_error_max'] = 0.0
            errors_data['summary']['arc_length_error_mean'] = 0.0

        if len(all_arc_length_relative_errors) > 0:
            errors_data['summary']['arc_length_error_relative_mean'] = \
                float(np.mean(np.abs(all_arc_length_relative_errors)))
        else:
            errors_data['summary']['arc_length_error_relative_mean'] = 0.0

        if len(all_normal_errors) > 0:
            all_normal_errors = np.array(all_normal_errors)
            errors_data['summary']['normal_angle_error_max'] = float(np.max(all_normal_errors))
            errors_data['summary']['normal_angle_error_mean'] = float(np.mean(all_normal_errors))
        else:
            errors_data['summary']['normal_angle_error_max'] = 0.0
            errors_data['summary']['normal_angle_error_mean'] = 0.0

        # Store as mesh attribute for repeated access
        self.InterfaceGeometricError = errors_data

        if verbose:
            self._print_interface_error_report(errors_data)

        return errors_data

    def _compute_element_interface_point_errors(self, elem_idx):
        """
        Compute point-to-interface distance for each interface approximation node.

        The interface is defined by PHI(X) = 0. For exact interface nodes, PHI should be ~0.
        Deviations indicate approximation error.

        Returns:
            np.ndarray of absolute distances (should be close to 0 for good approximations)
        """
        elem = self.Elements[elem_idx]

        if not hasattr(elem, 'InterfApprox') or elem.InterfApprox is None:
            return np.array([])

        interface_approx = elem.InterfApprox
        distances = []

        # Evaluate level-set at each interface approximation node
        if hasattr(interface_approx, 'Xint') and interface_approx.Xint is not None:
            for X_int in interface_approx.Xint:
                # Evaluate level-set at this point (should be ~0)
                phi_value = elem.PHI(X_int.reshape((1,2)))

                # For elements with smooth level-set, |PHI| approximates distance to interface
                # Distance is proportional to |PHI(X)| / ||grad PHI||, but ||grad PHI|| ~ O(1)
                distances.append(abs(phi_value))

        return np.array(distances)

    def _compute_element_interface_arc_length_error(self, elem_idx):
        """
        Compute arc-length approximation error on interface.

        Compares:
        - Exact arc length: integral of ||grad PHI|| / ||grad PHI|| along contour (= contour length)
        - Approximated arc length: sum of distances between consecutive interface nodes

        Returns:
            (arc_length_error, arc_length_exact, arc_length_approx)
        """
        elem = self.Elements[elem_idx]

        if not hasattr(elem, 'InterfApprox') or elem.InterfApprox is None:
            return None, None, None

        interface_approx = elem.InterfApprox

        if not hasattr(interface_approx, 'Xint') or interface_approx.Xint is None:
            return None, None, None

        Xint = interface_approx.Xint

        if len(Xint) < 2:
            return None, None, None

        # Compute arc length of piecewise-linear approximation
        arc_length_approx = 0.0
        for i in range(len(Xint) - 1):
            arc_length_approx += np.linalg.norm(Xint[i+1] - Xint[i])

        # Estimate exact arc length using adapted quadrature
        # The interface is parametrized, and we have quadrature points on it
        if hasattr(interface_approx, 'detJg1D') and interface_approx.detJg1D is not None and \
           hasattr(interface_approx, 'Wg') and interface_approx.Wg is not None:
            arc_length_exact = 0.0
            for ig in range(len(interface_approx.detJg1D)):
                arc_length_exact += interface_approx.detJg1D[ig] * interface_approx.Wg[ig]
        else:
            # Fallback: assume quadrature weight already includes Jacobian
            arc_length_exact = arc_length_approx

        arc_length_error = arc_length_exact - arc_length_approx

        return arc_length_error, arc_length_exact, arc_length_approx

    def _compute_element_interface_normal_error(self, elem_idx):
        """
        Compute angular error in normal vector approximation.

        At each interface approximation node, computes:
        - Exact normal: grad PHI / ||grad PHI|| (computed via finite differences)
        - Approximated normal: stored in InterfApprox.NormalVec

        Returns angle between them in degrees.

        Returns:
            List of angles in degrees
        """
        elem = self.Elements[elem_idx]

        if not hasattr(elem, 'InterfApprox') or elem.InterfApprox is None:
            return None

        interface_approx = elem.InterfApprox

        if not hasattr(interface_approx, 'Xint') or interface_approx.Xint is None:
            return None

        if not hasattr(interface_approx, 'NormalVec') or interface_approx.NormalVec is None:
            return None

        normal_errors = []
        h_fd = 1e-8  # Finite difference step for gradient computation

        for i, X_int in enumerate(interface_approx.Xint):
            # Compute exact gradient of level-set via finite differences
            grad_phi_exact = np.zeros(2)
            for j in range(2):
                X_plus = X_int.copy()
                X_minus = X_int.copy()
                X_plus[j] += h_fd
                X_minus[j] -= h_fd

                phi_plus = elem.PHI(X_plus.reshape((1,2)))
                phi_minus = elem.PHI(X_minus.reshape((1,2)))

                grad_phi_exact[j] = (phi_plus - phi_minus) / (2 * h_fd)

            # Normalize to get exact normal
            grad_norm = np.linalg.norm(grad_phi_exact)
            if grad_norm > 1e-14:
                n_exact = grad_phi_exact / grad_norm

                # Get approximated normal
                if i < len(interface_approx.NormalVec):
                    n_approx = interface_approx.NormalVec[i]

                    # Compute angle between vectors
                    dot_product = np.dot(n_exact, n_approx)
                    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure valid range for arccos
                    angle_rad = np.arccos(dot_product)
                    angle_deg = np.degrees(angle_rad)

                    normal_errors.append(angle_deg)

        return normal_errors

    def _print_interface_error_report(self, errors_data):
        """
        Print a structured report of interface geometric errors.
        """
        summary = errors_data['summary']

        EqPrint("\n" + "="*70)
        EqPrint("INTERFACE GEOMETRIC ERROR ANALYSIS")
        EqPrint("="*70)

        EqPrint(f"\nTotal cut elements analyzed: {summary['total_elements']}")

        EqPrint("\n[1] POINT-TO-INTERFACE DISTANCE ERRORS")
        EqPrint("-"*70)
        EqPrint(f"  Maximum distance error:  {summary['point_distance_max']:.6e} m")
        EqPrint(f"  Mean distance error:     {summary['point_distance_mean']:.6e} m")
        EqPrint(f"  Std dev distance error:  {summary['point_distance_std']:.6e} m")

        EqPrint("\n[2] ARC-LENGTH APPROXIMATION ERRORS")
        EqPrint("-"*70)
        EqPrint(f"  Maximum arc-length error: {summary['arc_length_error_max']:.6e} m")
        EqPrint(f"  Mean arc-length error:    {summary['arc_length_error_mean']:.6e} m")
        EqPrint(f"  Relative mean error:      {summary['arc_length_error_relative_mean']:.6e}")

        EqPrint("\n[3] NORMAL VECTOR APPROXIMATION ERRORS")
        EqPrint("-"*70)
        EqPrint(f"  Maximum angular error:    {summary['normal_angle_error_max']:.6e}°")
        EqPrint(f"  Mean angular error:       {summary['normal_angle_error_mean']:.6e}°")

        EqPrint("\n" + "="*70 + "\n")
        return



    def IntegrationNodesMesh(self):
        if type(self.Xg) == type(None):
            self.Xg = np.zeros([self.Ne*self.nge,self.dim])
            for ielem, ELEMENT in enumerate(self.Elements):
                self.Xg[ielem*self.nge:(ielem+1)*self.nge,:] = ELEMENT.Xg
        return
    
    
    ##################################################################################################
    ######################################### REPRESENTATION #########################################
    ##################################################################################################
    
    def PlotBoundary(self,ax=None):
        """
        Plots the computational domain's boundary on the provided matplotlib axis or creates a new figure.

        Parameters:
            ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. If None, a new figure and axis are created.
        """
        # GENERATE FIGURE IF NON EXISTENT
        if type(ax) == type(None):
            fig, ax = plt.subplots(1, 1, figsize=(5,6))
            ax.set_aspect('equal')
            ax.set_xlim(self.Rmin-self.padx,self.Rmax+self.padx)
            ax.set_ylim(self.Zmin-self.pady,self.Zmax+self.pady)
            ax.set_xlabel('R (in m)')
            ax.set_ylabel('Z (in m)')
            ax.set_title("Computational domain's boundary")
        # PLOT MESH BOUNDARY
        for iboun in range(self.Nbound):
            ax.plot(self.X[self.Tbound[iboun,:2],0],self.X[self.Tbound[iboun,:2],1],
                    linewidth = eqplot.compbounlinewidth, 
                    color = eqplot.compbouncolor)
        return
    
    def Plot(self,ax=None):
        """
        Plots the computational mesh of the domain including its boundary.

        Parameters:
            ax (matplotlib.axes.Axes, optional): Axis on which to plot the mesh. If None, a new figure and axis are created.
        """
        # GENERATE FIGURE IF NON EXISTENT
        if type(ax) == type(None):
            fig, ax = plt.subplots(1, 1, figsize=(5,6))
            ax.set_aspect('equal')
            ax.set_xlim(self.Rmin-self.padx,self.Rmax+self.padx)
            ax.set_ylim(self.Zmin-self.pady,self.Zmax+self.pady)
            ax.set_xlabel('R (in m)')
            ax.set_ylabel('Z (in m)')
            ax.set_title("Computational domain's mesh")
        # PLOT MESH
        for ielem in range(self.Ne):
            Xe = np.zeros([self.numedges+1,2])
            Xe[:-1,:] = self.X[self.T[ielem,:self.numedges],:]
            Xe[-1,:] = Xe[0,:]
            ax.plot(Xe[:,0], Xe[:,1], 
                     color = eqplot.meshcolor, 
                     linewidth = eqplot.meshlinewidth)
        # PLOT BOUNDARY
        self.PlotBoundary(ax = ax)
        return
    
    
    def PlotClassifiedElements(self,PlasmaLS = None, GHOSTFACES=False,**kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_aspect('equal')
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        ax.set_title("Classified mesh elements")
        if not kwargs:
            ax.set_xlim(self.Rmin-eqplot.padx,self.Rmax+eqplot.padx)
            ax.set_ylim(self.Zmin-eqplot.pady,self.Zmax+eqplot.pady)
        else: 
            ax.set_ylim(kwargs['zmin'],kwargs['zmax'])
            ax.set_xlim(kwargs['rmin'],kwargs['rmax'])
        
        # PLOT PLASMA REGION ELEMENTS
        for ELEMENT in self.Elements:
            Xe = np.zeros([ELEMENT.numedges+1,2])
            Xe[:-1,:] = self.X[ELEMENT.Te[:self.numedges],:]
            Xe[-1,:] = self.X[ELEMENT.Te[0],:]
            ax.plot(Xe[:,0], Xe[:,1], color=eqplot.Black, linewidth=1)
            ax.fill(Xe[:,0], Xe[:,1], color = ELEMENT.Color())
           
        # PLOT PLASMA BOUNDARY  
        if type(PlasmaLS) != type(None):
            ax.tricontour(self.X[:,0],self.X[:,1], PlasmaLS, levels=[0], 
                        colors = eqplot.plasmabouncolor,
                        linewidths = eqplot.plasmabounlinewidth)
                
        # PLOT GHOSTFACES 
        if GHOSTFACES:
            for ghostface in self.GhostFaces:
                ax.plot(self.X[ghostface[0][:2],0],self.X[ghostface[0][:2],1],
                         linewidth=2,
                         color=eqplot.ghostfacescolor)
        return
    
    

    def PlotActiveMesh(self, PlasmaLS = None, GHOSTFACES=False, **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_aspect('equal')
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        ax.set_title("Classified mesh elements")
        if not kwargs:
            ax.set_xlim(self.Rmin-eqplot.padx,self.Rmax+eqplot.padx)
            ax.set_ylim(self.Zmin-eqplot.pady,self.Zmax+eqplot.pady)
        else: 
            ax.set_ylim(kwargs['zmin'],kwargs['zmax'])
            ax.set_xlim(kwargs['rmin'],kwargs['rmax'])

        # PLOT PLASMA REGION ELEMENTS
        for ELEMENT in self.Elements:
            if ELEMENT.Dom < 1:
                color = eqplot.activemeshcolor
            else:
                color = eqplot.backmeshcolor
            Xe = np.zeros([ELEMENT.numedges+1,2])
            Xe[:-1,:] = self.X[ELEMENT.Te[:self.numedges],:]
            Xe[-1,:] = self.X[ELEMENT.Te[0],:]
            ax.plot(Xe[:,0], Xe[:,1], color=eqplot.Black, linewidth=1)
            ax.fill(Xe[:,0], Xe[:,1], color = color)

        # PLOT PLASMA BOUNDARY  
        if type(PlasmaLS) != type(None):
            ax.tricontour(self.X[:,0],self.X[:,1], PlasmaLS, levels=[0], 
                        colors = eqplot.plasmabouncolor,
                        linewidths = eqplot.plasmabounlinewidth)
                
        # PLOT GHOSTFACES 
        if GHOSTFACES:
            for ghostface in self.GhostFaces:
                ax.plot(self.X[ghostface[0][:2],0],self.X[ghostface[0][:2],1],
                         linewidth=2,
                         color=eqplot.ghostfacescolor)
        return
    