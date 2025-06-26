import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

class EquilipyUpdate:
    
    
    def CheckConvergence(self,VALUES):
        """
        Function to evaluate convergence criteria during iterative computation. 
        Based on the type of value being checked (`PSI_NORM` or `PSI_B`), it calculates 
        the L2 norm of the residual between consecutive iterations and determines if the solution 
        has converged to the desired tolerance.

        Input:
            VALUES (str) Specifies the variable type to check for convergence:
                - "PSI_NORM" : Normalized magnetic flux (used for internal convergence).
                - "PSI_B"    : Boundary flux (used for external convergence).
        """
        if VALUES == "PSI_NORM":
            # FOR THE LINEAR AND ZHENG MODELS (FIXED BOUNDARY) THE SOURCE TERM DOESN'T DEPEND ON PSI, THEREFORE A SINGLE INTERNAL ITERATION IS ENOUGH
            if not self.PlasmaCurrent.PSIdependent:
                self.int_cvg = True  # STOP INTERNAL WHILE LOOP 
                self.int_residu = 0
            else:
                # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
                if np.linalg.norm(self.PSI_NORM[:,1]) > 0:
                    L2residu = np.linalg.norm(self.PSI_NORM[:,1] - self.PSI_NORM[:,0])/np.linalg.norm(self.PSI_NORM[:,1])
                else: 
                    L2residu = np.linalg.norm(self.PSI_NORM[:,1] - self.PSI_NORM[:,0])
                if L2residu < self.int_tol:
                    self.int_cvg = True   # STOP INTERNAL WHILE LOOP 
                else:
                    self.int_cvg = False
                    
                self.int_residu = L2residu
                print("Internal iteration = ",self.int_it,", PSI_NORM residu = ", L2residu)
                print(" ")
            
        elif VALUES == "PSI_B":
            # FOR FIXED BOUNDARY PROBLEM, THE BOUNDARY VALUES ARE ALWAYS THE SAME, THEREFORE A SINGLE EXTERNAL ITERATION IS NEEDED
            if self.FIXED_BOUNDARY:
                self.ext_cvg = True  # STOP EXTERNAL WHILE LOOP 
                self.ext_residu = 0
            else:
                # COMPUTE L2 NORM OF RESIDUAL BETWEEN ITERATIONS
                if np.linalg.norm(self.PSI_B[:,1]) > 0:
                    L2residu = np.linalg.norm(self.PSI_B[:,1] - self.PSI_B[:,0])/np.linalg.norm(self.PSI_B[:,1])
                else: 
                    L2residu = np.linalg.norm(self.PSI_B[:,1] - self.PSI_B[:,0])
                if L2residu < self.ext_tol:
                    self.ext_cvg = True   # STOP EXTERNAL WHILE LOOP 
                else:
                    self.ext_cvg = False
                    
                self.ext_residu = L2residu
                print("External iteration = ",self.ext_it,", PSI_B residu = ", L2residu)
                print(" ")
        return 
    
    def UpdatePSI_B(self):
        """
        Updates the PSI_B arrays.
        """
        if self.ext_cvg == False:
            self.PSI_B[:,0] = self.PSI_B[:,1]
            self.PSI_NORMstar[:,0] = self.PSI_NORMstar[:,1]
            self.PSI_NORM[:,0] = self.PSI_NORM[:,1]
        elif self.ext_cvg == True:
            self.PSI_CONV = self.PSI_NORM[:,1]
        return
    
    def UpdateElementalPSI(self):
        """ 
        Function to update the elemental PSI values, respect to PSI_NORM.
        """
        for ELEMENT in self.MESH.Elements:
            ELEMENT.PSIe = self.PSI_NORM[ELEMENT.Te,0]  # TAKE VALUES OF ITERATION N
        return
    
    def UpdatePlasmaBoundaryValues(self):
        """
        Updates the plasma boundary PSI values constraints (PSIgseg) on the interface approximation segments integration points.
        """
        for ielem in self.MESH.PlasmaBoundElems:
            INTAPPROX = self.MESH.Elements[ielem].InterfApprox
            # INITIALISE BOUNDARY VALUES
            INTAPPROX.PSIg = np.zeros([INTAPPROX.ng])
            # FOR EACH INTEGRATION POINT ON THE PLASMA/VACUUM INTERFACE APPROXIMATION SEGMENT
            for ig in range(INTAPPROX.ng):
                # FIXED BOUNDARY PROBLEM -> ANALYTICAL SOLUTION PLASMA BOUNDARY VALUES 
                if self.FIXED_BOUNDARY:
                    INTAPPROX.PSIg[ig] = self.PlasmaCurrent.PSIanalytical(INTAPPROX.Xg[ig,:],NORMALISED=True)
                # FREE BOUNDARY PROBLEM -> PLASMA BOUNDARY VALUES = SEPARATRIX VALUE
                else:
                    INTAPPROX.PSIg[ig] = self.PSI_X
                    #INTAPPROX.PSIg[ig] = self.PSIseparatrix
        return
    
    def UpdateVacuumVesselBoundaryValues(self):
        
        PSI_Bextend = np.zeros([self.MESH.Nn])
        for inode in range(self.MESH.Nnbound):
            PSI_Bextend[self.MESH.BoundaryNodes[inode]] = self.PSI_B[inode,1]
        
        for ielem in self.MESH.DirichletElems:
            self.MESH.Elements[ielem].PSI_Be = PSI_Bextend[self.MESH.Elements[ielem].Te]
        
        return
    

    
    def ComputePSILevelSet(self,PSI):
        
        # OBTAIN POINTS CONFORMING THE NEW PLASMA DOMAIN BOUNDARY
        fig, ax = plt.subplots(figsize=(6, 8))
        cs = ax.tricontour(self.MESH.X[:,0],self.MESH.X[:,1], PSI-1.0, levels=[self.PSIseparatrix-1.0])

        paths = list()

        # CHECK IF CONTOUR SETS CONTAINS SADDLE POINT OR COMPUTATIONAL BOUNDARY POINTS (CLOSE ENOUGH) 
        for item in cs.collections:
            for path in item.get_paths():
                path_dict = dict()
                path_dict['coords'] = path.vertices
                path_dict['saddlepoint'] = False
                path_dict['compbound'] = False
                for point in path.vertices:
                    # COMPUTE DISTANCE TO SADDLE POINT
                    dist_saddle = np.linalg.norm(point-self.Xcrit[1,1,0:2])
                    # COMPUTE DISTANCE TO COMPUTATIONAL BOUNDARY NODES
                    dist_bound = np.sqrt((self.MESH.X[self.MESH.BoundaryNodes,0]-point[0])**2+(self.MESH.X[self.MESH.BoundaryNodes,1]-point[1])**2)
                    # CHECK IF CONTOUR CONTAINS SADDLE POINT
                    if  dist_saddle < 0.1:
                        path_dict['saddlepoint'] = True
                        # CHECK IF CONTOUR CONTAINS COMPUTATIONAL DOMAIN BOUNDARY POINTS
                    elif np.any(dist_bound <= 0.1):
                        path_dict['compbound'] = True
                paths.append(path_dict)

        # DIFFERENT PROCEDURES:
        # 1. DISCARD SETS WHICH DO NOT CONTAIN THE SADDLE POINT
        paths_temp = list()
        for path in paths:
            if path['saddlepoint']:
                paths_temp.append(path)
        paths = paths_temp.copy()        

        # IF THERE ARE MORE THAN 1 CONTOUR SET CONTAINING THE SADDLE POINT, REMOVE THE SETS CONTAINING COMPUTATIONAL BOUNDARY POINTS
        if len(paths) > 1:
            paths_temp = list()
            for path in paths:
                if not path['compbound']:
                    paths_temp.append(path)
            paths = paths_temp.copy()
            # TAKE THE REMAINING SET AS THE NEW PLASMA BOUNDARY SET
            if len(paths) == 1:
                plasmaboundary = paths[0]['coords']
                
        # IF A SINGLE CONTOUR REMAINS, CHECK WHETHER IT CONTAINS COMPUTATIONAL BOUNDARIES 
        else:
            # IF THE REMAINING SET CONTAINS BOTH SADDLE POINT AND COMPUTATIONAL BOUNDARY POINTS
            if paths[0]['compbound']:       
                plasmaboundary = list()
                oncontour = False
                firstpass = True
                secondpass = False
                counter = 0
                for point in paths[0]['coords']:
                    if np.linalg.norm(point-self.Xcrit[1,1,0:2]) < 0.3 and firstpass:
                        oncontour = True 
                        firstpass = False
                        plasmaboundary.append(point)
                    elif oncontour:
                        plasmaboundary.append(point)
                        counter += 1
                    if counter > 50:
                        secondpass = True
                    if np.linalg.norm(point-self.Xcrit[1,1,0:2]) < 0.3 and secondpass: 
                        oncontour = False 
                                
                plasmaboundary.append(plasmaboundary[0])
                plasmaboundary = np.array(plasmaboundary)
            # IF THE REMAINING SET DOES NOT CONTAIN ANY COMPUTATIONAL BOUNDARY POINT, TAKE IT AS THE NEW PLASMA BOUNDARY SET 
            else: 
                plasmaboundary = paths[0]['coords']

        fig.clear()
        plt.close(fig)

        # Create a Path object for the new plasma domain
        polygon_path = Path(plasmaboundary)
        # Check if the mesh points are inside the new plasma domain
        inside = polygon_path.contains_points(self.MESH.X)

        # FORCE PLASMA LEVEL-SET SIGN DEPENDING ON REGION
        PSILevSet = 1.0-PSI.copy()
        for inode in range(self.MESH.Nn):
            if inside[inode]:
                PSILevSet[inode] = -np.abs(PSILevSet[inode])
            else:
                PSILevSet[inode] = np.abs(PSILevSet[inode])
    
        return PSILevSet
    
    
    
    def UpdateElementalPlasmaLevSet(self):
        for ELEMENT in self.MESH.Elements:
            ELEMENT.LSe = self.PlasmaLS[self.MESH.T[ELEMENT.index,:],1]
        return
    
    
    def UpdatePlasmaRegion(self,RELAXATION=False):
        """
        If necessary, the level-set function is updated according to the new normalised solution's 0-level contour.
        If the new saddle point is close enough to the old one, the function exits early, assuming the plasma region is already well-defined.
        
        On the contrary, it updates the following:
            1. Plasma boundary level-set function values.
            2. Plasma region classification.
            3. Plasma boundary approximation and normal vectors.
            4. Numerical integration quadratures for the plasma and vacuum elements.
            5. Updates nodes on the plasma boundary approximation.
        """
                
        if not self.FIXED_BOUNDARY:
            # IN CASE WHERE THE NEW SADDLE POINT (N+1) CORRESPONDS (CLOSE TO) TO THE OLD SADDLE POINT, THEN THAT MEANS THAT THE PLASMA REGION
            # IS ALREADY WELL DEFINED BY THE OLD LEVEL-SET 
            
            if self.it >= self.it_plasma and np.linalg.norm(self.Xcrit[1,1,:-1]-self.Xcrit[0,1,:-1]) > 0.2:

                ###### UPDATE PLASMA REGION LEVEL-SET FUNCTION VALUES ACCORDING TO SOLUTION OBTAINED
                # . RECALL THAT PLASMA REGION IS DEFINED BY NEGATIVE VALUES OF LEVEL-SET -> NEED TO INVERT SIGN
                # . CLOSED GEOMETRY DEFINED BY 0-LEVEL CONTOUR BENEATH ACTIVE SADDLE POINT (DIVERTOR REGION) NEEDS TO BE
                #   DISCARTED BECAUSE THE LEVEL-SET DESCRIBES ONLY THE PLASMA REGION GEOMETRY -> NEED TO POST-PROCESS CUTFEM
                #   SOLUTION IN ORDER TO TAKE ITS 0-LEVEL CONTOUR ENCLOSING ONLY THE PLASMA REGION.  
                
                self.PlasmaLSstar[:,1] = self.ComputePSILevelSet(self.PSI_NORM[:,1])
                
                # AITKEN RELAXATION FOR PLASMA REGION EVOLUTION
                if RELAXATION:
                    residual1 = self.PlasmaLSstar[:,1] - self.PlasmaLS[:,1] 
                    if self.it > 2:
                        residual0 = self.PlasmaLSstar[:,0] - self.PlasmaLS[:,0]
                        self.alphaPHI = - (residual1-residual0)@residual1/np.linalg.norm(residual1-residual0)
                    newPlasmaLS = self.PlasmaLS[:,1] + self.alphaPHI*residual1
                    
                    # SHOULD THE CRITICAL POINT BE IN AGREEMENT WITH THE RELAXED PLASMA REGION?
                    
                else:
                    newPlasmaLS = self.PlasmaLSstar[:,1]
                
                # UPDATE PLASMA LS
                self.PlasmaLSstar[:,0] = self.PlasmaLSstar[:,1]
                self.PlasmaLS[:,0] = self.PlasmaLS[:,1]
                self.PlasmaLS[:,1] = newPlasmaLS
                
                ###### RECOMPUTE ALL PLASMA BOUNDARY ELEMENTS ATTRIBUTES
                # UPDATE PLASMA REGION LEVEL-SET ELEMENTAL VALUES     
                self.UpdateElementalPlasmaLevSet()
                # CLASSIFY ELEMENTS ACCORDING TO NEW LEVEL-SET
                self.PlasmaLS[:,1] = self.MESH.ClassifyElements(self.PlasmaLS[:,1])
                # RECOMPUTE PLASMA BOUNDARY APPROXIMATION and NORMAL VECTORS
                self.MESH.ComputePlasmaBoundaryApproximation()
                # REIDENTIFY PLASMA BOUNDARY GHOST FACES
                if self.GhostStabilization:
                    self.MESH.ComputePlasmaBoundaryGhostFaces()
                
                ###### RECOMPUTE NUMERICAL INTEGRATION QUADRATURES
                # COMPUTE STANDARD QUADRATURE ENTITIES FOR NON-CUT ELEMENTS
                for ielem in np.concatenate((self.MESH.PlasmaElems, self.MESH.VacuumElems), axis = 0):
                    self.MESH.Elements[ielem].ComputeStandardQuadrature2D(self.QuadratureOrder2D)
                # COMPUTE ADAPTED QUADRATURE ENTITIES FOR INTERFACE ELEMENTS
                for ielem in self.MESH.PlasmaBoundElems:
                    self.MESH.Elements[ielem].ComputeAdaptedQuadratures(self.QuadratureOrder2D,self.QuadratureOrder1D)
                # CHECK NORMAL VECTORS
                self.MESH.CheckPlasmaBoundaryApproximationNormalVectors()
                # COMPUTE PLASMA BOUNDARY GHOST FACES QUADRATURES
                if self.GhostStabilization:
                    for ielem in self.MESH.GhostElems: 
                        self.MESH.Elements[ielem].ComputeGhostFacesQuadratures(self.QuadratureOrder1D)
                    
                # RECOMPUTE NUMBER OF NODES ON PLASMA BOUNDARY APPROXIMATION 
                self.MESH.NnPB = self.MESH.ComputePlasmaBoundaryNumberNodes()
                
                # WRITE NEW PLASMA REGION DATA
                self.writePlasmaBoundaryData()
            
            # UPDATE CRITICAL VALUES
            self.Xcrit[0,:,:] = self.Xcrit[1,:,:]        
            return