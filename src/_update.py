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
import matplotlib.pyplot as plt
from matplotlib.path import Path

class EquilipyUpdate:
    
    
    def FixElementalPSI_P(self):
        """
        Updates the plasma boundary PSI values constraints (PSIg) on the interface approximation integration nodes.
        """
        for ielem in self.MESH.PlasmaBoundActiveElems:
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
                    INTAPPROX.PSIg[ig] = self.PSIseparatrix
        return
    
    
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
                    
                # CHECK CONVERGENCE
                if L2residu < self.int_tol:
                    self.int_cvg = True   # STOP INTERNAL WHILE LOOP 
                else:
                    self.int_cvg = False
                
                # UPDATE VALUES
                
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
    
    def UpdatePSI_NORM(self):
        """
        Updates the normalized PSI solution and critical point arrays for the new iteration.
        """
        # UPDATE NORMALISED SOLUTION ARRAY
        self.PSI_NORM[:,0] = self.PSI_NORM[:,1]
        # UPDATE CRITICAL VALUES
        self.Xcrit[0,:,:] = self.Xcrit[1,:,:]
        return
    
    def UpdatePSI_B(self):
        """
        Updates the boundary PSI_B arrays.
        """
        if self.ext_cvg == False:
            self.PSI_B[:,0] = self.PSI_B[:,1]
            self.PSI_NORM[:,0] = self.PSI_NORM[:,1]
        elif self.ext_cvg == True:
            self.PSI_CONV = self.PSI_NORM[:,1]
        return
    
    
    def UpdateElementalPSI(self):
        """ 
        Function to update the elemental PSI values, respect to PSI_NORM.
        """
        for ELEMENT in self.MESH.Elements:
            ELEMENT.PSIe = self.PSI_NORM[ELEMENT.Te,1]  # TAKE VALUES OF ITERATION N
        return
    
    
    def UpdateElementalPSI_B(self):
        """
        Updates the essential boundary conditions PSI_B for boundary elements.
        """
        PSI_Bextend = np.zeros([self.MESH.Nn])
        for inode in range(self.MESH.Nnbound):
            PSI_Bextend[self.MESH.BoundaryNodes[inode]] = self.PSI_B[inode,1]
        
        for ielem in self.MESH.DirichletElems:
            self.MESH.Elements[ielem].PSI_Be = PSI_Bextend[self.MESH.Elements[ielem].Te]
        
        return
    

    
    def ComputePSILevelSet(self,PSI_NORM=None):
        """
        Computes the level-set function representing the plasma boundary based on 
        the normalized poloidal flux (PSI_NORM).

        Parameters:
        -----------
        PSI_NORM : ndarray, optional
            Normalized PSI values to use for contour extraction. If not provided, 
            the method uses the latest solution (`self.PSI_NORM[:,1]`).

        Description:
        ------------
        This method identifies the plasma boundary by extracting the contour line 
        corresponding to the separatrix value (`self.PSI_NORMseparatrix`) from the 
        normalized PSI field. It ensures the extracted contour:
            - Contains the magnetic saddle point (X-point).
            - Does not lie along the computational boundary (if avoidable).

        If multiple valid contours exist, it selects the most physically consistent one.
        In cases where the contour also lies along the computational boundary, a segment 
        of the contour around the X-point is isolated.

        The selected contour is then optionally reduced to a maximum number of points 
        and used to define a signed distance level-set function (`self.PlasmaLS`):
            - Negative inside the plasma domain.
            - Positive outside the plasma domain.
        """
        if type(PSI_NORM) == type(None):
            psinorm = self.PSI_NORM[:,1]
        else:
            psinorm = PSI_NORM
        
        # OBTAIN POINTS CONFORMING THE NEW PLASMA DOMAIN BOUNDARY
        fig, ax = plt.subplots(figsize=(6, 8))
        cs = ax.tricontour(self.MESH.X[:,0],self.MESH.X[:,1], psinorm, levels=[self.PSI_NORMseparatrix])

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
                    if  dist_saddle < 0.2:
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
        
        # IF PLASMA BOUNDARY CURVE ABOVE maxnumpoints POINTS, BRING IT DOWN 
        maxnumpoints = 300
        for path in paths:
            if len(plasmaboundary[:,0]) > maxnumpoints:
                indices = np.linspace(0, len(plasmaboundary[:,0]) - 1, maxnumpoints, dtype=int)
                plasmaboundary = plasmaboundary[indices,:]
        
        """
        ### ACTIVATE FOR ULTRAFINE MESHES
        
        # CHECK RESULTING CURVE FOR ABRUPT DIRECTIONAL CHANGES
        smooth = True
        smoothplasmaboun = list()
        smoothplasmaboun.append(plasmaboundary[0,:])
        smoothplasmaboun.append(plasmaboundary[1,:])
        for inode in range(2,len(plasmaboundary[:,0])):
            # CHECK TANGENT VECTORS
            vect0 = plasmaboundary[inode,:] - plasmaboundary[inode-1,:]
            vect1 = plasmaboundary[inode-1,:] - plasmaboundary[inode-2,:]
            vect0 /= np.linalg.norm(vect0)
            vect1 /= np.linalg.norm(vect1) 
            if np.dot(vect0,vect1) < 0:
                smooth = not smooth
            if smooth:
                smoothplasmaboun.append(plasmaboundary[inode,:]) 
        plasmaboundary = np.array(smoothplasmaboun)
        """

        # Create a Path object for the new plasma domain
        polygon_path = Path(plasmaboundary)
        # Check if the mesh points are inside the new plasma domain
        inside = polygon_path.contains_points(self.MESH.X)

        # FORCE PLASMA LEVEL-SET SIGN DEPENDING ON REGION
        self.PlasmaLS = self.PSI_NORMseparatrix - psinorm.copy()
        for inode in range(self.MESH.Nn):
            if inside[inode]:
                self.PlasmaLS[inode] = -np.abs(self.PlasmaLS[inode])
            else:
                self.PlasmaLS[inode] = np.abs(self.PlasmaLS[inode])

        return 
    
    
    def UpdateElementalPlasmaLevSet(self):
        """
        Updates the elemental level-set values for each mesh element.
        """
        for ELEMENT in self.MESH.Elements:
            ELEMENT.LSe = self.PlasmaLS[self.MESH.T[ELEMENT.index,:]]
        return
    
    