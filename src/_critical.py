import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import root
from matplotlib.path import Path


##################################################################################################
############################### FIND CRITICAL POINTS #############################################
##################################################################################################

class EquilipyCritical:

    # SEARCH ELEMENT CONTAINING POINT IN MESH
    def SearchElement(self,X,searchelements):
        """
        Identify the element within a specified list searchelements and contains a given point X.

        Input:
            - X (array-like): Coordinates of the point to locate, specified as [x, y].
            - searchelements (list of int): List of element indices to search within.

        Output:
            elem (int or None): Index of the element containing the point X. 
                                Returns None if no element contains the point.
        """
        
        for elem in searchelements:
            if self.MESH.Elements[elem].isinside(X):
                return elem
            

    def FindCritical(self,PSI, X0 = None, nr=45, nz=65, tolbound=0.1):
            
        # 1. INTERPOLATE PSI VALUES ON A FINER STRUCTURED MESH USING PSI ON NODES
        # DEFINE FINER STRUCTURED MESH
        rfine = np.linspace(self.MESH.Rmin, self.MESH.Rmax, nr)
        zfine = np.linspace(self.MESH.Zmin, self.MESH.Zmax, nz)
        # INTERPOLATE PSI VALUES
        Rfine, Zfine = np.meshgrid(rfine,zfine, indexing='ij')
        PSIfine = griddata((self.MESH.X[:,0],self.MESH.X[:,1]), PSI, (Rfine, Zfine), method='cubic')
        # CORRECT VALUES AT INTERPOLATION POINTS OUTSIDE OF COMPUTATIONAL DOMAIN
        for ir in range(nr):
            for iz in range(nz):
                if np.isnan(PSIfine[ir,iz]):
                    PSIfine[ir,iz] = 0

        # 2. DEFINE GRAD(PSI) WITH FINER MESH VALUES USING FINITE DIFFERENCES
        dr = (self.MESH.Rmax-self.MESH.Rmin)/nr
        dz = (self.MESH.Zmax-self.MESH.Zmin)/nz
        gradPSIfine = np.gradient(PSIfine,dr,dz)
        
        # 3. LOOK FOR CRITICAL POINTS
        Xpoint = []
        Opoint = []
        if type(X0) == type(None):      # IF NO INITIAL GUESS POINTS ARE GIVEN, LOOK OVER WHOLE MESH
            
            # CREATE MASK FOR BACKGROUND MESH POINTS LYING OUTSIDE COMPUTATIONAL DOMAIN   
            grid_points = np.vstack([Rfine.ravel(), Zfine.ravel()]).T
            # Create polygon path and test containment
            path = Path(self.MESH.X[self.MESH.BoundaryVertices,:])
            mask = path.contains_points(grid_points)

            # Reshape mask to the grid shape
            mask_grid = mask.reshape(Rfine.shape)
            
            # 3. COMPUTE SQUARE MODUL OF POLOIDAL MAGNETIC FIELD Bp^2
            Bp2 = (gradPSIfine[0]**2 + gradPSIfine[1]**2)/Rfine**2

            # 4. FIND LOCAL MINIMA BY MINIMISING Bp^2
            for i in range(2, nr - 2):
                for j in range(2, nz - 2):
                    if (
                        (Bp2[i, j] < Bp2[i + 1, j + 1])
                        and (Bp2[i, j] < Bp2[i + 1, j])
                        and (Bp2[i, j] < Bp2[i + 1, j - 1])
                        and (Bp2[i, j] < Bp2[i - 1, j + 1])
                        and (Bp2[i, j] < Bp2[i - 1, j])
                        and (Bp2[i, j] < Bp2[i - 1, j - 1])
                        and (Bp2[i, j] < Bp2[i, j + 1])
                        and (Bp2[i, j] < Bp2[i, j - 1])
                    ):
                        # Found local minimum
                        # 5. CHECK IF POINT OUTSIDE OF COMPUTATIONAL DOMAIN
                        if mask_grid[i,j]:
                            # 6. LAUNCH NONLINEAR SOLVER
                            x0 = np.array([Rfine[i,j],Zfine[i,j]], dtype=float)
                            sol = root(gradPSI, x0, args=(Rfine,Zfine,gradPSIfine))

                            if sol.success == True:
                                # 7. INTERPOLATE VALUE OF PSI AT LOCAL EXTREMUM
                                elemcrit = self.SearchElement(sol.x,range(self.MESH.Ne))
                                if type(elemcrit) == type(None):
                                    # POINT OUTSIDE OF COMPUTATIONAL DOMAIN
                                    continue
                                else:
                                    PSIcrit = self.MESH.Elements[elemcrit].ElementalInterpolationPHYSICAL(sol.x,PSI[self.MESH.Elements[elemcrit].Te])
                                    
                                    # 8. CHECK LOCAL EXTREMUM HESSIAN 
                                    dPSIdrdr, dPSIdzdr, dPSIdzdz = hessianPSI(sol.x, gradPSIfine, Rfine, Zfine, dr, dz)
                                    if dPSIdrdr*dPSIdzdz-dPSIdzdr**2 > 0.0:
                                        # Found O-point
                                        Opoint.append((sol.x,PSIcrit,elemcrit))
                                    else:
                                        # Found X-point
                                        Xpoint.append((sol.x,PSIcrit,elemcrit))
                                        
        else:       # IF INITIAL GUESSES X0 ARE DEFINED
            for x0 in X0:
                sol = root(gradPSI, x0, args=(Rfine,Zfine,gradPSIfine))
                
                if not sol.success:
                    # LOOK IN SMALLER AREA NEAR GUESS WITH MORE RESOLUTION
                    rfinewindow = np.linspace(x0[0]-0.5, x0[0]+0.5, nr)
                    zfinewindow = np.linspace(x0[1]-0.5, x0[1]+0.5, nz)
                    # INTERPOLATE PSI VALUES
                    Rfinewindow, Zfinewindow = np.meshgrid(rfinewindow,zfinewindow, indexing='ij')
                    PSIfinewindow = griddata((self.MESH.X[:,0],self.MESH.X[:,1]), PSI, (Rfinewindow, Zfinewindow), method='cubic')
                    # CORRECT VALUES AT INTERPOLATION POINTS OUTSIDE OF COMPUTATIONAL DOMAIN
                    for ir in range(nr):
                        for iz in range(nz):
                            if np.isnan(PSIfinewindow[ir,iz]):
                                PSIfinewindow[ir,iz] = 0

                    # 2. DEFINE GRAD(PSI) WITH FINER MESH VALUES USING FINITE DIFFERENCES
                    drwindow = rfinewindow[1]-rfinewindow[0]
                    dzwindow = zfinewindow[1]-zfinewindow[0]
                    gradPSIfinewindow = np.gradient(PSIfinewindow,drwindow,dzwindow)
        
                    sol = root(gradPSI, x0, args=(Rfinewindow,Zfinewindow,gradPSIfinewindow))
                
                if sol.success:
                    # 4. LOCATE ELEMENT CONTAINING CRITICAL POINT
                    elemcrit = self.SearchElement(sol.x,range(self.MESH.Ne))
                    if type(elemcrit) == type(None):
                        # POINT OUTSIDE OF COMPUTATIONAL DOMAIN
                        continue
                    else:
                        # 5. INTERPOLATE CRITICAL PSI VALUE
                        PSIcrit = self.MESH.Elements[elemcrit].ElementalInterpolationPHYSICAL(sol.x,PSI[self.MESH.Elements[elemcrit].Te])
                        # 6. CHECK HESSIAN 
                        dPSIdrdr, dPSIdzdr, dPSIdzdz = hessianPSI(sol.x, gradPSIfine, Rfine, Zfine, dr, dz)
                        if dPSIdrdr*dPSIdzdz-dPSIdzdr**2 > 0.0:
                            # Found O-point
                            Opoint.append((sol.x,PSIcrit,elemcrit))
                        else:
                            # Found X-point
                            Xpoint.append((sol.x,PSIcrit,elemcrit))
        
        # Remove duplicates
        def remove_dup(points):
            result = []
            for p in points:
                dup = False
                for p2 in result:
                    if (p[0][0] - p2[0][0]) ** 2 + (p[0][1] - p2[0][1]) ** 2 < 1e-5:
                        dup = True  # Duplicate
                        break
                if not dup:
                    result.append(p)  # Add to the list
            return result

        Xpoint = remove_dup(Xpoint)
        Opoint = remove_dup(Opoint)  

        # Check distance to computational boundary
        def remove_closeboundary(points):
            for boundarypoint in self.MESH.BoundaryNodes:
                Xbound = self.MESH.X[boundarypoint,:]
                for p in points:
                    if np.linalg.norm(p[0]-Xbound) < tolbound:
                        points.remove(p)
            return points
                    
        Xpoint = remove_closeboundary(Xpoint)
        Opoint = remove_closeboundary(Opoint)    
                
        if len(Opoint) == 0:
            # Can't order primary O-point, X-point so return
            print("Warning: No O points found")
            return Opoint, Xpoint

        """
        # Find primary O-point by sorting by distance from middle of domain
        Rmid = 0.5 * (self.MESH.Rmax-self.MESH.Rmin)
        Zmid = 0.5 * (self.MESH.Zmax-self.MESH.Zmin)
        Opoint.sort(key=lambda x: (x[0] - Rmid) ** 2 + (x[1] - Zmid) ** 2)
        """    
        return Opoint, Xpoint
    
    
    def ComputeCriticalPSI(self):
        # DEFINE INITIAL GUESSES
        X0 = list()
        if self.it == 1:
            X0.append(np.array([self.R0_axis,self.Z0_axis],dtype=float))
            X0.append(np.array([self.R0_saddle,self.Z0_saddle],dtype=float))
        else:
            # TAKE PREVIOUS SOLUTION AS INITIAL GUESS
            X0.append(self.Xcrit[0,0,:-1])
            X0.append(self.Xcrit[0,1,:-1])
        
        Opoint, Xpoint = self.FindCritical(self.PSI.T[0], X0)   
        #print('Opoint = ', Opoint)
        #print('Xpoint = ', Xpoint)
        # O-point
        self.Xcrit[1,0,:-1] = Opoint[0][0] 
        self.PSI_0 = Opoint[0][1]
        self.Xcrit[1,0,-1] = Opoint[0][2] 
        print('LOCAL EXTREMUM AT ',self.Xcrit[1,0,:-1],' (ELEMENT ', int(self.Xcrit[1,0,-1]),') WITH VALUE PSI_0 = ',self.PSI_0)
        
        # X-point
        if not self.FIXED_BOUNDARY: 
            if not Xpoint and self.it > 1:
                print("SADDLE POINT NOT FOUND, TAKING PREVIOUS SOLUTION")
                self.Xcrit[1,1,:] = self.Xcrit[0,1,:]
            else:
                self.Xcrit[1,1,:-1] = Xpoint[0][0] 
                self.PSI_X = Xpoint[0][1]
                self.Xcrit[1,1,-1] = Xpoint[0][2]
            print('SADDLE POINT AT ',self.Xcrit[1,1,:-1],' (ELEMENT ', int(self.Xcrit[1,1,-1]),') WITH VALUE PSI_X = ',self.PSI_X)
        return


    ##################################################################################################
    #################################### NORMALISE SOLUTION ##########################################
    ##################################################################################################

    def NormalisePSI(self):
        """
        Normalize the magnetic flux function (PSI) based on critical PSI values (PSI_0 and PSI_X).
        """
        if not self.FIXED_BOUNDARY:
            for i in range(self.MESH.Nn):
                self.PSI_NORM[i,1] = (self.PSI[i]-self.PSI_0)/(self.PSI_X-self.PSI_0)
        else: 
            for i in range(self.MESH.Nn):
                self.PSI_NORM[i,1] = self.PSI[i]
        return 
    
    
# INTERPOLATION OF GRAD(PSI)
def gradPSI(X,Rfine,Zfine,gradPSIfine):
    dPSIdr = griddata((Rfine.flatten(),Zfine.flatten()), gradPSIfine[0].flatten(), (X[0],X[1]), method='cubic')
    dPSIdz = griddata((Rfine.flatten(),Zfine.flatten()), gradPSIfine[1].flatten(), (X[0],X[1]), method='cubic')
    GRAD = np.array([dPSIdr,dPSIdz])
    return GRAD

# INTERPOLATION OF HESSIAN(PSI)
def hessianPSI(X,gradPSIfine,Rfine,Zfine,dr,dz):
    # compute second derivatives on fine mesh
    dgradPSIdrfine = np.gradient(gradPSIfine[0],dr,dz)
    dgradPSIdzfine = np.gradient(gradPSIfine[1],dr,dz)
    # interpolate HESSIAN components on point 
    dPSIdrdr = griddata((Rfine.flatten(),Zfine.flatten()), dgradPSIdrfine[0].flatten(), (X[0],X[1]), method='cubic')
    dPSIdzdr = griddata((Rfine.flatten(),Zfine.flatten()), dgradPSIdrfine[1].flatten(), (X[0],X[1]), method='cubic')
    dPSIdzdz = griddata((Rfine.flatten(),Zfine.flatten()), dgradPSIdzfine[1].flatten(), (X[0],X[1]), method='cubic')
    return dPSIdrdr, dPSIdzdr, dPSIdzdz
    