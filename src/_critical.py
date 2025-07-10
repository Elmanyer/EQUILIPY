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
        """
        Finds critical points (O-points and X-points) of the poloidal flux (PSI) on a plasma mesh.

        Inputs:
            - PSI (array): Poloidal flux values at mesh nodes.
            - X0 (list of arrays, optional): Initial guess points for critical points. If None, searches entire domain.
            - nr (int, optional): Number of radial points for finer grid interpolation (default 45).
            - nz (int, optional): Number of vertical points for finer grid interpolation (default 65).
            - tolbound (float, optional): Tolerance distance to computational boundary to exclude points (default 0.1).

        Outputs:
            - Opoint (list): List of tuples (coordinates, PSI value, element index) for O-points (local minima of Bp²).
            - Xpoint (list): List of tuples (coordinates, PSI value, element index) for X-points (saddle points).

        Tasks:
            - Interpolates PSI on a finer structured mesh.
            - Computes gradient and searches for local minima of poloidal magnetic field squared Bp².
            - Uses nonlinear root finding to refine critical point locations.
            - Filters points outside computational domain and close to boundary.
            - Removes duplicates from found points.
        """
            
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
        """
        Computes and updates the critical points (O-point and X-point) of the poloidal flux PSI.

        Tasks:
            - Defines initial guess points based on iteration count.
            - Calls FindCritical() with initial guesses to locate critical points.
            - Updates internal state variables with found critical points and their PSI values.
            - If the saddle point (X-point) is not found and boundary is not fixed, it reuses previous solution.
        """
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
    """
    Interpolates the gradient of PSI at point X using cubic interpolation.

    Inputs:
        - X (array-like): Coordinates [r, z] where gradient is evaluated.
        - Rfine, Zfine (2D arrays): Coordinates of the fine interpolation grid.
        - gradPSIfine (list of 2D arrays): Gradient components of PSI on the fine grid [dPSIdr, dPSIdz].

    Returns:
        - GRAD (np.array): Interpolated gradient vector [dPSIdr, dPSIdz] at X.
    """
    dPSIdr = griddata((Rfine.flatten(),Zfine.flatten()), gradPSIfine[0].flatten(), (X[0],X[1]), method='cubic')
    dPSIdz = griddata((Rfine.flatten(),Zfine.flatten()), gradPSIfine[1].flatten(), (X[0],X[1]), method='cubic')
    GRAD = np.array([dPSIdr,dPSIdz])
    return GRAD


# INTERPOLATION OF HESSIAN(PSI)
def hessianPSI(X,gradPSIfine,Rfine,Zfine,dr,dz):
    """
    Computes interpolated Hessian components of PSI at point X using cubic interpolation.

    Inputs:
        - X (array-like): Coordinates [r, z] where Hessian is evaluated.
        - gradPSIfine (list of 2D arrays): Gradient components of PSI on fine mesh.
        - Rfine, Zfine (2D arrays): Coordinates of the fine interpolation grid.
        - dr, dz (floats): Grid spacing in r and z directions.

    Returns:
        - dPSIdrdr (float): ∂²PSI/∂r² at X.
        - dPSIdzdr (float): ∂²PSI/∂z∂r at X.
        - dPSIdzdz (float): ∂²PSI/∂z² at X.
    """
    # compute second derivatives on fine mesh
    dgradPSIdrfine = np.gradient(gradPSIfine[0],dr,dz)
    dgradPSIdzfine = np.gradient(gradPSIfine[1],dr,dz)
    # interpolate HESSIAN components on point 
    dPSIdrdr = griddata((Rfine.flatten(),Zfine.flatten()), dgradPSIdrfine[0].flatten(), (X[0],X[1]), method='cubic')
    dPSIdzdr = griddata((Rfine.flatten(),Zfine.flatten()), dgradPSIdrfine[1].flatten(), (X[0],X[1]), method='cubic')
    dPSIdzdz = griddata((Rfine.flatten(),Zfine.flatten()), dgradPSIdzfine[1].flatten(), (X[0],X[1]), method='cubic')
    return dPSIdrdr, dPSIdzdr, dPSIdzdz
    