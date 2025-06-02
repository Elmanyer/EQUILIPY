
from weakref import proxy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.path import Path
from AnalyticalSolutions import *
from functools import partial

class InitialPlasmaBoundary:
    
    def __init__(self,PROBLEM,GEOMETRY,**kwargs):
        # IMPORT PROBLEM DATA
        self.problem = proxy(PROBLEM)
        # INITIAL BOUNDARY PREDEFINED MODELS
        self.INITIAL_GEOMETRY = None
        self.LINEAR_SOLUTION = 0
        self.ZHENG_SOLUTION = 1
        self.F4E_HAMILTONIAN = 2
        self.OTHER_PARAMETRIATION = 3
        
        ##### PRE-DEFINED INITIAL PLASMA BOUNDARY GEOMETRIES
        match GEOMETRY:
            case 'LINEAR':
                # GEOMETRY PARAMETERS
                self.INITIAL_GEOMETRY = self.LINEAR_SOLUTION
                self.R0 = kwargs['R0']                    # MEAN RADIUS
                self.epsilon = kwargs['epsilon']          # INVERSE ASPECT RATIO
                self.kappa = kwargs['kappa']              # ELONGATION
                self.delta = kwargs['delta']              # TRIANGULARITY
                self.coeffs = ComputeLinearSolutionCoefficients(self.R0,self.epsilon,self.kappa,self.delta)
                # GEOMETRY LEVEL-SET FUNCTION
                self.PHI0 = partial(PSIanalyticalLINEAR, R0=self.R0, coeffs=self.coeffs)
            
            case 'ZHENG':
                # GEOMETRY PARAMETERS
                self.INITIAL_GEOMETRY = self.ZHENG_SOLUTION
                self.R0 = kwargs['R0']                     # MEAN RADIUS
                self.epsilon = kwargs['epsilon']           # INVERSE ASPECT RATIO
                self.kappa = kwargs['kappa']               # ELONGATION
                self.delta = kwargs['delta']               # TRIANGULARITY
                self.coeffs = ComputeZhengSolutionCoefficients(self.R0,self.epsilon,self.kappa,self.delta)
                # GEOMETRY LEVEL-SET FUNCTION
                self.PHI0 = self.ZHENG_LS
                
            case 'F4E':
                # GEOMETRY PARAMETERS
                self.INITIAL_GEOMETRY = self.F4E_HAMILTONIAN
                self.X_SADDLE = kwargs['Xsaddle']          # ACTIVE SADDLE POINT
                self.X_RIGHT = kwargs['Xright']            # POINT ON THE RIGHT
                self.X_LEFT = kwargs['Xleft']              # POINT ON THE LEFT
                self.X_TOP = kwargs['Xtop']                # POINT ON TOP
                self.coeffs = self.ComputeF4EPlasmaLScoeffs()
                # GEOMETRY LEVEL-SET FUNCTION
                self.PHI0 = self.F4EPlasmaLS
                
            case 'OTHER':
                self.INITIAL_GEOMETRY = self.OTHER_PARAMETRIATION      
                self.PHI0 = kwargs['PHI0'] 
        
        return
    
##################################################################################################
##################################### F4E PARAMETRISATION ########################################
##################################################################################################

    def ZHENG_LS(self,X):
        return -PSIanalyticalZHENG(X,self.coeffs)
        
##################################################################################################
##################################### F4E PARAMETRISATION ########################################
##################################################################################################
    
    def ComputeF4EPlasmaLScoeffs(self):
        """ # IN ORDER TO FIND THE CURVE PARAMETRIZING THE PLASMA REGION BOUNDARY, WE LOOK FOR THE COEFFICIENTS DEFINING
        # A 3rd ORDER HAMILTONIAN FROM WHICH WE WILL TAKE THE 0-LEVEL CURVE AS PLASMA REGION BOUNDARY. THAT IS
        #
        # H(x,y) = A00 + A10x + A01y + A20x**2 + A11xy + A02y**2 + A30x**3 + A21x**2y + A12xy**2 + A03y**3
        # 
        # HENCE, WE NEED TO IMPOSE CONSTRAINTS ON THE HAMILTONIAN FUNCTION IN ORDER TO SOLVE A SYSTEM OF EQUATIONS 
        # (LINEAR OR NONLINEAR). THE RESULTING SYSTEM WILL READ AS   Ax = b.
        # IN ORDER TO SIMPLIFY SUCH PROBLEM, WE ASSUME THAT:
        #   - ORIGIN (0,0) ON 0-LEVEL CURVE ---> A00 = 0
        #   - SADDLE POINT AT (0,0) ---> A10 = A01 = 0 
        # EVEN IF THAT IS NOT THE CASE IN THE PHYSICAL PLASMA REGION, WE ONLY NEED TO TRANSLATE THE REFERENCE FRAME 
        # RESPECT TO THE REAL SADDLE POINT LOCATION P0 IN ORDER TO WORK WITH EQUIVALENT PROBLEMS.
        # FURTHERMORE, WE CAN NORMALISE RESPECT TO A20 WITHOUT LOSS OF GENERALITY. THEREFORE, WE DEPART FROM 
        #
        # H(x,y) = x**2 + A11xy + A02y**2 + A30x**3 + A21x**2y + A12xy**2 + A03y**3
        # 
        # AS MENTIONED EARLIER, THE PROFILE WILL CORRESPOND TO THE 0-LEVEL CURVE, WHICH MEANS WE MUST OBTAIN THE 
        # COEFFICIENTS FOR 
        #
        # A11xy + A02y**2 + A30x**3 + A21x**2y + A12xy**2 + A03y**3 = -x**2
        #
        # WE NEED HENCE TO IMPOSE 6 CONSTRAINTS IN ORDER TO DETERMINE THE REMAINING COEFFICIENTS
        
        # For this method we constraint the curve to:
        # - go through points P1, P2 and P3 (CONTROL POINTS)
        # - have vertical tangents at points P1 and P2
        # - have a 90º angle at saddle point
        
        # where the control points are defined as:
        #      - P0: SADDLE POINT
        #      - P1: RIGHTMOST POINT
        #      - P2: LEFTMOST POINT
        #      - P3: TOP POINT
        
        # Input: - P0: SADDLE POINT COORDINATES
        #        - P1: RIGHTMOST POINT COORDINATES
        #        - P2: LEFTMOST POINT COORDINATES
        #        - P3: TOP POINT COORDINATES
        #        - X: NODAL COORDINATES MATRIX
        # """
        
        # THE FOLLOWING FUNCTIONS TRANSLATE THE CONSTRAINTS ON THE PROBLEM INTO EQUATIONS FOR THE FINAL SYSTEM OF EQUATIONS TO SOLVE
        def Point_on_curve(P):
            # Function returning the row coefficients in the system Ax=b corresponding to the equation 
            # obtained when constraining the curve to pass through point P. Such equation corresponds 
            # basically to   H(P) = 0.
            x, y = P
            Arow = [x*y, y**2, x**3, x**2*y, x*y**2, y**3]
            brow = -x**2
            return Arow, brow

        def VerticalTangent(P):
            # Function returning the row coefficients in the system Ax=b corresponding to the equation
            # obtained when constraining the curve to have a vertical tangent at point P. Such equation  
            # corresponds basically to   dH/dy(P) = 0.
            x, y = P
            Arow = [x, 2*y, 0, x**2, 2*x*y, 3*y**2]
            brow = 0
            return Arow, brow

        def HorizontalTangent(P):
            # Function returning the row coefficients in the system Ax=b corresponding to the equation
            # obtained when constraining the curve to have a horizontal tangent at point P. Such equation  
            # corresponds basically to   dH/dx(P) = 0.
            x, y = P
            Arow = [y, 0, 3*x**2, 2*x*y, y**2, 0]
            brow = -2*x
            return Arow, brow

        def RightAngle_SaddlePoint(A,b):
            # Function imposing a 90º angle at the closed surface saddle point at (0,0), which can be shown 
            # is equivalent to fixing  A02 = -1
            # Hence, what we need to do is take the second column of matrix A, corresponding to the A02 factors,
            # multiply them by -1 and pass them to the system's RHS, vector b. Then, we will reduce the system size.
            
            bred = np.delete(b+A[:,1].reshape((6,1)),5,0)     # pass second column to RHS and delete last row
            A = np.delete(A,1,1)    # delete second column 
            Ared = np.delete(A,5,0)    # delete last row
            return Ared, bred
        
        # 1. RESCALE POINT COORDINATES SO THAT THE SADDLE POINT IS LOCATED AT ORIGIN (0,0)
        P1star = self.X_RIGHT-self.X_SADDLE
        P2star = self.X_LEFT-self.X_SADDLE
        P3star = self.X_TOP-self.X_SADDLE

        # 2. COMPUTE HAMILTONIAN COEFFICIENTS
        # Build system matrices
        A = np.zeros([6,6])
        b = np.zeros([6,1])

        # Constraints on point P1 = (a1,b1)
        Arow11, brow11 = Point_on_curve(P1star)
        Arow12, brow12 = VerticalTangent(P1star)
        A[0,:] = Arow11
        b[0] = brow11
        A[1,:] = Arow12
        b[1] = brow12

        # Constraints on point P2 = (a2,b2)
        Arow21, brow21 = Point_on_curve(P2star)
        Arow22, brow22 = VerticalTangent(P2star)
        A[2,:] = Arow21
        b[2] = brow21
        A[3,:] = Arow22
        b[3] = brow22
        
        # Constraints on point P3 = (a3,b3)
        Arow31, brow31 = Point_on_curve(P3star)
        A[4,:] = Arow31
        b[4] = brow31

        # 90º on saddle point (0,0)
        Ared, bred = RightAngle_SaddlePoint(A,b)   # Now A = [5x5] and  b = [5x1]
        
        # Solve system of equations and obtain Hamiltonian coefficients
        Q, R = np.linalg.qr(Ared)
        y = np.dot(Q.T, bred)
        coeffs_red = np.linalg.solve(R, y)  # Hamiltonian coefficients  [5x1]
        
        coeffs = np.insert(coeffs_red,1,-1,0)        # insert second coefficient A02 = -1
        return coeffs
    
    def F4EPlasmaLS(self,X):
        Xstar = X[0] - self.X_SADDLE[0]
        Ystar = X[1] - self.X_SADDLE[1]
        
        # HAMILTONIAN  ->>  Z(x,y) = H(x,y) = x**2 + A11xy + A02y**2 + A30x**3 + A21x**2y + A12xy**2 + A03y**3
        LS = Xstar**2+self.coeffs[0]*Xstar*Ystar+self.coeffs[1]*Ystar**2+self.coeffs[2]*Xstar**3+self.coeffs[3]*Xstar**2*Ystar+self.coeffs[4]*Xstar*Ystar**2+self.coeffs[5]*Ystar**3
        
        # MODIFY HAMILTONIAN VALUES SO THAT OUTSIDE THE PLASMA REGION THE LEVEL-SET IS POSITIVE  
        if X[0] < self.X_LEFT[0] or X[1] < self.X_SADDLE[1]:
            LS = np.abs(LS)
        return LS
    
    
##################################################################################################
########################################## REPRESENTATION ########################################
##################################################################################################
    
    def ComputeField(self,X):
        
        PHI0 = np.zeros([np.shape(X)[0]])
        for inode in range(np.shape(X)[0]):
            PHI0[inode] = self.PHI0(X[inode,:])
        return PHI0
    
##################################################################################################
########################################## REPRESENTATION ########################################
##################################################################################################
    
    def Plot(self):
        # PREPARE RECTANGULAR REPRESENTATION DOMAIN
        Rmax = np.max(self.problem.X[:,0])+0.1
        Rmin = np.min(self.problem.X[:,0])-0.1
        Zmax = np.max(self.problem.X[:,1])+0.1
        Zmin = np.min(self.problem.X[:,1])-0.1
        Nr = 50
        Nz = 60
        Xrec = np.zeros([Nr*Nz,2])
        inode = 0
        for r in np.linspace(Rmin,Rmax,Nr):
            for z in np.linspace(Zmin,Zmax,Nz):
                Xrec[inode,:] = [r,z]
                inode += 1
        # COMPUTE INITIAL PLASMA BOUNDARY FIELD
        PHI0 = self.ComputeField(Xrec)
        #### FIGURE
        # PLOT PHI LEVEL-SET BACKGROUND VALUES 
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_aspect('equal')
        ax.tricontourf(Xrec[:,0],Xrec[:,1],PHI0,levels=30)
        # PLOT MESH
        triang = tri.Triangulation(self.problem.X[:, 0], self.problem.X[:, 1])
        # Define computational domain's boundary path
        compboundary = np.zeros([len(self.problem.BoundaryVertices)+1,2])
        compboundary[:-1,:] = self.problem.X[self.problem.BoundaryVertices,:]
        # Close path
        compboundary[-1,:] = compboundary[0,:]
        clip_path = Path(compboundary)
        # Mask triangles whose centroids are outside
        xmid = self.problem.X[:,0][triang.triangles].mean(axis=1)
        ymid = self.problem.X[:,1][triang.triangles].mean(axis=1)
        mask = ~clip_path.contains_points(np.column_stack((xmid, ymid)))
        triang.set_mask(mask)
        ax.triplot(triang, color='gray')
        ax.plot(self.problem.X[:, 0], self.problem.X[:, 1], 'o', markersize=2)
        
        # PLOT MESH BOUNDARY
        for iboun in range(self.problem.Nbound):
            ax.plot(self.problem.X[self.problem.Tbound[iboun,:2],0],self.problem.X[self.problem.Tbound[iboun,:2],1],linewidth = 4, color = 'grey')
        # PLOT LEVEL-SET CONTOURS
        ax.tricontour(Xrec[:,0],Xrec[:,1],PHI0,levels=30,colors='black', linewidths=1)
        # PLOT INITIAL PLASMA BOUNDARY
        ax.tricontour(Xrec[:,0],Xrec[:,1],PHI0,levels=[0],colors='red', linewidths=3)
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        ax.set_title('Initial plasma domain')
        return