
import numpy as np

##################################################################################################
######################################## LINEAR MODEL ############################################
##################################################################################################
    
def ComputeLinearSolutionCoefficients(epsilon,kappa,delta):
    """ 
    Computes the coeffients for the magnetic flux in the linear source term case, that is for 
            GRAD-SHAFRANOV EQ:  DELTA*(PSI) = R^2   (plasma current is linear such that Jphi = R/mu0)
    for which the exact solution is 
            PSI = R^4/8 + D1 + D2*R^2 + D3*(R^4-4*R^2*Z^2)
        This function returns coefficients D1, D2, D3
            
    Geometrical dimensionless parameters: 
            - epsilon: magnetic confinement cross-section inverse aspect ratio
            - kappa: magnetic confinement cross-section elongation
            - delta: magnetic confinement cross-section triangularity 
    """
            
    A = np.array([[1, (1+epsilon)**2, (1+epsilon)**4], 
                [1, (1-epsilon)**2, (1-epsilon)**4],
                [1, (1-delta*epsilon)**2, (1-delta*epsilon)**4-4*(1-delta*epsilon)**2*kappa**2*epsilon**2]])
    b = -(1/8)*np.array([[(1+epsilon)**4], [(1-epsilon)**4], [(1-delta*epsilon)**4]])
    
    coeffs = np.linalg.solve(A,b)
    return coeffs.T[0].tolist() 

def PSIanalyticalLINEAR(X,R0,coeffs,NORMALISED=False):
    # DIMENSIONLESS COORDINATES
    Xstar = X
    if not NORMALISED:
        Xstar = X/R0
    # ANALYTICAL SOLUTION
    PSIexact = (Xstar[0]**4)/8 + coeffs[0] + coeffs[1]*Xstar[0]**2 + coeffs[2]*(Xstar[0]**4-4*Xstar[0]**2*Xstar[1]**2)
    return PSIexact
    
##################################################################################################
######################################### ZHENG MODEL ############################################
##################################################################################################
    
def ComputeZhengSolutionCoefficients(R0,epsilon,kappa,delta):
    """ Computes the coefficients for the Grad-Shafranov equation analytical solution proposed in ZHENG paper. """
    Ri = R0*(1-epsilon)  # PLASMA SHAPE EQUATORIAL INNERMOST POINT R COORDINATE
    Ro = R0*(1+epsilon)  # PLASMA SHAPE EQUATORIAL OUTERMOST POINT R COORDINATE
    a = (Ro-Ri)/2                  # PLASMA MINOR RADIUS
    Rt = R0 - delta*a    # PLASMA SHAPE HIGHEST POINT R COORDINATE
    Zt = kappa*a              # PLASMA SHAPE HIGHEST POINT Z COORDINATE
    
    coeffs = np.zeros([6])
    
    # SET THE COEFFICIENT A2 TO 0 FOR SIMPLICITY
    coeffs[5] = 0
    # COMPUTE COEFFICIENT A1 BY IMPOSING A CONSTANT TOTAL TOROIDAL PLASMA CURRENT Ip
    #                   Jphi = (A1*R**2 - A2)/ R*mu0 
    # IF A2 = 0, WE HAVE THEN       Jphi = A1* (R/mu0)   THAT IS WHAT WE NEED TO INTEGRATE
    # HENCE,   A1 = Ip/integral(Jphi)
    
    #self.coeffsZHENG[4] = self.TOTAL_CURRENT/self.PlasmaDomainIntegral(fun)
    
    coeffs[4] = -0.1
    
    # FOR COEFFICIENTS C1, C2, C3 AND C4, WE SOLVE A LINEAR SYSTEM OF EQUATIONS BASED ON THE PLASMA SHAPE GEOMETRY
    A = np.array([[1,Ri**2,Ri**4,np.log(Ri)*Ri**2],
                    [1,Ro**2,Ro**4,np.log(Ro)*Ro**2],
                    [1,Rt**2,(Rt**2-4*Zt**2)*Rt**2,np.log(Rt)*Rt**2-Zt**2],
                    [0,2,4*(Rt**2-2*Zt**2),2*np.log(Rt)+1]])
    
    b = np.array([[-(coeffs[4]*Ri**4)/8],
                    [-(coeffs[4]*Ro**4)/8],
                    [-(coeffs[4]*Rt**4)/8+(coeffs[5]*Zt**2)/2],
                    [-(coeffs[4]*Rt**2)/2]])
    
    coeffs_red = np.linalg.solve(A,b)
    coeffs[:4] = coeffs_red.T[0].tolist()
    return coeffs

def PSIanalyticalZHENG(X,coeffs):
    PSIexact = coeffs[0]+coeffs[1]*X[0]**2+coeffs[2]*(X[0]**4-4*X[0]**2*X[1]**2)+coeffs[3]*(np.log(X[0])
                                *X[0]**2-X[1]**2)+(coeffs[4]*X[0]**4)/8 - (coeffs[5]*X[1]**2)/2
    return PSIexact
    

##################################################################################################
######################################## NONLINEAR MODEL #########################################
##################################################################################################

def PSIanalyticalNONLINEAR(X,R0,coeffs,NORMALISED=False):
    # DIMENSIONLESS COORDINATES
    Xstar = X
    if not NORMALISED:
        Xstar = X/R0
    # ANALYTICAL SOLUTION
    PSIexact = np.sin(coeffs[0]*(Xstar[0]+coeffs[2]))*np.cos(coeffs[1]*Xstar[1])  
    return PSIexact
                
                
##################################################################################################
##################################### F4E PARAMETRISATION ########################################
##################################################################################################
    
def ComputeF4EPlasmaLScoeffs(X_SADDLE,X_RIGHT,X_LEFT,X_TOP):
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
    P1star = X_RIGHT-X_SADDLE
    P2star = X_LEFT-X_SADDLE
    P3star = X_TOP-X_SADDLE

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


def F4EPlasmaLS(X,coeffs,X_SADDLE,X_LEFT):
    Xstar = X[0] - X_SADDLE[0]
    Ystar = X[1] - X_SADDLE[1]
    
    # HAMILTONIAN  ->>  Z(x,y) = H(x,y) = x**2 + A11xy + A02y**2 + A30x**3 + A21x**2y + A12xy**2 + A03y**3
    LS = Xstar**2 + coeffs[0]*Xstar*Ystar + coeffs[1]*Ystar**2 + coeffs[2]*Xstar**3 + coeffs[3]*Xstar**2*Ystar + coeffs[4]*Xstar*Ystar**2 + coeffs[5]*Ystar**3
    
    # MODIFY HAMILTONIAN VALUES SO THAT OUTSIDE THE PLASMA REGION THE LEVEL-SET IS POSITIVE  
    if X[0] < X_LEFT[0] or X[1] < X_SADDLE[1]:
        LS = np.abs(LS)
    return LS