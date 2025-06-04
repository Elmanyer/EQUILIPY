
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
                