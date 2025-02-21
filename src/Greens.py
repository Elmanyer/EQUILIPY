


import numpy as np

##################################################################################################
##################################### GREEN'S FORMALISM ##########################################
##################################################################################################

def ellipticK(k):
    """ 
    COMPLETE ELLIPTIC INTEGRAL OF 1rst KIND 
    """
    pk=1.0-k*k
    if k == 1:
        ellipticK=1.0e+16
    else:
        AK = (((0.01451196212*pk+0.03742563713)*pk +0.03590092383)*pk+0.09666344259)*pk+1.38629436112
        BK = (((0.00441787012*pk+0.03328355346)*pk+0.06880248576)*pk+0.12498593597)*pk+0.5
        ellipticK = AK-BK*np.log(pk)
    return ellipticK

def ellipticE(k):
    """
    COMPLETE ELLIPTIC INTEGRAL OF 2nd KIND
    """
    pk = 1 - k*k
    if k == 1:
        ellipticE = 1
    else:
        AE=(((0.01736506451*pk+0.04757383546)*pk+0.0626060122)*pk+0.44325141463)*pk+1
        BE=(((0.00526449639*pk+0.04069697526)*pk+0.09200180037)*pk+0.2499836831)*pk
        ellipticE = AE-BE*np.log(pk)
    return ellipticE

def GreensFunction(Xb,Xp):
    """ 
    GREEN FUNCTION ASSOCIATED TO THE GRAD-SHAFRANOV'S EQUATION ELLIPTIC OPERATOR. 
    """
    k= np.sqrt(4*Xb[0]*Xp[0]/((Xb[0]+Xp[0])**2 + (Xp[1]-Xb[1])**2))
    Greenfun = (1/(2*np.pi))*(np.sqrt(Xp[0]*Xb[0])/k)*((2-k**2)*ellipticK(k)-2*ellipticE(k))
    return Greenfun

def GreensBz(Xb,Xp,eps=1e-8):
    """
    Calculate vertical magnetic field at (R,Z)
    due to unit current at (Rc, Zc)

    Bz = (1/R) d psi/dR
    """

    return (GreensFunction(Xb,np.array([Xp[0]+eps,Xp[1]])) - GreensFunction(Xb,np.array([Xp[0]-eps,Xp[1]]))) / (2.0 * eps * Xp[0])

def GreensBr(Xb,Xp,eps=1e-8):
    """
    Calculate radial magnetic field at (R,Z)
    due to unit current at (Rc, Zc)

    Br = -(1/R) d psi/dZ
    """

    return (GreensFunction(Xb,np.array([Xp[0],Xp[1]-eps])) - GreensFunction(Xb,np.array([Xp[0],Xp[1]+eps]))) / (2.0 * eps * Xp[0])