from weakref import proxy
import numpy as np
from AnalyticalSolutions import *
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.integrate import quad

class CurrentModel:
    
    mu0 = 12.566370E-7           # H m-1    Magnetic permeability
    
    def __init__(self,PROBLEM,MODEL,**kwargs):
        # IMPORT PROBLEM DATA
        self.problem = proxy(PROBLEM)
        # PLASMA CURRENT PREDEFINED MODELS
        self.CURRENT_MODEL = None
        self.LINEAR_CURRENT = 0
        self.ZHENG_CURRENT = 1
        self.NONLINEAR_CURRENT = 2
        self.PROFILES_CURRENT = 3
        self.OTHER_CURRENT = 4
        
        self.PSIdependent = False
        
        ##### PRE-DEFINED PLASMA CURRENT MODELS
        match MODEL:
            # LINEAR PLASMA CURRENT 
            case 'LINEAR': 
                # MODEL PARAMETERS
                self.CURRENT_MODEL = self.LINEAR_CURRENT
                self.R0 = kwargs['R0']                    # MEAN RADIUS
                self.epsilon = kwargs['epsilon']          # INVERSE ASPECT RATIO
                self.kappa = kwargs['kappa']              # ELONGATION
                self.delta = kwargs['delta']              # TRIANGULARITY
                self.coeffs = ComputeLinearSolutionCoefficients(self.epsilon,self.kappa,self.delta)
                # MODEL PLASMA CURRENT
                self.Jphi = self.JphiLINEAR
                # MODEL ANALYTICAL SOLUTION
                self.PSIanalytical = partial(PSIanalyticalLINEAR, R0=self.R0, coeffs=self.coeffs)
          
            # ZHENG PLASMA CURRENT
            case 'ZHENG':
                # MODEL PARAMETERS
                self.CURRENT_MODEL = self.ZHENG_CURRENT
                self.R0 = kwargs['R0']                     # MEAN RADIUS
                self.epsilon = kwargs['epsilon']           # INVERSE ASPECT RATIO
                self.kappa = kwargs['kappa']               # ELONGATION
                self.delta = kwargs['delta']               # TRIANGULARITY
                self.coeffs = ComputeZhengSolutionCoefficients(self.R0,self.epsilon,self.kappa,self.delta)
                # MODEL PLASMA CURRENT
                self.Jphi = self.JphiZHENG
                # MODEL ANALYTICAL SOLUTION
                self.PSIanalytical = partial(PSIanalyticalZHENG, coeffs=self.coeffs)
            
            # NONLINEAR PLASMA CURRENT
            case 'NONLINEAR':
                # MODEL PARAMETERS
                self.CURRENT_MODEL = self.NONLINEAR_CURRENT
                self.PSIdependent = True
                self.R0 = kwargs['R0']
                self.coeffs = [1.15*np.pi,  # [Kr, 
                               1.15,        #  Kz,
                               -0.5]        #  R0] 
                # MODEL PLASMA CURRENT
                self.Jphi = self.JphiNONLINEAR
                # MODEL ANALYTICAL SOLUTION
                self.PSIanalytical = partial(PSIanalyticalNONLINEAR, R0=self.R0, coeffs=self.coeffs)
        
            # PROFILES PLASMA CURRENT 
            case 'PROFILES':
                # MODEL PARAMETERS
                self.CURRENT_MODEL = self.PROFILES_CURRENT
                self.PSIdependent = True
                self.P0 = kwargs['P0']                      # PRESSURE VALUE ON MAGNETIC AXIS
                self.n_p = kwargs['np']                     # EXPONENT FOR PRESSURE PROFILE p_hat FUNCTION
                self.G0 = kwargs['G0']                      # TOROIDAL FIELD FACTOR
                self.n_g = kwargs['ng']                     # EXPONENT FOR TOROIDAL FIELD PROFILE g_hat FUNCTION
                self.TOTAL_CURRENT = kwargs['Tcurrent']     # TOTAL CURRENT IN PLASMA (NORMALISATION PARAMETER)
                # MODEL PLASMA CURRENT 
                self.Jphi = self.JphiPROFILES
                
            case 'PROFILES_PCONSTRAIN':
                # MODEL PARAMETERS
                self.CURRENT_MODEL = self.PCONSTRAIN_CURRENT
                self.PSIdependent = True
                self.P0 = kwargs['P0']                      # PRESSURE VALUE ON MAGNETIC AXIS
                self.alpha_m = kwargs['alpha_m']
                self.alpha_n = kwargs['alpha_n']
                self.TOTAL_CURRENT = kwargs['Tcurrent']     # TOTAL CURRENT IN PLASMA (NORMALISATION PARAMETER)
                if 'Raxis' in kwargs:
                    self.Raxis = kwargs['Raxis']
                else:
                    self.Raxis = 1.0
                self.Beta0 = None                           # BETA CONSTRAIN FACTOR
                self.L = None                               # L CONSTRAIN FACTOR
                # MODEL PLASMA CURRENT 
                self.Jphi = self.JphiPCONSTRAIN
                
            case 'PROFILES_BetaCONSTRAIN':
                # MODEL PARAMETERS
                self.CURRENT_MODEL = self.BetaCONSTRAIN_CURRENT
                self.PSIdependent = True
                self.Betap = kwargs['Betap']                # POLOIDAL BETA
                self.alpha_m = kwargs['alpha_m']
                self.alpha_n = kwargs['alpha_n']
                self.TOTAL_CURRENT = kwargs['Tcurrent']     # TOTAL CURRENT IN PLASMA (NORMALISATION PARAMETER)
                if 'Raxis' in kwargs:
                    self.Raxis = kwargs['Raxis']
                else:
                    self.Raxis = 1.0
                self.Beta0 = None                           # BETA CONSTRAIN FACTOR
                self.L = None                               # L CONSTRAIN FACTOR
                # MODEL PLASMA CURRENT 
                self.Jphi = self.JphiPCONSTRAIN
                
            # USER DEFINED PLASMA CURRENT
            case 'OTHER':
                self.CURRENT_MODEL = self.OTHER_CURRENT
                self.Jphi = kwargs['Jphi']
                self.PSIdependent = kwargs['PSIdependent']
        
        return
    
    
    def SourceTerm(self,X,PSI):
        return self.mu0*X[0]*self.Jphi(X,PSI)

##################################################################################################
######################################## LINEAR MODEL ############################################
##################################################################################################

    def JphiLINEAR(self,X,PSI):
        return X[0]/self.mu0
    
##################################################################################################
######################################### ZHENG MODEL ############################################
##################################################################################################
    
    def JphiZHENG(self,X,PSI):
        return (self.coeffs[4]*X[0]**2 - self.coeffs[5])/ (X[0]*self.mu0)
    
##################################################################################################
##################################### NONLINEAR MODEL ############################################
##################################################################################################
    
    def JphiNONLINEAR(self,X,PSI):
        Xstar = X/self.R0
        Kr, Kz, r0 = self.coeffs
        Jphi = -((Kr**2+Kz**2)*PSI+(Kr/Xstar[0])*np.cos(Kr*(Xstar[0]+r0))*np.cos(Kz*Xstar[1])+Xstar[0]*(np.sin(Kr*(Xstar[0]+r0))**2*np.cos(Kz*Xstar[1])**2
                    -PSI**2+np.exp(-np.sin(Kr*(Xstar[0]+r0))*np.cos(Kz*Xstar[1]))-np.exp(-PSI)))/(Xstar[0]*self.mu0)
        return Jphi
    
##################################################################################################
###################################### PROFILES MODEL ############################################
##################################################################################################    

    def JphiPROFILES(self,X,PSI):
        return -X[0] * self.dPdPSI(PSI) - 0.5*self.dG2dPSI(PSI)/ (X[0]*self.mu0)
    
    # PLASMA PRESSURE MODELING
    def dPdPSI(self,PSI):
        """
        Compute the derivative of the plasma pressure profile with respect to PSI.

        Input:
            PSI (float): Poloidal flux function value.

        Output:
            dp (float): The computed derivative of the plasma pressure profile (dP/dPSI).
        """ 
        dp = self.P0*self.n_p*(PSI**(self.n_p-1))
        return dp
    
    ######## TOROIDAL FUNCTION MODELING
    def dG2dPSI(self,PSI):
        # FUNCTION MODELING TOROIDAL FIELD FUNCTION DERIVATIVE IN PLASMA REGION
        dg = (self.G0**2)*self.n_g*(PSI**(self.n_g-1))
        return dg
    
##################################################################################################
#################################### PCONSTRAIN MODEL ############################################
##################################################################################################   
    
    def JphiPCONSTRAIN(self,X,PSI):
        return self.L * (self.Beta0 * X[0] / self.Raxis + (1 - self.Beta0) * self.Raxis / X[0]) * (1.0 - PSI**self.alpha_m)**self.alpha_n
    
        
    def ComputeConstrains(self,problem):
        
        # Apply constraints to define constants L and Beta0

        # Need integral of current shape function (1-psi**alpha_m)**alpha_n to calculate P0
        # Note factor to convert from normalised psi integral
        shapeintegral, _ = quad(
            func = lambda x: (1.0 - x**self.alpha_m) ** self.alpha_n, 
            a = 0.0, 
            b = 1.0)
        shapeintegral *= problem.PSI_X - problem.PSI_0

        # Integrate current components
        def funIR(X,PSI):
            return (1.0 - PSI**self.alpha_m)**self.alpha_n * X[0] / self.Raxis
        
        def funI_R(X,PSI):
            return (1.0 - PSI**self.alpha_m)**self.alpha_n * self.Raxis / X[0] 

        IR = problem.IntegratePlasmaDomain(funIR)
        I_R = problem.IntegratePlasmaDomain(funI_R)
        
        # Pressure on axis is
        #
        # P0 = - (L*Beta0/Raxis) * shapeintegral
        #
        # Toroidal plasma current Ip is
        #
        # TotalCurrent = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R

        LBeta0 = -self.P0 * self.Raxis / shapeintegral

        self.L = self.TOTAL_CURRENT / I_R - LBeta0 * (IR / I_R - 1)
        self.Beta0 = LBeta0 / self.L
        return



##################################################################################################
####################################### CURRENT FIELD ############################################
##################################################################################################
    
    def ComputeField(self,X,PSI):
        
        Jphifield = np.zeros([np.shape(X)[0]])
        for inode in range(np.shape(X)[0]):
            Jphifield[inode] = self.Jphi(X[inode,:],PSI[inode])
        return Jphifield
    
##################################################################################################
###################################### REPRESENTATION ############################################
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
        # COMPUTE PLASMA CURRENT FIELD
        PSI0 = self.problem.initialPSI.ComputeField(self.problem.X)
        Jphi = self.ComputeField(self.problem.X,PSI0)
        # COMPUTE PLASMA DOMAIN BOUNDARY
        PHI0 = self.problem.initialPHI.ComputeField(Xrec)
        #### FIGURE
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_aspect('equal')
        ax.set_xlim(Rmin,Rmax)
        ax.set_ylim(Zmin,Zmax)
        # PLOT INITIAL PSI GUESS BACKGROUND VALUES
        contourf = ax.tricontourf(self.problem.X[:,0],self.problem.X[:,1],Jphi,levels=30)
        contour = ax.tricontour(self.problem.X[:,0],self.problem.X[:,1],Jphi,levels=30,colors='black', linewidths=1)
        # PLOT INITIAL PLASMA BOUNDARY
        cs = ax.tricontour(Xrec[:,0],Xrec[:,1],PHI0,levels=[0],colors='red', linewidths=3)
        # Loop over paths and extract vertices
        plasmabounpath = []
        for path in cs.collections[0].get_paths():
            v = path.vertices  # shape (N, 2) array of (x, y)
            plasmabounpath.append(v)
        plasmabounpath = plasmabounpath[0]
        # APPLY MASK TO NOT PLOT OUTSIDE OF MESH
        clip_path = Path(plasmabounpath)
        patch = PathPatch(clip_path, transform=ax.transData)
        for cont in [contourf, contour]:
            for coll in cont.collections:
                coll.set_clip_path(patch)
            
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
        
    
        # PLOT COLORBAR
        plt.colorbar(contourf, ax=ax)
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        ax.set_title('Initial poloidal magnetic flux guess')
        return
    