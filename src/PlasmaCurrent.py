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



from weakref import proxy
import numpy as np
from AnalyticalSolutions import *
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.integrate import quad
import _plot as eqplot

class CurrentModel:
    
    mu0 = 12.566370E-7           # H m-1    Magnetic permeability
    
    def __init__(self,EQUILIBRIUM,MODEL,**kwargs):
        # IMPORT PROBLEM DATA
        self.eq = proxy(EQUILIBRIUM)
        # PLASMA CURRENT PREDEFINED MODELS
        self.CURRENT_MODEL = None
        self.LINEAR_CURRENT = 0
        self.ZHENG_CURRENT = 1
        self.NONLINEAR_CURRENT = 2
        self.JARDIN_CURRENT = 3
        self.PCONSTRAIN_CURRENT = 4
        self.BetaCONSTRAIN_CURRENT = 5
        self.APEC_CURRENT = 6
        self.OTHER_CURRENT = 7
        
        self.PSIdependent = False
        self.DIMENSIONLESS = False
        
        ##### PRE-DEFINED PLASMA CURRENT MODELS
        match MODEL:
            
            ######## FIXED-BOUNDARY PROBLEM ANALYTICAL PLASMA CURRENT MODELS
            
            # LINEAR PLASMA CURRENT 
            case 'LINEAR': 
                # MODEL PARAMETERS
                self.CURRENT_MODEL = self.LINEAR_CURRENT
                self.DIMENSIONLESS = True
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
                self.DIMENSIONLESS = True
                self.PSIdependent = True
                self.R0 = kwargs['R0']
                self.coeffs = [1.15*np.pi,  # [Kr, 
                               1.15,        #  Kz,
                               -0.5]        #  R0] 
                # MODEL PLASMA CURRENT
                self.Jphi = self.JphiNONLINEAR
                # MODEL ANALYTICAL SOLUTION
                self.PSIanalytical = partial(PSIanalyticalNONLINEAR, R0=self.R0, coeffs=self.coeffs)
        
        
            ######## FREE-BOUNDARY PROBLEM PROFILES PLASMA CURRENT MODELS
        
            # JARDIN BOOK PLASMA CURRENT 
            case 'JARDIN':
                # MODEL PARAMETERS
                self.CURRENT_MODEL = self.JARDIN_CURRENT
                self.PSIdependent = True
                self.P0 = kwargs['P0']                      # PRESSURE VALUE ON MAGNETIC AXIS
                self.n_p = kwargs['np']                     # EXPONENT FOR PRESSURE PROFILE p_hat FUNCTION
                self.G0 = kwargs['G0']                      # TOROIDAL FIELD FACTOR
                self.n_g = kwargs['ng']                     # EXPONENT FOR TOROIDAL FIELD PROFILE g_hat FUNCTION
                self.TOTAL_CURRENT = kwargs['Tcurrent']     # TOTAL CURRENT IN PLASMA (NORMALISATION PARAMETER)
                # MODEL PLASMA CURRENT 
                self.Jphi = self.JphiPROFILES
                self.L = self.ComputeIpconstrain()
                
            # PLASMA CURRENT WITH CONSTRAINS ON PRESSURE AND TOTAL CURRENT
            case 'PCONSTRAIN':
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
                self.L, self.Beta0 = self.ComputePConstrains()
                # MODEL PLASMA CURRENT 
                self.Jphi = self.JphiPCONSTRAIN
                
            # PLASMA CURRENT WITH CONSTRAINS ON POLOIDAL BETA AND TOTAL CURRENT
            case 'BetaCONSTRAIN':
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
                
            # PLASMA CURRENT FROM APEC PAPER
            case 'APEC':
                # MODEL PARAMETERS
                self.CURRENT_MODEL = self.APEC_CURRENT
                self.PSIdependent = True
                self.Ii = kwargs['Ii']                      # PLASMA INTERNAL INDUCTANCE
                self.Betap = kwargs['Betap']                # POLOIDAL BETA
                self.R0 = kwargs['R0']                      # MEAN RADIUS
                self.TOTAL_CURRENT = kwargs['Tcurrent']     # TOTAL CURRENT IN PLASMA (NORMALISATION PARAMETER)
                self.L = None                               # RESCALING FACTOR
                # MODEL PLASMA CURRENT 
                self.Beta = 1.0/self.Betap                  # 
                self.alpha = 1.0/self.Ii                    # 
                self.Jphi = self.JphiAPEC
                self.L = self.ComputeIpconstrain()
                
            # USER DEFINED PLASMA CURRENT
            case 'OTHER':
                self.CURRENT_MODEL = self.OTHER_CURRENT
                self.Jphi = kwargs['Jphi']
                self.PSIdependent = kwargs['PSIdependent']
        
        return
    
    
    def Normalise(self):
        match self.CURRENT_MODEL:
            case self.JARDIN_CURRENT:
                self.L = self.ComputeIpconstrain()
            case self.APEC_CURRENT:
                self.L = self.ComputeIpconstrain()
            case self.PCONSTRAIN_CURRENT:
                self.L, self.Beta0 = self.ComputePConstrains()
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
        Kr, Kz, r0 = self.coeffs
        Jphi = -((Kr**2+Kz**2)*PSI+(Kr/X[0])*np.cos(Kr*(X[0]+r0))*np.cos(Kz*X[1])+X[0]*(np.sin(Kr*(X[0]+r0))**2*np.cos(Kz*X[1])**2
                    -PSI**2+np.exp(-np.sin(Kr*(X[0]+r0))*np.cos(Kz*X[1]))-np.exp(-PSI)))/(X[0]*self.mu0)
        return Jphi
    
##################################################################################################
######################################## JARDIN MODEL ############################################
##################################################################################################    

    def JphiPROFILES(self,X,PSI):
        # REVERSE NORMALISATION
        PSIstar = 1.0 - PSI
        return self.L * (-X[0] * self.dPdPSI(PSIstar) - 0.5*self.dG2dPSI(PSIstar))/ (X[0]*self.mu0)
    
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
    
    def ComputeIpconstrain(self):
        self.L = 1.0
        Tcurrent = self.eq.IntegratePlasmaDomain(self.Jphi)     
        L = self.TOTAL_CURRENT/Tcurrent
        return L
    
##################################################################################################
#################################### CONSTRAIN MODELS ############################################
##################################################################################################   
    
    def JphiPCONSTRAIN(self,X,PSI):
        return self.L * (self.Beta0 * X[0] / self.Raxis + (1 - self.Beta0) * self.Raxis / X[0]) * (1.0 - PSI**self.alpha_m)**self.alpha_n
    
        
    def ComputePConstrains(self):
        
        # Apply constraints to define constants L and Beta0

        # Need integral of current shape function (1-psi**alpha_m)**alpha_n to calculate P0
        # Note factor to convert from normalised psi integral
        shapeintegral, _ = quad(
            func = lambda x: (1.0 - x**self.alpha_m) ** self.alpha_n, 
            a = 0.0, 
            b = 1.0)
        shapeintegral *= self.eq.PSI_X - self.eq.PSI_0

        # Integrate current components
        def funIR(X,PSI):
            return (1.0 - PSI**self.alpha_m)**self.alpha_n * X[0] / self.Raxis
        
        def funI_R(X,PSI):
            return (1.0 - PSI**self.alpha_m)**self.alpha_n * self.Raxis / X[0] 

        IR = self.eq.IntegratePlasmaDomain(funIR)
        I_R = self.eq.IntegratePlasmaDomain(funI_R)
        
        # Pressure on axis is
        #
        # P0 = - (L*Beta0/Raxis) * shapeintegral
        #
        # Toroidal plasma current Ip is
        #
        # TotalCurrent = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R

        LBeta0 = -self.P0 * self.Raxis / shapeintegral
        L = self.TOTAL_CURRENT / I_R - LBeta0 * (IR / I_R - 1)
        Beta0 = LBeta0 / L
        return L, Beta0
    
    
    def ComputeBetaConstrains(self):
        
        # Need integral of jtorshape to calculate pressure
        # Note factor to convert from normalised psi integral
        def pshape(psinorm):
            shapeintegral, _ = quad(
                lambda x: (1.0 - x**self.alpha_m) ** self.alpha_n, psinorm, 1.0
            )
            shapeintegral *= self.eq.PSI_X - self.eq.PSI_0
            return shapeintegral

        # Pressure is
        #
        # p(psinorm) = - (L*Beta0/Raxis) * pshape(psinorm)

        # Integrate over plasma
        # betap = (2mu0) * (int(p)RdRdZ)/(int(B_poloidal**2)RdRdZ)
        #       = - (2L*Beta0*mu0/Raxis) * (pfunc*RdRdZ)/((int(B_poloidal**2)RdRdZ))

        def P(X,PSI):
            return X[0]*pshape(PSI)
        
        p_int = self.eq.IntegratePlasmaDomain(P)
        b_int = self.eq.IntegrateBpolPlasmaDomain()

        # self.betap = - (2*LBeta0*mu0/ self.Raxis) * (p_int/b_int)
        LBeta0 = (b_int / p_int) * (-self.betap * self.Raxis) / (2 * self.mu0)

        # Integrate current components
        def funIR(X,PSI):
            return (1.0 - PSI**self.alpha_m)**self.alpha_n * X[0] / self.Raxis
        
        def funI_R(X,PSI):
            return (1.0 - PSI**self.alpha_m)**self.alpha_n * self.Raxis / X[0] 

        IR = self.eq.IntegratePlasmaDomain(funIR)
        I_R = self.eq.IntegratePlasmaDomain(funI_R)

        # Toroidal plasma current Ip is
        #
        # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R
        #
        # L = self.Ip / ((Beta0*IR) + ((1.0-Beta0)*(I_R)))

        L = self.Ip / I_R - LBeta0 * (IR / I_R - 1)
        Beta0 = LBeta0 / L

        return L, Beta0
        
        
##################################################################################################
########################################## APEC MODEL ############################################
##################################################################################################

    def JphiAPEC(self,X,PSI):
        return self.L*(self.Beta*X[0]/self.R0 + (1-self.Beta)*self.R0/X[0])*(1-abs(PSI)**self.alpha)
    

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
        # COMPUTE PLASMA CURRENT FIELD
        Jphi = self.ComputeField(self.eq.MESH.X,self.eq.initialPSI.PSI0)
        
        #### FIGURE
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_aspect('equal')
        ax.set_xlim(self.eq.MESH.Rmin-eqplot.padx,self.eq.MESH.Rmax+eqplot.padx)
        ax.set_ylim(self.eq.MESH.Zmin-eqplot.pady,self.eq.MESH.Zmax+eqplot.pady)
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        ax.set_title('Plasma current')
        
        # Plot low-opacity background (outside plasma region)
        contourf_bg = ax.tricontourf(self.eq.MESH.X[:,0], self.eq.MESH.X[:,1], Jphi, levels=30, alpha=0.8)
        
        patch = PathPatch(self.eq.MESH.boundary_path, transform=ax.transData)
        for coll in contourf_bg.collections:
            coll.set_clip_path(patch)
        
        # PLOT INITIAL Jphi GUESS BACKGROUND VALUES
        contourf = ax.tricontourf(self.eq.MESH.X[:,0],self.eq.MESH.X[:,1],Jphi,levels=30, cmap = eqplot.currentcmap)
        contour = ax.tricontour(self.eq.MESH.X[:,0],self.eq.MESH.X[:,1],Jphi,levels=30,colors='black', linewidths=1)
        contour0 = ax.tricontour(self.eq.MESH.X[:,0],self.eq.MESH.X[:,1],Jphi,levels=[0],colors='black', linewidths=3)
        
        # PLOT INITIAL PLASMA BOUNDARY
        cs = ax.tricontour(self.eq.initialPHI.Xrec[:,0],self.eq.initialPHI.Xrec[:,1],self.eq.initialPHI.PHI0rec,levels=[0],
                           colors = eqplot.plasmabouncolor, 
                           linewidths = eqplot.plasmabounlinewidth)
        # Loop over paths and extract vertices
        plasmabounpath = []
        for path in cs.collections[0].get_paths():
            v = path.vertices  # shape (N, 2) array of (x, y)
            plasmabounpath.append(v)
        plasmabounpath = plasmabounpath[0]
        # APPLY MASK TO NOT PLOT OUTSIDE OF PLASMA REGION
        clip_path = Path(plasmabounpath)
        patch = PathPatch(clip_path, transform=ax.transData)
        for cont in [contourf, contour, contour0]:
            for coll in cont.collections:
                coll.set_clip_path(patch)
            
        # PLOT MESH BOUNDARY
        self.eq.MESH.PlotBoundary(ax = ax)
        # PLOT TOKAMAK FIRST WALL
        self.eq.TOKAMAK.PlotFirstWall(ax = ax)
        
        # PLOT COLORBAR
        plt.colorbar(contourf, ax=ax)
        return
    