import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from Element import ReferenceElementCoordinates
from itertools import chain


##################################################################################################
############################### RENDERING AND REPRESENTATION #####################################
##################################################################################################

dzoom = 0.1
Vermillion = '#D55E00'
Blue = '#0072B2'
BluishGreen = '#009E73'
Orange = '#E69F00'
SkyBlue = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow = '#F0E442'
Black = '#000000'
Grey = '#BBBBBB'
White = '#FFFFFF'
colorlist = [Blue, Vermillion, BluishGreen, Black,Grey, Orange, ReddishPurple, Yellow, SkyBlue]
markerlist = ['o','^', '<', '>', 'v', 's','p','*','D']

meshcolor = Black
nodescolor = Blue
nodesize = 1
compbouncolor = Orange
compbounlinewidth = 5
meshlinewidth = 0.5

plasmacmap = plt.get_cmap('jet_r')
Npsilevels = 30
plasmabouncolor = 'green'
plasmabounlinewidth = 3
firstwallcolor = 'gray'
firstwalllinewidth = 5
magneticaxiscolor = 'red'
saddlepointcolor = BluishGreen
magnetcolor = SkyBlue
padx = 0.1
pady = 0.1

Nphilevels = 30
phicmap = 'viridis'

currentcmap = 'viridis'

ghostfacescolor = Blue

class EquilipyPlotting:
    
    def PlotLayout(self,plotmesh=True):
        
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_aspect('equal')
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        ax.set_title('Simulation layout')
        
        # PLOT COMPUTATIONAL DOMAIN
        if plotmesh:
            self.MESH.Plot(ax = ax)
        else:
            self.MESH.PlotBoundary(ax = ax)
        # PLOT TOKAMAK
        self.TOKAMAK.PlotFirstWall(ax = ax)
        self.TOKAMAK.PlotMagnets(ax = ax)
        plt.show()
        return
    
    
    def PlotField(self,FIELD):
        
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        ax.set_xlim(self.MESH.Rmin-padx,self.MESH.Rmax+padx)
        ax.set_ylim(self.MESH.Zmin-pady,self.MESH.Zmax+pady)
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        
        contourf = ax.tricontourf(self.MESH.X[:,0],self.MESH.X[:,1], FIELD, levels=Npsilevels)
        contour1 = ax.tricontour(self.MESH.X[:,0],self.MESH.X[:,1], FIELD, levels=[0], colors = 'black')
        contour2 = ax.tricontour(self.MESH.X[:,0],self.MESH.X[:,1], self.PlasmaLS[:,1], levels=[0], 
                      colors = plasmabouncolor,
                      linewidths = plasmabounlinewidth)
        
        # PATH FIELD OUTSIDE COMPUTATIONAL DOMAIN'S BOUNDARY 
        patch = PathPatch(self.MESH.boundary_path, transform=ax.transData)
        for cont in [contourf,contour1,contour2]:
            for coll in cont.collections:
                coll.set_clip_path(patch)
                
        # PLOT COMPUTATIONAL DOMAIN
        self.MESH.Plot(ax = ax)
        # PLOT TOKAMAK
        self.TOKAMAK.PlotFirstWall(ax = ax)
                
        plt.colorbar(contourf, ax=ax)
        plt.show()
        return
    

    def PlotPSI(self):
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_aspect('equal')
        ax.set_xlim(self.MESH.Rmin-padx,self.MESH.Rmax+padx)
        ax.set_ylim(self.MESH.Zmin-pady,self.MESH.Zmax+pady)
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        ax.set_title('PSI')
        
        contourf = ax.tricontourf(self.MESH.X[:,0],self.MESH.X[:,1], self.PSI[:,0], levels = Npsilevels, cmap = plasmacmap)
        contour1 = ax.tricontour(self.MESH.X[:,0],self.MESH.X[:,1], self.PSI[:,0], levels=[self.PSI_X], colors = 'black')
        contour2 = ax.tricontour(self.MESH.X[:,0],self.MESH.X[:,1], self.PlasmaLS[:,1], levels=[0], 
                                 colors = plasmabouncolor,
                                 linewidths = plasmabounlinewidth)
        
        # PATH FIELD OUTSIDE COMPUTATIONAL DOMAIN'S BOUNDARY 
        patch = PathPatch(self.MESH.boundary_path, transform=ax.transData)
        for cont in [contourf,contour1,contour2]:
            for coll in cont.collections:
                coll.set_clip_path(patch)
        
        # PLOT MESH BOUNDARY
        self.MESH.PlotBoundary(ax = ax)
        # PLOT TOKAMAK FIRST WALL
        self.TOKAMAK.PlotFirstWall(ax = ax)
                
        plt.colorbar(contourf, ax=ax)
        plt.show()
        return


    

    def PlotError(self,RelativeError = False):
        if self.FIXED_BOUNDARY:
            AnaliticalNorm = np.zeros([self.MESH.Nn])
            for inode in range(self.MESH.Nn):
                AnaliticalNorm[inode] = self.PlasmaCurrent.PSIanalytical(self.MESH.X[inode,:])
                
            print('||PSIerror||_L2 = ', self.ErrorL2norm)
            print('relative ||PSIerror||_L2 = ', self.RelErrorL2norm)
            print('||PSIerror|| = ',np.linalg.norm(self.PSIerror))
            print('||PSIerror||/node = ',np.linalg.norm(self.PSIerror)/self.MESH.Nn)
            print('relative ||PSIerror|| = ',np.linalg.norm(self.PSIrelerror))
                
            # Compute global min and max across both datasets
            vmin = min(AnaliticalNorm)
            vmax = max(AnaliticalNorm)  
                
            fig, axs = plt.subplots(1, 4, figsize=(16,5),gridspec_kw={'width_ratios': [1,1,0.25,1]})
            
            # LEFT PLOT: ANALYTICAL SOLUTION
            axs[0].set_xlim(self.MESH.Rmin-padx,self.MESH.Rmax+padx)
            axs[0].set_ylim(self.MESH.Zmin-pady,self.MESH.Zmax+pady)
            axs[0].set_xlabel('R (in m)')
            axs[0].set_ylabel('Z (in m)')
            axs[0].set_title('PSI exact')
            a1 = axs[0].tricontourf(self.MESH.X[:,0],self.MESH.X[:,1], AnaliticalNorm, levels=Npsilevels, cmap=plasmacmap, vmin=vmin, vmax=vmax)
            axs[0].tricontour(self.MESH.X[:,0],self.MESH.X[:,1], self.PlasmaLS[:,1], levels=[0], 
                              colors = plasmabouncolor, 
                              linewidths=plasmabounlinewidth)
            axs[0].tricontour(self.MESH.X[:,0],self.MESH.X[:,1], AnaliticalNorm, levels=[0], colors = 'black')
            self.MESH.PlotBoundary(ax = axs[0])

            # CENTRAL PLOT: NUMERICAL SOLUTION
            axs[1].set_xlim(self.MESH.Rmin-padx,self.MESH.Rmax+padx)
            axs[1].set_ylim(self.MESH.Zmin-pady,self.MESH.Zmax+pady)
            axs[1].set_xlabel('R (in m)')
            axs[1].set_title('PSI numeric')
            axs[1].tricontourf(self.MESH.X[:,0],self.MESH.X[:,1], self.PSI_CONV, levels=Npsilevels, cmap=plasmacmap, vmin=vmin, vmax=vmax)
            axs[1].tricontour(self.MESH.X[:,0],self.MESH.X[:,1], self.PlasmaLS[:,1], levels=[0], 
                              colors = plasmabouncolor, 
                              linewidths=plasmabounlinewidth)
            axs[1].tricontour(self.MESH.X[:,0],self.MESH.X[:,1], self.PSI_CONV, levels=[0], colors = 'black')
            self.MESH.PlotBoundary(ax = axs[1])
            # COLORBAR
            fig.colorbar(a1, ax=axs[2], orientation="vertical", fraction=0.8, pad=-0.7)
            
            # RIGHT PLOT: ERROR
            axs[3].axis('off')
            axs[3].set_xlim(self.MESH.Rmin-padx,self.MESH.Rmax+padx)
            axs[3].set_ylim(self.MESH.Zmin-pady,self.MESH.Zmax+pady)
            axs[3].set_xlabel('R (in m)')
            if RelativeError:
                errorfield = self.PSIrelerror
                axs[3].set_title('PSI relative error')
            else:
                errorfield = self.PSIerror
                axs[3].set_title('PSI error')
            vmax = max(np.log(errorfield))
            a = axs[3].tricontourf(self.MESH.X[:,0],self.MESH.X[:,1], np.log(errorfield), levels=Npsilevels, vmax=vmax,vmin=-20)
            self.MESH.PlotBoundary(ax = axs[3])
            
            plt.colorbar(a, ax=axs[3])
            plt.show()
        return


    def PlotSolutionPSI(self):
        """ FUNCTION WHICH PLOTS THE FIELD VALUES FOR PSI, OBTAINED FROM SOLVING THE CUTFEM SYSTEM, 
        AND PSI_NORM IF NORMALISED. """
        
        def subplotfield(self,ax,field,normalised=True):
            ax.set_xlim(self.MESH.Rmin-padx,self.MESH.Rmax+padx)
            ax.set_ylim(self.MESH.Zmin-pady,self.MESH.Zmax+pady)
            ax.set_xlabel('R (in m)')
            ax.set_ylabel('Z (in m)')
            
            # PLOT PSI FIELD
            if normalised:
                psisep = self.PSI_NORMseparatrix
            else:
                psisep = self.PSI_X
            contourf = ax.tricontourf(self.MESH.X[:,0],self.MESH.X[:,1], field, levels=Npsilevels, cmap=plasmacmap)
            contour1 = ax.tricontour(self.MESH.X[:,0],self.MESH.X[:,1], field, levels=[psisep], colors = 'black',linewidths=2)
            contour2 = ax.tricontour(self.MESH.X[:,0],self.MESH.X[:,1], self.PlasmaLS[:,1], levels=[0], 
                                     colors = plasmabouncolor, 
                                     linewidths=plasmabounlinewidth)
            
            # Mask solution outside computational domain's boundary 
            patch = PathPatch(self.MESH.boundary_path, transform=ax.transData)
            for cont in [contourf,contour1,contour2]:
                for coll in cont.collections:
                    coll.set_clip_path(patch)
                    
            # PLOT MESH BOUNDARY
            self.MESH.PlotBoundary(ax = ax)
            # PLOT TOKAMAK FIRST WALL
            self.TOKAMAK.PlotFirstWall(ax = ax)
        
            plt.colorbar(contourf, ax=ax)
            return
        
        if not self.FIXED_BOUNDARY:
            psi_sol = " normalised solution PSI_NORM"
        else:
            psi_sol = " solution PSI"
        
        if self.it == 0:  # INITIAL GUESS PLOT
            fig, axs = plt.subplots(1, 1, figsize=(5,6))
            axs.set_aspect('equal')
            subplotfield(self,axs,self.PSI_NORM[:,0])
            axs.set_title('Initial PSI guess')
            plt.show(block=False)
            plt.pause(0.8)
            
        elif self.ext_cvg:  # CONVERGED SOLUTION PLOT
            fig, axs = plt.subplots(1, 1, figsize=(5,6))
            axs.set_aspect('equal')
            subplotfield(self,axs,self.PSI_CONV)
            axs.set_title('Converged'+psi_sol)
            plt.show()
            
        elif not self.FIXED_BOUNDARY:  # ITERATION SOLUTION FOR JARDIN PLASMA CURRENT (PLOT PSI and PSI_NORM)
            if not self.PSIrelax:
                fig, axs = plt.subplots(1, 2, figsize=(11,6))
                axs[0].set_aspect('equal')
                axs[1].set_aspect('equal')
                # LEFT PLOT: PSI at iteration N+1 WITHOUT NORMALISATION (SOLUTION OBTAINED BY SOLVING CUTFEM SYSTEM)
                subplotfield(self,axs[0],self.PSI[:,0],normalised=False)
                axs[0].set_title('PSI')
                # RIGHT PLOT: NORMALISED PSI at iteration N+1
                subplotfield(self,axs[1],self.PSI_NORM[:,1])
                axs[1].set_title('PSI_NORM')
                axs[1].yaxis.set_visible(False)
                ## PLOT LOCATION OF CRITICAL POINTS
                for i in range(2):
                    # LOCAL EXTREMUM
                    axs[i].scatter(self.Xcrit[1,0,0],self.Xcrit[1,0,1],marker = 'X',facecolor=magneticaxiscolor, edgecolor=Black, s = 100, linewidths = 1.5,zorder=5)
                    # SADDLE POINT
                    axs[i].scatter(self.Xcrit[1,1,0],self.Xcrit[1,1,1],marker = 'X',facecolor=saddlepointcolor, edgecolor=Black, s = 100, linewidths = 1.5,zorder=5)
                plt.suptitle("Iteration n = "+str(self.it))
                plt.show(block=False)
                plt.pause(0.8)
            else:
                fig, axs = plt.subplots(1, 3, figsize=(15,6))
                axs[0].set_aspect('equal')
                axs[1].set_aspect('equal')
                axs[2].set_aspect('equal')
                # LEFT PLOT: PSI at iteration N+1 WITHOUT NORMALISATION (SOLUTION OBTAINED BY SOLVING CUTFEM SYSTEM)
                subplotfield(self,axs[0],self.PSI[:,0],normalised=False)
                axs[0].set_title('PSI')
                # CENTER PLOT: NORMALISED PSI at iteration N+1
                subplotfield(self,axs[1],self.PSI_NORMstar[:,1])
                axs[1].set_title('PSI_NORMstar')
                axs[1].yaxis.set_visible(False)
                # RIGHT PLOT: RELAXED SOLUTION
                subplotfield(self,axs[2],self.PSI_NORMstar[:,1])
                axs[2].set_title('PSI_NORM')
                axs[2].yaxis.set_visible(False)
                
                ## PLOT LOCATION OF CRITICAL POINTS
                for i in range(2):
                    # LOCAL EXTREMUM
                    axs[i].scatter(self.Xcrit[1,0,0],self.Xcrit[1,0,1],marker = 'X',facecolor=magneticaxiscolor, edgecolor=Black, s = 100, linewidths = 1.5,zorder=5)
                    # SADDLE POINT
                    axs[i].scatter(self.Xcrit[1,1,0],self.Xcrit[1,1,1],marker = 'X',facecolor=saddlepointcolor, edgecolor=Black, s = 100, linewidths = 1.5,zorder=5)
                plt.suptitle("Iteration n = "+str(self.it))
                plt.show(block=False)
                plt.pause(0.8)
                
        else:  # ITERATION SOLUTION FOR ANALYTICAL PLASMA CURRENT CASES (PLOT PSI)
            fig, axs = plt.subplots(1, 1, figsize=(5,6))
            axs.set_aspect('equal')
            subplotfield(self,axs,self.PSI[:,0],normalised=False)
            axs.set_title('Poloidal magnetic flux PSI')
            axs.set_title("Iteration n = "+str(self.it)+ psi_sol)
            plt.show(block=False)
            plt.pause(0.8)
            
        return



    def PlotPlasmaBoundContrainedEdges(self):
        
        #### FIGURE
        # PLOT PHI LEVEL-SET BACKGROUND VALUES 
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_aspect('equal')
        ax.set_xlim(self.MESH.Rmin-padx,self.MESH.Rmax+padx)
        ax.set_ylim(self.MESH.Zmin-pady,self.MESH.Zmax+pady) 
        ax.set_xlabel('R (in m)')
        ax.set_ylabel('Z (in m)')
        # Plot low-opacity background (outside plasma region)
        ax.tricontourf(self.MESH.X[:,0],self.MESH.X[:,1],self.PlasmaLS[:,1],levels=Nphilevels)
        # PLOT PLASMA BOUNDARY
        ax.tricontour(self.MESH.X[:,0],self.MESH.X[:,1],self.PlasmaLS[:,1],levels=[0],
                      colors=plasmabouncolor, 
                      linewidths=plasmabounlinewidth)
        
        # PLOT CONSTRAINED EDGES
        for ielem in self.MESH.PlasmaBoundActiveElems:
            ax.plot([self.MESH.Elements[ielem].InterfApprox.Xint[1,0],self.MESH.Elements[ielem].InterfApprox.Xint[0,0]],
                    [self.MESH.Elements[ielem].InterfApprox.Xint[1,1],self.MESH.Elements[ielem].InterfApprox.Xint[0,1]],'-o',
                    color = ReddishPurple,
                    linewidth = 2)

        # PLOT MESH BOUNDARY
        self.MESH.PlotBoundary(ax = ax)
        # PLOT TOKAMAK FIRST WALL
        self.TOKAMAK.PlotFirstWall(ax = ax)
        return




    def PlotMagneticField(self):
        # COMPUTE MAGNETIC FIELD NORM
        Bnorm = np.zeros([self.MESH.Ne*self.nge])
        for inode in range(self.MESH.Ne*self.nge):
            Bnorm[inode] = np.linalg.norm(self.Brzfield[inode,:])
            
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        ax.set_xlim(self.MESH.Rmin,self.MESH.Rmax)
        ax.set_ylim(self.MESH.Zmin,self.MESH.Zmax)
        a = ax.tricontourf(self.Xg[:,0],self.Xg[:,1], Bnorm, levels=30)
        ax.tricontour(self.MESH.X[:,0],self.MESH.X[:,1], self.PlasmaLS, levels=[0], colors = 'red')
        plt.colorbar(a, ax=ax)
        plt.show()
        
        
        """
        if streamplot:
            R, Z, Br, Bz = self.ComputeMagnetsBfield(regular_grid=True)
            # Poloidal field magnitude
            Bp = np.sqrt(Br**2 + Br**2)
            plt.contourf(R, Z, np.log(Bp), 50)
            plt.streamplot(R, Z, Br, Bz)
            plt.show()
        """
        
        return


    def PlotNormalVectors(self):
        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        axs[0].set_xlim(self.MESH.Rmin-0.5,self.MESH.Rmax+0.5)
        axs[0].set_ylim(self.MESH.Zmin-0.5,self.MESH.Zmax+0.5)
        axs[1].set_xlim(6.5,7)
        if self.FIXED_BOUNDARY:
            axs[1].set_ylim(1.6,2)
        else:
            axs[1].set_ylim(2.2,2.6)

        for i in range(2):
            # PLOT PLASMA/VACUUM INTERFACE
            axs[i].tricontour(self.MESH.X[:,0],self.MESH.X[:,1], self.PlasmaLS, levels=[0], colors='green',linewidths=6)
            # PLOT NORMAL VECTORS
            for ielem in self.MESH.PlasmaBoundElems:
                ELEMENT = self.MESH.Elements[ielem]
                if i == 0:
                    dl = 5
                else:
                    dl = 10
                for j in range(ELEMENT.n):
                    plt.plot([ELEMENT.Xe[j,0], ELEMENT.Xe[int((j+1)%ELEMENT.n),0]], 
                            [ELEMENT.Xe[j,1], ELEMENT.Xe[int((j+1)%ELEMENT.n),1]], color='k', linewidth=1)
                INTAPPROX = ELEMENT.InterfApprox
                # PLOT INTERFACE APPROXIMATIONS
                for inode in range(INTAPPROX.n-1):
                    axs[0].plot(INTAPPROX.Xint[INTAPPROX.Tint[inode:inode+1],0],INTAPPROX.Xint[INTAPPROX.Tint[inode:inode+1],1], linestyle='-', color = 'red', linewidth = 2)
                    axs[1].plot(INTAPPROX.Xint[INTAPPROX.Tint[inode:inode+1],0],INTAPPROX.Xint[INTAPPROX.Tint[inode:inode+1],1], linestyle='-', marker='o',color = 'red', linewidth = 2)
                # PLOT NORMAL VECTORS
                for ig, vec in enumerate(INTAPPROX.NormalVec):
                    axs[i].arrow(INTAPPROX.Xg[ig,0],INTAPPROX.Xg[ig,1],vec[0]/dl,vec[1]/dl,width=0.005)
                
        axs[1].set_aspect('equal')
        plt.show()
        return







####################################################   

    def InspectElement(self,element_index,BOUNDARY,PSI,TESSELLATION,GHOSTFACES,NORMALS,QUADRATURE):
        ELEMENT = self.MESH.Elements[element_index]
        Xmin = np.min(ELEMENT.Xe[:,0])-self.MESH.meanLength/4
        Xmax = np.max(ELEMENT.Xe[:,0])+self.MESH.meanLength/4
        Ymin = np.min(ELEMENT.Xe[:,1])-self.MESH.meanLength/4
        Ymax = np.max(ELEMENT.Xe[:,1])+self.MESH.meanLength/4
            
        color = self.ElementColor(ELEMENT.Dom)
        colorlist = ['#009E73','#D55E00','#CC79A7','#56B4E9']

        fig, axs = plt.subplots(1, 2, figsize=(10,6))
        axs[0].set_xlim(self.MESH.Rmin-0.25,self.MESH.Rmax+0.25)
        axs[0].set_ylim(self.MESH.Zmin-0.25,self.MESH.Zmax+0.25)
        if PSI:
            axs[0].tricontourf(self.MESH.X[:,0],self.MESH.X[:,1], self.PSI_NORM[:,1], levels=30, cmap='plasma')
            axs[0].tricontour(self.MESH.X[:,0],self.MESH.X[:,1], self.PSI_NORM[:,1], levels=[0], colors = 'black')
        axs[0].tricontour(self.MESH.X[:,0],self.MESH.X[:,1], self.PlasmaLS, levels=[0], colors = 'red')
        # PLOT ELEMENT EDGES
        for iedge in range(ELEMENT.numedges):
            axs[0].plot([ELEMENT.Xe[iedge,0],ELEMENT.Xe[int((iedge+1)%ELEMENT.numedges),0]],[ELEMENT.Xe[iedge,1],ELEMENT.Xe[int((iedge+1)%ELEMENT.numedges),1]], color=color, linewidth=3)

        axs[1].set_xlim(Xmin,Xmax)
        axs[1].set_ylim(Ymin,Ymax)
        axs[1].set_aspect('equal')
        axs[1].tricontour(self.MESH.X[:,0],self.MESH.X[:,1], self.PlasmaLS, levels=[0], colors = 'red',linewidths=2)
        # PLOT ELEMENT EDGES
        for iedge in range(ELEMENT.numedges):
            axs[1].plot([ELEMENT.Xe[iedge,0],ELEMENT.Xe[int((iedge+1)%ELEMENT.numedges),0]],[ELEMENT.Xe[iedge,1],ELEMENT.Xe[int((iedge+1)%ELEMENT.numedges),1]], color=color, linewidth=8)
        for inode in range(ELEMENT.n):
            if ELEMENT.LSe[inode] < 0:
                cl = 'blue'
            else:
                cl = 'red'
            axs[1].scatter(ELEMENT.Xe[inode,0],ELEMENT.Xe[inode,1],s=120,color=cl,zorder=5)
        if TESSELLATION and (ELEMENT.Dom == 0 or ELEMENT.Dom == 2):
            for isub, SUBELEM in enumerate(ELEMENT.SubElements):
                # PLOT SUBELEMENT EDGES
                for i in range(SUBELEM.numedges):
                    axs[1].plot([SUBELEM.Xe[i,0], SUBELEM.Xe[(i+1)%SUBELEM.numedges,0]], [SUBELEM.Xe[i,1], SUBELEM.Xe[(i+1)%SUBELEM.numedges,1]], color=colorlist[isub], linewidth=3.5)
                axs[1].scatter(SUBELEM.Xe[:,0],SUBELEM.Xe[:,1], marker='o', s=60, color=colorlist[isub], zorder=5)
        if BOUNDARY:
            axs[1].scatter(ELEMENT.InterfApprox.Xint[:,0],ELEMENT.InterfApprox.Xint[:,1],marker='o',color='red',s=100, zorder=5)
        if GHOSTFACES:
            for FACE in ELEMENT.GhostFaces:
                axs[1].plot(FACE.Xseg[:2,0],FACE.Xseg[:2,1],color=colorlist[-1],linestyle='dashed',linewidth=3)
        if NORMALS:
            if BOUNDARY:
                for ig, vec in enumerate(ELEMENT.InterfApprox.NormalVec):
                    # PLOT NORMAL VECTORS
                    dl = 20
                    axs[1].arrow(ELEMENT.InterfApprox.Xg[ig,0],ELEMENT.InterfApprox.Xg[ig,1],vec[0]/dl,vec[1]/dl,width=0.005)
            if GHOSTFACES:
                for FACE in ELEMENT.GhostFaces:
                    # PLOT NORMAL VECTORS
                    Xsegmean = np.mean(FACE.Xseg, axis=0)
                    dl = 40
                    axs[1].arrow(Xsegmean[0],Xsegmean[1],FACE.NormalVec[0]/dl,FACE.NormalVec[1]/dl,width=0.005)
        if QUADRATURE:
            if ELEMENT.Dom == -1 or ELEMENT.Dom == 1 or ELEMENT.Dom == 2:
                # PLOT STANDARD QUADRATURE INTEGRATION POINTS
                axs[1].scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='black')
            else:
                if TESSELLATION:
                    for isub, SUBELEM in enumerate(ELEMENT.SubElements):
                        # PLOT QUADRATURE SUBELEMENTAL INTEGRATION POINTS
                        axs[1].scatter(SUBELEM.Xg[:,0],SUBELEM.Xg[:,1],marker='x',c=colorlist[isub], zorder=3)
                if BOUNDARY:
                    # PLOT PLASMA BOUNDARY INTEGRATION POINTS
                    axs[1].scatter(ELEMENT.InterfApprox.Xg[:,0],ELEMENT.InterfApprox.Xg[:,1],marker='x',color='grey',s=50, zorder=5)
                if GHOSTFACES:
                    # PLOT CUT EDGES QUADRATURES 
                    for FACE in ELEMENT.GhostFaces:
                        axs[1].scatter(FACE.Xg[:,0],FACE.Xg[:,1],marker='x',color='k',s=50, zorder=6)
        return


    def InspectGhostFace(self,BOUNDARY,index):
        if BOUNDARY == self.PLASMAbound:
            ghostface = self.MESH.GhostFaces[index]
        elif BOUNDARY == self.VACVESbound:
            ghostface == self.VacVessWallGhostFaces[index]

        # ISOLATE ELEMENTS
        ELEMS = [self.MESH.Elements[ghostface[1][0]],self.MESH.Elements[ghostface[2][0]]]
        FACES = [ELEMS[0].CutEdges[ghostface[1][1]],ELEMS[1].CutEdges[ghostface[2][1]]]
        
        color = self.ElementColor(ELEMS[0].Dom)
        colorlist = ['#009E73','#D55E00','#CC79A7','#56B4E9']
        
        Rmin = min((min(ELEMS[0].Xe[:,0]),min(ELEMS[1].Xe[:,0])))
        Rmax = max((max(ELEMS[0].Xe[:,0]),max(ELEMS[1].Xe[:,0])))
        Zmin = min((min(ELEMS[0].Xe[:,1]),min(ELEMS[1].Xe[:,1])))
        Zmax = max((max(ELEMS[0].Xe[:,1]),max(ELEMS[1].Xe[:,1])))
        
        fig, axs = plt.subplots(1, 1, figsize=(6,6))
        axs.set_xlim(Rmin,Rmax)
        axs.set_ylim(Zmin,Zmax)
        # PLOT ELEMENTAL EDGES:
        for ELEMENT in ELEMS:
            for iedge in range(ELEMENT.numedges):
                axs[1].plot([ELEMENT.Xe[iedge,0],ELEMENT.Xe[int((iedge+1)%ELEMENT.numedges),0]],[ELEMENT.Xe[iedge,1],ELEMENT.Xe[int((iedge+1)%ELEMENT.numedges),1]], color=color, linewidth=8)
            for inode in range(ELEMENT.n):
                if ELEMENT.LSe[inode] < 0:
                    cl = 'blue'
                else:
                    cl = 'red'
                axs[1].scatter(ELEMENT.Xe[inode,0],ELEMENT.Xe[inode,1],s=120,color=cl,zorder=5)
                
        # PLOT CUT EDGES
        for iedge, FACE in enumerate(FACES):
            axs.plot(FACE.Xseg[:2,0],FACE.Xseg[:2,1],color='#D55E00',linestyle='dashed',linewidth=3)
            
        for inode in range(FACES[0].n):
            axs.text(FACES[0].Xseg[inode,0]+0.03,FACES[0].Xseg[inode,1],str(inode),fontsize=12, color=colorlist[0])
        for inode in range(FACES[1].n):
            axs.text(FACES[1].Xseg[inode,0]-0.03,FACES[1].Xseg[inode,1],str(inode),fontsize=12, color=colorlist[1])
            
        for iedge, FACE in enumerate(FACES):
            # PLOT NORMAL VECTORS
            Xsegmean = np.mean(FACE.Xseg, axis=0)
            dl = 10
            axs.arrow(Xsegmean[0],Xsegmean[1],FACE.NormalVec[0]/dl,FACE.NormalVec[1]/dl,width=0.01, color=colorlist[iedge])
            # PLOT CUT EDGES QUADRATURES 
            for FACE in FACES:
                axs.scatter(FACE.Xg[:,0],FACE.Xg[:,1],marker='x',color='k',s=80, zorder=6)
                
        for inode in range(FACES[0].ng):
            axs.text(FACES[0].Xg[inode,0]+0.03,FACES[0].Xg[inode,1],str(inode),fontsize=12, color=colorlist[0])
        for inode in range(FACES[1].ng):
            axs.text(FACES[1].Xg[inode,0]-0.03,FACES[1].Xg[inode,1],str(inode),fontsize=12, color=colorlist[1])
            
        return
    

    def PlotInterfaceValues(self):
        """ Function which plots the values PSIgseg at the interface edges, for both the plasma/vacuum interface and the vacuum vessel first wall. """

        # IMPOSED BOUNDARY VALUES
        ### VACUUM VESSEL FIRST WALL
        PSI_Bg = self.PSI_B[:,1]
        PSI_B = self.PSI[self.MESH.BoundaryNodes]
            
        ### PLASMA BOUNDARY
        X_Dg = np.zeros([self.MESH.NnPB,self.MESH.dim])
        PSI_Dg = np.zeros([self.MESH.NnPB])
        PSI_D = np.zeros([self.MESH.NnPB])
        k = 0
        for ielem in self.MESH.PlasmaBoundElems:
            INTAPPROX = self.MESH.Elements[ielem].InterfApprox
            for inode in range(INTAPPROX.ng):
                X_Dg[k,:] = INTAPPROX.Xg[inode,:]
                PSI_Dg[k] = INTAPPROX.PSIg[inode]
                PSI_D[k] = self.MESH.Elements[ielem].ElementalInterpolationPHYSICAL(X_Dg[k,:],self.PSI[self.MESH.Elements[ielem].Te])
                k += 1
            
        fig, axs = plt.subplots(1, 2, figsize=(14,7))
        ### UPPER ROW SUBPLOTS 
        # LEFT SUBPLOT: CONSTRAINT VALUES ON PSI
        axs[0].set_aspect('equal')
        axs[0].set_ylim(self.MESH.Zmin-0.5,self.MESH.Zmax+0.5)
        axs[0].set_xlim(self.MESH.Rmin-0.5,self.MESH.Rmax+0.5)
        cmap = plt.get_cmap('jet')
        
        norm = plt.Normalize(np.min([PSI_Bg.min(),PSI_Dg.min()]),np.max([PSI_Bg.max(),PSI_Dg.max()]))
        linecolors_Dg = cmap(norm(PSI_Dg))
        linecolors_Bg = cmap(norm(PSI_Bg))
        axs[0].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_Dg)
        axs[0].scatter(self.MESH.X[self.MESH.BoundaryNodes,0],self.MESH.X[self.MESH.BoundaryNodes,1],color = linecolors_Bg)

        # RIGHT SUBPLOT: RESULTING VALUES ON CUTFEM SYSTEM 
        axs[1].set_aspect('equal')
        axs[1].set_ylim(self.MESH.Zmin-0.5,self.MESH.Zmax+0.5)
        axs[1].set_xlim(self.MESH.Rmin-0.5,self.MESH.Rmax+0.5)
        linecolors_D = cmap(norm(PSI_D))
        linecolors_B = cmap(norm(PSI_B))
        axs[1].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_D)
        axs[1].scatter(self.MESH.X[self.MESH.BoundaryNodes,0],self.MESH.X[self.MESH.BoundaryNodes,1],color = linecolors_B)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[1])

        plt.show()
        return


    def PlotPlasmaBoundaryConstraints(self):
        
        # COLLECT PSIgseg DATA ON PLASMA/VACUUM INTERFACE
        X_Dg = np.zeros([len(self.MESH.PlasmaBoundElems)*self.Ng1D,self.MESH.dim])
        PSI_Dexact = np.zeros([len(self.MESH.PlasmaBoundElems)*self.Ng1D])
        PSI_Dg = np.zeros([len(self.MESH.PlasmaBoundElems)*self.Ng1D])
        X_D = np.zeros([len(self.MESH.PlasmaBoundElems)*self.MESH.n,self.MESH.dim])
        PSI_D = np.zeros([len(self.MESH.PlasmaBoundElems)*self.MESH.n])
        error = np.zeros([len(self.MESH.PlasmaBoundElems)*self.MESH.n])
        k = 0
        l = 0
        for ielem in self.MESH.PlasmaBoundElems:
            for SEGMENT in self.MESH.Elements[ielem].InterfApprox.Segments:
                for inode in range(SEGMENT.ng):
                    X_Dg[k,:] = SEGMENT.Xg[inode,:]
                    if self.PLASMA_CURRENT != self.JARDIN_CURRENT:
                        PSI_Dexact[k] = self.PSIAnalyticalSolution(X_Dg[k,:],self.PLASMA_CURRENT)
                    else:
                        PSI_Dexact[k] = SEGMENT.PSIgseg[inode]
                    PSI_Dg[k] = SEGMENT.PSIgseg[inode]
                    k += 1
            for jnode in range(self.MESH.Elements[ielem].n):
                X_D[l,:] = self.MESH.Elements[ielem].Xe[jnode,:]
                PSI_Dexact_node = self.PSIAnalyticalSolution(X_D[l,:],self.PLASMA_CURRENT)
                PSI_D[l] = self.PSI[self.MESH.Elements[ielem].Te[jnode]]
                error[l] = np.abs(PSI_D[l]-PSI_Dexact_node)
                l += 1
            
        fig, axs = plt.subplots(1, 4, figsize=(18,6)) 
        # LEFT SUBPLOT: ANALYTICAL VALUES
        axs[0].set_aspect('equal')
        axs[0].set_ylim(self.MESH.Zmin-0.5,self.MESH.Zmax+0.5)
        axs[0].set_xlim(self.MESH.Rmin-0.5,self.MESH.Rmax+0.5)
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(PSI_Dexact.min(),PSI_Dexact.max())
        linecolors_Dexact = cmap(norm(PSI_Dexact))
        axs[0].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_Dexact)
        
        # CENTER SUBPLOT: CONSTRAINT VALUES ON PSI
        axs[1].set_aspect('equal')
        axs[1].set_ylim(self.MESH.Zmin-0.5,self.MESH.Zmax+0.5)
        axs[1].set_xlim(self.MESH.Rmin-0.5,self.MESH.Rmax+0.5)
        #norm = plt.Normalize(PSI_Dg.min(),PSI_Dg.max())
        linecolors_Dg = cmap(norm(PSI_Dg))
        axs[1].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_Dg)

        # RIGHT SUBPLOT: RESULTING VALUES ON CUTFEM SYSTEM 
        axs[2].set_aspect('equal')
        axs[2].set_ylim(self.MESH.Zmin-0.5,self.MESH.Zmax+0.5)
        axs[2].set_xlim(self.MESH.Rmin-0.5,self.MESH.Rmax+0.5)
        linecolors_D = cmap(norm(PSI_D))
        axs[2].scatter(X_D[:,0],X_D[:,1],color = linecolors_D)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[2])
        
        axs[3].set_aspect('equal')
        axs[3].set_ylim(self.MESH.Zmin-0.5,self.MESH.Zmax+0.5)
        axs[3].set_xlim(self.MESH.Rmin-0.5,self.MESH.Rmax+0.5)
        norm = plt.Normalize(np.log(error).min(),np.log(error).max())
        linecolors_error = cmap(norm(np.log(error)))
        axs[3].scatter(X_D[:,0],X_D[:,1],color = linecolors_error)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[3])

        plt.show()
        
        return 


    def PlotIntegrationQuadratures(self):
        
        plt.figure(figsize=(9,11))
        plt.ylim(self.MESH.Zmin-0.25,self.MESH.Zmax+0.25)
        plt.xlim(self.MESH.Rmin-0.25,self.MESH.Rmax+0.25)

        # PLOT NODES
        plt.plot(self.MESH.X[:,0],self.MESH.X[:,1],'.',color='black')
        Tmesh = self.MESH.T +1
        # PLOT PLASMA REGION ELEMENTS
        for elem in self.MESH.PlasmaElems:
            ELEMENT = self.MESH.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.MESH.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='red', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='red')
        # PLOT VACCUM ELEMENTS
        for elem in self.MESH.VacuumElems:
            ELEMENT = self.MESH.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.MESH.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='gray', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='gray')
        # PLOT EXTERIOR ELEMENTS IF EXISTING
        for elem in self.ExteriorElems:
            ELEMENT = self.MESH.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.MESH.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='black', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='black')
            
        # PLOT PLASMA BOUNDARY ELEMENTS
        for elem in self.MESH.PlasmaBoundElems:
            ELEMENT = self.MESH.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.MESH.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='gold', linewidth=1)
            # PLOT SUBELEMENT EDGES AND INTEGRATION POINTS
            for SUBELEM in ELEMENT.SubElements:
                # PLOT SUBELEMENT EDGES
                for i in range(self.MESH.n):
                    plt.plot([SUBELEM.Xe[i,0], SUBELEM.Xe[(i+1)%SUBELEM.n,0]], [SUBELEM.Xe[i,1], SUBELEM.Xe[(i+1)%SUBELEM.n,1]], color='gold', linewidth=1)
                # PLOT QUADRATURE INTEGRATION POINTS
                plt.scatter(SUBELEM.Xg[:,0],SUBELEM.Xg[:,1],marker='x',c='gold')
            # PLOT INTERFACE  APPROXIMATION AND INTEGRATION POINTS
            for SEGMENT in ELEMENT.InterfApprox.Segments:
                # PLOT INTERFACE APPROXIMATION
                plt.plot(SEGMENT.Xseg[:,0], SEGMENT.Xseg[:,1], color='green', linewidth=1)
                # PLOT INTERFACE QUADRATURE
                plt.scatter(SEGMENT.Xg[:,0],SEGMENT.Xg[:,1],marker='o',c='green')
                
        # PLOT VACUUM VESSEL FIRST WALL ELEMENTS
        for elem in self.MESH.FirstWallElems:
            ELEMENT = self.MESH.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.MESH.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='darkturquoise', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='darkturquoise')

        plt.show()
        return



    def PlotREFERENCE_PHYSICALelement(self,element_index,TESSELLATION,BOUNDARY,NORMALS,QUADRATURE):
        ELEMENT = self.MESH.Elements[element_index]
        Xmin = np.min(ELEMENT.Xe[:,0])-0.1
        Xmax = np.max(ELEMENT.Xe[:,0])+0.1
        Ymin = np.min(ELEMENT.Xe[:,1])-0.1
        Ymax = np.max(ELEMENT.Xe[:,1])+0.1
        if ELEMENT.ElType == 1:
            numedges = 3
        elif ELEMENT.ElType == 2:
            numedges = 4
            
        color = self.ElementColor(ELEMENT.Dom)
        colorlist = ['#009E73','#D55E00','#CC79A7','#56B4E9']

        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        XIe = ReferenceElementCoordinates(ELEMENT.ElType,ELEMENT.ElOrder)
        XImin = np.min(XIe[:,0])-0.4
        XImax = np.max(XIe[:,0])+0.25
        ETAmin = np.min(XIe[:,1])-0.4
        ETAmax = np.max(XIe[:,1])+0.25
        axs[0].set_xlim(XImin,XImax)
        axs[0].set_ylim(ETAmin,ETAmax)
        axs[0].tricontour(XIe[:,0],XIe[:,1], ELEMENT.LSe, levels=[0], colors = 'red',linewidths=2)
        # PLOT ELEMENT EDGES
        for iedge in range(ELEMENT.numedges):
            axs[0].plot([XIe[iedge,0],XIe[int((iedge+1)%ELEMENT.numedges),0]],[XIe[iedge,1],XIe[int((iedge+1)%ELEMENT.numedges),1]], color=color, linewidth=8)
        for inode in range(ELEMENT.n):
            if ELEMENT.LSe[inode] < 0:
                cl = 'blue'
            else:
                cl = 'red'
            axs[0].scatter(XIe[inode,0],XIe[inode,1],s=120,color=cl,zorder=5)

        if TESSELLATION and (ELEMENT.Dom == 0 or ELEMENT.Dom == 2):
            for isub, SUBELEM in enumerate(ELEMENT.SubElements):
                # PLOT SUBELEMENT EDGES
                for i in range(SUBELEM.numedges):
                    axs[0].plot([SUBELEM.XIe[i,0], SUBELEM.XIe[(i+1)%SUBELEM.numedges,0]], [SUBELEM.XIe[i,1], SUBELEM.XIe[(i+1)%SUBELEM.numedges,1]], color=colorlist[isub], linewidth=3.5)
                axs[0].scatter(SUBELEM.XIe[:,0],SUBELEM.XIe[:,1], marker='o', s=60, color=colorlist[isub], zorder=5)
        if BOUNDARY:
            axs[0].scatter(ELEMENT.InterfApprox.XIint[:,0],ELEMENT.InterfApprox.XIint[:,1],marker='o',color='red',s=100, zorder=5)
            for SEGMENT in ELEMENT.InterfApprox.Segments:
                axs[0].scatter(SEGMENT.XIseg[:,0],SEGMENT.XIseg[:,1],marker='o',color='green',s=30, zorder=5)
        if NORMALS:
            for SEGMENT in ELEMENT.InterfApprox.Segments:
                # PLOT NORMAL VECTORS
                Xsegmean = np.mean(SEGMENT.Xseg, axis=0)
                dl = 10
                #axs[0].arrow(Xsegmean[0],Xsegmean[1],SEGMENT.NormalVec[0]/dl,SEGMENT.NormalVec[1]/dl,width=0.01)
        if QUADRATURE:
            if ELEMENT.Dom == -1 or ELEMENT.Dom == 1 or ELEMENT.Dom == 3:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[0].scatter(ELEMENT.XIg[:,0],ELEMENT.XIg[:,1],marker='x',c='black')
            elif ELEMENT.Dom == 2:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[0].scatter(ELEMENT.XIg[:,0],ELEMENT.XIg[:,1],marker='x',c='black', zorder=5)
                # PLOT INTERFACE INTEGRATION POINTS
                for SEGMENT in ELEMENT.InterfApprox.Segments:
                    axs[0].scatter(SEGMENT.XIg[:,0],SEGMENT.XIg[:,1],marker='x',color='grey',s=50, zorder = 5)
                        
                        
        axs[1].set_xlim(Xmin,Xmax)
        axs[1].set_ylim(Ymin,Ymax)
        axs[1].tricontour(self.MESH.X[:,0],self.MESH.X[:,1], self.PlasmaLS, levels=[0], colors = 'red',linewidths=2)
        # PLOT ELEMENT EDGES
        for iedge in range(ELEMENT.numedges):
            axs[1].plot([ELEMENT.Xe[iedge,0],ELEMENT.Xe[int((iedge+1)%ELEMENT.numedges),0]],[ELEMENT.Xe[iedge,1],ELEMENT.Xe[int((iedge+1)%ELEMENT.numedges),1]], color=color, linewidth=8)
        for inode in range(ELEMENT.n):
            if ELEMENT.LSe[inode] < 0:
                cl = 'blue'
            else:
                cl = 'red'
            axs[1].scatter(ELEMENT.Xe[inode,0],ELEMENT.Xe[inode,1],s=120,color=cl,zorder=5)
        if TESSELLATION and (ELEMENT.Dom == 0 or ELEMENT.Dom == 2):
            for isub, SUBELEM in enumerate(ELEMENT.SubElements):
                # PLOT SUBELEMENT EDGES
                for i in range(SUBELEM.numedges):
                    axs[1].plot([SUBELEM.Xe[i,0], SUBELEM.Xe[(i+1)%SUBELEM.numedges,0]], [SUBELEM.Xe[i,1], SUBELEM.Xe[(i+1)%SUBELEM.numedges,1]], color=colorlist[isub], linewidth=3.5)
                axs[1].scatter(SUBELEM.Xe[:,0],SUBELEM.Xe[:,1], marker='o', s=60, color=colorlist[isub], zorder=5)
        if BOUNDARY:
            axs[1].scatter(ELEMENT.InterfApprox.Xint[:,0],ELEMENT.InterfApprox.Xint[:,1],marker='o',color='red',s=100, zorder=5)
            for SEGMENT in ELEMENT.InterfApprox.Segments:
                axs[1].scatter(SEGMENT.Xseg[:,0],SEGMENT.Xseg[:,1],marker='o',color='green',s=30, zorder=5)
        if NORMALS:
            for SEGMENT in ELEMENT.InterfApprox.Segments:
                # PLOT NORMAL VECTORS
                Xsegmean = np.mean(SEGMENT.Xseg, axis=0)
                dl = 10
                axs[1].arrow(Xsegmean[0],Xsegmean[1],SEGMENT.NormalVec[0]/dl,SEGMENT.NormalVec[1]/dl,width=0.01)
        if QUADRATURE:
            if ELEMENT.Dom == -1 or ELEMENT.Dom == 1 or ELEMENT.Dom == 3:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[1].scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='black')
            elif ELEMENT.Dom == 2:
                # PLOT QUADRATURE INTEGRATION POINTS
                axs[1].scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='black', zorder=5)
                # PLOT INTERFACE INTEGRATION POINTS
                for SEGMENT in ELEMENT.InterfApprox.Segments:
                    axs[1].scatter(SEGMENT.Xg[:,0],SEGMENT.Xg[:,1],marker='x',color='grey',s=50, zorder = 5)
                
        return


    def InspectElement2(self):
        
        QUADRATURES = False

        Nx = 40
        Ny = 40
        dx = 0.01
        xgrid, ygrid = np.meshgrid(np.linspace(min(self.Xe[:,0])-dx,max(self.Xe[:,0])+dx,Nx),np.linspace(min(self.Xe[:,1])-dx,max(self.Xe[:,1])+dx,Ny),indexing='ij')
        def parabolicLS(r,z):
            return (r-6)**2+z**2-4
        LS = parabolicLS(xgrid,ygrid)
        LSint = np.zeros([Nx,Ny])
        for i in range(Nx):
            for j in range(Ny):
                LSint[i,j] = self.ElementalInterpolationPHYSICAL([xgrid[i,j],ygrid[i,j]],self.LSe)

        fig, axs = plt.subplots(1, 1, figsize=(6,6))
        plt.xlim(min(self.Xe[:,0])-dx,max(self.Xe[:,0])+dx)
        plt.ylim(min(self.Xe[:,1])-dx,max(self.Xe[:,1])+dx)
        Xe = np.zeros([self.MESH.numedges+1,2])
        Xe[:-1,:] = self.Xe[:self.MESH.numedges,:]
        Xe[-1,:] = self.Xe[0,:]
        plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=10)
        for inode in range(self.MESH.n):
            if self.LSe[inode] < 0:
                cl = 'blue'
            else:
                cl = 'red'
            plt.scatter(self.Xe[inode,0],self.Xe[inode,1],s=180,color=cl,zorder=5)
        # PLOT PLASMA BOUNDARY
        plt.contour(xgrid,ygrid,LS, levels=[0], colors='red',linewidths=6)
        # PLOT TESSELLATION 
        colorlist = ['#009E73','darkviolet','#D55E00','#CC79A7','#56B4E9']
        #colorlist = ['orange','gold','grey','cyan']
        for isub, SUBELEM in enumerate(self.SubElements):
            # PLOT SUBELEMENT EDGES
            for iedge in range(SUBELEM.numedges):
                inode = iedge
                jnode = int((iedge+1)%SUBELEM.numedges)
                if iedge == self.interfedge[isub]:
                    inodeHO = SUBELEM.numedges+(self.MESH.ElOrder-1)*inode
                    xcoords = [SUBELEM.Xe[inode,0],SUBELEM.Xe[inodeHO:inodeHO+(self.MESH.ElOrder-1),0],SUBELEM.Xe[jnode,0]]
                    xcoords = list(chain.from_iterable([x] if not isinstance(x, np.ndarray) else x for x in xcoords))
                    ycoords = [SUBELEM.Xe[inode,1],SUBELEM.Xe[inodeHO:inodeHO+(self.MESH.ElOrder-1),1],SUBELEM.Xe[jnode,1]]
                    ycoords = list(chain.from_iterable([y] if not isinstance(y, np.ndarray) else y for y in ycoords))
                    plt.plot(xcoords,ycoords, color=colorlist[isub], linewidth=3)
                else:
                    plt.plot([SUBELEM.Xe[inode,0],SUBELEM.Xe[jnode,0]],[SUBELEM.Xe[inode,1],SUBELEM.Xe[jnode,1]], color=colorlist[isub], linewidth=3)
            
            plt.scatter(SUBELEM.Xe[:,0],SUBELEM.Xe[:,1], marker='o', s=60, color=colorlist[isub], zorder=5)
            # PLOT SUBELEMENT QUADRATURE
            if QUADRATURES:
                plt.scatter(SUBELEM.Xg[:,0],SUBELEM.Xg[:,1], marker='x', s=60, color=colorlist[isub], zorder=5)
        # PLOT LEVEL-SET INTERPOLATION
        plt.contour(xgrid,ygrid,LSint,levels=[0],colors='lime')

        # PLOT INTERFACE QUADRATURE
        if QUADRATURES:
            plt.scatter(self.InterfApprox.Xg[:,0],self.InterfApprox.Xg[:,1],s=80,marker='X',color='green',zorder=7)
            dl = 100
            for ig, vec in enumerate(self.InterfApprox.NormalVec):
                plt.arrow(self.InterfApprox.Xg[ig,0],self.InterfApprox.Xg[ig,1],vec[0]/dl,vec[1]/dl,width=0.001)
        plt.show()

        return