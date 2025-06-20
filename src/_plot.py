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

class EquilipyPlotting:
    
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
    # exec(open("./colourMaps.py").read()) # load colour map
    # VermBlue = CBWcm['VeBu']             # Vermillion (-) White (0) Blue (+)
    # BlueVerm = CBWcm['BuVe']             # Blue (-) White (0) Vermillion (+)
    colorlist = [Blue, Vermillion, BluishGreen, Black,Grey, Orange, ReddishPurple, Yellow, SkyBlue]
    markerlist = ['o','^', '<', '>', 'v', 's','p','*','D']

    plasmacmap = plt.get_cmap('jet_r')
    #plasmacmap = plt.get_cmap('winter_r')
    plasmabouncolor = 'green'
    vacvesswallcolor = 'gray'
    magneticaxiscolor = 'red'
    saddlepointcolor = BluishGreen

    def PlotPSI(self):
        fig, ax = plt.subplots(1, 1, figsize=(5,6))
        ax.set_aspect('equal')
        ax.set_xlim(self.Mesh.Rmin-self.dzoom,self.Mesh.Rmax+self.dzoom)
        ax.set_ylim(self.Mesh.Zmin-self.dzoom,self.Mesh.Zmax+self.dzoom)
        contourf = ax.tricontourf(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PSI[:,0], levels=30, cmap = self.plasmacmap)
        contour1 = ax.tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PSI[:,0], levels=[self.PSI_X], colors = 'black')
        contour2 = ax.tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PlasmaLS[:,1], levels=[0], colors = self.plasmabouncolor)
        
        # Mask solution outside computational domain's boundary 
        compboundary = np.zeros([len(self.Mesh.BoundaryVertices)+1,2])
        compboundary[:-1,:] = self.Mesh.X[self.Mesh.BoundaryVertices,:]
        # Close path
        compboundary[-1,:] = compboundary[0,:]
        clip_path = Path(compboundary)
        patch = PathPatch(clip_path, transform=ax.transData)
        for cont in [contourf,contour1,contour2]:
            for coll in cont.collections:
                coll.set_clip_path(patch)
        # Plot computational domain's boundary
        for iboun in range(self.Mesh.Nbound):
            ax.plot(self.Mesh.X[self.Mesh.Tbound[iboun,:2],0],self.Mesh.X[self.Mesh.Tbound[iboun,:2],1],linewidth = 3, color = 'grey')
                
        plt.colorbar(contourf, ax=ax)
        plt.show()
        return


    def PlotFIELD(self,FIELD,plotnodes):
        
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        ax.set_xlim(self.Mesh.Rmin-self.dzoom,self.Mesh.Rmax+self.dzoom)
        ax.set_ylim(self.Mesh.Zmin-self.dzoom,self.Mesh.Zmax+self.dzoom)
        a = ax.tricontourf(self.Mesh.X[plotnodes,0],self.Mesh.X[plotnodes,1], FIELD[plotnodes], levels=30)
        ax.tricontour(self.Mesh.X[plotnodes,0],self.Mesh.X[plotnodes,1], FIELD[plotnodes], levels=[0], colors = 'black')
        ax.tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PlasmaLS[:,1], levels=[0], colors = 'red')
        plt.colorbar(a, ax=ax)
        plt.show()
        
        return

    def PlotError(self,RelativeError = False):
        
        AnaliticalNorm = np.zeros([self.Mesh.Nn])
        for inode in range(self.Mesh.Nn):
            AnaliticalNorm[inode] = self.PlasmaCurrent.PSIanalytical(self.Mesh.X[inode,:])
            
        print('||PSIerror||_L2 = ', self.ErrorL2norm)
        print('relative ||PSIerror||_L2 = ', self.RelErrorL2norm)
        print('plasma boundary subelements ||PSIerror||_L2 = ', self.ErrorL2normPlasmaBound)
        print('plasma boundary subelements relative ||PSIerror||_L2 = ', self.RelErrorL2normPlasmaBound)
        print('interface ||PSIerror||_L2 = ', self.ErrorL2normINT)
        print('interface relative ||PSIerror||_L2 = ', self.RelErrorL2normINT)
        print('||PSIerror|| = ',np.linalg.norm(self.PSIerror))
        print('||PSIerror||/node = ',np.linalg.norm(self.PSIerror)/self.Mesh.Nn)
        print('relative ||PSIerror|| = ',np.linalg.norm(self.PSIrelerror))
        print('||jump(grad)||_L2 = ', self.InterfGradJumpErrorL2norm)
            
        # Compute global min and max across both datasets
        vmin = min(AnaliticalNorm)
        vmax = max(AnaliticalNorm)  
            
        fig, axs = plt.subplots(1, 4, figsize=(16,5),gridspec_kw={'width_ratios': [1,1,0.25,1]})
        axs[0].set_xlim(self.Mesh.Rmin-self.dzoom,self.Mesh.Rmax+self.dzoom)
        axs[0].set_ylim(self.Mesh.Zmin-self.dzoom,self.Mesh.Zmax+self.dzoom)
        a1 = axs[0].tricontourf(self.Mesh.X[:,0],self.Mesh.X[:,1], AnaliticalNorm, levels=30, vmin=vmin, vmax=vmax)
        axs[0].tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PlasmaLS[:,1], levels=[0], colors = 'red')
        axs[0].tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], AnaliticalNorm, levels=[0], colors = 'black')

        axs[1].set_xlim(self.Mesh.Rmin-self.dzoom,self.Mesh.Rmax+self.dzoom)
        axs[1].set_ylim(self.Mesh.Zmin-self.dzoom,self.Mesh.Zmax+self.dzoom)
        a2 = axs[1].tricontourf(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PSI_CONV, levels=30, vmin=vmin, vmax=vmax)
        axs[1].tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PlasmaLS[:,1], levels=[0], colors = 'red')
        axs[1].tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PSI_CONV, levels=[0], colors = 'black')

        fig.colorbar(a1, ax=axs[2], orientation="vertical", fraction=0.8, pad=-0.7)
        axs[2].axis('off')
        
        axs[2].set_xlim(self.Mesh.Rmin-self.dzoom,self.Mesh.Rmax+self.dzoom)
        axs[2].set_ylim(self.Mesh.Zmin-self.dzoom,self.Mesh.Zmax+self.dzoom)
        if RelativeError:
            errorfield = self.PSIrelerror
        else:
            errorfield = self.PSIerror
        vmax = max(np.log(errorfield))
        a = axs[3].tricontourf(self.Mesh.X[:,0],self.Mesh.X[:,1], np.log(errorfield), levels=30 , vmax=vmax,vmin=-20)
        plt.colorbar(a, ax=axs[3])

        plt.show()
        return


    def PlotElementalInterfaceApproximation(self,interface_index):
        self.Mesh.Elements[self.Mesh.PlasmaBoundElems[interface_index]].PlotInterfaceApproximation(self.InitialPlasmaLevelSetFunction)
        return


    def PlotSolutionPSI(self):
        """ FUNCTION WHICH PLOTS THE FIELD VALUES FOR PSI, OBTAINED FROM SOLVING THE CUTFEM SYSTEM, 
        AND PSI_NORM IF NORMALISED. """
        
        def subplotfield(self,ax,field,normalised=True):
            if normalised:
                psisep = self.PSIseparatrix
            else:
                psisep = self.PSI_X
            contourf = ax.tricontourf(self.Mesh.X[:,0],self.Mesh.X[:,1], field, levels=50, cmap=self.plasmacmap)
            contour1 = ax.tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], field, levels=[psisep], colors = 'black',linewidths=2)
            contour2 = ax.tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PlasmaLS[:,1], levels=[0], colors = self.plasmabouncolor, linewidths=3)
            # Mask solution outside computational domain's boundary 
            compboundary = np.zeros([len(self.Mesh.BoundaryVertices)+1,2])
            compboundary[:-1,:] = self.Mesh.X[self.Mesh.BoundaryVertices,:]
            # Close path
            compboundary[-1,:] = compboundary[0,:]
            clip_path = Path(compboundary)
            patch = PathPatch(clip_path, transform=ax.transData)
            for cont in [contourf,contour1,contour2]:
                for coll in cont.collections:
                    coll.set_clip_path(patch)
            # Plot computational domain's boundary
            for iboun in range(self.Mesh.Nbound):
                ax.plot(self.Mesh.X[self.Mesh.Tbound[iboun,:2],0],self.Mesh.X[self.Mesh.Tbound[iboun,:2],1],linewidth = 4, color = self.vacvesswallcolor)
            ax.set_xlim(self.Mesh.Rmin-self.dzoom,self.Mesh.Rmax+self.dzoom)
            ax.set_ylim(self.Mesh.Zmin-self.dzoom,self.Mesh.Zmax+self.dzoom)
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
            
        elif self.converg_EXT:  # CONVERGED SOLUTION PLOT
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
                    axs[i].scatter(self.Xcrit[1,0,0],self.Xcrit[1,0,1],marker = 'X',facecolor=self.magneticaxiscolor, edgecolor='k', s = 100, linewidths = 1.5,zorder=5)
                    # SADDLE POINT
                    axs[i].scatter(self.Xcrit[1,1,0],self.Xcrit[1,1,1],marker = 'X',facecolor=self.saddlepointcolor, edgecolor='k', s = 100, linewidths = 1.5,zorder=5)
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
                    axs[i].scatter(self.Xcrit[1,0,0],self.Xcrit[1,0,1],marker = 'X',facecolor=self.magneticaxiscolor, edgecolor='k', s = 100, linewidths = 1.5,zorder=5)
                    # SADDLE POINT
                    axs[i].scatter(self.Xcrit[1,1,0],self.Xcrit[1,1,1],marker = 'X',facecolor=self.saddlepointcolor, edgecolor='k', s = 100, linewidths = 1.5,zorder=5)
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


    def PlotMagneticField(self):
        # COMPUTE MAGNETIC FIELD NORM
        Bnorm = np.zeros([self.Mesh.Ne*self.nge])
        for inode in range(self.Mesh.Ne*self.nge):
            Bnorm[inode] = np.linalg.norm(self.Brzfield[inode,:])
            
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        ax.set_xlim(self.Mesh.Rmin,self.Mesh.Rmax)
        ax.set_ylim(self.Mesh.Zmin,self.Mesh.Zmax)
        a = ax.tricontourf(self.Xg[:,0],self.Xg[:,1], Bnorm, levels=30)
        ax.tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PlasmaLS, levels=[0], colors = 'red')
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

    def InspectElement(self,element_index,BOUNDARY,PSI,TESSELLATION,GHOSTFACES,NORMALS,QUADRATURE):
        ELEMENT = self.Mesh.Elements[element_index]
        Xmin = np.min(ELEMENT.Xe[:,0])-self.Mesh.meanLength/4
        Xmax = np.max(ELEMENT.Xe[:,0])+self.Mesh.meanLength/4
        Ymin = np.min(ELEMENT.Xe[:,1])-self.Mesh.meanLength/4
        Ymax = np.max(ELEMENT.Xe[:,1])+self.Mesh.meanLength/4
            
        color = self.ElementColor(ELEMENT.Dom)
        colorlist = ['#009E73','#D55E00','#CC79A7','#56B4E9']

        fig, axs = plt.subplots(1, 2, figsize=(10,6))
        axs[0].set_xlim(self.Mesh.Rmin-0.25,self.Mesh.Rmax+0.25)
        axs[0].set_ylim(self.Mesh.Zmin-0.25,self.Mesh.Zmax+0.25)
        if PSI:
            axs[0].tricontourf(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PSI_NORM[:,1], levels=30, cmap='plasma')
            axs[0].tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PSI_NORM[:,1], levels=[0], colors = 'black')
        axs[0].tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PlasmaLS, levels=[0], colors = 'red')
        # PLOT ELEMENT EDGES
        for iedge in range(ELEMENT.numedges):
            axs[0].plot([ELEMENT.Xe[iedge,0],ELEMENT.Xe[int((iedge+1)%ELEMENT.numedges),0]],[ELEMENT.Xe[iedge,1],ELEMENT.Xe[int((iedge+1)%ELEMENT.numedges),1]], color=color, linewidth=3)

        axs[1].set_xlim(Xmin,Xmax)
        axs[1].set_ylim(Ymin,Ymax)
        axs[1].set_aspect('equal')
        axs[1].tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PlasmaLS, levels=[0], colors = 'red',linewidths=2)
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
            ghostface = self.Mesh.GhostFaces[index]
        elif BOUNDARY == self.VACVESbound:
            ghostface == self.VacVessWallGhostFaces[index]

        # ISOLATE ELEMENTS
        ELEMS = [self.Mesh.Elements[ghostface[1][0]],self.Mesh.Elements[ghostface[2][0]]]
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


    def PlotLevelSetEvolution(self,Zlow,Rleft):
        
        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        axs[0].set_xlim(self.Mesh.Rmin,self.Mesh.Rmax)
        axs[0].set_ylim(self.Mesh.Zmin,self.Mesh.Zmax)
        a = axs[0].tricontourf(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PSI_NORM[:,1], levels=30)
        axs[0].tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PSI_NORM[:,1], levels=[0], colors = 'black')
        plt.colorbar(a, ax=axs[0])

        axs[1].set_xlim(self.Mesh.Rmin,self.Mesh.Rmax)
        axs[1].set_ylim(self.Mesh.Zmin,self.Mesh.Zmax)
        a = axs[1].tricontourf(self.Mesh.X[:,0],self.Mesh.X[:,1], np.sign(self.PlasmaLS), levels=30)
        axs[1].tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PSI_NORM[:,1], levels=[0], colors = 'black',linewidths = 3)
        axs[1].tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PlasmaLS, levels=[0], colors = 'red',linestyles = 'dashed')
        axs[1].tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PlasmaLS_ALL[:,self.it-1], levels=[0], colors = 'orange',linestyles = 'dashed')
        axs[1].plot([self.Mesh.Rmin,self.Mesh.Rmax],[Zlow,Zlow],color = 'green')
        axs[1].plot([Rleft,Rleft],[self.Mesh.Zmin,self.Mesh.Zmax],color = 'green')

        plt.show()
        
        return


    def PlotMesh(self):
        plt.figure(figsize=(7,10))
        plt.ylim(np.min(self.Mesh.X[:,1]),np.max(self.Mesh.X[:,1]))
        plt.xlim(np.min(self.Mesh.X[:,0]),np.max(self.Mesh.X[:,0]))
        # Plot nodes
        plt.plot(self.Mesh.X[:,0],self.Mesh.X[:,1],'.')
        # Plot element edges
        for e in range(self.Mesh.Ne):
            for i in range(self.Mesh.numedges):
                plt.plot([self.Mesh.X[self.Mesh.T[e,i],0], self.Mesh.X[self.Mesh.T[e,int((i+1)%self.Mesh.n)],0]], 
                        [self.Mesh.X[self.Mesh.T[e,i],1], self.Mesh.X[self.Mesh.T[e,int((i+1)%self.Mesh.n)],1]], color='black', linewidth=1)
        plt.show()
        return

    def PlotClassifiedElements(self,GHOSTFACES,**kwargs):
        plt.figure(figsize=(5,6))
        if not kwargs:
            plt.ylim(self.Mesh.Zmin-0.25,self.Mesh.Zmax+0.25)
            plt.xlim(self.Mesh.Rmin-0.25,self.Mesh.Rmax+0.25)
        else: 
            plt.ylim(kwargs['zmin'],kwargs['zmax'])
            plt.xlim(kwargs['rmin'],kwargs['rmax'])
        
        # PLOT PLASMA REGION ELEMENTS
        for elem in self.Mesh.PlasmaElems:
            ELEMENT = self.Mesh.Elements[elem]
            Xe = np.zeros([ELEMENT.numedges+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.Mesh.numedges,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'red')
        # PLOT VACCUM ELEMENTS
        for elem in self.Mesh.VacuumElems:
            ELEMENT = self.Mesh.Elements[elem]
            Xe = np.zeros([ELEMENT.numedges+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.Mesh.numedges,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'gray')
        # PLOT PLASMA BOUNDARY ELEMENTS
        for elem in self.Mesh.PlasmaBoundElems:
            ELEMENT = self.Mesh.Elements[elem]
            Xe = np.zeros([ELEMENT.numedges+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.Mesh.numedges,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'gold')
        # PLOT VACUUM VESSEL FIRST WALL ELEMENTS
        for elem in self.Mesh.FirstWallElems:
            ELEMENT = self.Mesh.Elements[elem]
            Xe = np.zeros([ELEMENT.numedges+1,2])
            Xe[:-1,:] = ELEMENT.Xe[:self.Mesh.numedges,:]
            Xe[-1,:] = ELEMENT.Xe[0,:]
            plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
            plt.fill(Xe[:,0], Xe[:,1], color = 'cyan')
            
        # PLOT PLASMA BOUNDARY  
        plt.tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PlasmaLS, levels=[0], colors='green',linewidths=3)
                
        # PLOT GHOSTFACES 
        if GHOSTFACES:
            for ghostface in self.Mesh.GhostFaces:
                plt.plot(self.Mesh.X[ghostface[0][:2],0],self.Mesh.X[ghostface[0][:2],1],linewidth=2,color='#56B4E9')
            
        #colorlist = ['#009E73','#D55E00','#CC79A7','#56B4E9']
        plt.show()
        return
        

    def PlotNormalVectors(self):
        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        axs[0].set_xlim(self.Mesh.Rmin-0.5,self.Mesh.Rmax+0.5)
        axs[0].set_ylim(self.Mesh.Zmin-0.5,self.Mesh.Zmax+0.5)
        axs[1].set_xlim(6.5,7)
        if self.FIXED_BOUNDARY:
            axs[1].set_ylim(1.6,2)
        else:
            axs[1].set_ylim(2.2,2.6)

        for i in range(2):
            # PLOT PLASMA/VACUUM INTERFACE
            axs[i].tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PlasmaLS, levels=[0], colors='green',linewidths=6)
            # PLOT NORMAL VECTORS
            for ielem in self.Mesh.PlasmaBoundElems:
                ELEMENT = self.Mesh.Elements[ielem]
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


    def PlotInterfaceValues(self):
        """ Function which plots the values PSIgseg at the interface edges, for both the plasma/vacuum interface and the vacuum vessel first wall. """

        # IMPOSED BOUNDARY VALUES
        ### VACUUM VESSEL FIRST WALL
        PSI_Bg = self.PSI_B[:,1]
        PSI_B = self.PSI[self.Mesh.BoundaryNodes]
            
        ### PLASMA BOUNDARY
        X_Dg = np.zeros([self.Mesh.NnPB,self.Mesh.dim])
        PSI_Dg = np.zeros([self.Mesh.NnPB])
        PSI_D = np.zeros([self.Mesh.NnPB])
        k = 0
        for ielem in self.Mesh.PlasmaBoundElems:
            INTAPPROX = self.Mesh.Elements[ielem].InterfApprox
            for inode in range(INTAPPROX.ng):
                X_Dg[k,:] = INTAPPROX.Xg[inode,:]
                PSI_Dg[k] = INTAPPROX.PSIg[inode]
                PSI_D[k] = self.Mesh.Elements[ielem].ElementalInterpolationPHYSICAL(X_Dg[k,:],self.PSI[self.Mesh.Elements[ielem].Te])
                k += 1
            
        fig, axs = plt.subplots(1, 2, figsize=(14,7))
        ### UPPER ROW SUBPLOTS 
        # LEFT SUBPLOT: CONSTRAINT VALUES ON PSI
        axs[0].set_aspect('equal')
        axs[0].set_ylim(self.Mesh.Zmin-0.5,self.Mesh.Zmax+0.5)
        axs[0].set_xlim(self.Mesh.Rmin-0.5,self.Mesh.Rmax+0.5)
        cmap = plt.get_cmap('jet')
        
        norm = plt.Normalize(np.min([PSI_Bg.min(),PSI_Dg.min()]),np.max([PSI_Bg.max(),PSI_Dg.max()]))
        linecolors_Dg = cmap(norm(PSI_Dg))
        linecolors_Bg = cmap(norm(PSI_Bg))
        axs[0].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_Dg)
        axs[0].scatter(self.Mesh.X[self.Mesh.BoundaryNodes,0],self.Mesh.X[self.Mesh.BoundaryNodes,1],color = linecolors_Bg)

        # RIGHT SUBPLOT: RESULTING VALUES ON CUTFEM SYSTEM 
        axs[1].set_aspect('equal')
        axs[1].set_ylim(self.Mesh.Zmin-0.5,self.Mesh.Zmax+0.5)
        axs[1].set_xlim(self.Mesh.Rmin-0.5,self.Mesh.Rmax+0.5)
        linecolors_D = cmap(norm(PSI_D))
        linecolors_B = cmap(norm(PSI_B))
        axs[1].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_D)
        axs[1].scatter(self.Mesh.X[self.Mesh.BoundaryNodes,0],self.Mesh.X[self.Mesh.BoundaryNodes,1],color = linecolors_B)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[1])

        plt.show()
        return


    def PlotPlasmaBoundaryConstraints(self):
        
        # COLLECT PSIgseg DATA ON PLASMA/VACUUM INTERFACE
        X_Dg = np.zeros([len(self.Mesh.PlasmaBoundElems)*self.Ng1D,self.Mesh.dim])
        PSI_Dexact = np.zeros([len(self.Mesh.PlasmaBoundElems)*self.Ng1D])
        PSI_Dg = np.zeros([len(self.Mesh.PlasmaBoundElems)*self.Ng1D])
        X_D = np.zeros([len(self.Mesh.PlasmaBoundElems)*self.Mesh.n,self.Mesh.dim])
        PSI_D = np.zeros([len(self.Mesh.PlasmaBoundElems)*self.Mesh.n])
        error = np.zeros([len(self.Mesh.PlasmaBoundElems)*self.Mesh.n])
        k = 0
        l = 0
        for ielem in self.Mesh.PlasmaBoundElems:
            for SEGMENT in self.Mesh.Elements[ielem].InterfApprox.Segments:
                for inode in range(SEGMENT.ng):
                    X_Dg[k,:] = SEGMENT.Xg[inode,:]
                    if self.PLASMA_CURRENT != self.JARDIN_CURRENT:
                        PSI_Dexact[k] = self.PSIAnalyticalSolution(X_Dg[k,:],self.PLASMA_CURRENT)
                    else:
                        PSI_Dexact[k] = SEGMENT.PSIgseg[inode]
                    PSI_Dg[k] = SEGMENT.PSIgseg[inode]
                    k += 1
            for jnode in range(self.Mesh.Elements[ielem].n):
                X_D[l,:] = self.Mesh.Elements[ielem].Xe[jnode,:]
                PSI_Dexact_node = self.PSIAnalyticalSolution(X_D[l,:],self.PLASMA_CURRENT)
                PSI_D[l] = self.PSI[self.Mesh.Elements[ielem].Te[jnode]]
                error[l] = np.abs(PSI_D[l]-PSI_Dexact_node)
                l += 1
            
        fig, axs = plt.subplots(1, 4, figsize=(18,6)) 
        # LEFT SUBPLOT: ANALYTICAL VALUES
        axs[0].set_aspect('equal')
        axs[0].set_ylim(self.Mesh.Zmin-0.5,self.Mesh.Zmax+0.5)
        axs[0].set_xlim(self.Mesh.Rmin-0.5,self.Mesh.Rmax+0.5)
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(PSI_Dexact.min(),PSI_Dexact.max())
        linecolors_Dexact = cmap(norm(PSI_Dexact))
        axs[0].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_Dexact)
        
        # CENTER SUBPLOT: CONSTRAINT VALUES ON PSI
        axs[1].set_aspect('equal')
        axs[1].set_ylim(self.Mesh.Zmin-0.5,self.Mesh.Zmax+0.5)
        axs[1].set_xlim(self.Mesh.Rmin-0.5,self.Mesh.Rmax+0.5)
        #norm = plt.Normalize(PSI_Dg.min(),PSI_Dg.max())
        linecolors_Dg = cmap(norm(PSI_Dg))
        axs[1].scatter(X_Dg[:,0],X_Dg[:,1],color = linecolors_Dg)

        # RIGHT SUBPLOT: RESULTING VALUES ON CUTFEM SYSTEM 
        axs[2].set_aspect('equal')
        axs[2].set_ylim(self.Mesh.Zmin-0.5,self.Mesh.Zmax+0.5)
        axs[2].set_xlim(self.Mesh.Rmin-0.5,self.Mesh.Rmax+0.5)
        linecolors_D = cmap(norm(PSI_D))
        axs[2].scatter(X_D[:,0],X_D[:,1],color = linecolors_D)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[2])
        
        axs[3].set_aspect('equal')
        axs[3].set_ylim(self.Mesh.Zmin-0.5,self.Mesh.Zmax+0.5)
        axs[3].set_xlim(self.Mesh.Rmin-0.5,self.Mesh.Rmax+0.5)
        norm = plt.Normalize(np.log(error).min(),np.log(error).max())
        linecolors_error = cmap(norm(np.log(error)))
        axs[3].scatter(X_D[:,0],X_D[:,1],color = linecolors_error)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[3])

        plt.show()
        
        return 


    def PlotIntegrationQuadratures(self):
        
        plt.figure(figsize=(9,11))
        plt.ylim(self.Mesh.Zmin-0.25,self.Mesh.Zmax+0.25)
        plt.xlim(self.Mesh.Rmin-0.25,self.Mesh.Rmax+0.25)

        # PLOT NODES
        plt.plot(self.Mesh.X[:,0],self.Mesh.X[:,1],'.',color='black')
        Tmesh = self.Mesh.T +1
        # PLOT PLASMA REGION ELEMENTS
        for elem in self.Mesh.PlasmaElems:
            ELEMENT = self.Mesh.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.Mesh.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='red', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='red')
        # PLOT VACCUM ELEMENTS
        for elem in self.Mesh.VacuumElems:
            ELEMENT = self.Mesh.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.Mesh.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='gray', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='gray')
        # PLOT EXTERIOR ELEMENTS IF EXISTING
        for elem in self.ExteriorElems:
            ELEMENT = self.Mesh.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.Mesh.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='black', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='black')
            
        # PLOT PLASMA BOUNDARY ELEMENTS
        for elem in self.Mesh.PlasmaBoundElems:
            ELEMENT = self.Mesh.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.Mesh.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='gold', linewidth=1)
            # PLOT SUBELEMENT EDGES AND INTEGRATION POINTS
            for SUBELEM in ELEMENT.SubElements:
                # PLOT SUBELEMENT EDGES
                for i in range(self.Mesh.n):
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
        for elem in self.Mesh.FirstWallElems:
            ELEMENT = self.Mesh.Elements[elem]
            # PLOT ELEMENT EDGES
            for i in range(self.Mesh.n):
                plt.plot([ELEMENT.Xe[i,0], ELEMENT.Xe[(i+1)%ELEMENT.n,0]], [ELEMENT.Xe[i,1], ELEMENT.Xe[(i+1)%ELEMENT.n,1]], color='darkturquoise', linewidth=1)
            # PLOT QUADRATURE INTEGRATION POINTS
            plt.scatter(ELEMENT.Xg[:,0],ELEMENT.Xg[:,1],marker='x',c='darkturquoise')

        plt.show()
        return


    @staticmethod
    def ElementColor(dom):
        if dom == -1:
            color = 'red'
        elif dom == 0:
            color = 'gold'
        elif dom == 1:
            color = 'grey'
        elif dom == 2:
            color = 'cyan'
        elif dom == 3:
            color = 'black'
        return color

    def PlotREFERENCE_PHYSICALelement(self,element_index,TESSELLATION,BOUNDARY,NORMALS,QUADRATURE):
        ELEMENT = self.Mesh.Elements[element_index]
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
        axs[1].tricontour(self.Mesh.X[:,0],self.Mesh.X[:,1], self.PlasmaLS, levels=[0], colors = 'red',linewidths=2)
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
        Xe = np.zeros([self.Mesh.numedges+1,2])
        Xe[:-1,:] = self.Xe[:self.Mesh.numedges,:]
        Xe[-1,:] = self.Xe[0,:]
        plt.plot(Xe[:,0], Xe[:,1], color='black', linewidth=10)
        for inode in range(self.Mesh.n):
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
                    inodeHO = SUBELEM.numedges+(self.Mesh.ElOrder-1)*inode
                    xcoords = [SUBELEM.Xe[inode,0],SUBELEM.Xe[inodeHO:inodeHO+(self.Mesh.ElOrder-1),0],SUBELEM.Xe[jnode,0]]
                    xcoords = list(chain.from_iterable([x] if not isinstance(x, np.ndarray) else x for x in xcoords))
                    ycoords = [SUBELEM.Xe[inode,1],SUBELEM.Xe[inodeHO:inodeHO+(self.Mesh.ElOrder-1),1],SUBELEM.Xe[jnode,1]]
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