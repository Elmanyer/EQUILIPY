import numpy as np
from shutil import copy2

##################################################################################################
####################################### EQUILIPY OUTPUT ##########################################
##################################################################################################

class EquilipyOutput:
    
    def __init__(self):
        
        # INITIATE OUTPUT SWITCHES
        self.out_PSIcrit = False
        self.out_proparams = False          # SIMULATION PARAMETERS 
        self.out_elemsClas = False          # CLASSIFICATION OF MESH ELEMENTS
        self.out_plasmaLS = False           # PLASMA BOUNDARY LEVEL-SET FIELD VALUES 
        self.out_plasmaBC = False           # PLASMA BOUNDARY CONDITION VALUES 
        self.out_plasmaapprox = False       # PLASMA BOUNDARY APPROXIMATION DATA 
        self.out_ghostfaces = False         # GHOST STABILISATION FACES DATA 
        self.out_elemsys = False            # ELEMENTAL MATRICES
        self.plotelemsClas = False          # ELEMENTS CLASSIFICATION PLOTS AT EACH ITERATION
        self.plotPSI = False                # PSI SOLUTION PLOTS AT EACH ITERATION
        self.out_pickle = False             # SIMULATION DATA PYTHON PICKLE
        
        # INITIATE OUTPUT FILES
        self.outputdir = None
        self.file_proparams = None          # OUTPUT FILE CONTAINING THE SIMULATION PARAMETERS 
        self.file_PSI = None                # OUTPUT FILE CONTAINING THE PSI FIELD VALUES OBTAINED BY SOLVING THE CutFEM SYSTEM
        self.file_PSIcrit = None            # OUTPUT FILE CONTAINING THE CRITICAL PSI VALUES
        self.file_PSI_NORM = None           # OUTPUT FILE CONTAINING THE PSI_NORM FIELD VALUES (AFTER NORMALISATION OF PSI FIELD)
        self.file_PSI_B = None              # OUTPUT FILE CONTAINING THE PSI_B BOUNDARY VALUES
        self.file_RESIDU = None             # OUTPUT FILE CONTAINING THE RESIDU FOR EACH ITERATION
        self.file_elemsClas = None          # OUTPUT FILE CONTAINING THE CLASSIFICATION OF MESH ELEMENTS
        self.file_plasmaLS = None           # OUTPUT FILE CONTAINING THE PLASMA BOUNDARY LEVEL-SET FIELD VALUES
        self.file_plasmaBC = None           # OUTPUT FILE CONTAINING THE PLASMA BOUNDARY CONDITION VALUES
        self.file_plasmaapprox = None       # OUTPUT FILE CONTAINING THE PLASMA BOUNDARY APPROXIMATION DATA
        self.file_ghostfaces = None         # OUTPUT FILE CONTAINING THE GHOST STABILISATION FACES DATA
        self.file_L2error = None            # OUTPUT FILE CONTAINING THE ERROR FIELD AND THE L2 ERROR NORM FOR THE CONVERGED SOLUTION 
        self.file_elemsys = None            # OUTPUT FILE CONTAINING THE ELEMENTAL MATRICES FOR EACH ITERATION
        self.file_globalsys = None
        
        super().__init__()
        return

    def openOUTPUTfiles(self):
        """
        Open files for selected output. 
        """
        if self.out_elemsClas:
            self.file_elemsClas = open(self.outputdir+'/MeshElementsClassification.dat', 'w')
            self.file_elemsClas.write('MESH_ELEMENTS_CLASSIFICATION_FILE\n')
        
        if self.out_plasmaLS:
            self.file_plasmaLS = open(self.outputdir+'/PlasmaBoundLS.dat', 'w')
            self.file_plasmaLS.write('PLASMA_BOUNDARY_LEVEL_SET_FILE\n')
            
        if self.out_plasmaapprox:
            self.file_plasmaapprox = open(self.outputdir+'/PlasmaBoundApprox.dat', 'w')
            self.file_plasmaapprox.write('PLASMA_BOUNDARY_APPROXIMATION_FILE\n')
            
        if self.out_plasmaBC:
            self.file_plasmaBC = open(self.outputdir+'/PlasmaBC.dat', 'w')
            self.file_plasmaBC.write('PLASMA_BOUNDARY_CONSTRAINT_VALUES_FILE\n')
            
        if self.out_ghostfaces:
            self.file_ghostfaces = open(self.outputdir+'/GhostFaces.dat', 'w')
            self.file_ghostfaces.write('GHOST_STABILISATION_FACES_FILE\n')
        
        self.file_PSI = open(self.outputdir+'/UNKNO.dat', 'w')
        self.file_PSI.write('PSI_FIELD\n')
            
        self.file_PSI_NORM = open(self.outputdir+'/PSIpol.dat', 'w')
        self.file_PSI_NORM.write('PSIpol_FIELD\n')
        
        if self.out_PSIcrit:
            self.file_PSIcrit = open(self.outputdir+'/PSIcrit.dat', 'w')
            self.file_PSIcrit.write('PSIcrit_VALUES\n')

        self.file_PSI_B = open(self.outputdir+'/PSIpol_B.dat', 'w')
        self.file_PSI_B.write('PSIpol_B_VALUES\n')
        
        self.file_RESIDU = open(self.outputdir+'/Residu.dat', 'w')
        self.file_RESIDU.write('RESIDU_VALUES\n')
        
        if self.out_elemsys:
            self.file_elemsys = open(self.outputdir+'/ElementalMatrices.dat', 'w')
            self.file_elemsys.write('ELEMENTAL_MATRICES_FILE\n')
            
            self.file_globalsys = open(self.outputdir+'/GlobalMatrices.dat', 'w')
            self.file_globalsys.write('GLOBAL_MATRICES_FILE\n')
        
        return

    def closeOUTPUTfiles(self):
        """
        Close files for selected output.
        """
        if self.out_elemsClas:
            self.file_elemsClas.write('END_MESH_ELEMENTS_CLASSIFICATION_FILE')
            self.file_elemsClas.close()
        
        if self.out_plasmaLS:
            self.file_plasmaLS.write('END_PLASMA_BOUNDARY_LEVEL_SET_FILE')
            self.file_plasmaLS.close()
        
        if self.out_plasmaapprox:
            self.file_plasmaapprox.write('END_PLASMA_BOUNDARY_APPROXIMATION_FILE\n')
            self.file_plasmaapprox.close()
            
        if self.out_plasmaBC:
            self.file_plasmaBC.write('END_PLASMA_BOUNDARY_CONSTRAINT_VALUES_FILE')
            self.file_plasmaBC.close()
            
        if self.out_ghostfaces:
            self.file_ghostfaces.write('END_GHOST_STABILISATION_FACES_FILE\n')
            self.file_ghostfaces.close()
            
        self.file_PSI.write('END_PSI_FIELD')
        self.file_PSI.close()
        
        self.file_PSI_NORM.write('END_PSIpol_FIELD')
        self.file_PSI_NORM.close()
        
        if self.out_PSIcrit:
            self.file_PSIcrit.write('END_PSIcrit_VALUES')
            self.file_PSIcrit.close()
        
        self.file_PSI_B.write('END_PSIpol_B_VALUES')
        self.file_PSI_B.close()
        
        self.file_RESIDU.write('END_RESIDU_VALUES')
        self.file_RESIDU.close()
        
        if self.out_elemsys:
            self.file_elemsys.write('END_ELEMENTAL_MATRICES_FILE\n')
            self.file_elemsys.close()
            
            self.file_globalsys.write('END_GLOBAL_MATRICES_FILE\n')
            self.file_globalsys.close()
            
        return

    def copysimfiles(self):
        """
        Copies the simulation files (DOM.DAT, GEO.DAT, and EQU.DAT) to the output directory for the given case and mesh.

        This function handles the copying of essential simulation data files from the mesh folder and case file location
        to the output directory. The files copied include the mesh domain data (`dom.dat`), geometry data (`geo.dat`), and 
        equilibrium data (`equ.dat`).

        Steps:
            1. Copies the mesh domain file (`dom.dat`) from the mesh folder to the output directory.
            2. Copies the geometry file (`geo.dat`) from the mesh folder to the output directory.
            3. Copies the equilibrium data file (`equ.dat`) from the case file to the output directory.
        """
        
        # COPY DOM.DAT FILE
        MeshDataFile = self.mesh_folder +'/' + self.MESH +'.dom.dat'
        copy2(MeshDataFile,self.outputdir+'/'+self.CASE+'-'+self.MESH+'.dom.dat')
        # COPY GEO.DAT FILE
        MeshFile = self.mesh_folder +'/' + self.MESH +'.geo.dat'
        copy2(MeshFile,self.outputdir+'/'+self.CASE+'-'+self.MESH+'.geo.dat')
        
        return

    def writeparams(self):
        """
        Write simulation parameters in output file.
        """
        if self.out_proparams:
            self.file_proparams = open(self.outputdir+'/PARAMETERS.dat', 'w')
            self.file_proparams.write('SIMULATION_PARAMETERS_FILE\n')
            self.file_proparams.write('\n')
            
            self.file_proparams.write('MESH_PARAMETERS\n')
            self.file_proparams.write("    NPOIN = {:d}\n".format(self.Mesh.Nn))
            self.file_proparams.write("    NELEM = {:d}\n".format(self.Mesh.Ne))
            self.file_proparams.write("    ELEMENT = {:d}\n".format(self.Mesh.ElTypeALYA))
            self.file_proparams.write("    NBOUN = {:d}\n".format(self.Mesh.Nbound))
            self.file_proparams.write("    DIM = {:d}\n".format(self.Mesh.dim))
            self.file_proparams.write('END_MESH_PARAMETERS\n')
            self.file_proparams.write('\n')
            
            self.file_proparams.write('PROBLEM_TYPE_PARAMETERS\n')
            if self.FIXED_BOUNDARY:
                self.file_proparams.write("    PLASMA_BOUNDARY = FIXED")
            else:
                self.file_proparams.write("    PLASMA_BOUNDARY = FREE")
            self.file_proparams.write("    PLASMA_CURRENT = {:d}\n".format(self.PlasmaCurrent.CURRENT_MODEL))
            self.file_proparams.write('END_PROBLEM_TYPE_PARAMETERS\n')
            self.file_proparams.write('\n')
            
            """
            self.file_proparams.write('VACUUM_VESSEL_GEOMETRY_PARAMETERS\n')
            self.file_proparams.write("    R0 = {:f}\n".format(self.R0))
            self.file_proparams.write("    EPSILON = {:f}\n".format(self.epsilon))
            self.file_proparams.write("    KAPPA = {:f}\n".format(self.kappa))
            self.file_proparams.write("    DELTA = {:f}\n".format(self.delta))
            self.file_proparams.write('END_VACUUM_VESSEL_GEOMETRY_PARAMETERS\n')
            self.file_proparams.write('\n')
            
            if 
                self.file_proparams.write('PLASMA_REGION_GEOMETRY_PARAMETERS\n')
                self.file_proparams.write("    CONTROL_POINTS = {:d}\n".format(self.CONTROL_POINTS))
                self.file_proparams.write("    R_SADDLE = {:f}\n".format(self.R_SADDLE))
                self.file_proparams.write("    Z_SADDLE = {:f}\n".format(self.Z_SADDLE))
                self.file_proparams.write("    R_RIGHTMOST = {:f}\n".format(self.R_RIGHTMOST))
                self.file_proparams.write("    Z_RIGHTMOST = {:f}\n".format(self.Z_RIGHTMOST))
                self.file_proparams.write("    R_LEFTMOST = {:f}\n".format(self.R_LEFTMOST))
                self.file_proparams.write("    Z_LEFTMOST = {:f}\n".format(self.Z_LEFTMOST))
                self.file_proparams.write("    R_TOP = {:f}\n".format(self.R_TOP))
                self.file_proparams.write("    Z_TOP = {:f}\n".format(self.Z_TOP))
                self.file_proparams.write('END_PLASMA_REGION_GEOMETRY_PARAMETERS\n')
                self.file_proparams.write('\n')
            
            if self.PLASMA_CURRENT == self.JARDIN_CURRENT:
                self.file_proparams.write('PLASMA_CURRENT_MODEL_PARAMETERS\n')
                self.file_proparams.write("    B0 = {:f}\n".format(self.B0))
                self.file_proparams.write("    q0 = {:f}\n".format(self.q0))
                self.file_proparams.write("    n_p = {:f}\n".format(self.n_p))
                self.file_proparams.write("    g0 = {:f}\n".format(self.G0))
                self.file_proparams.write("    n_g = {:f}\n".format(self.n_g))
                self.file_proparams.write("    TOTAL_PLASMA_CURRENT = {:f}\n".format(self.TOTAL_CURRENT))
                self.file_proparams.write('END_PLASMA_CURRENT_MODEL_PARAMETERS\n')
                self.file_proparams.write('\n')
            """
            
            """
            if not self.FIXED_BOUNDARY:
                self.file_proparams.write('EXTERNAL_COILS_PARAMETERS\n')
                self.file_proparams.write("    N_COILS = {:d}\n".format(len(self.COILS)))
                for COIL in self.COILS:
                    self.file_proparams.write("    Rposi = {:f}\n".format(COIL.X[0]))
                    self.file_proparams.write("    Zposi = {:f}\n".format(COIL.X[1]))
                    self.file_proparams.write("    Inten = {:f}\n".format(COIL.I))
                    self.file_proparams.write('\n')
                self.file_proparams.write('END_EXTERNAL_COILS_PARAMETERS\n')
                self.file_proparams.write('\n')
                
                self.file_proparams.write('EXTERNAL_SOLENOIDS_PARAMETERS\n')
                self.file_proparams.write("    N_SOLENOIDS = {:d}\n".format(len(self.SOLENOIDS)))
                for SOLENOID in self.SOLENOIDS:
                    self.file_proparams.write("    Rposi = {:f}\n".format(SOLENOID.Xe[0,0]))
                    self.file_proparams.write("    Zlow = {:f}\n".format(SOLENOID.Xe[0,1]))
                    self.file_proparams.write("    Zup = {:f}\n".format(SOLENOID.Xe[1,1]))
                    self.file_proparams.write("    Inten = {:f}\n".format(SOLENOID.I))
                    self.file_proparams.write("    Nturns = {:d}\n".format(SOLENOID.Nturns))
                    self.file_proparams.write('\n')
                self.file_proparams.write('END_EXTERNAL_SOLENOIDS_PARAMETERS\n')
                self.file_proparams.write('\n')
            """
            
            self.file_proparams.write('NUMERICAL_TREATMENT_PARAMETERS\n')
            self.file_proparams.write("    GHOST_STABILIZATION = {0}\n".format(self.GhostStabilization))
            self.file_proparams.write("    QUADRATURE_ORDER = {:d}\n".format(self.QuadratureOrder2D))
            self.file_proparams.write("    MAX_EXT_IT = {:d}\n".format(self.EXT_ITER))
            self.file_proparams.write("    EXT_TOL = {:e}\n".format(self.EXT_TOL))
            self.file_proparams.write("    MAX_INT_IT = {:d}\n".format(self.INT_ITER))
            self.file_proparams.write("    INT_TOL = {:e}\n".format(self.INT_TOL))
            self.file_proparams.write("    Beta = {:f}\n".format(self.beta))
            self.file_proparams.write("    Zeta = {:f}\n".format(self.zeta))
            self.file_proparams.write("    Lambda0 = {:f}\n".format(self.lambda0))
            self.file_proparams.write("    EXTR_R0 = {:f}\n".format(self.EXTR_R0))
            self.file_proparams.write("    EXTR_Z0 = {:f}\n".format(self.EXTR_Z0))
            self.file_proparams.write("    SADD_R0 = {:f}\n".format(self.SADD_R0))
            self.file_proparams.write("    SADD_Z0 = {:f}\n".format(self.SADD_Z0))
            self.file_proparams.write("    OPTI_ITMAX = {:d}\n".format(self.OPTI_ITMAX))
            self.file_proparams.write("    OPTI_TOL = {:f}\n".format(self.OPTI_TOL))
            self.file_proparams.write('END_NUMERICAL_TREATMENT_PARAMETERS\n')
            self.file_proparams.write('\n')
            
            self.file_proparams.write('OUTPUT_PARAMETERS\n')
            self.file_proparams.write("    OUT_PROPARAMS = {0}\n".format(self.out_proparams))
            self.file_proparams.write("    OUT_LSPLASMA = {0}\n".format(self.out_plasmaLS))
            self.file_proparams.write("    OUT_CLASELEMS = {0}\n".format(self.out_elemsClas))
            self.file_proparams.write("    OUT_APPROXPLASMA = {0}\n".format(self.out_plasmaapprox))
            self.file_proparams.write("    OUT_BCPLASMA = {0}\n".format(self.out_plasmaBC))
            self.file_proparams.write("    OUT_GHOSTFACES = {0}\n".format(self.out_ghostfaces))
            self.file_proparams.write("    OUT_ELEMSYSTEMS = {0}\n".format(self.out_elemsys))
            self.file_proparams.write('END_OUTPUT_PARAMETERS\n')
            self.file_proparams.write('\n')
            
            self.file_proparams.write('END_SIMULATION_PARAMETERS_FILE\n')
            self.file_proparams.close()
        return

    def writePSI(self):
        # WRITE PSI (CUTFEM SYSTEM SOLUTION)
        self.file_PSI.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
        for inode in range(self.Mesh.Nn):
            self.file_PSI.write("{:d} {:e}\n".format(inode+1,float(self.PSI[inode])))
        self.file_PSI.write('END_ITERATION\n')
        # WRITE NORMALISED PSI
        self.file_PSI_NORM.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
        for inode in range(self.Mesh.Nn):
            self.file_PSI_NORM.write("{:d} {:e}\n".format(inode+1,self.PSI_NORM[inode,1]))
        self.file_PSI_NORM.write('END_ITERATION\n')
        
        if self.out_pickle:
            self.PSIIt_sim.append((self.it_INT,self.it_EXT))
            self.PSI_sim.append(self.PSI.copy())
            self.PSI_NORM_sim.append(self.PSI_NORM[:,1].copy())
        return

    def writePSI_B(self):
        if not self.FIXED_BOUNDARY:
            self.file_PSI_B.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
        for inode in range(self.Mesh.Nnbound):
            self.file_PSI_B.write("{:d} {:e}\n".format(inode+1,self.PSI_B[inode,1]))
        if not self.FIXED_BOUNDARY:
            self.file_PSI_B.write('END_ITERATION\n')
            
        if self.out_pickle:
            self.PSI_B_sim.append(self.PSI_B[:,1].copy())
        return

    def writeresidu(self,which_loop):
        if which_loop == "INTERNAL":
            if self.it_INT == 1:
                self.file_RESIDU.write("INTERNAL_LOOP_STRUCTURE\n")
            self.file_RESIDU.write("  INTERNAL_ITERATION = {:d} \n".format(self.it_INT))
            self.file_RESIDU.write("      INTERNAL_RESIDU = {:f} \n".format(self.residu_INT))
            
            if self.out_pickle:
                self.Residu_sim.append(self.residu_INT)
            
        elif which_loop == "EXTERNAL":
            self.file_RESIDU.write("END_INTERNAL_LOOP_STRUCTURE\n")
            self.file_RESIDU.write("EXTERNAL_ITERATION = {:d} \n".format(self.it_EXT))
            self.file_RESIDU.write("  EXTERNAL_RESIDU = {:f} \n".format(self.residu_EXT))
            
            if self.out_pickle:
                self.Residu_sim.append(self.residu_EXT)
        return

    def writePSIcrit(self):
        if self.out_PSIcrit:
            self.file_PSIcrit.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            self.file_PSIcrit.write("{:f}  {:f}  {:f}  {:f}\n".format(self.Xcrit[1,0,-1],self.Xcrit[1,0,0],self.Xcrit[1,0,1],self.PSI_0))
            self.file_PSIcrit.write("{:f}  {:f}  {:f}  {:f}\n".format(self.Xcrit[1,1,-1],self.Xcrit[1,1,0],self.Xcrit[1,1,1],self.PSI_X))
            self.file_PSIcrit.write('END_ITERATION\n')
            
            if self.out_pickle:
                PSIcrit = np.concatenate((self.Xcrit[1,:,:],np.array([[self.PSI_0],[self.PSI_X]])),axis=1)
                self.PSIcrit_sim.append(PSIcrit)
        return

    def writeElementsClassification(self):
        MeshClassi = self.ObtainClassification()
        if self.out_elemsClas:
            if not self.FIXED_BOUNDARY:
                self.file_elemsClas.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            for ielem in range(self.Mesh.Ne):
                self.file_elemsClas.write("{:d} {:d}\n".format(ielem+1,MeshClassi[ielem]))
            if not self.FIXED_BOUNDARY:
                self.file_elemsClas.write('END_ITERATION\n')
                
        if self.out_pickle:
            self.MeshElements_sim.append(MeshClassi)
        return

    def writePlasmaLS(self):
        if self.out_plasmaLS:
            if not self.FIXED_BOUNDARY:
                self.file_plasmaLS.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            for inode in range(self.Mesh.Nn):
                self.file_plasmaLS.write("{:d} {:e}\n".format(inode+1,self.PlasmaLS[inode,1]))
            if not self.FIXED_BOUNDARY:
                self.file_plasmaLS.write('END_ITERATION\n')
                
        if self.out_pickle:
            self.PlasmaLS_sim.append(self.PlasmaLS[:,1].copy())
            self.PlasmaUpdateIt_sim.append(self.it)
        return

    def writePlasmaBC(self):
        if self.out_plasmaBC:
            if not self.FIXED_BOUNDARY:
                self.file_plasmaBC.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            for ielem in self.Mesh.PlasmaBoundElems:
                INTAPPROX = self.Mesh.Elements[ielem].InterfApprox
                for ig in range(INTAPPROX.ng):
                    self.file_plasmaBC.write("{:d} {:d} {:d} {:f} {:f} {:f}\n".format(self.Mesh.Elements[ielem].index,INTAPPROX.index,ig,INTAPPROX.Xg[ig,0],INTAPPROX.Xg[ig,1],INTAPPROX.PSIg[ig]))
            if not self.FIXED_BOUNDARY:
                self.file_plasmaBC.write("END_ITERATION\n")
        return

    def writePlasmaapprox(self):
        if self.out_plasmaapprox:
            if not self.FIXED_BOUNDARY:
                self.file_plasmaapprox.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            for ielem in self.Mesh.PlasmaBoundElems:
                INTAPPROX = self.Mesh.Elements[ielem].InterfApprox
                for inode in range(INTAPPROX.n):
                    self.file_plasmaapprox.write("{:d} {:d} {:d} {:f} {:f}\n".format(self.Mesh.Elements[ielem].index,INTAPPROX.index,inode,INTAPPROX.Xint[inode,0],INTAPPROX.Xint[inode,1]))
            if not self.FIXED_BOUNDARY:
                self.file_plasmaapprox.write("END_ITERATION\n")
                
        if self.out_pickle:
            plasmaapprox = np.zeros([len(self.Mesh.PlasmaBoundElems)*self.Mesh.Elements[self.Mesh.PlasmaBoundElems[0]].InterfApprox.n, 5])
            counter = 0
            for ielem in self.Mesh.PlasmaBoundElems:
                INTAPPROX = self.Mesh.Elements[ielem].InterfApprox
                for inode in range(INTAPPROX.n):
                    plasmaapprox[counter,0] = self.Mesh.Elements[ielem].index
                    plasmaapprox[counter,1] = INTAPPROX.index
                    plasmaapprox[counter,1] = inode
                    plasmaapprox[counter,1] = INTAPPROX.Xint[inode,0]
                    plasmaapprox[counter,1] = INTAPPROX.Xint[inode,1]
                    counter += 1
            self.PlasmaBoundApprox_sim.append(plasmaapprox)
        return

    def writeGhostFaces(self):
        if self.out_ghostfaces:
            if not self.FIXED_BOUNDARY:
                self.file_ghostfaces.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.it_EXT,self.it_INT))
            for FACE in self.Mesh.GhostFaces:
                self.file_ghostfaces.write("{:d} {:d} {:d} {:d}\n".format(FACE[1][0],FACE[1][1],FACE[2][0],FACE[2][1]))
            if not self.FIXED_BOUNDARY:
                self.file_ghostfaces.write("END_ITERATION\n")
                
        if self.GhostStabilization and self.out_pickle:
            self.PlasmaBoundGhostFaces_sim.append(self.Mesh.GhostFaces.copy())
        return

    def writePlasmaBoundaryData(self):
        if self.Mesh.PlasmaBoundElems.size > 0:
            self.writePlasmaLS()
            self.writeElementsClassification()
            self.writePlasmaapprox()
            self.writeGhostFaces()
        return


    def writeerror(self):
        self.file_L2error = open(self.outputdir+'/PSIerror.dat', 'w')
        
        AnaliticalNorm = np.zeros([self.Mesh.Nn])
        self.PSIerror = np.zeros([self.Mesh.Nn])
        self.PSIrelerror = np.zeros([self.Mesh.Nn])
        for inode in range(self.Mesh.Nn):
            AnaliticalNorm[inode] = self.PlasmaCurrent.PSIanalytical(self.Mesh.X[inode,:])
            self.PSIerror[inode] = abs(AnaliticalNorm[inode]-self.PSI_CONV[inode])
            self.PSIrelerror[inode] = self.PSIerror[inode]/abs(AnaliticalNorm[inode])
            if self.PSIerror[inode] < 1e-16:
                self.PSIerror[inode] = 1e-16
                self.PSIrelerror[inode] = 1e-16
                
        self.file_L2error.write('PSI_ERROR_FIELD\n')
        for inode in range(self.Mesh.Nn):
            self.file_L2error.write("{:d} {:e}\n".format(inode+1,self.PSIerror[inode])) 
        self.file_L2error.write('END_PSI_ERROR_FIELD\n')
        
        self.file_L2error.write("L2ERROR = {:e}".format(self.ErrorL2norm))

        self.file_L2error.close()
        return


    def writeSimulationPickle(self):
        if self.out_pickle:
            import pickle
            # RESTORE STATE OF OUTPUT FILES
            self.file_proparams = None           
            self.file_PSI = None                
            self.file_PSIcrit = None            
            self.file_PSI_NORM = None           
            self.file_PSI_B = None              
            self.file_RESIDU = None             
            self.file_elemsClas = None          
            self.file_plasmaLS = None           
            self.file_plasmaBC = None           
            self.file_plasmaapprox = None       
            self.file_ghostfaces = None         
            self.file_L2error = None            
            self.file_elemsys = None   
            self.file_globalsys = None    
                    
            # Serialize the simulation using Pickle
            with open(self.outputdir+'/'+self.CASE+'-'+self.MESH+'.pickle', 'wb') as pickle_file:
                pickle.dump(self, pickle_file)
        return