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
import os
from shutil import copy2

##################################################################################################
####################################### EQUILIPY OUTPUT ##########################################
##################################################################################################

class EquilipyOutput:
    
    def __init__(self):
        
        # INITIATE OUTPUT SWITCHES
        self.out_proparams = False          # SIMULATION PARAMETERS 
        self.out_boundaries = False
        self.out_elemsClas = False          # CLASSIFICATION OF MESH ELEMENTS
        self.out_plasmaLS = False           # PLASMA BOUNDARY LEVEL-SET FIELD VALUES 
        self.out_plasmaBC = False           # PLASMA BOUNDARY CONDITION VALUES 
        self.out_plasmaapprox = False       # PLASMA BOUNDARY APPROXIMATION DATA 
        self.out_ghostfaces = False         # GHOST STABILISATION FACES DATA 
        self.out_quadratures = False
        self.out_elemsys = False            # ELEMENTAL MATRICES
        self.plotelemsClas = False          # ELEMENTS CLASSIFICATION PLOTS AT EACH ITERATION
        self.out_PSIcrit = False
        self.plotPSI = False                # PSI SOLUTION PLOTS AT EACH ITERATION
        self.out_pickle = False             # SIMULATION DATA PYTHON PICKLE
        
        # INITIATE OUTPUT FILES
        self.outputdir = None
        self.file_proparams = None          # OUTPUT FILE CONTAINING THE SIMULATION PARAMETERS 
        self.file_boundaries = None
        self.file_elemsClas = None          # OUTPUT FILE CONTAINING THE CLASSIFICATION OF MESH ELEMENTS
        self.file_elemgroups = None
        self.file_plasmaLS = None           # OUTPUT FILE CONTAINING THE PLASMA BOUNDARY LEVEL-SET FIELD VALUES
        self.file_plasmaBC = None           # OUTPUT FILE CONTAINING THE PLASMA BOUNDARY CONDITION VALUES
        self.file_plasmaapprox = None       # OUTPUT FILE CONTAINING THE PLASMA BOUNDARY APPROXIMATION DATA
        self.file_ghostfaces = None         # OUTPUT FILE CONTAINING THE GHOST STABILISATION FACES DATA
        self.file_quadratures = None
        self.file_elemsys = None            # OUTPUT FILE CONTAINING THE ELEMENTAL MATRICES FOR EACH ITERATION
        self.file_globalsys = None
        self.file_PSI = None                # OUTPUT FILE CONTAINING THE PSI FIELD VALUES OBTAINED BY SOLVING THE CutFEM SYSTEM
        self.file_PSIcrit = None            # OUTPUT FILE CONTAINING THE CRITICAL PSI VALUES
        self.file_PSI_NORM = None           # OUTPUT FILE CONTAINING THE PSI_NORM FIELD VALUES (AFTER NORMALISATION OF PSI FIELD)
        self.file_PSI_B = None              # OUTPUT FILE CONTAINING THE PSI_B BOUNDARY VALUES
        self.file_RESIDU = None             # OUTPUT FILE CONTAINING THE RESIDU FOR EACH ITERATION
        self.file_PSIerror = None           # OUTPUT FILE CONTAINING THE ERROR FIELD AND THE L2 ERROR NORM FOR THE CONVERGED SOLUTION 
        
        super().__init__()
        return
    
    def InitialiseOutput(self):
        
        # Check if the directory exists
        if self.FIXED_BOUNDARY:
            bound = 'FIXED_BOUNDARY'
        else:
            bound = 'FREE_BOUNDARY'
        self.outputdir = self.pwd + '/RESULTS/'+bound+'/' + self.CASE + '-' + self.MESH.name
        if not os.path.exists(self.outputdir):
            # Create the directory
            os.makedirs(self.outputdir)
        # COPY SIMULATION FILES
        self.copysimfiles()
        # WRITE SIMULATION PARAMETERS FILE (IF ON)
        self.writeparams() 
        return


    def openOUTPUTfiles(self):
        """
        Open files for selected output. 
        """
        if self.out_boundaries:
            self.file_boundaries = open(self.outputdir+'/Boundaries.dat', 'w')
            self.file_boundaries.write('COMPUTATIONAL_DOMAIN_BOUNDARIES_FILE\n')
            
        if self.out_plasmaLS:
            self.file_plasmaLS = open(self.outputdir+'/PlasmaBoundLS.dat', 'w')
            self.file_plasmaLS.write('PLASMA_BOUNDARY_LEVEL_SET_FILE\n')
            
        if self.out_elemsClas:
            self.file_elemsClas = open(self.outputdir+'/ElemsClassification.dat', 'w')
            self.file_elemsClas.write('MESH_ELEMENTS_CLASSIFICATION_FILE\n')
            
            self.file_elemgroups = open(self.outputdir+'/ElementGroups.dat', 'w')
            self.file_elemgroups.write('ELEMENT_GROUPS_FILE\n')
            
        if self.out_plasmaapprox:
            self.file_plasmaapprox = open(self.outputdir+'/PlasmaBoundApprox.dat', 'w')
            self.file_plasmaapprox.write('PLASMA_BOUNDARY_APPROXIMATION_FILE\n')
            
        if self.out_plasmaBC:
            self.file_plasmaBC = open(self.outputdir+'/PlasmaBC.dat', 'w')
            self.file_plasmaBC.write('PLASMA_BOUNDARY_CONSTRAINT_VALUES_FILE\n')
            
        if self.out_quadratures:
            self.file_quadratures = open(self.outputdir+'/Quadratures.dat','w')
            self.file_quadratures.write("NUMERICAL_INTEGRATION_QUADRATURES_FILE\n")
            
        if self.out_ghostfaces:
            self.file_ghostfaces = open(self.outputdir+'/GhostFaces.dat', 'w')
            self.file_ghostfaces.write('GHOST_STABILIZATION_FACES_FILE\n')
            
        if self.out_elemsys:
            self.file_elemsys = open(self.outputdir+'/ElemMatrices.dat', 'w')
            self.file_elemsys.write('ELEMENTAL_MATRICES_FILE\n')
            
            self.file_globalsys = open(self.outputdir+'/GlobalMatrices.dat', 'w')
            self.file_globalsys.write('GLOBAL_MATRICES_FILE\n')
            
        
        self.file_PSI = open(self.outputdir+'/PSI.dat', 'w')
        self.file_PSI.write('PSI_FIELD_FILE\n')
            
        self.file_PSI_NORM = open(self.outputdir+'/PSI_NORM.dat', 'w')
        self.file_PSI_NORM.write('PSI_NORM_FIELD_FILE\n')
        
        if self.out_PSIcrit:
            self.file_PSIcrit = open(self.outputdir+'/PSIcrit.dat', 'w')
            self.file_PSIcrit.write('PSIcrit_VALUES_FILE\n')

        self.file_PSI_B = open(self.outputdir+'/PSI_B.dat', 'w')
        self.file_PSI_B.write('PSI_B_VALUES_FILE\n')
        
        self.file_RESIDU = open(self.outputdir+'/Residu.dat', 'w')
        self.file_RESIDU.write('RESIDU_VALUES_FILE\n')
        
        return


    def closeOUTPUTfiles(self):
        """
        Close files for selected output.
        """ 
        if self.out_boundaries:
            self.file_boundaries.write('END_COMPUTATIONAL_DOMAIN_BOUNDARIES_FILE')
            self.file_boundaries.close()
            
        if self.out_elemsClas:
            self.file_elemsClas.write('END_MESH_ELEMENTS_CLASSIFICATION_FILE')
            self.file_elemsClas.close()
            
            self.file_elemgroups.write('END_ELEMENT_GROUPS_FILE')
            self.file_elemgroups.close()
        
        if self.out_plasmaLS:
            self.file_plasmaLS.write('END_PLASMA_BOUNDARY_LEVEL_SET_FILE')
            self.file_plasmaLS.close()
        
        if self.out_plasmaapprox:
            self.file_plasmaapprox.write('END_PLASMA_BOUNDARY_APPROXIMATION_FILE')
            self.file_plasmaapprox.close()
            
        if self.out_plasmaBC:
            self.file_plasmaBC.write('END_PLASMA_BOUNDARY_CONSTRAINT_VALUES_FILE')
            self.file_plasmaBC.close()
            
        if self.out_ghostfaces:
            self.file_ghostfaces.write('END_GHOST_STABILISATION_FACES_FILE')
            self.file_ghostfaces.close()
            
        if self.out_quadratures:
            self.file_quadratures.write("END_NUMERICAL_INTEGRATION_QUADRATURES_FILE")
            self.file_quadratures.close()
            
        
        if self.out_elemsys:
            self.file_elemsys.write('END_ELEMENTAL_MATRICES_FILE')
            self.file_elemsys.close()
            
            self.file_globalsys.write('END_GLOBAL_MATRICES_FILE')
            self.file_globalsys.close()
            
        self.file_PSI.write('END_PSI_FIELD_FILE')
        self.file_PSI.close()
        
        self.file_PSI_NORM.write('END_PSI_NORM_FIELD_FILE')
        self.file_PSI_NORM.close()
        
        if self.out_PSIcrit:
            self.file_PSIcrit.write('END_PSIcrit_VALUES_FILE')
            self.file_PSIcrit.close()
        
        self.file_PSI_B.write('END_PSI_B_VALUES_FILE')
        self.file_PSI_B.close()
        
        self.file_RESIDU.write('END_RESIDU_VALUES_FILE')
        self.file_RESIDU.close()
            
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
        MeshDataFile = self.MESH.directory +'/' + self.MESH.name +'.dom.dat'
        copy2(MeshDataFile,self.outputdir+'/'+self.CASE+'-'+self.MESH.name+'.dom.dat')
        # COPY GEO.DAT FILE
        MeshFile = self.MESH.directory +'/' + self.MESH.name +'.geo.dat'
        copy2(MeshFile,self.outputdir+'/'+self.CASE+'-'+self.MESH.name+'.geo.dat')
        
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
            self.file_proparams.write("    NPOIN = {:d}\n".format(self.MESH.Nn))
            self.file_proparams.write("    NELEM = {:d}\n".format(self.MESH.Ne))
            self.file_proparams.write("    ELEM = {:d}\n".format(self.MESH.ElTypeALYA))
            self.file_proparams.write("    NBOUN = {:d}\n".format(self.MESH.Nbound))
            self.file_proparams.write("    DIM = {:d}\n".format(self.MESH.dim))
            self.file_proparams.write('END_MESH_PARAMETERS\n')
            self.file_proparams.write('\n')
            
            self.file_proparams.write('PROBLEM_TYPE_PARAMETERS\n') 
            self.file_proparams.write("    PLASMA_BOUNDARY = {}".format(str(self.FIXED_BOUNDARY)))
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
            self.file_proparams.write("    QUADRATURE_ORDER_2D = {:d}\n".format(self.QuadratureOrder2D))
            self.file_proparams.write("    QUADRATURE_ORDER_1D = {:d}\n".format(self.QuadratureOrder1D))
            self.file_proparams.write("    MAX_EXT_IT = {:d}\n".format(self.ext_maxiter))
            self.file_proparams.write("    EXT_TOL = {:e}\n".format(self.ext_tol))
            self.file_proparams.write("    MAX_INT_IT = {:d}\n".format(self.int_maxiter))
            self.file_proparams.write("    INT_TOL = {:e}\n".format(self.int_tol))
            self.file_proparams.write("    BETA = {:f}\n".format(self.beta))
            self.file_proparams.write("    ZETA = {:f}\n".format(self.zeta))
            self.file_proparams.write("    R0_axis = {:f}\n".format(self.R0_axis))
            self.file_proparams.write("    Z0_axis = {:f}\n".format(self.Z0_axis))
            self.file_proparams.write("    R0_saddle = {:f}\n".format(self.R0_saddle))
            self.file_proparams.write("    Z0_saddle = {:f}\n".format(self.Z0_saddle))
            self.file_proparams.write("    OPTI_ITMAX = {:d}\n".format(self.opti_maxiter))
            self.file_proparams.write("    OPTI_TOL = {:f}\n".format(self.opti_tol))
            self.file_proparams.write('END_NUMERICAL_TREATMENT_PARAMETERS\n')
            self.file_proparams.write('\n')
            
            self.file_proparams.write('OUTPUT_PARAMETERS\n')
            self.file_proparams.write("    OUT_PROPARAMS = {0}\n".format(self.out_proparams))
            self.file_proparams.write("    OUT_BOUNDARIES = {0}\n".format(self.out_boundaries))
            self.file_proparams.write("    OUT_LSPLASMA = {0}\n".format(self.out_plasmaLS))
            self.file_proparams.write("    OUT_BCPLASMA = {0}\n".format(self.out_plasmaBC))
            self.file_proparams.write("    OUT_CLASELEMS = {0}\n".format(self.out_elemsClas))
            self.file_proparams.write("    OUT_APPROXPLASMA = {0}\n".format(self.out_plasmaapprox))
            self.file_proparams.write("    OUT_GHOSTFACES = {0}\n".format(self.out_ghostfaces))
            self.file_proparams.write("    OUT_QUADRATURES = {0}\n".format(self.out_quadratures))
            self.file_proparams.write("    OUT_ELEMSYSTEMS = {0}\n".format(self.out_elemsys))
            self.file_proparams.write('END_OUTPUT_PARAMETERS\n')
            self.file_proparams.write('\n')
            
            self.file_proparams.write('END_SIMULATION_PARAMETERS_FILE')
            self.file_proparams.close()
        return
    
    
    def writeboundaries(self):
        if self.out_boundaries:
            self.file_boundaries.write("BOUNDARY_NODES (NNBOUN = {:d})\n".format(self.MESH.Nnbound))
            for inode, nodeindex in enumerate(self.MESH.BoundaryNodes):
                self.file_boundaries.write("{:d} {:d}\n".format(inode, nodeindex))
            self.file_boundaries.write("END_BOUNDARY_NODES\n")
            self.file_boundaries.write("BOUNDARY_CONNECTIVITIES (NBOUN = {:d})\n".format(self.MESH.Nbound))
            kboun = 0
            for ielem in self.MESH.BoundaryElems: 
                ELEM = self.MESH.Elements[ielem]
                for iboun in range(np.shape(ELEM.Teboun)[0]):
                    kboun += 1
                    values = " ".join(str(val) for val in ELEM.Teboun[iboun, :])
                    self.file_boundaries.write("{:d} {:d} {}\n".format(kboun, ELEM.index, values))
            self.file_boundaries.write("END_BOUNDARY_CONNECTIVITIES\n")
            self.file_boundaries.write("DIRICHLET_ELEMENTS (NDIRICH = {:d})\n".format(len(self.MESH.DirichletElems)))
            kboun = 0
            for ielem in self.MESH.DirichletElems: 
                ELEM = self.MESH.Elements[ielem]
                for iboun in range(np.shape(ELEM.Teboun)[0]):
                    kboun += 1
                    values = " ".join(str(val) for val in ELEM.Teboun[iboun])
                    self.file_boundaries.write("{:d} {:d} {}\n".format(kboun, ELEM.index, values))
            self.file_boundaries.write("END_DIRICHLET_ELEMENTS\n")
        return
    
    
    def writeElementsClassification(self):
        MeshClassi = self.MESH.ObtainClassification()
        if self.out_elemsClas:
            if not self.FIXED_BOUNDARY:
                self.file_elemsClas.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.ext_it,self.int_it))
                self.file_elemgroups.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.ext_it,self.int_it))
            
            for ielem in range(self.MESH.Ne):
                self.file_elemsClas.write("{:d} {:d}\n".format(ielem+1,MeshClassi[ielem]))
            
            self.file_elemgroups.write("PLASMA_ELEMENTS (N = {:d})\n".format(len(self.MESH.PlasmaElements)))
            for ielem, elemindex in enumerate(self.MESH.PlasmaElements):
                self.file_elemgroups.write("{d:} {:d}\n".format(ielem+1, elemindex))
            self.file_elemgroups.write("END_PLASMA_ELEMENTS\n")
            
            self.file_elemgroups.write("PLASMA_BOUNDARY_ELEMENTS (N = {:d})\n".format(len(self.MESH.PlasmaBoundaryElements)))
            for ielem, elemindex in enumerate(self.MESH.PlasmaBoundaryElements):
                self.file_elemgroups.write("{d:} {:d}\n".format(ielem+1, elemindex))
            self.file_elemgroups.write("END_PLASMA_BOUNDARY_ELEMENTS\n")
            
            self.file_elemgroups.write("VACUUM_ELEMENTS (N = {:d})\n".format(len(self.MESH.VacuumElements)))
            for ielem, elemindex in enumerate(self.MESH.VacuumElements):
                self.file_elemgroups.write("{d:} {:d}\n".format(ielem+1, elemindex))
            self.file_elemgroups.write("END_VACUUM_ELEMENTS\n")
            
            self.file_elemgroups.write("BOUNDARY_ELEMENTS (N = {:d})\n".format(len(self.MESH.BoundaryElements)))
            for ielem, elemindex in enumerate(self.MESH.BoundaryElements):
                self.file_elemgroups.write("{d:} {:d}\n".format(ielem+1, elemindex))
            self.file_elemgroups.write("END_BOUNDARY_ELEMENTS\n")
            
            self.file_elemgroups.write("NON_CUT_ELEMENTS (N = {:d})\n".format(len(self.MESH.NonCutElements)))
            for ielem, elemindex in enumerate(self.MESH.NonCutElements):
                self.file_elemgroups.write("{d:} {:d}\n".format(ielem+1, elemindex))
            self.file_elemgroups.write("END_NON_CUT_ELEMENTS\n")
            
            if not self.FIXED_BOUNDARY:
                self.file_elemsClas.write('END_ITERATION\n')
                self.file_elemgroups.write('END_ITERATION\n')
                
        if self.out_pickle:
            self.MeshElements_sim.append(MeshClassi)
        return

    def writePlasmaLS(self):
        if self.out_plasmaLS:
            if not self.FIXED_BOUNDARY:
                self.file_plasmaLS.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.ext_it,self.int_it))
            for inode in range(self.MESH.Nn):
                self.file_plasmaLS.write("{:d} {:e}\n".format(inode+1,self.PlasmaLS[inode]))
            if not self.FIXED_BOUNDARY:
                self.file_plasmaLS.write('END_ITERATION\n')
                
        if self.out_pickle:
            self.PlasmaLS_sim.append(self.PlasmaLS.copy())
            self.PlasmaUpdateIt_sim.append(self.it)
        return

    def writePlasmaBC(self):
        if self.out_plasmaBC:
            if not self.FIXED_BOUNDARY:
                self.file_plasmaBC.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.ext_it,self.int_it))
            for ielem in self.MESH.PlasmaBoundActiveElems:
                INTAPPROX = self.MESH.Elements[ielem].InterfApprox
                for ig in range(INTAPPROX.ng):
                    self.file_plasmaBC.write("{:d} {:d} {:d} {:f} {:f} {:f}\n".format(self.MESH.Elements[ielem].index,INTAPPROX.index,ig,INTAPPROX.Xg[ig,0],INTAPPROX.Xg[ig,1],INTAPPROX.PSIg[ig]))
            if not self.FIXED_BOUNDARY:
                self.file_plasmaBC.write("END_ITERATION\n")
        return

    def writePlasmaapprox(self):
        if self.out_plasmaapprox:
            if not self.FIXED_BOUNDARY:
                self.file_plasmaapprox.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.ext_it,self.int_it))
            self.file_plasmaapprox.write("NNPB = {:d}\n".format(self.MESH.NnPB))
            for ielem in self.MESH.PlasmaBoundActiveElems:
                INTAPPROX = self.MESH.Elements[ielem].InterfApprox
                for inode in range(INTAPPROX.n):
                    self.file_plasmaapprox.write("{:d} {:d} {:f} {:f}\n".format(self.MESH.Elements[ielem].index,inode,INTAPPROX.Xint[inode,0],INTAPPROX.Xint[inode,1]))
            if not self.FIXED_BOUNDARY:
                self.file_plasmaapprox.write("END_ITERATION\n")
                
        if self.out_pickle:
            plasmaapprox = np.zeros([len(self.MESH.PlasmaBoundActiveElems)*self.MESH.Elements[self.MESH.PlasmaBoundActiveElems[0]].InterfApprox.n, 5])
            counter = 0
            for ielem in self.MESH.PlasmaBoundActiveElems:
                INTAPPROX = self.MESH.Elements[ielem].InterfApprox
                for inode in range(INTAPPROX.n):
                    plasmaapprox[counter,0] = self.MESH.Elements[ielem].index
                    plasmaapprox[counter,1] = INTAPPROX.index
                    plasmaapprox[counter,1] = inode
                    plasmaapprox[counter,1] = INTAPPROX.Xint[inode,0]
                    plasmaapprox[counter,1] = INTAPPROX.Xint[inode,1]
                    counter += 1
            self.PlasmaBoundApprox_sim.append(plasmaapprox)
        return
    

    def writePSI(self):
        # WRITE PSI (CUTFEM SYSTEM SOLUTION)
        self.file_PSI.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.ext_it,self.int_it))
        for inode in range(self.MESH.Nn):
            self.file_PSI.write("{:d} {:e}\n".format(inode+1,float(self.PSI[inode])))
        self.file_PSI.write('END_ITERATION\n')
        # WRITE NORMALISED PSI
        self.file_PSI_NORM.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.ext_it,self.int_it))
        for inode in range(self.MESH.Nn):
            self.file_PSI_NORM.write("{:d} {:e}\n".format(inode+1,self.PSI_NORM[inode,1]))
        self.file_PSI_NORM.write('END_ITERATION\n')
        
        if self.out_pickle:
            self.PSIIt_sim.append((self.int_it,self.ext_it))
            self.PSI_sim.append(self.PSI.copy())
            self.PSI_NORM_sim.append(self.PSI_NORM[:,1].copy())
        return

    def writePSI_B(self):
        if not self.FIXED_BOUNDARY:
            self.file_PSI_B.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.ext_it,self.int_it))
        for inode in range(self.MESH.Nnbound):
            self.file_PSI_B.write("{:d} {:e}\n".format(inode+1,self.PSI_B[inode,1]))
        if not self.FIXED_BOUNDARY:
            self.file_PSI_B.write('END_ITERATION\n')
            
        if self.out_pickle:
            self.PSI_B_sim.append(self.PSI_B[:,1].copy())
        return

    def writeresidu(self,which_loop):
        if which_loop == "INTERNAL":
            if self.int_it == 1:
                self.file_RESIDU.write("INTERNAL_LOOP_STRUCTURE\n")
            self.file_RESIDU.write("  INTERNAL_ITERATION = {:d} \n".format(self.int_it))
            self.file_RESIDU.write("      INTERNAL_RESIDU = {:f} \n".format(self.int_residu))
            
            if self.out_pickle:
                self.Residu_sim.append(self.int_residu)
            
        elif which_loop == "EXTERNAL":
            self.file_RESIDU.write("END_INTERNAL_LOOP_STRUCTURE\n")
            self.file_RESIDU.write("EXTERNAL_ITERATION = {:d} \n".format(self.ext_it))
            self.file_RESIDU.write("  EXTERNAL_RESIDU = {:f} \n".format(self.ext_residu))
            
            if self.out_pickle:
                self.Residu_sim.append(self.ext_residu)
        return

    def writePSIcrit(self):
        if self.out_PSIcrit:
            self.file_PSIcrit.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.ext_it,self.int_it))
            self.file_PSIcrit.write("{:f}  {:f}  {:f}  {:f}\n".format(self.Xcrit[1,0,-1],self.Xcrit[1,0,0],self.Xcrit[1,0,1],self.PSI_0))
            self.file_PSIcrit.write("{:f}  {:f}  {:f}  {:f}\n".format(self.Xcrit[1,1,-1],self.Xcrit[1,1,0],self.Xcrit[1,1,1],self.PSI_X))
            self.file_PSIcrit.write('END_ITERATION\n')
            
            if self.out_pickle:
                PSIcrit = np.concatenate((self.Xcrit[1,:,:],np.array([[self.PSI_0],[self.PSI_X]])),axis=1)
                self.PSIcrit_sim.append(PSIcrit)
        return

    

    def writeGhostFaces(self):
        if self.out_ghostfaces:
            if not self.FIXED_BOUNDARY:
                self.file_ghostfaces.write("ITERATION {:d} (EXT_it = {:d}, INT_it = {:d})\n".format(self.it,self.ext_it,self.int_it))
            self.file_ghostfaces.write("GHOST_ELEMENTS (NGE = {:d})\n".format(len(self.MESH.GhostElems)))
            for ielem, elemindex in enumerate(self.MESH.GhostElems):
                self.file_ghostfaces.write("{:d} {:d}".format(ielem+1, elemindex))
            self.file_ghostfaces.write("END_GHOST_ELEMENTS\n")
            self.file_ghostfaces.write("GHOST_FACES (NGF = {:d})\n".format(len(self.MESH.GhostFaces)))
            for iface, FACE in enumerate(self.MESH.GhostFaces):
                self.file_ghostfaces.write("{:d} {:d} {:d} {:d} {:d}\n".format(iface+1, FACE[1][0], FACE[1][1], FACE[2][0], FACE[2][1]))
            self.file_ghostfaces.write("END_GHOST_FACES\n")
            if not self.FIXED_BOUNDARY: 
                self.file_ghostfaces.write("END_ITERATION\n")
                
        if self.GhostStabilization and self.out_pickle:
            self.PlasmaBoundGhostFaces_sim.append(self.MESH.GhostFaces.copy())
        return

    def writePlasmaBoundaryData(self):
        if self.MESH.PlasmaBoundElems.size > 0:
            self.writePlasmaLS()
            self.writeElementsClassification()
            self.writePlasmaapprox()
            self.writeGhostFaces()
        return


    def writeerror(self):
        self.file_PSIerror = open(self.outputdir+'/PSIerror.dat', 'w')
        self.file_PSIerror.write('PSI_ERROR_FILE\n')
        self.file_PSIerror.write('PSI_ERROR_FIELD\n')
        for inode in range(self.MESH.Nn):
            self.file_PSIerror.write("{:d} {:e}\n".format(inode+1,self.PSIerror[inode])) 
        self.file_PSIerror.write('END_PSI_ERROR_FIELD\n')
        
        self.file_PSIerror.write("EUCLIDEAN_ERROR = {:e}\n".format(self.ErrorEuclinorm))
        self.file_PSIerror.write("L2ERROR = {:e}\n".format(self.ErrorL2norm))
        self.file_PSIerror.write("RelL2ERROR = {:e}\n".format(self.RelErrorL2norm))
        self.file_PSIerror.write('END_PSI_ERROR_FILE\n')
        self.file_PSIerror.close()
        return


    def writeSimulationPickle(self):
        if self.out_pickle:
            import pickle
            self.file_proparams = None
            self.file_boundaries = None                      
            self.file_elemsClas = None 
            self.file_elemgroups = None         
            self.file_plasmaLS = None           
            self.file_plasmaBC = None           
            self.file_plasmaapprox = None       
            self.file_ghostfaces = None         
            self.file_quadratures = None            
            self.file_elemsys = None   
            self.file_globalsys = None 
            self.file_PSI = None                
            self.file_PSIcrit = None            
            self.file_PSI_NORM = None           
            self.file_PSI_B = None              
            self.file_RESIDU = None
            self.file_PSIerror = None     
                    
            # Serialize the simulation using Pickle
            with open(self.outputdir+'/'+self.CASE+'-'+self.MESH.name+'.pickle', 'wb') as pickle_file:
                pickle.dump(self, pickle_file)
        return