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
# Date: October 2024
# Institution: Barcelona Supercomputing Center (BSC)
# Department: Computer Applications in Science and Engineering (CASE)
# Research Group: Nuclear Fusion  

# This script constitutes the test-suite for the FIXED-boundary plasma equilibrium problem 
# where the plasma current is modelled using an expression depending on profiles *p* (plasma pressure) 
# and *g* (toroidal function). There is no analytical solution for this case. 

# After selecting the MESH, the file may be executed to launch the solver. EQUILIPY's output 
# can be turned ON and OFF by change the bolean output parameters.


import sys
sys.path.append('../src/')

from GradShafranovSolver import *

### SELECT MESH FOLDER...
###### LINEAR TRIANGULAR ELEMENT MESH
#MESH = 'TRI03-COARSE'
#MESH = 'TRI03-MEDIUM'
MESH = 'TRI03-INTERMEDIATE'
#MESH = 'TRI03-FINE'
#MESH = 'TRI03-SUPERFINE'
#MESH = 'TRI03-MEGAFINE'
#MESH = 'TRI03-ULTRAFINE'

###### QUADRATIC TRIANGULAR ELEMENT MESH
#MESH = 'TRI06-COARSE'
#MESH = 'TRI06-MEDIUM'
#MESH = 'TRI06-INTERMEDIATE'
#MESH = 'TRI06-FINE'
#MESH = 'TRI06-SUPERFINE'
#MESH = 'TRI06-MEGAFINE'
#MESH = 'TRI06-ULTRAFINE'

###### CUBIC TRIANGULAR ELEMENT MESH
#MESH = 'TRI10-COARSE'
#MESH = 'TRI10-MEDIUM'
#MESH = 'TRI10-INTERMEDIATE'
#MESH = 'TRI10-FINE'
#MESH = 'TRI10-SUPERFINE'


### SELECT SOLUTION CASE FILE:
CASE = 'TS-FREE-PROFILES'   
#CASE = 'TS-FREE-PROFILES-RECTANGLE'

##############################################################

## CREATE GRAD-SHAFRANOV PROBLEM 
Problem = GradShafranovSolver(MESH,CASE)
## DECLARE SWITCHS:
##### GHOST PENALTY STABILISATION
Problem.GhostStabilization = True
##### OUTPUT PLOTS IN RUNTIME
Problem.plotelemsClas = False      # OUTPUT SWITCH FOR ELEMENTS CLASSIFICATION PLOTS AT EACH ITERATION
Problem.plotout_PSI = True         # OUTPUT SWITCH FOR PSI SOLUTION PLOTS AT EACH ITERATION
##### OUTPUT FILES
Problem.out_proparams = True       # OUTPUT SWITCH FOR SIMULATION PARAMETERS 
Problem.out_elemsClas = True       # OUTPUT SWITCH FOR CLASSIFICATION OF MESH ELEMENTS
Problem.out_plasmaLS = True        # OUTPUT SWITCH FOR PLASMA BOUNDARY LEVEL-SET FIELD VALUES
Problem.out_plasmaBC = True        # OUTPUT SWITCH FOR PLASMA BOUNDARY CONDITION VALUES 
Problem.out_plasmaapprox = True    # OUTPUT SWITCH FOR PLASMA BOUNDARY APPROXIMATION DATA 
Problem.out_ghostfaces = True      # OUTPUT SWITCH FOR GHOST STABILISATION FACES DATA 
Problem.out_elemsys = False        # OUTPUT SWITCH FOR ELEMENTAL MATRICES
##### OUTPUT PICKLING
Problem.out_pickle = True          # OUTPUT SWITCH FOR SIMULATION DATA PYTHON PICKLE

##############################################################

## COMPUTE PLASMA EQUILIBRIUM
Problem.EQUILI()