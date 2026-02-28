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


# This script contains the definition for class InterfaceApprox, an object representing
# the elemental approximation of an interface parametrised by a level-set function.

class InterfaceApprox:
    """
    Class representing an interface approximation consisting of a collection of segments.
    This class manages the properties and methods related to the interface, including its 
    segmentation and nodal coordinates in both physical and reference space.
    """
    
    def __init__(self,index,n,Xint,XIint,Tint,ElIntNodes):
        """
        Initializes the interface approximation with the specified global index and number of segments.

        Input:
            - index (int): Global index of the element in the computational mesh.
            - Nsegments (int): Number of segments conforming the interface approximation.
        """
        
        self.index = index            # GLOBAL INDEX OF INTERFACE APPROXIMATION
        self.n = n                    # NODES
        self.Xint = Xint              # INTERFACE APPROXIMATION NODAL COORDINATES MATRIX (PHYSICAL SPACE)
        self.XIint = XIint            # INTERFACE APPROXIMATION NODAL COORDINATES MATRIX (REFERENCE SPACE)
        self.Tint = Tint              # INTERFACE APPROXIMATION SEGMENTS CONNECTIVITY 
        self.ElIntNodes = ElIntNodes  # ELEMENTAL VERTICES INDEXES ON EDGES CUTING THE INTERFACE 
        self.PSIg = None              # PSI VALUE ON SEGMENT GAUSS INTEGRATION NODES
        
        # QUADRATURE FOR INTEGRATION ALONG INTERFACE 
        self.ng = None              # NUMBER OF GAUSS INTEGRATION NODES 
        self.Wg = None              # GAUSS INTEGRATION WEIGHTS 
        self.XIg = None             # GAUSS INTEGRATION NODAL COORDINATES (REFERENCE SPACE)
        self.Xg = None              # GAUSS INTEGRATION NODAL COORDINATES (PHYSICAL SPACE)
        self.Ng = None              # REFERENCE SHAPE FUNCTIONS EVALUATED AT GAUSS INTEGRATION NODES 
        self.dNdxig = None          # REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT GAUSS INTEGRATION NODES 
        self.dNdetag = None         # REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO ETA EVALUATED AT GAUSS INTEGRATION NODES
        self.invJg = None
        self.detJg = None           # MATRIX DETERMINANTS OF JACOBIAN OF TRANSFORMATION FROM 1D REFERENCE ELEMENT TO 2D PHYSICAL
        self.detJg1D = None
        
        # NORMAL VECTOR (OUTWARDS RESPECT TO BOUNDARY)
        self.NormalVec = None       # NORMAL VECTOR AT GAUSS INTEGRATION NODE (PHYSICAL SPACE)
        self.NormalVecREF = None 
        return