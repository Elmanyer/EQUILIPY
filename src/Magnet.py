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



from Greens import *

class Coil:
    """
    Class representing a tokamak's external coil (confinement magnet).
    """
    
    def __init__(self,name,dim,X,I):
        """
        Constructor to initialize the Coil object with the provided attributes.

        Input:
            - index (int): The index of the coil in the global system.
            - dim (int): The spatial dimension of the coil coordinates.
            - X (numpy.ndarray): A 1D array representing the position coordinates of the coil in physical space.
            - I (float): The current carried by the coil.
        """
        
        self.name = name        # IDENTIFICATION
        self.dim = dim          # SPATIAL DIMENSION
        self.X = X              # POSITION COORDINATES
        self.I = I              # CURRENT
        
        return
    
    def Psi(self,X):
        """
        Calculate poloidal flux psi at X=(R,Z) due to coil
        """
        return GreensFunction(self.X,X) * self.I 
    
    def Br(self,X):
        """
        Calculate radial magnetic field Br at X=(R,Z)
        """
        return GreensBr(self.X,X) * self.I 

    def Bz(self,X):
        """
        Calculate vertical magnetic field Bz at X=(R,Z)
        """
        return GreensBz(self.X,X) * self.I 
    
    
class ShapedCoil:
    
    def __init__(self, shape, current=0.0, Nturns=1, ElOrder=2):
        
        # Find the geometric middle of the coil
        # The R,Z properties have accessor functions to handle modifications
        self.Rcentre = sum(r for r, z in shape) / len(shape)
        self.Zcentre = sum(z for r, z in shape) / len(shape)

        self.current = current
        self.Nturns = Nturns
        self.shape = shape
        
        
        return
    
    
    def triangulate(self):
        """
        Use the ear clipping method to turn an arbitrary polygon into triangles

        Input

        polygon   [ (r1, z1), (r2, z2), ... ]
        """

        nvert = len(self.shape)  # Number of vertices
        assert nvert > 2

        triangles = []
        while nvert > 3:
            # Find an "ear"
            for i in range(nvert):
                vert = self.shape[i]
                next_vert = self.shape[(i + 1) % nvert]
                prev_vert = self.shape[i - 1]

                # Take cross-product of edge from prev->vert and vert->next
                # to check whether the angle is > 180 degrees
                cross = (vert[1] - prev_vert[1]) * (next_vert[0] - vert[0]) - (
                    vert[0] - prev_vert[0]
                ) * (next_vert[1] - vert[1])
                if cross < 0:
                    continue  # Skip this vertex

                # Check these edges don't intersect with other edges
                r1 = [prev_vert[0], vert[0], next_vert[0]]
                z1 = [prev_vert[1], vert[1], next_vert[1]]

                r2 = []
                z2 = []
                if i < nvert - 1:
                    r2 += [v[0] for v in self.shape[(i + 1) :]]
                    z2 += [v[1] for v in self.shape[(i + 1) :]]
                if i > 0:
                    r2 += [v[0] for v in self.shape[:i]]
                    z2 += [v[1] for v in self.shape[:i]]

                # (r1,z1) is the line along two edges of the triangle
                # (r2,z2) is the rest of the polygon
                if intersect(r1, z1, r2, z2, closed1=False, closed2=False):
                    continue  # Skip

                # Found an ear
                triangles.append([prev_vert, vert, next_vert])
                # Remove this vertex
                del self.shape[i]
                nvert -= 1
                break

        # Reduced to a single triangle
        triangles.append(self.shape)
        return triangles
    
    
class Solenoid:
    """
    Class representing a tokamak's external solenoid (confinement magnet).
    """
    
    def __init__(self,name,dim,Xe,I,Nturns):
        """
        Constructor to initialize the Solenoid object with the provided attributes.

        Input:
            - index (int): The index of the solenoid in the global system.
            - dim (int): The spatial dimension of the solenoid coordinates.
            - X (numpy.ndarray): Solenoid nodal coordinates matrix.
            - I (float): The current carried by the solenoid.
        """
        
        self.name = name        # IDENTIFICATION
        self.dim = dim          # SPATIAL DIMENSION
        self.Xe = Xe            # POSITION COORDINATES MATRIX
        self.I = I              # CURRENT
        self.Nturns = Nturns    # NUMBER OF TURNS
        
        # NUMERICAL INTEGRATION QUADRATURE
        self.ng = None          # NUMBER OF GAUSS INTEGRATION NODES FOR STANDARD 1D QUADRATURE
        self.XIg = None         # GAUSS INTEGRATION NODES (REFERENCE SPACE)
        self.Xg = None          # GAUSS INTEGRATION NODES (PHYSICAL SPACE)
        self.Wg = None          # GAUSS INTEGRATION WEIGTHS 
        self.Ng = None          # REFERENCE SHAPE FUNCTIONS EVALUATED AT GAUSS INTEGRATION NODES 
        self.dNdxig = None      # REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT GAUSS INTEGRATION NODES
        self.detJg = None       # DETERMINANT OF JACOBIAN OF TRANSFORMATION FROM 1D REFERENCE ELEMENT TO 2D PHYSICAL SOLENOID
        
        # TRANSFORM SOLENOID INTO COIL STRUCTURE EQUIVALENT
        self.Xcoils = self.Solenoid_coils()
        
        #self.COILS = list()
        #for icoil, Xcoil in enumerate(self.Xcoils):
        #    self.COILS.append(Coil(index = icoil,
        #                           dim = self.dim,
        #                           X = Xcoil,
        #                           I = self.I))
        return
    
    def Solenoid_coils(self):
        """
        Calculate the position of the individual coils constituting the solenoid.
        """
        if self.Nturns == 0:
            Xcoils = np.zeros([np.mean(self.Xe[:,0]),np.mean(self.Xe[:,1])])
        else:
            Xcoils = np.linspace(self.Xe[0],self.Xe[1],self.Nturns)
        return Xcoils
    
    def Psi(self,X):
        """
        Calculate poloidal flux psi at (R,Z) due to solenoid
        """
        Psi_sole = 0.0
        for Xcoil in self.Xcoils:
            Psi_sole += GreensFunction(Xcoil,X) * self.I
        return Psi_sole

    def Br(self,X):
        """
        Calculate radial magnetic field Br at (R,Z) due to solenoid
        """
        Br_sole = 0.0
        for Xcoil in self.Xcoils:
            Br_sole += GreensBr(Xcoil,X) * self.I
        return Br_sole

    def Bz(self,X):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to solenoid
        """
        Bz_sole = 0.0
        for Xcoil in self.Xcoils:
            Bz_sole += GreensBz(Xcoil,X) * self.I
        return Bz_sole



def intersect(r1, z1, r2, z2, closed1=True, closed2=True):
    """Test if two polynomials intersect. The polynomials consist of
    (r1, z1) and (r2, z2) line segments. All inputs are expected to be lists.

    Returns True or False.
    """

    assert len(r1) == len(z1)
    assert len(r2) == len(z2)

    n1 = len(r1) if closed1 else len(r1) - 1
    n2 = len(r2) if closed2 else len(r2) - 1

    for i in range(n1):
        for j in range(n2):
            # Test for intersection between two line segments:
            # (r1[i],z1[i]) -- (r1[i+1],z1[i+1])
            # (r1[j],z1[j]) -- (r1[j+1],z1[j+1])
            # Note that since polynomials are closed the indices wrap around
            ip = (i + 1) % n1
            jp = (j + 1) % n2

            a = r1[ip] - r1[i]
            b = r2[jp] - r2[j]
            c = z1[ip] - z1[i]
            d = z2[jp] - z2[j]

            dr = r2[jp] - r1[i]
            dz = z2[jp] - z1[i]

            det = a * d - b * c

            if abs(det) < 1e-6:
                continue  # Almost certainly doesn't intersect

            alpha = (d * dr - b * dz) / det  # Location along line 1 [0,1]
            beta = (a * dz - c * dr) / det  # Location along line 2 [0,1]

            if (alpha > 0.0) & (alpha < 1.0) & (beta > 0.0) & (beta < 1.0):
                return True
    # Tested all combinations, none intersect
    return False


def area(polygon):
    """
    Calculate the area of a polygon. Can be positive (clockwise) or negative (anticlockwise)

    Input

    polygon   [ (r1, z1), (r2, z2), ... ]
    """
    nvert = len(polygon)  # Number of vertices

    # Integrate using trapezium rule. The sign of (r2-r1) ensures that
    # positive and negative areas leave only the area of the polygon.
    area = 0.0
    for i in range(nvert):
        r1, z1 = polygon[i]
        r2, z2 = polygon[(i + 1) % nvert]  # Next vertex in periodic list
        area += (r2 - r1) * (z1 + z2)  # 2*area
    return 0.5 * area


def clockwise(polygon):
    """
    Detect whether a polygon is clockwise or anti-clockwise
    True -> clockwise
    False -> anticlockwise

    Input

    polygon   [ (r1, z1), (r2, z2), ... ]
    """
    # Work out the winding direction by calculating the area
    return area(polygon) > 0


def triangulate(polygon):
    """
    Use the ear clipping method to turn an arbitrary polygon into triangles

    Input

    polygon   [ (r1, z1), (r2, z2), ... ]
    """

    if clockwise(polygon):
        # Copy input into list
        polygon = list(iter(polygon))
    else:
        polygon = list(reversed(polygon))
    # Now polygon should be clockwise

    nvert = len(polygon)  # Number of vertices
    assert nvert > 2

    triangles = []
    while nvert > 3:
        # Find an "ear"
        for i in range(nvert):
            vert = polygon[i]
            next_vert = polygon[(i + 1) % nvert]
            prev_vert = polygon[i - 1]

            # Take cross-product of edge from prev->vert and vert->next
            # to check whether the angle is > 180 degrees
            cross = (vert[1] - prev_vert[1]) * (next_vert[0] - vert[0]) - (
                vert[0] - prev_vert[0]
            ) * (next_vert[1] - vert[1])
            if cross < 0:
                continue  # Skip this vertex

            # Check these edges don't intersect with other edges
            r1 = [prev_vert[0], vert[0], next_vert[0]]
            z1 = [prev_vert[1], vert[1], next_vert[1]]

            r2 = []
            z2 = []
            if i < nvert - 1:
                r2 += [v[0] for v in polygon[(i + 1) :]]
                z2 += [v[1] for v in polygon[(i + 1) :]]
            if i > 0:
                r2 += [v[0] for v in polygon[:i]]
                z2 += [v[1] for v in polygon[:i]]

            # (r1,z1) is the line along two edges of the triangle
            # (r2,z2) is the rest of the polygon
            if intersect(r1, z1, r2, z2, closed1=False, closed2=False):
                continue  # Skip

            # Found an ear
            triangles.append([prev_vert, vert, next_vert])
            # Remove this vertex
            del polygon[i]
            nvert -= 1
            break

    # Reduced to a single triangle
    triangles.append(polygon)
    return triangles