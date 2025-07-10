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


from Greens import *
from GaussQuadrature import *
from ShapeFunctions import *
from Element import compute_quadrilateral_area
import matplotlib.pyplot as plt

dcoil = 0.2
coilcolor = '#56B4E9'

class Coil:
    
    """ Class defining a tokamak's external magnet as a single point coil. """
    
    def __init__(self,name,X,I,Nturns=1):
        """
        Coil object constructor.

        Input:
            - name (str): Identifier name of the coil.
            - X (np.ndarray): Position vector of the coil center (shape: dim,).
            - I (float): Electric current carried by the coil [A].
            - Nturns (int): Number of turns in the coil (total current = I × Nturns).
        """
        self.name = name        # IDENTIFICATION
        self.X = X              # POSITION COORDINATES
        self.dim = len(X)       # SPATIAL DIMENSION
        self.I = I              # CURRENT
        self.Nturns = Nturns    # NUMBER OF TURNS (TOTAL CURRENT = I*Nturns)
        
        return
    
    def Psi(self,X):
        """
        Calculate poloidal flux psi at X=(R,Z) due to coil
        """
        return GreensFunction(self.X,X) * self.I * self.Nturns
    
    def Br(self,X):
        """
        Calculate radial magnetic field Br at X=(R,Z)
        """
        return GreensBr(self.X,X) * self.I * self.Nturns

    def Bz(self,X):
        """
        Calculate vertical magnetic field Bz at X=(R,Z)
        """
        return GreensBz(self.X,X) * self.I * self.Nturns
    
    def Plot(self,ax):
        """
        Plots the coil as a circle on the given axis.

        Input:
            - ax (matplotlib.axes.Axes): Matplotlib axis where the coil will be plotted.
        """
        ax.add_patch(plt.Circle((self.X[0],self.X[1]),dcoil,facecolor=coilcolor,edgecolor='k',linewidth=2))
        ax.text(self.X[0]+0.4,self.X[1]+0.4,self.name)
        return
    
    
class RectangularMultiCoil:
    
    """ Class defining a tokamak's external magnet constituted of multiple single coils. """
    
    def __init__(self,name,Xe,I, nr=1, nz=4, padr = 0.05, padz = 0.1, Xcoils=None):
        """
        Solenoid object constructor.

        Input:
            - name (str): Identifier name of the solenoid.
            - Xe (np.ndarray): Matrix of vertex coordinates (n_vertices x dim).
            - I (float): Electric current carried by the solenoid [A].
            - nr (int): Number of coils in the radial direction.
            - nz (int): Number of coils in the vertical direction.
            - padr (float): Radial padding between coils.
            - padz (float): Vertical padding between coils.
            - Xcoils (list[np.ndarray] or None): Predefined list of coil positions. If None, coils are automatically distributed.
        """
        
        self.name = name            # IDENTIFICATION
        self.Xe = Xe                # VERTICES COORDINATES MATRIX
        self.dim = np.shape(Xe)[1]  # SPATIAL DIMENSION
        self.I = I                  # CURRENT
        self.n = nr*nz              # NUMBER OF COILS (IF Xcoils not specified)
        self.Xcoils = Xcoils        # POSITIONS OF COILS CONSTITUTING MULTICOIL 
        
        # TRANSFORM SOLENOID INTO COIL STRUCTURE EQUIVALENT
        if type(Xcoils) == type(None):
            self.Xcoils = self.Distribute_coils(nr,nz,padr,padz)
            
        # CHECK THAT ncoils IS EQUAL TO THE DEFINED NUMBER OF COILS' POSITIONS
        if self.n != np.shape(self.Xcoils)[0]:
            self.n = np.shape(self.Xcoils)[0]
            print('Multicoil '+self.name+': number of coils corrected to '+ str(self.n))
        
        self.COILS = list()
        for icoil, xcoil in enumerate(self.Xcoils):
            self.COILS.append(Coil(name = 'coil '+str(icoil),
                                   X = xcoil,
                                   I = self.I/self.n))
        return
    
    def Distribute_coils(self, nr, nz, padr, padz):
        """
        Distributes coil centers uniformly within the solenoid's geometry.

        Input:
            - nr (int): Number of coils along the horizontal (radial) direction.
            - nz (int): Number of coils along the vertical (axial) direction.
            - padr (float): Fractional horizontal padding (relative to width).
            - padz (float): Fractional vertical padding (relative to height).

        Output:
            - Xcoils (np.ndarray): Array of coil center positions (n_coils x 2).
        """
        # Determine domain boundaries
        xmax = np.max(self.Xe[:, 0])
        xmin = np.min(self.Xe[:, 0])
        ymax = np.max(self.Xe[:, 1])
        ymin = np.min(self.Xe[:, 1])
        
        width = xmax - xmin
        height = ymax - ymin

        # Compute padding
        pad_x = width * padr
        pad_y = height * padz

        # Handle x (horizontal) positions
        if nr == 1:
            x = np.array([(xmin + xmax) / 2.0])  # center x
        else:
            x = np.linspace(xmin + pad_x, xmax - pad_x, nr)

        # Handle y (vertical) positions
        if nz == 1:
            y = np.array([(ymin + ymax) / 2.0])  # center y
        else:
            y = np.linspace(ymin + pad_y, ymax - pad_y, nz)
        
        grid_x, grid_y = np.meshgrid(x, y)
        Xcoils = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        return Xcoils
    
    def Psi(self,X):
        """
        Calculate poloidal flux psi at (R,Z) due to solenoid
        """
        Psi_sole = 0.0
        for coil in self.COILS:
            Psi_sole += coil.Psi(X)
        return Psi_sole 

    def Br(self,X):
        """
        Calculate radial magnetic field Br at (R,Z) due to solenoid
        """
        Br_sole = 0.0
        for coil in self.COILS:
            Br_sole += coil.Br(X)
        return Br_sole

    def Bz(self,X):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to solenoid
        """
        Bz_sole = 0.0
        for coil in self.COILS:
            Bz_sole += coil.Bz(X)
        return Bz_sole
    
    def Plot(self,ax):
        """
        Plot the solenoid and its coils on the given matplotlib axis.

        Input:
            - ax (matplotlib.axes.Axes): The axis to plot on.
        """
        for coil in self.COILS:
            ax.add_patch(plt.Circle((coil.X[0],coil.X[1]),dcoil,facecolor=coilcolor,edgecolor='k',linewidth=2))
        Xe = np.zeros([5,2])
        Xe[:-1,:] = self.Xe[:4,:]
        Xe[-1,:] = self.Xe[0,:]
        ax.plot(Xe[:,0], Xe[:,1], color='black', linewidth=1)
        ax.text(np.mean(self.Xe[:,0])+0.4,np.mean(self.Xe[:,1])+0.4,self.name)
        return


class QuadrilateralCoil:
    
    """ Class defining a tokamak's external magnet with quadrilateral (non-null surface) cross-section. """
    
    def __init__(self, name, Itotal, ElOrder = 2, QuadOrder = 4, **kwargs):
        """
        Quadrilateral magnet object constructor.

        Input:
            - name (str): Identifier name of the magnet.
            - Itotal (float): Total current passing through the magnet cross-section [A].
            - ElOrder (int, optional): Finite element order (default: 2).
            - QuadOrder (int, optional): Quadrature order for numerical integration (default: 4).
            - kwargs (optional):
                - Xvertices (np.ndarray): Vertices coordinates (n_vertices × dim) defining the cross-section.
                - Xcenter (np.ndarray): Center coordinates of the magnet cross-section.
                - Area (float): Cross-section area (required if Xcenter is given).
        """
        
        self.name = name                    # IDENTIFICATION
        self.I = Itotal                     # TOTAL CURRENT PASSING THOUGH MAGNET CROSS-SECTION
        self.area = None                    # CROSS-SECTION AREA
        self.Xvertices = None               # VERTICES COORDINATES
        self.Xcenter = None                 # CENTER COORDINATES
        # QUADRILATERAL SHAPED MAGNET
        self.ElOrder = ElOrder              # ASSOCIATED ELEMENT ORDER
        self.ElType = 2                     # ASSOCIATED ELEMENT TYPE (= 2 -> QUADRILATERAL)
        self.numedges = 4                   # ASSOCIATED ELEMENT NUMBER OF EDGES
        
        # CASE WHERE THE VERTICES COORDINATES ARE GIVEN TO DEFINE CROSS-SECTION
        if 'Xvertices' in kwargs:
            self.dim = np.shape(kwargs['Xvertices'])[1]    # SPATIAL DIMENSION
            self.Xvertices = kwargs['Xvertices']          # VERTICES COORDINATES MATRIX
            self.Xcenter = np.array([np.mean(self.Xvertices[:,0]),np.mean(self.Xvertices[:,1])])
        # CASE WHERE THE VERTICES ARE COMPUTED FROM CENTER POINT AND TOTAL AREA (ASSUMED SQUARE CROSS-SECTION)
        elif 'Xcenter' in kwargs:
            self.dim = np.shape(kwargs['Xcenter'])[0]    # SPATIAL DIMENSION
            # COMPUTE QUADRILATERAL MAGNET VERTICES
            self.Xcenter = kwargs['Xcenter']
            self.area = kwargs['Area']
            h = np.sqrt(self.area)/2
            self.Xvertices = np.array([[self.Xcenter[0] - h, self.Xcenter[1] - h],  # bottom-left
                                       [self.Xcenter[0] + h, self.Xcenter[1] - h],  # bottom-right
                                       [self.Xcenter[0] + h, self.Xcenter[1] + h],  # top-right
                                       [self.Xcenter[0] - h, self.Xcenter[1] + h]])  # top-left       
        
        # NUMERICAL INTEGRATION QUADRATURE
        self.ng = None              # NUMBER OF GAUSS INTEGRATION NODES FOR STANDARD 1D QUADRATURE
        self.XIg = None             # GAUSS INTEGRATION NODES (REFERENCE SPACE)
        self.Xg = None              # GAUSS INTEGRATION NODES (PHYSICAL SPACE)
        self.Wg = None              # GAUSS INTEGRATION WEIGTHS 
        self.Ng = None              # REFERENCE SHAPE FUNCTIONS EVALUATED AT GAUSS INTEGRATION NODES 
        self.dNdxig = None          # REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO XI EVALUATED AT GAUSS INTEGRATION NODES
        self.dNdetag = None         # REFERENCE SHAPE FUNCTIONS DERIVATIVES RESPECT TO ETA EVALUATED AT GAUSS INTEGRATION NODES
        self.invJg = None           # INVERSE MATRIX OF JACOBIAN OF TRANSFORMATION FROM 2D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT, EVALUATED AT GAUSS INTEGRATION NODES
        self.detJg = None           # MATRIX DETERMINANT OF JACOBIAN OF TRANSFORMATION FROM 2D REFERENCE ELEMENT TO 2D PHYSICAL ELEMENT, EVALUATED AT GAUSS INTEGRATION NODES 
        
        # COMPUTE QUADRILATERAL COIL AREA
        if type(self.area) == type(None):
            self.area = compute_quadrilateral_area(self.Xvertices)
        # COMPUTE QUADRILATERAL HIGH-ORDER NODES
        self.Xe = self.HO_quadrilateral()
        # COMPUTE NUMERICAL INTEGRATION QUADRATURE
        self.ComputeQuadrature(QuadOrder)
        self.CheckQuadrature()
        return
    
    
    def HO_quadrilateral(self):
        """
        Generates a high-order quadrilateral element from a linear one with nodal vertices coordinates Xvertices, incorporating high-order 
        nodes on the edges and interior.

        Output: 
            XeHO (numpy.ndarray): An array containing the coordinates of the high-order element nodes, including those on 
                the edges and interior.
        """
        
        XeHO = self.Xvertices.copy()
        for iedge in range(self.numedges):
            inode = iedge
            jnode = (iedge+1)%self.numedges
            for k in range(1,self.ElOrder):
                HOnode = np.array([self.Xvertices[inode,0]+((self.Xvertices[jnode,0]-self.Xvertices[inode,0])/self.ElOrder)*k,self.Xvertices[inode,1]+((self.Xvertices[jnode,1]-self.Xvertices[inode,1])/self.ElOrder)*k])
                XeHO = np.concatenate((XeHO,HOnode.reshape((1,2))), axis=0)
        # INTERIOR HIGH-ORDER NODES:
        if self.ElOrder == 2:
            HOnode = np.array([np.mean(XeHO[:,0]),np.mean(XeHO[:,1])])
            XeHO = np.concatenate((XeHO,HOnode.reshape((1,2))), axis=0)
        elif self.ElOrder == 3:
            for k in range(1,self.ElOrder):
                dx = (XeHO[12-k,0]-XeHO[5+k,0])/self.ElOrder
                dy = (XeHO[12-k,1]-XeHO[5+k,1])/self.ElOrder
                for j in range(1,self.ElOrder):
                    if k == 1:
                        HOnode = XeHO[11,:] - np.array([dx*j,dy*j])
                    elif k == 2:
                        HOnode = XeHO[7,:] + np.array([dx*j,dy*j])
                    XeHO = np.concatenate((XeHO,HOnode.reshape((1,2))), axis=0)
        return XeHO
    
    
    def ComputeQuadrature(self,QuadOrder):
        """
        Computes the numerical integration quadrature for the quadrilateral magnet.

        Input:
            - QuadOrder (int): Order of the Gauss quadrature to use.

        Sets the following attributes:
            - XIg (np.ndarray): Gauss integration nodes in reference element space.
            - Wg (np.ndarray): Corresponding Gauss integration weights.
            - ng (int): Number of Gauss integration nodes.
            - Ng (np.ndarray): Reference shape functions evaluated at Gauss nodes.
            - dNdxig (np.ndarray): Derivatives of shape functions w.r.t. xi at Gauss nodes.
            - dNdetag (np.ndarray): Derivatives of shape functions w.r.t. eta at Gauss nodes.
            - Xg (np.ndarray): Gauss nodes mapped to physical element coordinates.
            - invJg (np.ndarray): Inverse Jacobian matrices at each Gauss node.
            - detJg (np.ndarray): Determinants of Jacobian matrices at each Gauss node (absolute value).
        """
        #### REFERENCE ELEMENT QUADRATURE TO INTEGRATE SURFACES
        self.XIg, self.Wg, self.ng = GaussQuadrature(self.ElType,QuadOrder)
        # EVALUATE REFERENCE SHAPE FUNCTIONS 
        self.Ng, self.dNdxig, self.dNdetag = EvaluateReferenceShapeFunctions(self.XIg, self.ElType, self.ElOrder)
        
        # COMPUTE MAPPED GAUSS NODES
        self.Xg = self.Ng @ self.Xe       
        # COMPUTE JACOBIAN INVERSE AND DETERMINANT
        self.invJg = np.zeros([self.ng,self.dim,self.dim])
        self.detJg = np.zeros([self.ng])
        for ig in range(self.ng):
            self.invJg[ig,:,:], self.detJg[ig] = Jacobian(self.Xe,self.dNdxig[ig,:],self.dNdetag[ig,:])
            self.detJg[ig] = abs(self.detJg[ig])
        return
    
    def CheckQuadrature(self):
        """
        Verifies the numerical integration quadrature by checking the integrated area.

        Raises:
            - ValueError: If the computed integral differs from the coil area by more than 1e-6.
        """
        # CHECK NUMERICAL INTEGRATION QUADRATURE BY INTEGRATING AREA
        integral = 0
        for ig in range(self.ng):
            integral += self.detJg[ig]*self.Wg[ig] 
        if abs(integral - self.area) > 1e-6:
            raise ValueError('Quadrilateral coil '+self.name+': error in integration quadrature.')
        return
    
    
    def Psi(self,X):
        """
        Calculate poloidal flux psi at (R,Z) due to quadrilateral coil
        """
        Psi_coil = 0.0
        # INTEGRATE ALONG SOLENOID AREA 
        for ig in range(self.ng):
            Psi_coil += GreensFunction(self.Xg[ig,:],X) * self.detJg[ig] * self.Wg[ig] * self.I/self.area
        return Psi_coil  
    
    def Br(self,X):
        """
        Calculate radial magnetic field Br at (R,Z) due to quadrilateral coil
        """
        Br_coil = 0.0
        # INTEGRATE ALONG SOLENOID AREA 
        for ig in range(self.ng):
            Br_coil += GreensBr(self.Xg[ig,:],X) * self.detJg[ig] * self.Wg[ig] * self.I/self.area
        return Br_coil

    def Bz(self,X):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to quadrilateral coil
        """
        Bz_coil = 0.0
        # INTEGRATE ALONG SOLENOID AREA 
        for ig in range(self.ng):
            Br_coil += GreensBz(self.Xg[ig,:],X) * self.detJg[ig] * self.Wg[ig] * self.I/self.area
        return Bz_coil
    
    def Plot(self,ax):
        """
        Plots the quadrilateral coil on the given matplotlib axis.

        Input:
            - ax (matplotlib.axes.Axes): Axis object where the coil will be plotted.
        """
        ax.add_patch(plt.Polygon(self.Xvertices,closed=True,facecolor=coilcolor,edgecolor='k',linewidth=3))
        ax.text(self.Xcenter[0]+0.5,self.Xcenter[1], self.name)
        return
    


class ShapedCoil:
    
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