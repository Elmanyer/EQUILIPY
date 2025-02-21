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


# This script contains the information regarding the FEM interpolating shape functions.

import numpy as np

def ShapeFunctionsReference(X, elemType, elemOrder, node):
    """ 
    Nodal shape function in reference element, for element type and order elemType and elemOrder respectively, evaluated at point X.
    
    Input: 
        - X: coordinates of point on which to evaluate shape function (natural coordinates) 
        - elemType: 0=line, 1=tri, 2=quad
        - elemOrder: order of element
        - node: local nodal index 
    
    Output: 
        - N: nodal shape function evaluated at X
        - dNdxi: nodal shape function derivative respect to xi evaluated at point X
        - dNdeta: nodal shape function derivative respect to eta evaluated at point X
    """

    N = 0
    dNdxi = 0
    dNdeta = 0

    match elemType:
        case 0:    # LINE (1D ELEMENT)
            xi = X
            match elemOrder:
                case 0:
                    # --1--
                    match node:
                        case 1:
                            N = 1
                            dNdxi = 0
                case 1:
                    # 1---2
                    match node:
                        case 1:
                            N = (1-xi)/2
                            dNdxi = -1/2
                        case 2:
                            N = (1+xi)/2
                            dNdxi = 1/2       
                case 2:         
                    # 1---3---2
                    match node:
                        case 1:
                            N = -xi*(1-xi)/2
                            dNdxi = xi-0.5
                        case 2:
                            N = xi*(xi+1)/2
                            dNdxi = xi+0.5
                        case 3:
                            N = 1-xi**2
                            dNdxi = -2*xi
                case 3:         
                    # 1-3-4-2
                    match node:
                        case 1:
                            N = -9/16*(xi+1/3)*(xi-1/3)*(xi-1)
                            dNdxi = -9/16*((xi-1/3)*(xi-1)+(xi+1/3)*(xi-1)+(xi+1/3)*(xi-1/3))
                        case 2:
                            N =  9/16*(xi+1)*(xi+1/3)*(xi-1/3)
                            dNdxi = 9/16*((xi+1/3)*(xi-1/3)+(xi+1)*(xi-1/3)+(xi+1)*(xi+1/3))
                        case 3:
                            N = 27/16*(xi+1)*(xi-1/3)*(xi-1)
                            dNdxi = 27/16*((xi-1/3)*(xi-1)+ (xi+1)*(xi-1)+ (xi+1)*(xi-1/3))
                        case 4:
                            N = -27/16*(xi+1)*(xi+1/3)*(xi-1)
                            dNdxi = -27/16*((xi+1/3)*(xi-1)+(xi+1)*(xi-1)+(xi+1)*(xi+1/3))
    
        case 1:   # TRIANGLE
            xi = X[0]
            eta = X[1]
            match elemOrder:
                case 1:
                    # 2
                    # |\
                    # | \
                    # 3--1
                    match node:
                        case 1:
                            N = xi
                            dNdxi = 1
                            dNdeta = 0
                        case 2:
                            N = eta
                            dNdxi = 0
                            dNdeta = 1
                        case 3:
                            N = 1-(xi+eta)
                            dNdxi = -1
                            dNdeta = -1
                case 2:
                    # 2
                    # |\
                    # 5 4
                    # |  \
                    # 3-6-1
                    match node:
                        case 1:
                            N = xi*(2*xi-1)
                            dNdxi = 4*xi-1
                            dNdeta = 0
                        case 2:
                            N = eta*(2*eta-1)
                            dNdxi = 0
                            dNdeta = 4*eta-1
                        case 3:
                            N = (1-2*(xi+eta))*(1-(xi+eta))
                            dNdxi = -3+4*(xi+eta)
                            dNdeta = -3+4*(xi+eta)
                        case 4:
                            N = 4*xi*eta
                            dNdxi = 4*eta
                            dNdeta = 4*xi
                        case 5:
                            N = 4*eta*(1-(xi+eta))
                            dNdxi = -4*eta
                            dNdeta = 4*(1-xi-2*eta)
                        case 6:
                            N = 4*xi*(1-(xi+eta))
                            dNdxi = 4*(1-2*xi-eta)
                            dNdeta = -4*xi
                case 3:
                    #  2
                    # | \
                    # 6  5 
                    # |   \
                    # 7 10 4
                    # |     \
                    # 3-8--9-1
                    match node:
                        case 1:
                            N = (9/2)*(1/3-xi)*(2/3-xi)*xi
                            dNdxi = -(9/2)*((2/3-xi)*xi+(1/3-xi)*xi-(1/3-xi)*(2/3-xi))
                            dNdeta = 0
                        case 2:
                            N = (9/2)*(1/3-eta)*(2/3-eta)*eta 
                            dNdxi = 0
                            dNdeta = -(9/2)*((2/3-eta)*eta+(1/3-eta)*eta-(1/3-eta)*(2/3-eta))
                        case 3:
                            N = (9/2)*(1-xi-eta)*(2/3-xi-eta)*(1/3-xi-eta)  
                            dNdxi = -(9/2)*((1-xi-eta)*(2/3-xi-eta)+(1-xi-eta)*(1/3-xi-eta)+(2/3-xi-eta)*(1/3-xi-eta))
                            dNdeta = -(9/2)*((1-xi-eta)*(2/3-xi-eta)+(1-xi-eta)*(1/3-xi-eta)+(2/3-xi-eta)*(1/3-xi-eta))
                        case 4:
                            N = -3*(9/2)*(1/3-xi)*xi*eta
                            dNdxi = -3*(9/2)*((1/3-xi)*eta-xi*eta)
                            dNdeta = -3*(9/2)*((1/3-xi)*xi)
                        case 5:
                            N = -3*(9/2)*xi*(1/3-eta)*eta 
                            dNdxi = -3*(9/2)*((1/3-eta)*eta)
                            dNdeta = -3*(9/2)*((1/3-eta)*xi-xi*eta)
                        case 6:
                            N = -3*(9/2)*(1-xi-eta)*(1/3-eta)*eta
                            dNdxi = 3*(9/2)*((1/3-eta)*eta)
                            dNdeta = -3*(9/2)*(-(1/3-eta)*eta-(1-xi-eta)*eta+(1-xi-eta)*(1/3-eta))
                        case 7:
                            N = 3*(9/2)*(1-xi-eta)*(2/3-xi-eta)*eta
                            dNdxi = 3*(9/2)*(-(1-xi-eta)*eta-(2/3-xi-eta)*eta)
                            dNdeta = 3*(9/2)*(-(1-xi-eta)*eta-(2/3-xi-eta)*eta+(1-xi-eta)*(2/3-xi-eta))
                        case 8:
                            N = 3*(9/2)*(1-xi-eta)*(2/3-xi-eta)*xi 
                            dNdxi = 3*(9/2)*((1-xi-eta)*(2/3-xi-eta)-(1-xi-eta)*xi-(2/3-xi-eta)*xi)
                            dNdeta = 3*(9/2)*(-(1-xi-eta)*xi-(2/3-xi-eta)*xi)
                        case 9:
                            N = -3*(9/2)*(1-xi-eta)*(1/3-xi)*xi 
                            dNdxi = -3*(9/2)*((1-xi-eta)*(1/3-xi)-(1-xi-eta)*xi-(1/3-xi)*xi)
                            dNdeta = -3*(9/2)*(-(1/3-xi)*xi)
                        case 10:
                            N = 6*(9/2)*(1-xi-eta)*xi*eta
                            dNdxi = 6*(9/2)*((1-xi-eta)*eta-xi*eta)
                            dNdeta = 6*(9/2)*((1-xi-eta)*xi-xi*eta)
                            
        case 2:    # QUADRILATERAL
            xi = X[0]
            eta = X[1]
            match elemOrder:
                case 1: 
                    # 4-----3
                    # |     |
                    # |     |
                    # 1-----2
                    match node:
                        case 1:
                            N = (1-xi)*(1-eta)/4
                            dNdxi = (eta-1)/4
                            dNdeta = (xi-1)/4
                        case 2:
                            N = (1+xi)*(1-eta)/4
                            dNdxi = (1-eta)/4
                            dNdeta = -(1+xi)/4
                        case 3:
                            N = (1+xi)*(1+eta)/4
                            dNdxi = (1+eta)/4
                            dNdeta = (1+xi)/4
                        case 4:
                            N = (1-xi)*(1+eta)/4
                            dNdxi = -(1+eta)/4
                            dNdeta = (1-xi)/4
                case 2:
                    # 4---7---3
                    # |       |
                    # 8   9   6
                    # |       |
                    # 1---5---2
                    match node: 
                        case 1:
                            N = xi*(xi-1)*eta*(eta-1)/4
                            dNdxi = (xi-1/2)*eta*(eta-1)/2
                            dNdeta = xi*(xi-1)*(eta-1/2)/2
                        case 2:
                            N = xi*(xi+1)*eta*(eta-1)/4
                            dNdxi = (xi+1/2)*eta*(eta-1)/2
                            dNdeta = xi*(xi+1)*(eta-1/2)/2
                        case 3:
                            N = xi*(xi+1)*eta*(eta+1)/4
                            dNdxi = (xi+1/2)*eta*(eta+1)/2
                            dNdeta = xi*(xi+1)*(eta+1/2)/2
                        case 4:
                            N = xi*(xi-1)*eta*(eta+1)/4
                            dNdxi = (xi-1/2)*eta*(eta+1)/2
                            dNdeta = xi*(xi-1)*(eta+1/2)/2
                        case 5:
                            N = (1-xi**2)*eta*(eta-1)/2
                            dNdxi = -xi*eta*(eta-1)
                            dNdeta = (1-xi**2)*(eta-1/2)
                        case 6:
                            N = xi*(xi+1)*(1-eta**2)/2
                            dNdxi = (xi+1/2)*(1-eta**2)
                            dNdeta = xi*(xi+1)*(-eta)
                        case 7:
                            N = (1-xi**2)*eta*(eta+1)/2
                            dNdxi = -xi*eta*(eta+1)
                            dNdeta = (1-xi**2)*(eta+1/2)
                        case 8:
                            N = xi*(xi-1)*(1-eta**2)/2
                            dNdxi = (xi-1/2)*(1-eta**2)
                            dNdeta = xi*(xi-1)*(-eta)
                        case 9:
                            N = (1-xi**2)*(1-eta**2)
                            dNdxi = -2*xi*(1-eta**2)
                            dNdeta = (1-xi**2)*(-2*eta)
                case 3:
                    # 4---10--9---3
                    # |           |
                    # 11  16  15  8
                    # |           |
                    # 12  13  14  7
                    # |           |
                    # 1---5---6---2
                    a = 81./256.
                    c = 1./3.
                    s1 = 1. + xi
                    s2 = c + xi
                    s3 = c - xi
                    s4 = 1. - xi
                    t1 = 1. + eta
                    t2 = c + eta
                    t3 = c - eta
                    t4 = 1. - eta
                    match node:
                        case 1:
                            N = a*s2*s3*s4*t2*t3*t4
                            dNdxi = a*t2*t3*t4*(-s2*s3-s2*s4+s3*s4)
                            dNdeta = a*s2*s3*s4*(-t2*t3-t2*t4+t3*t4)
                        case 2:
                            N = a*s1*s2*s3*t2*t3*t4
                            dNdxi = a*t2*t3*t4*(-s1*s2+s1*s3+s2*s3)
                            dNdeta = a*s1*s2*s3*(-t2*t3-t2*t4+t3*t4)
                        case 3:
                            N = a*s1*s2*s3*t1*t2*t3
                            dNdxi = a*t1*t2*t3*(-s1*s2+s1*s3+s2*s3)
                            dNdeta = a*s1*s2*s3*(-t1*t2+t1*t3+t2*t3)
                        case 4:
                            N = a*s2*s3*s4*t1*t2*t3
                            dNdxi = a*t1*t2*t3*(-s2*s3-s2*s4+s3*s4)
                            dNdeta = a*s2*s3*s4*(-t1*t2+t1*t3+t2*t3)
                        case 5:
                            N = -3.0*a*s1*s3*s4*t2*t3*t4 
                            dNdxi = -3.0*a*t2*t3*t4*(-s1*s3-s1*s4+s3*s4)
                            dNdeta = -3.0*a *s1*s3*s4*(-t2*t3-t2*t4+t3*t4)
                        case 6:
                            N = -3.0*a*s1*s2*s4*t2*t3*t4
                            dNdxi = -3.0*a*t2*t3*t4*(-s1*s2+s1*s4+s2*s4)
                            dNdeta = -3.0*a *s1*s2*s4*(-t2*t3-t2*t4+t3*t4)
                        case 7:
                            N = -3.0*a*s1*s2*s3*t1*t3*t4
                            dNdxi = -3.0*a*t1*t3*t4*(-s1*s2+s1*s3+s2*s3)
                            dNdeta = -3.0*a *s1*s2*s3*(-t1*t3-t1*t4+t3*t4)
                        case 8:
                            N = -3.0*a*s1*s2*s3*t1*t2*t4
                            dNdxi = -3.0*a*t1*t2*t4*(-s1*s2+s1*s3+s2*s3)
                            dNdeta = -3.0*a *s1*s2*s3*(-t1*t2+t1*t4+t2*t4)
                        case 9:
                            N = -3.0*a*s1*s2*s4*t1*t2*t3  
                            dNdxi = -3.0*a*t1*t2*t3*(-s1*s2+s1*s4+s2*s4)
                            dNdeta = -3.0*a *s1*s2*s4*(-t1*t2+t1*t3+t2*t3)
                        case 10:
                            N = -3.0*a*s1*s3*s4*t1*t2*t3 
                            dNdxi = -3.0*a*t1*t2*t3*(-s1*s3-s1*s4+s3*s4)
                            dNdeta = -3.0*a *s1*s3*s4*(-t1*t2+t1*t3+t2*t3)
                        case 11:
                            N = -3.0*a*s2*s3*s4*t1*t2*t4
                            dNdxi = -3.0*a*t1*t2*t4*(-s2*s3-s2*s4+s3*s4)
                            dNdeta = -3.0*a *s2*s3*s4*(-t1*t2+t1*t4+t2*t4)
                        case 12:
                            N = -3.0*a*s2*s3*s4*t1*t3*t4
                            dNdxi = -3.0*a*t1*t3*t4*(-s2*s3-s2*s4+s3*s4)
                            dNdeta = -3.0*a *s2*s3*s4*(-t1*t3-t1*t4+t3*t4)
                        case 13:
                            N = 9.0*a*s1*s3*s4*t1*t3*t4
                            dNdxi = 9.0*a*t1*t3*t4*(-s1*s3-s1*s4+s3*s4)
                            dNdeta = 9.0*a *s1*s3*s4*(-t1*t3-t1*t4+t3*t4)
                        case 14:
                            N = 9.0*a*s1*s2*s4*t1*t3*t4
                            dNdxi = 9.0*a*t1*t3*t4*(-s1*s2+s1*s4+s2*s4)
                            dNdeta = 9.0*a *s1*s2*s4*(-t1*t3-t1*t4+t3*t4)
                        case 15:
                            N = 9.0*a*s1*s2*s4*t1*t2*t4
                            dNdxi = 9.0*a*t1*t2*t4*(-s1*s2+s1*s4+s2*s4)
                            dNdeta = 9.0*a *s1*s2*s4*(-t1*t2+t1*t4+t2*t4)
                        case 16:
                            N = 9.0*a*s1*s3*s4*t1*t2*t4
                            dNdxi = 9.0*a*t1*t2*t4*(-s1*s3-s1*s4+s3*s4)
                            dNdeta = 9.0*a *s1*s3*s4*(-t1*t2+t1*t4+t2*t4)
    return N, dNdxi, dNdeta

def EvaluateReferenceShapeFunctions(X, elemType, elemOrder):
    """ 
    Evaluates nodal shape functions in the reference space for the selected element type and order at points defined by coordinates X
    
    Input: 
        - X: coordinates of points on which to evaluate shape functions
        - elemType: 0=line, 1=tri, 2=quad
        - elemOrder: order of element

    Output: 
        - N: shape functions evaluated at points with coordinates X
        - dNdxi: shape functions derivatives respect to xi evaluated at points with coordinates X
        - dNdeta: shape functions derivatives respect to eta evaluated at points with coordinates X
    """
    
    from Element import ElementalNumberOfNodes
    ## NUMBER OF NODAL SHAPE FUNCTIONS
    n, foo = ElementalNumberOfNodes(elemType, elemOrder)
    ## NUMBER OF GAUSS INTEGRATION NODES
    if elemType == 0:
        ng = len(X)
    else: 
        ng = len(X[:,0])
        
    N = np.zeros([ng,n])
    dNdxi = np.zeros([ng,n])
    dNdeta = np.zeros([ng,n])
    
    for i in range(n):
        for ig in range(ng):
            N[ig,i], dNdxi[ig,i], dNdeta[ig,i] = ShapeFunctionsReference(X[ig,:],elemType, elemOrder, i+1)
            
    return N, dNdxi, dNdeta
    

def Jacobian(X,dNdxi,dNdeta):
    """ 
    Function that computes the Jacobian of the mapping between physical and natural coordinates 
        
    Input: 
        - X: elemental physical coordinates 
        - dNdxi: shape functions derivatives respect to xi evaluated at Gauss integration nodes
        - dNdeta: shape functions derivatives respect to eta evaluated at Gauss integration nodes
    
    Output: 
        - invJ: Jacobian inverse
        - detJ: Jacobian determinant 
    """
    
    J = np.zeros([2,2])
    # COMPUTE JACOBIAN
    for i in range(len(X[:,0])):
        J += np.array([[X[i,0]*dNdxi[i], X[i,1]*dNdxi[i]] ,
                       [X[i,0]*dNdeta[i], X[i,1]*dNdeta[i]]])
        # COMPUTE INVERSE MATRIX AND JACOBIAN
    invJ = np.linalg.inv(J)
    detJ = np.linalg.det(J)
    return invJ, detJ


def Jacobian1D(X,dNdxi):
    """
    Calculates the Jacobian determinant of the transformation between the 1D reference element and the physical space.

    Input:
        - X (numpy.ndarray): A 2D array of nodal coordinates in physical space, where each row corresponds to a node's 
                        physical coordinates.
                        Shape: (n_nodes, dimension), where `n_nodes` is the number of nodes in the element and 
                        `dimension` is the spatial dimension.          
        - dNdxi (numpy.ndarray): A 1D array of shape function derivatives with respect to the reference coordinate (xi).
                            Length: `n_nodes`, where `n_nodes` is the number of nodes in the element.

    Output:
        detJ (float): The determinant of the Jacobian matrix, which represents the scale factor when mapping from 
                    the reference space to the physical space.
    """
    J = np.zeros([2])
    for i in range(len(X[:,0])):
        J += dNdxi[i]*X[i,:]
    detJ = np.linalg.norm(J)
    return detJ