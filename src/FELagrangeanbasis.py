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


# This script contains the information regarding the FEM interpolating shape functions.

import numpy as np

def RefLagrangeBasis(X, elemType, elemOrder, node, deriv=1):
    """ 
    Nodal shape function in reference element, for element type and order elemType and elemOrder respectively, evaluated at point X.
    
    Input: 
        - X: coordinates of point on which to evaluate shape function (natural coordinates) 
        - elemType: 0=line, 1=tri, 2=quad
        - elemOrder: order of element
        - node: local nodal index 
        - deriv: order of derivative to evaluate (default=1, only first derivatives are implemented)
    
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
            # Extract scalar from array for 1D case
            xi = float(X.flat[0]) if hasattr(X, 'flat') else float(X)
            match elemOrder:
                case 0:
                    # --1--
                    N = 1
                case 1:
                    # 1---2
                    match node:
                        case 1:
                            N = (1-xi)/2
                            if deriv >= 1:
                                dNdxi = -1/2
                        case 2:
                            N = (1+xi)/2
                            if deriv >= 1:
                                dNdxi = 1/2 
                case 2:         
                    # 1---3---2
                    match node:
                        case 1:
                            N = -xi*(1-xi)/2
                            if deriv >= 1:
                                dNdxi = xi - 0.5
                            if deriv >= 2:
                                d2Ndxi2 = 1.0
                        case 2: 
                            N = xi*(xi+1)/2
                            if deriv >= 1:
                                dNdxi = xi + 0.5
                            if deriv >= 2:
                                d2Ndxi2 = 1.0
                        case 3: 
                            N = 1 - xi**2
                            if deriv >= 1:
                                dNdxi = -2*xi
                            if deriv >= 2:
                                d2Ndxi2 = -2.0
                case 3:         
                    # 1-3-4-2
                    match node:
                        case 1:
                            N = -9/16*(xi+1/3)*(xi-1/3)*(xi-1)
                            if deriv >= 1:
                                dNdxi = -9/16*((xi-1/3)*(xi-1)+(xi+1/3)*(xi-1)+(xi+1/3)*(xi-1/3))
                            if deriv >= 2:
                                d2Ndxi2 = -9/16 * (6*xi - 2)
                            if deriv >= 3:
                                d3Ndxi3 = -27/8
                        case 2:
                            N =  9/16*(xi+1)*(xi+1/3)*(xi-1/3)
                            if deriv >= 1:
                                dNdxi = 9/16*((xi+1/3)*(xi-1/3)+(xi+1)*(xi-1/3)+(xi+1)*(xi+1/3))
                            if deriv >= 2:
                                d2Ndxi2 = 9/16 * (6*xi + 2)
                            if deriv >= 3:
                                d3Ndxi3 = 27/8   
                        case 3:
                            N = 27/16*(xi+1)*(xi-1/3)*(xi-1)
                            if deriv >= 1:
                                dNdxi = 27/16*((xi-1/3)*(xi-1)+ (xi+1)*(xi-1)+ (xi+1)*(xi-1/3))
                            if deriv >= 2:
                                d2Ndxi2 = 27/16 * (6*xi - 2/3)
                            if deriv >= 3:
                                d3Ndxi3 = 81/8   
                        case 4:
                            N = -27/16*(xi+1)*(xi+1/3)*(xi-1)
                            if deriv >= 1:
                                dNdxi = -27/16*((xi+1/3)*(xi-1)+(xi+1)*(xi-1)+(xi+1)*(xi+1/3))
                            if deriv >= 2:
                                d2Ndxi2 = -27/16 * (6*xi + 2/3)
                            if deriv >= 3:
                                d3Ndxi3 = -81/8  

            match deriv:
                case 0:
                    return N
                case 1:
                    return N, dNdxi
                case 2:
                    return N, dNdxi, d2Ndxi2
                case 3:
                    return N, dNdxi, d2Ndxi2, d3Ndxi3

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
                            if deriv >= 1:
                                dNdxi = 1
                                dNdeta = 0
                        case 2:
                            N = eta
                            if deriv >= 1:
                                dNdxi = 0
                                dNdeta = 1
                        case 3:
                            N = 1-(xi+eta)
                            if deriv >= 1:
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
                            if deriv >= 1:
                                dNdxi = 4*xi-1
                                dNdeta = 0
                            if deriv >= 2:
                                Hess = np.array([[4.0, 0.0], [0.0, 0.0]])
                        case 2:
                            N = eta*(2*eta-1)
                            if deriv >= 1:
                                dNdxi = 0
                                dNdeta = 4*eta-1
                            if deriv >= 2:
                                Hess = np.array([[0.0, 0.0], [0.0, 4.0]])
                        case 3:
                            N = (1-2*(xi+eta))*(1-(xi+eta))
                            if deriv >= 1:
                                dNdxi = -3+4*(xi+eta)
                                dNdeta = -3+4*(xi+eta)
                            if deriv >= 2:
                                Hess = np.array([[4.0, 4.0], [4.0, 4.0]])
                        case 4:
                            N = 4*xi*eta
                            if deriv >= 1:
                                dNdxi = 4*eta
                                dNdeta = 4*xi
                            if deriv >= 2:
                                Hess = np.array([[0.0, 4.0], [4.0, 0.0]])
                        case 5:
                            N = 4*eta*(1-(xi+eta))
                            if deriv >= 1:
                                dNdxi = -4*eta
                                dNdeta = 4*(1-xi-2*eta)
                            if deriv >= 2:
                                Hess = np.array([[0.0, -4.0], [-4.0, -8.0]])
                        case 6:
                            N = 4*xi*(1-(xi+eta))
                            if deriv >= 1:
                                dNdxi = 4*(1-2*xi-eta)
                                dNdeta = -4*xi
                            if deriv >= 2:
                                Hess = np.array([[-8.0, -4.0], [-4.0, 0.0]])
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
                            if deriv >= 1:
                                dNdxi = -(9/2)*((2/3-xi)*xi+(1/3-xi)*xi-(1/3-xi)*(2/3-xi))
                                dNdeta = 0
                            if deriv >= 2:
                                Hess = np.array([[27*xi - 9, 0], [0, 0]])
                            if deriv >= 3:
                                J3 = np.array([[[27, 0], [0, 0]], [[0, 0], [0,0]]])
                        case 2: 
                            N = (9/2)*(1/3-eta)*(2/3-eta)*eta 
                            if deriv >= 1:
                                dNdxi = 0
                                dNdeta = -(9/2)*((2/3-eta)*eta+(1/3-eta)*eta-(1/3-eta)*(2/3-eta))
                            if deriv >= 2:
                                Hess = np.array([[0, 0], [0, 27*eta - 9]])
                            if deriv >= 3:
                                J3 = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 27]]])
                        case 3: 
                            N = (9/2)*(1-xi-eta)*(2/3-xi-eta)*(1/3-xi-eta)  
                            if deriv >= 1:
                                dNdxi = -(9/2)*((1-xi-eta)*(2/3-xi-eta)+(1-xi-eta)*(1/3-xi-eta)+(2/3-xi-eta)*(1/3-xi-eta))
                                dNdeta = -(9/2)*((1-xi-eta)*(2/3-xi-eta)+(1-xi-eta)*(1/3-xi-eta)+(2/3-xi-eta)*(1/3-xi-eta))
                            if deriv >= 2:
                                val = 27*(1 - xi - eta) - 9 
                                Hess = np.array([[val, val], [val, val]])
                            if deriv >= 3:
                                    J3 = np.full((2, 2, 2), -27.0)
                        case 4: 
                            N = -3*(9/2)*(1/3-xi)*xi*eta
                            if deriv >= 1:
                                dNdxi = -3*(9/2)*((1/3-xi)*eta-xi*eta)
                                dNdeta = -3*(9/2)*((1/3-xi)*xi)
                            if deriv >= 2:
                                Hess = np.array([[27*eta, 27*xi - 4.5], [27*xi - 4.5, 0]])
                            if deriv >= 3:
                                    # d3N/dxi2deta = 27, others 0
                                    J3 = np.zeros((2,2,2))
                                    J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = 27
                        case 5: 
                            N = -3*(9/2)*xi*(1/3-eta)*eta 
                            if deriv >= 1:
                                dNdxi = -3*(9/2)*((1/3-eta)*eta)
                                dNdeta = -3*(9/2)*((1/3-eta)*xi-xi*eta)
                            if deriv >= 2:
                                    Hess = np.array([[0, 27*eta - 4.5], [27*eta - 4.5, 27*xi]])
                            if deriv >= 3:
                                # d3N/dxi deta2 = 27
                                J3 = np.zeros((2,2,2))
                                J3[0,1,1] = J3[1,0,1] = J3[1,1,0] = 27
                        case 6: 
                            N = -3*(9/2)*(1-xi-eta)*(1/3-eta)*eta
                            if deriv >= 1:
                                dNdxi = 3*(9/2)*((1/3-eta)*eta)
                                dNdeta = -3*(9/2)*(-(1/3-eta)*eta-(1-xi-eta)*eta+(1-xi-eta)*(1/3-eta))
                            if deriv >= 2:
                                Hess = np.array([[0, -27*eta + 4.5], [-27*eta + 4.5, 54*eta - 27*(1-xi-eta) - 9]])
                            if deriv >= 3:
                                    J3 = np.zeros((2,2,2))
                                    J3[1,1,1], J3[1,1,0], J3[1,0,1], J3[0,1,1] = 81, -27, -27, -27
                        case 7: 
                            N = 3*(9/2)*(1-xi-eta)*(2/3-xi-eta)*eta
                            if deriv >= 1:
                                dNdxi = 3*(9/2)*(-(1-xi-eta)*eta-(2/3-xi-eta)*eta)
                                dNdeta = 3*(9/2)*(-(1-xi-eta)*eta-(2/3-xi-eta)*eta+(1-xi-eta)*(2/3-xi-eta))
                            if deriv >= 2:
                                Hess = np.array([[27*eta, 27*(1-xi-eta) + 27*eta - 4.5], [27*(1-xi-eta) + 27*eta - 4.5, 54*eta - 54*(1-xi-eta) + 9]])
                            if deriv >= 3:
                                J3 = np.array([[ [0, -27], [-27, 54] ], [ [-27, 54], [54, -81] ]])
                        case 8: 
                            N = 3*(9/2)*(1-xi-eta)*(2/3-xi-eta)*xi 
                            if deriv >= 1:
                                dNdxi = 3*(9/2)*((1-xi-eta)*(2/3-xi-eta)-(1-xi-eta)*xi-(2/3-xi-eta)*xi)
                                dNdeta = 3*(9/2)*(-(1-xi-eta)*xi-(2/3-xi-eta)*xi)
                            if deriv >= 2:
                                Hess = np.array([[54*xi - 54*(1-xi-eta) + 9, 27*xi + 27*(1-xi-eta) - 4.5], [27*xi + 27*(1-xi-eta) - 4.5, 27*xi]])
                            if deriv >= 3:
                                J3 = np.array([[ [-81, 54], [54, -27] ], [ [54, -27], [-27, 0] ]])
                        case 9: 
                            N = -3*(9/2)*(1-xi-eta)*(1/3-xi)*xi 
                            if deriv >= 1:
                                dNdxi = -3*(9/2)*((1-xi-eta)*(1/3-xi)-(1-xi-eta)*xi-(1/3-xi)*xi)
                                dNdeta = -3*(9/2)*(-(1/3-xi)*xi)
                            if deriv >= 2:
                                Hess = np.array([[27*(1-xi-eta) - 54*xi + 9, -27*xi + 4.5], [-27*xi + 4.5, 0]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0], J3[0,0,1], J3[0,1,0], J3[1,0,0] = 81, -27, -27, -27
                        case 10: 
                            N = 6*(9/2)*(1-xi-eta)*xi*eta
                            if deriv >= 1:
                                dNdxi = 6*(9/2)*((1-xi-eta)*eta-xi*eta)
                                dNdeta = 6*(9/2)*((1-xi-eta)*xi-xi*eta)
                            if deriv >= 2:
                                Hess = np.array([[-54*eta, 27*((1-xi-eta) - xi - eta)], [27*((1-xi-eta) - xi - eta), -54*xi]])
                            if deriv >= 3:
                                # Pure derivatives are zero, cross derivatives are -54
                                J3 = np.zeros((2,2,2))
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = -54
                                J3[1,1,0] = J3[1,0,1] = J3[0,1,1] = -54

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
                            if deriv >= 1:
                                dNdxi = (eta-1)/4
                                dNdeta = (xi-1)/4
                        case 2:
                            N = (1+xi)*(1-eta)/4
                            if deriv >= 1:
                                dNdxi = (1-eta)/4
                                dNdeta = -(1+xi)/4
                        case 3:
                            N = (1+xi)*(1+eta)/4
                            if deriv >= 1:
                                dNdxi = (1+eta)/4
                                dNdeta = (1+xi)/4
                        case 4:
                            N = (1-xi)*(1+eta)/4
                            if deriv >= 1:
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
                            if deriv >= 1:
                                dNdxi = (xi-1/2)*eta*(eta-1)/2
                                dNdeta = xi*(xi-1)*(eta-1/2)/2
                            if deriv >= 2:
                                Hess = np.array([[eta*(eta-1)/2, (xi-1/2)*(eta-1/2)], [(xi-1/2)*(eta-1/2), xi*(xi-1)/2]])
                        case 2:
                            N = xi*(xi+1)*eta*(eta-1)/4
                            if deriv >= 1:
                                dNdxi = (xi+1/2)*eta*(eta-1)/2
                                dNdeta = xi*(xi+1)*(eta-1/2)/2
                            if deriv >= 2:
                                Hess = np.array([[eta*(eta-1)/2, (xi+1/2)*(eta-1/2)], [(xi+1/2)*(eta-1/2), xi*(xi+1)/2]])
                        case 3:
                            N = xi*(xi+1)*eta*(eta+1)/4
                            if deriv >= 1:
                                dNdxi = (xi+1/2)*eta*(eta+1)/2
                                dNdeta = xi*(xi+1)*(eta+1/2)/2
                            if deriv >= 2:
                                Hess = np.array([[eta*(eta+1)/2, (xi+1/2)*(eta+1/2)], [(xi+1/2)*(eta+1/2), xi*(xi+1)/2]])
                        case 4:
                            N = xi*(xi-1)*eta*(eta+1)/4
                            if deriv >= 1:
                                dNdxi = (xi-1/2)*eta*(eta+1)/2
                                dNdeta = xi*(xi-1)*(eta+1/2)/2
                            if deriv >= 2:
                                Hess = np.array([[eta*(eta+1)/2, (xi-1/2)*(eta+1/2)], [(xi-1/2)*(eta+1/2), xi*(xi-1)/2]])
                        case 5:
                            N = (1-xi**2)*eta*(eta-1)/2
                            if deriv >= 1:
                                dNdxi = -xi*eta*(eta-1)
                                dNdeta = (1-xi**2)*(eta-1/2)
                            if deriv >= 2:
                                Hess = np.array([[-eta*(eta-1), -2*xi*eta + xi], [-2*xi*eta + xi, 1-xi**2]])
                        case 6:
                            N = xi*(xi+1)*(1-eta**2)/2
                            if deriv >= 1:
                                dNdxi = (xi+1/2)*(1-eta**2)
                                dNdeta = xi*(xi+1)*(-eta)
                            if deriv >= 2:
                                # Calculating the Hessian for the quadratic terms
                                Hess = np.array([[1 - eta**2,    -2*xi*eta - eta], [-2*xi*eta - eta,   -xi**2 - xi]])
                        case 7:
                            N = (1-xi**2)*eta*(eta+1)/2
                            if deriv >= 1:
                                dNdxi = -xi*eta*(eta+1)
                                dNdeta = (1-xi**2)*(eta+1/2)
                            if deriv >= 2:
                                Hess = np.array([[-eta*(eta+1), -2*xi*eta - xi], [-2*xi*eta - xi, 1-xi**2]])
                        case 8:
                            N = xi*(xi-1)*(1-eta**2)/2
                            if deriv >= 1:
                                dNdxi = (xi-1/2)*(1-eta**2)
                                dNdeta = xi*(xi-1)*(-eta)
                            if deriv >= 2:
                                Hess = np.array([[1 - eta**2,    -2*xi*eta - eta], [-2*xi*eta - eta,   -xi**2 + xi]])
                        case 9:
                            N = (1-xi**2)*(1-eta**2)
                            if deriv >= 1:
                                dNdxi = -2*xi*(1-eta**2)
                                dNdeta = (1-xi**2)*(-2*eta)
                            if deriv >= 2:
                                Hess = np.array([[-2*(1-eta**2), -2*xi*(-2*eta)], [-2*xi*(-2*eta), -2*(1-xi**2)]])
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
                            if deriv >= 1:
                                dNdxi = a*t2*t3*t4*(-s2*s3-s2*s4+s3*s4)
                                dNdeta = a*s2*s3*s4*(-t2*t3-t2*t4+t3*t4)
                            if deriv >= 2:
                                Hess = np.array([[a*t2*t3*t4*(-s2 - s3 - s4), a*(-s2*s3-s2*s4+s3*s4)*(-t2*t3-t2*t4+t3*t4)], 
                                                    [a*(-s2*s3-s2*s4+s3*s4)*(-t2*t3-t2*t4+t3*t4), a*s2*s3*s4*(-t2 - t3 - t4)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = a*t2*t3*t4*(-s2 - s3 - s4)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = a*(-s2*s3-s2*s4+s3*s4)*(-t2*t3-t2*t4+t3*t4)
                                J3[1,1,0] = J3[1,0,1] = J3[0,1,1] = a*s2*s3*s4*(-t2 - t3 - t4)
                        case 2:
                            N = a*s1*s2*s3*t2*t3*t4
                            if deriv >= 1:
                                dNdxi = a*t2*t3*t4*(-s1*s2+s1*s3+s2*s3)
                                dNdeta = a*s1*s2*s3*(-t2*t3-t2*t4+t3*t4)
                            if deriv >= 2:
                                Hess = np.array([[a*t2*t3*t4*(-s1 - s2 - s3), a*(-s1*s2-s1*s3+s2*s3)*(-t2*t3-t2*t4+t3*t4)], 
                                                     [a*(-s1*s2-s1*s3+s2*s3)*(-t2*t3-t2*t4+t3*t4), a*s1*s2*s3*(-t2 - t3 - t4)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = a*t2*t3*t4*(-s1 - s2 - s3)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = a*(-s1*s2-s1*s3+s2*s3)*(-t2*t3-t2*t4+t3*t4)
                                J3[1,1,0] = J3[1,0,1] = J3[0,1,1] = a*s1*s2*s3*(-t2 - t3 - t4)

                        case 3:
                            N = a*s1*s2*s3*t1*t2*t3
                            if deriv >= 1:
                                dNdxi = a*t1*t2*t3*(-s1*s2+s1*s3+s2*s3)
                                dNdeta = a*s1*s2*s3*(-t1*t2+t1*t3+t2*t3)
                            if deriv >= 2:
                                Hess = np.array([[a*t1*t2*t3*(-s1 - s2 - s3), a*(-s1*s2-s1*s3+s2*s3)*(-t1*t2+t1*t3+t2*t3)],
                                                     [a*(-s1*s2-s1*s3+s2*s3)*(-t1*t2+t1*t3+t2*t3), a*s1*s2*s3*(-t1 - t2 - t3)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = a*t1*t2*t3*(-s1 - s2 - s3)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = a*(-s1*s2-s1*s3+s2*s3)*(-t1*t2+t1*t3+t2*t3)
                                J3[1,1,0] = J3[1,0,1] = J3[0,1,1] = a*s1*s2*s3*(-t1 - t2 - t3)
                        case 4:
                            N = a*s2*s3*s4*t1*t2*t3
                            if deriv >= 1:
                                dNdxi = a*t1*t2*t3*(-s2*s3-s2*s4+s3*s4)
                                dNdeta = a*s2*s3*s4*(-t1*t2+t1*t3+t2*t3)
                            if deriv >= 2:
                                Hess = np.array([[a*t1*t2*t3*(-s2 - s3 - s4), a*(-s2*s3-s2*s4+s3*s4)*(-t1*t2+t1*t3+t2*t3)],
                                                     [a*(-s2*s3-s2*s4+s3*s4)*(-t1*t2+t1*t3+t2*t3), a*s2*s3*s4*(-t1 - t2 - t3)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = a*t1*t2*t3*(-s2 - s3 - s4)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = a*(-s2*s3-s2*s4+s3*s4)*(-t1*t2+t1*t3+t2*t3)
                                J3[1,1,0] = J3[1,0,1] = J3[0,1,1] = a*s2*s3*s4*(-t1 - t2 - t3)
                        case 5:
                            N = -3.0*a*s1*s3*s4*t2*t3*t4 
                            if deriv >= 1:
                                dNdxi = -3.0*a*t2*t3*t4*(-s1*s3-s1*s4+s3*s4)
                                dNdeta = -3.0*a *s1*s3*s4*(-t2*t3-t2*t4+t3*t4)
                            if deriv >= 2:
                                Hess = np.array([[-3.0*a*t2*t3*t4*(-s1 - s3 - s4), -3.0*a*(-s1*s3-s1*s4+s3*s4)*(-t2*t3-t2*t4+t3*t4)], 
                                                     [-3.0*a*(-s1*s3-s1*s4+s3*s4)*(-t2*t3-t2*t4+t3*t4), -3.0*a*s1*s3*s4*(-t2 - t3 - t4)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = -3.0*a*t2*t3*t4*(-s1 - s3 - s4)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = -3.0*a*(-s1*s3-s1*s4+s3*s4)*(-t2*t3-t2*t4+t3*t4)
                                J3[1,1,0] = J3[1,0,1] = J3[0,1,1] = -3.0*a*s1*s3*s4*(-t2 - t3 - t4)
                        case 6:
                            N = -3.0*a*s1*s2*s4*t2*t3*t4
                            if deriv >= 1:
                                dNdxi = -3.0*a*t2*t3*t4*(-s1*s2+s1*s4+s2*s4)
                                dNdeta = -3.0*a *s1*s2*s4*(-t2*t3-t2*t4+t3*t4)
                            if deriv >= 2:
                                Hess = np.array([[-3.0*a*t2*t3*t4*(-s1 - s2 + s4), -3.0*a*(-s1*s2+s1*s4+s2*s4)*(-t2*t3-t2*t4+t3*t4)], 
                                                    [-3.0*a*(-s1*s2+s1*s4+s2*s4)*(-t2*t3-t2*t4+t3*t4), -3.0*a*s1*s2*s4*(-t2 - t3 - t4)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = -3.0*a*t2*t3*t4*(-s1 - s2 + s4)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = -3.0*a*(-s1*s2+s1*s4+s2*s4)*(-t2*t3-t2*t4+t3*t4)
                                J3[0,1,1] = J3[1,0,1] = J3[1,1,0] = -3.0*a*s1*s2*s4*(-t2 - t3 - t4)
                        case 7:
                            # Node 7 at (1, -1/3) - right edge, s4=0 and t2=0
                            N = -3.0*a*s1*s2*s3*t1*t3*t4
                            if deriv >= 1:
                                dNdxi = -3.0*a*t1*t3*t4*(-s1*s2+s1*s3+s2*s3)
                                dNdeta = -3.0*a*s1*s2*s3*(-t1*t3+t1*t4+t3*t4)
                            if deriv >= 2:
                                Hess = np.array([[-3.0*a*t1*t3*t4*(-s1 - s2 + s3), -3.0*a*(-s1*s2+s1*s3+s2*s3)*(-t1*t3+t1*t4+t3*t4)],
                                                     [-3.0*a*(-s1*s2+s1*s3+s2*s3)*(-t1*t3+t1*t4+t3*t4), -3.0*a*s1*s2*s3*(-t1 - t3 + t4)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = -3.0*a*t1*t3*t4*(-s1 - s2 + s3)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = -3.0*a*(-s1*s2+s1*s3+s2*s3)*(-t1*t3+t1*t4+t3*t4)
                                J3[0,1,1] = J3[1,0,1] = J3[1,1,0] = -3.0*a*s1*s2*s3*(-t1 - t3 + t4)
                        case 8:
                            N = -3.0*a*s1*s2*s3*t1*t2*t4
                            if deriv >= 1:
                                dNdxi = -3.0*a*t1*t2*t4*(-s1*s2+s1*s3+s2*s3)
                                dNdeta = -3.0*a *s1*s2*s3*(-t1*t2+t1*t4+t2*t4)
                            if deriv >= 2:
                                Hess = np.array([[-3.0*a*t1*t2*t4*(-s1 - s2 + s3), -3.0*a*(-s1*s2+s1*s3+s2*s3)*(-t1*t2+t1*t4+t2*t4)], 
                                                     [-3.0*a*(-s1*s2+s1*s3+s2*s3)*(-t1*t2+t1*t4+t2*t4), -3.0*a*s1*s2*s3*(-t1 - t2 + t4)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = -3.0*a*t1*t2*t4*(-s1 - s2 + s3)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = -3.0*a*(-s1*s2+s1*s3+s2*s3)*(-t1*t2+t1*t4+t2*t4)
                                J3[0,1,1] = J3[1,0,1] = J3[1,1,0] = -3.0*a*s1*s2*s3*(-t1 - t2 + t4)
                        case 9:
                            N = -3.0*a*s1*s2*s4*t1*t2*t3  
                            if deriv >= 1:
                                dNdxi = -3.0*a*t1*t2*t3*(-s1*s2+s1*s4+s2*s4)
                                dNdeta = -3.0*a *s1*s2*s4*(-t1*t2+t1*t3+t2*t3)
                            if deriv >= 2:
                                Hess = np.array([[-3.0*a*t1*t2*t3*(-s1 - s2 + s4), -3.0*a*(-s1*s2+s1*s4+s2*s4)*(-t1*t2+t1*t3+t2*t3)], 
                                                     [-3.0*a*(-s1*s2+s1*s4+s2*s4)*(-t1*t2+t1*t3+t2*t3), -3.0*a*s1*s2*s4*(-t1 - t2 + t3)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = -3.0*a*t1*t2*t3*(-s1 - s2 + s4)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = -3.0*a*(-s1*s2+s1*s4+s2*s4)*(-t1*t2+t1*t3+t2*t3)
                                J3[0,1,1] = J3[1,0,1] = J3[1,1,0] = -3.0*a*s1*s2*s4*(-t1 - t2 + t3)
                        case 10:
                            N = -3.0*a*s1*s3*s4*t1*t2*t3 
                            if deriv >= 1:
                                dNdxi = -3.0*a*t1*t2*t3*(-s1*s3-s1*s4+s3*s4)
                                dNdeta = -3.0*a *s1*s3*s4*(-t1*t2+t1*t3+t2*t3)
                            if deriv >= 2:
                                Hess = np.array([[-3.0*a*t1*t2*t3*(-s1 - s3 + s4), -3.0*a*(-s1*s3-s1*s4+s3*s4)*(-t1*t2+t1*t3+t2*t3)], 
                                                     [-3.0*a*(-s1*s3-s1*s4+s3*s4)*(-t1*t2+t1*t3+t2*t3), -3.0*a*s1*s3*s4*(-t1 - t2 + t3)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = -3.0*a*t1*t2*t3*(-s1 - s3 + s4)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = -3.0*a*(-s1*s3-s1*s4+s3*s4)*(-t1*t2+t1*t3+t2*t3)
                                J3[0,1,1] = J3[1,0,1] = J3[1,1,0] = -3.0*a*s1*s3*s4*(-t1 - t2 + t3)
                        case 11:
                            N = -3.0*a*s2*s3*s4*t1*t2*t4
                            if deriv >= 1:
                                dNdxi = -3.0*a*t1*t2*t4*(-s2*s3-s2*s4+s3*s4)
                                dNdeta = -3.0*a *s2*s3*s4*(-t1*t2+t1*t4+t2*t4)
                            if deriv >= 2:
                                Hess = np.array([[-3.0*a*t1*t2*t4*(-s2 - s3 + s4), -3.0*a*(-s2*s3-s2*s4+s3*s4)*(-t1*t2+t1*t4+t2*t4)], 
                                                     [-3.0*a*(-s2*s3-s2*s4+s3*s4)*(-t1*t2+t1*t4+t2*t4), -3.0*a*s2*s3*s4*(-t1 - t2 + t4)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = -3.0*a*t1*t2*t4*(-s2 - s3 + s4)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = -3.0*a*(-s2*s3-s2*s4+s3*s4)*(-t1*t2+t1*t4+t2*t4)
                                J3[0,1,1] = J3[1,0,1] = J3[1,1,0] = -3.0*a*s2*s3*s4*(-t1 - t2 + t4)
                        case 12:
                            N = -3.0*a*s2*s3*s4*t1*t3*t4
                            if deriv >= 1:
                                dNdxi = -3.0*a*t1*t3*t4*(-s2*s3-s2*s4+s3*s4)
                                dNdeta = -3.0*a *s2*s3*s4*(-t1*t3-t1*t4+t3*t4)
                            if deriv >= 2:
                                Hess = np.array([[-3.0*a*t1*t3*t4*(-s2 - s3 + s4), -3.0*a*(-s2*s3-s2*s4+s3*s4)*(-t1*t3-t1*t4+t3*t4)], 
                                                     [-3.0*a*(-s2*s3-s2*s4+s3*s4)*(-t1*t3-t1*t4+t3*t4), -3.0*a*s2*s3*s4*(-t1 - t3 + t4)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = -3.0*a*t1*t3*t4*(-s2 - s3 + s4)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = -3.0*a*(-s2*s3-s2*s4+s3*s4)*(-t1*t3-t1*t4+t3*t4)
                                J3[0,1,1] = J3[1,0,1] = J3[1,1,0] = -3.0*a*s2*s3*s4*(-t1 - t3 + t4)
                        case 13:
                            N = 9.0*a*s1*s3*s4*t1*t3*t4
                            if deriv >= 1:
                                dNdxi = 9.0*a*t1*t3*t4*(-s1*s3-s1*s4+s3*s4)
                                dNdeta = 9.0*a *s1*s3*s4*(-t1*t3-t1*t4+t3*t4)
                            if deriv >= 2:
                                Hess = np.array([[9.0*a*t1*t3*t4*(-s1 - s3 + s4), 9.0*a*(-s1*s3-s1*s4+s3*s4)*(-t1*t3-t1*t4+t3*t4)], 
                                                     [9.0*a*(-s1*s3-s1*s4+s3*s4)*(-t1*t3-t1*t4+t3*t4), 9.0*a*s1*s3*s4*(-t1 - t3 + t4)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = 9.0*a*t1*t3*t4*(-s1 - s3 + s4)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = 9.0*a*(-s1*s3-s1*s4+s3*s4)*(-t1*t3-t1*t4+t3*t4)
                                J3[0,1,1] = J3[1,0,1] = J3[1,1,0] = 9.0*a*s1*s3*s4*(-t1 - t3 + t4)
                        case 14:
                            N = 9.0*a*s1*s2*s4*t1*t3*t4
                            if deriv >= 1:
                                dNdxi = 9.0*a*t1*t3*t4*(-s1*s2+s1*s4+s2*s4)
                                dNdeta = 9.0*a *s1*s2*s4*(-t1*t3-t1*t4+t3*t4)
                            if deriv >= 2:
                                Hess = np.array([[9.0*a*t1*t3*t4*(-s1 - s2 + s4), 9.0*a*(-s1*s2-s1*s4+s2*s4)*(-t1*t3-t1*t4+t3*t4)], 
                                                     [9.0*a*(-s1*s2-s1*s4+s2*s4)*(-t1*t3-t1*t4+t3*t4), 9.0*a*s1*s2*s4*(-t1 - t3 + t4)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = 9.0*a*t1*t3*t4*(-s1 - s2 + s4)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = 9.0*a*(-s1*s2-s1*s4+s2*s4)*(-t1*t3-t1*t4+t3*t4)
                                J3[0,1,1] = J3[1,0,1] = J3[1,1,0] = 9.0*a*s1*s2*s4*(-t1 - t3 + t4)
                        case 15:
                            N = 9.0*a*s1*s2*s4*t1*t2*t4
                            if deriv >= 1:
                                dNdxi = 9.0*a*t1*t2*t4*(-s1*s2+s1*s4+s2*s4)
                                dNdeta = 9.0*a *s1*s2*s4*(-t1*t2+t1*t4+t2*t4)
                            if deriv >= 2:
                                Hess = np.array([[9.0*a*t1*t2*t4*(-s1 - s2 + s4), 9.0*a*(-s1*s2-s1*s4+s2*s4)*(-t1*t2-t1*t4+t2*t4)], 
                                                     [9.0*a*(-s1*s2-s1*s4+s2*s4)*(-t1*t2-t1*t4+t2*t4), 9.0*a*s1*s2*s4*(-t1 - t2 + t4)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = 9.0*a*t1*t2*t4*(-s1 - s2 + s4)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = 9.0*a*(-s1*s2-s1*s4+s2*s4)*(-t1*t2-t1*t4+t2*t4)
                                J3[0,1,1] = J3[1,0,1] = J3[1,1,0] = 9.0*a*s1*s2*s4*(-t1 - t2 + t4)
                        case 16:
                            N = 9.0*a*s1*s3*s4*t1*t2*t4
                            if deriv >= 1:
                                dNdxi = 9.0*a*t1*t2*t4*(-s1*s3-s1*s4+s3*s4)
                                dNdeta = 9.0*a *s1*s3*s4*(-t1*t2+t1*t4+t2*t4)
                            if deriv >= 2:
                                Hess = np.array([[9.0*a*t1*t2*t4*(-s1 - s3 + s4), 9.0*a*(-s1*s3-s1*s4+s3*s4)*(-t1*t2-t1*t4+t2*t4)], 
                                                     [9.0*a*(-s1*s3-s1*s4+s3*s4)*(-t1*t2-t1*t4+t2*t4), 9.0*a*s1*s3*s4*(-t1 - t2 + t4)]])
                            if deriv >= 3:
                                J3 = np.zeros((2,2,2))
                                J3[0,0,0] = 9.0*a*t1*t2*t4*(-s1 - s3 + s4)
                                J3[0,0,1] = J3[0,1,0] = J3[1,0,0] = 9.0*a*(-s1*s3-s1*s4+s3*s4)*(-t1*t2-t1*t4+t2*t4)
                                J3[0,1,1] = J3[1,0,1] = J3[1,1,0] = 9.0*a*s1*s3*s4*(-t1 - t2 + t4)
    match deriv:
        case 0:
            return N
        case 1:
            return N, np.array([dNdxi, dNdeta])
        case 2:
            return N, np.array([dNdxi, dNdeta]), Hess
        case 3:
            return N, np.array([dNdxi, dNdeta]), Hess, J3


def EvalRefLagrangeBasis(X, elemType, elemOrder, deriv=1):
    """ 
    Evaluates nodal shape functions in the reference space for the selected element type and order at points defined by coordinates X
    
    Input: 
        - X: coordinates of points on which to evaluate shape functions
        - elemType: 0=line, 1=tri, 2=quad
        - elemOrder: order of element
        - deriv: 0=only shape functions, 1=shape functions and first derivatives, 2=shape functions, first and second derivatives, 3=shape functions, first, second and third derivatives

    Output: 
        - N: shape functions evaluated at points with coordinates X
        - dN: shape functions derivatives up to order deriv evaluated at points with coordinates X
    """
    
    from Element import ElementalNumberOfNodes
    ## NUMBER OF NODAL SHAPE FUNCTIONS
    n, foo = ElementalNumberOfNodes(elemType, elemOrder)
    ## NUMBER OF GAUSS INTEGRATION NODES
    ng = np.shape(X)[0]
    N = np.zeros([ng,n])
    if elemType == 0:
        dim = 1
    else:
        dim = 2
    
    match deriv:
        case 0:
            for i in range(n):
                for ig in range(ng):
                    N[ig,i] = RefLagrangeBasis(X[ig],elemType, elemOrder, i+1, deriv)
            return N
        case 1:
            gradN = np.zeros([ng,n,dim])
            for i in range(n):
                for ig in range(ng):
                    N[ig,i], gradN[ig,i] = RefLagrangeBasis(X[ig],elemType, elemOrder, i+1, deriv)
            return N, [gradN]
        case 2:
            gradN = np.zeros([ng,n,dim])
            HessN = np.zeros([ng,n,dim,dim])
            for i in range(n):
                for ig in range(ng):
                    N[ig,i], gradN[ig,i], HessN[ig,i]  = RefLagrangeBasis(X[ig],elemType, elemOrder, i+1, deriv)
            return N, [gradN, HessN]
        case 3:
            gradN = np.zeros([ng,n,dim])
            HessN = np.zeros([ng,n,dim,dim])
            J3N = np.zeros([ng,n,dim,dim,dim])
            for i in range(n):
                for ig in range(ng):
                    N[ig,i], gradN[ig,i], HessN[ig,i], J3N[ig,i]  = RefLagrangeBasis(X[ig],elemType, elemOrder, i+1, deriv)
            return N, [gradN, HessN, J3N]
        

def Jacobian(X, dN):
    """
    Compute the p-th order derivative of the forward coordinate map  x(ξ)
    at a single Gauss point.
 
    Parameters
    ----------
    X  : (n_nodes, 2)      physical coordinates of the element nodes
    dN : (n_nodes, 2, ...) p-th order reference derivatives of shape functions
                           at ONE Gauss point.  The ellipsis represents p-1
                           additional ref-index axes.
 
    Returns
    -------
    F  : (2, 2, ...)       p-th order forward map derivative tensor
                           F[a, i, j, ...] = ∂^p x_a / ∂ξ_i ∂ξ_j ...
                           (first axis = physical coord, remaining = ref coords)
    """
    return np.einsum('na,n...->a...', X, dN)

def PhysicalGradient(dN, invJ, order = 1, maps = None):
    # ------------------------------------------------------------------ #
    #  ORDER 1  —  exact for any mapping                                 #
    #  Single term: contraction of reference gradient with inverse Jacobian #
    #  dN_phys[n, a] = Σ_i  dN[n,i] · invJ[i,a]                          #
    # ------------------------------------------------------------------ #
    dN_phys1 = np.einsum('ni,ia->na', dN[0], invJ)
    if order == 1:
        return dN_phys1
    
    # ------------------------------------------------------------------ #
    #  ORDER 2                                                           #
    #  Term 1: ref Hessian contracted twice with invJ                    #
    #  Term 2: physical gradient contracted with D                       #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  Correction tensor D                                               #
    #  H_fwd[a, i, j]  =  ∂²x_a / ∂ξ_i ∂ξ_j (map Hessian)                #
    #  D    [k, a, b]  =  ∂²ξ_k / ∂x_a ∂x_b                              #
    # ------------------------------------------------------------------ #
    H_fwd = maps[1]
    D = -np.einsum('kl,lmn,ma,nb->kab', invJ, H_fwd, invJ, invJ)

    term1 = np.einsum('nij,ia,jb->nab', dN[1], invJ, invJ)
    term2 = np.einsum('ni,iab->nab', dN[0], D)
    dN_phys2 = term1 + term2

    if order == 2:
        return dN_phys2 


    # ------------------------------------------------------------------ #
    #  ORDER 3                                                           #
    #  Term A: ref 3rd-deriv contracted three times with invJ            #
    #  Term B: ref Hessian × invJ × D  — three (a,b,c) permutations      #
    #  Term C: physical gradient contracted with E                       #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  Correction tensor E                                               #
    #  E[k, a, b, c] = ∂³ξ_k / ∂x_a ∂x_b ∂x_c = E1 + E2                  #
    #                                                                    #
    #  Two contributions:                                                #
    #    E1: from the T_fwd term (map 3rd order derivative)              #
    #    E2: from differentiating each invJ factor inside D              #
    #        (i.e. ∂invJ/∂x = −invJt · H_fwd · invJ contracted)          #
    # ------------------------------------------------------------------ #
    T_fwd = maps[2]
 
    # E1: T_fwd contracted four times with invJ
    E1 = -np.einsum('kl,lmno,ma,nb,oc->kabc', invJ, T_fwd, invJ, invJ, invJ)

    # E2: differentiate each of the three invJ factors in D
    # ∂invJ[i,a]/∂x_c = −Σ_p invJ[i,p] · D[p,a,c]

    # Use the clean closed-form (avoids index bookkeeping errors):
    #   E = E1  +  three permutations of  (D contracted with D via H_fwd)
    # The three permutations arise because ∂invJ[i,a]/∂x_c = −Σ_p invJ[i,p]·D[p,a,c]
    # applied to each of the three invJ slots in D:
    E2 = (
        np.einsum('kp,pac,lmn,ma,nb->kabc',   invJ, D, H_fwd, invJ, invJ)  # slot 1 (k,l)
      + np.einsum('kl,lmn,pa,pmc,nb->kabc',   invJ, H_fwd, D, invJ, invJ)  # slot 2 (m,a)
      + np.einsum('kl,lmn,ma,pb,pnc->kabc',   invJ, H_fwd, invJ, D, invJ)  # slot 3 (n,b)
    )
 
    E = E1 + E2   # shape: (2, 2, 2, 2)

    termA = np.einsum('nijk,ia,jb,kc->nabc', dN[2], invJ, invJ, invJ)
 
    # Term B: differentiate each of the two invJ in term1 of order-2 w.r.t. x_c
    # gives three symmetry-equivalent permutations of free physical indices
    termB = (
        np.einsum('nij,ip,pac,jb->nabc', dN[1], invJ, D, invJ)   # perm on a
        + np.einsum('nij,ia,jp,pbc->nabc', dN[1], invJ, invJ, D)   # perm on b  
        + np.einsum('nij,ip,pbc,ja->nabc', dN[1], invJ, D, invJ)   # perm on c
        )
 
    termC = np.einsum('ni,iabc->nabc', dN[0], E)

    dN_phys = termA + termB + termC

    return dN_phys
 
 
def _apply_normal(dN_phys, n):
    """
    If a direction vector n is given, contract every physical index with n,
    returning the directional (normal) derivative of each order as a scalar
    per shape function per Gauss point.
 
    For order p the physical tensor has shape (ng, n_nodes, 2, ..., 2) with
    p trailing physical axes.  Contracting with n p times gives (ng, n_nodes).
    """
    if n is None:
        return dN_phys
 
    n = np.asarray(n, dtype=float)
    result = []
    for p, T in enumerate(dN_phys):
        # T shape: (ng, n_nodes, 2, 2, ...) with p+1 trailing physical axes (order = p+1)
        out = T
        for _ in range(p + 1):
            # contract the last axis with n
            out = np.einsum('...a,a->...', out, n)
        result.append(out)   # shape: (ng, n_nodes)
    return result
    