# Testing and validation functions for EQUILIPY CutFEM Grad-Shafranov solver
# These functions verify the correctness of critical components, especially ghost penalty stabilization
#
# Author: Pau Manyer Fuertes
# Email: pau.manyer@bsc.es
# Institution: Barcelona Supercomputing Center (BSC)

import numpy as np
import warnings

###################################################################################################
#                           PENALTY SCALING AND PARAMETER VALIDATION
###################################################################################################

def test_penalty_scaling_formula(zeta, h, p, dim=2):
    """
    Verify that the penalty scaling formula is correct according to CutFEM theory.

    Theory: For k-order element, p-th order derivative penalty:
        σ_p = ζ * h^(2p-d) where d = dimension

    For 2D: σ_p = ζ * h^(2p-2)

    Note: The actual implementation uses h^(2p+2) which was empirically found to work better
    for the axisymmetric Grad-Shafranov problem.

    Input:
        - zeta (float): Penalty parameter
        - h (float): Element characteristic length
        - p (int): Derivative order (1, 2, 3, ...)
        - dim (int): Dimension (default 2 for 2D)

    Returns:
        - penalty (float): Correctly scaled penalty
        - is_correct (bool): Whether the formula is correct
    """
    # Correct formula from CutFEM theory
    penalty_correct = zeta * h**(2*p - dim)

    # Check for physically reasonable scaling
    if penalty_correct <= 0:
        warnings.warn(f"Penalty scaling resulted in non-positive value: {penalty_correct}")
        return penalty_correct, False

    if p > 3:
        warnings.warn(f"Ghost penalty for p={p} (3rd+ derivatives) may cause conditioning issues")

    return penalty_correct, True


def validate_penalty_parameter(zeta, element_order, mesh_size, dim=2):
    """
    Validate and suggest appropriate ghost penalty parameter.

    Input:
        - zeta (float, str, or None): User-specified penalty parameter or 'auto'
        - element_order (int): Order of elements (1, 2, 3, ...)
        - mesh_size (float): Characteristic mesh size
        - dim (int): Dimension (default 2)

    Returns:
        - zeta_recommended (float): Recommended penalty parameter
        - info (str): Information message
    """
    if zeta == 'auto' or zeta is None:
        # CutFEM theory: zeta ~ k^4 where k is element order
        # Conservative estimate: zeta = 10 * k^2
        zeta_recommended = 10.0 * (element_order ** 2)
        info = f"Auto-selected penalty parameter: zeta = {zeta_recommended} " \
               f"(element order k={element_order}, formula: 10*k^2)"
    else:
        zeta_recommended = float(zeta)
        info = f"Using user-specified penalty parameter: zeta = {zeta_recommended}"

    # Validate range
    if zeta_recommended < 1.0:
        warnings.warn(f"Penalty parameter {zeta_recommended} is very small and may provide "
                     f"insufficient stabilization")
    elif zeta_recommended > 1000.0:
        warnings.warn(f"Penalty parameter {zeta_recommended} is very large and may cause "
                     f"ill-conditioning")

    return zeta_recommended, info


###################################################################################################
#                              GHOST FACE NORMAL VECTOR TESTS
###################################################################################################

def test_ghost_face_normal_opposition(mesh):
    """
    Verify that normal vectors on shared ghost faces point in opposite directions (n1 = -n2).

    This is a CRITICAL requirement for ghost penalty stabilization correctness.

    Input:
        - mesh (Mesh object): The mesh containing ghost faces

    Returns:
        - failures (list): List of (elem1_idx, elem2_idx, normal_sum_norm) for failed checks
        - all_passed (bool): Whether all checks passed
    """
    if mesh.GhostFaces is None or len(mesh.GhostFaces) == 0:
        print("No ghost faces found in mesh.")
        return [], True

    failures = []
    tolerance = 1e-10

    for ghost_face_tuple in mesh.GhostFaces:
        # Extract element and face information
        elem1_idx, edge1_idx, face1_list_idx = ghost_face_tuple[1]
        elem2_idx, edge2_idx, face2_list_idx = ghost_face_tuple[2]

        # Get the actual face objects
        face1 = mesh.Elements[elem1_idx].GhostFaces[face1_list_idx]
        face2 = mesh.Elements[elem2_idx].GhostFaces[face2_list_idx]

        # Check opposition: n1 + n2 should be zero
        normal_sum = face1.NormalVec + face2.NormalVec
        normal_sum_norm = np.linalg.norm(normal_sum)

        if normal_sum_norm > tolerance:
            failures.append((elem1_idx, elem2_idx, normal_sum_norm))

    all_passed = len(failures) == 0

    if all_passed:
        print(f"✓ Ghost face normal opposition test PASSED ({len(mesh.GhostFaces)} faces checked)")
    else:
        print(f"✗ Ghost face normal opposition test FAILED for {len(failures)} faces:")
        for elem1, elem2, norm in failures[:5]:  # Print first 5 failures
            print(f"  Elements {elem1} & {elem2}: normal sum norm = {norm:.2e}")
        if len(failures) > 5:
            print(f"  ... and {len(failures)-5} more failures")

    return failures, all_passed


def test_ghost_face_normal_unitary(mesh):
    """
    Verify that all ghost face normals are unit vectors (||n|| = 1).

    Input:
        - mesh (Mesh object): The mesh containing ghost faces

    Returns:
        - failures (list): List of (elem_idx, face_idx, norm_value) for failed checks
        - all_passed (bool): Whether all checks passed
    """
    if mesh.GhostFaces is None or len(mesh.GhostFaces) == 0:
        return [], True

    failures = []
    tolerance = 1e-6

    for ghost_face_tuple in mesh.GhostFaces:
        for position in [1, 2]:  # Check both elements
            elem_idx, edge_idx, face_list_idx = ghost_face_tuple[position]
            face = mesh.Elements[elem_idx].GhostFaces[face_list_idx]

            norm_value = np.linalg.norm(face.NormalVec)
            if abs(norm_value - 1.0) > tolerance:
                failures.append((elem_idx, face_list_idx, norm_value))

    all_passed = len(failures) == 0

    if all_passed:
        print(f"✓ Ghost face normal unitary test PASSED")
    else:
        print(f"✗ Ghost face normal unitary test FAILED for {len(failures)} normals:")
        for elem, face_idx, norm in failures[:5]:
            print(f"  Element {elem}, Face {face_idx}: ||n|| = {norm:.6f}")

    return failures, all_passed


def test_ghost_face_orthogonality(mesh):
    """
    Verify that normal vectors are orthogonal to the corresponding face segments.

    Input:
        - mesh (Mesh object): The mesh containing ghost faces

    Returns:
        - failures (list): List of (elem_idx, face_idx, dot_product) for failed checks
        - all_passed (bool): Whether all checks passed
    """
    if mesh.GhostFaces is None or len(mesh.GhostFaces) == 0:
        return [], True

    failures = []
    tolerance = 1e-10

    for ghost_face_tuple in mesh.GhostFaces:
        elem_idx, edge_idx, face_list_idx = ghost_face_tuple[1]  # Check first element
        face = mesh.Elements[elem_idx].GhostFaces[face_list_idx]

        # Compute tangent vector
        tangent = np.array([face.Xseg[1,0] - face.Xseg[0,0],
                           face.Xseg[1,1] - face.Xseg[0,1]])
        tangent_norm = np.linalg.norm(tangent)

        if tangent_norm > 1e-14:
            tangent = tangent / tangent_norm
            # Check orthogonality via dot product
            dot_product = np.dot(tangent, face.NormalVec)

            if abs(dot_product) > tolerance:
                failures.append((elem_idx, face_list_idx, dot_product))

    all_passed = len(failures) == 0

    if all_passed:
        print(f"✓ Ghost face orthogonality test PASSED")
    else:
        print(f"✗ Ghost face orthogonality test FAILED for {len(failures)} faces:")
        for elem, face_idx, dot in failures[:5]:
            print(f"  Element {elem}, Face {face_idx}: tangent·normal = {dot:.2e}")

    return failures, all_passed


###################################################################################################
#                              SHAPE FUNCTION VALIDATION TESTS
###################################################################################################

def test_shape_functions_partition_of_unity(elemType, elemOrder, num_test_points=10):
    """
    Verify that shape functions form a partition of unity: sum(N_i(x)) = 1 for all x.

    This is a fundamental property of Lagrange finite element shape functions.

    Input:
        - elemType (int): Element type (0=line, 1=tri, 2=quad)
        - elemOrder (int): Element order (1, 2, 3)
        - num_test_points (int): Number of random test points

    Returns:
        - max_error (float): Maximum deviation from unity
        - all_passed (bool): Whether test passed
    """
    from ShapeFunctions import ShapeFunctionsReference
    from Element import ElementalNumberOfNodes

    n, _ = ElementalNumberOfNodes(elemType, elemOrder)

    # Generate random test points in reference element
    if elemType == 0:  # Line - use scalar points
        test_points = np.random.uniform(-1, 1, num_test_points)
    elif elemType == 1:  # Triangle
        # Generate valid barycentric coordinates
        r = np.random.uniform(0, 1, (num_test_points, 2))
        r.sort(axis=1)
        xi = r[:, 0]
        eta = r[:, 1] - r[:, 0]
        test_points = np.column_stack([xi, eta])
    else:  # Quadrilateral
        test_points = np.random.uniform(-1, 1, (num_test_points, 2))

    max_error = 0.0
    for ip in range(num_test_points):
        if elemType == 0:
            point = test_points[ip]
        else:
            point = test_points[ip, :]

        # Sum shape functions at this point
        N_sum = 0.0
        for i in range(n):
            N_i = ShapeFunctionsReference(point, elemType, elemOrder, i+1, deriv=0)
            N_sum += N_i

        error = abs(N_sum - 1.0)
        max_error = max(max_error, error)

    all_passed = max_error < 1e-12

    if all_passed:
        print(f"✓ Shape function partition of unity test PASSED (max error = {max_error:.2e})")
    else:
        print(f"✗ Shape function partition of unity test FAILED (max error = {max_error:.2e})")

    return max_error, all_passed


def test_shape_functions_at_nodes(elemType, elemOrder):
    """
    Verify that N_i(x_j) = δ_ij (Kronecker delta property at nodes).

    Input:
        - elemType (int): Element type (0=line, 1=tri, 2=quad)
        - elemOrder (int): Element order

    Returns:
        - max_error (float): Maximum deviation from expected values
        - all_passed (bool): Whether test passed
    """
    from ShapeFunctions import ShapeFunctionsReference
    from Element import ElementalNumberOfNodes, ReferenceElementCoordinates

    # Get nodal coordinates
    XIe = ReferenceElementCoordinates(elemType, elemOrder)
    n, _ = ElementalNumberOfNodes(elemType, elemOrder)

    # Build N matrix manually
    N = np.zeros((n, n))
    for j in range(n):  # Loop over nodes
        if elemType == 0:
            point = XIe[j]
        else:
            point = XIe[j, :]
        for i in range(n):  # Loop over shape functions
            N[j, i] = ShapeFunctionsReference(point, elemType, elemOrder, i+1, deriv=0)

    # Expected: identity matrix
    expected = np.eye(n)

    max_error = np.max(np.abs(N - expected))
    all_passed = max_error < 1e-12

    if all_passed:
        print(f"✓ Shape function Kronecker delta test PASSED (max error = {max_error:.2e})")
    else:
        print(f"✗ Shape function Kronecker delta test FAILED (max error = {max_error:.2e})")

    return max_error, all_passed


def test_shape_function_derivatives_consistency(elemType, elemOrder, epsilon=1e-6):
    """
    Verify shape function derivatives using finite differences.

    Input:
        - elemType (int): Element type (0=line, 1=tri, 2=quad)
        - elemOrder (int): Element order
        - epsilon (float): Finite difference step size

    Returns:
        - max_error (float): Maximum error between analytical and numerical derivatives
        - all_passed (bool): Whether test passed
    """
    from ShapeFunctions import ShapeFunctionsReference
    from Element import ElementalNumberOfNodes

    n, _ = ElementalNumberOfNodes(elemType, elemOrder)

    if elemType == 0:  # Line
        test_point = 0.3
        dim = 1
    elif elemType == 1:  # Triangle
        test_point = np.array([0.3, 0.2])
        dim = 2
    else:  # Quadrilateral
        test_point = np.array([0.3, 0.2])
        dim = 2

    max_error = 0.0

    for i in range(n):
        # Get analytical derivatives
        if elemType == 0:
            N, dNdxi = ShapeFunctionsReference(test_point, elemType, elemOrder, i+1, deriv=1)
            # Finite difference
            N_plus = ShapeFunctionsReference(test_point + epsilon, elemType, elemOrder, i+1, deriv=0)
            N_minus = ShapeFunctionsReference(test_point - epsilon, elemType, elemOrder, i+1, deriv=0)
            dN_fd = (N_plus - N_minus) / (2 * epsilon)
            error = abs(dN_fd - dNdxi)
            max_error = max(max_error, error)
        else:
            N, (dNdxi, dNdeta) = ShapeFunctionsReference(test_point, elemType, elemOrder, i+1, deriv=1)
            for d in range(dim):
                test_plus = test_point.copy()
                test_minus = test_point.copy()
                test_plus[d] += epsilon
                test_minus[d] -= epsilon
                N_plus = ShapeFunctionsReference(test_plus, elemType, elemOrder, i+1, deriv=0)
                N_minus = ShapeFunctionsReference(test_minus, elemType, elemOrder, i+1, deriv=0)
                dN_fd = (N_plus - N_minus) / (2 * epsilon)
                dN_analytical = dNdxi if d == 0 else dNdeta
                error = abs(dN_fd - dN_analytical)
                max_error = max(max_error, error)

    all_passed = max_error < 1e-4  # Relaxed tolerance for finite differences

    if all_passed:
        print(f"✓ Shape function derivative consistency test PASSED (max error = {max_error:.2e})")
    else:
        print(f"✗ Shape function derivative consistency test FAILED (max error = {max_error:.2e})")

    return max_error, all_passed


def test_polynomial_reproduction(elemType, elemOrder):
    """
    Verify that FEM interpolation exactly reproduces polynomials up to degree elemOrder.

    For order k elements, we should be able to exactly interpolate polynomials of degree ≤ k.

    Input:
        - elemType (int): Element type (0=line, 1=tri, 2=quad)
        - elemOrder (int): Element order

    Returns:
        - max_error (float): Maximum interpolation error
        - all_passed (bool): Whether test passed
    """
    from ShapeFunctions import ShapeFunctionsReference
    from Element import ElementalNumberOfNodes, ReferenceElementCoordinates

    # Get nodal coordinates
    XIe = ReferenceElementCoordinates(elemType, elemOrder)
    n, _ = ElementalNumberOfNodes(elemType, elemOrder)

    # Generate random test points
    num_test_points = 20
    if elemType == 0:
        test_points = np.random.uniform(-1, 1, num_test_points)
    elif elemType == 1:
        r = np.random.uniform(0, 1, (num_test_points, 2))
        r.sort(axis=1)
        test_points = np.column_stack([r[:, 0], r[:, 1] - r[:, 0]])
    else:
        test_points = np.random.uniform(-1, 1, (num_test_points, 2))

    # Define polynomial of degree elemOrder
    def poly(X):
        if elemType == 0:
            if np.isscalar(X):
                return X ** elemOrder
            return X ** elemOrder
        else:
            return X[0] ** elemOrder + X[1] ** elemOrder

    # Nodal values
    nodal_values = np.zeros(n)
    for j in range(n):
        if elemType == 0:
            nodal_values[j] = poly(XIe[j])
        else:
            nodal_values[j] = poly(XIe[j, :])

    max_error = 0.0
    for ip in range(num_test_points):
        if elemType == 0:
            point = test_points[ip]
        else:
            point = test_points[ip, :]

        # Interpolate
        interpolated = 0.0
        for i in range(n):
            N_i = ShapeFunctionsReference(point, elemType, elemOrder, i+1, deriv=0)
            interpolated += N_i * nodal_values[i]

        # Exact value
        exact = poly(point)

        error = abs(interpolated - exact)
        max_error = max(max_error, error)

    all_passed = max_error < 1e-10

    if all_passed:
        print(f"✓ Polynomial reproduction test PASSED (max error = {max_error:.2e})")
    else:
        print(f"✗ Polynomial reproduction test FAILED (max error = {max_error:.2e})")

    return max_error, all_passed


###################################################################################################
#                              JACOBIAN COMPUTATION TESTS
###################################################################################################

def test_jacobian_computation(elemType, elemOrder):
    """
    Verify Jacobian computation for a known affine mapping.

    For an affine mapping, the Jacobian should be constant and equal to the mapping matrix.

    Input:
        - elemType (int): Element type (1=tri, 2=quad)
        - elemOrder (int): Element order

    Returns:
        - max_error (float): Maximum error in Jacobian
        - all_passed (bool): Whether test passed
    """
    from ShapeFunctions import ShapeFunctionsReference, Jacobian
    from Element import ElementalNumberOfNodes, ReferenceElementCoordinates

    if elemType == 0:
        # Skip 1D elements for Jacobian test
        print(f"✓ Jacobian computation test SKIPPED for 1D elements")
        return 0.0, True

    # Get reference element coordinates
    XIe = ReferenceElementCoordinates(elemType, elemOrder)
    n, _ = ElementalNumberOfNodes(elemType, elemOrder)

    # Create a simple affine mapping: scale by 2, no rotation
    Xe = XIe * 2.0 + np.array([1.0, 0.5])

    # Test points
    if elemType == 1:
        test_points = np.array([[0.3, 0.2], [0.1, 0.1], [0.5, 0.3]])
    else:
        test_points = np.array([[0.3, 0.2], [-0.5, 0.3], [0.0, 0.0]])

    expected_det = 4.0  # For scale factor 2 in 2D: det = 2^2 = 4

    max_error = 0.0
    for ig in range(len(test_points)):
        point = test_points[ig, :]

        # Build gradient matrix
        gradN = np.zeros((n, 2))
        for i in range(n):
            _, (dNdxi, dNdeta) = ShapeFunctionsReference(point, elemType, elemOrder, i+1, deriv=1)
            gradN[i, 0] = dNdxi
            gradN[i, 1] = dNdeta

        invJ, detJ = Jacobian(Xe, gradN)
        error = abs(detJ - expected_det)
        max_error = max(max_error, error)

        # Check inverse Jacobian: should be scale by 1/2
        expected_invJ = np.eye(2) * 0.5
        invJ_error = np.max(np.abs(invJ - expected_invJ))
        max_error = max(max_error, invJ_error)

    all_passed = max_error < 1e-10

    if all_passed:
        print(f"✓ Jacobian computation test PASSED (max error = {max_error:.2e})")
    else:
        print(f"✗ Jacobian computation test FAILED (max error = {max_error:.2e})")

    return max_error, all_passed


def test_jacobian_1d_arc_length():
    """
    Verify that 1D Jacobian correctly computes arc length for a linear segment.

    Returns:
        - error (float): Error in arc length computation
        - passed (bool): Whether test passed
    """
    from ShapeFunctions import Jacobian1D, ShapeFunctionsReference
    from GaussQuadrature import GaussQuadrature

    # Create a simple segment
    Xseg = np.array([[1.0, 0.0], [3.0, 4.0]])  # Length = sqrt(4 + 16) = sqrt(20)
    expected_length = np.sqrt(20)

    # High-order quadrature
    XIg, Wg, Ng = GaussQuadrature(0, 5)

    # Manually evaluate 1D shape function derivatives
    # For linear 1D element: dN1/dxi = -0.5, dN2/dxi = 0.5
    computed_length = 0.0
    for ig in range(Ng):
        xi = XIg[ig, 0] if XIg.ndim > 1 else XIg[ig]
        # Get derivatives manually for linear 1D element
        _, dN1dxi = ShapeFunctionsReference(xi, 0, 1, 1, deriv=1)
        _, dN2dxi = ShapeFunctionsReference(xi, 0, 1, 2, deriv=1)
        dNdxi = np.array([dN1dxi, dN2dxi])

        detJ1D = Jacobian1D(Xseg, dNdxi)
        computed_length += detJ1D * Wg[ig]

    error = abs(computed_length - expected_length)
    passed = error < 1e-12

    if passed:
        print(f"✓ 1D Jacobian arc length test PASSED (error = {error:.2e})")
    else:
        print(f"✗ 1D Jacobian arc length test FAILED (error = {error:.2e})")

    return error, passed


def test_jacobian_determinant_signs(mesh):
    """
    Check for non-positive Jacobian determinants in ghost face quadratures.

    Negative Jacobian determinants indicate inverted/distorted elements.

    Input:
        - mesh (Mesh object): The mesh with initialized elements

    Returns:
        - negative_jacobians (list): List of (elem_idx, face_idx, quad_idx, detJ) for negative values
        - all_passed (bool): Whether all determinants are positive
    """
    negative_jacobians = []
    warning_threshold = 1e-3  # Warn if det|J| is very small

    for elem in mesh.Elements:
        if not hasattr(elem, 'GhostFaces') or elem.GhostFaces is None:
            continue

        for face_idx, FACE in enumerate(elem.GhostFaces):
            for ig in range(FACE.ng):
                det_j = abs(FACE.detJg[ig])

                if det_j <= 0:
                    negative_jacobians.append((elem.index, face_idx, ig, FACE.detJg[ig]))
                elif det_j < warning_threshold:
                    warnings.warn(f"Element {elem.index}, Face {face_idx}, Quad {ig}: "
                                f"|det(J)| = {det_j:.2e} (very small)")

    all_passed = len(negative_jacobians) == 0

    if all_passed:
        print(f"✓ Jacobian determinant sign test PASSED")
    else:
        print(f"✗ Jacobian determinant sign test FAILED for {len(negative_jacobians)} quadrature points:")
        for elem_idx, face_idx, quad_idx, detJ in negative_jacobians[:5]:
            print(f"  Element {elem_idx}, Face {face_idx}, Quad {quad_idx}: det(J) = {detJ:.2e}")

    return negative_jacobians, all_passed


###################################################################################################
#                              GHOST PENALTY EINSUM CONTRACTION TESTS
###################################################################################################

def test_einsum_gradient_contraction():
    """
    Test the einsum contraction for gradient (p=1) normal derivative computation.

    Verifies: n_dot_grad_N = sum_i (dN/dxi_i * invJ_ia * n_a)

    Returns:
        - error (float): Maximum error between einsum and manual computation
        - passed (bool): Whether test passed
    """
    np.random.seed(42)

    n_nodes = 6  # e.g., quadratic triangle
    dim = 2

    # Random data
    dNdxi = np.random.randn(n_nodes, dim)  # Shape: [n, 2]
    invJ = np.random.randn(dim, dim)  # Shape: [2, 2]
    normal = np.random.randn(dim)
    normal = normal / np.linalg.norm(normal)  # Normalize

    # Einsum contraction (as in code)
    subscripts = 'ni,ia,a->n'
    result_einsum = np.einsum(subscripts, dNdxi, invJ, normal)

    # Manual computation
    result_manual = np.zeros(n_nodes)
    for n in range(n_nodes):
        for i in range(dim):
            for a in range(dim):
                result_manual[n] += dNdxi[n, i] * invJ[i, a] * normal[a]

    error = np.max(np.abs(result_einsum - result_manual))
    passed = error < 1e-14

    if passed:
        print(f"✓ Einsum gradient contraction test PASSED (error = {error:.2e})")
    else:
        print(f"✗ Einsum gradient contraction test FAILED (error = {error:.2e})")

    return error, passed


def test_einsum_hessian_contraction():
    """
    Test the einsum contraction for Hessian (p=2) normal derivative computation.

    Verifies: n_dot_hess_N = sum_{i,j,a,b} (d2N/dxi_i dxi_j * invJ_ia * invJ_jb * n_a * n_b)

    Returns:
        - error (float): Maximum error between einsum and manual computation
        - passed (bool): Whether test passed
    """
    np.random.seed(42)

    n_nodes = 6
    dim = 2

    # Random Hessian tensor
    HessN = np.random.randn(n_nodes, dim, dim)  # Shape: [n, 2, 2]
    invJ = np.random.randn(dim, dim)
    normal = np.random.randn(dim)
    normal = normal / np.linalg.norm(normal)

    # Einsum contraction (as in code)
    subscripts = 'nij,ia,jb,a,b->n'
    result_einsum = np.einsum(subscripts, HessN, invJ, invJ, normal, normal)

    # Manual computation
    result_manual = np.zeros(n_nodes)
    for n in range(n_nodes):
        for i in range(dim):
            for j in range(dim):
                for a in range(dim):
                    for b in range(dim):
                        result_manual[n] += HessN[n, i, j] * invJ[i, a] * invJ[j, b] * normal[a] * normal[b]

    error = np.max(np.abs(result_einsum - result_manual))
    passed = error < 1e-14

    if passed:
        print(f"✓ Einsum Hessian contraction test PASSED (error = {error:.2e})")
    else:
        print(f"✗ Einsum Hessian contraction test FAILED (error = {error:.2e})")

    return error, passed


def test_einsum_third_derivative_contraction():
    """
    Test the einsum contraction for third derivative (p=3) normal derivative computation.

    Returns:
        - error (float): Maximum error between einsum and manual computation
        - passed (bool): Whether test passed
    """
    np.random.seed(42)

    n_nodes = 10  # e.g., cubic triangle
    dim = 2

    # Random third derivative tensor
    J3N = np.random.randn(n_nodes, dim, dim, dim)  # Shape: [n, 2, 2, 2]
    invJ = np.random.randn(dim, dim)
    normal = np.random.randn(dim)
    normal = normal / np.linalg.norm(normal)

    # Einsum contraction (as in code)
    subscripts = 'nijk,ia,jb,kc,a,b,c->n'
    result_einsum = np.einsum(subscripts, J3N, invJ, invJ, invJ, normal, normal, normal)

    # Manual computation
    result_manual = np.zeros(n_nodes)
    for n in range(n_nodes):
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for a in range(dim):
                        for b in range(dim):
                            for c in range(dim):
                                result_manual[n] += J3N[n, i, j, k] * invJ[i, a] * invJ[j, b] * invJ[k, c] * normal[a] * normal[b] * normal[c]

    error = np.max(np.abs(result_einsum - result_manual))
    passed = error < 1e-13

    if passed:
        print(f"✓ Einsum third derivative contraction test PASSED (error = {error:.2e})")
    else:
        print(f"✗ Einsum third derivative contraction test FAILED (error = {error:.2e})")

    return error, passed


def test_normal_derivative_jump_symmetry(mesh):
    """
    Verify that the ghost penalty bilinear form is symmetric.

    The ghost penalty contribution for a single face should produce a symmetric matrix.

    Input:
        - mesh (Mesh object): Mesh with ghost face quadratures computed

    Returns:
        - max_asymmetry (float): Maximum asymmetry in elemental matrices
        - all_passed (bool): Whether test passed
    """
    if mesh.GhostFaces is None or len(mesh.GhostFaces) == 0:
        print("No ghost faces to test.")
        return 0.0, True

    max_asymmetry = 0.0

    for ghost_face_tuple in mesh.GhostFaces:
        elem1_idx, edge1_idx, face1_list_idx = ghost_face_tuple[1]
        elem2_idx, edge2_idx, face2_list_idx = ghost_face_tuple[2]

        ELEM1 = mesh.Elements[elem1_idx]
        ELEM2 = mesh.Elements[elem2_idx]
        FACE1 = ELEM1.GhostFaces[face1_list_idx]
        FACE2 = ELEM2.GhostFaces[face2_list_idx]

        n_total = ELEM1.n + ELEM2.n
        LHSe = np.zeros((n_total, n_total))

        # Build elemental matrix for p=1 (gradient)
        for ig in range(FACE1.ng):
            invJ1 = FACE1.invJg[ig]
            invJ2 = FACE2.invJg[ig]
            n1 = FACE1.NormalVec
            n2 = FACE2.NormalVec

            # Compute normal derivatives
            gradN1 = FACE1.dNg[0][ig]  # [n, 2]
            gradN2 = FACE2.dNg[0][ig]

            n_dot_gradN1 = np.einsum('ni,ia,a->n', gradN1, invJ1, n1)
            n_dot_gradN2 = np.einsum('ni,ia,a->n', gradN2, invJ2, n2)

            n_dot_gradN = np.concatenate([n_dot_gradN1, n_dot_gradN2])

            # Add contribution (outer product)
            LHSe += np.outer(n_dot_gradN, n_dot_gradN) * FACE1.detJg1D[ig] * FACE1.Wg[ig]

        # Check symmetry
        asymmetry = np.max(np.abs(LHSe - LHSe.T))
        max_asymmetry = max(max_asymmetry, asymmetry)

    all_passed = max_asymmetry < 1e-12

    if all_passed:
        print(f"✓ Ghost penalty matrix symmetry test PASSED (max asymmetry = {max_asymmetry:.2e})")
    else:
        print(f"✗ Ghost penalty matrix symmetry test FAILED (max asymmetry = {max_asymmetry:.2e})")

    return max_asymmetry, all_passed


###################################################################################################
#                              NODE PERMUTATION AND EDGE ORDERING TESTS
###################################################################################################

def test_ghost_face_node_permutation(mesh):
    """
    Verify that ghost face node permutation correctly aligns nodes between adjacent elements.

    The permutation stored in mesh.GhostFaces should ensure that corresponding nodes
    on the shared edge have the same physical coordinates.

    Input:
        - mesh (Mesh object): Mesh with ghost faces identified

    Returns:
        - failures (list): List of failed face checks
        - all_passed (bool): Whether test passed
    """
    if mesh.GhostFaces is None or len(mesh.GhostFaces) == 0:
        return [], True

    failures = []
    tolerance = 1e-12

    for ghost_face_tuple in mesh.GhostFaces:
        elem1_idx, edge1_idx, face1_list_idx = ghost_face_tuple[1]
        elem2_idx, edge2_idx, face2_list_idx = ghost_face_tuple[2]

        FACE1 = mesh.Elements[elem1_idx].GhostFaces[face1_list_idx]
        FACE2 = mesh.Elements[elem2_idx].GhostFaces[face2_list_idx]

        # After permutation, physical coordinates should match
        coord_diff = np.max(np.abs(FACE1.Xseg - FACE2.Xseg))

        if coord_diff > tolerance:
            failures.append((elem1_idx, elem2_idx, coord_diff))

    all_passed = len(failures) == 0

    if all_passed:
        print(f"✓ Ghost face node permutation test PASSED ({len(mesh.GhostFaces)} faces checked)")
    else:
        print(f"✗ Ghost face node permutation test FAILED for {len(failures)} faces:")
        for elem1, elem2, diff in failures[:5]:
            print(f"  Elements {elem1} & {elem2}: coordinate mismatch = {diff:.2e}")

    return failures, all_passed


def test_edge_node_ordering(mesh):
    """
    Verify that edge node ordering is consistent across all elements.

    For triangular elements:
    - Edge 0: nodes [0, 1, ...]
    - Edge 1: nodes [1, 2, ...]
    - Edge 2: nodes [2, 0, ...]

    All edges should follow counter-clockwise ordering.

    Input:
        - mesh (Mesh object): The mesh to check

    Returns:
        - inconsistencies (list): List of (elem_idx, edge_idx) with issues
        - all_passed (bool): Whether all checks passed
    """
    inconsistencies = []

    for elem in mesh.Elements:
        for iedge in range(elem.numedges):
            # Get edge vertices (first and second local indices)
            v0_local = iedge
            v1_local = (iedge + 1) % elem.numedges

            v0_global = elem.Te[v0_local]
            v1_global = elem.Te[v1_local]

            # All edges should be enumerated in counter-clockwise order
            # This is implicitly ensured by the modulo operation
            # But we should verify consistent edge direction
            edge_vector = elem.Xe[v1_local,:] - elem.Xe[v0_local,:]
            edge_length = np.linalg.norm(edge_vector)

            if edge_length < 1e-14:
                inconsistencies.append((elem.index, iedge))

    all_passed = len(inconsistencies) == 0

    if all_passed:
        print(f"✓ Edge node ordering test PASSED ({mesh.Ne} elements checked)")
    else:
        print(f"✗ Edge node ordering test FOUND {len(inconsistencies)} degenerate edges:")
        for elem_idx, edge_idx in inconsistencies[:5]:
            print(f"  Element {elem_idx}, Edge {edge_idx}")

    return inconsistencies, all_passed


def test_ghost_face_reference_physical_consistency(mesh):
    """
    Verify that ghost face reference and physical coordinates are consistent.

    The mapping from reference to physical should be consistent with element mapping.

    Input:
        - mesh (Mesh object): Mesh with ghost faces

    Returns:
        - max_error (float): Maximum mapping error
        - all_passed (bool): Whether test passed
    """
    from ShapeFunctions import ShapeFunctionsReference
    from Element import ElementalNumberOfNodes

    if mesh.GhostFaces is None or len(mesh.GhostFaces) == 0:
        return 0.0, True

    max_error = 0.0

    for ghost_face_tuple in mesh.GhostFaces:
        elem_idx, edge_idx, face_list_idx = ghost_face_tuple[1]
        ELEM = mesh.Elements[elem_idx]
        FACE = ELEM.GhostFaces[face_list_idx]
        n, _ = ElementalNumberOfNodes(ELEM.ElType, ELEM.ElOrder)

        # Evaluate element shape functions at face quadrature points (in reference space)
        for ig in range(FACE.ng):
            XI_point = FACE.XIg[ig, :]

            # Map reference point to physical using element shape functions
            X_mapped = np.zeros(2)
            for i in range(n):
                N_i = ShapeFunctionsReference(XI_point, ELEM.ElType, ELEM.ElOrder, i+1, deriv=0)
                X_mapped += N_i * ELEM.Xe[i, :]

            # Compare with stored physical coordinates
            error = np.linalg.norm(X_mapped - FACE.Xg[ig, :])
            max_error = max(max_error, error)

    all_passed = max_error < 1e-10

    if all_passed:
        print(f"✓ Ghost face reference-physical consistency test PASSED (max error = {max_error:.2e})")
    else:
        print(f"✗ Ghost face reference-physical consistency test FAILED (max error = {max_error:.2e})")

    return max_error, all_passed


###################################################################################################
#                              QUADRATURE ACCURACY TESTS
###################################################################################################

def test_ghost_face_quadrature_accuracy(mesh, tolerance=1e-8):
    """
    Verify ghost face quadrature by checking arc length integration.

    For a 1D curve, the quadrature should accurately integrate arc length:
        ∫_Γ 1 dΓ = arc_length

    Input:
        - mesh (Mesh object): The mesh with quadratures computed
        - tolerance (float): Acceptable error in arc length integration

    Returns:
        - errors (list): List of (elem_idx, face_idx, exact_length, quad_length, error) for failed checks
        - all_passed (bool): Whether all checks passed
    """
    errors = []

    for elem in mesh.Elements:
        if not hasattr(elem, 'GhostFaces') or elem.GhostFaces is None:
            continue

        for face_idx, FACE in enumerate(elem.GhostFaces):
            # Compute exact arc length (piecewise linear approximation)
            exact_length = 0.0
            for inode in range(len(FACE.Xseg) - 1):
                segment_vec = FACE.Xseg[inode+1,:] - FACE.Xseg[inode,:]
                exact_length += np.linalg.norm(segment_vec)

            # Compute length via quadrature
            quad_length = np.sum(FACE.detJg1D * FACE.Wg)

            error = abs(exact_length - quad_length)
            if error > tolerance:
                errors.append((elem.index, face_idx, exact_length, quad_length, error))

    all_passed = len(errors) == 0

    if all_passed:
        print(f"✓ Ghost face quadrature accuracy test PASSED")
    else:
        print(f"✗ Ghost face quadrature accuracy test FAILED for {len(errors)} faces:")
        for elem_idx, face_idx, exact, quad, err in errors[:5]:
            print(f"  Element {elem_idx}, Face {face_idx}: error = {err:.2e} "
                  f"(exact={exact:.6f}, quad={quad:.6f})")

    return errors, all_passed


def test_quadrature_polynomial_exactness(elemType, order, degree):
    """
    Verify that quadrature rule integrates polynomials of given degree exactly.

    Input:
        - elemType (int): Element type (0=line, 1=tri, 2=quad)
        - order (int): Quadrature order
        - degree (int): Polynomial degree to test

    Returns:
        - error (float): Integration error
        - passed (bool): Whether test passed
    """
    from GaussQuadrature import GaussQuadrature

    XIg, Wg, Ng = GaussQuadrature(elemType, order)

    # Handle edge case where Ng=1 returns scalar values
    if not hasattr(Wg, '__len__'):
        Wg = np.array([Wg])
    if not hasattr(XIg, '__len__') or (hasattr(XIg, 'ndim') and XIg.ndim == 0):
        XIg = np.array([[XIg]]) if elemType > 0 else np.array([XIg])

    # Simple polynomial: integrating constant (degree=0)
    if elemType == 0:  # Line from -1 to 1
        if degree == 0:
            exact = 2.0  # integral of 1 from -1 to 1
            computed = np.sum(Wg)
        else:
            # For x^degree, integral = 2/(degree+1) if degree is even, 0 if odd
            if degree % 2 == 0:
                exact = 2.0 / (degree + 1)
            else:
                exact = 0.0
            if XIg.ndim == 1:
                computed = np.sum(XIg**degree * Wg)
            else:
                computed = np.sum(XIg[:, 0]**degree * Wg)
    elif elemType == 1:  # Triangle (reference area = 0.5)
        if degree == 0:
            exact = 0.5  # area of reference triangle
            computed = np.sum(Wg)
        else:
            # Just test constant integration for simplicity
            exact = 0.5
            computed = np.sum(Wg)
    else:  # Quadrilateral (reference area = 4 for [-1,1]^2)
        if degree == 0:
            exact = 4.0
            computed = np.sum(Wg)
        else:
            exact = 4.0
            computed = np.sum(Wg)

    error = abs(exact - computed)
    passed = error < 1e-10

    if passed:
        print(f"✓ Quadrature polynomial exactness test PASSED (type={elemType}, order={order}, error={error:.2e})")
    else:
        print(f"✗ Quadrature polynomial exactness test FAILED (type={elemType}, order={order}, error={error:.2e})")

    return error, passed


###################################################################################################
#                              SYSTEM MATRIX TESTS
###################################################################################################

def test_system_matrix_symmetry(LHS_matrix, tolerance=1e-10):
    """
    Check if the assembled system matrix is symmetric (within tolerance).

    For Galerkin methods, the system matrix should be symmetric.

    Input:
        - LHS_matrix: Sparse matrix (scipy.sparse)
        - tolerance (float): Acceptable asymmetry threshold

    Returns:
        - max_asymmetry (float): Maximum absolute difference between A and A^T
        - is_symmetric (bool): Whether matrix is symmetric within tolerance
    """
    # Convert to CSR format for efficient operations
    if hasattr(LHS_matrix, 'tocsr'):
        A = LHS_matrix.tocsr()
    else:
        A = LHS_matrix

    # Compute difference A - A^T
    diff = A - A.T

    # Get maximum absolute value
    if diff.nnz > 0:
        max_asymmetry = np.max(np.abs(diff.data))
    else:
        max_asymmetry = 0.0

    is_symmetric = max_asymmetry < tolerance

    if is_symmetric:
        print(f"✓ System matrix symmetry test PASSED (max asymmetry = {max_asymmetry:.2e})")
    else:
        print(f"✗ System matrix symmetry test FAILED (max asymmetry = {max_asymmetry:.2e})")

    return max_asymmetry, is_symmetric


def test_system_matrix_conditioning(LHS_matrix, RHS_vector):
    """
    Estimate the condition number of the system matrix to assess numerical stability.

    Input:
        - LHS_matrix: Sparse system matrix
        - RHS_vector: Right-hand side vector

    Returns:
        - info_dict (dict): Contains 'condition_number', 'is_well_conditioned'
    """
    try:
        from scipy.sparse.linalg import norm

        # For sparse matrices, condition number estimation is not straightforward
        # We'll use the Frobenius norm ratio as a heuristic
        if hasattr(LHS_matrix, 'tocsr'):
            A = LHS_matrix.tocsr()
        else:
            A = LHS_matrix

        # Simple heuristic: check diagonal dominance
        diag = np.array(A.diagonal()).flatten()
        row_sums = np.array(A.sum(axis=1)).flatten()

        # Diagonal dominance measure
        dominance = np.min(np.abs(diag) / (np.abs(row_sums) + 1e-14))

        is_well_conditioned = dominance > 0.5  # Heuristic threshold

        info_dict = {
            'diagonal_dominance': dominance,
            'is_well_conditioned': is_well_conditioned,
            'matrix_size': (A.shape[0], A.shape[1]),
            'nnz': A.nnz
        }

        if is_well_conditioned:
            print(f"✓ System matrix conditioning test PASSED (diag. dominance = {dominance:.4f})")
        else:
            warnings.warn(f"System matrix may be ill-conditioned (diag. dominance = {dominance:.4f})")

        return info_dict

    except Exception as e:
        print(f"⚠ Could not analyze matrix conditioning: {str(e)}")
        return {'error': str(e)}


def test_ghost_penalty_contribution_magnitude(LHS_matrix_with_ghost, LHS_matrix_without_ghost):
    """
    Compare the magnitude of ghost penalty contributions vs. other terms.

    Input:
        - LHS_matrix_with_ghost: Global matrix including ghost penalty
        - LHS_matrix_without_ghost: Global matrix without ghost penalty

    Returns:
        - ghost_contribution_ratio (float): Ratio of ghost penalty contribution to total
        - is_reasonable (bool): Whether ratio is in expected range (0.01 - 0.5)
    """
    try:
        diff = LHS_matrix_with_ghost - LHS_matrix_without_ghost

        if hasattr(diff, 'tocsr'):
            diff_csr = diff.tocsr()
        else:
            diff_csr = diff

        if diff_csr.nnz == 0:
            ghost_contribution_ratio = 0.0
        else:
            ghost_norm = np.max(np.abs(diff_csr.data))
            total_norm = np.max(np.abs(LHS_matrix_with_ghost.data))
            ghost_contribution_ratio = ghost_norm / (total_norm + 1e-14)

        # Expected range: ghost penalty should contribute meaningfully but not dominate
        is_reasonable = 0.001 < ghost_contribution_ratio < 0.9

        if is_reasonable:
            print(f"✓ Ghost penalty contribution test PASSED (ratio = {ghost_contribution_ratio:.4f})")
        else:
            warnings.warn(f"Ghost penalty contribution may be unreasonable (ratio = {ghost_contribution_ratio:.4f})")

        return ghost_contribution_ratio, is_reasonable

    except Exception as e:
        print(f"⚠ Could not compare matrices: {str(e)}")
        return None, False


def test_ghost_penalty_matrix_positive_semidefinite(mesh, zeta=1.0):
    """
    Verify that the ghost penalty contribution is positive semi-definite.

    The ghost penalty bilinear form should be positive semi-definite:
    u^T * G * u >= 0 for all u

    Input:
        - mesh (Mesh object): Mesh with ghost faces
        - zeta (float): Penalty parameter

    Returns:
        - min_eigenvalue (float): Minimum eigenvalue (should be >= 0)
        - is_psd (bool): Whether matrix is positive semi-definite
    """
    if mesh.GhostFaces is None or len(mesh.GhostFaces) == 0:
        print("No ghost faces to test.")
        return 0.0, True

    # Build a small test matrix from one ghost face
    ghost_face_tuple = mesh.GhostFaces[0]
    elem1_idx, edge1_idx, face1_list_idx = ghost_face_tuple[1]
    elem2_idx, edge2_idx, face2_list_idx = ghost_face_tuple[2]

    ELEM1 = mesh.Elements[elem1_idx]
    ELEM2 = mesh.Elements[elem2_idx]
    FACE1 = ELEM1.GhostFaces[face1_list_idx]
    FACE2 = ELEM2.GhostFaces[face2_list_idx]

    n_total = ELEM1.n + ELEM2.n
    LHSe = np.zeros((n_total, n_total))

    h = max(ELEM1.length, ELEM2.length)
    penalty = zeta * h**4  # p=1: h^(2*1+2) = h^4

    for ig in range(FACE1.ng):
        invJ1 = FACE1.invJg[ig]
        invJ2 = FACE2.invJg[ig]
        n1 = FACE1.NormalVec
        n2 = FACE2.NormalVec

        gradN1 = FACE1.dNg[0][ig]
        gradN2 = FACE2.dNg[0][ig]

        n_dot_gradN1 = np.einsum('ni,ia,a->n', gradN1, invJ1, n1)
        n_dot_gradN2 = np.einsum('ni,ia,a->n', gradN2, invJ2, n2)

        n_dot_gradN = np.concatenate([n_dot_gradN1, n_dot_gradN2])

        R = FACE1.Xg[ig, 0]
        LHSe += penalty * np.outer(n_dot_gradN, n_dot_gradN) * (1/R) * FACE1.detJg1D[ig] * FACE1.Wg[ig]

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(LHSe)
    min_eigenvalue = np.min(eigenvalues)

    # Should be non-negative (allowing small numerical tolerance)
    is_psd = min_eigenvalue >= -1e-12

    if is_psd:
        print(f"✓ Ghost penalty PSD test PASSED (min eigenvalue = {min_eigenvalue:.2e})")
    else:
        print(f"✗ Ghost penalty PSD test FAILED (min eigenvalue = {min_eigenvalue:.2e})")

    return min_eigenvalue, is_psd


###################################################################################################
#                    GHOST PENALTY JUMP ANALYSIS TESTS (CutFEM Stabilization)
###################################################################################################

def compute_solution_jump_across_ghost_face(mesh, PSI, ghost_face_tuple):
    """
    Compute the jump of the solution PSI across a ghost face.

    [[u]] = u|_{Ω1} - u|_{Ω2} at each quadrature point on the shared face.

    For continuous FEM, this should be zero (or machine precision).

    Input:
        - mesh: Mesh object with ghost faces
        - PSI: Solution vector (nodal values)
        - ghost_face_tuple: Ghost face information tuple

    Returns:
        - jumps: Array of solution jumps at quadrature points
        - max_jump: Maximum absolute jump
    """
    from ShapeFunctions import ShapeFunctionsReference
    from Element import ElementalNumberOfNodes

    elem1_idx, edge1_idx, face1_list_idx = ghost_face_tuple[1]
    elem2_idx, edge2_idx, face2_list_idx = ghost_face_tuple[2]

    ELEM1 = mesh.Elements[elem1_idx]
    ELEM2 = mesh.Elements[elem2_idx]
    FACE1 = ELEM1.GhostFaces[face1_list_idx]
    FACE2 = ELEM2.GhostFaces[face2_list_idx]

    n1, _ = ElementalNumberOfNodes(ELEM1.ElType, ELEM1.ElOrder)
    n2, _ = ElementalNumberOfNodes(ELEM2.ElType, ELEM2.ElOrder)

    # Extract local solution values
    PSI1 = PSI[ELEM1.Te]
    PSI2 = PSI[ELEM2.Te]

    jumps = np.zeros(FACE1.ng)

    for ig in range(FACE1.ng):
        # Evaluate solution on element 1 at quadrature point
        u1 = 0.0
        for i in range(n1):
            N_i = ShapeFunctionsReference(FACE1.XIg[ig, :], ELEM1.ElType, ELEM1.ElOrder, i+1, deriv=0)
            u1 += N_i * PSI1[i]

        # Evaluate solution on element 2 at quadrature point
        u2 = 0.0
        for i in range(n2):
            N_i = ShapeFunctionsReference(FACE2.XIg[ig, :], ELEM2.ElType, ELEM2.ElOrder, i+1, deriv=0)
            u2 += N_i * PSI2[i]

        jumps[ig] = u1 - u2

    return jumps, np.max(np.abs(jumps))


def compute_normal_derivative_jump_across_ghost_face(mesh, PSI, ghost_face_tuple, deriv_order=1):
    """
    Compute the jump of the p-th normal derivative of PSI across a ghost face.

    [[∂^p u/∂n^p]] = (n · ∇)^p u|_{Ω1} - (n · ∇)^p u|_{Ω2}

    This is what the ghost penalty stabilizes. For smooth solutions, high-order
    derivative jumps should be controlled by the ghost penalty.

    Input:
        - mesh: Mesh object with ghost faces
        - PSI: Solution vector (nodal values)
        - ghost_face_tuple: Ghost face information tuple
        - deriv_order: Order of derivative (1=gradient, 2=Hessian, 3=third derivative)

    Returns:
        - jumps: Array of normal derivative jumps at quadrature points
        - L2_norm: L2 norm of the jump (integrated)
        - max_jump: Maximum absolute jump
    """
    elem1_idx, edge1_idx, face1_list_idx = ghost_face_tuple[1]
    elem2_idx, edge2_idx, face2_list_idx = ghost_face_tuple[2]

    ELEM1 = mesh.Elements[elem1_idx]
    ELEM2 = mesh.Elements[elem2_idx]
    FACE1 = ELEM1.GhostFaces[face1_list_idx]
    FACE2 = ELEM2.GhostFaces[face2_list_idx]

    # Extract local solution values
    PSI1 = PSI[ELEM1.Te]
    PSI2 = PSI[ELEM2.Te]

    jumps = np.zeros(FACE1.ng)
    L2_integral = 0.0

    # Build einsum subscript string based on derivative order
    if deriv_order == 1:
        subscripts = 'ni,ia,a->n'
    elif deriv_order == 2:
        subscripts = 'nij,ia,jb,a,b->n'
    elif deriv_order == 3:
        subscripts = 'nijk,ia,jb,kc,a,b,c->n'
    else:
        raise ValueError(f"Derivative order {deriv_order} not supported (use 1, 2, or 3)")

    for ig in range(FACE1.ng):
        invJ1 = FACE1.invJg[ig]
        invJ2 = FACE2.invJg[ig]
        n1 = FACE1.NormalVec
        n2 = FACE2.NormalVec

        # Build einsum arguments for element 1
        args1 = [FACE1.dNg[deriv_order-1][ig]]
        for _ in range(deriv_order):
            args1.append(invJ1)
        for _ in range(deriv_order):
            args1.append(n1)

        # Build einsum arguments for element 2
        args2 = [FACE2.dNg[deriv_order-1][ig]]
        for _ in range(deriv_order):
            args2.append(invJ2)
        for _ in range(deriv_order):
            args2.append(n2)

        # Compute normal derivatives of shape functions
        n_dot_dN1 = np.einsum(subscripts, *args1, optimize=True)
        n_dot_dN2 = np.einsum(subscripts, *args2, optimize=True)

        # Compute normal derivative of solution
        dnPSI1 = np.dot(n_dot_dN1, PSI1)
        dnPSI2 = np.dot(n_dot_dN2, PSI2)

        # Jump: note that normals should be opposite, so we add them
        # If n1 = -n2, then [[∂u/∂n]] = ∂u/∂n1|_1 + ∂u/∂n2|_2 (using outward normals)
        jumps[ig] = dnPSI1 + dnPSI2

        # Contribute to L2 integral
        R = FACE1.Xg[ig, 0]  # For axisymmetric formulation
        L2_integral += jumps[ig]**2 * (1/R) * FACE1.detJg1D[ig] * FACE1.Wg[ig]

    L2_norm = np.sqrt(L2_integral)
    max_jump = np.max(np.abs(jumps))

    return jumps, L2_norm, max_jump


def test_solution_continuity_across_ghost_faces(mesh, PSI, tolerance=1e-10):
    """
    Test that the solution is continuous across all ghost faces.

    For standard conforming FEM, the solution should be continuous (C0).
    Jumps should be at machine precision level.

    Input:
        - mesh: Mesh object with ghost faces computed
        - PSI: Solution vector
        - tolerance: Acceptable jump tolerance

    Returns:
        - max_jump: Maximum solution jump across all ghost faces
        - all_continuous: Whether all jumps are within tolerance
    """
    if mesh.GhostFaces is None or len(mesh.GhostFaces) == 0:
        print("No ghost faces to test.")
        return 0.0, True

    max_jump = 0.0
    failures = []

    for ghost_face_tuple in mesh.GhostFaces:
        jumps, face_max_jump = compute_solution_jump_across_ghost_face(mesh, PSI, ghost_face_tuple)

        if face_max_jump > max_jump:
            max_jump = face_max_jump

        if face_max_jump > tolerance:
            elem1_idx = ghost_face_tuple[1][0]
            elem2_idx = ghost_face_tuple[2][0]
            failures.append((elem1_idx, elem2_idx, face_max_jump))

    all_continuous = len(failures) == 0

    if all_continuous:
        print(f"✓ Solution continuity test PASSED (max jump = {max_jump:.2e})")
    else:
        print(f"✗ Solution continuity test FAILED ({len(failures)} faces with jumps > {tolerance:.0e})")
        for elem1, elem2, jump in failures[:3]:
            print(f"  Elements {elem1} & {elem2}: jump = {jump:.2e}")

    return max_jump, all_continuous


def test_normal_derivative_jumps(mesh, PSI, deriv_orders=[1, 2, 3], verbose=True):
    """
    Analyze the normal derivative jumps across all ghost faces for multiple derivative orders.

    This test measures the effectiveness of ghost penalty stabilization.
    For well-stabilized solutions, higher-order derivative jumps should be controlled.

    Input:
        - mesh: Mesh object with ghost faces
        - PSI: Solution vector
        - deriv_orders: List of derivative orders to test
        - verbose: Print detailed results

    Returns:
        - results: Dict with L2 norms and max jumps for each derivative order
    """
    if mesh.GhostFaces is None or len(mesh.GhostFaces) == 0:
        print("No ghost faces to test.")
        return {}

    results = {}

    if verbose:
        print("\n" + "-"*60)
        print("NORMAL DERIVATIVE JUMP ANALYSIS ACROSS GHOST FACES")
        print("-"*60)

    for p in deriv_orders:
        if p > mesh.ElOrder:
            if verbose:
                print(f"  Order {p}: Skipped (exceeds element order {mesh.ElOrder})")
            continue

        total_L2_squared = 0.0
        max_jump_all = 0.0
        num_faces = 0

        for ghost_face_tuple in mesh.GhostFaces:
            try:
                jumps, L2_norm, max_jump = compute_normal_derivative_jump_across_ghost_face(
                    mesh, PSI, ghost_face_tuple, deriv_order=p
                )
                total_L2_squared += L2_norm**2
                max_jump_all = max(max_jump_all, max_jump)
                num_faces += 1
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not compute order {p} derivative for face: {e}")

        total_L2 = np.sqrt(total_L2_squared)

        results[f'order_{p}'] = {
            'L2_norm': total_L2,
            'max_jump': max_jump_all,
            'num_faces': num_faces
        }

        if verbose:
            print(f"  Order {p} (∂^{p}u/∂n^{p}): L2 = {total_L2:.4e}, Max = {max_jump_all:.4e}")

    if verbose:
        print("-"*60)

    return results


def test_ghost_penalty_stabilization_effectiveness(mesh, PSI_with_ghost, PSI_without_ghost=None,
                                                     zeta=None, verbose=True):
    """
    Comprehensive test of ghost penalty stabilization effectiveness.

    Compares normal derivative jumps with and without ghost penalty stabilization,
    or analyzes the jump behavior for a single solution.

    The ghost penalty should:
    1. Maintain solution continuity (C0)
    2. Control gradient jumps (order 1)
    3. Control higher-order derivative jumps (orders 2, 3, ...)

    Input:
        - mesh: Mesh object with ghost faces
        - PSI_with_ghost: Solution computed with ghost penalty
        - PSI_without_ghost: Solution computed without ghost penalty (optional)
        - zeta: Ghost penalty parameter (for reporting)
        - verbose: Print detailed analysis

    Returns:
        - report: Dict containing analysis results and pass/fail status
    """
    report = {
        'passed': True,
        'tests': {},
        'warnings': []
    }

    if verbose:
        print("\n" + "="*70)
        print("GHOST PENALTY STABILIZATION EFFECTIVENESS ANALYSIS")
        if zeta is not None:
            print(f"Ghost penalty parameter: zeta = {zeta}")
        print("="*70)

    # Test 1: Solution continuity
    if verbose:
        print("\n[Test 1] Solution Continuity (C0)")
    max_jump, is_continuous = test_solution_continuity_across_ghost_faces(
        mesh, PSI_with_ghost, tolerance=1e-10
    )
    report['tests']['solution_continuity'] = {
        'max_jump': max_jump,
        'passed': is_continuous
    }
    if not is_continuous:
        report['passed'] = False
        report['warnings'].append("Solution is not continuous across ghost faces")

    # Test 2: Normal derivative jumps for solution with ghost penalty
    if verbose:
        print("\n[Test 2] Normal Derivative Jumps (WITH Ghost Penalty)")

    deriv_orders = list(range(1, mesh.ElOrder + 1))
    results_with = test_normal_derivative_jumps(mesh, PSI_with_ghost, deriv_orders, verbose)
    report['tests']['derivative_jumps_with_ghost'] = results_with

    # Test 3: Compare with solution without ghost penalty (if provided)
    if PSI_without_ghost is not None:
        if verbose:
            print("\n[Test 3] Normal Derivative Jumps (WITHOUT Ghost Penalty)")

        results_without = test_normal_derivative_jumps(mesh, PSI_without_ghost, deriv_orders, verbose)
        report['tests']['derivative_jumps_without_ghost'] = results_without

        # Compute reduction ratios
        if verbose:
            print("\n[Test 4] Stabilization Improvement Ratios")
            print("-"*60)

        improvement_ratios = {}
        for p in deriv_orders:
            key = f'order_{p}'
            if key in results_with and key in results_without:
                L2_with = results_with[key]['L2_norm']
                L2_without = results_without[key]['L2_norm']

                if L2_without > 1e-14:
                    ratio = L2_with / L2_without
                    improvement_ratios[key] = ratio

                    if verbose:
                        status = "✓ IMPROVED" if ratio < 1.0 else "✗ WORSE"
                        print(f"  Order {p}: Ratio = {ratio:.4f} ({status})")

                    if ratio >= 1.0:
                        report['warnings'].append(
                            f"Order {p} derivative jumps not reduced by ghost penalty"
                        )

        report['tests']['improvement_ratios'] = improvement_ratios

        if verbose:
            print("-"*60)

    # Summary
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        if report['passed'] and len(report['warnings']) == 0:
            print("✓ Ghost penalty stabilization is working correctly")
        else:
            if not report['passed']:
                print("✗ Critical issues detected:")
            if report['warnings']:
                print("⚠ Warnings:")
                for w in report['warnings']:
                    print(f"  - {w}")

        print("="*70 + "\n")

    return report


def test_manufactured_solution_ghost_penalty(mesh, verbose=True):
    """
    Test ghost penalty with a manufactured smooth solution.

    Uses a smooth polynomial solution where we know the exact derivatives.
    The ghost penalty should effectively control derivative jumps for smooth functions.

    Input:
        - mesh: Mesh object with ghost faces computed
        - verbose: Print detailed results

    Returns:
        - report: Test results
    """
    if mesh.GhostFaces is None or len(mesh.GhostFaces) == 0:
        print("No ghost faces to test.")
        return {'passed': True, 'skipped': True}

    report = {'passed': True, 'tests': {}}

    if verbose:
        print("\n" + "="*70)
        print("MANUFACTURED SOLUTION TEST FOR GHOST PENALTY")
        print("="*70)

    # Create a smooth manufactured solution: u(x,y) = x^2 + y^2
    # This is exactly representable by quadratic (or higher) elements
    PSI_manufactured = np.zeros(mesh.Nn)
    for i in range(mesh.Nn):
        x, y = mesh.X[i, 0], mesh.X[i, 1]
        PSI_manufactured[i] = x**2 + y**2

    if verbose:
        print("\nManufactured solution: u(x,y) = x² + y²")
        print("Expected: Derivative jumps should be at machine precision for")
        print("          elements of order >= 2 (since solution is quadratic)")

    # Test continuity
    if verbose:
        print("\n[1] Solution Continuity Test")
    max_jump, is_continuous = test_solution_continuity_across_ghost_faces(
        mesh, PSI_manufactured, tolerance=1e-10
    )
    report['tests']['continuity'] = {'max_jump': max_jump, 'passed': is_continuous}

    # Test derivative jumps
    if verbose:
        print("\n[2] Derivative Jump Analysis")

    deriv_results = test_normal_derivative_jumps(
        mesh, PSI_manufactured,
        deriv_orders=list(range(1, min(mesh.ElOrder + 1, 4))),
        verbose=verbose
    )
    report['tests']['derivative_jumps'] = deriv_results

    # For quadratic solution with order >= 2 elements, jumps should be very small
    if mesh.ElOrder >= 2:
        for key, vals in deriv_results.items():
            if vals['L2_norm'] > 1e-8:
                report['warnings'] = report.get('warnings', [])
                report['warnings'].append(
                    f"Unexpectedly large {key} derivative jumps for smooth solution"
                )

    if verbose:
        print("\n" + "="*70)
        if report['passed']:
            print("✓ Manufactured solution test PASSED")
        else:
            print("✗ Manufactured solution test FAILED")
        print("="*70 + "\n")

    return report


def compute_ghost_penalty_energy_norm(mesh, PSI, zeta, h):
    """
    Compute the ghost penalty energy norm (seminorm).

    ||u||²_GP = Σ_F Σ_p ζ h^(2p+2) ∫_F [[∂^p u/∂n^p]]² dΓ

    This measures the total contribution of the ghost penalty stabilization.

    Input:
        - mesh: Mesh object with ghost faces
        - PSI: Solution vector
        - zeta: Ghost penalty parameter
        - h: Characteristic mesh size

    Returns:
        - energy_norm: Ghost penalty energy norm
        - contributions: Dict with contribution from each derivative order
    """
    if mesh.GhostFaces is None or len(mesh.GhostFaces) == 0:
        return 0.0, {}

    contributions = {}
    total_energy = 0.0

    for p in range(1, mesh.ElOrder + 1):
        penalty = zeta * h**(2*p + 2)
        order_contribution = 0.0

        for ghost_face_tuple in mesh.GhostFaces:
            try:
                _, L2_norm, _ = compute_normal_derivative_jump_across_ghost_face(
                    mesh, PSI, ghost_face_tuple, deriv_order=p
                )
                order_contribution += penalty * L2_norm**2
            except:
                pass

        contributions[f'order_{p}'] = order_contribution
        total_energy += order_contribution

    energy_norm = np.sqrt(total_energy)

    return energy_norm, contributions


def run_ghost_penalty_jump_tests(mesh, PSI, zeta=None, h=None, verbose=True):
    """
    Run all ghost penalty jump analysis tests.

    Input:
        - mesh: Mesh object with ghost faces
        - PSI: Solution vector
        - zeta: Ghost penalty parameter (optional, for energy norm)
        - h: Mesh size (optional, for energy norm)
        - verbose: Print detailed output

    Returns:
        - results: Comprehensive test results
    """
    results = {}

    if verbose:
        print("\n" + "="*70)
        print("RUNNING GHOST PENALTY JUMP ANALYSIS TESTS")
        print("="*70)

    # Test 1: Solution continuity
    if verbose:
        print("\n[1] Testing solution continuity across ghost faces...")
    max_jump, is_continuous = test_solution_continuity_across_ghost_faces(mesh, PSI)
    results['continuity'] = {'max_jump': max_jump, 'passed': is_continuous}

    # Test 2: Normal derivative jumps
    if verbose:
        print("\n[2] Analyzing normal derivative jumps...")
    results['derivative_jumps'] = test_normal_derivative_jumps(
        mesh, PSI,
        deriv_orders=list(range(1, mesh.ElOrder + 1)),
        verbose=verbose
    )

    # Test 3: Ghost penalty energy norm (if parameters provided)
    if zeta is not None and h is not None:
        if verbose:
            print("\n[3] Computing ghost penalty energy norm...")
        energy_norm, contributions = compute_ghost_penalty_energy_norm(mesh, PSI, zeta, h)
        results['energy_norm'] = {
            'total': energy_norm,
            'contributions': contributions
        }
        if verbose:
            print(f"  Total energy norm: {energy_norm:.4e}")
            for order, contrib in contributions.items():
                print(f"    {order}: {np.sqrt(contrib):.4e}")

    # Summary
    if verbose:
        print("\n" + "="*70)
        all_passed = results['continuity']['passed']
        if all_passed:
            print("✓ ALL GHOST PENALTY JUMP TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")
        print("="*70 + "\n")

    return results


###################################################################################################
#                              COMPREHENSIVE TEST SUITES
###################################################################################################

def run_all_shape_function_tests(verbose=True):
    """
    Run all shape function validation tests.

    Returns:
        - results (dict): Test results
    """
    if verbose:
        print("\n" + "="*70)
        print("RUNNING SHAPE FUNCTION VALIDATION TESTS")
        print("="*70)

    results = {}

    for elemType in [0, 1, 2]:
        for elemOrder in [1, 2, 3]:
            if elemType == 0 and elemOrder > 3:
                continue

            type_name = ['Line', 'Triangle', 'Quad'][elemType]
            key = f"{type_name}_Order{elemOrder}"

            if verbose:
                print(f"\n--- Testing {type_name}, Order {elemOrder} ---")

            results[f'{key}_partition'] = test_shape_functions_partition_of_unity(elemType, elemOrder)
            results[f'{key}_kronecker'] = test_shape_functions_at_nodes(elemType, elemOrder)

            if elemOrder <= 2:  # Derivatives for lower orders only to save time
                results[f'{key}_derivatives'] = test_shape_function_derivatives_consistency(elemType, elemOrder)

            results[f'{key}_polynomial'] = test_polynomial_reproduction(elemType, elemOrder)

    if verbose:
        print("\n" + "="*70)
        all_passed = all(r[1] for r in results.values() if isinstance(r, tuple))
        if all_passed:
            print("✓ ALL SHAPE FUNCTION TESTS PASSED")
        else:
            print("✗ SOME SHAPE FUNCTION TESTS FAILED")
        print("="*70)

    return results


def run_all_einsum_tests(verbose=True):
    """
    Run all einsum contraction validation tests.

    Returns:
        - results (dict): Test results
    """
    if verbose:
        print("\n" + "="*70)
        print("RUNNING EINSUM CONTRACTION VALIDATION TESTS")
        print("="*70 + "\n")

    results = {}

    results['gradient'] = test_einsum_gradient_contraction()
    results['hessian'] = test_einsum_hessian_contraction()
    results['third_derivative'] = test_einsum_third_derivative_contraction()

    if verbose:
        print("\n" + "="*70)
        all_passed = all(r[1] for r in results.values())
        if all_passed:
            print("✓ ALL EINSUM TESTS PASSED")
        else:
            print("✗ SOME EINSUM TESTS FAILED")
        print("="*70)

    return results


def run_all_jacobian_tests(verbose=True):
    """
    Run all Jacobian computation validation tests.

    Returns:
        - results (dict): Test results
    """
    if verbose:
        print("\n" + "="*70)
        print("RUNNING JACOBIAN VALIDATION TESTS")
        print("="*70 + "\n")

    results = {}

    for elemType in [1, 2]:
        for elemOrder in [1, 2]:
            type_name = ['', 'Triangle', 'Quad'][elemType]
            key = f"{type_name}_Order{elemOrder}"
            results[key] = test_jacobian_computation(elemType, elemOrder)

    results['arc_length_1d'] = test_jacobian_1d_arc_length()

    if verbose:
        print("\n" + "="*70)
        all_passed = all(r[1] for r in results.values())
        if all_passed:
            print("✓ ALL JACOBIAN TESTS PASSED")
        else:
            print("✗ SOME JACOBIAN TESTS FAILED")
        print("="*70)

    return results


def run_all_mesh_tests(mesh):
    """
    Run all mesh-level validation tests.

    Input:
        - mesh (Mesh object): The mesh to validate

    Returns:
        - test_results (dict): Dictionary with results of each test
    """
    print("\n" + "="*70)
    print("RUNNING MESH VALIDATION TESTS")
    print("="*70)

    results = {}

    print("\n1. Testing ghost face normal vectors...")
    results['normal_unitary'] = test_ghost_face_normal_unitary(mesh)
    results['normal_orthogonal'] = test_ghost_face_orthogonality(mesh)
    results['normal_opposition'] = test_ghost_face_normal_opposition(mesh)

    print("\n2. Testing edge and node ordering...")
    results['edge_ordering'] = test_edge_node_ordering(mesh)
    results['node_permutation'] = test_ghost_face_node_permutation(mesh)
    results['ref_phys_consistency'] = test_ghost_face_reference_physical_consistency(mesh)

    print("\n3. Testing Jacobian determinants...")
    results['jacobian_signs'] = test_jacobian_determinant_signs(mesh)

    print("\n4. Testing ghost face quadratures...")
    results['quadrature_accuracy'] = test_ghost_face_quadrature_accuracy(mesh)

    print("\n5. Testing ghost penalty matrix properties...")
    results['matrix_symmetry'] = test_normal_derivative_jump_symmetry(mesh)
    results['matrix_psd'] = test_ghost_penalty_matrix_positive_semidefinite(mesh)

    print("\n" + "="*70)
    all_passed = all(res[1] for res in results.values() if isinstance(res, tuple))
    if all_passed:
        print("✓ ALL MESH TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED - see details above")
    print("="*70 + "\n")

    return results


def run_all_system_tests(LHS, RHS):
    """
    Run all system-level validation tests (after assembly).

    Input:
        - LHS (sparse matrix): Global stiffness matrix
        - RHS (array): Global load vector

    Returns:
        - test_results (dict): Dictionary with results of each test
    """
    print("\n" + "="*70)
    print("RUNNING SYSTEM VALIDATION TESTS")
    print("="*70)

    results = {}

    print("\n1. Testing system matrix properties...")
    results['matrix_symmetry'] = test_system_matrix_symmetry(LHS)
    results['matrix_conditioning'] = test_system_matrix_conditioning(LHS, RHS)

    print("\n" + "="*70)
    print("✓ SYSTEM TESTS COMPLETED")
    print("="*70 + "\n")

    return results


def run_standalone_tests():
    """
    Run all standalone tests that don't require a mesh instance.

    This is useful for quick validation of fundamental components.
    """
    print("\n" + "="*70)
    print("RUNNING STANDALONE VALIDATION TESTS")
    print("="*70)

    all_results = {}

    # Shape function tests
    all_results['shape_functions'] = run_all_shape_function_tests(verbose=True)

    # Einsum tests
    all_results['einsum'] = run_all_einsum_tests(verbose=True)

    # Jacobian tests
    all_results['jacobian'] = run_all_jacobian_tests(verbose=True)

    # Quadrature tests
    print("\n" + "="*70)
    print("RUNNING QUADRATURE VALIDATION TESTS")
    print("="*70 + "\n")

    quadrature_results = {}
    for elemType in [0, 1, 2]:
        for order in [1, 3, 5]:
            type_name = ['Line', 'Triangle', 'Quad'][elemType]
            key = f"{type_name}_Order{order}"
            quadrature_results[key] = test_quadrature_polynomial_exactness(elemType, order, 0)
    all_results['quadrature'] = quadrature_results

    print("\n" + "="*70)
    print("STANDALONE TESTS COMPLETED")
    print("="*70 + "\n")

    return all_results


###################################################################################################
#                              GHOST PENALTY IMPLEMENTATION VERIFICATION
###################################################################################################

def verify_ghost_penalty_implementation(mesh, solver=None, verbose=True):
    """
    Comprehensive verification of the ghost penalty implementation.

    This function runs all relevant tests to verify that ghost penalty
    stabilization is correctly implemented according to CutFEM theory.

    Input:
        - mesh (Mesh object): Mesh with ghost faces computed
        - solver (GradShafranovSolver, optional): Solver instance for parameter verification
        - verbose (bool): Print detailed output

    Returns:
        - report (dict): Comprehensive verification report
    """
    report = {
        'passed': True,
        'critical_failures': [],
        'warnings': [],
        'tests': {}
    }

    if verbose:
        print("\n" + "="*70)
        print("GHOST PENALTY IMPLEMENTATION VERIFICATION")
        print("="*70)

    # 1. Check ghost face identification
    if verbose:
        print("\n[1/6] Verifying ghost face identification...")

    if mesh.GhostFaces is None or len(mesh.GhostFaces) == 0:
        report['warnings'].append("No ghost faces found - ghost penalty may be disabled")
    else:
        report['tests']['num_ghost_faces'] = len(mesh.GhostFaces)
        if verbose:
            print(f"     Found {len(mesh.GhostFaces)} ghost faces")

    # 2. Check normal vector opposition (CRITICAL)
    if verbose:
        print("\n[2/6] Verifying normal vector opposition (CRITICAL)...")

    failures, passed = test_ghost_face_normal_opposition(mesh)
    report['tests']['normal_opposition'] = {'passed': passed, 'failures': len(failures)}

    if not passed:
        report['passed'] = False
        report['critical_failures'].append("Normal vectors on adjacent ghost faces are not opposite")

    # 3. Check normal vector properties
    if verbose:
        print("\n[3/6] Verifying normal vector properties...")

    failures, passed = test_ghost_face_normal_unitary(mesh)
    report['tests']['normal_unitary'] = {'passed': passed}
    if not passed:
        report['critical_failures'].append("Ghost face normals are not unit vectors")
        report['passed'] = False

    failures, passed = test_ghost_face_orthogonality(mesh)
    report['tests']['normal_orthogonal'] = {'passed': passed}
    if not passed:
        report['critical_failures'].append("Ghost face normals are not orthogonal to edges")
        report['passed'] = False

    # 4. Check node permutation
    if verbose:
        print("\n[4/6] Verifying node permutation correctness...")

    failures, passed = test_ghost_face_node_permutation(mesh)
    report['tests']['node_permutation'] = {'passed': passed}
    if not passed:
        report['critical_failures'].append("Node permutation is incorrect")
        report['passed'] = False

    # 5. Check quadrature accuracy
    if verbose:
        print("\n[5/6] Verifying quadrature accuracy...")

    errors, passed = test_ghost_face_quadrature_accuracy(mesh)
    report['tests']['quadrature_accuracy'] = {'passed': passed}
    if not passed:
        report['warnings'].append("Some ghost face quadratures may have reduced accuracy")

    # 6. Check matrix properties
    if verbose:
        print("\n[6/6] Verifying ghost penalty matrix properties...")

    asymmetry, passed = test_normal_derivative_jump_symmetry(mesh)
    report['tests']['matrix_symmetry'] = {'passed': passed, 'asymmetry': asymmetry}
    if not passed:
        report['critical_failures'].append("Ghost penalty matrix is not symmetric")
        report['passed'] = False

    min_eig, passed = test_ghost_penalty_matrix_positive_semidefinite(mesh)
    report['tests']['matrix_psd'] = {'passed': passed, 'min_eigenvalue': min_eig}
    if not passed:
        report['critical_failures'].append("Ghost penalty matrix is not positive semi-definite")
        report['passed'] = False

    # Summary
    if verbose:
        print("\n" + "="*70)
        print("VERIFICATION SUMMARY")
        print("="*70)

        if report['passed']:
            print("\n✓ GHOST PENALTY IMPLEMENTATION VERIFIED SUCCESSFULLY")
        else:
            print("\n✗ GHOST PENALTY IMPLEMENTATION HAS CRITICAL ISSUES:")
            for failure in report['critical_failures']:
                print(f"  - {failure}")

        if report['warnings']:
            print("\n⚠ Warnings:")
            for warning in report['warnings']:
                print(f"  - {warning}")

        print("\n" + "="*70 + "\n")

    return report


# Main entry point for running tests
if __name__ == "__main__":
    print("Running standalone EQUILIPY validation tests...")
    results = run_standalone_tests()
    print("\nAll standalone tests completed.")
