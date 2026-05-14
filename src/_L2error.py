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


from _header import EQUILIPY_ROOT
from _logging import EqPrint
import numpy as np
from Element import *

class EquilipyL2error:
    
    """
    A class to compute errors between the numerical and analytical solutions of the magnetic flux function (PSI) in equilibrium simulations.
    """

    def __init__(self):
        """
        Constructor to initialize error tracking attributes.
        """
        # INITIATE ERROR ARRAYS
        # STANDARD METRICS
        self.PSIexact = None                    # ANALYTICAL SOLUTION FIELD
        self.PSIerror = None                    # ABSOLUTE PSI FIELD ERROR 
        self.ErrorEuclinorm = None              # EUCLIDEAN NORM ERROR
        self.ErrorL2norm = None                 # L2 INTEGRAL NORM ERROR 
        self.RelErrorL2norm = None              # L2 INTEGRAL NORM RELATIVE ERROR
        # CUTFEM METRICS
        self.CutElemsErrorL2norm = None         # L2 ERROR IN CUT ELEMENTS
        self.CutElemsRelErrorL2norm = None      # L2 RELATIVE ERROR IN CUT ELEMENTS
        self.InterfaceErrorL2norm = None        # L2 ERROR ON THE PLASMA INTERFACE
        self.MaxSolJumpGF = None                # MAXIMUM SOLUTION JUMP AT GHOST FACES
        self.MaxGradJumpGF = None               # MAXIMUM GRADIENT JUMP AT GHOST FACES
        self.dpsidnJumpGF = None                # NORMAL DERIVATIVE JUMP AT GHOST FACES 
        self.CutFEMDiagnostics = None           # DICTIONARY TO STORE ALL CUTFEM ERROR DIAGNOSTICS (FOR FUTURE ANALYSIS)

        super().__init__()
        return
    
    
    def ComputeEuclierrorField(self):
        """
        Computes the error between the numerical and analytical PSI solutions.

        Computes:
            self.PSIexact          : Analytical PSI at each node.
            self.PSIerror          : Absolute pointwise error.
            self.ErrorEuclinorm    : Euclidean norm of absolute error.
        """
        # COMPUTE ERROR FIELDS
        self.PSIexact = np.zeros([self.MESH.Nn])
        self.PSIerror = np.zeros([self.MESH.Nn])
        for inode in range(self.MESH.Nn):
            self.PSIexact[inode] = self.PlasmaCurrent.PSIanalytical(self.MESH.X[inode,:])
            self.PSIerror[inode] = abs(self.PSI_NORM[inode,1] - self.PSIexact[inode])
            if self.PSIerror[inode] < 1e-16:
                self.PSIerror[inode] = 1e-16
    
        # COMPUTE EUCLIDEAN NORM ERROR
        self.ErrorEuclinorm = np.linalg.norm(self.PSIerror)
        return
    
    
    def ComputeL2errorPlasma(self):
        """
        Computes the L2 integral norm error of the PSI field by integrating the squared difference between the analytical solution and the 
        computed solution over the plasma region.
        
        Computes:
            self.ErrorL2norm        : L2 INTEGRAL NORM ERROR 
            self.RelErrorL2norm     : L2 INTEGRAL NORM RELATIVE ERROR
        """
        ErrorL2norm = 0
        PSIexactL2norm = 0
        # INTEGRATE OVER PLASMA ELEMENTS
        for elem in self.MESH.PlasmaElems:
            # ISOLATE ELEMENT
            ELEMENT = self.MESH.Elements[elem]
            # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
            PSIg = ELEMENT.Nrefg @ ELEMENT.PSIe
            # LOOP OVER GAUSS NODES
            for ig in range(ELEMENT.ng):
                ErrorL2norm += (PSIg[ig]-self.PlasmaCurrent.PSIanalytical(ELEMENT.Xg[ig,:],NORMALISED=True))**2*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                PSIexactL2norm += self.PlasmaCurrent.PSIanalytical(ELEMENT.Xg[ig,:],NORMALISED=True)**2*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                    
        # INTEGRATE OVER INTERFACE ELEMENTS, FOR SUBELEMENTS INSIDE PLASMA REGION
        for elem in self.MESH.PlasmaBoundElems:
            # ISOLATE ELEMENT
            ELEMENT = self.MESH.Elements[elem]
            # LOOP OVER SUBELEMENTS
            for SUBELEM in ELEMENT.SubElements:
                # INTEGRATE IN SUBDOMAIN INSIDE PLASMA REGION
                if SUBELEM.Dom < 0:
                    # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
                    PSIg = SUBELEM.Nrefg @ ELEMENT.PSIe
                    # LOOP OVER GAUSS NODES
                    for ig in range(SUBELEM.ng):
                        ErrorL2norm += (PSIg[ig]-self.PlasmaCurrent.PSIanalytical(SUBELEM.Xg[ig,:],NORMALISED=True))**2*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]
                        PSIexactL2norm += self.PlasmaCurrent.PSIanalytical(SUBELEM.Xg[ig,:],NORMALISED=True)**2*SUBELEM.detJg[ig]*SUBELEM.Wg[ig]                  
        
        self.RelErrorL2norm = np.sqrt(ErrorL2norm/PSIexactL2norm)
        self.ErrorL2norm = np.sqrt(ErrorL2norm)
        return
    

    def ComputeL2error(self):
        """
        Computes the L2 integral norm error of the PSI field by integrating the squared difference between the analytical solution and the 
        computed solution over the whole domain.
        
        Computes:
            self.ErrorL2norm        : L2 INTEGRAL NORM ERROR 
            self.RelErrorL2norm     : L2 INTEGRAL NORM RELATIVE ERROR
        """
        ErrorL2norm = 0
        PSIexactL2norm = 0
        # INTEGRATE OVER ALL ELEMENTS
        for ELEMENT in self.MESH.Elements:
            # MAPP GAUSS NODAL PSI VALUES FROM REFERENCE ELEMENT TO PHYSICAL SUBELEMENT
            PSIg = ELEMENT.Nrefg @ ELEMENT.PSIe
            # LOOP OVER GAUSS NODES
            for ig in range(ELEMENT.ng):
                ErrorL2norm += (PSIg[ig]-self.PlasmaCurrent.PSIanalytical(ELEMENT.Xg[ig,:],NORMALISED=True))**2*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                PSIexactL2norm += self.PlasmaCurrent.PSIanalytical(ELEMENT.Xg[ig,:],NORMALISED=True)**2*ELEMENT.detJg[ig]*ELEMENT.Wg[ig]
                    
        self.RelErrorL2norm = np.sqrt(ErrorL2norm/PSIexactL2norm)
        self.ErrorL2norm = np.sqrt(ErrorL2norm)
        return
    

    def _compute_interface_error(self):
        """
        Helper: Compute L2 error on the plasma boundary interface.

        Integrates the squared difference between analytical and numerical solutions
        along the plasma boundary (interface line). Used internally by
        _compute_cutfem_errors to avoid redundant computation.

        This error measures how well the solution fits on the interface itself,
        independent of element-wise errors. It's particularly important for CutFEM
        since the interface cuts through elements.

        Returns:
            - interface_L2: L2 norm of absolute error on interface (scalar)
            - interface_rel_L2: Relative L2 error on interface (normalized by exact solution norm)
        """
        ErrorL2norm = 0.0
        PSIexactL2norm = 0.0

        # INTEGRATE OVER CUT ELEMENTS' INTERFACE
        for elem in self.MESH.PlasmaBoundElems:
            INTAPPROX = self.MESH.Elements[elem].InterfApprox
            PSIg = INTAPPROX.Nrefg @ self.MESH.Elements[elem].PSIe

            for ig in range(INTAPPROX.ng):
                psi_exact = self.PlasmaCurrent.PSIanalytical(INTAPPROX.Xg[ig, :])
                weight = INTAPPROX.detJg1D[ig] * INTAPPROX.Wg[ig]

                ErrorL2norm += (PSIg[ig] - psi_exact)**2 * weight
                PSIexactL2norm += psi_exact**2 * weight

        if ErrorL2norm == 0:
            return 0.0, 0.0
        else:
            return np.sqrt(ErrorL2norm), np.sqrt(ErrorL2norm / PSIexactL2norm)
    
    
    def _compute_interface_jump(self):
        """
        Computes L2 norm of normal derivative jumps on the plasma interface.

        This function is distinct from ghost face diagnostics because it:
        - Evaluates gradients at shifted points perpendicular to the interface
        - Computes the jump in normal derivatives across the interface
        - Is used for analyzing interface gradient continuity

        Returns:
            - ErrorL2norm: L2 norm of gradient jump on interface
            - JumpError: Point-wise gradient jump values at interface Gauss nodes
            - JumpRelError: Relative gradient jump values
        """
        JumpError = np.zeros([self.MESH.NnPB])
        JumpRelError = np.zeros([self.MESH.NnPB])
        ErrorL2norm = 0
        dn = 1e-4
        knode = 0

        # INTEGRATE OVER CUT ELEMENTS' INTERFACE
        for elem in self.MESH.PlasmaBoundElems:
            ELEMENT = self.MESH.Elements[elem]
            INTAPPROX = ELEMENT.InterfApprox
            PSIg = INTAPPROX.Nrefg @ ELEMENT.PSIe

            # LOOP OVER GAUSS NODES
            for ig in range(INTAPPROX.ng):
                # Evaluate gradients at shifted points (perpendicular to interface)
                XIgplus = INTAPPROX.XIg[ig, :] + dn * INTAPPROX.NormalVecREF[ig]
                XIgminus = INTAPPROX.XIg[ig, :] - dn * INTAPPROX.NormalVecREF[ig]

                Ngplus, dNgplus = EvalRefLagrangeBasis(XIgplus.reshape((1, 2)), ELEMENT.ElType, ELEMENT.ElOrder)
                Ngminus, dNgminus = EvalRefLagrangeBasis(XIgminus.reshape((1, 2)), ELEMENT.ElType, ELEMENT.ElOrder)

                # Compute Jacobians at shifted points and compute their transposes inverse
                Jgplus = Jacobian(ELEMENT.Xe, dNgplus[0][0, :, :])
                Jgminus = Jacobian(ELEMENT.Xe, dNgminus[0][0, :, :])

                invJTplus = np.linalg.inv(Jgplus.T)   # (J^T)^{-1}
                invJTminus = np.linalg.inv(Jgminus.T)  # (J^T)^{-1}

                PSIgplus = Ngplus[0] @ ELEMENT.PSIe
                PSIgminus = Ngminus[0] @ ELEMENT.PSIe

                # Physical gradients: ∇_x = (J^T)^{-1} ∇_ξ
                Ngradplus = invJTplus @ dNgplus[0][0, :, :].T
                Ngradminus = invJTminus @ dNgminus[0][0, :, :].T

                # Compute gradient difference (jump)
                diffgrad = 0.0
                grad = 0.0
                for inode in range(ELEMENT.n):
                    diffgrad += (Ngradplus[:, inode] * PSIgplus - Ngradminus[:, inode] * PSIgminus) @ INTAPPROX.NormalVec[ig]
                    # Use physical space derivatives for consistency with physical space normal vector
                    phys_grad_n = INTAPPROX.NormalVec[ig] @ INTAPPROX.dNg[0][ig, inode, :]
                    grad += phys_grad_n * PSIg[ig]

                JumpError[knode] = diffgrad
                JumpRelError[knode] = diffgrad / abs(grad) if abs(grad) > 1e-16 else 0.0
                knode += 1

                ErrorL2norm += diffgrad**2 * INTAPPROX.detJg1D[ig] * INTAPPROX.Wg[ig]

        return np.sqrt(ErrorL2norm), JumpError, JumpRelError


    def _compute_element_L2_errors(self, element_indices):
        """
        Helper: Compute L2 error for a specified set of elements.

        Computes the integral-norm error (||u_h - u_exact||_L2) over the given elements
        by integrating the squared difference at all quadrature points. The 1/R factor
        accounts for the axisymmetric geometry (Grad-Shafranov problem is in (R,Z) cylindrical coordinates).

        This function is refactored from duplicated code in _compute_cutfem_errors
        to compute errors for both cut and interior elements without code repetition.

        Args:
            element_indices: List of element indices to compute errors for

        Returns:
            - L2_squared: Sum of squared L2 contributions (before taking sqrt)
            - exact_L2_squared: Sum of squared analytical solution norm (for relative error)

        Note: Both return values are squared; caller should apply sqrt() if needed.
        """
        L2_squared = 0.0
        exact_L2_squared = 0.0

        for elem_idx in element_indices:
            elem = self.MESH.Elements[elem_idx]
            if elem.ng is None or elem.ng == 0:
                continue

            PSI_local = self.PSI[elem.Te]
            for ig in range(elem.ng):
                X_point = elem.Xg[ig, :]
                R = X_point[0]
                psi_num = np.dot(elem.Nrefg[ig, :], PSI_local)

                try:
                    psi_exact = self.PlasmaCurrent.PSIanalytical(X_point)
                except:
                    continue

                weight = (1.0 / R) * elem.detJg[ig] * elem.Wg[ig]
                L2_squared += (psi_num - psi_exact)**2 * weight
                exact_L2_squared += psi_exact**2 * weight

        return L2_squared, exact_L2_squared

    def _compute_ghost_face_derivative_jumps(self, deriv_order):
        """
        Helper: Compute p-th order normal derivative jumps at ghost faces.

        Evaluates the jump in p-th order normal derivatives across internal ghost faces
        (interfaces between cut and plasma elements). This is essential for diagnosing
        whether ghost penalty stabilization is working: large jumps indicate inadequate
        penalty parameter or formula scaling.

        The jump is computed as:
            [[∂^p u/∂n^p]] = (n·∇)^p u+ + (n·∇)^p u-  (normals are opposite)

        This function refactors the repetitive derivative jump computation loop
        (previously duplicated for orders 1, 2, 3 in _compute_cutfem_errors).

        Args:
            deriv_order: Derivative order (1=gradient jump, 2=hessian jump, 3=3rd deriv jump)

        Returns:
            - L2_norm: L2 integral norm of the derivative jump across all ghost faces
            - max_jump: Maximum magnitude of derivative jump (useful for peak detection)

        Raises:
            Silently skips Gauss points with numerical issues (try/except block).
        """

        if deriv_order == 1:
            subscripts = 'ni,i->n'
        elif deriv_order == 2:
            subscripts = 'nij,i,j->n'
        elif deriv_order == 3:
            subscripts = 'nijk,i,j,k->n'
        else:
            return 0.0, 0.0

        total_L2_squared = 0.0
        max_jump = 0.0

        for ghost_face_tuple in self.MESH.GhostFaces:
            elem1_idx, edge1_idx, face1_list_idx = ghost_face_tuple[1]
            elem2_idx, edge2_idx, face2_list_idx = ghost_face_tuple[2]

            ELEM1 = self.MESH.Elements[elem1_idx]
            ELEM2 = self.MESH.Elements[elem2_idx]
            FACE1 = ELEM1.GhostFaces[face1_list_idx]
            FACE2 = ELEM2.GhostFaces[face2_list_idx]

            PSI1 = self.PSI[ELEM1.Te]
            PSI2 = self.PSI[ELEM2.Te]

            for ig in range(FACE1.ng):
                try:
                    n1 = FACE1.NormalVec
                    n2 = FACE2.NormalVec

                    # Use physical space derivatives directly (dNg already transformed)
                    # Build einsum arguments for p-th derivative
                    args1 = [FACE1.dNg[deriv_order - 1][ig]]
                    args2 = [FACE2.dNg[deriv_order - 1][ig]]
                    for _ in range(deriv_order):
                        args1.append(n1)
                        args2.append(n2)

                    n_dot_dN1 = np.einsum(subscripts, *args1, optimize=True)
                    n_dot_dN2 = np.einsum(subscripts, *args2, optimize=True)

                    dnPSI1 = np.dot(n_dot_dN1, PSI1)
                    dnPSI2 = np.dot(n_dot_dN2, PSI2)

                    jump = dnPSI1 + dnPSI2  # Normals are opposite
                    R = FACE1.Xg[ig, 0]

                    total_L2_squared += jump**2 * (1.0 / R) * FACE1.detJg1D[ig] * FACE1.Wg[ig]
                    max_jump = max(max_jump, abs(jump))
                except:
                    continue

        return np.sqrt(total_L2_squared), max_jump

    def _compute_cutfem_errors(self, verbose=True):
        """
        Computes comprehensive error diagnostics for CutFEM solutions, including:
        - L2 error in cut elements vs interior elements
        - Solution continuity (jumps) at ghost faces
        - Normal derivative jumps at ghost faces (orders 1, 2, 3)
        - Error distribution near vs far from interface

        Returns:
            - diagnostics (dict): Dictionary containing all error metrics
        """
        from FELagrangeanbasis import RefLagrangeBasis
        from Element import ElementalNumberOfNodes

        diagnostics = {
            'cut_elements': {},
            'interior_elements': {},
            'ghost_faces': {},
            'interface': {},
            'summary': {}
        }

        mesh = self.MESH
        PSI = self.PSI
        analytical_solution = self.PlasmaCurrent.PSIanalytical

        # ==========================================================================
        # 1. COMPUTE L2 ERROR IN CUT ELEMENTS VS INTERIOR ELEMENTS
        # ==========================================================================
        # Identify element sets
        cut_elements = [i for i, elem in enumerate(mesh.Elements) if elem.Dom == 1]
        interior_elements = [i for i, elem in enumerate(mesh.Elements) if elem.Dom == 2]

        # Compute errors using helper function
        cut_L2_squared, exact_L2_squared_cut = self._compute_element_L2_errors(cut_elements)
        interior_L2_squared, exact_L2_squared_interior = self._compute_element_L2_errors(interior_elements)

        self.CutElemsErrorL2norm = np.sqrt(cut_L2_squared) if cut_L2_squared > 0 else 0.0
        interior_L2 = np.sqrt(interior_L2_squared) if interior_L2_squared > 0 else 0.0
        self.CutElemsRelErrorL2norm = self.CutElemsErrorL2norm / np.sqrt(exact_L2_squared_cut) if exact_L2_squared_cut > 0 else 0.0
        interior_rel_L2 = interior_L2 / np.sqrt(exact_L2_squared_interior) if exact_L2_squared_interior > 0 else 0.0

        diagnostics['cut_elements'] = {
            'count': len(cut_elements),
            'L2_error': self.CutElemsErrorL2norm,
            'relative_L2_error': self.CutElemsRelErrorL2norm
        }
        diagnostics['interior_elements'] = {
            'count': len(interior_elements),
            'L2_error': interior_L2,
            'relative_L2_error': interior_rel_L2
        }
        diagnostics['summary']['error_ratio_cut_interior'] = self.CutElemsErrorL2norm / (interior_L2 + 1e-16)

        # ==========================================================================
        # 2. COMPUTE SOLUTION CONTINUITY AT GHOST FACES
        # ==========================================================================
        if mesh.GhostFaces is not None and len(mesh.GhostFaces) > 0:
            solution_jumps = []
            gradient_jumps = []

            for ghost_face_tuple in mesh.GhostFaces:
                elem1_idx, edge1_idx, face1_list_idx = ghost_face_tuple[1]
                elem2_idx, edge2_idx, face2_list_idx = ghost_face_tuple[2]

                ELEM1 = mesh.Elements[elem1_idx]
                ELEM2 = mesh.Elements[elem2_idx]
                FACE1 = ELEM1.GhostFaces[face1_list_idx]
                FACE2 = ELEM2.GhostFaces[face2_list_idx]

                n1, _ = ElementalNumberOfNodes(ELEM1.ElType, ELEM1.ElOrder)
                n2, _ = ElementalNumberOfNodes(ELEM2.ElType, ELEM2.ElOrder)

                PSI1 = PSI[ELEM1.Te]
                PSI2 = PSI[ELEM2.Te]

                for ig in range(FACE1.ng):
                    # Solution jump
                    u1 = 0.0
                    u2 = 0.0
                    for i in range(n1):
                        N_i = RefLagrangeBasis(FACE1.XIg[ig, :], ELEM1.ElType, ELEM1.ElOrder, i+1, deriv=0)
                        u1 += N_i * PSI1[i]
                    for i in range(n2):
                        N_i = RefLagrangeBasis(FACE2.XIg[ig, :], ELEM2.ElType, ELEM2.ElOrder, i+1, deriv=0)
                        u2 += N_i * PSI2[i]

                    solution_jumps.append(abs(u1 - u2))

                    # Gradient jump (normal component)
                    n1_vec = FACE1.NormalVec
                    n2_vec = FACE2.NormalVec

                    # Use physical space derivatives directly (dNg already transformed)
                    gradN1 = FACE1.dNg[0][ig]
                    gradN2 = FACE2.dNg[0][ig]

                    n_dot_gradN1 = np.einsum('ni,i->n', gradN1, n1_vec)
                    n_dot_gradN2 = np.einsum('ni,i->n', gradN2, n2_vec)

                    grad_u1 = np.dot(n_dot_gradN1, PSI1)
                    grad_u2 = np.dot(n_dot_gradN2, PSI2)

                    # Normal vectors are opposite, so we add
                    gradient_jumps.append(abs(grad_u1 + grad_u2))

            diagnostics['ghost_faces'] = {
                'count': len(mesh.GhostFaces),
                'solution_jump_max': np.max(solution_jumps) if solution_jumps else 0,
                'solution_jump_mean': np.mean(solution_jumps) if solution_jumps else 0,
                'solution_jump_std': np.std(solution_jumps) if solution_jumps else 0,
                'gradient_jump_max': np.max(gradient_jumps) if gradient_jumps else 0,
                'gradient_jump_mean': np.mean(gradient_jumps) if gradient_jumps else 0,
                'gradient_jump_std': np.std(gradient_jumps) if gradient_jumps else 0,
                'continuity_ok': np.max(solution_jumps) < 1e-10 if solution_jumps else True
            }

            # ==========================================================================
            # 3. COMPUTE NORMAL DERIVATIVE JUMPS (ORDERS 1, 2, 3)
            # ==========================================================================
            for deriv_order in range(1, min(mesh.ElOrder + 1, 4)):
                L2_norm, max_jump = self._compute_ghost_face_derivative_jumps(deriv_order)
                diagnostics[f'normal_deriv_order_{deriv_order}'] = {
                    'L2_norm': L2_norm,
                    'max_jump': max_jump
                }
        else:
            diagnostics['ghost_faces'] = {
                'count': 0,
                'message': 'No ghost faces computed'
            }

        # ==========================================================================
        # 4. COMPUTE ERROR ON INTERFACE (INLINE - previously duplicated)
        # ==========================================================================
        interface_L2, interface_rel_L2 = self._compute_interface_error()
        diagnostics['interface'] = {
            'L2_error': interface_L2,
            'relative_L2_error': interface_rel_L2
        }

        # ==========================================================================
        # 5. SUMMARY
        # ==========================================================================
        diagnostics['summary']['total_L2_error'] = self.ErrorL2norm if hasattr(self, 'ErrorL2norm') else None
        diagnostics['summary']['total_relative_L2_error'] = self.RelErrorL2norm if hasattr(self, 'RelErrorL2norm') else None
        diagnostics['summary']['ghost_penalty_enabled'] = self.GhostStabilization if hasattr(self, 'GhostStabilization') else None
        diagnostics['summary']['zeta'] = self.zeta if hasattr(self, 'zeta') else None

        # Store diagnostics in solver object
        self.CutFEMDiagnostics = diagnostics

        # ==========================================================================
        # 6. PRINT REPORT IF VERBOSE
        # ==========================================================================
        if verbose:
            EqPrint("="*70)
            EqPrint("CUTFEM ERROR DIAGNOSTICS REPORT")
            EqPrint("="*70)

            EqPrint("[1] ELEMENT-WISE ERROR DISTRIBUTION")
            EqPrint("-"*50)
            EqPrint(f"  Cut elements ({len(cut_elements)}):     L2 = {self.CutElemsErrorL2norm:.4e}, Rel = {self.CutElemsRelErrorL2norm:.4e}")
            EqPrint(f"  Interior elements ({len(interior_elements)}):  L2 = {interior_L2:.4e}, Rel = {interior_rel_L2:.4e}")
            EqPrint(f"  Error ratio (cut/interior): {diagnostics['summary']['error_ratio_cut_interior']:.4f}")

            if 'ghost_faces' in diagnostics and diagnostics['ghost_faces'].get('count', 0) > 0:
                gf = diagnostics['ghost_faces']
                EqPrint(f"[2] GHOST FACE SOLUTION QUALITY ({gf['count']} faces)")
                EqPrint("-"*50)
                EqPrint(f"  Solution jump:   max = {gf['solution_jump_max']:.4e}, mean = {gf['solution_jump_mean']:.4e}")
                EqPrint(f"  Gradient jump:   max = {gf['gradient_jump_max']:.4e}, mean = {gf['gradient_jump_mean']:.4e}")
                EqPrint(f"  Continuity OK:   {'✓ YES' if gf['continuity_ok'] else '✗ NO'}")

                EqPrint(f"[3] NORMAL DERIVATIVE JUMPS")
                EqPrint("-"*50)
                for p in range(1, 4):
                    key = f'normal_deriv_order_{p}'
                    if key in diagnostics:
                        d = diagnostics[key]
                        EqPrint(f"  Order {p}: [[∂^{p}u/∂n^{p}]]  L2 = {d['L2_norm']:.4e}, Max = {d['max_jump']:.4e}")

            EqPrint(f"[4] INTERFACE ERROR")
            EqPrint("-"*50)
            EqPrint(f"  Interface L2 error:     {interface_L2:.4e}")
            EqPrint(f"  Interface relative L2:  {interface_rel_L2:.4e}")

            EqPrint("="*70)
            if diagnostics['ghost_faces'].get('continuity_ok', True):
                EqPrint("✓ Solution continuity verified - CutFEM stabilization working")
            else:
                EqPrint("✗ Solution discontinuity detected - check ghost penalty parameters")
            EqPrint("="*70 + "\n")

        return diagnostics