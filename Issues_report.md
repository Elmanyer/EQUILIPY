## Context

This is a CutFEM (Cut Finite Element Method) solver for the Grad-Shafranov PDE:
  -∇·(1/R ∇ψ) = μ₀ R Jφ
implemented in Python. Ghost penalty stabilization has been added to recover
optimal O(h^{p+1}) convergence, but it is not yet achieved.

The relevant source files are:
  - GradShafranovSolver.py  →  IntegrateGhostStabilizationTerms()
  - Element.py              →  GhostFacesNormals(), ComputeGhostFacesQuadratures()
  - Mesh.py                 →  IdentifyPlasmaBoundaryGhostFaces()
  - ShapeFunctions.py       →  EvaluateReferenceShapeFunctions()

---

## Issues to fix

### Issue 1 — Wrong physical derivative for p≥2 (missing chain rule)
File: GradShafranovSolver.py → IntegrateGhostStabilizationTerms()

The p-th order physical normal derivative is computed by contracting the
reference-space tensor with p copies of invJ and p copies of the normal:

  p=2: einsum('nij,ia,jb,a,b->n', Hess_ref, invJ, invJ, n, n)

This is only exact for AFFINE (straight-sided, constant-Jacobian) elements.
For isoparametric elements with ElOrder ≥ 2, the chain rule gives:

  d²u/dx_a dx_b = Σ_{ij} (d²u/dξ_i dξ_j)(dξ_i/dx_a)(dξ_j/dx_b)
                + Σ_i   (du/dξ_i)(d²ξ_i/dx_a dx_b)

The second term (involving second derivatives of the inverse map) is currently
dropped. Add logic to detect non-affine elements (check if detJ varies across
Gauss points) and either:
  (a) compute the missing term from the Jacobian second derivatives, or
  (b) raise a clear warning if ElOrder ≥ 2 with non-affine mapping and
      document the O(h) consistency error this introduces.

---

### Issue 2 — Constant normal vector ignores curved face geometry
File: Element.py → GhostFacesNormals()

The outward normal is computed from only the two corner vertices:

  dx = FACE.Xseg[1,0] - FACE.Xseg[0,0]
  dy = FACE.Xseg[1,1] - FACE.Xseg[0,1]
  ntest = np.array([-dy, dx])

This gives a single constant NormalVec for the entire face, ignoring
mid-edge or high-order nodes. For quadratic/cubic faces this is only
first-order accurate in the normal direction.

Fix: compute the normal at each quadrature point using the face's
isoparametric mapping. At each Gauss point ξ_g on the face, the tangent
vector is t = dX/dξ evaluated via the face shape functions; the outward
normal is n = [-t_y, t_x] / |t|. Store NormalVec as an array of shape
[ng, 2] instead of a single vector [2], and update all downstream
consumers (IntegrateGhostStabilizationTerms) to index NormalVec[ig]
instead of NormalVec.

---

### Issue 3 — Ghost penalty scaling exponent may be sub-optimal
File: GradShafranovSolver.py → IntegrateGhostStabilizationTerms()

Current code (line ~418):
  penalty = self.zeta * h**(2*p - 1)

The comment notes this was empirically tested and h^(2p+2) may be better.
The standard ghost penalty theory for -∇·(1/R ∇ψ) is not identical to
the Euclidean Laplacian case because of the 1/R weight. 

Implement a scaling study helper: for a range of exponent values (2p-2,
2p-1, 2p, 2p+1, 2p+2), run the solver on a known analytical case at
multiple mesh refinements and record the L2 convergence rate. Report the
exponent that first achieves O(h^{p+1}). Make the exponent configurable
via a parameter (e.g. self.ghost_penalty_exponent, defaulting to 2*p-1).

---

### Issue 4 — 1/R weight applied uniformly for all derivative orders p
File: GradShafranovSolver.py → IntegrateGhostStabilizationTerms()

The ghost penalty integrand uses (1/R) for all p:
  LHSe[i,j] += penalty * n_dot_dNg[i] * n_dot_dNg[j] * (1/R) * detJg1D * Wg

For p=1 this is consistent with the bilinear form ∫(1/R)∇u·∇v.
For p≥2, verify whether the (1/R) factor should still appear or whether
it should be (1/R^p) or none. Derive the consistent ghost penalty from
the adjoint of the Grad-Shafranov operator and implement the correct
weight for each derivative order.

---

### Issue 5 — Ghost face set may be too narrow (single-layer patch)
File: Mesh.py → IdentifyPlasmaBoundaryGhostFaces()

Ghost faces are only created on edges shared between a cut element
(PlasmaBoundElems) and an immediate neighbour with Dom < 1. For nearly
degenerate cuts (cut element very small), the stabilization band should
extend to a full patch of width proportional to element order p.

Consider extending the ghost face identification to include faces between
non-cut elements that are within p layers of a cut element, following
the "extended ghost penalty" construction of Burman & Hansbo (2012).

---

## Acceptance criteria

After applying fixes, run the fixed-boundary convergence test
(FIXED_BOUNDARY=True, GhostStabilization=True) on triangular meshes
with ElOrder=1, 2, 3 at 4–5 refinement levels. The L2 error computed
by ComputeL2errorPlasma() must converge at O(h^{p+1}) with the slope
measured via numpy.polyfit on log(h) vs log(error) ≥ p+0.9.