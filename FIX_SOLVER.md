# FIX_SOLVER.md — CutFEM Solver Bug-Fix Reference

**Purpose:** This document is written for an agent that must find and fix the same class of bugs
in a **Grad-Shafranov CutFEM solver** (or any CutFEM solver for an elliptic PDE on a domain
defined by a level-set function). Every bug described here was found and fixed during a
systematic validation campaign on both a 2D Poisson CutFEM solver and the GS CutFEM solver.
The bugs are ordered by severity (most impactful first).

**CRITICAL DISTINCTION — GS vs Poisson sign convention:**
The GS equation is `Δ*ψ = f` (positive LHS operator), while the Poisson solver uses `−Δu = f`
(negative LHS). Because the LHS is assembled as `LHSe[i,j] -= ...` in both cases (INV-1), the
sign implications for the RHS differ. See BUG-4 for the exact consequence.

The bugs fall into five categories:
1. **Quadrature table error** — wrong Gauss point storage index
2. **Sign convention errors** — LHS=−K convention applied inconsistently
3. **Sub-element DOF indexing** — wrong node count for high-order sub-elements
4. **Degenerate cut element handling** — machine-epsilon level-set values
5. **Parameter scaling** — Nitsche penalty and ghost penalty too large for p=1 elements

Six architectural invariants (INV-1 through INV-6) are documented in the prerequisites
section. Violating any one of them produces wrong results regardless of whether all
individual bugs are fixed. **INV-6 (only assemble plasma-side elements) is a particularly
common mistake when porting a standard FEM solver to CutFEM.**

Each section gives: (a) the symptom you will observe, (b) the exact location and wrong code,
(c) the correct fix, (d) the mathematical root cause, (e) a diagnostic to confirm the bug.

---

## Architecture prerequisites — read this first

Before hunting bugs, understand these invariants. Violating any one of them will produce wrong
results even if all individual components are correct.

### INV-1: Global LHS = −K sign convention

This solver assembles the stiffness matrix with a **global minus sign**:
```python
LHSe[i,j] -= (1/R) * grad_N_i @ grad_N_j * detJ * w    # note: -=
```
The system solved is `−K · PSI = RHS`. The minus signs in LHS and RHS cancel in the solve.
This is a non-standard convention but is internally consistent. **Every new term added to the
system must follow this convention.**

The GS weak form of `Δ*ψ = f` gives:
```
(1/R) ∇ψ·∇v = −(f, v)    [after IBP, with zero Dirichlet BC]
```
So `K·ψ = −(f,v)` and the assembled system is `(−K)·ψ = (f,v)`, i.e. `LHS·ψ = RHS` where
`RHS[i] = +(f, N_i)`.

### INV-2: Nitsche signs with LHS=−K

The symmetric Nitsche weak form for `ψ = ψ_D` on the cut interface Γ is:
```
a_N(ψ,v) = −∫_Γ (n·(1/R)∇ψ) v − ∫_Γ (n·(1/R)∇v) ψ_D + (β/h) ∫_Γ ψ v
```
With `LHS = −K`, the Nitsche bilinear terms contribute `−a_N` to the LHS (an extra sign flip).
The **correct** implementation is:

| Term | LHS code | RHS code |
|------|----------|----------|
| Consistency | `LHSe[i,j] += N_i * (n·∇N_j/R) * w` | — |
| Symmetry    | `LHSe[i,j] += (n·∇N_i/R) * N_j * w` | — |
| Penalty     | `LHSe[i,j] -= (β/h) * N_i * N_j * w` | — |
| RHS symmetry | — | `RHSe[i] += PSIg * (n·∇N_i/R) * w` |
| RHS penalty  | — | `RHSe[i] -= (β/h) * PSIg * N_i * w` |

**Common mistake:** writing `LHSe[i,j] -= N_i * (n·∇N_j/R) * w` for consistency/symmetry. This is
wrong with LHS=−K; it would be correct for the standard LHS=+K convention.

**Diagnostic:** Assemble the full system with PSI set to the exact analytical solution.
Compute `residual = LHS @ PSI_exact - RHS` at all free (non-Dirichlet) DOFs. The max residual
should be < 1e-8 × ‖LHS‖ if Nitsche is consistent. If it is O(1) or larger, signs are wrong.

### INV-3: Ghost penalty jump sign convention

The ghost penalty penalises the jump of the p-th normal derivative across a ghost face F shared
between elements 0 and 1. Each element uses its OWN outward normal `n_i`. The jump is:

```
[[D^p_n u]]_F = (D^p_n u|_1 using n_1) − (−1)^p × (D^p_n u|_0 using n_0)
```

Converting to concatenated form where both sides use their own outward normal:
```python
sign_p = (-1) ** (p + 1)   # +1 for p=1 (odd); -1 for p=2 (even)
n_dot_dNg = np.concatenate((sign_p * n_dot_dNg0, n_dot_dNg1), axis=0)
```

**Common mistake:** `np.concatenate((n_dot_dNg0, -n_dot_dNg1))` — this computes the SUM of
normal derivatives (not the jump), because n_0 = −n_1. For p=1 this gives 2·∇u·n ≠ 0 for any
smooth function, completely breaking the ghost penalty.

### INV-4: Sub-element DOF count = parent DOF count

When a high-order element (e.g. QUA09 with n=9 DOFs) is tessellated into lower-order
sub-elements for integration over the cut domain, the sub-elements must use the PARENT's
DOF count for all arrays (`PSIe`, loop bounds, stiffness accumulation). The sub-element
has its own geometric nodes for computing the integration map, but the basis functions
evaluated at sub-element quadrature points are the PARENT's basis functions at the
mapped parent reference coordinates.

```python
SUBELEM.n = parent.n          # 9 for QUA09, not 6 for TRI6
SUBELEM.PSIe = np.zeros(parent.n)
```

### INV-5: Ghost penalty parameter scaling

The penalty `ζ·h^{2p−1}·[[∂^p_n u_h]]²` scales as ζ times a mesh-independent quantity (the
ratio ghost/domain ~ ζ, h-independent). With `ζ=1e8`, the ghost penalty is 1e8 times larger
than the domain stiffness. This dominates the system for any non-polynomial test case or for
p=1 elements. Use `ζ = O(1–100)` for general problems.

**Exception:** If the exact solution lies in the FEM space (e.g. u=x²+y² for p=2 elements),
the ghost terms are identically zero and ζ has no effect on the solution. In that case, large ζ
provides numerical conditioning of sliver DOFs without corrupting the result. This special case
does NOT generalise.

### INV-6: Only assemble elements inside the embedded domain (Dom ≤ 0)

The domain integral and interface terms must be assembled **only for elements on the plasma
side**. Exterior elements (Dom=+1, fully outside Ω⁻) must NOT contribute to the domain
stiffness.

**Why exterior elements must be skipped:** The GS PDE is only defined in the plasma domain Ω⁻.
Assembling vacuum elements adds stiffness contributions enforcing `Δ*ψ=0` in the vacuum. Since
the exact solution satisfies `Δ*ψ=f ≠ 0` in the vacuum for the LINEAR test case, the assembled
vacuum solution differs from `ψ_exact` there. Cut element boundary nodes are shared between
plasma K⁻ and adjacent vacuum elements — the Nitsche coupling propagates the O(h) vacuum
mismatch into the plasma solution, saturating convergence at O(h) regardless of element order.

**Root cause of O(h) saturation (confirmed in GS solver):**
- With vacuum stiffness assembled: TRI03 rates ≈ 0.93, 0.35, 0.87 (O(h), not O(h²))
- After INV-6 fix: TRI03 rates ≈ 3.08, 1.99, 2.00 — achieving optimal O(h²)

**Correct implementation in `src/GradShafranovSolver.py`:**

```python
for ielem in self.MESH.NonCutElems:
    ELEMENT = self.MESH.Elements[ielem]
    # INV-6: Skip pure vacuum interior elements (Dom>0, no Dirichlet BCs).
    # Their DOFs are conditioned by ghost penalty and the zero-diagonal fix.
    if ELEMENT.Dom > 0 and ELEMENT.Teboun is None:
        continue
    SourceTermg = np.zeros([ELEMENT.ng])
    if ELEMENT.Dom < 0:   # plasma interior: compute source term
        PSIg = ELEMENT.Nrefg @ ELEMENT.PSIe
        for ig in range(ELEMENT.ng):
            SourceTermg[ig] = self.PlasmaCurrent.SourceTerm(ELEMENT.Xg[ig,:], PSIg[ig])
    LHSe, RHSe = ELEMENT.IntegrateElementalDomainTerms(SourceTermg)
    if ELEMENT.Teboun is not None:
        LHSe, RHSe = ELEMENT.PrescribeDirichletBC(LHSe, RHSe)
    # ... assemble into global system ...
```

For the cut element loop (`PlasmaBoundElems`), additionally skip vacuum sub-elements:
```python
for SUBELEM in ELEMENT.SubElements:
    if SUBELEM.Dom >= 0:   # skip vacuum sub-elements (Dom=0 is interface, Dom>0 is vacuum)
        continue
    # ... integrate plasma sub-element ...
```

**Ghost penalty involves exterior DOFs — this is fine:** Ghost faces can connect a cut element
to an adjacent exterior element. The ghost penalty assembly uses both elements' DOFs. The
exterior element DOFs then have rows in the global system populated only by ghost penalty terms
(no domain stiffness). With ghost stabilization ON and ζ=O(10–100), these rows are
well-conditioned.

**Zero-diagonal sweep after all assembly:**
After completing all loops (NonCutElems, PlasmaBoundElems, PlasmaBoundActiveElems, ghost faces),
some vacuum-interior DOFs that are not reached by any ghost face will have zero diagonal entries,
which makes the system singular. Fix:
```python
LHS_csr = self.LHS.tocsr()
diag = np.array(LHS_csr.diagonal())
for i in range(self.MESH.Nn):
    if abs(diag[i]) < 1e-30:
        self.LHS[i, i] = 1.0
```

**Diagnostic:** After assembly, check that all rows corresponding to interior vacuum nodes
(Dom=+1, not on the outer boundary) have zero domain stiffness contributions (only ghost
contributions if adjacent to the cut, or a unit diagonal from the zero-diagonal sweep if not).

---

## BUG-1: Missing Gauss point in 2D quadrilateral quadrature table

**STATUS IN GS SOLVER: NOT PRESENT** — The QUA order-2 block in `src/GaussQuadrature.py`
correctly stores 4 points at indices 0, 1, 2, 3. Verified by monomial test:
`∫_{[-1,1]²} x²y² = 4/9` — result 0.4444444444442 (error 2.78e-16, machine precision).

### Symptom
All integrals over quadrilateral elements produce systematically wrong values. L2 errors are
large and do not decrease at the correct rate.

### Location
`src/GaussQuadrature.py` — the function that builds the 2D Gauss point table for
quadrilateral reference elements (the tensor product of 1D rules).

### Wrong code
```python
zg[2,:] = [a, a]   # stored correctly
zg[2,:] = [-a, a]  # OVERWRITES zg[2] — the value [a,a] is lost; zg[3,:] stays [0,0]
```

### Correct fix
```python
zg[2,:] = [a, a]
zg[3,:] = [-a, a]  # was: zg[2,:] = [-a, a]
```

### QUA quadrature order limitation
QUA elements in this solver support quadrature orders 1–5 only (25 Gauss points maximum).
Requesting order 6 or higher causes `UnboundLocalError: cannot access local variable 'zg'`
because no case exists for those orders. **Always use `QuadratureOrder2D = 5` for QUA element
runs.** TRI elements support up to order 8.

### Diagnostic
```python
from GaussQuadrature import GaussQuadrature
zg, wg, Ng = GaussQuadrature(2, 2)   # QUA, order 2
result = sum(zg[i,0]**2 * zg[i,1]**2 * wg[i] for i in range(Ng))
assert abs(result - 4/9) < 1e-12, f"BUG-1: got {result}, expected {4/9:.6f}"
print(f"BUG-1 check: {result:.10f} (expect 0.4444...) — {'OK' if abs(result-4/9)<1e-12 else 'FAIL'}")
```

---

## BUG-2: Wrong domain flags in L2 error diagnostic

### Symptom
A secondary L2 error diagnostic function reports large errors or errors concentrated in the
wrong region. The PRIMARY error metric is unaffected.

### Location
`src/_L2error.py` — inside `_compute_cutfem_errors()` or a similar diagnostic function.

### Wrong code
```python
cut_elements      = [e for e in elements if e.Dom == 1]   # WRONG: Dom=1 is vacuum
interior_elements = [e for e in elements if e.Dom == 2]   # WRONG: Dom=2 is ∂Ω_comp
```

### Correct fix
```python
cut_elements      = [e for e in elements if e.Dom == 0]   # 0 = cut
interior_elements = [e for e in elements if e.Dom == -1]  # −1 = interior (plasma)
```

### Domain flag reference
| Dom | Meaning |
|-----|---------|
| −1  | Interior (plasma) — fully inside Ω⁻ |
|  0  | Cut — element is intersected by Γ |
| +1  | Exterior (vacuum) — fully outside Ω⁻ |
| +2  | Computational boundary — nodes on ∂Ω_comp |

---

## BUG-3: False quadrature warning for sub-elements (not a real bug)

### Explanation
Sub-elements are created by tessellating the parent reference element. Adapted quadrature maps
standard Gauss points through `ξ_parent = J_interp(ξ_sub)`. A check that tests whether Gauss
points are inside the sub-element's own reference domain will fail, even though the points are
correct. Disable or skip the quadrature check for sub-elements. The warning is a false positive.

---

## BUG-4: Source-term sign in domain integration

### CRITICAL: GS vs Poisson sign difference

**This bug description applies differently for the GS and Poisson solvers:**

**Poisson solver (−Δu = f):** RHS must be `RHSe[i] -= f * N_i * detJ * w` (minus sign).
With LHS=−K, the system is `(−K)u = RHS`. Poisson gives `K·u = (f,v)`, so `(−K)u = −(f,v)`,
i.e. `RHS = −(f,v)`. The source term must use `-=`.

**GS solver (Δ*ψ = f, positive LHS operator):** RHS must be `RHSe[i] += (1/R)*f*N_i*detJ*w`
(plus sign — CORRECT). With LHS=−K, the system is `(−K)ψ = RHS`. GS IBP gives `K·ψ = −(f,v)`,
so `(−K)ψ = (f,v)`, i.e. `RHS = +(f,v)`. The source term correctly uses `+=`.

**The current GS solver code at `src/Element.py` line 1238 uses `+=` — this is CORRECT.**
Do NOT change it to `-=`. The Poisson solver fix (BUG-4 = `+=`→`-=`) does NOT apply to the GS solver.

### Symptom (Poisson only)
The Poisson solver produces large L2 errors (O(1) or larger). The solution has the wrong sign
or wrong magnitude. Reducing the mesh size does not help.

### Location (Poisson only)
`src/Element.py` — `IntegrateElementalDomainTerms`

### Wrong code (Poisson solver only)
```python
RHSe[i] += SourceTerm(Xg[ig]) * Nrefg[ig,i] * detJg[ig] * Wg[ig]   # WRONG for Poisson (−Δu=f)
```

### Correct fix (Poisson solver only)
```python
RHSe[i] -= SourceTerm(Xg[ig]) * Nrefg[ig,i] * detJg[ig] * Wg[ig]   # CORRECT for −Δu=f
```

### GS solver (current code — correct, do not change)
```python
RHSe[i] += (1/self.Xg[ig,0]) * SourceTermg[ig] * self.Nrefg[ig,i] * self.detJg[ig] * self.Wg[ig]
```
This is correct for `Δ*ψ = f` with the LHS=−K convention.

---

## BUG-5: Sub-element node count wrong for high-order parent elements

### Symptom
High-order elements (p=2: QUA09, TRI06) produce large L2 errors (often > 10) at all mesh sizes,
with no sign of convergence. Lower-order elements (p=1: QUA04, TRI03) work correctly.

### Location
`src/Element.py` — `ReferenceElementTessellation` — the function that creates sub-elements.

### Wrong code
```python
# SUBELEM.n is set to 6 (TRI6 sub-element) instead of 9 (QUA9 parent)
SUBELEM.PSIe = np.zeros(SUBELEM.n)    # WRONG: 6 zeros, not 9
```

### Correct fix
```python
SUBELEM.n    = self.n                  # CORRECT: parent's DOF count
SUBELEM.PSIe = np.zeros(self.n)
```

---

## BUG-6: Ghost penalty jump sign wrong for odd derivative orders (p=1)

### Symptom
Ghost stabilization is ENABLED, but the solver performs WORSE than with ghost disabled. For
smooth test cases, the ghost-penalized system has larger errors. The LHS matrix is not symmetric.

### Location
`src/GradShafranovSolver.py` — `IntegrateGhostStabilizationTerms`

### Wrong code
```python
n_dot_dNg = np.concatenate((n_dot_dNg0, -n_dot_dNg1), axis=0)   # WRONG
```

### Correct fix
```python
sign_p = (-1) ** (p + 1)   # +1 for p=1 (odd); -1 for p=2 (even)
n_dot_dNg = np.concatenate((sign_p * n_dot_dNg0, n_dot_dNg1), axis=0)   # CORRECT
```

Apply this for ALL orders p in the loop `for p in range(1, ElOrder+1)`.

### Diagnostic
Set PSI to the exact analytical solution. For each ghost face, compute the jump:
```python
jump = sign_p * (n_dot_dNg0 @ PSI0_exact) + (n_dot_dNg1 @ PSI1_exact)
assert abs(jump) < 1e-8, f"BUG-6: jump={jump:.3e} at ghost face (should be ~0)"
```
For a smooth exact solution, the p-th normal derivative jump must be zero to machine precision.

---

## BUG-7: Machine-epsilon level-set values at interface cardinal points create degenerate cut elements

### Symptom
For specific mesh sizes, the solver crashes or produces anomalously large L2 errors. The affected
mesh sizes correspond to cases where mesh nodes coincide exactly with the interface extrema.

### Location
`src/Mesh.py` — `ClassifyElements`

### Correct fix
Add BEFORE standard element classification:
```python
for ielem in range(self.Ne):
    elem = self.Elements[ielem]
    corners = elem.LSe[:elem.numedges]
    max_phi = np.max(np.abs(corners))
    if max_phi > 0:
        tol = max_phi * 1e-10
        for _ci in range(elem.numedges):
            if abs(corners[_ci]) < tol:
                other_signs = [corners[j] for j in range(elem.numedges) if j != _ci]
                n_vacuum = sum(v > 0 for v in other_signs)
                n_plasma = sum(v < 0 for v in other_signs)
                if n_vacuum >= n_plasma:
                    elem.LSe[_ci] = tol
                else:
                    elem.LSe[_ci] = -tol
```

### Root cause
Floating-point evaluation of the level-set at a point exactly on the interface returns ±2.22e-16.
This creates a degenerate sub-element with plasma fraction ≈ 1e-16, causing Nitsche penalty
entries of O(β/h_eff) where h_eff ≈ 1e-16·h — i.e. O(10²²), corrupting the entire system.

---

## BUG-8: Nitsche penalty `beta` too large for p=1 elements

### Symptom
p=1 elements (QUA04, TRI03) show suboptimal convergence rate (~0.5 instead of 2.0) even after
all code bugs are fixed. p=2 elements work correctly with the same parameters.

### Root cause
The Nitsche penalty term amplifies the interface discretization error by β/h. For p=1 elements
with a non-polynomial exact solution, the FEM interpolant has O(h²) error at the interface.
The penalty amplifies this to O(β·h), which is O(1e10·h) for β=1e10. At all mesh sizes, this
is larger than domain stiffness (O(1)), so the solution minimizes interface residual rather than
solving the PDE in the domain.

### Validated parameter ranges (GS solver, LINEAR test case):
| Element | Optimal beta | Note |
|---------|-------------|------|
| TRI03   | 10–100      | `beta=100` confirmed rate 2.00 |
| TRI06   | 10–100      | `beta=100` confirmed rate 2.99 |
| QUA04   | 10–100      | `beta=100` confirmed rate ≥ 1.98 |
| QUA09   | 100         | `beta=100` confirmed rate 3.54 |

**Do not use beta > 1000 for any GS element type.** The GS exact solution is NOT a low-degree
polynomial, so large beta always degrades convergence.

---

## BUG-9: Ghost penalty `zeta` too large for p=1 elements

### Symptom
Same as BUG-8: suboptimal convergence rate. Both BUG-8 and BUG-9 must be fixed together.

### Root cause
The ghost penalty `ζ·h^{2p−1}·[[∂^p_n u_h]]²` scales as ζ times a mesh-independent quantity.
With ζ=1e8, ghost dominates by 1e8 × domain stiffness at all mesh sizes. The solution minimizes
gradient jumps instead of solving the PDE.

### Validated parameter ranges (GS solver, LINEAR test case):
| Element | Optimal zeta | Note |
|---------|-------------|------|
| TRI03   | 0           | Unstructured mesh; ghost not needed. zeta=0 confirms rate 2.00 |
| TRI06   | 0           | Unstructured mesh; ghost not needed. zeta=0 confirms rate 2.99 |
| QUA04   | 0–1         | Small ghost only. `zeta=100` degrades p=1 rates significantly |
| QUA09   | 100         | Structured rectangular mesh requires strong ghost (penalty=100·h³) |

**QUA09 structured mesh explanation:** Rectangular meshes produce more severe degenerate cut
configurations than unstructured TRI meshes. Near-degenerate cut elements (very thin slivers)
require strong ghost penalty to condition their DOFs. `zeta=100` with p=2 gives `penalty=100·h³`,
which is strong enough to stabilize but small enough not to dominate the O(h³) solution error.

**QUA04 note:** Large zeta (`zeta=100`) degrades p=1 convergence because `penalty=100·h¹` is
O(1) at h=0.1, comparable to domain stiffness. Use `zeta=0` or `zeta=1` for QUA04.

---

## BUG-10: Straight-line 1D Nitsche quadrature ignores curved interface for high-order elements

**STATUS IN GS SOLVER: FIXED** — `src/Element.py` `ComputeAdaptedQuadrature1D` correctly uses
`self.ElOrder` for the 1D basis order and all `XIint` nodes for curved interface quadrature.

### Symptom
TRI06 (p=2) absolute errors are ~10⁴ larger than expected. Rate for TRI06 is limited to ~2
instead of the theoretical ~3 for p=2 CutFEM.

### Root cause
The curved interface (quadratic approximation) was mapped using only the 2 endpoints, giving a
straight-line arc-length element with O(h²) geometric error. This limited Nitsche integration
accuracy to O(h²), capping the consistency residual at O(h²) instead of O(h³).

### Fix
```python
if self.ElOrder == 1:
    XIint_1D = self.InterfApprox.XIint[:2, :]
    Xint_1D  = self.InterfApprox.Xint[:2, :]
    order_1D = 1
else:
    # natural order [endpoint0, endpoint1, midpoint] matches 1D basis [ξ=-1, ξ=+1, ξ=0]
    XIint_1D = self.InterfApprox.XIint   # all nodes, no reorder needed
    Xint_1D  = self.InterfApprox.Xint
    order_1D = self.ElOrder
N1D, dNdxi1D = EvalRefLagrangeBasis(XIg1Dstand, 0, order_1D, deriv=deriv_order)
```

---

## Summary table: bugs, impact, fix location, GS solver status

| Bug | Impact | File | Fix | GS status |
|-----|--------|------|-----|-----------|
| BUG-1: Missing 4th QUA Gauss point | All QUA integrals wrong | `GaussQuadrature.py` | `zg[2,:]` → `zg[3,:]` | **NOT PRESENT** (verified) |
| BUG-2: Wrong Dom flags in L2 diagnostic | Diagnostic only | `_L2error.py` | `Dom==1→0`, `Dom==2→-1` | **FIXED** |
| BUG-3: False quadrature warning | No real impact | `Element.py` | Ignore for sub-elements | N/A |
| BUG-4: Source-term sign | All L2 errors O(1) | `Element.py` | `-=` for Poisson; `+=` for GS (CORRECT, no change) | **N/A — GS uses `+=` correctly** |
| BUG-5: Wrong sub-element n | p=2 errors O(10) at all h | `Element.py` | `SUBELEM.n = parent.n` | **FIXED** |
| BUG-6: Wrong ghost jump sign | Ghost breaks convergence | `GradShafranovSolver.py` | `sign_p = (-1)**(p+1)` | **FIXED** |
| BUG-7: Machine-epsilon phi | Crash at specific h | `Mesh.py` | Snap near-zero phi | **FIXED** |
| BUG-8: `beta` too large for p=1 | p=1 rate ~0.5 | test scripts | `beta=100–1000` for all GS families | **FIXED** (use 100) |
| BUG-9: `zeta` too large | Rate degradation | test scripts | `zeta=0` for TRI, `zeta=100` for QUA09 | **FIXED** |
| BUG-10: Straight-line 1D Nitsche | TRI06 errors ~10⁴ too large | `Element.py` | All XIint nodes + order-ElOrder basis | **FIXED** |
| INV-6: Vacuum stiffness assembled | O(h) saturation for all elements | `GradShafranovSolver.py` | Skip `Dom>0 and Teboun is None` + zero-diagonal sweep | **FIXED** (critical fix) |

---

## Validated convergence rates (GS solver, LINEAR test case)

Assembly: INV-6 fix applied, vacuum sub-elements skipped, zero-diagonal sweep.
Parameters and quadrature orders as noted.

| Element | beta | zeta | QuadOrder2D | h=1.0→0.5 | h=0.5→0.1 | h=0.1→0.06 | h=0.06→0.02 | Status |
|---------|------|------|-------------|-----------|-----------|------------|-------------|--------|
| TRI03   | 100  |  0   |  8          | —         | **3.08**  | **1.99**   | **2.00**    | O(h²) ✓ |
| TRI06   | 100  |  0   |  8          | —         | **6.21**  | **2.98**   | **2.99**    | O(h³) ✓ |
| QUA04   | 100  |  0   |  5          | —         | **3.29**  | **1.98**   | ~2.0 (est.) | O(h²) ✓ |
| QUA09   | 100  | 100  |  5          | —         | **4.80**  | **3.78**   | **3.54**    | O(h³) ✓ |

Notes:
- Asymptotic rates (h≤0.1) match theoretical optimum for all families.
- TRI03/TRI06 super-convergent pre-asymptotic rate at h=0.5→0.1 (mesh-dependent).
- QUA04 h=0.06→0.02 not re-run after confirming zeta=0 is optimal (zeta=100 sub-optimal run gave 2.59).
- QUA09 requires structured rectangular mesh ghost stabilization (`zeta=100`) for O(h³) rates.

---

## Recommended validation sequence

Run these checks in order. Each check isolates one component.

### V1: QUA quadrature exactness (catches BUG-1)
```python
from GaussQuadrature import GaussQuadrature
zg, wg, Ng = GaussQuadrature(2, 2)
result = sum(zg[i,0]**2 * zg[i,1]**2 * wg[i] for i in range(Ng))
assert abs(result - 4/9) < 1e-12, f"BUG-1: got {result}"
```

### V2: LHS symmetry (catches asymmetric Nitsche sign errors)
```python
LHS_arr = solver.LHS.toarray()
sym_err = np.max(np.abs(LHS_arr - LHS_arr.T))
assert sym_err < 1e-10 * np.max(np.abs(LHS_arr)), f"LHS not symmetric: {sym_err:.2e}"
```

### V3: Patch test — residual of exact solution (catches INV-2, INV-6, BUG-4 violations)
```python
# After AssembleGlobalSystem (before SolveSystem):
PSI_exact = np.array([solver.PlasmaCurrent.PSIanalytical(solver.MESH.X[i,:])
                      for i in range(solver.MESH.Nn)])
LHS_arr = solver.LHS.toarray()
residual = LHS_arr @ PSI_exact - solver.RHS.flatten()
norm_LHS = np.max(np.abs(LHS_arr))
max_res = np.max(np.abs(residual))
print(f"Patch test residual: {max_res:.2e}  (should be < 1e-8 * {norm_LHS:.2e})")
```

### V4: Ghost jump at exact solution (catches BUG-6)
```python
# For each ghost face, compute jump of normal derivative at exact solution
# Should be < 1e-8 for smooth PSI_exact
```

### V5: INV-6 diagnostic
```python
# After AssembleGlobalSystem, check that VacuumElem interior nodes have no domain stiffness
for ielem in solver.MESH.VacuumElems:
    ELEMENT = solver.MESH.Elements[ielem]
    if ELEMENT.Teboun is None:   # pure vacuum interior
        for i in range(ELEMENT.n):
            node = ELEMENT.Te[i]
            # row should have zero or only ghost contributions
```

### V6: Full convergence study
Run with validated parameters (beta=100, zeta=0 for TRI, zeta=100 for QUA09):
- TRI03: target rate ≥ 1.8, expect 2.0 at fine meshes
- TRI06: target rate ≥ 2.5, expect 3.0 at fine meshes
- QUA04: target rate ≥ 1.8, expect 2.0 at fine meshes
- QUA09: target rate ≥ 2.5, expect 3.0 at fine meshes (requires zeta=100)

---

## Application to Grad-Shafranov solver — key differences from Poisson

1. **Source term sign:** GS uses `Δ*ψ = f` (positive LHS operator). RHSe `+=` is CORRECT.
   Do NOT apply the Poisson BUG-4 fix (`+=`→`-=`) to the GS solver.

2. **Non-polynomial exact solution:** The GS LINEAR test solution is NOT a low-degree polynomial.
   BUG-8 and BUG-9 apply to ALL element types (including p=2). Use β=O(100) and ζ=O(0–100).

3. **QUA quadrature order:** Maximum supported order for QUA is 5. Use `QuadratureOrder2D=5`
   for all QUA runs; use `QuadratureOrder2D=8` for TRI runs.

4. **Structured vs unstructured mesh ghost:** QUA09 on structured rectangular meshes requires
   `zeta=100` for O(h³) convergence. TRI03/TRI06 on unstructured meshes are stable at `zeta=0`.

5. **INV-6 is the most critical fix:** Assembling vacuum elements saturated convergence at O(h)
   for ALL element families. The fix is a single guard in the NonCutElems assembly loop.

---

*End of FIX_SOLVER.md — Last updated: 2026-05-18*
