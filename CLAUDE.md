# CLAUDE.md — EQUILIPY GradShafranov CutFEM Solver

Developer reference for the EQUILIPY Grad-Shafranov (GS) CutFEM solver. Covers the assembly
workflow, sign conventions, all applied bug fixes, validated parameters, and convergence results.

---

## Project overview

EQUILIPY solves the Grad-Shafranov equation for MHD plasma equilibria using CutFEM (Cut Finite
Element Method). The plasma boundary is implicitly defined by a level-set function; the mesh does
not conform to it. Nitsche's method enforces the interface condition weakly; ghost penalty
stabilization conditions near-degenerate DOFs.

**Repository layout:**
- `src/` — solver source (GradShafranovSolver.py, Element.py, Mesh.py, etc.)
- `TESTs/` — test scripts and notebooks
- `MESHES/` — pre-built mesh files (TRI03, TRI06, QUA04, QUA09 families)
- `FIX_SOLVER.md` — detailed bug-fix reference with diagnostics

---

## GS equation and sign conventions

Two formulations are supported, selected by `FIXED_BOUNDARY`:

**FIXED_BOUNDARY = True** — solve only on plasma domain Ω⁻:
```
Δ*ψ = f(ψ)    on Ω⁻
ψ = ψ_D        on Γ (plasma boundary, Nitsche)
ψ = 0          on ∂Ω_comp (outer wall, strong Dirichlet)
```

**FIXED_BOUNDARY = False** — solve on full computational domain Ω = Ω⁻ ∪ Γ ∪ Ω⁺:
```
Δ*ψ = f(ψ)    on Ω⁻  (plasma)
Δ*ψ = 0       on Ω⁺  (vacuum)
ψ = ψ_B        on ∂Ω_comp (outer wall, from Biot-Savart)
```
The plasma boundary Γ is an evolving interior interface tracked by the level-set; Nitsche enforces
weak continuity across it. The level-set is updated each inner iteration until convergence.

where `Δ* = R ∂/∂R (1/R ∂/∂R) + ∂²/∂Z²` and `f = μ₀·R·j_φ`.

**IBP of GS operator:** `∇·((1/R)∇ψ) = (1/R)Δ*ψ`, so:
```
∫_Ω (1/R) ∇ψ·∇v = −∫_Ω (1/R) f v    (ignoring boundary terms)
```

### LHS = −K convention (INV-1)

The stiffness matrix is assembled with a global minus sign:
```python
LHSe[i,j] -= (1/R) * grad_N_i @ grad_N_j * detJ * w
```
The system is `(−K)·ψ = RHS`. Since GS IBP gives `K·ψ = −(f,v)`, the system becomes
`(−K)·ψ = (f,v)`, so `RHS[i] = +(f, N_i)`:
```python
RHSe[i] += (1/R) * f * N_i * detJ * w    # += is CORRECT for GS
```

**This is the opposite of the Poisson solver** (−Δu=f uses `-=`). Never apply the Poisson
sign fix to the GS solver. See `FIX_SOLVER.md` BUG-4 for the full explanation.

### Domain (Dom) flag convention

| Dom | Meaning |
|-----|---------|
| −1  | Interior plasma — fully inside Ω⁻ |
|  0  | Cut element — crossed by plasma boundary Γ |
| +1  | Vacuum interior — fully outside Ω⁻ |
| +2  | Outer computational boundary — nodes on ∂Ω_comp |

---

## Assembly workflow (`AssembleGlobalSystem`)

The assembly consists of five phases. Phases 1 and 2 are sensitive to `FIXED_BOUNDARY`;
the remaining phases are identical in both modes.

### `FIXED_BOUNDARY` switch summary

| Phase | FIXED_BOUNDARY = True | FIXED_BOUNDARY = False |
|-------|----------------------|------------------------|
| 1 – NonCutElems | Skip `Dom>0, Teboun=None` (INV-6) | Assemble all elements |
| 2 – PlasmaBoundElems sub-elems | Plasma sub-elems only (`Dom<0`) | Both plasma and vacuum sub-elems |
| 3 – Nitsche interface | Enforce ψ=ψ_D (known exact) | Enforce ψ=ψ_D (previous iterate) |
| 4 – Ghost penalty | Unchanged | Unchanged |
| 5 – Zero-diagonal sweep | Patches unreachable vacuum DOFs | No-op (all DOFs assembled) |

### Phase 1 — Non-cut elements (`self.MESH.NonCutElems`)

Iterates over all non-cut elements. The guard is active only in FIXED_BOUNDARY mode (INV-6):

```python
for ielem in self.MESH.NonCutElems:
    ELEMENT = self.MESH.Elements[ielem]
    # FIXED_BOUNDARY (INV-6): skip pure vacuum interior; their DOFs are
    # conditioned by ghost penalty + zero-diagonal sweep.
    # FREE_BOUNDARY: include vacuum elements so Δ*ψ=0 is solved on Ω⁺.
    if self.FIXED_BOUNDARY and ELEMENT.Dom > 0 and ELEMENT.Teboun is None:
        continue
    # Source term: only for plasma interior (Dom < 0)
    SourceTermg = np.zeros([ELEMENT.ng])
    if ELEMENT.Dom < 0:
        PSIg = ELEMENT.Nrefg @ ELEMENT.PSIe
        for ig in range(ELEMENT.ng):
            SourceTermg[ig] = self.PlasmaCurrent.SourceTerm(ELEMENT.Xg[ig,:], PSIg[ig])
    LHSe, RHSe = ELEMENT.IntegrateElementalDomainTerms(SourceTermg)
    if ELEMENT.Teboun is not None:
        LHSe, RHSe = ELEMENT.PrescribeDirichletBC(LHSe, RHSe)
    # assemble into self.LHS, self.RHS ...
```

Elements always assembled (both modes):
- `Dom < 0` (plasma interior): full source term
- `Dom = +2` or `Teboun is not None` (outer boundary): zero source + Dirichlet BC

Elements skipped only in FIXED_BOUNDARY mode (INV-6):
- `Dom > 0 and Teboun is None` (pure vacuum interior)

### Phase 2 — Cut element sub-elements (`self.MESH.PlasmaBoundElems`)

Each cut element is tessellated into plasma and vacuum sub-elements. In FIXED_BOUNDARY mode,
only plasma sub-elements (Dom < 0) contribute; in FREE_BOUNDARY mode, both sides are assembled
(vacuum sub-elements use zero source term, representing `Δ*ψ = 0`):

```python
for ielem in self.MESH.PlasmaBoundElems:
    ELEMENT = self.MESH.Elements[ielem]
    for SUBELEM in ELEMENT.SubElements:
        # FIXED_BOUNDARY: skip vacuum sub-elements (one-sided IBP flux is O(h),
        # no Nitsche cancellation → saturates convergence).
        # FREE_BOUNDARY: include vacuum sub-elements; source term stays zero.
        if self.FIXED_BOUNDARY and SUBELEM.Dom >= 0:
            continue
        SourceTermg = np.zeros([SUBELEM.ng])
        if SUBELEM.Dom < 0:   # source only in plasma
            PSIg = SUBELEM.Nrefg @ ELEMENT.PSIe
            for ig in range(SUBELEM.ng):
                SourceTermg[ig] = self.PlasmaCurrent.SourceTerm(SUBELEM.Xg[ig,:], PSIg[ig])
        LHSe, RHSe = SUBELEM.IntegrateElementalDomainTerms(SourceTermg)
        # assemble using ELEMENT.Te (parent node indices) ...
```

Why vacuum sub-elements are skipped in FIXED_BOUNDARY mode: their IBP boundary flux
`∫_∂K⁺ (1/R) N_k (n⁺·∇ψ) dΓ` is O(h) and has no Nitsche cancellation term (one-sided
Nitsche uses only the plasma side n⁻). Including it saturates convergence to O(h).
In FREE_BOUNDARY mode this is acceptable — full-domain accuracy is needed for level-set
evolution, not optimal convergence to a fixed exact solution.

### Phase 3 — Nitsche interface terms (`self.MESH.PlasmaBoundActiveElems`)

Symmetric Nitsche enforcement of `ψ = ψ_D` on the plasma boundary Γ. Uses one-sided
plasma normal n⁻. Sign convention (with LHS=−K):

| Term | LHS | RHS |
|------|-----|-----|
| Consistency | `+= N_i (n⁻·∇N_j/R)` | — |
| Symmetry    | `+= (n⁻·∇N_i/R) N_j` | — |
| Penalty     | `−= (β/h) N_i N_j` | — |
| RHS symmetry | — | `+= ψ_D (n⁻·∇N_i/R)` |
| RHS penalty  | — | `−= (β/h) ψ_D N_i` |

### Phase 4 — Ghost penalty stabilization (`IntegrateGhostStabilizationTerms`)

Optional (activated by `self.GhostStabilization = True`). Penalizes normal derivative jumps
across ghost faces shared between cut elements and their vacuum neighbors:

```python
penalty = self.zeta * h**(2*p - 1)   # for derivative order p
sign_p = (-1)**(p + 1)               # +1 for p=1, -1 for p=2
n_dot_dNg = np.concatenate((sign_p * n_dot_dNg0, n_dot_dNg1), axis=0)
jump = n_dot_dNg @ PSI_vals           # should be ~0 at exact solution
LHSe -= penalty * np.outer(n_dot_dNg, n_dot_dNg)   # -=: consistent with LHS=−K
```

### Phase 5 — Zero-diagonal sweep

After all assembly, some vacuum-interior DOFs unreachable by ghost faces have zero diagonal,
making the system singular. Fixed by setting unit diagonal:

```python
LHS_csr = self.LHS.tocsr()
diag = np.array(LHS_csr.diagonal())
for i in range(self.MESH.Nn):
    if abs(diag[i]) < 1e-30:
        self.LHS[i, i] = 1.0
```

---

## All bug fixes applied to the GS solver

### INV-6: Vacuum elements assembled into domain stiffness [FIXED — FIXED_BOUNDARY only]

**Root cause:** The original assembly iterated over `self.MESH.NonCutElems` without filtering,
assembling domain stiffness `∫_K (1/R) ∇N_i·∇N_j` for vacuum elements (Dom=+1). Since the GS
LINEAR exact solution satisfies `Δ*ψ=f ≠ 0` everywhere, the assembled vacuum system (with f=0)
differs from ψ_exact in the vacuum. Cut element boundary nodes (shared between plasma K⁻ and
adjacent vacuum elements) propagated this O(h) mismatch into the plasma solution through Nitsche
coupling, saturating convergence at O(h) for ALL element families regardless of polynomial order.

**Fix:** Guards conditioned on `self.FIXED_BOUNDARY`:
- Phase 1: `if self.FIXED_BOUNDARY and ELEMENT.Dom > 0 and ELEMENT.Teboun is None: continue`
- Phase 2: `if self.FIXED_BOUNDARY and SUBELEM.Dom >= 0: continue`

In FREE_BOUNDARY mode both guards are inactive — the full domain is assembled so the
level-set can evolve (vacuum contribution `Δ*ψ=0` is physically correct in Ω⁺).

**Impact:** TRI03 rates went from {0.93, 0.35, 0.87} → {3.08, 1.99, 2.00} (O(h) → O(h²)).

**File:** `src/GradShafranovSolver.py`, NonCutElems loop (Phase 1) and PlasmaBoundElems loop (Phase 2).

---

### BUG-6: Ghost penalty jump sign wrong for p=1 (odd derivative orders) [FIXED]

**Root cause:** The jump concatenation used `concat(dNg0, -dNg1)` instead of the correct formula
`concat(sign_p * dNg0, dNg1)`. For p=1, sign_p=+1, so the correct code is `concat(+dNg0, dNg1)`.
The wrong code effectively computed the sum of normal derivatives instead of the jump, making the
ghost penalty non-zero even at smooth functions and breaking LHS symmetry.

**Fix:** Applied `sign_p = (-1)**(p+1)` in `IntegrateGhostStabilizationTerms`.

**File:** `src/GradShafranovSolver.py`.

---

### BUG-5: Sub-element DOF count wrong for high-order parents [FIXED]

**Root cause:** When a QUA09 parent (n=9) was tessellated into sub-triangles, the sub-elements
inherited their own geometry-based DOF count instead of the parent's. Assembly loops only covered
6 DOFs instead of 9, leaving 3 parent nodes with zero stiffness contributions from cut elements.

**Fix:** `SUBELEM.n = self.n` (parent DOF count) in `ReferenceElementTessellation`.

**File:** `src/Element.py`.

---

### BUG-7: Machine-epsilon level-set values creating degenerate cut elements [FIXED]

**Root cause:** Level-set evaluation at interface cardinal points returned ±2.22e-16 instead of 0.
This created sub-elements with plasma fraction ≈ 1e-16, causing Nitsche penalty matrix entries
of O(10²²) that corrupted the entire global system.

**Fix:** Near-zero phi snapping with relative tolerance 1e-10 × max corner phi in `ClassifyElements`.

**File:** `src/Mesh.py`.

---

### BUG-10: Straight-line 1D Nitsche quadrature for high-order elements [FIXED]

**Root cause:** The Nitsche 1D integration over the interface approximation used only 2 endpoints
(straight-line arc length), ignoring the quadratic midpoint for p=2 elements. This limited
Nitsche consistency to O(h²), capping TRI06 rates at 2 instead of 3.

**Fix:** Use `order_1D = self.ElOrder` and all `XIint` nodes in `ComputeAdaptedQuadrature1D`.
Node order [endpoint0, endpoint1, midpoint] already matches the 1D Lagrange basis convention
[ξ=−1, ξ=+1, ξ=0] — no reordering needed.

**File:** `src/Element.py`.

---

### BUG-8/9: Nitsche beta and ghost zeta too large for p=1 elements [FIXED]

**Root cause:** Large β (1e10) and ζ (1e8) dominated the system over domain stiffness at all
mesh sizes, regardless of element order, because the GS exact solution is NOT a polynomial.

**Fix:** Use β=100 and ζ=0 (TRI) or ζ=100 (QUA09 structured mesh) for the GS LINEAR case.

**Files:** Test scripts and notebooks (not source code).

---

### BUG-2: Wrong Dom flags in L2 diagnostic [FIXED]

**Fix:** `Dom==0` for cut elements, `Dom==-1` for interior plasma (not 1 and 2).

**File:** `src/_L2error.py`.

---

### BUG-1: QUA Gauss point missing [NOT PRESENT in GS solver]

**Verified:** `∫_{[-1,1]²} x²y²` = 0.4444444444 (error 2.78e-16). QUA order-2 rule has 4 correct
points. No fix needed.

However, note the **QUA maximum quadrature order limitation**: only orders 1–5 are implemented.
Requesting order 6–8 causes `UnboundLocalError`. Always use `QuadratureOrder2D=5` for QUA elements.

---

## Solver parameters

### Element-type-specific settings (validated)

| Element | QuadOrder2D | beta | zeta | GhostStabilization |
|---------|-------------|------|------|--------------------|
| TRI03   | 8           | 100  | 0    | False (or True with zeta=0) |
| TRI06   | 8           | 100  | 0    | False (or True with zeta=0) |
| QUA04   | 5           | 100  | 0–1  | False (or True with small zeta) |
| QUA09   | 5           | 100  | 100  | True |

### Common solver config (FIXED_BOUNDARY mode)

```python
SOLVER_CONFIG = {
    'FIXED_BOUNDARY': True,
    'RunTests': False,
    'PARALLEL': False,
    'QuadratureOrder2D': 8,      # override to 5 for QUA elements
    'QuadratureOrder1D': 6,
    'ext_maxiter': 1,            # single outer iteration for FIXED_BOUNDARY
    'ext_tol': 1.0e-3,
    'int_maxiter': 50,
    'int_tol': 1.0e-10,
    'tol_saddle': 0.1,
    'Nconstrainedges': -1,
    'R0_axis': 6.0, 'Z0_axis': 1.0,
    'R0_saddle': 5.0, 'Z0_saddle': -3.5,
    'opti_maxiter': 50, 'opti_tol': 1.0e-6,
    'dim': 2,
}
```

---

## Validated convergence rates

**Test case:** LINEAR model — `j_φ = R/μ₀`, `f = R²`, exact solution `ψ_exact = R*⁴/8 + c₀ + c₁R*² + c₂(R*⁴−4R*²Z*²)`.

All runs used `FIXED_BOUNDARY=True`, `ext_maxiter=1`, `beta=100`.

| Element | zeta | h=1.0→0.5 rate | h=0.5→0.1 rate | h=0.1→0.06 rate | h=0.06→0.02 rate | Asymptotic |
|---------|------|---------------|----------------|-----------------|------------------|------------|
| TRI03   |  0   | (coarse)      | 3.08           | **1.99**        | **2.00**         | O(h²) ✓   |
| TRI06   |  0   | (coarse)      | 6.21           | **2.98**        | **2.99**         | O(h³) ✓   |
| QUA04   |  0   | (coarse)      | 3.29           | **1.98**        | ~2.0 (est.)      | O(h²) ✓   |
| QUA09   | 100  | (coarse)      | 4.80           | **3.78**        | **3.54**         | O(h³) ✓   |

All element families achieve optimal CutFEM convergence rates in the asymptotic regime (h ≤ 0.1).
Pre-asymptotic super-convergence at coarse meshes is expected and not a reliability concern.

---

## Import system

Source files use a centralized header-based import system. Test scripts must add the `src/`
directory to `sys.path` and import `_header` first:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from _header import EQUILIPY_ROOT
```

The `_header.py` module sets `EQUILIPY_ROOT` and any other workspace-relative path constants
needed by the solver. Do not use absolute paths in test scripts.

---

## Test scripts

| Script | Purpose |
|--------|---------|
| `TESTs/convergence_analysis.py` | Full convergence study with parameter sweeps for all element types |
| `TESTs/run_fine_convergence.py` | Fine mesh convergence (h up to 0.02) for TRI03 and TRI06 |
| `TESTs/run_residual_diag.py` | Patch test: assembles system, checks residual of PSI_exact |
| `TESTs/run_pointwise_diag.py` | Point-wise convergence at key nodes (magnetic axis, etc.) |

---

## Key files

| File | Role |
|------|------|
| `src/GradShafranovSolver.py` | Main solver: assembly, solve, L2 error computation |
| `src/Element.py` | Element integrals: domain terms, Nitsche interface, ghost quadratures |
| `src/Mesh.py` | Mesh loading, element classification, level-set snapping |
| `src/GaussQuadrature.py` | Gauss point tables (TRI orders 1–8, QUA orders 1–5) |
| `src/FELagrangeanbasis.py` | Shape functions and physical gradients (orders 1–3) |
| `src/PlasmaCurrent.py` | Current models: LINEAR, NONLINEAR, etc.; PSIanalytical |
| `src/_L2error.py` | L2 error computation over plasma domain |
| `FIX_SOLVER.md` | Full bug-fix reference with diagnostics and root-cause analysis |

---

*Last updated: 2026-06-30 — Added FIXED_BOUNDARY domain-integration switch: free-boundary mode now assembles the full computational domain (Δ*ψ=0 in vacuum) so the plasma level-set can evolve. Optimal convergence confirmed for all 4 element families in fixed-boundary mode.*
