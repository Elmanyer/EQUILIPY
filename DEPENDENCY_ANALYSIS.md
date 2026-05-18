# EQUILIPY Repository Dependency Analysis

## Executive Summary
The repository has **critical path management issues** that prevent postprocessing scripts from running. Key problems:
1. Hardcoded absolute paths (e.g., `/home/elmanyer/Documents/...`)
2. Mixed import styles (relative paths, sys.path.append, direct imports)
3. Missing Python package structure (`__init__.py` files)
4. Inconsistent working directory assumptions

---

## Directory Structure

```
EQUILIPY/
├── src/                          # Core EQUILIPY solver modules
│   ├── GradShafranovSolver.py   # Main solver (imports Mesh, Element, etc.)
│   ├── Mesh.py
│   ├── Element.py
│   ├── GaussQuadrature.py
│   ├── FELagrangeanbasis.py
│   ├── Tokamak.py
│   ├── Magnet.py
│   ├── PlasmaCurrent.py
│   ├── InitialPlasmaBoundary.py
│   ├── InitialPSIGuess.py
│   ├── AnalyticalSolutions.py
│   ├── Greens.py
│   ├── Segment.py
│   ├── InterfaceApprox.py
│   ├── _Bfield.py
│   ├── _critical.py
│   ├── _initialisation.py
│   ├── _L2error.py
│   ├── _logging.py
│   ├── _output.py
│   ├── _plot.py
│   ├── _test.py
│   └── _update.py
│
├── TESTs/                        # Test files (WORKING - uses relative paths)
│   ├── TestFIXED-CutFEM-LINEAR.py
│   ├── TestFIXED-FEM-LINEAR.py
│   ├── TestFREE-ITER-SHAPEDCOILS.ipynb
│   └── [multiple test files]
│
├── postprocess/                  # Postprocessing modules (BROKEN)
│   ├── src/
│   │   ├── RenderFigure.py       # ❌ FAILS: can't import EQUILIPYpostprocess
│   │   ├── EQUILIPYpostprocess.py # ❌ Has hardcoded path: /home/elmanyer/...
│   │   ├── _ReadOutputFiles.py
│   │   ├── _LatexFigures.py
│   │   └── [other files]
│   ├── Figures/
│   │   └── [figure generation notebooks]
│   └── tests/
│       └── [test notebooks]
│
└── MESHES/                       # Mesh files

```

---

## Dependency Tree

### Forward Dependencies (What imports what)

```
GradShafranovSolver.py (src/)
├── imports: Mesh, Element, InitialPSIGuess, InitialPlasmaBoundary, Tokamak, 
│           PlasmaCurrent, Magnet, _output, _plot, _test, _update, _logging
│
├── TESTs/TestFIXED-CutFEM-LINEAR.py
│   └── sys.path.append('../src/') → imports GradShafranovSolver
│
└── postprocess/src/EQUILIPYpostprocess.py ❌ PROBLEM
    ├── Line 7: sys.path.append('/home/elmanyer/Documents/BSC/EQUILI/EQUILIPY/EQUILIPY/')
    │           (HARDCODED PATH - breaks for other users)
    ├── Line 10: from src.GradShafranovSolver import *
    │           (imports solver from src/)
    ├── Line 9: from _ReadOutputFiles import ReadOutputEQUILIPY
    ├── Line 11: from _LatexFigures import Equilipylatex
    │
    └── postprocess/src/RenderFigure.py ❌ PROBLEM
        ├── Line 2: from EQUILIPYpostprocess import *
        │          (same directory, but no PYTHONPATH setup)
        ├── Line 4: directory = "/home/elmanyer/Documents/..." 
        │          (HARDCODED PATH)
        └── Uses: EquilibriumResults class from EQUILIPYpostprocess

_LatexFigures.py (postprocess/src/)
├── imports: _plot (relative import, same directory)
└── Uses: classes from GradShafranovSolver (indirectly through EquilibriumResults)

_ReadOutputFiles.py (postprocess/src/)
├── imports: various standard libraries
└── Provides: ReadOutputEQUILIPY base class
```

---

## Current Import Patterns

### ✅ WORKING (TESTs)
```python
# TESTs/TestFIXED-CutFEM-LINEAR.py (relative path from TESTs dir)
import sys
sys.path.append('../src/')
from GradShafranovSolver import *
```
**Why it works**: TESTs is one level above src, so `../src/` resolves correctly.

### ❌ BROKEN (postprocess)
```python
# postprocess/src/EQUILIPYpostprocess.py (hardcoded absolute path)
sys.path.append('/home/elmanyer/Documents/BSC/EQUILI/EQUILIPY/EQUILIPY/')
from src.GradShafranovSolver import *
from _ReadOutputFiles import ReadOutputEQUILIPY
```
**Problems**:
- Hardcoded path `/home/elmanyer/...` - breaks for other users
- Relative import `_ReadOutputFiles` assumes same directory
- `src.GradShafranovSolver` assumes `src` is in appended path

```python
# postprocess/src/RenderFigure.py (no path setup)
from EQUILIPYpostprocess import *
directory = "/home/elmanyer/Documents/BSC/EQUILI/EQUILIPY/EQUILIPY/RESULTS/"
```
**Problems**:
- No sys.path management
- Hardcoded absolute paths
- Can't find EQUILIPYpostprocess (same directory but no PYTHONPATH)

---

## Issues Summary

| Issue | Location | Impact | Severity |
|-------|----------|--------|----------|
| Hardcoded paths | EQUILIPYpostprocess.py:7, RenderFigure.py:4 | Breaks on other machines | **CRITICAL** |
| Missing sys.path setup | RenderFigure.py | ModuleNotFoundError | **CRITICAL** |
| No __init__.py files | src/, postprocess/src/ | Not proper packages | **HIGH** |
| Mixed import styles | All files | Confusing, inconsistent | **MEDIUM** |
| Relative paths fragile | TESTs files | Breaks if files move | **MEDIUM** |

---

## Recommended Architecture

### Solution: Workspace-Relative Import Header System

Create `__init__.py` files that establish paths relative to the workspace root:

```
EQUILIPY/
├── __init__.py (sets up workspace root)
├── src/
│   └── __init__.py (imports all solver modules)
├── postprocess/
│   ├── __init__.py
│   └── src/
│       └── __init__.py (imports postprocess modules)
└── TESTs/
    └── (test files use workspace header)
```

Each module will start with a "header" that:
1. Gets the workspace root dynamically
2. Adds it to sys.path
3. Makes all imports relative to workspace root

---

## Implementation Plan

### Phase 1: Create Package Structure
- [ ] Add `__init__.py` to `src/`
- [ ] Add `__init__.py` to `postprocess/src/`
- [ ] Add root `__init__.py` to workspace

### Phase 2: Create Import Headers
- [ ] Add workspace header function to a new `_header.py` in src/
- [ ] Update all imports in src/ files to use relative imports with explicit paths
- [ ] Update all imports in postprocess/ files to use workspace-relative paths
- [ ] Update all imports in TESTs/ files

### Phase 3: Replace Hardcoded Paths
- [ ] Replace all hardcoded `/home/elmanyer/...` paths with workspace-relative paths
- [ ] Use `os.path.join()` with workspace root for file operations
- [ ] Make paths configurable or derivable from results structure

### Phase 4: Testing
- [ ] Test RenderFigure.py from postprocess/
- [ ] Test all TESTs files
- [ ] Test postprocess notebooks

---

## Example Header Code

```python
# Standard header for all modules
import sys
import os
from pathlib import Path

# Get workspace root (parent of wherever this module is)
def _get_workspace_root():
    """Get the EQUILIPY workspace root directory."""
    current = Path(__file__).resolve()
    while current.name != 'EQUILIPY' and current.parent != current:
        current = current.parent
    return current if current.name == 'EQUILIPY' else Path.cwd()

_WORKSPACE_ROOT = _get_workspace_root()
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
```

