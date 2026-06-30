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
# Date: June 2026
# Institution: Barcelona Supercomputing Center (BSC)
# Department: Computer Applications in Science and Engineering (CASE)
# Research Group: Nuclear Fusion


"""
Universal import header for EQUILIPY workspace.
Ensures all modules can find each other regardless of execution location.

This module should be imported at the top of every EQUILIPY module to guarantee
that the workspace root is in sys.path and all relative imports work correctly.
"""
import sys
import os
from pathlib import Path


def _get_equilipy_root():
    """
    Dynamically locate the EQUILIPY workspace root directory.

    Searches from current file location upward until finding the 'EQUILIPY' folder,
    or a directory containing both 'src' and 'TESTs' subdirectories.

    Returns:
        Path: The EQUILIPY workspace root directory

    Raises:
        RuntimeError: If EQUILIPY root cannot be found
    """
    current = Path(__file__).resolve().parent
    attempts = 0
    max_attempts = 10

    # Search upward from this file's location
    while attempts < max_attempts and current.parent != current:
        current = current.parent
        attempts += 1

        # Check if this is the EQUILIPY root
        if current.name == 'EQUILIPY':
            return current
        if (current / 'src').is_dir() and (current / 'TESTs').is_dir():
            return current

    # Fallback: try current working directory
    cwd = Path.cwd()
    if (cwd / 'src').is_dir() and (cwd / 'TESTs').is_dir():
        return cwd

    raise RuntimeError(
        f"Could not locate EQUILIPY workspace root. "
        f"Started from {Path(__file__).resolve().parent}\n"
        f"Make sure _header.py is being imported from within the EQUILIPY workspace."
    )


# Initialize workspace root and add to path if needed
EQUILIPY_ROOT = _get_equilipy_root()

# Add key directories to sys.path for imports
# src/ - GradShafranovSolver, Mesh, Element, etc.
# postprocess/src/ - EQUILIPYpostprocess, etc.
src_dir = EQUILIPY_ROOT / 'src'
postprocess_src_dir = EQUILIPY_ROOT / 'postprocess' / 'src'

if str(EQUILIPY_ROOT) not in sys.path:
    sys.path.insert(0, str(EQUILIPY_ROOT))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(postprocess_src_dir) not in sys.path:
    sys.path.insert(0, str(postprocess_src_dir))
