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
EQUILIPY Solver Package

Core solver and numerical methods for Grad-Shafranov problem solving.
All modules are set up with workspace-relative imports via _header.py
"""

__all__ = [
    'GradShafranovSolver',
    'Mesh',
    'Element',
    'FELagrangeanbasis',
    'Tokamak',
    'Magnet',
    'Greens',
    'Segment',
    'InitialPlasmaBoundary',
    'InitialPSIGuess',
    'PlasmaCurrent',
    'AnalyticalSolutions',
    'GaussQuadrature',
    'InterfaceApprox',
]
