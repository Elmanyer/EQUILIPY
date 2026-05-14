#!/bin/bash

# Full Parameter Optimization for EQUILIPY CutFEM
# Runs optimization for TRI03, TRI06, TRI10 on all mesh levels

set -e  # Exit on error

echo "=========================================="
echo "EQUILIPY FULL CONVERGENCE ANALYSIS"
echo "=========================================="
echo ""
echo "This script will optimize parameters (beta, zeta) for:"
echo "  - TRI03 (Linear P1)"
echo "  - TRI06 (Quadratic P2)"
echo "  - TRI10 (Cubic P3)"
echo ""
echo "On each mesh level:"
echo "  - COARSE"
echo "  - MEDIUM"
echo "  - INTERMEDIATE"
echo "  - FINE"
echo "  - SUPERFINE"
echo "  - MEGAFINE"
echo ""
echo "=========================================="
echo ""

# Array of mesh levels
MESH_LEVELS=("COARSE" "MEDIUM" "INTERMEDIATE" "FINE" "SUPERFINE" "MEGAFINE")

# Run optimization for each mesh level
for MESH_LEVEL in "${MESH_LEVELS[@]}"; do
    echo ""
    echo "============================================"
    echo "Optimizing parameters on $MESH_LEVEL mesh"
    echo "============================================"

    python convergence_analysis.py \
        --elements TRI03 TRI06 TRI10 \
        --opt-mesh-level "$MESH_LEVEL" \
        --mesh-levels "$MESH_LEVEL" \
        --no-plots \
        --no-fine-tune

    echo ""
    echo "✓ $MESH_LEVEL mesh optimization complete"
    echo ""
done

echo ""
echo "=========================================="
echo "ALL OPTIMIZATIONS COMPLETE!"
echo "=========================================="
echo ""
echo "Results have been saved to:"
echo "  - convergence_results.csv"
echo "  - Console output above"
echo ""
