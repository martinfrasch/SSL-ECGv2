#!/bin/bash
#
# compare_methods.sh - Run ablation study comparing subject-wise vs within-subject CV
#
# This script runs experiments with both the CORRECT (subject-wise) and INCORRECT
# (within-subject) methods to quantify the impact of the data leakage bug.
#
# Usage:
#   bash scripts/compare_methods.sh
#
# Results will be saved with different timestamps, allowing direct comparison.

set -e  # Exit on error

echo "======================================================================"
echo "Ablation Study: Subject-Wise vs Within-Subject Cross-Validation"
echo "======================================================================"
echo ""
echo "This script will run two experiments:"
echo "  1. CORRECT: Subject-wise CV (no data leakage)"
echo "  2. INCORRECT: Within-subject CV (causes 20-30% inflation)"
echo ""
echo "Results will be saved separately for comparison."
echo ""
read -p "Press Enter to continue..."

# Change to codes directory
cd codes

# Experiment 1: CORRECT method (subject-wise)
echo ""
echo "======================================================================"
echo "Experiment 1: Running with CORRECT subject-wise cross-validation"
echo "======================================================================"
echo ""

python train_fixed.py \
    --random_seed 42 \
    --subject_wise True \
    --normalize_features True \
    --validate_split True \
    --epochs 30 \
    --total_folds 5

echo ""
echo "✓ Experiment 1 complete!"
echo ""
sleep 2

# Experiment 2: INCORRECT method (within-subject) for comparison
echo ""
echo "======================================================================"
echo "Experiment 2: Running with INCORRECT within-subject split"
echo "⚠️  WARNING: This is for comparison only - results are inflated!"
echo "======================================================================"
echo ""

python train_fixed.py \
    --random_seed 42 \
    --subject_wise False \
    --normalize_features True \
    --validate_split False \
    --epochs 30 \
    --total_folds 5

echo ""
echo "✓ Experiment 2 complete!"
echo ""

# Summary
echo ""
echo "======================================================================"
echo "Both experiments complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Compare results in output/ER_result/"
echo "2. Look for te_stress.csv files with different timestamps"
echo "3. Expect ~20-30% performance difference between methods"
echo ""
echo "Example analysis:"
echo "  cd ../output/ER_result"
echo "  # Compare the two most recent te_stress.csv files"
echo ""
echo "======================================================================"
