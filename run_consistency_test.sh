#!/bin/bash
# Consistency test: Run training 10 times and collect results
# Usage: ./run_consistency_test.sh

set -e

cd /Users/mzus/dev/TimeWaste
source .venv/bin/activate

RESULTS_FILE="consistency_test_results.csv"
LOG_DIR="consistency_logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize results file with header
echo "run,accuracy,f1_macro,f1_weighted,ic,ic_pvalue,dir_accuracy,sharpe,ann_return,vs_naive" > "$RESULTS_FILE"

echo "========================================"
echo "Starting 10-run consistency test"
echo "========================================"

for i in $(seq 2 10); do
    echo ""
    echo "========================================"
    echo "Run $i/10 - Starting at $(date)"
    echo "========================================"
    
    MODEL_DIR="models/consistency-test-run-$i"
    LOG_FILE="$LOG_DIR/run_$i.log"
    
    # Run training (suppress harmless RuntimeWarning about module import)
    python -W ignore::RuntimeWarning -m tweet_classifier.train \
        --data-path output/test2_entry_fix.csv \
        --output-dir "$MODEL_DIR" \
        --epochs 5 \
        --batch-size 16 \
        --evaluate-test 2>&1 | tee "$LOG_FILE"
    
    # Extract metrics from log (macOS compatible - no grep -P)
    ACCURACY=$(grep "Test Accuracy:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    F1_MACRO=$(grep "Test F1 (macro):" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    F1_WEIGHTED=$(grep "Test F1 (weighted):" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    
    # Extract IC and p-value using sed (macOS compatible)
    IC_LINE=$(grep "Information Coefficient:" "$LOG_FILE" | tail -1)
    IC=$(echo "$IC_LINE" | sed 's/.*Information Coefficient: \([0-9.-]*\).*/\1/')
    IC_PVALUE=$(echo "$IC_LINE" | sed 's/.*p=\([0-9.]*\)).*/\1/')
    
    DIR_ACC=$(grep "Directional Accuracy:" "$LOG_FILE" | tail -1 | sed 's/.*: \([0-9.]*\)%.*/\1/')
    SHARPE=$(grep "Simulated Sharpe" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    ANN_RETURN=$(grep "Annualized Return" "$LOG_FILE" | tail -1 | sed 's/.*: \([0-9.]*\)%.*/\1/')
    VS_NAIVE=$(grep "Improvement vs Naive:" "$LOG_FILE" | tail -1 | sed 's/.*: \([+-]*[0-9.]*\)%.*/\1/')
    
    # Append to results
    echo "$i,$ACCURACY,$F1_MACRO,$F1_WEIGHTED,$IC,$IC_PVALUE,$DIR_ACC,$SHARPE,$ANN_RETURN,$VS_NAIVE" >> "$RESULTS_FILE"
    
    echo ""
    echo "Run $i complete: IC=$IC (p=$IC_PVALUE), Accuracy=$ACCURACY"
    echo "----------------------------------------"
done

echo ""
echo "========================================"
echo "All 10 runs complete!"
echo "========================================"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Logs saved to: $LOG_DIR/"
echo ""
echo "Summary:"
cat "$RESULTS_FILE"
echo ""
echo "To compute statistics, run:"
echo "  python -c \"import pandas as pd; df=pd.read_csv('$RESULTS_FILE'); print(df.describe())\""

