#!/bin/bash

echo "========================================="
echo "Parsing Test 2 benchmark results"
echo "========================================="
echo ""

# Prometheus configuration
PROMETHEUS_URL="http://34.132.51.122:9090/"
PROMETHEUS_USER="admin"
PROMETHEUS_PASSWORD="omar"

# Counter for tracking
total=0
success=0
failed=0

# ==============================================================================
# Parse Test 2: 10K input + 2K output (port 8002 only)
# ==============================================================================
file="benchmark_results/test2_10Kin_2Kout_port8002.json"
if [ -f "$file" ]; then
    ((total++))
    filename=$(basename "$file")
    output_file="${file%.json}_parsed.json"
    echo "[1] Parsing: $filename"
    echo "  Output: $(basename $output_file)"

    python parse_benchmark_results.py "$file" "$output_file" \
        --prometheus-url "$PROMETHEUS_URL" \
        --prometheus-user "$PROMETHEUS_USER" \
        --prometheus-password "$PROMETHEUS_PASSWORD"

    if [ $? -eq 0 ]; then
        ((success++))
        echo "  ✓ Success"
    else
        ((failed++))
        echo "  ✗ Failed"
    fi
    echo ""
else
    echo "⚠ Warning: $file not found, skipping..."
    echo ""
fi

# ==============================================================================
# Parse Test 2: 2K input + 10K output (port 8002 only)
# ==============================================================================
file="benchmark_results/test2_2Kin_10Kout_port8002.json"
if [ -f "$file" ]; then
    ((total++))
    filename=$(basename "$file")
    output_file="${file%.json}_parsed.json"
    echo "[2] Parsing: $filename"
    echo "  Output: $(basename $output_file)"

    python parse_benchmark_results.py "$file" "$output_file" \
        --prometheus-url "$PROMETHEUS_URL" \
        --prometheus-user "$PROMETHEUS_USER" \
        --prometheus-password "$PROMETHEUS_PASSWORD"

    if [ $? -eq 0 ]; then
        ((success++))
        echo "  ✓ Success"
    else
        ((failed++))
        echo "  ✗ Failed"
    fi
    echo ""
else
    echo "⚠ Warning: $file not found, skipping..."
    echo ""
fi

# ==============================================================================
# SUMMARY
# ==============================================================================
echo "========================================="
echo "Parsing complete!"
echo "========================================="
echo "Total files: $total"
echo "Successful: $success"
echo "Failed: $failed"
echo ""
if [ $success -gt 0 ]; then
    echo "Parsed files saved as: *_parsed.json in benchmark_results/"
fi
echo "========================================="
echo ""
