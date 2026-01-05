#!/bin/bash

echo "========================================="
echo "Parsing all 2K input + 10K output benchmark results"
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

# Find and parse all test3_2Kin_10Kout_port*.json files
for file in benchmark_results/test3_2Kin_10Kout_port*.json; do
    if [ -f "$file" ]; then
        ((total++))
        filename=$(basename "$file")
        # Generate output filename by adding _parsed before .json
        output_file="${file%.json}_parsed.json"
        echo "[$total] Parsing: $filename"
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
    fi
done

echo "========================================="
echo "Parsing complete!"
echo "========================================="
echo "Total files: $total"
echo "Successful: $success"
echo "Failed: $failed"
echo ""
echo "Parsed files saved as: *_parsed.json"
echo "========================================="
