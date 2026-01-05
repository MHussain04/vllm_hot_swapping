#!/usr/bin/env python3
"""
Parse vllm bench serve detailed JSON output and create a custom compact format.
Optionally fetch GPU metrics from Prometheus.

Usage:
    python parse_benchmark_results.py <input_json_file> <output_json_file> [--prometheus-url <url>] [--prometheus-user <user>] [--prometheus-password <password>]

Examples:
    python parse_benchmark_results.py benchmark_results/test.json custom_results.json
    python parse_benchmark_results.py benchmark_results/test.json custom_results.json --prometheus-url http://34.132.51.122:9090 --prometheus-user admin --prometheus-password omar
"""

import json
import sys
import math
from pathlib import Path
from datetime import datetime
import statistics

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def extract_summary_metrics(data):
    """
    Extract aggregate/summary metrics from the benchmark output.
    """
    summary = {
        'completed': data.get('completed'),
        'failed': data.get('failed'),
        'total_input_tokens': data.get('total_input_tokens'),
        'total_output_tokens': data.get('total_output_tokens'),
        'request_throughput': data.get('request_throughput'),
        'request_goodput': data.get('request_goodput'),
        'output_throughput': data.get('output_throughput'),
        'total_token_throughput': data.get('total_token_throughput'),

        'mean_ttft_ms': data.get('mean_ttft_ms'),
        'median_ttft_ms': data.get('median_ttft_ms'),
        'std_ttft_ms': data.get('std_ttft_ms'),
        'p99_ttft_ms': data.get('p99_ttft_ms'),

        'mean_tpot_ms': data.get('mean_tpot_ms'),
        'median_tpot_ms': data.get('median_tpot_ms'),
        'std_tpot_ms': data.get('std_tpot_ms'),
        'p99_tpot_ms': data.get('p99_tpot_ms'),

        'mean_itl_ms': data.get('mean_itl_ms'),
        'median_itl_ms': data.get('median_itl_ms'),
        'std_itl_ms': data.get('std_itl_ms'),
        'p99_itl_ms': data.get('p99_itl_ms'),
    }

    # Remove None values
    return {k: v for k, v in summary.items() if v is not None}


def extract_request_metrics(data):
    """
    Extract per-request metrics from parallel arrays.
    Calculate e2el = ttft + sum(itls) for each request.
    """
    custom_requests = {}

    input_lens = data.get('input_lens', [])
    output_lens = data.get('output_lens', [])
    ttfts = data.get('ttfts', [])
    itls = data.get('itls', [])

    num_requests = len(ttfts)

    for idx in range(num_requests):
        ttft = ttfts[idx]
        itl_list = itls[idx]
        e2el = ttft + sum(itl_list)

        custom_requests[f"request_id {idx}"] = {
            'prompt_tokens': input_lens[idx],
            'output_tokens': output_lens[idx],
            'ttft': ttft,
            'e2el': e2el
        }

    return custom_requests


def query_prometheus_range(prometheus_url, metric, start_time, end_time, step='1s', auth=None, return_timestamps=False):
    """
    Query Prometheus for a metric over a time range.

    Args:
        return_timestamps: If True, returns (timestamps, values). If False, returns just values.

    Returns:
        If return_timestamps=False: [value1, value2, ...]
        If return_timestamps=True: ([timestamp1, timestamp2, ...], [value1, value2, ...])
    """
    if not HAS_REQUESTS:
        print("Warning: 'requests' library not installed, skipping Prometheus query")
        return ([], []) if return_timestamps else []

    try:
        url = f"{prometheus_url}/api/v1/query_range"
        params = {
            'query': metric,
            'start': start_time,
            'end': end_time,
            'step': step
        }

        response = requests.get(url, params=params, auth=auth, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data['status'] != 'success':
            print(f"Warning: Prometheus query failed for {metric}: {data}")
            return ([], []) if return_timestamps else []

        # Extract values from response
        results = data['data']['result']
        if not results:
            print(f"Warning: No data returned for {metric}")
            return ([], []) if return_timestamps else []

        # Get values from first result (assuming single GPU)
        values = results[0]['values']
        # values are [timestamp, "value_str"] pairs

        if return_timestamps:
            timestamps = [v[0] for v in values]
            vals = [float(v[1]) for v in values]
            return (timestamps, vals)
        else:
            return [float(v[1]) for v in values]

    except Exception as e:
        print(f"Warning: Failed to query Prometheus for {metric}: {e}")
        return ([], []) if return_timestamps else []


def fetch_gpu_metrics(prometheus_url, start_time, end_time, auth=None, min_util_threshold=5.0):
    """
    Fetch GPU metrics from Prometheus and calculate statistics.

    Smart approach:
    1. Query GPU utilization for full window
    2. Find when GPU first becomes active (>= threshold) = actual benchmark start
    3. Use refined window (actual_start to end_time) for memory queries
    4. This excludes setup/initialization period automatically

    Args:
        start_time: Initial start time (from date - duration)
        end_time: End time (from date field in JSON)
        min_util_threshold: Minimum GPU utilization % to consider as "active" (default 5%)
    """
    print(f"\nFetching GPU metrics from Prometheus...")
    print(f"  Initial time range: {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}")

    # Calculate dynamic step to stay under Prometheus 11,000 point limit
    duration = end_time - start_time
    step_seconds = max(1, math.ceil(duration / 10000))
    step = f"{step_seconds}s"
    print(f"  Using step={step} for Prometheus queries (duration: {duration:.0f}s)")

    gpu_metrics = {}

    # Query GPU utilization with timestamps (filter for training_node)
    gpu_timestamps, gpu_util_values = query_prometheus_range(
        prometheus_url,
        'DCGM_FI_DEV_GPU_UTIL{gpu_tags=~".*training_node.*"}',
        start_time,
        end_time,
        step=step,
        auth=auth,
        return_timestamps=True
    )

    if gpu_util_values and gpu_timestamps:
        # Find indices where GPU is active (>= threshold)
        active_indices = [i for i, v in enumerate(gpu_util_values) if v >= min_util_threshold]

        if active_indices:
            # Find actual benchmark start (first active timestamp)
            actual_start_time = gpu_timestamps[active_indices[0]]
            actual_start_dt = datetime.fromtimestamp(actual_start_time)

            # Calculate GPU util stats from active values only
            active_util_values = [gpu_util_values[i] for i in active_indices]
            gpu_metrics['avg_gpu_util_%'] = statistics.mean(active_util_values)
            gpu_metrics['max_gpu_util_%'] = max(active_util_values)
            gpu_metrics['min_gpu_util_%'] = min(active_util_values)

            filtered_count = len(gpu_util_values) - len(active_util_values)
            print(f"  ✓ GPU utilization (active periods): avg={gpu_metrics['avg_gpu_util_%']:.2f}%, max={gpu_metrics['max_gpu_util_%']:.2f}%, min={gpu_metrics['min_gpu_util_%']:.2f}%")
            print(f"    Filtered {filtered_count} idle samples below {min_util_threshold}%")
            print(f"  ✓ Refined benchmark window: {actual_start_dt} to {datetime.fromtimestamp(end_time)}")
            print(f"    (Excluded {actual_start_time - start_time:.1f}s of setup/initialization)")

            # Use refined window for memory queries
            refined_start_time = actual_start_time
        else:
            print(f"  ⚠ Warning: All GPU utilization samples below {min_util_threshold}% threshold")
            print(f"    Using original time window for memory queries")
            refined_start_time = start_time
    else:
        print(f"  ⚠ Warning: No GPU utilization data available")
        refined_start_time = start_time

    # Query GPU memory used with refined window (filter for training_node)
    memory_used_values = query_prometheus_range(
        prometheus_url,
        'DCGM_FI_DEV_FB_USED{gpu_tags=~".*training_node.*"}',
        refined_start_time,
        end_time,
        step=step,
        auth=auth
    )

    if memory_used_values:
        gpu_metrics['avg_gpu_memory_used_mb'] = statistics.mean(memory_used_values)
        gpu_metrics['max_gpu_memory_used_mb'] = max(memory_used_values)
        print(f"  ✓ GPU memory used: avg={gpu_metrics['avg_gpu_memory_used_mb']:.2f}MB, max={gpu_metrics['max_gpu_memory_used_mb']:.2f}MB")

    # Query GPU memory free with refined window (filter for training_node)
    memory_free_values = query_prometheus_range(
        prometheus_url,
        'DCGM_FI_DEV_FB_FREE{gpu_tags=~".*training_node.*"}',
        refined_start_time,
        end_time,
        step=step,
        auth=auth
    )

    if memory_free_values:
        gpu_metrics['avg_gpu_memory_free_mb'] = statistics.mean(memory_free_values)
        gpu_metrics['min_gpu_memory_free_mb'] = min(memory_free_values)
        print(f"  ✓ GPU memory free: avg={gpu_metrics['avg_gpu_memory_free_mb']:.2f}MB, min={gpu_metrics['min_gpu_memory_free_mb']:.2f}MB")

    return gpu_metrics


def parse_vllm_benchmark_output(input_file, prometheus_url=None, prometheus_auth=None):
    """
    Parse vllm bench serve detailed JSON output.
    Optionally fetch GPU metrics from Prometheus.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    summary = extract_summary_metrics(data)
    requests = extract_request_metrics(data)

    # Fetch GPU metrics if Prometheus URL provided
    if prometheus_url:
        date_str = data.get('date')
        duration = data.get('duration')

        if date_str and duration:
            try:
                # Parse date string: "20251204-125327" -> datetime
                # IMPORTANT: 'date' field represents END time (when file was saved)
                end_dt = datetime.strptime(date_str, '%Y%m%d-%H%M%S')
                end_time = end_dt.timestamp()
                start_time = end_time - duration

                # Fetch GPU metrics
                gpu_metrics = fetch_gpu_metrics(prometheus_url, start_time, end_time, auth=prometheus_auth)

                # Add GPU metrics to summary
                summary.update(gpu_metrics)

            except Exception as e:
                print(f"Warning: Failed to fetch GPU metrics: {e}")
        else:
            print("Warning: 'date' or 'duration' not found in benchmark output, skipping GPU metrics")

    return {
        'summary': summary,
        'requests': requests
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_benchmark_results.py <input_json_file> <output_json_file> [--prometheus-url <url>] [--prometheus-user <user>] [--prometheus-password <password>]")
        print("\nExamples:")
        print("  python parse_benchmark_results.py benchmark_results/test.json custom_results.json")
        print("  python parse_benchmark_results.py benchmark_results/test.json custom_results.json --prometheus-url http://34.132.51.122:9090 --prometheus-user admin --prometheus-password omar")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Parse optional arguments
    prometheus_url = None
    prometheus_user = None
    prometheus_password = None

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--prometheus-url' and i + 1 < len(sys.argv):
            prometheus_url = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--prometheus-user' and i + 1 < len(sys.argv):
            prometheus_user = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--prometheus-password' and i + 1 < len(sys.argv):
            prometheus_password = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    if prometheus_url:
        print(f"Prometheus URL: {prometheus_url}")
        if prometheus_user:
            print(f"Prometheus authentication: user={prometheus_user}")

    # Create auth tuple if credentials provided
    prometheus_auth = None
    if prometheus_user and prometheus_password:
        prometheus_auth = (prometheus_user, prometheus_password)

    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    print(f"Parsing benchmark results from: {input_file}")

    try:
        custom_output = parse_vllm_benchmark_output(input_file, prometheus_url, prometheus_auth)

        with open(output_file, 'w') as f:
            json.dump(custom_output, f, indent=2)

        print(f"\n✓ Successfully created custom results file: {output_file}")
        print(f"✓ Summary metrics: {len(custom_output['summary'])} fields")
        print(f"✓ Per-request data: {len(custom_output['requests'])} requests")

    except Exception as e:
        print(f"✗ Error parsing benchmark results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
