#!/usr/bin/env python3
"""
vLLM Benchmark Runner - Connects to existing server
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
BASE_DIR = Path.home() / "vllm_benchmarking"
RESULTS_DIR = BASE_DIR / "results"
STATE_FILE = BASE_DIR / "benchmark_state.json"
LOG_FILE = BASE_DIR / "benchmark.log"

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SWEEP_POINTS = [1024, 2048, 4096, 8192, 16384, 24576, 32768]

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

def generate_configs():
    configs = []
    # Input-fixed sweep
    for input_len in SWEEP_POINTS:
        for output_len in SWEEP_POINTS:
            configs.append((input_len, output_len, "input_fixed"))
    # Output-fixed sweep
    for output_len in SWEEP_POINTS:
        for input_len in SWEEP_POINTS:
            configs.append((input_len, output_len, "output_fixed"))
    return configs

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"completed": [], "start_time": datetime.now().isoformat()}

def save_state(state):
    state["last_update"] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, indent=2, fp=f)

def is_completed(config, state):
    config_key = f"{config[0]}_{config[1]}_{config[2]}"
    return config_key in state["completed"]

def mark_completed(config, state):
    config_key = f"{config[0]}_{config[1]}_{config[2]}"
    if config_key not in state["completed"]:
        state["completed"].append(config_key)
    save_state(state)

def run_benchmark(input_len, output_len, sweep_type):
    result_filename = f"input_{input_len}_output_{output_len}_{sweep_type}.json"
    
    log(f"Running: input={input_len}, output={output_len}, type={sweep_type}")
    
    cmd = [
        "vllm", "bench", "serve",
        "--model", MODEL_NAME,
        "--backend", "openai-chat",
        "--endpoint", "/v1/chat/completions",
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", "50",
        "--max-concurrency", "24",
        "--ignore-eos",
        "--save-result",
        "--result-dir", str(RESULTS_DIR),
        "--result-filename", result_filename
    ]
    
    log(f"Benchmark command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, text=True)
        
        if result.returncode == 0:
            log("✓ Benchmark completed successfully")
            return True
        else:
            log(f"✗ Benchmark failed")
            return False
            
    except Exception as e:
        log(f"✗ Error: {str(e)}")
        return False

def main():
    log("=" * 80)
    log("vLLM Benchmark Runner")
    log("=" * 80)
    log("Make sure vLLM server is running in another terminal!")
    log("=" * 80)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    state = load_state()
    configs = generate_configs()
    
    completed_count = len(state["completed"])
    total_count = len(configs)
    
    if completed_count > 0:
        log(f"RESUMING: {completed_count}/{total_count} already completed")
    
    log(f"Total configurations: {total_count}")
    
    try:
        for idx, (input_len, output_len, sweep_type) in enumerate(configs):
            config_num = idx + 1
            
            if is_completed((input_len, output_len, sweep_type), state):
                log(f"[{config_num}/{total_count}] SKIPPING (already done)")
                continue
            
            log("=" * 80)
            log(f"Configuration {config_num}/{total_count}")
            log(f"Completed: {len(state['completed'])}/{total_count}")
            log("=" * 80)
            
            success = run_benchmark(input_len, output_len, sweep_type)
            
            if success:
                mark_completed((input_len, output_len, sweep_type), state)
                log(f"✓ Progress: {len(state['completed'])}/{total_count}")
            else:
                log("✗ Failed, will retry on next run")
        
        log("=" * 80)
        log("🎉 ALL BENCHMARKS COMPLETED!")
        log("=" * 80)
        
    except KeyboardInterrupt:
        log("Interrupted by user")
    finally:
        completed = len(state["completed"])
        log(f"Final: {completed}/{total_count} completed")

if __name__ == "__main__":
    main()
