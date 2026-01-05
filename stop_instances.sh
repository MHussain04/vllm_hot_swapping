#!/bin/bash

echo "Stopping all vLLM instances..."

if [ -f instance_pids.txt ]; then
    while read pid; do
        if kill -0 $pid 2>/dev/null; then
            echo "  Killing PID $pid..."
            kill $pid
        fi
    done < instance_pids.txt

    echo "Waiting for processes to terminate..."
    sleep 3

    # Force kill if still running
    while read pid; do
        if kill -0 $pid 2>/dev/null; then
            echo "  Force killing PID $pid..."
            kill -9 $pid
        fi
    done < instance_pids.txt

    rm instance_pids.txt
    echo "✓ All instances stopped"
else
    echo "No instance_pids.txt found. Trying to kill by port..."
    for port in {8002..8013}; do
        pid=$(lsof -ti:$port)
        if [ ! -z "$pid" ]; then
            echo "  Killing process on port $port (PID: $pid)"
            kill $pid 2>/dev/null || kill -9 $pid 2>/dev/null
        fi
    done
    echo "✓ Cleanup complete"
fi
