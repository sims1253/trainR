#!/usr/bin/env python3
"""Mini benchmark to test all models on a single task."""
import subprocess

import yaml

# Load models
with open('configs/llm.yaml') as f:
    config = yaml.safe_load(f)

models = list(config.get('models', {}).keys())
task = "tasks/mined/tidyverse_readr_1615.json"

print(f"Testing {len(models)} models on {task}")
print("=" * 60)

results = []
for model in models:
    print(f"\n[{model}]")
    cmd = [
        "uv", "run", "python", "scripts/run_benchmark.py",
        "--task", task,
        "--worker-model", model
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        # Parse output for key metrics
        output = result.stdout + result.stderr
        lines = output.split('\n')
        for line in lines:
            if 'Input Tokens:' in line or 'Output Tokens:' in line or 'Pass Rate:' in line or 'Latency:' in line:
                print(f"  {line.strip()}")
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[:200] if result.stderr else 'unknown'}")
    except subprocess.TimeoutExpired:
        print("  TIMEOUT after 600s")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 60)
print("Mini benchmark complete!")
