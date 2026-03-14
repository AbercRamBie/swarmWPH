#!/usr/bin/env python3
"""
Generate the 216-entry benchmark configuration YAML.

Matrix: 4 algorithms x 2 energy models x 3 predator counts x 3 prey counts x 3 seeds
= 216 total test configurations.

Usage:
    python config/generate_algorithm_benchmarks.py
    # Produces: config/algorithm_benchmark_configs.yaml
"""
import yaml
from pathlib import Path

ALGORITHMS = [
    {"name": "wolf_pack_formation", "short": "wpf"},
    {"name": "strombom", "short": "strom"},
    {"name": "simple_apf", "short": "apf"},
    {"name": "wolf_apf", "short": "wapf"},
]

ENERGY_MODELS = [
    {"name": "stolaroff", "config_file": "config/benchmark_stolaroff.yaml", "short": "M1"},
    {"name": "turtlebot3", "config_file": "config/benchmark_turtlebot3.yaml", "short": "M2"},
]

PREDATOR_COUNTS = [5, 10, 15]
PREY_COUNTS = [10, 20, 30]
SEEDS = [42, 100, 200]

def generate():
    benchmark_runs = []

    for algo in ALGORITHMS:
        for model in ENERGY_MODELS:
            for pred_count in PREDATOR_COUNTS:
                for prey_count in PREY_COUNTS:
                    for seed in SEEDS:
                        name = (
                            f"{algo['short']}_{model['short']}_"
                            f"{pred_count}P_{prey_count}Pr_s{seed}"
                        )
                        entry = {
                            "name": name,
                            "config_file": model["config_file"],
                            "predator_count": pred_count,
                            "prey_count": prey_count,
                            "seed": seed,
                            "algorithm_name": algo["name"],
                            "algorithm_parameters": {},
                        }
                        benchmark_runs.append(entry)

    config = {
        "benchmark_runs": benchmark_runs,
    }

    output_path = Path(__file__).parent / "algorithm_benchmark_configs.yaml"
    with open(output_path, "w") as f:
        f.write("# Algorithm Benchmark Configurations\n")
        f.write(f"# Auto-generated: {len(benchmark_runs)} tests\n")
        f.write(f"# Matrix: {len(ALGORITHMS)} algos x {len(ENERGY_MODELS)} models x "
                f"{len(PREDATOR_COUNTS)} pred x {len(PREY_COUNTS)} prey x {len(SEEDS)} seeds\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return len(benchmark_runs)

if __name__ == "__main__":
    generate()
