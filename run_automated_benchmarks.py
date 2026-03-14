import argparse
import sys
import yaml
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from src.core.simulation import run_simulation
from src.utils.config_loader import load_config

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_benchmark_configs(config_path: str) -> List[Dict]:
    """Load benchmark configurations from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['benchmark_runs']

def run_benchmark_suite(benchmark_configs: List[Dict], output_dir: str, max_frames: int = None):
    """Run all benchmark configurations and save results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    csv_path = run_dir / "results.csv"
    json_path = run_dir / "summary.json"

    latest_link = Path(output_dir) / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)

    results = []
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'benchmark_name', 'algorithm_name', 'energy_model', 'predator_count', 'prey_count', 'seed',
        'delivered_count', 'completion_rate', 'total_energy', 'energy_per_delivery',
        'frames', 'timeout', 'avg_per_predator_cost', 'avg_duty_cycle'
    ])

    for i, bench_config in enumerate(benchmark_configs, 1):
        name = bench_config['name']
        config_file = bench_config['config_file']
        pred_count = bench_config['predator_count']
        prey_count = bench_config['prey_count']
        seed = bench_config['seed']
        algo_name = bench_config.get('algorithm_name', 'wolf_pack_formation')
        algo_params = bench_config.get('algorithm_parameters', {})
        config = load_config(config_file)
        config['predators']['count'] = pred_count
        config['prey']['count'] = prey_count
        config['algorithm']['name'] = algo_name
        config['algorithm']['parameters'] = algo_params
        if max_frames is not None:
            config['simulation']['max_frames'] = max_frames

        try:
            result = run_simulation(config, seed=seed)
            completion_rate = result.delivered_count / prey_count if prey_count > 0 else 0
            energy_per_delivery = result.total_energy_consumed / result.delivered_count if result.delivered_count > 0 else float('inf')
            avg_per_predator = sum(result.per_predator_costs) / len(result.per_predator_costs) if result.per_predator_costs else 0
            energy_model_name = config_file.split('_')[-1].replace('.yaml', '')

            avg_duty_cycle = (sum(result.per_predator_duty_cycle) /
                             len(result.per_predator_duty_cycle)
                             if result.per_predator_duty_cycle else 0.0)

            csv_writer.writerow([
                name, result.algorithm_name, energy_model_name, pred_count, prey_count, seed,
                result.delivered_count, completion_rate, result.total_energy_consumed, energy_per_delivery,
                result.frames, result.timeout, avg_per_predator, avg_duty_cycle
            ])
            csv_file.flush()

            results.append({
                'benchmark_name': name,
                'algorithm_name': result.algorithm_name,
                'energy_model': energy_model_name,
                'predator_count': pred_count,
                'prey_count': prey_count,
                'seed': seed,
                'delivered_count': result.delivered_count,
                'completion_rate': completion_rate,
                'total_energy_consumed': result.total_energy_consumed,
                'energy_per_delivery': energy_per_delivery,
                'frames': result.frames,
                'timeout': result.timeout,
                'avg_per_predator_cost': avg_per_predator,
                'avg_duty_cycle': avg_duty_cycle,
            })

        except Exception as e:
            results.append({'benchmark_name': name, 'error': str(e)})
    csv_file.close()

    summary = {
        'timestamp': timestamp,
        'run_directory': str(run_dir),
        'total_benchmarks': len(benchmark_configs),
        'successful': sum(1 for r in results if 'error' not in r),
        'failed': sum(1 for r in results if 'error' in r),
        'results': results,
    }

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    try:
        import subprocess
        result = subprocess.run([
            sys.executable,
            "analyze_and_plot_benchmarks.py",
            str(csv_path),
            "--output", str(plots_dir)
        ], capture_output=True, text=True, cwd=str(Path(__file__).parent))

        if result.returncode == 0:
            print("Plots generated in plots/")
        else:
            print(f"Plot generation failed. Run manually: python3 analyze_and_plot_benchmarks.py {csv_path} --output {plots_dir}")
    except Exception as e:
        print(f"Plot generation error: {e}")
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Run automated benchmark suite with varied configurations"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/automated_benchmark_configs.yaml",
        help="Path to benchmark configurations YAML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/automated_benchmarks",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Override maximum simulation frames (default: use config value)"
    )
    args = parser.parse_args()
    benchmark_configs = load_benchmark_configs(args.config)
    run_benchmark_suite(benchmark_configs, args.output, args.max_frames)
    return 0

if __name__ == "__main__":
    sys.exit(main())
