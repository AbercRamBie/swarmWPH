import argparse
import sys
import yaml
import json
import csv
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import random
from src.utils.config_loader import load_config
from src.rendering import PygameRenderer
from src.core.simulation import initialize_simulation_components
from src.core.states import PredatorMode
from src.utils.math_helpers import inside_rectangle

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def update_latest_reference(output_dir: Path, run_dir: Path) -> None:
    """Update output_dir/latest to point at run_dir, with Windows-safe fallbacks."""
    latest_path = output_dir / "latest"

    if latest_path.exists() or latest_path.is_symlink():
        if latest_path.is_symlink() or latest_path.is_file():
            latest_path.unlink()
        else:
            shutil.rmtree(latest_path)
    try:
        latest_path.symlink_to(run_dir.name, target_is_directory=True)
        return
    except OSError as exc:
        if os.name != 'nt' or getattr(exc, 'winerror', None) != 1314:
            raise
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(latest_path), str(run_dir.resolve())],
            check=True,
            capture_output=True,
            text=True,
        )
        return
    except Exception:
        shutil.copytree(run_dir, latest_path)

def load_benchmark_configs(config_path: str) -> List[Dict]:
    """Load benchmark configurations from YAML file."""

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['benchmark_runs']

def run_single_visual_test(
    name: str,
    config: dict,
    seed: int,
    fps: int = 60,
    max_frames: int = 15000,
) -> Dict:
    """Run a single benchmark test with visualization."""
    config['simulation']['headless'] = False
    config['simulation']['max_frames'] = max_frames
    rng = random.Random(seed)

    predators, prey_list, charging_stations, energy_model, goal_center, algorithm = \
        initialize_simulation_components(config, rng)

    sim_config = config['simulation']
    arena_bounds = (sim_config['arena_width'], sim_config['arena_height'])
    goal_size = sim_config['goal_zone_size']
    goal_margin = sim_config['goal_zone_margin']
    goal_radius = goal_size / 2
    goal_x = arena_bounds[0] - goal_margin - goal_size
    goal_y = goal_margin
    comm_radius = config['communication'].get('comm_radius', 250)
    conflict_rounds = config['communication'].get('conflict_rounds', 3)

    renderer = PygameRenderer(
        screen_width=arena_bounds[0],
        screen_height=arena_bounds[1],
        fps=fps,
        show_stats=True,
    )

    energy_model_name = energy_model.get_model_name()
    renderer.set_title(energy_model_name, name)

    frame = 0
    delivered_count = 0
    total_energy = 0.0
    running = True

    try:
        while running and frame < max_frames:
            if not renderer.handle_events():
                break

            delivered_count = sum(1 for prey in prey_list if prey.delivered)
            if delivered_count == len(prey_list):
                for _ in range(fps * 2):
                    if not renderer.handle_events():
                        break
                    renderer.render_frame(
                        predators,
                        prey_list,
                        (goal_center[0], goal_center[1], goal_radius),
                        charging_stations,
                        frame,
                        delivered_count,
                        total_energy,
                    )
                break

            msg_count = algorithm.assign_targets(predators, prey_list, comm_radius, conflict_rounds)

            for predator in predators:
                if msg_count > 0:
                    comm_cost = predator.energy_model.compute_communication_cost(msg_count // len(predators), 1/60)
                    predator.energy_remaining -= comm_cost
                    predator.total_energy_consumed += comm_cost

            predator_positions = [p.position for p in predators]
            for predator in predators:
                old_energy = predator.energy_remaining
                predator.update(
                    prey_list=prey_list,
                    predator_positions=predator_positions,
                    goal_center=goal_center,
                    charging_stations=charging_stations,
                    arena_bounds=arena_bounds,
                    delta_time=1/60,
                    all_predators=predators,
                )
                energy_spent = old_energy - predator.energy_remaining
                total_energy += energy_spent

            for prey in prey_list:
                prey.update(prey_list=prey_list, predator_positions=predator_positions, arena_bounds=arena_bounds)

            for prey in prey_list:
                if not prey.delivered:
                    if inside_rectangle(prey.position, goal_x, goal_y, goal_size, goal_size):
                        prey.delivered = True

            renderer.render_frame(
                predators,
                prey_list,
                (goal_center[0], goal_center[1], goal_radius),
                charging_stations,
                frame,
                delivered_count,
                total_energy,
            )

            frame += 1

        completion_rate = delivered_count / len(prey_list) if len(prey_list) > 0 else 0
        energy_per_delivery = total_energy / delivered_count if delivered_count > 0 else float('inf')
        avg_per_predator = sum(p.total_energy_consumed for p in predators) / len(predators) if predators else 0

        # Compute duty cycle per predator
        per_predator_duty = []
        for p in predators:
            if frame > 0:
                per_predator_duty.append(p.pursue_frames / frame if hasattr(p, 'pursue_frames') else 1.0)
            else:
                per_predator_duty.append(0.0)
        avg_duty_cycle = sum(per_predator_duty) / len(per_predator_duty) if per_predator_duty else 0.0

        return {
            'benchmark_name': name,
            'algorithm_name': algorithm.get_algorithm_name(),
            'energy_model': energy_model_name,
            'predator_count': len(predators),
            'prey_count': len(prey_list),
            'seed': seed,
            'delivered_count': delivered_count,
            'completion_rate': completion_rate,
            'total_energy_consumed': total_energy,
            'energy_per_delivery': energy_per_delivery,
            'frames': frame,
            'timeout': frame >= max_frames,
            'avg_per_predator_cost': avg_per_predator,
            'avg_duty_cycle': avg_duty_cycle,
        }

    finally:
        renderer.close()


def run_visual_benchmark_suite(
    benchmark_configs: List[Dict],
    output_dir: str,
    fps: int = 60,
    max_frames: int = 15000,
):
    """Run all benchmark configurations with visualization."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    csv_path = run_dir / "results.csv"
    json_path = run_dir / "summary.json"
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

        # Algorithm override from benchmark config
        algo_name = bench_config.get('algorithm_name', 'wolf_pack_formation')
        algo_params = bench_config.get('algorithm_parameters', {})
        config = load_config(config_file)
        config['predators']['count'] = pred_count
        config['prey']['count'] = prey_count
        config['algorithm']['name'] = algo_name
        config['algorithm']['parameters'] = algo_params

        try:
            result = run_single_visual_test(name=name, config=config, seed=seed, fps=fps, max_frames=max_frames)

            csv_writer.writerow([
                result['benchmark_name'],
                result['algorithm_name'],
                result['energy_model'],
                result['predator_count'],
                result['prey_count'],
                result['seed'],
                result['delivered_count'],
                result['completion_rate'],
                result['total_energy_consumed'],
                result['energy_per_delivery'],
                result['frames'],
                result['timeout'],
                result['avg_per_predator_cost'],
                result['avg_duty_cycle'],
            ])
            csv_file.flush()
            results.append(result)

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

    update_latest_reference(Path(output_dir), run_dir)
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Run automated benchmark suite with VISUAL simulation"
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
        default="results/automated_benchmarks_visual",
        help="Output directory for results"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second for visualization (default: 60)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=15000,
        help="Maximum simulation frames per test (default: 15000)"
    )
    args = parser.parse_args()
    # Load benchmark configurations
    benchmark_configs = load_benchmark_configs(args.config)

    # Run visual benchmark suite
    run_visual_benchmark_suite(
        benchmark_configs,
        args.output,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
