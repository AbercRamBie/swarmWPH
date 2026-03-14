#!/usr/bin/env python3
"""
run_visual_benchmark_suite.py — Visual benchmark across models and configurations.

Runs both real-world energy models with different predator-prey configurations,
showing each simulation visually and collecting metrics for comparison.

Usage:
    python run_visual_benchmark_suite.py
    python run_visual_benchmark_suite.py --fps 30 --fast
"""

import argparse
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from src.utils.config_loader import load_config
from src.rendering import PygameRenderer
from src.core.simulation import initialize_simulation_components
from src.core.states import PredatorMode
from src.utils.math_helpers import inside_rectangle
import random

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

TEST_CONFIGS = [
    (5, 10, "Small: 5 predators, 10 prey"),
    (10, 20, "Medium: 10 predators, 20 prey"),
    (15, 30, "Large: 15 predators, 30 prey"),
    (8, 20, "Outnumbered: 8 predators, 20 prey"),
    (12, 15, "Balanced: 12 predators, 15 prey"),
]

# Energy models to test
ENERGY_MODELS = [
    {
        'name': 'M1_Stolaroff',
        'config_file': 'config/benchmark_stolaroff.yaml',
        'description': 'Physics-based Drone'
    },
    {
        'name': 'M2_TurtleBot3',
        'config_file': 'config/benchmark_turtlebot3.yaml',
        'description': 'Empirical Ground Robot'
    },
]

def run_single_visual_test(
    config: dict,
    model_name: str,
    test_name: str,
    seed: int,
    fps: int,
    fast_mode: bool = False,
):
    """
    Run a single visual simulation test.
    
    Returns dict with results or None if user quit.
    """
    rng = random.Random(seed)
    predators, prey_list, charging_stations, energy_model, goal_center, algorithm = \
        initialize_simulation_components(config, rng)

    sim_config = config['simulation']
    max_frames = sim_config['max_frames']
    arena_bounds = (sim_config['arena_width'], sim_config['arena_height'])
    
    goal_size = sim_config['goal_zone_size']
    goal_radius = goal_size / 2
    goal_margin = sim_config['goal_zone_margin']
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

    renderer.set_title(
        energy_model_name=energy_model.get_model_name(),
        config_name=test_name
    )

    frame = 0
    total_energy = 0.0
    pursue_frames = 0  
    capture_times = []  
    running = True
    user_quit = False

    no_progress_frames = 0
    last_delivered_count = 0
    STAGNATION_THRESHOLD = 5000  

    try:
        while running and frame < max_frames:
            
            if not renderer.handle_events():
                user_quit = True
                break
            
            delivered_count = sum(1 for prey in prey_list if prey.delivered)
            if delivered_count == len(prey_list):
                # Show completion for a moment
                if not fast_mode:
                    for _ in range(fps):
                        if not renderer.handle_events():
                            user_quit = True
                            break
                        renderer.render_frame(
                            predators, prey_list,
                            (goal_center[0], goal_center[1], goal_radius),
                            charging_stations, frame, delivered_count, total_energy,
                        )
                break

            msg_count = algorithm.assign_targets(
                predators, prey_list, comm_radius, conflict_rounds,
            )

            for predator in predators:
                if msg_count > 0:
                    comm_cost = predator.energy_model.compute_communication_cost(
                        msg_count // len(predators), 1/60,
                    )
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

                if predator.mode == PredatorMode.PURSUE:
                    pursue_frames += 1

            for prey in prey_list:
                prey.update(
                    prey_list=prey_list,
                    predator_positions=predator_positions,
                    arena_bounds=arena_bounds,
                )

            for prey in prey_list:
                if not prey.delivered:
                    if inside_rectangle(prey.position, goal_x, goal_y, goal_size, goal_size):
                        prey.delivered = True
                        capture_times.append(frame)  # Record when this prey was captured

            delivered_count = sum(1 for prey in prey_list if prey.delivered)
            renderer.render_frame(
                predators, prey_list,
                (goal_center[0], goal_center[1], goal_radius),
                charging_stations, frame, delivered_count, total_energy,
            )

            active_count = sum(1 for p in predators if p.mode == PredatorMode.PURSUE)

            if delivered_count == last_delivered_count:
                no_progress_frames += 1
            else:
                no_progress_frames = 0
                last_delivered_count = delivered_count

            frame += 1

    finally:
        renderer.close()

    if user_quit:
        return None

    delivered_count = sum(1 for prey in prey_list if prey.delivered)
    completion_rate = delivered_count / len(prey_list)
    avg_duty_cycle = pursue_frames / (frame * len(predators)) if frame > 0 else 0.0
    avg_cost_per_delivery = total_energy / delivered_count if delivered_count > 0 else None

    avg_time_per_capture = None
    avg_time_per_capture_seconds = None
    if capture_times:        
        if len(capture_times) > 1:
            time_diffs = [capture_times[i] - capture_times[i-1] for i in range(1, len(capture_times))]
            time_diffs.insert(0, capture_times[0]) 
            avg_time_per_capture = sum(time_diffs) / len(time_diffs)
        else:
            avg_time_per_capture = capture_times[0]
        avg_time_per_capture_seconds = avg_time_per_capture / 60.0

    is_timeout = frame >= max_frames
    is_failure = (active_count == 0 and no_progress_frames >= STAGNATION_THRESHOLD)
    is_success = completion_rate == 1.0 and not is_timeout and not is_failure

    result = {
        'model_name': model_name,
        'model_description': ENERGY_MODELS[[m['name'] for m in ENERGY_MODELS].index(model_name)]['description'],
        'test_config': test_name,
        'num_predators': len(predators),
        'num_prey': len(prey_list),
        'seed': seed,
        'frames': frame,
        'delivered': delivered_count,
        'completion_rate': completion_rate,
        'timeout': is_timeout,
        'failure': is_failure,
        'success': is_success,
        'total_energy': total_energy,
        'avg_cost_per_delivery': avg_cost_per_delivery,
        'avg_duty_cycle': avg_duty_cycle,
        'makespan': frame,
        'avg_time_per_capture_frames': avg_time_per_capture,
        'avg_time_per_capture_seconds': avg_time_per_capture_seconds,
    }
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Visual benchmark suite across models and configurations"
    )
    parser.add_argument(
        "--fps", type=int, default=60,
        help="Frames per second for visualization (default: 60, try 30 for faster)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip completion celebration delays",
    )
    parser.add_argument(
        "--output", type=str, default="results/visual_benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--algorithm", type=str, default=None,
        choices=["wolf_pack_formation", "strombom", "simple_apf", "wolf_apf"],
        help="Override herding algorithm (default: use config value)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    test_number = 0
    total_tests = len(ENERGY_MODELS) * len(TEST_CONFIGS)
    for pred_count, prey_count, config_desc in TEST_CONFIGS:
        for model_info in ENERGY_MODELS:
            test_number += 1
            config = load_config(model_info['config_file'])
            config['predators']['count'] = pred_count
            config['prey']['count'] = prey_count
            if args.algorithm:
                config['algorithm']['name'] = args.algorithm
                config['algorithm']['parameters'] = {}
            result = run_single_visual_test(
                config=config,
                model_name=model_info['name'],
                test_name=config_desc,
                seed=args.seed,
                fps=args.fps,
                fast_mode=args.fast,
            )

            if result is not None:
                all_results.append(result)

    if all_results:       
        csv_path = output_dir / f"visual_benchmark_{timestamp}_detailed.csv"
        fieldnames = list(all_results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        summary = {}
        for config_desc in [c[2] for c in TEST_CONFIGS]:
            config_results = [r for r in all_results if r['test_config'] == config_desc]
            if config_results:
                summary[config_desc] = {}
                for model_name in [m['name'] for m in ENERGY_MODELS]:
                    model_results = [r for r in config_results if r['model_name'] == model_name]
                    if model_results:
                        r = model_results[0]  # Single run per config
                        summary[config_desc][model_name] = {
                            'delivered': r['delivered'],
                            'total_prey': r['num_prey'],
                            'completion_rate': r['completion_rate'],
                            'total_energy': r['total_energy'],
                            'duty_cycle': r['avg_duty_cycle'],
                        }

        json_path = output_dir / f"visual_benchmark_{timestamp}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        model_stats = {}
        for model_name in [m['name'] for m in ENERGY_MODELS]:
            model_results = [r for r in all_results if r['model_name'] == model_name]
            if model_results:
                total_tests = len(model_results)
                successes = sum(1 for r in model_results if r.get('success', False))
                failures = sum(1 for r in model_results if r.get('failure', False))
                timeouts = sum(1 for r in model_results if r.get('timeout', False) and not r.get('failure', False))
                model_stats[model_name] = {
                    'total': total_tests,
                    'success': successes,
                    'failure': failures,
                    'timeout': timeouts,
                    'success_rate': successes / total_tests if total_tests > 0 else 0,
                    'failure_rate': failures / total_tests if total_tests > 0 else 0,
                }

        for model_name, stats in model_stats.items():
            config_results = [r for r in all_results if r['test_config'] == config_desc]
            for r in config_results:
                time_per_cap = r.get('avg_time_per_capture_seconds', 0)
                time_str = f"{time_per_cap:.1f}s" if time_per_cap else "N/A"

                if r.get('success', False):
                    status = "OK"
                elif r.get('failure', False):
                    status = "FAIL"
                elif r.get('timeout', False):
                    status = "TIME"
                else:
                    status = "?"

    return 0
if __name__ == "__main__":
    sys.exit(main())
