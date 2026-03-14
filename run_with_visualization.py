import argparse
import sys
from pathlib import Path
from src.utils.config_loader import load_config
from src.rendering import PygameRenderer
from src.core.simulation import initialize_simulation_components
from src.core.states import PredatorMode
import random

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_visual_simulation(config: dict, seed: int = None, fps: int = 60):
    """
    Run simulation with pygame visualization.
    Args:
        config: Configuration dictionary
        seed: Random seed for reproducibility
        fps: Target frames per second for visualization
    """
    config['simulation']['headless'] = False

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    predators, prey_list, charging_stations, energy_model, goal_center, algorithm = \
        initialize_simulation_components(config, rng)

    sim_config = config['simulation']
    max_frames = sim_config['max_frames']
    arena_bounds = (sim_config['arena_width'], sim_config['arena_height'])
    goal_size = sim_config['goal_zone_size']
    goal_radius = goal_size / 2
    comm_radius = config['communication'].get('comm_radius', 250)
    conflict_rounds = config['communication'].get('conflict_rounds', 3)
    renderer = PygameRenderer(
        screen_width=arena_bounds[0],
        screen_height=arena_bounds[1],
        fps=fps,
        show_stats=True,
    )
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
            msg_count = algorithm.assign_targets(
                predators,
                prey_list,
                comm_radius,
                conflict_rounds,
            )
            for predator in predators:
                if msg_count > 0:
                    comm_cost = predator.energy_model.compute_communication_cost(
                        msg_count // len(predators), 
                        1/60,
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

            for prey in prey_list:
                prey.update(
                    prey_list=prey_list,
                    predator_positions=predator_positions,
                    arena_bounds=arena_bounds,
                )

            from src.utils.math_helpers import inside_rectangle
            goal_size = config['simulation']['goal_zone_size']
            goal_margin = config['simulation']['goal_zone_margin']
            goal_x = arena_bounds[0] - goal_margin - goal_size
            goal_y = goal_margin

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
            if frame % 1000 == 0:
                active_count = sum(1 for p in predators if p.mode == PredatorMode.PURSUE)
                print(f"Frame {frame}: {delivered_count}/{len(prey_list)} delivered, "
                      f"{active_count}/{len(predators)} predators active, "
                      f"energy: {total_energy:.1f}")
        if delivered_count > 0:
            avg_cost_per_delivery = total_energy / delivered_count
        completion_rate = delivered_count / len(prey_list)

    finally:
        renderer.close()

def main():
    parser = argparse.ArgumentParser(
        description="Run energy-aware swarm simulation with visualization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target frames per second",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["wolf_pack_formation", "strombom", "simple_apf", "wolf_apf"],
        help="Override herding algorithm (default: use config value)",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    if args.algorithm:
        config['algorithm']['name'] = args.algorithm
        config['algorithm']['parameters'] = {}
    run_visual_simulation(config, seed=args.seed, fps=args.fps)
    return 0

if __name__ == "__main__":
    sys.exit(main())
