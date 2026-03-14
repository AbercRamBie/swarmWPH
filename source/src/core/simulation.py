"""
simulation.py — Headless simulation engine for swarm herding.

This module runs the herding simulation without any rendering.
It is the core of the benchmarking pipeline.

Process flow:
    1. Load config and create energy model
    2. Create predators and prey with seeded RNG
    3. Main loop:
        a. Decentralised prey assignment (with comm cost)
        b. Update predators (with energy model cost)
        c. Update prey (flocking + avoidance)
        d. Check deliveries (prey reaching goal zone)
        e. Record metrics for this frame
    4. Return SimulationResult with all collected data

Key features:
    - Fully headless (no pygame imports)
    - Seeded RNG for reproducibility
    - Energy model injection
    - Comprehensive metrics tracking
"""

import random
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from src.core.predator import Predator
from src.core.prey import Prey
from src.core.assignment import assign_prey_to_predators
from src.core.charging_station import create_stations_from_config
from src.energy import create_energy_model
from src.algorithms import create_herding_algorithm
from src.utils.math_helpers import inside_rectangle


logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """
    Complete results from one simulation run.

    Attributes:
        frames: Total frames simulated
        delivered_count: Number of prey successfully delivered
        timeout: Whether max_frames was reached
        total_energy_consumed: Sum of all predator energy consumption
        per_predator_costs: List of total cost per predator
        per_predator_duty_cycle: List of duty cycle per predator
        makespan: Total frames until completion
        avg_cost_per_delivery: Average energy per delivered prey
        config: Configuration used for this run
    """
    frames: int
    delivered_count: int
    timeout: bool
    total_energy_consumed: float
    per_predator_costs: List[float]
    per_predator_duty_cycle: List[float]
    makespan: int
    avg_cost_per_delivery: Optional[float]
    config: dict
    energy_model_name: str
    seed: Optional[int]
    algorithm_name: str = "wolf_pack_formation"


def initialize_simulation_components(config: dict, rng: random.Random):
    """
    Initialize simulation components (predators, prey, stations, energy model).

    This is extracted for reuse by both headless and visual simulations.

    Args:
        config: Full configuration dictionary
        rng: Seeded random number generator

    Returns:
        Tuple of (predators, prey_list, charging_stations, energy_model, goal_center)
    """
    # Extract config sections
    sim_cfg = config["simulation"]
    pred_cfg = config["predators"]
    prey_cfg = config["prey"]
    charging_cfg = config["charging"]

    # Arena setup
    arena_width = sim_cfg["arena_width"]
    arena_height = sim_cfg["arena_height"]

    # Goal zone setup
    goal_size = sim_cfg["goal_zone_size"]
    goal_margin = sim_cfg["goal_zone_margin"]
    goal_x = arena_width - goal_margin - goal_size
    goal_y = goal_margin
    goal_center = [goal_x + goal_size / 2, goal_y + goal_size / 2]

    # Create energy model
    energy_model = create_energy_model(config["energy_model"])
    logger.info(f"Using energy model: {energy_model.get_model_name()}")

    # Create herding algorithm
    algorithm = create_herding_algorithm(config["algorithm"])
    logger.info(f"Using algorithm: {algorithm.get_algorithm_name()}")

    # Create charging stations
    charging_stations = create_stations_from_config(
        charging_cfg, arena_width, arena_height
    )
    logger.info(f"Created {len(charging_stations)} charging stations")

    # Create predators
    predators = []
    spawn_anchor = pred_cfg["spawn_anchor"]
    spawn_spread = pred_cfg["spawn_spread"]

    # Handle negative spawn coordinates (relative to bottom edge)
    spawn_x = spawn_anchor[0]
    spawn_y = spawn_anchor[1]
    if spawn_y < 0:
        spawn_y = arena_height + spawn_y

    for i in range(pred_cfg["count"]):
        # Random position around spawn anchor
        angle = rng.uniform(0, 2 * 3.14159)
        radius = rng.uniform(0, spawn_spread)
        pos_x = spawn_x + radius * rng.choice([-1, 1]) * rng.random()
        pos_y = spawn_y + radius * rng.choice([-1, 1]) * rng.random()

        # Clamp to arena
        pos_x = max(0, min(arena_width, pos_x))
        pos_y = max(0, min(arena_height, pos_y))

        predator = Predator(
            position=[pos_x, pos_y],
            config=pred_cfg,
            energy_model=energy_model,
            predator_id=i,
            rng=rng,
            algorithm=algorithm,
        )
        predators.append(predator)

    logger.info(f"Created {len(predators)} predators")

    # Create prey (spawned in center region)
    prey_list = []
    for i in range(prey_cfg["count"]):
        # Spawn prey in center 50% of arena
        pos_x = rng.uniform(0.25 * arena_width, 0.75 * arena_width)
        pos_y = rng.uniform(0.25 * arena_height, 0.75 * arena_height)

        prey = Prey(
            position=[pos_x, pos_y],
            config=prey_cfg,
            prey_id=i,
            rng=rng,
        )
        prey_list.append(prey)

    logger.info(f"Created {len(prey_list)} prey")

    return predators, prey_list, charging_stations, energy_model, goal_center, algorithm


def run_simulation(config: dict, seed: Optional[int] = None) -> SimulationResult:
    """
    Run one complete herding simulation episode.

    Args:
        config: Full configuration dictionary (from config_loader)
        seed: Random seed for reproducibility (None = use system random)

    Returns:
        SimulationResult with complete episode metrics

    Raises:
        ValueError: If configuration is invalid
    """
    # Setup logging
    if seed is not None:
        logger.info(f"Starting simulation with seed {seed}")
    else:
        logger.info("Starting simulation with random seed")

    # Create seeded RNG
    rng = random.Random(seed)

    # Initialize simulation components
    predators, prey_list, charging_stations, energy_model, goal_center, algorithm = \
        initialize_simulation_components(config, rng)

    # Extract remaining config
    sim_cfg = config["simulation"]
    comm_cfg = config["communication"]

    # Arena and timing
    arena_width = sim_cfg["arena_width"]
    arena_height = sim_cfg["arena_height"]
    arena_bounds = (arena_width, arena_height)
    max_frames = sim_cfg["max_frames"]
    fps = sim_cfg["fps"]
    delta_time = 1.0 / fps

    # Goal zone coordinates
    goal_size = sim_cfg['goal_zone_size']
    goal_margin = sim_cfg['goal_zone_margin']
    goal_x = arena_width - goal_margin - goal_size
    goal_y = goal_margin

    # Metrics tracking
    total_delivered = 0
    frame = 0

    # Main simulation loop
    logger.info("Starting simulation loop")
    for frame in range(1, max_frames + 1):
        # Check termination condition
        active_prey = [p for p in prey_list if not p.delivered]
        if not active_prey:
            logger.info(f"All prey delivered at frame {frame}")
            break

        # Step 1: Prey assignment (via algorithm)
        message_count = algorithm.assign_targets(
            predators,
            prey_list,
            comm_cfg["comm_radius"],
            comm_cfg["conflict_rounds"],
        )

        # Apply communication cost to predators
        for pred in predators:
            if message_count > 0:
                comm_cost = pred.energy_model.compute_communication_cost(
                    message_count // len(predators),  # Distribute messages
                    delta_time,
                )
                pred.energy_remaining -= comm_cost
                pred.total_energy_consumed += comm_cost

        # Step 2: Update predators
        predator_positions = [pred.position for pred in predators]
        for pred in predators:
            pred.update(
                prey_list,
                predator_positions,
                goal_center,
                charging_stations,
                arena_bounds,
                delta_time,
                all_predators=predators,
            )

        # Update positions after movement
        predator_positions = [pred.position for pred in predators]

        # Step 3: Update prey
        for prey in prey_list:
            if not prey.delivered:
                prey.update(prey_list, predator_positions, arena_bounds)

        # Step 4: Check for deliveries
        for prey in prey_list:
            if not prey.delivered:
                if inside_rectangle(
                    prey.position, goal_x, goal_y, goal_size, goal_size
                ):
                    prey.delivered = True
                    total_delivered += 1
                    logger.debug(f"Prey {prey.prey_id} delivered at frame {frame}")

        # Log progress every 1000 frames
        if frame % 1000 == 0:
            active_count = sum(1 for p in predators if not p.disengaged)
            logger.info(
                f"Frame {frame}: {total_delivered}/{len(prey_list)} delivered, "
                f"{active_count}/{len(predators)} predators active"
            )

    # Determine timeout
    timeout = (frame >= max_frames)
    if timeout:
        logger.warning(f"Simulation timeout at frame {max_frames}")

    # Compute final metrics
    total_energy = sum(pred.total_energy_consumed for pred in predators)
    per_predator_costs = [pred.total_energy_consumed for pred in predators]

    # Duty cycle: fraction of time in pursue mode
    per_predator_duty_cycle = []
    for pred in predators:
        total_frames_active = pred.frames_in_pursue + pred.frames_in_search + pred.frames_idle
        if total_frames_active > 0:
            duty_cycle = pred.frames_in_pursue / total_frames_active
        else:
            duty_cycle = 0.0
        per_predator_duty_cycle.append(duty_cycle)

    # Average cost per delivery
    if total_delivered > 0:
        avg_cost_per_delivery = total_energy / total_delivered
    else:
        avg_cost_per_delivery = None

    logger.info(
        f"Simulation complete: {total_delivered}/{len(prey_list)} delivered "
        f"in {frame} frames, total energy: {total_energy:.2f}"
    )

    return SimulationResult(
        frames=frame,
        delivered_count=total_delivered,
        timeout=timeout,
        total_energy_consumed=total_energy,
        per_predator_costs=per_predator_costs,
        per_predator_duty_cycle=per_predator_duty_cycle,
        makespan=frame,
        avg_cost_per_delivery=avg_cost_per_delivery,
        config=config,
        energy_model_name=energy_model.get_model_name(),
        seed=seed,
        algorithm_name=algorithm.get_algorithm_name(),
    )
