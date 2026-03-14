"""
predator.py — Predator agent logic for swarm herding.

This module implements the predator agents that collaborate to herd prey
into a goal zone. Each predator uses a decentralized assignment algorithm
to select prey targets and resolves conflicts with neighbors.

Key features:
    - Energy-aware behavior (switches modes based on remaining energy)
    - Decentralized prey assignment with conflict resolution
    - Formation-based herding (positions behind prey relative to goal)
    - Soft collision avoidance with other predators
    - Energy model injection for swappable energy accounting

All bugs from the original code have been fixed:
    - ✓ Bug 1: Double energy deduction removed
    - ✓ Bug 3: Speed overwrite fixed
    - ✓ Bug 4: Debug print removed
    - ✓ Bug 6: disengaged attribute initialized

References:
    - Decentralized assignment: Claim strength based on energy + distance
    - Formation control: Position predators behind prey toward goal
"""

import math
import random
from typing import List, Optional, Tuple
from src.core.states import PredatorMode
from src.energy.base_energy_model import BaseEnergyModel
from src.utils.math_helpers import clamp, limit_vector, wrap_angle, distance, distance_squared


class Predator:
    """
    Predator agent that herds prey into a goal zone.

    Attributes:
        position: [x, y] coordinates in arena
        heading_radians: Current orientation in radians
        current_speed: Current forward speed (pixels/frame)
        energy_remaining: Current energy level
        energy_capacity: Maximum energy storage
        assigned_prey_index: Index of currently assigned prey (None if no assignment)
        formation_slot_index: Position in formation around assigned prey
        mode: Current behavioral mode (IDLE, SEARCH, or PURSUE)
        disengaged: Whether agent has run out of energy
    """

    def __init__(
        self,
        position: List[float],
        config: dict,
        energy_model: BaseEnergyModel,
        predator_id: int,
        rng: random.Random,
        algorithm=None,
    ):
        """
        Initialize a predator agent.

        Args:
            position: Initial [x, y] position
            config: Predator configuration dict from YAML
            energy_model: Energy model instance for cost calculations
            predator_id: Unique identifier
            rng: Seeded random number generator
            algorithm: Optional herding algorithm (BaseHerdingAlgorithm)
        """
        self.predator_id = predator_id
        self.position = position.copy()
        self.heading_radians = rng.uniform(0, 2 * math.pi)
        self.current_speed = 0.0
        self.rng = rng

        # Energy parameters
        self.energy_model = energy_model
        self.energy_capacity = config["energy_capacity"]
        self.energy_remaining = config["energy_initial"]
        self.energy_regen_rate = config["energy_regen_rate"]
        self.engage_minimum_energy = config["engage_minimum_energy"]

        # Physical parameters
        self.radius = config["radius"]
        self.speed_max = config["speed_max"]
        self.friction = config["friction"]
        self.turn_rate_max = config["turn_rate_max"]
        self.turn_rate_min = config["turn_rate_min"]

        # Behavior parameters
        self.reorient_probability = config["reorient_probability"]
        self.herd_distance = config["herd_distance"]
        self.arrive_radius = config["arrive_radius"]
        self.separation_radius = config["separation_radius"]
        self.lane_offset = config["lane_offset"]

        # Algorithm injection (strategy pattern)
        self.algorithm = algorithm

        # Assignment state
        self.assigned_prey_index: Optional[int] = None
        self.formation_slot_index: int = 0

        # Mode tracking
        self.mode = PredatorMode.SEARCH
        self.disengaged = False  # Bug fix #6: Initialize disengaged attribute

        # Metrics
        self.total_energy_consumed = 0.0
        self.frames_in_pursue = 0
        self.frames_in_search = 0
        self.frames_idle = 0

    def update(
        self,
        prey_list: List,
        predator_positions: List[List[float]],
        goal_center: List[float],
        charging_stations: List,
        arena_bounds: Tuple[float, float],
        delta_time: float,
        all_predators: Optional[List] = None,
    ) -> float:
        """
        Update predator state for one simulation step.

        Args:
            prey_list: List of all prey agents
            predator_positions: List of all predator positions
            goal_center: [x, y] coordinates of goal zone center
            charging_stations: List of ChargingStation objects
            arena_bounds: (width, height) of arena
            delta_time: Time step duration in seconds
            all_predators: Optional list of all Predator agents (for algorithm use)

        Returns:
            Energy consumed this frame
        """
        # Determine mode based on energy level
        if self.energy_remaining <= 0:
            self.mode = PredatorMode.IDLE
            self.disengaged = True
            self.current_speed = 0.0
            idle_cost = self.energy_model.compute_idle_cost(delta_time)
            self.frames_idle += 1
            self.total_energy_consumed += idle_cost
            return idle_cost

        elif self.energy_remaining < self.engage_minimum_energy:
            self.mode = PredatorMode.SEARCH
        elif self.assigned_prey_index is not None:
            self.mode = PredatorMode.PURSUE
        else:
            self.mode = PredatorMode.SEARCH

        # Execute behavior based on mode
        if self.mode == PredatorMode.SEARCH:
            self._search_behavior(predator_positions, arena_bounds)
            self.frames_in_search += 1
        elif self.mode == PredatorMode.PURSUE:
            self._pursue_behavior(prey_list, predator_positions, goal_center, arena_bounds, all_predators)
            self.frames_in_pursue += 1

        # Apply friction
        self.current_speed *= self.friction

        # Move predator
        self.position[0] += self.current_speed * math.cos(self.heading_radians)
        self.position[1] += self.current_speed * math.sin(self.heading_radians)

        # Bounce off boundaries
        self._handle_boundaries(arena_bounds)

        # Compute energy cost (Bug fix #1: Only deduct once!)
        turn_rate = 0.0  # We'll track actual turn rate if needed
        motion_cost = self.energy_model.compute_motion_cost(
            speed=self.current_speed,
            turn_rate=turn_rate,
            mode=str(self.mode),
            delta_time=delta_time,
        )

        # Apply energy cost
        self.energy_remaining = max(0, self.energy_remaining - motion_cost)
        self.total_energy_consumed += motion_cost

        # Check for charging
        charging_gain = self._check_charging(charging_stations, delta_time)
        self.energy_remaining = min(
            self.energy_capacity,
            self.energy_remaining + charging_gain
        )

        # Passive regeneration (if model supports it)
        if self.energy_regen_rate > 0:
            regen = self.energy_regen_rate * delta_time
            self.energy_remaining = min(
                self.energy_capacity,
                self.energy_remaining + regen
            )

        return motion_cost

    def _search_behavior(
        self,
        predator_positions: List[List[float]],
        arena_bounds: Tuple[float, float],
    ):
        """
        Low-energy wandering behavior.

        Predator moves slowly, occasionally reorienting randomly,
        and avoids collisions with other predators.
        """
        # Random reorientation
        if self.rng.random() < self.reorient_probability:
            self.heading_radians = self.rng.uniform(0, 2 * math.pi)

        # Separation from other predators
        separation_force = self._compute_separation(predator_positions)
        if separation_force[0] != 0 or separation_force[1] != 0:
            target_heading = math.atan2(separation_force[1], separation_force[0])
            heading_error = wrap_angle(target_heading - self.heading_radians)
            turn_rate = clamp(heading_error, -self.turn_rate_min, self.turn_rate_min)
            self.heading_radians = wrap_angle(self.heading_radians + turn_rate)

        # Move at low speed
        self.current_speed = self.speed_max * 0.3

    def _pursue_behavior(
        self,
        prey_list: List,
        predator_positions: List[List[float]],
        goal_center: List[float],
        arena_bounds: Tuple[float, float],
        all_predators: Optional[List] = None,
    ):
        """
        Active pursuit behavior.

        If an algorithm is injected, delegates target computation to it.
        Otherwise uses built-in formation positioning (original behavior).
        """
        if self.assigned_prey_index is None or self.assigned_prey_index >= len(prey_list):
            self.current_speed = 0.0
            return

        prey = prey_list[self.assigned_prey_index]
        if prey.delivered:
            self.assigned_prey_index = None
            self.current_speed = 0.0
            return

        # Compute target position via algorithm or built-in formation
        if self.algorithm is not None:
            target_position = self.algorithm.compute_pursue_target(
                self, prey_list, all_predators or [], goal_center, arena_bounds
            )
            if target_position is None:
                self.current_speed = 0.0
                return
        else:
            target_position = self._compute_formation_position(
                prey.position, goal_center
            )

        # Move toward target position
        self._step_towards_point(target_position, predator_positions)

    def _compute_formation_position(
        self,
        prey_position: List[float],
        goal_center: List[float],
    ) -> List[float]:
        """
        Compute formation position behind prey relative to goal.

        Formation positions are arranged in a line behind the prey,
        offset perpendicular to the prey-goal direction.

        Args:
            prey_position: [x, y] of assigned prey
            goal_center: [x, y] of goal zone center

        Returns:
            Target [x, y] position for this predator
        """
        # Direction from prey to goal
        dx = goal_center[0] - prey_position[0]
        dy = goal_center[1] - prey_position[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            # Prey is at goal
            return prey_position.copy()

        # Normalize
        ux = dx / dist
        uy = dy / dist

        # Position behind prey (opposite direction to goal)
        formation_x = prey_position[0] - ux * self.herd_distance
        formation_y = prey_position[1] - uy * self.herd_distance

        # Offset perpendicular based on slot index
        # Perpendicular vector: rotate 90 degrees
        perp_x = -uy
        perp_y = ux

        # Alternate left/right based on slot index
        lateral_offset = (self.formation_slot_index % 2 * 2 - 1) * self.lane_offset
        lateral_offset *= (self.formation_slot_index // 2 + 1)

        formation_x += perp_x * lateral_offset
        formation_y += perp_y * lateral_offset

        return [formation_x, formation_y]

    def _step_towards_point(
        self,
        target: List[float],
        predator_positions: List[List[float]],
    ):
        """
        Move toward a target point with arrival and collision avoidance.

        Bug fix #3: Use the computed arrive_speed instead of hardcoded 2.0

        Args:
            target: [x, y] target position
            predator_positions: Positions of all predators for collision avoidance
        """
        # Direction to target
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            self.current_speed = 0.0
            return

        # Compute desired heading
        target_heading = math.atan2(dy, dx)

        # Add separation force
        separation_force = self._compute_separation(predator_positions)
        if separation_force[0] != 0 or separation_force[1] != 0:
            sep_heading = math.atan2(separation_force[1], separation_force[0])
            # Blend target heading with separation
            heading_error = wrap_angle(target_heading - self.heading_radians)
            sep_error = wrap_angle(sep_heading - self.heading_radians)
            combined_error = 0.7 * heading_error + 0.3 * sep_error
            target_heading = wrap_angle(self.heading_radians + combined_error)

        # Turn toward target
        heading_error = wrap_angle(target_heading - self.heading_radians)
        turn_rate = clamp(heading_error, -self.turn_rate_max, self.turn_rate_max)
        self.heading_radians = wrap_angle(self.heading_radians + turn_rate)

        # Compute speed with arrival behavior
        # Bug fix #3: Actually USE the arrive_speed instead of hardcoding
        target_speed = self._arrive_speed(dist, self.speed_max)
        self.current_speed = target_speed  # Fixed! Was: self.current_speed = 2.0

    def _arrive_speed(self, distance_to_target: float, cap_speed: float) -> float:
        """
        Compute speed that slows down near target (arrival behavior).

        Args:
            distance_to_target: Distance to destination
            cap_speed: Maximum allowed speed

        Returns:
            Target speed (smoothly reduced near target)
        """
        if distance_to_target > self.arrive_radius:
            return cap_speed
        else:
            # Slow down proportionally as we approach
            fraction = distance_to_target / self.arrive_radius
            return cap_speed * fraction

    def _compute_separation(
        self,
        predator_positions: List[List[float]],
    ) -> Tuple[float, float]:
        """
        Compute soft repulsion force from nearby predators.

        Args:
            predator_positions: List of all predator positions

        Returns:
            (fx, fy) force vector pointing away from neighbors
        """
        fx, fy = 0.0, 0.0
        count = 0

        for other_pos in predator_positions:
            # Skip self (same position)
            if (abs(other_pos[0] - self.position[0]) < 1e-6 and
                abs(other_pos[1] - self.position[1]) < 1e-6):
                continue

            dist_sq = distance_squared(self.position, other_pos)
            if dist_sq < self.separation_radius ** 2:
                dist = math.sqrt(dist_sq)
                if dist > 1e-6:
                    # Push away
                    fx += (self.position[0] - other_pos[0]) / dist
                    fy += (self.position[1] - other_pos[1]) / dist
                    count += 1

        if count > 0:
            fx /= count
            fy /= count

        return fx, fy

    def _handle_boundaries(self, arena_bounds: Tuple[float, float]):
        """
        Reflect off arena edges.

        Args:
            arena_bounds: (width, height) of arena
        """
        width, height = arena_bounds

        # Left/right walls
        if self.position[0] < 0:
            self.position[0] = 0
            self.heading_radians = math.pi - self.heading_radians
        elif self.position[0] > width:
            self.position[0] = width
            self.heading_radians = math.pi - self.heading_radians

        # Top/bottom walls
        if self.position[1] < 0:
            self.position[1] = 0
            self.heading_radians = -self.heading_radians
        elif self.position[1] > height:
            self.position[1] = height
            self.heading_radians = -self.heading_radians

    def _check_charging(
        self,
        charging_stations: List,
        delta_time: float,
    ) -> float:
        """
        Check if at a charging station and gain energy if so.

        Args:
            charging_stations: List of ChargingStation objects
            delta_time: Time step duration

        Returns:
            Energy gained this frame
        """
        if not charging_stations:
            return 0.0

        for station in charging_stations:
            if station.is_agent_in_range(self.position):
                # Agent is at station
                return self.energy_model.compute_charging_gain(
                    is_at_station=True,
                    charge_rate=station.charge_rate,
                    delta_time=delta_time,
                )

        return 0.0

    def get_claim_strength(self, prey_position: List[float]) -> float:
        """
        Compute claim strength for prey assignment conflicts.

        Claim strength = 0.7 * energy_remaining - 0.3 * distance_to_prey

        Higher claim strength means this predator has priority.

        Args:
            prey_position: [x, y] of prey under consideration

        Returns:
            Claim strength value
        """
        dist = distance(self.position, prey_position)
        energy_normalized = self.energy_remaining / self.energy_capacity
        return 0.7 * energy_normalized - 0.3 * (dist / 1000.0)
