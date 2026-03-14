"""
prey.py — Prey agent logic for swarm herding simulation.

Prey agents use flocking behavior (cohesion, alignment, separation)
combined with predator avoidance to move around the arena. They flee
from nearby predators and try to maintain spacing from other prey.

Bug fixes applied:
    - ✓ Bug 2: separatePrey() comparison fixed (was comparing i to radius instead of self_index)

References:
    - Reynolds flocking model (Boids, 1987)
    - Predator-prey dynamics
"""

import math
import random
from typing import List, Tuple
from src.utils.math_helpers import clamp, limit_vector, distance, distance_squared


class Prey:
    """
    Prey agent that flocks and avoids predators.

    Attributes:
        position: [x, y] coordinates in arena
        velocity: [vx, vy] velocity vector
        delivered: Whether this prey has reached the goal zone
        radius: Visual radius (pixels)
    """

    def __init__(
        self,
        position: List[float],
        config: dict,
        prey_id: int,
        rng: random.Random,
    ):
        """
        Initialize a prey agent.

        Args:
            position: Initial [x, y] position
            config: Prey configuration dict from YAML
            prey_id: Unique identifier
            rng: Seeded random number generator
        """
        self.prey_id = prey_id
        self.position = position.copy()
        self.rng = rng

        # Initialize random velocity
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(0, config["speed_max"])
        self.velocity = [speed * math.cos(angle), speed * math.sin(angle)]

        # Physical parameters
        self.radius = config["radius"]
        self.speed_max = config["speed_max"]

        # Behavior parameters
        self.avoid_predator_radius = config["avoid_predator_radius"]
        self.avoid_predator_strength = config["avoid_predator_strength"]
        self.separation_radius = config["separation_radius"]
        self.separation_strength = config["separation_strength"]
        self.boundary_margin = config["boundary_margin"]
        self.boundary_push_strength = config["boundary_push_strength"]
        self.predator_collision_buffer = config["predator_collision_buffer"]
        self.wander_jitter = config["wander_jitter"]

        # State
        self.delivered = False

    def update(
        self,
        prey_list: List['Prey'],
        predator_positions: List[List[float]],
        arena_bounds: Tuple[float, float],
    ):
        """
        Update prey state for one simulation step.

        Combines multiple behavioral forces:
            - Boundary repulsion
            - Predator avoidance
            - Prey separation, alignment, cohesion
            - Random wander

        Args:
            prey_list: List of all prey agents
            predator_positions: List of all predator positions
            arena_bounds: (width, height) of arena
        """
        if self.delivered:
            return  # Already delivered, don't move

        # Compute behavioral forces
        boundary_force = self._avoid_boundaries(arena_bounds)
        predator_force = self._avoid_predators(predator_positions)
        separation_force = self._separate_prey(prey_list)
        alignment_force = self._align_prey(prey_list)
        cohesion_force = self._cohere_prey(prey_list)
        wander_force = self._wander()

        # Combine forces with weights
        total_force = [0.0, 0.0]
        total_force[0] += boundary_force[0] * self.boundary_push_strength
        total_force[1] += boundary_force[1] * self.boundary_push_strength

        total_force[0] += predator_force[0] * self.avoid_predator_strength
        total_force[1] += predator_force[1] * self.avoid_predator_strength

        total_force[0] += separation_force[0] * self.separation_strength
        total_force[1] += separation_force[1] * self.separation_strength

        total_force[0] += alignment_force[0] * 0.5
        total_force[1] += alignment_force[1] * 0.5

        total_force[0] += cohesion_force[0] * 0.3
        total_force[1] += cohesion_force[1] * 0.3

        total_force[0] += wander_force[0] * 0.2
        total_force[1] += wander_force[1] * 0.2

        # Apply force to velocity
        self.velocity[0] += total_force[0]
        self.velocity[1] += total_force[1]

        # Limit speed
        self.velocity = list(limit_vector(
            self.velocity[0],
            self.velocity[1],
            self.speed_max
        ))

        # Update position
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

        # Hard boundary clamp (safety)
        width, height = arena_bounds
        self.position[0] = clamp(self.position[0], 0, width)
        self.position[1] = clamp(self.position[1], 0, height)

        # Resolve hard collisions with predators
        self._resolve_predator_collisions(predator_positions)

    def _avoid_boundaries(self, arena_bounds: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute repulsion force from arena edges.

        Args:
            arena_bounds: (width, height) of arena

        Returns:
            (fx, fy) force vector pointing away from edges
        """
        width, height = arena_bounds
        fx, fy = 0.0, 0.0

        # Left wall
        if self.position[0] < self.boundary_margin:
            fx += (self.boundary_margin - self.position[0]) / self.boundary_margin

        # Right wall
        if self.position[0] > width - self.boundary_margin:
            fx -= (self.position[0] - (width - self.boundary_margin)) / self.boundary_margin

        # Top wall
        if self.position[1] < self.boundary_margin:
            fy += (self.boundary_margin - self.position[1]) / self.boundary_margin

        # Bottom wall
        if self.position[1] > height - self.boundary_margin:
            fy -= (self.position[1] - (height - self.boundary_margin)) / self.boundary_margin

        return fx, fy

    def _avoid_predators(self, predator_positions: List[List[float]]) -> Tuple[float, float]:
        """
        Compute repulsion force from nearby predators.

        Args:
            predator_positions: List of all predator positions

        Returns:
            (fx, fy) force vector pointing away from predators
        """
        fx, fy = 0.0, 0.0
        count = 0

        for pred_pos in predator_positions:
            dist_sq = distance_squared(self.position, pred_pos)
            if dist_sq < self.avoid_predator_radius ** 2:
                dist = math.sqrt(dist_sq)
                if dist > 1e-6:
                    # Flee away from predator
                    # Force strength inversely proportional to distance
                    strength = (self.avoid_predator_radius - dist) / self.avoid_predator_radius
                    fx += (self.position[0] - pred_pos[0]) / dist * strength
                    fy += (self.position[1] - pred_pos[1]) / dist * strength
                    count += 1

        if count > 0:
            fx /= count
            fy /= count

        return fx, fy

    def _separate_prey(self, prey_list: List['Prey']) -> Tuple[float, float]:
        """
        Compute separation force from nearby prey (avoid crowding).

        Bug fix #2: Compare i to self_index instead of radius!

        Args:
            prey_list: List of all prey agents

        Returns:
            (fx, fy) force vector pointing away from neighbors
        """
        fx, fy = 0.0, 0.0
        count = 0

        self_index = -1
        for i, prey in enumerate(prey_list):
            if prey.prey_id == self.prey_id:
                self_index = i
                break

        for i, prey in enumerate(prey_list):
            # Bug fix #2: Was "if i == self.radius:", now correctly:
            if i == self_index:
                continue

            dist_sq = distance_squared(self.position, prey.position)
            if dist_sq < self.separation_radius ** 2:
                dist = math.sqrt(dist_sq)
                if dist > 1e-6:
                    fx += (self.position[0] - prey.position[0]) / dist
                    fy += (self.position[1] - prey.position[1]) / dist
                    count += 1

        if count > 0:
            fx /= count
            fy /= count

        return fx, fy

    def _align_prey(self, prey_list: List['Prey']) -> Tuple[float, float]:
        """
        Compute alignment force (steer toward average heading of neighbors).

        Args:
            prey_list: List of all prey agents

        Returns:
            (fx, fy) force vector aligned with neighbors
        """
        avg_vx, avg_vy = 0.0, 0.0
        count = 0

        for prey in prey_list:
            if prey.prey_id == self.prey_id:
                continue

            dist_sq = distance_squared(self.position, prey.position)
            if dist_sq < self.separation_radius ** 2:
                avg_vx += prey.velocity[0]
                avg_vy += prey.velocity[1]
                count += 1

        if count > 0:
            avg_vx /= count
            avg_vy /= count
            # Difference between average velocity and own velocity
            return avg_vx - self.velocity[0], avg_vy - self.velocity[1]

        return 0.0, 0.0

    def _cohere_prey(self, prey_list: List['Prey']) -> Tuple[float, float]:
        """
        Compute cohesion force (steer toward average position of neighbors).

        Args:
            prey_list: List of all prey agents

        Returns:
            (fx, fy) force vector pointing toward group center
        """
        avg_x, avg_y = 0.0, 0.0
        count = 0

        for prey in prey_list:
            if prey.prey_id == self.prey_id:
                continue

            dist_sq = distance_squared(self.position, prey.position)
            if dist_sq < self.separation_radius ** 2:
                avg_x += prey.position[0]
                avg_y += prey.position[1]
                count += 1

        if count > 0:
            avg_x /= count
            avg_y /= count
            # Direction toward center
            dx = avg_x - self.position[0]
            dy = avg_y - self.position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 1e-6:
                return dx / dist, dy / dist

        return 0.0, 0.0

    def _wander(self) -> Tuple[float, float]:
        """
        Add random jitter to movement.

        Returns:
            (fx, fy) random force vector
        """
        fx = (self.rng.random() - 0.5) * self.wander_jitter
        fy = (self.rng.random() - 0.5) * self.wander_jitter
        return fx, fy

    def _resolve_predator_collisions(self, predator_positions: List[List[float]]):
        """
        Hard collision resolution: push prey away from overlapping predators.

        Args:
            predator_positions: List of all predator positions
        """
        for pred_pos in predator_positions:
            dist = distance(self.position, pred_pos)
            min_dist = self.radius + self.predator_collision_buffer

            if dist < min_dist and dist > 1e-6:
                # Push apart
                overlap = min_dist - dist
                dx = self.position[0] - pred_pos[0]
                dy = self.position[1] - pred_pos[1]
                ux = dx / dist
                uy = dy / dist
                self.position[0] += ux * overlap
                self.position[1] += uy * overlap
