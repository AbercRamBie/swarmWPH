"""
simple_apf.py — Simple Artificial Potential Field (APF) herding.

A baseline algorithm with no inter-agent coordination.
Each predator independently selects the nearest undelivered prey
and navigates using attractive/repulsive potential fields.

Strategy:
    - Assignment: Greedy nearest prey (no communication, no conflict resolution)
    - Pursuit: Attraction toward assigned prey position +
      Repulsion from other predators (collision avoidance)

This represents a naive approach where predators act independently,
which typically leads to inefficient prey assignment overlap
and lack of coordinated herding behavior.

References:
    - Khatib (1986), "Real-time obstacle avoidance for manipulators
      and mobile robots", International Journal of Robotics Research
"""

import math
from typing import List, Optional, Dict, Any
from src.algorithms.base_algorithm import BaseHerdingAlgorithm
from src.utils.math_helpers import distance


class SimpleAPF(BaseHerdingAlgorithm):
    """
    Simple APF (Artificial Potential Field) herding algorithm.

    Parameters:
        k_attract: Attraction gain toward prey (default: 1.0)
        k_repel: Repulsion gain from other predators (default: 0.5)
        repel_radius: Distance within which repulsion activates (default: 50)
    """

    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.k_attract = parameters.get("k_attract", 1.0)
        self.k_repel = parameters.get("k_repel", 0.5)
        self.repel_radius = parameters.get("repel_radius", 50)

    def assign_targets(
        self,
        predators: List,
        prey_list: List,
        comm_radius: float,
        conflict_rounds: int,
    ) -> int:
        """
        Greedy nearest-prey assignment with no communication.

        Each predator independently picks the closest undelivered prey.
        No conflict resolution — multiple predators may target the same prey.

        Returns 0 messages (no communication).
        """
        for pred in predators:
            if pred.disengaged or pred.energy_remaining <= 0:
                pred.assigned_prey_index = None
                continue

            closest_idx = None
            closest_dist = float('inf')

            for i, prey in enumerate(prey_list):
                if prey.delivered:
                    continue
                dist = distance(pred.position, prey.position)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i

            pred.assigned_prey_index = closest_idx
            pred.formation_slot_index = 0

        return 0  # No communication

    def compute_pursue_target(
        self,
        predator,
        prey_list: List,
        all_predators: List,
        goal_center: List[float],
        arena_bounds: tuple,
    ) -> Optional[List[float]]:
        """
        Compute APF-based target position.

        Target = prey_position + repulsion_offset_from_other_predators

        The predator is attracted to a point behind the prey (relative to goal)
        and repelled from nearby predators.
        """
        if predator.assigned_prey_index is None:
            return None
        if predator.assigned_prey_index >= len(prey_list):
            return None

        prey = prey_list[predator.assigned_prey_index]
        if prey.delivered:
            return None

        # Attractive target: position behind prey relative to goal
        # (similar to formation but without slot offsets)
        dx_goal = goal_center[0] - prey.position[0]
        dy_goal = goal_center[1] - prey.position[1]
        dist_to_goal = math.sqrt(dx_goal * dx_goal + dy_goal * dy_goal)

        if dist_to_goal < 1e-6:
            attract_x = prey.position[0]
            attract_y = prey.position[1]
        else:
            ux = dx_goal / dist_to_goal
            uy = dy_goal / dist_to_goal
            # Position behind prey (opposite to goal direction)
            attract_x = prey.position[0] - ux * predator.herd_distance * self.k_attract
            attract_y = prey.position[1] - uy * predator.herd_distance * self.k_attract

        # Repulsive force from other predators
        repel_x, repel_y = 0.0, 0.0
        for other in all_predators:
            if other.predator_id == predator.predator_id:
                continue
            dist = distance(predator.position, other.position)
            if dist < self.repel_radius and dist > 1e-6:
                # Repulsion strength inversely proportional to distance
                strength = self.k_repel * (self.repel_radius - dist) / self.repel_radius
                repel_x += (predator.position[0] - other.position[0]) / dist * strength * predator.herd_distance
                repel_y += (predator.position[1] - other.position[1]) / dist * strength * predator.herd_distance

        # Combined target
        target_x = attract_x + repel_x
        target_y = attract_y + repel_y

        # Clamp to arena bounds
        width, height = arena_bounds
        target_x = max(0, min(width, target_x))
        target_y = max(0, min(height, target_y))

        return [target_x, target_y]

    def get_algorithm_name(self) -> str:
        return "Simple APF"

    def get_algorithm_parameters(self) -> Dict[str, Any]:
        return {
            "type": "simple_apf",
            "k_attract": self.k_attract,
            "k_repel": self.k_repel,
            "repel_radius": self.repel_radius,
        }
