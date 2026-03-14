"""
strombom_shepherding.py — Strombom (2014) collect-or-drive shepherding.

Based on: Strombom et al. (2014) "Solving the shepherding problem:
heuristics for herding autonomous, interacting agents"
Journal of the Royal Society Interface, 11(100), 20140719.

Strategy:
    The algorithm switches between two modes each frame based on
    the spatial distribution of the flock:

    COLLECT mode: When the farthest prey is more than f_n * N^(2/3)
    from the flock centroid, predators target the farthest outlier
    to bring it back to the group.

    DRIVE mode: When the flock is cohesive enough, predators position
    behind the flock centroid (relative to the goal) and push the
    entire group toward the goal zone.

    Multi-predator extension: Predators are distributed with lateral
    offsets to cover the flock width. In collect mode, each predator
    targets a different outlier (sorted by distance from centroid).
"""

import math
from typing import List, Optional, Dict, Any
from src.algorithms.base_algorithm import BaseHerdingAlgorithm
from src.utils.math_helpers import distance


class StrombomShepherding(BaseHerdingAlgorithm):
    """
    Strombom (2014) collect-or-drive shepherding algorithm.

    Parameters:
        f_n: Threshold factor for collect/drive switching (default: 1.05)
        driving_distance: How far behind the flock centroid to position (default: herd_distance)
        collect_distance: How far behind the outlier to position (default: herd_distance)
    """

    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.f_n = parameters.get("f_n", 1.05)
        self.driving_distance = parameters.get("driving_distance", 120)
        self.collect_distance = parameters.get("collect_distance", 120)

    def assign_targets(
        self,
        predators: List,
        prey_list: List,
        comm_radius: float,
        conflict_rounds: int,
    ) -> int:
        """
        Assign prey targets based on collect-or-drive heuristic.

        In COLLECT mode: each predator targets a different outlier
        (sorted by distance from centroid, farthest first).

        In DRIVE mode: all predators target a virtual point behind
        the centroid — assigned_prey_index is set to -1 (sentinel)
        to indicate "drive the flock" rather than "pursue individual prey".

        Returns 0 messages (no inter-agent communication needed).
        """
        # Get active (undelivered) prey
        active_prey = [(i, p) for i, p in enumerate(prey_list) if not p.delivered]

        if not active_prey:
            for pred in predators:
                pred.assigned_prey_index = None
            return 0

        # Compute flock centroid
        cx = sum(p.position[0] for _, p in active_prey) / len(active_prey)
        cy = sum(p.position[1] for _, p in active_prey) / len(active_prey)

        # Find distances from centroid for each active prey
        prey_distances = []
        for idx, prey in active_prey:
            dist = distance([cx, cy], prey.position)
            prey_distances.append((idx, prey, dist))

        # Sort by distance from centroid (farthest first)
        prey_distances.sort(key=lambda x: x[2], reverse=True)

        # Threshold for collect/drive switching
        n_active = len(active_prey)
        threshold = self.f_n * (n_active ** (2.0 / 3.0))
        farthest_dist = prey_distances[0][2] if prey_distances else 0

        # Get active predators
        active_preds = [p for p in predators if not p.disengaged and p.energy_remaining > 0]

        if farthest_dist > threshold:
            # COLLECT mode: target outliers
            for i, pred in enumerate(active_preds):
                if i < len(prey_distances):
                    pred.assigned_prey_index = prey_distances[i][0]
                else:
                    # More predators than outliers: wrap around
                    pred.assigned_prey_index = prey_distances[i % len(prey_distances)][0]
                pred.formation_slot_index = i
        else:
            # DRIVE mode: all predators drive the flock
            # Use index of nearest undelivered prey to centroid as reference
            # (the predator will compute drive position in compute_pursue_target)
            nearest_to_centroid = min(active_prey, key=lambda x: distance([cx, cy], x[1].position))
            for i, pred in enumerate(active_preds):
                pred.assigned_prey_index = nearest_to_centroid[0]
                pred.formation_slot_index = i

        # Deactivated predators get no assignment
        for pred in predators:
            if pred.disengaged or pred.energy_remaining <= 0:
                pred.assigned_prey_index = None

        return 0  # No communication messages

    def compute_pursue_target(
        self,
        predator,
        prey_list: List,
        all_predators: List,
        goal_center: List[float],
        arena_bounds: tuple,
    ) -> Optional[List[float]]:
        """
        Compute target position based on collect-or-drive mode.

        Recomputes the mode each frame (stateless per predator).
        """
        if predator.assigned_prey_index is None:
            return None
        if predator.assigned_prey_index >= len(prey_list):
            return None

        # Get active prey
        active_prey = [(i, p) for i, p in enumerate(prey_list) if not p.delivered]
        if not active_prey:
            return None

        # Compute flock centroid
        cx = sum(p.position[0] for _, p in active_prey) / len(active_prey)
        cy = sum(p.position[1] for _, p in active_prey) / len(active_prey)
        centroid = [cx, cy]

        # Find farthest distance from centroid
        farthest_dist = max(distance(centroid, p.position) for _, p in active_prey)

        # Threshold
        n_active = len(active_prey)
        threshold = self.f_n * (n_active ** (2.0 / 3.0))

        if farthest_dist > threshold:
            # COLLECT: position behind the assigned outlier, away from centroid
            target_prey = prey_list[predator.assigned_prey_index]
            return self._collect_position(target_prey.position, centroid)
        else:
            # DRIVE: position behind centroid, away from goal
            return self._drive_position(
                centroid, goal_center, predator.formation_slot_index,
                len([p for p in all_predators if not p.disengaged and p.energy_remaining > 0])
            )

    def _collect_position(
        self,
        outlier_pos: List[float],
        centroid: List[float],
    ) -> List[float]:
        """Position behind the outlier, opposite to centroid direction."""
        dx = outlier_pos[0] - centroid[0]
        dy = outlier_pos[1] - centroid[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            return [outlier_pos[0] + self.collect_distance, outlier_pos[1]]

        # Unit vector from centroid to outlier
        ux = dx / dist
        uy = dy / dist

        # Position behind outlier (further from centroid)
        return [
            outlier_pos[0] + ux * self.collect_distance,
            outlier_pos[1] + uy * self.collect_distance,
        ]

    def _drive_position(
        self,
        centroid: List[float],
        goal_center: List[float],
        slot_index: int,
        total_active: int,
    ) -> List[float]:
        """Position behind flock centroid relative to goal, with lateral spread."""
        dx = goal_center[0] - centroid[0]
        dy = goal_center[1] - centroid[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            return centroid.copy()

        # Unit vector from centroid toward goal
        ux = dx / dist
        uy = dy / dist

        # Position behind centroid (away from goal)
        base_x = centroid[0] - ux * self.driving_distance
        base_y = centroid[1] - uy * self.driving_distance

        # Lateral spread perpendicular to driving direction
        perp_x = -uy
        perp_y = ux

        if total_active > 1:
            # Spread predators across the flock width
            lateral_offset = (slot_index - (total_active - 1) / 2.0) * 55
            base_x += perp_x * lateral_offset
            base_y += perp_y * lateral_offset

        return [base_x, base_y]

    def get_algorithm_name(self) -> str:
        return "Strombom Shepherding"

    def get_algorithm_parameters(self) -> Dict[str, Any]:
        return {
            "type": "strombom",
            "f_n": self.f_n,
            "driving_distance": self.driving_distance,
            "collect_distance": self.collect_distance,
        }
