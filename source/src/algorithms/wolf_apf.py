"""
wolf_apf.py — Wolf Pack + APF hybrid herding algorithm.

Based on: Sun et al. (2022) "Multi-robot target encirclement via
a Wolf Pack Algorithm with APF" — role-differentiated pursuit
with artificial potential field collision avoidance.

Strategy:
    Role assignment based on energy ranking:
        - Alpha (top 1): Leads the pack, pursues directly behind prey
        - Beta (next 40%): Flanking positions at configurable angle offset
        - Omega (remaining): Follow alpha at distance, provide rear support

    All roles use APF repulsion from other predators for collision avoidance.

    Assignment: Alpha and Beta each target the nearest undelivered prey.
    Omega predators follow the alpha rather than targeting prey directly.
"""

import math
from typing import List, Optional, Dict, Any
from src.algorithms.base_algorithm import BaseHerdingAlgorithm
from src.utils.math_helpers import distance


# Role constants
ROLE_ALPHA = "alpha"
ROLE_BETA = "beta"
ROLE_OMEGA = "omega"


class WolfAPF(BaseHerdingAlgorithm):
    """
    Wolf Pack + APF hybrid herding algorithm.

    Parameters:
        alpha_count: Number of alpha wolves (default: 1)
        beta_fraction: Fraction of remaining wolves that are beta (default: 0.4)
        flank_angle: Flanking angle for beta wolves in radians (default: pi/3)
        follow_distance: Distance omega wolves maintain from alpha (default: 150)
        k_repel: APF repulsion gain (default: 0.5)
        repel_radius: Distance for APF repulsion (default: 50)
    """

    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.alpha_count = parameters.get("alpha_count", 1)
        self.beta_fraction = parameters.get("beta_fraction", 0.4)
        self.flank_angle = parameters.get("flank_angle", math.pi / 3)
        self.follow_distance = parameters.get("follow_distance", 150)
        self.k_repel = parameters.get("k_repel", 0.5)
        self.repel_radius = parameters.get("repel_radius", 50)

        # Role assignments (recomputed each frame)
        self._roles: Dict[int, str] = {}

    def assign_targets(
        self,
        predators: List,
        prey_list: List,
        comm_radius: float,
        conflict_rounds: int,
    ) -> int:
        """
        Assign roles and targets based on energy ranking.

        1. Rank active predators by energy (highest first)
        2. Top alpha_count -> ALPHA role
        3. Next beta_fraction of remaining -> BETA role
        4. Rest -> OMEGA role
        5. Alpha/Beta: assigned nearest undelivered prey
        6. Omega: assigned same prey as their nearest alpha

        Returns message count for role broadcast.
        """
        active_prey = [(i, p) for i, p in enumerate(prey_list) if not p.delivered]

        if not active_prey:
            for pred in predators:
                pred.assigned_prey_index = None
            self._roles.clear()
            return 0

        # Get active predators sorted by energy (highest first)
        active_preds = [p for p in predators if not p.disengaged and p.energy_remaining > 0]
        active_preds.sort(key=lambda p: p.energy_remaining, reverse=True)

        # Assign roles
        self._roles.clear()
        n_active = len(active_preds)
        n_alpha = min(self.alpha_count, n_active)
        n_remaining = n_active - n_alpha
        n_beta = int(n_remaining * self.beta_fraction)

        alphas = []
        betas = []
        omegas = []

        for i, pred in enumerate(active_preds):
            if i < n_alpha:
                self._roles[pred.predator_id] = ROLE_ALPHA
                alphas.append(pred)
            elif i < n_alpha + n_beta:
                self._roles[pred.predator_id] = ROLE_BETA
                betas.append(pred)
            else:
                self._roles[pred.predator_id] = ROLE_OMEGA
                omegas.append(pred)

        # Alpha: target nearest undelivered prey
        for pred in alphas:
            closest_idx = self._find_nearest_prey(pred, prey_list)
            pred.assigned_prey_index = closest_idx
            pred.formation_slot_index = 0

        # Beta: target nearest undelivered prey (may overlap with alpha)
        for i, pred in enumerate(betas):
            closest_idx = self._find_nearest_prey(pred, prey_list)
            pred.assigned_prey_index = closest_idx
            pred.formation_slot_index = i + 1  # Used for flank side

        # Omega: follow nearest alpha's target
        for pred in omegas:
            if alphas:
                # Find nearest alpha
                nearest_alpha = min(alphas, key=lambda a: distance(pred.position, a.position))
                pred.assigned_prey_index = nearest_alpha.assigned_prey_index
            else:
                # No alpha, fall back to nearest prey
                pred.assigned_prey_index = self._find_nearest_prey(pred, prey_list)
            pred.formation_slot_index = 0

        # Deactivated predators
        for pred in predators:
            if pred.disengaged or pred.energy_remaining <= 0:
                pred.assigned_prey_index = None
                self._roles[pred.predator_id] = ROLE_OMEGA

        # Role broadcast: 1 message per active predator
        return n_active

    def compute_pursue_target(
        self,
        predator,
        prey_list: List,
        all_predators: List,
        goal_center: List[float],
        arena_bounds: tuple,
    ) -> Optional[List[float]]:
        """
        Compute target based on wolf role.

        Alpha: directly behind prey, opposite to goal
        Beta: flanking position at flank_angle offset from prey-goal line
        Omega: follow alpha at follow_distance
        """
        if predator.assigned_prey_index is None:
            return None
        if predator.assigned_prey_index >= len(prey_list):
            return None

        prey = prey_list[predator.assigned_prey_index]
        if prey.delivered:
            return None

        role = self._roles.get(predator.predator_id, ROLE_OMEGA)

        if role == ROLE_ALPHA:
            base_target = self._alpha_target(predator, prey.position, goal_center)
        elif role == ROLE_BETA:
            base_target = self._beta_target(predator, prey.position, goal_center)
        else:
            base_target = self._omega_target(predator, all_predators, prey.position, goal_center)

        # Add APF repulsion from other predators
        repel_x, repel_y = self._compute_repulsion(predator, all_predators)
        target_x = base_target[0] + repel_x
        target_y = base_target[1] + repel_y

        # Clamp to arena bounds
        width, height = arena_bounds
        target_x = max(0, min(width, target_x))
        target_y = max(0, min(height, target_y))

        return [target_x, target_y]

    def _alpha_target(
        self,
        predator,
        prey_pos: List[float],
        goal_center: List[float],
    ) -> List[float]:
        """Alpha: position directly behind prey, opposite to goal."""
        dx = goal_center[0] - prey_pos[0]
        dy = goal_center[1] - prey_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            return prey_pos.copy()

        ux = dx / dist
        uy = dy / dist

        return [
            prey_pos[0] - ux * predator.herd_distance,
            prey_pos[1] - uy * predator.herd_distance,
        ]

    def _beta_target(
        self,
        predator,
        prey_pos: List[float],
        goal_center: List[float],
    ) -> List[float]:
        """Beta: flanking position at angle offset from prey-goal line."""
        dx = goal_center[0] - prey_pos[0]
        dy = goal_center[1] - prey_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            return prey_pos.copy()

        # Base angle from prey to goal
        base_angle = math.atan2(dy, dx)

        # Alternate flanking side based on slot index
        side = 1 if predator.formation_slot_index % 2 == 0 else -1
        flank_angle = base_angle + math.pi + side * self.flank_angle

        return [
            prey_pos[0] + predator.herd_distance * math.cos(flank_angle),
            prey_pos[1] + predator.herd_distance * math.sin(flank_angle),
        ]

    def _omega_target(
        self,
        predator,
        all_predators: List,
        prey_pos: List[float],
        goal_center: List[float],
    ) -> List[float]:
        """Omega: follow nearest alpha at follow_distance."""
        # Find nearest alpha
        nearest_alpha = None
        nearest_dist = float('inf')

        for other in all_predators:
            if self._roles.get(other.predator_id) == ROLE_ALPHA:
                dist = distance(predator.position, other.position)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_alpha = other

        if nearest_alpha is not None:
            # Position behind alpha (relative to alpha's heading toward prey)
            dx = prey_pos[0] - nearest_alpha.position[0]
            dy = prey_pos[1] - nearest_alpha.position[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 1e-6:
                return nearest_alpha.position.copy()

            ux = dx / dist
            uy = dy / dist

            # Follow behind alpha
            return [
                nearest_alpha.position[0] - ux * self.follow_distance,
                nearest_alpha.position[1] - uy * self.follow_distance,
            ]
        else:
            # No alpha available, fall back to basic behind-prey position
            return self._alpha_target(predator, prey_pos, goal_center)

    def _compute_repulsion(
        self,
        predator,
        all_predators: List,
    ) -> tuple:
        """Compute APF repulsion force from nearby predators."""
        repel_x, repel_y = 0.0, 0.0

        for other in all_predators:
            if other.predator_id == predator.predator_id:
                continue
            dist = distance(predator.position, other.position)
            if dist < self.repel_radius and dist > 1e-6:
                strength = self.k_repel * (self.repel_radius - dist) / self.repel_radius
                repel_x += (predator.position[0] - other.position[0]) / dist * strength * predator.herd_distance
                repel_y += (predator.position[1] - other.position[1]) / dist * strength * predator.herd_distance

        return repel_x, repel_y

    @staticmethod
    def _find_nearest_prey(predator, prey_list: List) -> Optional[int]:
        """Find index of nearest undelivered prey."""
        closest_idx = None
        closest_dist = float('inf')

        for i, prey in enumerate(prey_list):
            if prey.delivered:
                continue
            dist = distance(predator.position, prey.position)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i

        return closest_idx

    def get_algorithm_name(self) -> str:
        return "Wolf+APF"

    def get_algorithm_parameters(self) -> Dict[str, Any]:
        return {
            "type": "wolf_apf",
            "alpha_count": self.alpha_count,
            "beta_fraction": self.beta_fraction,
            "flank_angle": self.flank_angle,
            "follow_distance": self.follow_distance,
            "k_repel": self.k_repel,
            "repel_radius": self.repel_radius,
        }
