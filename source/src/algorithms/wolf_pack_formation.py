"""
wolf_pack_formation.py — Wolf Pack Formation herding algorithm.

Wraps the existing decentralized assignment and formation-based
herding behavior. This is the default algorithm and produces
identical results to the original simulation when selected.

Strategy:
    - Assignment: Greedy closest-prey + claim-strength conflict resolution
    - Pursuit: Formation positions behind prey relative to goal
      with lateral lane offsets per formation slot
"""

from typing import List, Optional, Dict, Any
from src.algorithms.base_algorithm import BaseHerdingAlgorithm
from src.core.assignment import assign_prey_to_predators


class WolfPackFormation(BaseHerdingAlgorithm):
    """
    Original wolf pack formation algorithm.

    Delegates assignment to the existing decentralized system
    and uses the predator's built-in formation positioning.
    """

    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)

    def assign_targets(
        self,
        predators: List,
        prey_list: List,
        comm_radius: float,
        conflict_rounds: int,
    ) -> int:
        """Delegate to existing decentralized assignment."""
        return assign_prey_to_predators(
            predators, prey_list, comm_radius, conflict_rounds
        )

    def compute_pursue_target(
        self,
        predator,
        prey_list: List,
        all_predators: List,
        goal_center: List[float],
        arena_bounds: tuple,
    ) -> Optional[List[float]]:
        """Use predator's built-in formation positioning."""
        if predator.assigned_prey_index is None:
            return None
        if predator.assigned_prey_index >= len(prey_list):
            return None

        prey = prey_list[predator.assigned_prey_index]
        if prey.delivered:
            return None

        return predator._compute_formation_position(
            prey.position, goal_center
        )

    def get_algorithm_name(self) -> str:
        return "Wolf Pack Formation"

    def get_algorithm_parameters(self) -> Dict[str, Any]:
        return {"type": "wolf_pack_formation", **self._parameters}
