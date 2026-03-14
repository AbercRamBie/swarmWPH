"""
base_algorithm.py — Abstract interface for herding algorithms.

Every herding algorithm inherits from BaseHerdingAlgorithm.
This ensures they all provide the same methods, making them
interchangeable in the simulation via config.

The simulation calls:
    1. assign_targets() once per frame to set predator assignments
    2. compute_pursue_target() per predator per frame to get target position
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class BaseHerdingAlgorithm(ABC):
    """
    Abstract base class for all herding algorithms.

    Every algorithm must implement:
        - assign_targets(): set predator prey assignments each frame
        - compute_pursue_target(): compute [x,y] target for a pursuing predator
        - get_algorithm_name(): human-readable name for logs/plots
        - get_algorithm_parameters(): dict of params for reproducibility
    """

    def __init__(self, parameters: Dict[str, Any]):
        """
        Args:
            parameters: Algorithm-specific parameters from YAML config
                        under algorithm.parameters.
        """
        self._parameters = parameters

    @abstractmethod
    def assign_targets(
        self,
        predators: List,
        prey_list: List,
        comm_radius: float,
        conflict_rounds: int,
    ) -> int:
        """
        Assign prey targets to predators for this frame.

        Sets predator.assigned_prey_index and predator.formation_slot_index.

        Args:
            predators: List of all Predator agents
            prey_list: List of all Prey agents
            comm_radius: Communication range for neighbor detection
            conflict_rounds: Rounds of conflict resolution

        Returns:
            Total number of communication messages sent
        """
        pass

    @abstractmethod
    def compute_pursue_target(
        self,
        predator,
        prey_list: List,
        all_predators: List,
        goal_center: List[float],
        arena_bounds: tuple,
    ) -> Optional[List[float]]:
        """
        Compute the target [x, y] position for a pursuing predator.

        Called each frame for each predator in PURSUE mode.
        The predator then steers toward this position using its
        existing _step_towards_point() method.

        Args:
            predator: The predator agent computing its target
            prey_list: List of all Prey agents
            all_predators: List of all Predator agents
            goal_center: [x, y] of goal zone center
            arena_bounds: (width, height) of arena

        Returns:
            [x, y] target position, or None if no valid target
        """
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return human-readable name for logs and plot labels."""
        pass

    @abstractmethod
    def get_algorithm_parameters(self) -> Dict[str, Any]:
        """Return all algorithm parameters as a dict (for logging/reproducibility)."""
        pass
