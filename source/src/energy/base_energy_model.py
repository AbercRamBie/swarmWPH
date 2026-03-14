"""
base_energy_model.py — Abstract interface for energy models.

Every energy model in this project inherits from BaseEnergyModel.
This ensures they all provide the same methods, making them
interchangeable in the simulation without any code changes.

The simulation calls these methods each frame for each predator.
The model returns how much energy was consumed. The simulation
then deducts that from the predator's energy storage.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseEnergyModel(ABC):
    """
    Abstract base class for all energy models.

    Every energy model must implement:
        - compute_motion_cost(): energy spent moving this frame
        - compute_idle_cost(): energy spent doing nothing this frame
        - compute_communication_cost(): energy spent on comms this frame
        - get_model_name(): human-readable name for logs and plots
        - get_model_parameters(): dict of all parameters for reproducibility

    Optional:
        - compute_charging_gain(): energy gained from charging this frame
    """

    def __init__(self, config_parameters: Dict[str, Any]):
        """
        Args:
            config_parameters: Model-specific parameters from the YAML config
                               under energy_model.parameters.
        """
        self._parameters = config_parameters

    @abstractmethod
    def compute_motion_cost(
        self,
        speed: float,
        turn_rate: float,
        mode: str,
        mass: float = 0.0,
        delta_time: float = 1.0 / 60.0,
    ) -> float:
        """
        Compute energy consumed by movement in one simulation step.

        Args:
            speed: Current speed of the agent (pixels/frame in sim, converted internally).
            turn_rate: Current turning rate (radians/frame).
            mode: Agent's behavioural mode — "search", "pursue", or "idle".
            mass: Total mass of the agent (relevant for drone/ground models).
            delta_time: Duration of this simulation step in seconds.

        Returns:
            Energy consumed (positive float). Units depend on the model's
            internal representation, but must be consistent with energy_capacity.
        """
        pass

    @abstractmethod
    def compute_idle_cost(self, delta_time: float = 1.0 / 60.0) -> float:
        """
        Compute energy consumed while stationary (base operations).

        Args:
            delta_time: Duration of this simulation step in seconds.

        Returns:
            Energy consumed (positive float).
        """
        pass

    @abstractmethod
    def compute_communication_cost(
        self,
        message_count: int,
        delta_time: float = 1.0 / 60.0,
    ) -> float:
        """
        Compute energy consumed by inter-agent communication.

        Args:
            message_count: Number of messages sent/received this step.
            delta_time: Duration of this simulation step in seconds.

        Returns:
            Energy consumed (positive float).
        """
        pass

    def compute_charging_gain(
        self,
        is_at_station: bool,
        charge_rate: float,
        delta_time: float = 1.0 / 60.0,
    ) -> float:
        """
        Compute energy gained from charging this step.
        Default implementation: simple constant-rate charging.
        Override in models with special charging behaviour.

        Args:
            is_at_station: Whether the agent is within a charging station's radius.
            charge_rate: Energy units gained per second while charging.
            delta_time: Duration of this simulation step in seconds.

        Returns:
            Energy gained (positive float), or 0.0 if not charging.
        """
        if is_at_station:
            return charge_rate * delta_time
        return 0.0

    @abstractmethod
    def get_model_name(self) -> str:
        """Return human-readable name for logs and plot labels."""
        pass

    @abstractmethod
    def get_model_parameters(self) -> Dict[str, Any]:
        """Return all model parameters as a dict (for logging/reproducibility)."""
        pass
