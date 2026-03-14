"""
energy_factory.py — Factory for creating energy model instances from config.

Usage:
    model = create_energy_model(config["energy_model"])
"""

from typing import Dict, Any
from src.energy.base_energy_model import BaseEnergyModel
from src.energy.stolaroff_drone import StolaroffDroneModel
from src.energy.turtlebot3_empirical import TurtleBot3EmpiricalModel


# Registry: model name string -> model class
_MODEL_REGISTRY = {
    "stolaroff_drone": StolaroffDroneModel,
    "turtlebot3_empirical": TurtleBot3EmpiricalModel,
}


def create_energy_model(energy_config: Dict[str, Any]) -> BaseEnergyModel:
    """
    Create an energy model instance from a config dictionary.

    Args:
        energy_config: Dictionary with keys:
            - "name": string matching a key in _MODEL_REGISTRY
            - "parameters": dict of model-specific parameters

    Returns:
        An instance of the requested energy model.

    Raises:
        ValueError: If the model name is not recognised.

    Example:
        >>> config = {"name": "stolaroff_drone", "parameters": {"body_mass_kg": 1.5}}
        >>> model = create_energy_model(config)
    """
    model_name = energy_config.get("name", "stolaroff_drone")
    parameters = energy_config.get("parameters", {})

    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown energy model: '{model_name}'\n"
            f"Available models: {available}\n"
            f"Set energy_model.name in your YAML config to one of these."
        )

    model_class = _MODEL_REGISTRY[model_name]
    return model_class(parameters)
