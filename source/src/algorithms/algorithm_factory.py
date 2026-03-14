"""
algorithm_factory.py — Factory for creating herding algorithm instances from config.

Usage:
    algorithm = create_herding_algorithm(config["algorithm"])
"""

from typing import Dict, Any
from src.algorithms.base_algorithm import BaseHerdingAlgorithm
from src.algorithms.wolf_pack_formation import WolfPackFormation
from src.algorithms.strombom_shepherding import StrombomShepherding
from src.algorithms.simple_apf import SimpleAPF
from src.algorithms.wolf_apf import WolfAPF


# Registry: algorithm name string -> algorithm class
_ALGORITHM_REGISTRY: Dict[str, type] = {
    "wolf_pack_formation": WolfPackFormation,
    "strombom": StrombomShepherding,
    "simple_apf": SimpleAPF,
    "wolf_apf": WolfAPF,
}


def create_herding_algorithm(algorithm_config: Dict[str, Any]) -> BaseHerdingAlgorithm:
    """
    Create a herding algorithm instance from a config dictionary.

    Args:
        algorithm_config: Dictionary with keys:
            - "name": string matching a key in _ALGORITHM_REGISTRY
            - "parameters": dict of algorithm-specific parameters

    Returns:
        An instance of the requested herding algorithm.

    Raises:
        ValueError: If the algorithm name is not recognised.
    """
    algo_name = algorithm_config.get("name", "wolf_pack_formation")
    parameters = algorithm_config.get("parameters", {})

    if algo_name not in _ALGORITHM_REGISTRY:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown herding algorithm: '{algo_name}'\n"
            f"Available algorithms: {available}\n"
            f"Set algorithm.name in your YAML config to one of these."
        )

    algo_class = _ALGORITHM_REGISTRY[algo_name]
    return algo_class(parameters)


def register_algorithm(name: str, algo_class: type):
    """Register a new algorithm class in the factory."""
    _ALGORITHM_REGISTRY[name] = algo_class
