"""
src/algorithms — Herding algorithm implementations.

Four herding algorithms are available:
    - Wolf Pack Formation (existing, default)
    - Strombom (2014) collect-or-drive shepherding
    - Simple APF (artificial potential field)
    - Wolf+APF (Sun 2022, role-based with APF)

Use the factory to create algorithms from config:
    from src.algorithms import create_herding_algorithm
    algorithm = create_herding_algorithm(config["algorithm"])
"""

from src.algorithms.base_algorithm import BaseHerdingAlgorithm
from src.algorithms.algorithm_factory import create_herding_algorithm

__all__ = [
    "BaseHerdingAlgorithm",
    "create_herding_algorithm",
]
