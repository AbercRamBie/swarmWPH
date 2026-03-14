"""
src/energy — Energy model implementations.

Two real-world energy models are implemented here:
    - M1: Stolaroff Drone (physics-based quadrotor)
    - M2: TurtleBot3 Empirical (data-fitted ground robot)

Use the factory to create models from config:
    from src.energy import create_energy_model
    model = create_energy_model(config["energy_model"])
"""

from src.energy.base_energy_model import BaseEnergyModel
from src.energy.energy_factory import create_energy_model
from src.energy.stolaroff_drone import StolaroffDroneModel
from src.energy.turtlebot3_empirical import TurtleBot3EmpiricalModel

__all__ = [
    "BaseEnergyModel",
    "create_energy_model",
    "StolaroffDroneModel",
    "TurtleBot3EmpiricalModel",
]
