"""
charging_station.py — Fixed charging station entity.

Charging stations are stationary points in the arena where predators
can replenish their energy. They are only active when the config
has charging.enabled = True.

Stations are used by:
    - M2 (TurtleBot3): simulates fixed charging + solar harvesting
    - M3 (Discrete Rate): simulates fixed charging stations from Paper 3
    - M4 (Quadratic): simulates the existing soft regeneration but localised

Stations are NOT used by:
    - M1 (Stolaroff): single-mission drones with no charging
"""

from typing import List
from src.utils.math_helpers import distance


class ChargingStation:
    """
    A fixed point in the arena where agents can recharge.

    Attributes:
        position: [x, y] coordinates in the arena.
        radius: How close an agent must be to receive charge.
        charge_rate: Energy units per frame delivered to agents within radius.
        station_id: Unique identifier for logging.
    """

    def __init__(self, position: List[float], radius: float,
                 charge_rate: float, station_id: int):
        self.position = position
        self.radius = radius
        self.charge_rate = charge_rate
        self.station_id = station_id

    def is_agent_in_range(self, agent_position: List[float]) -> bool:
        """Check if an agent is close enough to charge."""
        return distance(self.position, agent_position) <= self.radius


def create_stations_from_config(config: dict, arena_width: float,
                                 arena_height: float) -> List[ChargingStation]:
    """
    Create charging stations based on configuration.

    If station_positions is "auto", stations are evenly spaced along
    the bottom edge of the arena (opposite to the goal zone, which
    is at the top-right).

    Args:
        config: The charging section of the YAML config.
        arena_width: Width of the simulation arena in pixels.
        arena_height: Height of the simulation arena in pixels.

    Returns:
        List of ChargingStation instances.
    """
    if not config.get("enabled", False):
        return []

    count = config.get("station_count", 2)
    radius = config.get("station_radius", 40)
    charge_rate = config.get("charge_rate", 0.5)
    positions = config.get("station_positions", "auto")

    stations = []
    if positions == "auto":
        # Place stations evenly along the bottom edge
        margin = 80
        for i in range(count):
            if count == 1:
                x = arena_width / 2
            else:
                x = margin + (arena_width - 2 * margin) * i / (count - 1)
            y = arena_height - margin
            stations.append(ChargingStation([x, y], radius, charge_rate, station_id=i))
    else:
        # Explicit positions from config
        for i, pos in enumerate(positions):
            stations.append(ChargingStation(pos, radius, charge_rate, station_id=i))

    return stations
