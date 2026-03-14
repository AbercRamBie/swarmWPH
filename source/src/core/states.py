"""
states.py — State enums for predator behavior modes.

Predators operate in different modes depending on their energy level
and whether they have an assigned prey to herd.
"""

from enum import Enum


class PredatorMode(Enum):
    """
    Behavioral modes for predator agents.

    IDLE: Agent has no energy and cannot move.
    SEARCH: Agent has low energy, wanders randomly looking for opportunities.
    PURSUE: Agent is actively herding an assigned prey toward the goal.
    """
    IDLE = "idle"
    SEARCH = "search"
    PURSUE = "pursue"

    def __str__(self):
        return self.value
