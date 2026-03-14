"""
metric_definitions.py — Formulas for computing performance metrics.

Defines the exact calculations for:
    - Energy efficiency (η)
    - Duty cycle (D)
    - Task completion rate (TCR)
    - Cost per delivery (CPD)
    - Normalized efficiency vs theoretical bound
"""

from typing import List
import math


def compute_energy_efficiency(useful_energy: float, total_energy: float) -> float:
    """
    Compute energy efficiency.

    η = useful_energy / total_energy

    Where useful_energy is energy spent during frames where the predator
    was pursuing AND the prey moved closer to the goal.

    Args:
        useful_energy: Energy spent on productive work
        total_energy: Total energy consumed

    Returns:
        Efficiency in [0, 1], or 0 if no energy consumed
    """
    if total_energy <= 0:
        return 0.0
    return min(1.0, useful_energy / total_energy)


def compute_duty_cycle(frames_pursue: int, total_frames: int) -> float:
    """
    Compute duty cycle.

    D = frames_pursue / total_frames

    Args:
        frames_pursue: Number of frames in pursue mode
        total_frames: Total simulation frames

    Returns:
        Duty cycle in [0, 1], or 0 if no frames
    """
    if total_frames <= 0:
        return 0.0
    return frames_pursue / total_frames


def compute_task_completion_rate(delivered: int, total_prey: int) -> float:
    """
    Compute task completion rate.

    TCR = delivered / total_prey

    Args:
        delivered: Number of prey delivered
        total_prey: Total number of prey

    Returns:
        Completion rate in [0, 1]
    """
    if total_prey <= 0:
        return 0.0
    return delivered / total_prey


def compute_cost_per_delivery(total_energy: float, delivered: int) -> float:
    """
    Compute average energy cost per delivered prey.

    CPD = total_energy / delivered

    Args:
        total_energy: Total energy consumed
        delivered: Number of prey delivered

    Returns:
        Average cost per delivery, or float('inf') if no deliveries
    """
    if delivered <= 0:
        return float('inf')
    return total_energy / delivered


def compute_theoretical_upper_bound(
    energy_model_name: str,
    e_motion: float,
    e_idle: float,
) -> float:
    """
    Compute theoretical maximum energy efficiency.

    η_max ≈ 0.5 * (1 - e_idle / e_motion)

    The 0.5 factor comes from the round-trip nature of herding:
    predators must travel TO prey AND BACK for the next one.

    Args:
        energy_model_name: Name of energy model (for logging)
        e_motion: Energy cost of motion at optimal speed
        e_idle: Energy cost of idle/stationary state

    Returns:
        Theoretical maximum efficiency in [0, 0.5]
    """
    if e_motion <= 0:
        return 0.0

    # Basic bound: half the energy is "wasted" on return trips
    eta_max = 0.5 * (1.0 - e_idle / e_motion)

    # Clamp to [0, 0.5]
    return max(0.0, min(0.5, eta_max))


def compute_normalized_efficiency(
    actual_efficiency: float,
    theoretical_max: float,
) -> float:
    """
    Compute efficiency relative to theoretical upper bound.

    η_norm = η / η_max

    This shows how close the algorithm gets to theoretical optimum.

    Args:
        actual_efficiency: Measured energy efficiency
        theoretical_max: Theoretical maximum efficiency

    Returns:
        Normalized efficiency (can exceed 1.0 if bound is pessimistic)
    """
    if theoretical_max <= 0:
        return 0.0
    return actual_efficiency / theoretical_max
