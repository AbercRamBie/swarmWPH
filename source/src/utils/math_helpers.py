"""
math_helpers.py — Shared mathematical utilities for the swarm simulation.

These small functions are used by both predator and prey agents for
vector arithmetic, angle wrapping, and distance calculations.
"""

import math
from typing import Tuple, List


def clamp(value: float, minimum: float, maximum: float) -> float:
    """
    Constrain a value to lie within [minimum, maximum].

    Args:
        value: The number to clamp.
        minimum: The lowest allowed value.
        maximum: The highest allowed value.

    Returns:
        The clamped value.

    Example:
        >>> clamp(15.0, 0.0, 10.0)
        10.0
    """
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def limit_vector(vx: float, vy: float, max_magnitude: float) -> Tuple[float, float]:
    """
    Scale a 2D vector so its magnitude does not exceed max_magnitude.
    If the vector is already shorter, return it unchanged.

    Args:
        vx: X component of the vector.
        vy: Y component of the vector.
        max_magnitude: Maximum allowed length.

    Returns:
        Tuple (vx, vy) scaled down if necessary.
    """
    magnitude_squared = vx * vx + vy * vy
    if magnitude_squared <= max_magnitude * max_magnitude:
        return vx, vy
    magnitude = math.sqrt(max(1e-9, magnitude_squared))
    scale = max_magnitude / magnitude
    return vx * scale, vy * scale


def wrap_angle(angle_radians: float) -> float:
    """
    Wrap an angle to the range (-π, π].

    Args:
        angle_radians: Any angle in radians.

    Returns:
        Equivalent angle in (-π, π].
    """
    while angle_radians > math.pi:
        angle_radians -= 2.0 * math.pi
    while angle_radians < -math.pi:
        angle_radians += 2.0 * math.pi
    return angle_radians


def distance_squared(position_a: List[float], position_b: List[float]) -> float:
    """
    Compute squared Euclidean distance between two 2D points.
    Use this instead of distance() when you only need to compare distances
    (avoids the expensive sqrt).

    Args:
        position_a: [x, y] of point A.
        position_b: [x, y] of point B.

    Returns:
        Squared distance.
    """
    dx = position_a[0] - position_b[0]
    dy = position_a[1] - position_b[1]
    return dx * dx + dy * dy


def distance(position_a: List[float], position_b: List[float]) -> float:
    """
    Compute Euclidean distance between two 2D points.

    Args:
        position_a: [x, y] of point A.
        position_b: [x, y] of point B.

    Returns:
        Distance as a float.
    """
    return math.sqrt(distance_squared(position_a, position_b))


def inside_rectangle(position: List[float], rect_x: float, rect_y: float,
                     rect_width: float, rect_height: float) -> bool:
    """
    Check if a 2D point is inside an axis-aligned rectangle.

    Args:
        position: [x, y] of the point.
        rect_x: Left edge of rectangle.
        rect_y: Top edge of rectangle.
        rect_width: Width of rectangle.
        rect_height: Height of rectangle.

    Returns:
        True if point is inside the rectangle.
    """
    return (rect_x <= position[0] <= rect_x + rect_width and
            rect_y <= position[1] <= rect_y + rect_height)
