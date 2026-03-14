"""
src/metrics — Metrics tracking and analysis.

Provides tools for collecting and analyzing simulation performance:
    - MetricTracker: Real-time metric collection
    - EpisodeLogger: CSV logging
    - metric_definitions: Standard formulas for efficiency, duty cycle, etc.
"""

from src.metrics.metric_tracker import MetricTracker, FrameMetrics, AgentMetrics
from src.metrics.episode_logger import EpisodeLogger
from src.metrics.metric_definitions import (
    compute_energy_efficiency,
    compute_duty_cycle,
    compute_task_completion_rate,
    compute_cost_per_delivery,
    compute_theoretical_upper_bound,
    compute_normalized_efficiency,
)

__all__ = [
    "MetricTracker",
    "FrameMetrics",
    "AgentMetrics",
    "EpisodeLogger",
    "compute_energy_efficiency",
    "compute_duty_cycle",
    "compute_task_completion_rate",
    "compute_cost_per_delivery",
    "compute_theoretical_upper_bound",
    "compute_normalized_efficiency",
]
