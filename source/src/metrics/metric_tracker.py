"""
metric_tracker.py — Real-time metric collection during simulation.

Tracks per-frame and per-agent metrics for detailed analysis.
"""

from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class FrameMetrics:
    """Metrics collected for a single simulation frame."""
    frame_number: int
    delivered_count: int
    active_predators: int
    total_energy_consumed: float
    useful_energy_this_frame: float  # Energy spent when prey moved closer to goal


@dataclass
class AgentMetrics:
    """Per-agent cumulative metrics."""
    agent_id: int
    total_energy: float = 0.0
    useful_energy: float = 0.0  # Energy during frames that helped delivery
    frames_idle: int = 0
    frames_search: int = 0
    frames_pursue: int = 0
    deliveries_contributed: int = 0


class MetricTracker:
    """
    Tracks metrics throughout a simulation episode.

    Collects both frame-level and agent-level data for detailed analysis.
    """

    def __init__(self, num_predators: int):
        self.num_predators = num_predators
        self.frame_history: List[FrameMetrics] = []
        self.agent_metrics: Dict[int, AgentMetrics] = {
            i: AgentMetrics(agent_id=i) for i in range(num_predators)
        }

    def record_frame(
        self,
        frame_number: int,
        delivered_count: int,
        active_predators: int,
        total_energy_consumed: float,
        useful_energy: float,
    ):
        """Record metrics for a single frame."""
        self.frame_history.append(
            FrameMetrics(
                frame_number=frame_number,
                delivered_count=delivered_count,
                active_predators=active_predators,
                total_energy_consumed=total_energy_consumed,
                useful_energy_this_frame=useful_energy,
            )
        )

    def update_agent(
        self,
        agent_id: int,
        energy_spent: float,
        is_useful: bool,
        mode: str,
    ):
        """
        Update per-agent metrics.

        Args:
            agent_id: Predator ID
            energy_spent: Energy consumed this frame
            is_useful: Whether this energy helped delivery
            mode: Agent mode ("idle", "search", "pursue")
        """
        if agent_id not in self.agent_metrics:
            return

        metrics = self.agent_metrics[agent_id]
        metrics.total_energy += energy_spent

        if is_useful:
            metrics.useful_energy += energy_spent

        if mode == "idle":
            metrics.frames_idle += 1
        elif mode == "search":
            metrics.frames_search += 1
        elif mode == "pursue":
            metrics.frames_pursue += 1

    def compute_efficiency(self) -> float:
        """
        Compute overall energy efficiency.

        η = useful_energy / total_energy

        Returns:
            Energy efficiency in [0, 1]
        """
        total_useful = sum(a.useful_energy for a in self.agent_metrics.values())
        total_consumed = sum(a.total_energy for a in self.agent_metrics.values())

        if total_consumed > 0:
            return total_useful / total_consumed
        return 0.0

    def compute_duty_cycle(self, agent_id: int) -> float:
        """
        Compute duty cycle for a specific agent.

        D = frames_pursue / total_frames

        Args:
            agent_id: Predator ID

        Returns:
            Duty cycle in [0, 1]
        """
        if agent_id not in self.agent_metrics:
            return 0.0

        metrics = self.agent_metrics[agent_id]
        total_frames = metrics.frames_idle + metrics.frames_search + metrics.frames_pursue

        if total_frames > 0:
            return metrics.frames_pursue / total_frames
        return 0.0

    def get_summary(self) -> Dict:
        """
        Get a summary of all collected metrics.

        Returns:
            Dictionary with aggregate statistics
        """
        if not self.agent_metrics:
            return {}

        avg_duty_cycle = sum(
            self.compute_duty_cycle(i) for i in range(self.num_predators)
        ) / self.num_predators

        return {
            "energy_efficiency": self.compute_efficiency(),
            "avg_duty_cycle": avg_duty_cycle,
            "total_energy": sum(a.total_energy for a in self.agent_metrics.values()),
            "useful_energy": sum(a.useful_energy for a in self.agent_metrics.values()),
            "total_frames": len(self.frame_history),
        }
