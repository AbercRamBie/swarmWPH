"""
stolaroff_drone.py — Stolaroff Quadrotor Power Consumption Model (M1)

Source: Qin & Pournaras (2023), Appendix A, Eqs. 13–17
        Originally from: Stolaroff et al. (2018)

This model computes power consumption of a multirotor drone based on
momentum theory. The core idea: a drone generates thrust by pushing
air downward with its propellers. Power depends on how much thrust
is needed (to overcome gravity + drag) and how efficiently the
rotors can generate that thrust.

Key equations:
    Eq. 13: T = (m_b + m_c) · g + F_d                     (total thrust)
    Eq. 14: F_d = (m_b + m_c) · g · tan(θ)                (drag force)
    Eq. 15: P_fly = (v·sinθ + v_i) · T / ε                (flying power)
    Eq. 16: v_i = 2T / (π·d²·r·ρ·√((v·cosθ)²+(v·sinθ+v_i)²))  (induced velocity)
    Eq. 17: P_hover = T^(3/2) / (ε · √(½·π·d²·n·ρ))      (hovering power)

Adaptation for herding simulation:
    - "Flying" = predator is moving (speed > 0)
    - "Hovering" = predator is stationary but active (speed ≈ 0)
    - Speed in simulation (pixels/frame) is converted to m/s using a scaling factor
    - Turning cost is added as a percentage overhead on flying power
      (drones expend extra energy during banked turns)
"""

import math
from typing import Dict, Any
from src.energy.base_energy_model import BaseEnergyModel
from src.utils.constants import (
    GRAVITY_M_S2,
    AIR_DENSITY_KG_M3,
    DJI_PHANTOM4_BODY_MASS_KG,
    DJI_PHANTOM4_BATTERY_MASS_KG,
    DJI_PHANTOM4_PROPELLER_DIAMETER_M,
    DJI_PHANTOM4_PROPELLER_COUNT,
    DJI_PHANTOM4_GROUND_SPEED_M_S,
    DJI_PHANTOM4_DRAG_FORCE_N,
    DJI_PHANTOM4_POWER_EFFICIENCY,
    DJI_PHANTOM4_BATTERY_CAPACITY_J,
    DEFAULT_PIXELS_PER_METRE,
    DEFAULT_SECONDS_PER_FRAME,
)


class StolaroffDroneModel(BaseEnergyModel):
    """
    Stolaroff quadrotor energy model.

    Computes per-step energy consumption based on whether the drone is
    flying (moving forward) or hovering (stationary). Both are derived
    from momentum theory — how much thrust is needed and how much air
    must be moved to produce that thrust.

    NOTE: No charging is supported in this model. Drones have a single
    mission budget (battery capacity). When energy reaches zero, the
    agent becomes inactive.
    """

    def __init__(self, config_parameters: Dict[str, Any]):
        super().__init__(config_parameters)

        # --- Load parameters with defaults from DJI Phantom 4 Pro ---
        self.body_mass_kg = config_parameters.get("body_mass_kg", DJI_PHANTOM4_BODY_MASS_KG)
        self.battery_mass_kg = config_parameters.get("battery_mass_kg", DJI_PHANTOM4_BATTERY_MASS_KG)
        self.propeller_diameter_m = config_parameters.get("propeller_diameter_m", DJI_PHANTOM4_PROPELLER_DIAMETER_M)
        self.propeller_count = config_parameters.get("propeller_count", DJI_PHANTOM4_PROPELLER_COUNT)
        self.ground_speed_m_s = config_parameters.get("ground_speed_m_s", DJI_PHANTOM4_GROUND_SPEED_M_S)
        self.drag_force_n = config_parameters.get("drag_force_n", DJI_PHANTOM4_DRAG_FORCE_N)
        self.power_efficiency = config_parameters.get("power_efficiency", DJI_PHANTOM4_POWER_EFFICIENCY)
        self.battery_capacity_j = config_parameters.get("battery_capacity_j", DJI_PHANTOM4_BATTERY_CAPACITY_J)

        # Scaling factors to map simulation units to real-world units
        self.pixels_per_metre = config_parameters.get("pixels_per_metre", DEFAULT_PIXELS_PER_METRE)

        # Turn cost overhead: percentage extra power during turns
        self.turn_cost_overhead_fraction = config_parameters.get("turn_cost_overhead_fraction", 0.15)

        # Communication cost per message (as fraction of hovering power per step)
        self.comm_cost_fraction = config_parameters.get("comm_cost_fraction", 0.001)

        # --- Pre-compute constants that don't change per step ---
        self.total_mass_kg = self.body_mass_kg + self.battery_mass_kg
        self.weight_n = self.total_mass_kg * GRAVITY_M_S2
        self.total_thrust_n = self.weight_n + self.drag_force_n   # Eq. 13

        # Pitch angle from drag (Eq. 14 rearranged: θ = arctan(F_d / weight))
        self.pitch_angle_rad = math.atan2(self.drag_force_n, self.weight_n)

        # Hovering power (Eq. 17) — constant for a given drone
        disc_area = 0.5 * math.pi * (self.propeller_diameter_m ** 2) * self.propeller_count
        self.hovering_power_w = (self.total_thrust_n ** 1.5) / (
            self.power_efficiency * math.sqrt(disc_area * AIR_DENSITY_KG_M3)
        )

        # Flying power (Eqs. 15–16) — computed once for cruise speed
        self.flying_power_w = self._compute_flying_power(self.ground_speed_m_s)

        # Scale to simulation energy units:
        # Map battery_capacity_j -> predator's energy_capacity in config
        # This scale_factor converts Watts·seconds to simulation energy units
        self.energy_capacity_sim = config_parameters.get("energy_capacity_sim", 100.0)
        self.scale_factor = self.energy_capacity_sim / self.battery_capacity_j

    def _compute_induced_velocity(self, forward_speed_m_s: float) -> float:
        """
        Solve Eq. 16 iteratively for induced velocity v_i.

        The equation is implicit (v_i appears on both sides), so we use
        fixed-point iteration. Convergence is fast (usually 5–10 iterations).

        Args:
            forward_speed_m_s: Forward flight speed in m/s.

        Returns:
            Induced velocity in m/s.
        """
        thrust = self.total_thrust_n
        diameter = self.propeller_diameter_m
        rotor_count = self.propeller_count
        rho = AIR_DENSITY_KG_M3
        theta = self.pitch_angle_rad

        v_cos = forward_speed_m_s * math.cos(theta)
        v_sin = forward_speed_m_s * math.sin(theta)

        disc_area_factor = math.pi * (diameter ** 2) * rotor_count * rho

        # Initial guess: hovering induced velocity
        v_i = math.sqrt(thrust / (0.5 * disc_area_factor)) if disc_area_factor > 0 else 1.0

        # Fixed-point iteration (Eq. 16)
        for _ in range(20):
            denominator = disc_area_factor * math.sqrt(v_cos ** 2 + (v_sin + v_i) ** 2)
            if denominator < 1e-12:
                break
            v_i_new = (2.0 * thrust) / denominator
            if abs(v_i_new - v_i) < 1e-9:
                break
            v_i = v_i_new

        return v_i

    def _compute_flying_power(self, forward_speed_m_s: float) -> float:
        """
        Compute flying power using Eq. 15.

        P_fly = (v · sin(θ) + v_i) · T / ε

        Args:
            forward_speed_m_s: Forward flight speed in m/s.

        Returns:
            Flying power in Watts.
        """
        theta = self.pitch_angle_rad
        v_i = self._compute_induced_velocity(forward_speed_m_s)

        climb_component = forward_speed_m_s * math.sin(theta)
        power_w = (climb_component + v_i) * self.total_thrust_n / self.power_efficiency

        return max(0.0, power_w)

    def compute_motion_cost(
        self,
        speed: float,
        turn_rate: float,
        mode: str,
        mass: float = 0.0,
        delta_time: float = DEFAULT_SECONDS_PER_FRAME,
    ) -> float:
        """
        Compute energy consumed by drone movement in one simulation step.

        If the drone is moving (speed > threshold), use flying power.
        If stationary, use hovering power.
        Turning adds an overhead proportional to turn rate.

        Args:
            speed: Speed in pixels/frame.
            turn_rate: Turn rate in radians/frame.
            mode: "search" or "pursue" (affects nothing in this model — power is physics-based).
            mass: Unused (mass is fixed for drones).
            delta_time: Step duration in seconds.

        Returns:
            Energy consumed in simulation units.
        """
        # Convert simulation speed to real-world speed
        # speed is pixels/frame. We need metres/second.
        # speed / pixels_per_metre gives metres/frame
        # (metres/frame) / delta_time gives metres/second
        speed_m_s = speed / self.pixels_per_metre / delta_time if delta_time > 0 else 0.0

        # Choose flying or hovering based on speed
        speed_threshold_m_s = 0.1  # Below this, considered hovering
        if speed_m_s > speed_threshold_m_s:
            power_w = self._compute_flying_power(speed_m_s)
        else:
            power_w = self.hovering_power_w

        # Add turning overhead
        turn_overhead = abs(turn_rate) * self.turn_cost_overhead_fraction * power_w
        total_power_w = power_w + turn_overhead

        # Convert to energy: Power (W) × Time (s) = Energy (J)
        energy_j = total_power_w * delta_time

        # Scale to simulation units
        return energy_j * self.scale_factor

    def compute_idle_cost(self, delta_time: float = DEFAULT_SECONDS_PER_FRAME) -> float:
        """Idle = hovering in place. Drones cannot "turn off" mid-mission."""
        energy_j = self.hovering_power_w * delta_time
        return energy_j * self.scale_factor

    def compute_communication_cost(
        self,
        message_count: int,
        delta_time: float = DEFAULT_SECONDS_PER_FRAME,
    ) -> float:
        """Communication cost as fraction of hovering power."""
        base = self.hovering_power_w * delta_time * self.scale_factor
        return base * self.comm_cost_fraction * message_count

    def get_model_name(self) -> str:
        return "Stolaroff Quadrotor (Qin & Pournaras 2023)"

    def get_model_parameters(self) -> Dict[str, Any]:
        return {
            "body_mass_kg": self.body_mass_kg,
            "battery_mass_kg": self.battery_mass_kg,
            "propeller_diameter_m": self.propeller_diameter_m,
            "propeller_count": self.propeller_count,
            "ground_speed_m_s": self.ground_speed_m_s,
            "drag_force_n": self.drag_force_n,
            "power_efficiency": self.power_efficiency,
            "battery_capacity_j": self.battery_capacity_j,
            "hovering_power_w": self.hovering_power_w,
            "flying_power_w": self.flying_power_w,
            "scale_factor": self.scale_factor,
        }
