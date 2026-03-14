"""
turtlebot3_empirical.py — TurtleBot3 Empirical Energy Model (M2)

Source: Mokhtari et al. (2025), Section 4.3, Equations 27–32
        Robotics and Autonomous Systems, Vol. 186, Article 104898

This model is fitted to real measurements of a TurtleBot3 Burger ground robot.
Unlike physics-based models, this uses regression coefficients derived from
experimental data.

Key equations:
    P_r = P_s + P_act                                    (Eq. 28)
    P_act = P_stdby + Σ(σ_0 + σ_1·θ̇²_wi + σ_2·sgn(θ̇_wi)·θ̇_wi) + σ_3·m  (Eq. 30)
    E_o = ∫(P_s + P_act) dt                              (Eq. 29/32)

Components:
    - P_s: Static power (sensing + computing + communication) ≈ 6.35 W
    - P_act: Actuation power (motors)
    - σ coefficients: Regression parameters fitted to experimental data

Adaptations for simulation:
    - Convert simulation speed (pixels/frame) to wheel angular velocity
    - Use differential drive kinematics (2 wheels)
    - Support both fixed charging stations and solar harvesting
"""

from typing import Dict, Any
import math
from src.energy.base_energy_model import BaseEnergyModel
from src.utils.constants import (
    TURTLEBOT3_STATIC_POWER_W,
    TURTLEBOT3_STATIC_POWER_STD_W,
    TURTLEBOT3_ACTIVE_WHEELS,
    TURTLEBOT3_MOTOR_STANDBY_POWER_W,
    DEFAULT_PIXELS_PER_METRE,
    DEFAULT_SECONDS_PER_FRAME,
)


class TurtleBot3EmpiricalModel(BaseEnergyModel):
    """
    TurtleBot3 empirical energy model.

    Power = static power + actuation power.
    Actuation power depends on wheel speeds (derived from linear + angular velocity).

    Supports:
        - Fixed charging stations
        - Solar harvesting (as constant background charging)
    """

    def __init__(self, config_parameters: Dict[str, Any]):
        super().__init__(config_parameters)

        # --- Static power (sensing + computing + communication) ---
        self.p_static_w = config_parameters.get("p_static_w", TURTLEBOT3_STATIC_POWER_W)

        # --- Motor standby power (motors energized but not turning) ---
        self.p_motor_standby_w = config_parameters.get("p_motor_standby_w", TURTLEBOT3_MOTOR_STANDBY_POWER_W)

        # --- Regression coefficients for actuation power (Eq. 30) ---
        # These are fitted from experimental data (Kunze 2015, adapted by Mokhtari 2025)
        self.sigma_0 = config_parameters.get("sigma_0", 0.25)      # Constant friction per motor
        self.sigma_1 = config_parameters.get("sigma_1", 0.008)     # Speed-dependent loss (∝ ω²)
        self.sigma_2 = config_parameters.get("sigma_2", 0.015)     # Direction-dependent loss (∝ |ω|)
        self.sigma_3 = config_parameters.get("sigma_3", 0.05)      # Payload-dependent loss (∝ mass)

        # --- Robot geometry (differential drive) ---
        self.wheel_radius_m = config_parameters.get("wheel_radius_m", 0.033)  # TurtleBot3 Burger
        self.wheel_base_m = config_parameters.get("wheel_base_m", 0.16)       # Distance between wheels
        self.robot_mass_kg = config_parameters.get("robot_mass_kg", 1.0)      # Base mass (without payload)

        # --- Scaling factors ---
        self.pixels_per_metre = config_parameters.get("pixels_per_metre", DEFAULT_PIXELS_PER_METRE)

        # --- Communication cost per message ---
        self.comm_cost_per_message_w_s = config_parameters.get("comm_cost_per_message_w_s", 0.01)

        # --- Solar harvesting rate (Watts, continuous) ---
        self.solar_harvest_rate_w = config_parameters.get("solar_harvest_rate_w", 0.0)

        # --- Scale factor to convert Joules to simulation energy units ---
        self.energy_capacity_sim = config_parameters.get("energy_capacity_sim", 100.0)
        # Assume a TurtleBot3 battery capacity of ~50 Wh = 180,000 J
        self.battery_capacity_j = config_parameters.get("battery_capacity_j", 180_000)
        self.scale_factor = self.energy_capacity_sim / self.battery_capacity_j

    def _linear_and_angular_to_wheel_speeds(
        self, linear_speed_m_s: float, angular_speed_rad_s: float
    ) -> tuple[float, float]:
        """
        Convert linear and angular velocity to left and right wheel angular velocities.

        Differential drive kinematics:
            v_left  = (linear_speed - angular_speed * wheel_base / 2) / wheel_radius
            v_right = (linear_speed + angular_speed * wheel_base / 2) / wheel_radius

        Args:
            linear_speed_m_s: Forward speed in m/s.
            angular_speed_rad_s: Turning rate in rad/s.

        Returns:
            Tuple (omega_left, omega_right) in rad/s.
        """
        half_base = self.wheel_base_m / 2.0
        v_left_m_s = linear_speed_m_s - angular_speed_rad_s * half_base
        v_right_m_s = linear_speed_m_s + angular_speed_rad_s * half_base

        omega_left = v_left_m_s / self.wheel_radius_m if self.wheel_radius_m > 0 else 0.0
        omega_right = v_right_m_s / self.wheel_radius_m if self.wheel_radius_m > 0 else 0.0

        return omega_left, omega_right

    def _compute_actuation_power(self, omega_left: float, omega_right: float, mass: float) -> float:
        """
        Compute actuation power using Eq. 30.

        P_act = P_stdby + Σ(σ_0 + σ_1·θ̇²_wi + σ_2·sgn(θ̇_wi)·θ̇_wi) + σ_3·m

        Args:
            omega_left: Left wheel angular velocity (rad/s).
            omega_right: Right wheel angular velocity (rad/s).
            mass: Total mass (robot + payload) in kg.

        Returns:
            Actuation power in Watts.
        """
        p_act = self.p_motor_standby_w

        # Left wheel contribution
        p_act += self.sigma_0
        p_act += self.sigma_1 * (omega_left ** 2)
        p_act += self.sigma_2 * math.copysign(1.0, omega_left) * omega_left

        # Right wheel contribution
        p_act += self.sigma_0
        p_act += self.sigma_1 * (omega_right ** 2)
        p_act += self.sigma_2 * math.copysign(1.0, omega_right) * omega_right

        # Payload contribution
        p_act += self.sigma_3 * mass

        return max(0.0, p_act)

    def compute_motion_cost(
        self,
        speed: float,
        turn_rate: float,
        mode: str,
        mass: float = 0.0,
        delta_time: float = DEFAULT_SECONDS_PER_FRAME,
    ) -> float:
        """
        Compute energy consumed by movement.

        Args:
            speed: Linear speed in pixels/frame.
            turn_rate: Angular velocity in radians/frame.
            mode: "search", "pursue", or "idle" (unused in this model).
            mass: Payload mass in kg (added to robot base mass).
            delta_time: Timestep duration in seconds.

        Returns:
            Energy consumed in simulation units.
        """
        # Convert simulation units to physical units
        linear_speed_m_s = speed / self.pixels_per_metre / delta_time if delta_time > 0 else 0.0
        angular_speed_rad_s = turn_rate / delta_time if delta_time > 0 else 0.0

        # Compute wheel speeds
        omega_left, omega_right = self._linear_and_angular_to_wheel_speeds(
            linear_speed_m_s, angular_speed_rad_s
        )

        # Compute power
        total_mass = self.robot_mass_kg + mass
        p_actuation = self._compute_actuation_power(omega_left, omega_right, total_mass)
        p_total = self.p_static_w + p_actuation

        # Convert to energy: Power × Time
        energy_j = p_total * delta_time

        # Scale to simulation units
        return energy_j * self.scale_factor

    def compute_idle_cost(self, delta_time: float = DEFAULT_SECONDS_PER_FRAME) -> float:
        """
        Energy consumed while stationary.

        When idle, only static power and motor standby power are consumed.

        Args:
            delta_time: Timestep duration in seconds.

        Returns:
            Energy consumed in simulation units.
        """
        p_total = self.p_static_w + self.p_motor_standby_w
        energy_j = p_total * delta_time
        return energy_j * self.scale_factor

    def compute_communication_cost(
        self,
        message_count: int,
        delta_time: float = DEFAULT_SECONDS_PER_FRAME,
    ) -> float:
        """
        Energy consumed by communication.

        Args:
            message_count: Number of messages sent/received.
            delta_time: Timestep duration in seconds.

        Returns:
            Total communication cost in simulation units.
        """
        energy_j = self.comm_cost_per_message_w_s * message_count * delta_time
        return energy_j * self.scale_factor

    def compute_charging_gain(
        self,
        is_at_station: bool,
        charge_rate: float,
        delta_time: float = DEFAULT_SECONDS_PER_FRAME,
    ) -> float:
        """
        Energy gained from charging or solar harvesting.

        Args:
            is_at_station: Whether at a charging station.
            charge_rate: Charging rate in simulation units per frame (if at station).
            delta_time: Timestep duration in seconds.

        Returns:
            Energy gained in simulation units.
        """
        if is_at_station:
            # Use provided charge_rate (already in simulation units)
            return charge_rate
        else:
            # Solar harvesting (continuous background charging)
            energy_j = self.solar_harvest_rate_w * delta_time
            return energy_j * self.scale_factor

    def get_model_name(self) -> str:
        return "TurtleBot3 Empirical (Mokhtari et al. 2025)"

    def get_model_parameters(self) -> Dict[str, Any]:
        return {
            "p_static_w": self.p_static_w,
            "p_motor_standby_w": self.p_motor_standby_w,
            "sigma_0": self.sigma_0,
            "sigma_1": self.sigma_1,
            "sigma_2": self.sigma_2,
            "sigma_3": self.sigma_3,
            "wheel_radius_m": self.wheel_radius_m,
            "wheel_base_m": self.wheel_base_m,
            "robot_mass_kg": self.robot_mass_kg,
            "solar_harvest_rate_w": self.solar_harvest_rate_w,
            "scale_factor": self.scale_factor,
        }
