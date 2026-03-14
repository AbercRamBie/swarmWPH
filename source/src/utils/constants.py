"""
constants.py — Physical and simulation constants.

All physical constants used by the energy models live here.
This ensures consistency across models and makes it easy to
find and verify values.
"""

# ---- Gravitational acceleration ----
GRAVITY_M_S2 = 9.81  # metres per second squared

# ---- Air density at sea level, 15°C ----
AIR_DENSITY_KG_M3 = 1.225  # kilograms per cubic metre

# ---- DJI Phantom 4 Pro parameters (Qin & Pournaras 2023, Table 3) ----
# Used by the Stolaroff drone energy model
DJI_PHANTOM4_BODY_MASS_KG = 1.07
DJI_PHANTOM4_BATTERY_MASS_KG = 0.31
DJI_PHANTOM4_PROPELLER_DIAMETER_M = 0.35
DJI_PHANTOM4_PROPELLER_COUNT = 4
DJI_PHANTOM4_GROUND_SPEED_M_S = 6.94
DJI_PHANTOM4_DRAG_FORCE_N = 4.1134
DJI_PHANTOM4_POWER_EFFICIENCY = 0.8
DJI_PHANTOM4_BATTERY_CAPACITY_J = 275_000  # 275 kJ

# ---- TurtleBot3 Burger parameters (Mokhtari et al. 2025, Section 4.3) ----
# Used by the TurtleBot3 empirical energy model
TURTLEBOT3_STATIC_POWER_W = 6.35         # Gaussian mean from Fig. 5
TURTLEBOT3_STATIC_POWER_STD_W = 0.34     # Gaussian std dev
TURTLEBOT3_ACTIVE_WHEELS = 2
TURTLEBOT3_MOTOR_STANDBY_POWER_W = 0.5   # Estimated from experimental data

# ---- Simulation-to-real-world scaling ----
# Our simulation uses pixels and frames. These constants map to physical units.
# Adjustable via config, but these are sensible defaults.
DEFAULT_PIXELS_PER_METRE = 100.0          # 100 pixels = 1 metre
DEFAULT_SECONDS_PER_FRAME = 1.0 / 60.0   # at 60 FPS, each frame = 0.0167 seconds
