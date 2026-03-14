"""
config_loader.py — Safe YAML configuration loading with validation.

Loads a YAML config file, merges it with defaults, and validates
that all required keys exist and have correct types.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


# ----- Default values for every config key -----
# If a key is missing from the user's YAML, these defaults are used.
# This prevents KeyError crashes and documents every available option.

DEFAULTS = {
    "simulation": {
        "fps": 60,
        "max_frames": 30000,
        "arena_width": 1400,
        "arena_height": 900,
        "random_seed": None,         # None = use system random
        "headless": False,           # True = no pygame window
        "goal_zone_size": 180,
        "goal_zone_margin": 20,
    },
    "energy_model": {
        "name": "stolaroff_drone",  # stolaroff_drone | turtlebot3_empirical
        "parameters": {},                  # Model-specific overrides (see individual model files)
    },
    "algorithm": {
        "name": "wolf_pack_formation",  # wolf_pack_formation | strombom | simple_apf | wolf_apf
        "parameters": {},               # Algorithm-specific overrides
    },
    "charging": {
        "enabled": False,
        "station_count": 2,
        "station_positions": "auto",       # "auto" = evenly spaced along bottom edge
        "charge_rate": 0.5,                # energy units restored per frame at station
        "station_radius": 40,              # how close agent must be to charge
    },
    "predators": {
        "count": 10,
        "radius": 11,
        "color": [220, 80, 80],
        "speed_max": 3.2,
        "friction": 0.995,
        "turn_rate_max": 0.10,
        "turn_rate_min": 0.05,
        "reorient_probability": 0.01,
        "herd_distance": 120,
        "arrive_radius": 90,
        "separation_radius": 34,
        "lane_offset": 55,
        "spawn_anchor": [60, -60],         # [x, y] where y=-60 means "60 from bottom"
        "spawn_spread": 40,
        "energy_capacity": 100.0,
        "energy_initial": 100.0,
        "energy_regen_rate": 0.12,
        "engage_minimum_energy": 12.0,
    },
    "prey": {
        "count": 20,
        "radius": 8,
        "color": [100, 150, 255],
        "speed_max": 2.0,
        "avoid_predator_radius": 120,
        "avoid_predator_strength": 2.0,
        "separation_radius": 25,
        "separation_strength": 1.5,
        "boundary_margin": 40,
        "boundary_push_strength": 1.25,
        "predator_collision_buffer": 1.0,
        "wander_jitter": 0.5,
    },
    "communication": {
        "comm_radius": 250,
        "conflict_rounds": 3,
        "cost_per_message": 0.005,
        "cost_idle_per_frame": 0.0,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge 'override' into 'base'.
    Values in 'override' take priority. Keys only in 'base' are kept.

    Args:
        base: The default configuration dictionary.
        override: The user-provided configuration dictionary.

    Returns:
        Merged dictionary.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_existing_path(path_str: str) -> Optional[Path]:
    """
    Resolve a config path robustly across different launch directories.

    Resolution order:
    1) Absolute path (if provided)
    2) Relative to current working directory
    3) Relative to project root (two levels above this file)
    """
    candidate = Path(path_str).expanduser()

    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    project_root = Path(__file__).resolve().parents[2]
    root_candidate = project_root / candidate
    if root_candidate.exists():
        return root_candidate

    return None


def load_config(config_path: str, override_path: Optional[str] = None) -> dict:
    """
    Load a YAML configuration file, merge with defaults, and validate.

    Args:
        config_path: Path to the primary YAML config file.
        override_path: Optional path to a second YAML that overrides values.

    Returns:
        Fully merged and validated configuration dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If a config value has an invalid type.
    """
    resolved_config_path = _resolve_existing_path(config_path)
    if resolved_config_path is None:
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Expected a YAML file. See config/default_config.yaml for an example."
        )

    with open(resolved_config_path, "r", encoding="utf-8") as file:
        user_config = yaml.safe_load(file) or {}

    # Merge: defaults ← user config
    merged = deep_merge(DEFAULTS, user_config)

    # Apply overrides if provided (used by benchmarking to swap models)
    if override_path:
        resolved_override_path = _resolve_existing_path(override_path)
        if resolved_override_path is not None:
            with open(resolved_override_path, "r", encoding="utf-8") as file:
                overrides = yaml.safe_load(file) or {}
            merged = deep_merge(merged, overrides)

    # --- Validation ---
    _validate_positive_int(merged, "predators", "count")
    _validate_positive_int(merged, "prey", "count")
    _validate_positive_float(merged, "predators", "speed_max")
    _validate_positive_float(merged, "predators", "energy_capacity")

    valid_models = ["stolaroff_drone", "turtlebot3_empirical"]
    model_name = merged["energy_model"]["name"]
    if model_name not in valid_models:
        raise ValueError(
            f"Unknown energy model: '{model_name}'\n"
            f"Valid options: {valid_models}"
        )

    valid_algorithms = ["wolf_pack_formation", "strombom", "simple_apf", "wolf_apf"]
    algo_name = merged["algorithm"]["name"]
    if algo_name not in valid_algorithms:
        raise ValueError(
            f"Unknown algorithm: '{algo_name}'\n"
            f"Valid options: {valid_algorithms}"
        )

    return merged


def _validate_positive_int(config: dict, section: str, key: str):
    value = config.get(section, {}).get(key)
    if value is None or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"config['{section}']['{key}'] must be a positive integer, got: {value}")


def _validate_positive_float(config: dict, section: str, key: str):
    value = config.get(section, {}).get(key)
    if value is None or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"config['{section}']['{key}'] must be a positive number, got: {value}")
