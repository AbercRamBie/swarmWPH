# Energy-Aware Swarm Herding Simulation

A simulation framework for benchmarking energy-constrained predator-prey herding across multiple algorithms and energy models.

## Overview

This project implements a predator-prey herding simulation where a swarm of energy-constrained predator agents collaboratively herd prey agents into a goal zone. The system compares 4 herding algorithms against 2 real-world energy models, evaluated through a full-factorial 216-test benchmark suite.

### Herding Algorithms

| Algorithm | Source | Description |
|-----------|--------|-------------|
| **Wolf Pack Formation** | Custom | Decentralized role-based formation with conflict-resolved task assignment |
| **Strombom Shepherding** | Strombom et al. (2014) | Collect-or-drive switching based on flock compactness |
| **Simple APF** | Baseline | Greedy nearest-prey pursuit with artificial potential field repulsion |
| **Wolf + APF** | Sun et al. (2022) | Energy-ranked role hierarchy (alpha/beta/omega) with APF flanking |

### Energy Models

| Model | Source | Description |
|-------|--------|-------------|
| **Stolaroff Drone** | Qin & Pournaras (2023) | Physics-based quadrotor model using momentum theory |
| **TurtleBot3 Empirical** | Mokhtari et al. (2025) | Data-fitted ground robot model with static + actuation power |

## Installation

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip

### Setup

```bash
conda create -n swarm_energy python=3.10 -y
conda activate swarm_energy
pip install -r requirements.txt
```

## Quick Start

### Run with Visualization

```bash
# Default config (Wolf Pack Formation, Stolaroff energy model)
python run_with_visualization.py

# Choose algorithm
python run_with_visualization.py --algorithm strombom --seed 42 --fps 120

# Use TurtleBot3 energy model
python run_with_visualization.py --config config/benchmark_turtlebot3.yaml --algorithm wolf_apf
```

Available algorithms: `wolf_pack_formation`, `strombom`, `simple_apf`, `wolf_apf`

### Run Headless Benchmarks

```bash
# Full 216-test algorithm comparison suite (4 algos x 2 models x 3 pred x 3 prey x 3 seeds)
python run_automated_benchmarks.py --config config/algorithm_benchmark_configs.yaml

# Visual benchmark suite (same tests, with pygame window)
python run_automated_benchmarks_visual.py --config config/algorithm_benchmark_configs.yaml --fps 120

# Single-algorithm visual benchmark
python run_visual_benchmark_suite.py --algorithm wolf_pack_formation --fps 60
```

### Analyze Results

```bash
python analyze_and_plot_benchmarks.py results/algorithm_benchmarks/latest/results.csv \
    --output results/algorithm_benchmarks/latest/plots
```

## Configuration

Edit `config/default_config.yaml` or pass an override file with `--config`.

```yaml
algorithm:
  name: wolf_pack_formation    # wolf_pack_formation | strombom | simple_apf | wolf_apf
  parameters: {}

energy_model:
  name: stolaroff_drone        # stolaroff_drone | turtlebot3_empirical
  parameters: {}

predators:
  count: 10
  speed_max: 3.2
  energy_capacity: 200.0

prey:
  count: 20
  speed_max: 2.0

charging:
  enabled: false
  station_count: 2
  charge_rate: 0.5
```

## Project Structure

```
swarmWPH_WS/
├── config/
│   ├── default_config.yaml              # Full default configuration
│   ├── benchmark_stolaroff.yaml         # Stolaroff drone overrides
│   ├── benchmark_turtlebot3.yaml        # TurtleBot3 overrides
│   ├── algorithm_benchmark_configs.yaml # 216-entry benchmark matrix
│   └── generate_algorithm_benchmarks.py # Script to regenerate the matrix
│
├── src/
│   ├── algorithms/                      # Herding algorithm implementations
│   │   ├── base_algorithm.py            # Abstract interface
│   │   ├── algorithm_factory.py         # Registry + factory
│   │   ├── wolf_pack_formation.py       # Decentralized wolf pack (default)
│   │   ├── strombom_shepherding.py      # Strombom 2014 collect-or-drive
│   │   ├── simple_apf.py               # Greedy APF baseline
│   │   └── wolf_apf.py                 # Role-based APF (Sun 2022)
│   │
│   ├── core/                            # Simulation engine (no pygame dependency)
│   │   ├── simulation.py                # Headless simulation loop + initialization
│   │   ├── predator.py                  # Predator agent with energy + algorithm delegation
│   │   ├── prey.py                      # Prey agent with flocking behavior
│   │   ├── assignment.py                # Decentralized conflict-resolved task assignment
│   │   ├── charging_station.py          # Charging station entities
│   │   └── states.py                    # Behavior mode enums
│   │
│   ├── energy/                          # Energy model implementations
│   │   ├── base_energy_model.py         # Abstract interface
│   │   ├── energy_factory.py            # Registry + factory
│   │   ├── stolaroff_drone.py           # Qin & Pournaras (2023) quadrotor
│   │   └── turtlebot3_empirical.py      # Mokhtari et al. (2025) ground robot
│   │
│   ├── metrics/                         # Metrics collection
│   │   ├── metric_tracker.py            # Real-time metric collection
│   │   ├── episode_logger.py            # CSV logging
│   │   └── metric_definitions.py        # Standard metric formulas
│   │
│   ├── rendering/                       # Visualization (optional, requires pygame)
│   │   └── pygame_renderer.py           # Real-time pygame rendering
│   │
│   └── utils/                           # Utilities
│       ├── config_loader.py             # YAML loading with defaults
│       ├── constants.py                 # Physical constants
│       └── math_helpers.py              # Vector math helpers
│
├── tests/                               # Unit tests (pytest)
│   ├── test_algorithms.py               # Algorithm interface + behavior tests
│   ├── test_assignment.py               # Task assignment tests
│   ├── test_energy_models.py            # Energy model tests
│   └── test_simulation.py               # Integration tests
│
├── results/                             # Benchmark outputs (auto-generated)
│   └── algorithm_benchmarks/            # 216-test suite results
│
├── docs/Instructions/                   # Original assignment materials + references
│
├── run_with_visualization.py            # Single visual simulation run
├── run_automated_benchmarks.py          # Headless benchmark runner
├── run_automated_benchmarks_visual.py   # Visual benchmark runner
├── run_visual_benchmark_suite.py        # Visual suite with algorithm override
├── analyze_and_plot_benchmarks.py       # Plot generation from CSV results
├── requirements.txt
└── README.md
```

## Testing

```bash
pytest tests/ -v
```

43/44 tests pass. One pre-existing flaky test (`test_different_seeds_different_results`) occasionally fails due to both seeds hitting the frame timeout with identical completion counts.

## Benchmark Results (216-Test Suite)

Full results are in `results/algorithm_benchmarks/latest/`. Key findings from the 4-algorithm x 2-model x 9-config x 3-seed benchmark:

| Algorithm | Completion Rate | Energy/Delivery | Timeouts |
|-----------|:--------------:|:--------------:|:--------:|
| Wolf Pack Formation | 99.9% | 1.15 | 0/54 |
| Strombom | 81.4% | 11.35 | 17/54 |
| Simple APF | 96.1% | 2.84 | 4/54 |
| Wolf + APF | 93.6% | 2.81 | 5/54 |

See `results/algorithm_benchmarks/latest/BENCHMARK_INFERENCE_REPORT.md` for the full statistical analysis with Mann-Whitney U significance tests.

## References

1. Strombom et al. (2014). "Solving the shepherding problem: heuristics for herding autonomous, interacting agents." JRSI.
2. Sun et al. (2022). "Multi-robot target encirclement via role assignment." Applied Sciences.
3. Qin & Pournaras (2023). Transportation Research Part C, 157, 104387.
4. Mokhtari et al. (2025). Robotics and Autonomous Systems, 186, 104898.
