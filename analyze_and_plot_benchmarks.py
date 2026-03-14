#!/usr/bin/env python3
"""
analyze_and_plot_benchmarks.py — Generate plots from algorithm benchmark results.

Creates comprehensive visualizations comparing herding algorithms:
1. Algorithm Comparison: completion rate, energy/delivery, makespan
2. Algorithm x Model Heatmap: mean performance across all combinations
3. Scalability Analysis: performance vs swarm size per algorithm
4. Seed Variance: box plots of energy spread per algorithm
5. Summary Table: mean +/- std for all metrics

Usage:
    python3 analyze_and_plot_benchmarks.py results.csv
    python3 analyze_and_plot_benchmarks.py results.csv --output plots/
"""

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Algorithm display config
ALGO_COLORS = {
    'Wolf Pack Formation': '#E74C3C',
    'Strombom Shepherding': '#3498DB',
    'Simple APF': '#2ECC71',
    'Wolf+APF': '#9B59B6',
}
ALGO_SHORT = {
    'Wolf Pack Formation': 'WPF',
    'Strombom Shepherding': 'Strombom',
    'Simple APF': 'APF',
    'Wolf+APF': 'W+APF',
}


def load_benchmark_data(csv_path: str) -> pd.DataFrame:
    """Load benchmark results from CSV file."""
    df = pd.read_csv(csv_path)

    # Normalize energy model names
    def normalize_model_name(name):
        if pd.isna(name):
            return name
        name_lower = str(name).lower()
        if 'stolaroff' in name_lower or 'quadrotor' in name_lower:
            return 'Stolaroff'
        elif 'turtlebot' in name_lower:
            return 'TurtleBot3'
        return name

    df['energy_model'] = df['energy_model'].apply(normalize_model_name)

    # Handle inf values in energy_per_delivery
    df['energy_per_delivery'] = df['energy_per_delivery'].replace([np.inf, -np.inf], np.nan)

    print(f"Loaded {len(df)} benchmark results from {csv_path}")
    if 'algorithm_name' in df.columns:
        algos = df['algorithm_name'].unique()
        print(f"Algorithms: {', '.join(algos)}")
    models = df['energy_model'].unique()
    print(f"Energy models: {', '.join(models)}")
    return df


def _get_color(algo_name):
    return ALGO_COLORS.get(algo_name, '#95A5A6')


def _get_short(algo_name):
    return ALGO_SHORT.get(algo_name, algo_name)


def plot_algorithm_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Plot 1: Algorithm Performance Comparison
    Bar charts comparing completion rate, energy/delivery, and makespan.
    """
    if 'algorithm_name' not in df.columns:
        print("Skipping algorithm comparison (no algorithm_name column)")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    algos = df['algorithm_name'].unique()
    algo_labels = [_get_short(a) for a in algos]
    colors = [_get_color(a) for a in algos]

    # 1a: Completion Rate
    means = df.groupby('algorithm_name')['completion_rate'].mean()
    stds = df.groupby('algorithm_name')['completion_rate'].std()
    vals = [means.get(a, 0) * 100 for a in algos]
    errs = [stds.get(a, 0) * 100 for a in algos]

    bars = ax1.bar(algo_labels, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5,
                   yerr=errs, capsize=5)
    ax1.set_ylabel('Completion Rate (%)', fontweight='bold')
    ax1.set_title('Task Completion Rate', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 110)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 1b: Energy per Delivery
    valid = df.dropna(subset=['energy_per_delivery'])
    means = valid.groupby('algorithm_name')['energy_per_delivery'].mean()
    stds = valid.groupby('algorithm_name')['energy_per_delivery'].std()
    vals = [means.get(a, 0) for a in algos]
    errs = [stds.get(a, 0) for a in algos]

    bars = ax2.bar(algo_labels, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5,
                   yerr=errs, capsize=5)
    ax2.set_ylabel('Energy per Delivery', fontweight='bold')
    ax2.set_title('Energy Cost per Delivery', fontsize=13, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 1c: Makespan (frames)
    means = df.groupby('algorithm_name')['frames'].mean()
    stds = df.groupby('algorithm_name')['frames'].std()
    vals = [means.get(a, 0) for a in algos]
    errs = [stds.get(a, 0) for a in algos]

    bars = ax3.bar(algo_labels, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5,
                   yerr=errs, capsize=5)
    ax3.set_ylabel('Frames', fontweight='bold')
    ax3.set_title('Average Makespan', fontsize=13, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{v:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Algorithm Performance Comparison (All Configurations)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / '01_algorithm_comparison.eps'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_algorithm_model_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Plot 2: Algorithm x Energy Model Heatmap
    Shows mean completion rate and energy/delivery for each combination.
    """
    if 'algorithm_name' not in df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # 2a: Completion Rate heatmap
    pivot_cr = df.pivot_table(
        values='completion_rate', index='algorithm_name', columns='energy_model', aggfunc='mean'
    ) * 100
    pivot_cr.index = [_get_short(a) for a in pivot_cr.index]

    sns.heatmap(pivot_cr, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax1,
                vmin=0, vmax=100, cbar_kws={'label': 'Completion Rate (%)'}, linewidths=1)
    ax1.set_title('Completion Rate (%) by Algorithm x Model', fontsize=24, fontweight='bold')
    ax1.set_ylabel('Algorithm', fontsize=20)
    ax1.set_xlabel('Energy Model', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    # 2b: Energy per Delivery heatmap
    valid = df.dropna(subset=['energy_per_delivery'])
    pivot_epd = valid.pivot_table(
        values='energy_per_delivery', index='algorithm_name', columns='energy_model', aggfunc='mean'
    )
    pivot_epd.index = [_get_short(a) for a in pivot_epd.index]

    sns.heatmap(pivot_epd, annot=True, fmt='.2f', cmap='YlGnBu_r', ax=ax2,
                cbar_kws={'label': 'Energy / Delivery'}, linewidths=1)
    ax2.set_title('Energy per Delivery by Algorithm x Model', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Algorithm')
    ax2.set_xlabel('Energy Model')

    plt.suptitle('Algorithm x Energy Model Performance Matrix', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / '02_algorithm_model_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_scalability_by_algorithm(df: pd.DataFrame, output_dir: Path):
    """
    Plot 3: Scalability Analysis by Algorithm
    Energy/delivery vs predator count, one line per algorithm.
    """
    if 'algorithm_name' not in df.columns:
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    algos = df['algorithm_name'].unique()

    # 3a: Energy/delivery vs predator count
    for algo in algos:
        algo_df = df[df['algorithm_name'] == algo].dropna(subset=['energy_per_delivery'])
        if algo_df.empty:
            continue
        grouped = algo_df.groupby('predator_count')['energy_per_delivery'].agg(['mean', 'std']).reset_index()
        ax1.errorbar(grouped['predator_count'], grouped['mean'], yerr=grouped['std'],
                    marker='o', linewidth=2, markersize=8, capsize=4,
                    label=_get_short(algo), color=_get_color(algo))

    ax1.set_xlabel('Predator Count', fontweight='bold')
    ax1.set_ylabel('Energy per Delivery', fontweight='bold')
    ax1.set_title('Energy Efficiency vs Swarm Size', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 3b: Completion rate vs predator count
    for algo in algos:
        algo_df = df[df['algorithm_name'] == algo]
        grouped = algo_df.groupby('predator_count')['completion_rate'].agg(['mean', 'std']).reset_index()
        ax2.errorbar(grouped['predator_count'], grouped['mean'] * 100, yerr=grouped['std'] * 100,
                    marker='o', linewidth=2, markersize=8, capsize=4,
                    label=_get_short(algo), color=_get_color(algo))

    ax2.set_xlabel('Predator Count', fontweight='bold')
    ax2.set_ylabel('Completion Rate (%)', fontweight='bold')
    ax2.set_title('Task Completion vs Swarm Size', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3c: Makespan vs prey count
    for algo in algos:
        algo_df = df[df['algorithm_name'] == algo]
        grouped = algo_df.groupby('prey_count')['frames'].agg(['mean', 'std']).reset_index()
        ax3.errorbar(grouped['prey_count'], grouped['mean'], yerr=grouped['std'],
                    marker='s', linewidth=2, markersize=8, capsize=4,
                    label=_get_short(algo), color=_get_color(algo))

    ax3.set_xlabel('Prey Count', fontweight='bold')
    ax3.set_ylabel('Frames (Makespan)', fontweight='bold')
    ax3.set_title('Makespan vs Task Size', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 3d: Duty cycle vs predator count
    if 'avg_duty_cycle' in df.columns:
        for algo in algos:
            algo_df = df[df['algorithm_name'] == algo]
            grouped = algo_df.groupby('predator_count')['avg_duty_cycle'].agg(['mean', 'std']).reset_index()
            ax4.errorbar(grouped['predator_count'], grouped['mean'], yerr=grouped['std'],
                        marker='D', linewidth=2, markersize=8, capsize=4,
                        label=_get_short(algo), color=_get_color(algo))

        ax4.set_xlabel('Predator Count', fontweight='bold')
        ax4.set_ylabel('Duty Cycle', fontweight='bold')
        ax4.set_title('Duty Cycle vs Swarm Size', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No duty cycle data', transform=ax4.transAxes,
                ha='center', va='center', fontsize=14)

    plt.suptitle('Scalability Analysis by Algorithm', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / '03_scalability_by_algorithm.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_seed_variance(df: pd.DataFrame, output_dir: Path):
    """
    Plot 4: Seed Variance Analysis
    Box plots showing energy spread per algorithm across seeds.
    """
    if 'algorithm_name' not in df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    algos = df['algorithm_name'].unique()
    algo_labels = [_get_short(a) for a in algos]
    palette = {_get_short(a): _get_color(a) for a in algos}

    # 4a: Energy per delivery box plot
    valid = df.dropna(subset=['energy_per_delivery']).copy()
    valid['algo_short'] = valid['algorithm_name'].apply(_get_short)

    sns.boxplot(data=valid, x='algo_short', y='energy_per_delivery', palette=palette,
                ax=ax1, order=algo_labels)
    ax1.set_xlabel('Algorithm', fontweight='bold')
    ax1.set_ylabel('Energy per Delivery', fontweight='bold')
    ax1.set_title('Energy Cost Variance Across Seeds', fontsize=13, fontweight='bold')

    # 4b: Completion rate box plot
    df_copy = df.copy()
    df_copy['algo_short'] = df_copy['algorithm_name'].apply(_get_short)

    sns.boxplot(data=df_copy, x='algo_short', y='completion_rate', palette=palette,
                ax=ax2, order=algo_labels)
    ax2.set_xlabel('Algorithm', fontweight='bold')
    ax2.set_ylabel('Completion Rate', fontweight='bold')
    ax2.set_title('Completion Rate Variance Across Seeds', fontsize=13, fontweight='bold')

    plt.suptitle('Seed Variance Analysis (Robustness)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / '04_seed_variance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_table(df: pd.DataFrame, output_dir: Path):
    """
    Plot 5: Summary Statistics Table
    Mean +/- std for all metrics, all algorithms, both models.
    """
    if 'algorithm_name' not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('tight')
    ax.axis('off')

    algos = sorted(df['algorithm_name'].unique())
    models = sorted(df['energy_model'].unique())

    # Build table data
    header = ['Algorithm', 'Model', 'Completion %', 'Energy/Del', 'Makespan', 'Duty Cycle', 'Tests']
    rows = []

    for algo in algos:
        for model in models:
            subset = df[(df['algorithm_name'] == algo) & (df['energy_model'] == model)]
            if subset.empty:
                continue

            cr = subset['completion_rate']
            valid_epd = subset['energy_per_delivery'].dropna()
            frames = subset['frames']
            dc = subset['avg_duty_cycle'] if 'avg_duty_cycle' in subset.columns else pd.Series([0])

            row = [
                _get_short(algo),
                model,
                f'{cr.mean()*100:.1f} +/- {cr.std()*100:.1f}',
                f'{valid_epd.mean():.2f} +/- {valid_epd.std():.2f}' if len(valid_epd) > 0 else 'N/A',
                f'{frames.mean():.0f} +/- {frames.std():.0f}',
                f'{dc.mean():.3f} +/- {dc.std():.3f}',
                str(len(subset)),
            ]
            rows.append(row)

    table_data = [header] + rows
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.10, 0.18, 0.18, 0.18, 0.14, 0.06])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(header)):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(weight='bold', color='white')

    # Alternate row colors by algorithm
    algo_colors_light = {
        'WPF': '#FADBD8',
        'Strombom': '#D6EAF8',
        'APF': '#D5F5E3',
        'W+APF': '#E8DAEF',
    }
    for i, row in enumerate(rows, 1):
        color = algo_colors_light.get(row[0], '#F0F0F0')
        for j in range(len(header)):
            table[(i, j)].set_facecolor(color)

    ax.set_title('Performance Summary: Mean +/- Std Across All Configurations',
                fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = output_dir / '05_summary_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_model_comparison_per_algorithm(df: pd.DataFrame, output_dir: Path):
    """
    Plot 6: M1 vs M2 comparison within each algorithm.
    Grouped bar chart showing how energy model affects each algorithm.
    """
    if 'algorithm_name' not in df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    algos = sorted(df['algorithm_name'].unique())
    models = sorted(df['energy_model'].unique())
    x = np.arange(len(algos))
    width = 0.35

    model_colors = {'Stolaroff': '#E74C3C', 'TurtleBot3': '#3498DB'}

    # 6a: Completion Rate
    for i, model in enumerate(models):
        vals = []
        for algo in algos:
            subset = df[(df['algorithm_name'] == algo) & (df['energy_model'] == model)]
            vals.append(subset['completion_rate'].mean() * 100 if len(subset) > 0 else 0)
        offset = (i - 0.5) * width
        bars = ax1.bar(x + offset, vals, width, label=model,
                      color=model_colors.get(model, '#95A5A6'), alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{v:.0f}%', ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('Algorithm', fontweight='bold')
    ax1.set_ylabel('Completion Rate (%)', fontweight='bold')
    ax1.set_title('Completion Rate: Stolaroff vs TurtleBot3', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([_get_short(a) for a in algos])
    ax1.legend()
    ax1.set_ylim(0, 110)

    # 6b: Energy per Delivery
    for i, model in enumerate(models):
        vals = []
        for algo in algos:
            subset = df[(df['algorithm_name'] == algo) & (df['energy_model'] == model)]
            valid = subset['energy_per_delivery'].dropna()
            vals.append(valid.mean() if len(valid) > 0 else 0)
        offset = (i - 0.5) * width
        bars = ax2.bar(x + offset, vals, width, label=model,
                      color=model_colors.get(model, '#95A5A6'), alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel('Algorithm', fontweight='bold')
    ax2.set_ylabel('Energy per Delivery', fontweight='bold')
    ax2.set_title('Energy Cost: Stolaroff vs TurtleBot3', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([_get_short(a) for a in algos])
    ax2.legend()

    plt.suptitle('Energy Model Impact on Algorithm Performance', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / '06_model_comparison_per_algorithm.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_all_plots(csv_path: str, output_dir: str):
    """Generate all analysis plots."""
    df = load_benchmark_data(csv_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("GENERATING ANALYSIS PLOTS")
    print(f"{'='*70}")
    print(f"Input: {csv_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    plot_algorithm_comparison(df, output_path)
    plot_algorithm_model_heatmap(df, output_path)
    plot_scalability_by_algorithm(df, output_path)
    plot_seed_variance(df, output_path)
    plot_summary_table(df, output_path)
    plot_model_comparison_per_algorithm(df, output_path)

    print(f"\n{'='*70}")
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"Location: {output_dir}")
    print(f"Total plots: 6")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis plots from algorithm benchmark results"
    )
    parser.add_argument(
        "csv_file", type=str,
        help="Path to benchmark results CSV file"
    )
    parser.add_argument(
        "--output", type=str,
        default="results/automated_benchmarks/plots",
        help="Output directory for plots"
    )

    args = parser.parse_args()
    generate_all_plots(args.csv_file, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
