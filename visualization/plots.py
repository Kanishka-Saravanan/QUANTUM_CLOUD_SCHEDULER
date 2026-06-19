"""
Plots - Visualization of Experiment Results

Generates comparison graphs for cost, makespan, SLA violations,
and resource utilization. Saves to results folder.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def _ensure_results_dir(results_dir: str = "results") -> Path:
    """Ensure results directory exists."""
    p = Path(results_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _extract_plot_data(
    summary: Dict[int, Dict[str, Dict[str, float]]],
    metric_key: str,
) -> tuple:
    """
    Extract data for bar charts: sizes, algorithms, and matrix of values.
    """
    sizes = sorted(summary.keys())
    algorithms = list(next(iter(summary.values())).keys())
    data = np.array([
        [summary[s][a].get(metric_key, 0) for a in algorithms]
        for s in sizes
    ]).T
    return sizes, algorithms, data


def plot_cost_comparison(
    summary: Dict[int, Dict[str, Dict[str, float]]],
    results_dir: str = "results",
) -> str:
    """
    Generate cost comparison bar chart.

    Returns:
        Path to saved figure.
    """
    sizes, algorithms, data = _extract_plot_data(summary, "cost")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sizes))
    width = 0.25
    for i, alg in enumerate(algorithms):
        offset = (i - len(algorithms) / 2 + 0.5) * width
        ax.bar(x + offset, data[i], width, label=alg)
    ax.set_xlabel("Workload Size (tasks)")
    ax.set_ylabel("Total Cost")
    ax.set_title("Scheduling Cost Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = _ensure_results_dir(results_dir) / "cost_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_makespan_comparison(
    summary: Dict[int, Dict[str, Dict[str, float]]],
    results_dir: str = "results",
) -> str:
    """
    Generate makespan comparison bar chart.
    """
    sizes, algorithms, data = _extract_plot_data(summary, "makespan")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sizes))
    width = 0.25
    for i, alg in enumerate(algorithms):
        offset = (i - len(algorithms) / 2 + 0.5) * width
        ax.bar(x + offset, data[i], width, label=alg)
    ax.set_xlabel("Workload Size (tasks)")
    ax.set_ylabel("Makespan")
    ax.set_title("Makespan Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = _ensure_results_dir(results_dir) / "makespan_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_sla_violations(
    summary: Dict[int, Dict[str, Dict[str, float]]],
    results_dir: str = "results",
) -> str:
    """
    Generate SLA violations comparison bar chart.
    """
    sizes, algorithms, data = _extract_plot_data(summary, "sla_violations")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sizes))
    width = 0.25
    for i, alg in enumerate(algorithms):
        offset = (i - len(algorithms) / 2 + 0.5) * width
        ax.bar(x + offset, data[i], width, label=alg)
    ax.set_xlabel("Workload Size (tasks)")
    ax.set_ylabel("SLA Violations")
    ax.set_title("SLA Violations Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = _ensure_results_dir(results_dir) / "sla_violations.png"
    fig.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_resource_utilization(
    summary: Dict[int, Dict[str, Dict[str, float]]],
    results_dir: str = "results",
) -> str:
    """
    Generate resource utilization comparison bar chart.
    """
    sizes, algorithms, data = _extract_plot_data(summary, "mean_utilization")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sizes))
    width = 0.25
    for i, alg in enumerate(algorithms):
        offset = (i - len(algorithms) / 2 + 0.5) * width
        ax.bar(x + offset, data[i], width, label=alg)
    ax.set_xlabel("Workload Size (tasks)")
    ax.set_ylabel("Mean Resource Utilization")
    ax.set_title("Resource Utilization Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = _ensure_results_dir(results_dir) / "resource_utilization.png"
    fig.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_resource_imbalance(
    summary: Dict[int, Dict[str, Dict[str, float]]],
    results_dir: str = "results",
) -> str:
    """
    Generate resource imbalance comparison (lower is better).
    """
    sizes, algorithms, data = _extract_plot_data(summary, "resource_imbalance")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sizes))
    width = 0.25
    for i, alg in enumerate(algorithms):
        offset = (i - len(algorithms) / 2 + 0.5) * width
        ax.bar(x + offset, data[i], width, label=alg)
    ax.set_xlabel("Workload Size (tasks)")
    ax.set_ylabel("Resource Imbalance")
    ax.set_title("Resource Imbalance (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = _ensure_results_dir(results_dir) / "resource_imbalance.png"
    fig.savefig(path, dpi=150)
    plt.close()
    return str(path)


def generate_all_plots(
    summary: Dict[int, Dict[str, Dict[str, float]]],
    results_dir: str = "results",
) -> List[str]:
    """
    Generate all comparison plots and save to results directory.

    Args:
        summary: Output of ExperimentRunner.get_summary_for_dashboard().
        results_dir: Directory for saved figures.

    Returns:
        List of paths to saved figures.
    """
    paths = [
        plot_cost_comparison(summary, results_dir),
        plot_makespan_comparison(summary, results_dir),
        plot_sla_violations(summary, results_dir),
        plot_resource_utilization(summary, results_dir),
        plot_resource_imbalance(summary, results_dir),
    ]
    return paths
