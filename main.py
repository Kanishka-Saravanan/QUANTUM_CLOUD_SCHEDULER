#!/usr/bin/env python3
"""
Quantum-Enhanced Cloud Task Scheduling using Hybrid Optimization
Main entry point for experiments and visualization
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_runner import ExperimentRunner
from experiments.research_analysis import print_research_summary
from visualization.plots import generate_all_plots


def main() -> int:


    print("\n============================================================")
    print("  QUANTUM + ML CLOUD TASK SCHEDULING SYSTEM")
    print("============================================================\n")

    runner = ExperimentRunner(
        results_dir=str(PROJECT_ROOT / "results"),
        seed=42
    )

    print("Running experiments (50, 100, 200, 500 tasks)...\n")

    all_results = runner.run_all()

    out_path = runner.save_results(all_results)

    print(f"Results saved to: {out_path}")

    summary = runner.get_summary_for_dashboard(all_results)

    print("\nGenerating plots...\n")

    plot_paths = generate_all_plots(
        summary,
        results_dir=str(PROJECT_ROOT / "results")
    )

    for p in plot_paths:
        print(f"Saved: {p}")

    print_research_summary(summary)

    print("\nExperiment pipeline complete.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())