
# Quantum-Enhanced Cloud Task Scheduling using Hybrid Optimization

A research-quality Python prototype that simulates cloud resource scheduling and compares classical scheduling algorithms (FIFO, Round Robin) with a quantum-inspired optimization approach using QUBO (Quadratic Unconstrained Binary Optimization) and QAOA (Quantum Approximate Optimization Algorithm).

## Project Overview

This project addresses the challenge of efficient task-to-VM assignment in cloud environments. The goal is to reduce:

- **Scheduling Cost** — Total cost of running tasks on VMs
- **SLA Violations** — Tasks missing their deadlines
- **Makespan** — Total time to complete all tasks
- **Resource Imbalance** — Uneven utilization across VMs

By formulating the scheduling problem as a QUBO and solving it with QAOA (or a classical QUBO solver when the problem is large), we explore whether quantum-inspired optimization can improve upon simple classical heuristics.

## Architecture

```
quantum_cloud_scheduler
│
├── core/                     # Core simulation components
│   ├── scheduler_engine.py   # Orchestrates scheduling decisions
│   ├── resource_manager.py   # VM pool and capacity modeling
│   └── workload_manager.py   # Synthetic workload generation
│
├── algorithms/
│   ├── classical/            # Classical schedulers
│   │   ├── fifo.py
│   │   └── round_robin.py
│   └── quantum/              # Quantum-inspired optimization
│       ├── qubo_builder.py   # QUBO formulation of scheduling
│       └── qaoa_optimizer.py # QAOA-based optimizer
│
├── metrics/                  # Evaluation metrics
│   ├── cost_metrics.py
│   ├── performance_metrics.py
│   └── sla_metrics.py
│
├── visualization/            # Plots and dashboard
│   ├── plots.py
│   └── dashboard.py
│
├── experiments/
│   └── experiment_runner.py  # Batch experiment execution
│
├── datasets/                 # (Optional) input datasets
├── results/                  # Output: JSON results + plots
└── main.py                   # Entry point
```

## Algorithms

### Classical

- **FIFO (First-In-First-Out)**: Tasks are ordered by arrival time. Each task is assigned to the VM that minimizes execution cost (greedy cost-based placement).
- **Round Robin**: Tasks are distributed cyclically across VMs to balance load, with assignments ordered by arrival time.

### Quantum-Inspired

- **QUBO Formulation**: The scheduling problem is mapped to binary variables \(x_{ij} = 1\) if task \(i\) is assigned to VM \(j\). The objective combines:
  - Total scheduling cost
  - Resource load balance (quadratic terms)
  - SLA violation penalties
- **QAOA Optimizer**: Formulates scheduling as QUBO and solves it using Qiskit. For tractability, large problems are reduced to a small subset of tasks/VMs. The solver uses `NumPyMinimumEigensolver` (exact classical) for reliability; QAOA can be enabled for small instances when a quantum simulator is available.

## Experimental Setup

- **Workload Sizes**: 50, 100, 200, 500 tasks
- **VM Pool**: 8 heterogeneous VMs (variable CPU, cost, energy)
- **Metrics**: Cost, Makespan, SLA Violations, Resource Utilization, Resource Imbalance
- **Random Seed**: 42 (reproducible experiments)

## Installation

```bash
cd quantum_cloud_scheduler
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- qiskit
- qiskit-aer
- qiskit-algorithms
- qiskit-optimization
- numpy
- pandas
- matplotlib

## Running the Project

From the project root:

```bash
python main.py
```

This will:

1. Run experiments for workload sizes 50, 100, 200, 500
2. Compare FIFO, Round Robin, and Quantum Optimized scheduling
3. Save results to `results/experiment_results.json`
4. Generate comparison plots in `results/`:
   - `cost_comparison.png`
   - `makespan_comparison.png`
   - `sla_violations.png`
   - `resource_utilization.png`
   - `resource_imbalance.png`
5. Print a formatted dashboard to the console

## Results

Results are written to `results/experiment_results.json` in structured form. Each experiment includes:

- Workload size and number of VMs
- Per-algorithm metrics: cost, makespan, SLA violations, utilization, imbalance
- Timestamps for reproducibility

Plots are saved as PNG files in the `results/` directory for visual comparison.



