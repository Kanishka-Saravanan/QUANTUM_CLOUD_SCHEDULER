"""
Experiment Runner - Batch Experiments for Algorithm Comparison

Runs scheduling experiments across workload sizes (50, 100, 200, 500 tasks),
compares FIFO, Round Robin, ML Scheduler, and Quantum Optimization.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from ml.ml_scheduler import train_ml_model, ml_schedule

from core.workload_manager import WorkloadManager
from core.resource_manager import ResourceManager
from core.scheduler_engine import SchedulerEngine, SchedulingResult

from algorithms.classical.fifo import FIFOScheduler
from algorithms.classical.round_robin import RoundRobinScheduler
from algorithms.quantum.qaoa_optimizer import schedule_quantum

from metrics.cost_metrics import CostMetrics
from metrics.performance_metrics import PerformanceMetrics
from metrics.sla_metrics import SLAMetrics


WORKLOAD_SIZES = [50, 100, 200, 500]

# Only classical + quantum here
ALGORITHMS = [
    ("FIFO", lambda t, v, **kw: FIFOScheduler.schedule(t, v, **kw)),
    ("Round Robin", lambda t, v, **kw: RoundRobinScheduler.schedule(t, v, **kw)),
    ("Quantum Optimized", lambda t, v, **kw: schedule_quantum(t, v, **kw)),
]


class ExperimentRunner:
    def __init__(self, results_dir: str = "results", seed: int = 42):

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        self.workload_mgr = WorkloadManager(seed=seed)
        self.resource_mgr = ResourceManager(seed=seed + 1)

    def run_single(self, num_tasks: int, num_vms: int = 8):

        # Generate workload
        tasks = self.workload_mgr.generate_workload(num_tasks)
        vms = self.resource_mgr.create_default_pool(num_vms=num_vms)

        # Metrics
        cost_metrics = CostMetrics(tasks, vms)
        perf_metrics = PerformanceMetrics(tasks, vms)
        sla_metrics = SLAMetrics(tasks, vms)

        engine = SchedulerEngine(tasks, vms)
        results = {}

        # -----------------------------
        # Run Classical + Quantum
        # -----------------------------
        for name, scheduler_fn in ALGORITHMS:
            result = engine.run(scheduler_fn, algorithm_name=name)

            results[name] = {
                "cost": cost_metrics.compute(result),
                "performance": perf_metrics.compute(result),
                "sla": sla_metrics.compute(result),
            }

        # -----------------------------
        # ML Scheduler (SAFE)
        # -----------------------------
        try:
            ml_model = train_ml_model(tasks, vms)
            ml_assignments = ml_schedule(tasks, vms, ml_model)

            # Convert to SchedulingResult
            ml_result = SchedulingResult(
                algorithm_name="ML Scheduler",
                assignments=ml_assignments,
                metadata={
                    "num_tasks": len(tasks),
                    "num_vms": len(vms)
                }
            )

        except Exception as e:
            print(f"[WARNING] ML Scheduler failed: {e}")

            # fallback to FIFO
            fallback = engine.run(
                lambda t, v, **kw: FIFOScheduler.schedule(t, v, **kw),
                algorithm_name="FIFO"
            )
            ml_result = fallback

        results["ML Scheduler"] = {
            "cost": cost_metrics.compute(ml_result),
            "performance": perf_metrics.compute(ml_result),
            "sla": sla_metrics.compute(ml_result),
        }

        return {
            "workload_size": num_tasks,
            "num_vms": num_vms,
            "algorithm_results": results,
            "timestamp": datetime.now().isoformat(),
        }

    def run_all(self) -> List[Dict[str, Any]]:

        all_results = []

        for size in WORKLOAD_SIZES:
            exp = self.run_single(num_tasks=size)
            all_results.append(exp)

        return all_results

    def save_results(self, all_results):

        def to_serializable(obj):
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_serializable(x) for x in obj]
            if hasattr(obj, "item"):
                return obj.item()
            return obj

        path = self.results_dir / "experiment_results.json"

        with open(path, "w") as f:
            json.dump(to_serializable(all_results), f, indent=2)

        return str(path)

    def get_summary_for_dashboard(self, all_results):

        summary = {}

        for exp in all_results:
            size = exp["workload_size"]
            summary[size] = {}

            for alg_name, alg_data in exp["algorithm_results"].items():

                summary[size][alg_name] = {
                    "cost": alg_data["cost"]["total_cost"],
                    "makespan": alg_data["performance"]["makespan"],
                    "sla_violations": alg_data["sla"]["sla_violations"],
                    "sla_violation_rate": alg_data["sla"]["sla_violation_rate"],
                    "mean_utilization": alg_data["performance"]["mean_utilization"],
                    "resource_imbalance": alg_data["performance"]["resource_imbalance"],
                    "total_energy": alg_data["cost"]["total_energy"],
                }

        return summary