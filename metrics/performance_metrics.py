"""
Performance Metrics - Makespan and Resource Utilization

Computes makespan, resource utilization, and load balance for scheduling results.
"""

from typing import List, Dict, Any

from core.workload_manager import Task
from core.resource_manager import VirtualMachine
from core.scheduler_engine import ScheduleAssignment, SchedulingResult


class PerformanceMetrics:
    """
    Computes performance-related metrics for scheduling results.
    """

    def __init__(
        self,
        tasks: List[Task],
        vms: List[VirtualMachine],
    ):
        """
        Initialize with task and VM references.

        Args:
            tasks: Original task list.
            vms: Original VM list.
        """
        self.tasks = {t.task_id: t for t in tasks}
        self.vms = {v.vm_id: v for v in vms}

    def makespan(self, assignments: List[ScheduleAssignment]) -> float:
        """
        Compute makespan (maximum completion time across all VMs).

        Args:
            assignments: Task-to-VM assignments.

        Returns:
            Makespan value.
        """
        if not assignments:
            return 0.0
        return max(a.completion_time for a in assignments)

    def resource_utilization(
        self,
        assignments: List[ScheduleAssignment],
    ) -> Dict[str, float]:
        """
        Compute per-VM utilization and overall statistics.

        Utilization = (busy time) / (makespan) for each VM.

        Args:
            assignments: Task-to-VM assignments.

        Returns:
            Dict with 'mean_utilization', 'std_utilization', 'min_utilization',
            'max_utilization', 'resource_imbalance'.
        """
        if not assignments:
            return {
                "mean_utilization": 0.0,
                "std_utilization": 0.0,
                "min_utilization": 0.0,
                "max_utilization": 0.0,
                "resource_imbalance": 0.0,
            }

        vm_busy: Dict[int, float] = {}
        for a in assignments:
            dur = a.completion_time - a.start_time
            vm_busy[a.vm_id] = vm_busy.get(a.vm_id, 0.0) + dur

        span = self.makespan(assignments)
        if span <= 0:
            span = 1.0

        vm_ids = set(vm_busy.keys()) | {v.vm_id for v in self.vms.values()}
        utilizations = [
            vm_busy.get(vid, 0.0) / span
            for vid in vm_ids
        ]

        import numpy as np
        arr = np.array(utilizations)
        mean_u = float(np.mean(arr))
        std_u = float(np.std(arr)) if len(arr) > 1 else 0.0
        min_u = float(np.min(arr))
        max_u = float(np.max(arr))
        # Resource imbalance: std or (max - min)
        imbalance = max_u - min_u if len(arr) > 1 else 0.0

        return {
            "mean_utilization": mean_u,
            "std_utilization": std_u,
            "min_utilization": min_u,
            "max_utilization": max_u,
            "resource_imbalance": imbalance,
        }

    def compute(self, result: SchedulingResult) -> Dict[str, Any]:
        """
        Compute all performance metrics for a scheduling result.

        Args:
            result: SchedulingResult from an algorithm run.

        Returns:
            Dict with makespan and utilization metrics.
        """
        mspan = self.makespan(result.assignments)
        util = self.resource_utilization(result.assignments)
        return {
            "makespan": mspan,
            **util,
        }
