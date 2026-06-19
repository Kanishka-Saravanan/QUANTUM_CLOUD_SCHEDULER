"""
Cost Metrics - Scheduling Cost and Energy Consumption

Computes total scheduling cost and energy consumption for task assignments.
"""

from typing import List, Dict, Any

from core.workload_manager import Task
from core.resource_manager import VirtualMachine
from core.scheduler_engine import ScheduleAssignment, SchedulingResult


class CostMetrics:
    """
    Computes cost-related metrics for scheduling results.

    Requires task and VM lookups to compute execution cost and energy.
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

    def total_cost(self, assignments: List[ScheduleAssignment]) -> float:
        """
        Compute total scheduling cost.

        Cost = sum over assignments of (execution_time * cost_per_time_unit).

        Args:
            assignments: Task-to-VM assignments.

        Returns:
            Total cost in arbitrary units.
        """
        total = 0.0
        for a in assignments:
            task = self.tasks.get(a.task_id)
            vm = self.vms.get(a.vm_id)
            if task and vm:
                exec_time = a.completion_time - a.start_time
                total += exec_time * vm.cost_per_time_unit
        return total

    def total_energy(self, assignments: List[ScheduleAssignment]) -> float:
        """
        Compute total energy consumption.

        Energy = sum over assignments of (execution_time * energy_per_time_unit).

        Args:
            assignments: Task-to-VM assignments.

        Returns:
            Total energy in arbitrary units.
        """
        total = 0.0
        for a in assignments:
            task = self.tasks.get(a.task_id)
            vm = self.vms.get(a.vm_id)
            if task and vm:
                exec_time = a.completion_time - a.start_time
                total += exec_time * vm.energy_per_time_unit
        return total

    def compute(self, result: SchedulingResult) -> Dict[str, float]:
        """
        Compute all cost-related metrics for a scheduling result.

        Args:
            result: SchedulingResult from an algorithm run.

        Returns:
            Dict with 'total_cost' and 'total_energy'.
        """
        return {
            "total_cost": self.total_cost(result.assignments),
            "total_energy": self.total_energy(result.assignments),
        }
