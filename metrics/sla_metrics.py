"""
SLA Metrics - Service Level Agreement Violation Tracking

Computes SLA violations based on task deadlines vs. completion times.
"""

from typing import List, Dict, Any

from core.workload_manager import Task
from core.resource_manager import VirtualMachine
from core.scheduler_engine import ScheduleAssignment, SchedulingResult


class SLAMetrics:
    """
    Computes SLA-related metrics for scheduling results.

    SLA violation: task completes after its deadline.
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

    def sla_violations(self, assignments: List[ScheduleAssignment]) -> int:
        """
        Count tasks that completed after their deadline.

        Absolute deadline = arrival_time + deadline (deadline is duration from arrival).

        Args:
            assignments: Task-to-VM assignments.

        Returns:
            Number of SLA violations.
        """
        count = 0
        for a in assignments:
            task = self.tasks.get(a.task_id)
            if task:
                absolute_deadline = task.arrival_time + task.deadline
                if a.completion_time > absolute_deadline:
                    count += 1
        return count

    def sla_violation_rate(
        self,
        assignments: List[ScheduleAssignment],
    ) -> float:
        """
        Fraction of tasks that violated SLA.

        Args:
            assignments: Task-to-VM assignments.

        Returns:
            Violation rate in [0, 1].
        """
        if not assignments:
            return 0.0
        return self.sla_violations(assignments) / len(assignments)

    def compute(self, result: SchedulingResult) -> Dict[str, Any]:
        """
        Compute all SLA metrics for a scheduling result.

        Args:
            result: SchedulingResult from an algorithm run.

        Returns:
            Dict with 'sla_violations' and 'sla_violation_rate'.
        """
        violations = self.sla_violations(result.assignments)
        rate = self.sla_violation_rate(result.assignments)
        return {
            "sla_violations": violations,
            "sla_violation_rate": rate,
        }
