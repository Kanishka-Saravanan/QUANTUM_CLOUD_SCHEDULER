"""
FIFO Scheduler - First-In-First-Out Cloud Task Scheduling

Assigns tasks to VMs in arrival order, selecting the VM that minimizes
cost for each task (greedy cost-based placement).
"""

from typing import List, Any

from core.workload_manager import Task
from core.resource_manager import VirtualMachine
from core.scheduler_engine import ScheduleAssignment


def schedule_fifo(
    tasks: List[Task],
    vms: List[VirtualMachine],
    **kwargs: Any,
) -> List[ScheduleAssignment]:
    """
    Schedule tasks using FIFO (First-In-First-Out) policy.

    Tasks are sorted by arrival time. Each task is assigned to the VM
    that minimizes scheduling cost (execution_time * cost_per_unit).

    Args:
        tasks: List of tasks to schedule.
        vms: List of available VMs.
        **kwargs: Ignored (for API compatibility).

    Returns:
        List of ScheduleAssignment objects.
    """
    if not tasks or not vms:
        return []

    # Sort by arrival time (FIFO)
    sorted_tasks = sorted(tasks, key=lambda t: (t.arrival_time, t.task_id))

    assignments: List[ScheduleAssignment] = []
    vm_availability: List[float] = [0.0] * len(vms)

    for task in sorted_tasks:
        best_vm_idx = 0
        best_cost = float("inf")
        best_start = 0.0

        # Find VM with minimum cost that can fit the task
        for idx, vm in enumerate(vms):
            if task.cpu_required > vm.cpu_capacity:
                continue
            start = max(task.arrival_time, vm_availability[idx])
            completion = start + task.execution_time
            cost = task.execution_time * vm.cost_per_time_unit
            if cost < best_cost:
                best_cost = cost
                best_vm_idx = idx
                best_start = start

        vm = vms[best_vm_idx]
        completion_time = best_start + task.execution_time
        vm_availability[best_vm_idx] = completion_time

        assignments.append(
            ScheduleAssignment(
                task_id=task.task_id,
                vm_id=vm.vm_id,
                start_time=best_start,
                completion_time=completion_time,
            )
        )

    return assignments


class FIFOScheduler:
    """FIFO scheduling algorithm - entry point for experiments."""

    @staticmethod
    def schedule(
        tasks: List[Task],
        vms: List[VirtualMachine],
        **kwargs: Any,
    ) -> List[ScheduleAssignment]:
        """Schedule tasks using FIFO policy."""
        return schedule_fifo(tasks, vms, **kwargs)
