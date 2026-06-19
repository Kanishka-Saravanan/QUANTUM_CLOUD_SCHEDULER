"""
Round Robin Scheduler - Cyclic VM Assignment for Cloud Tasks

Distributes tasks across VMs in round-robin fashion to balance load.
Tasks are still ordered by arrival time but VM selection is cyclic.
"""

from typing import List, Any

from core.workload_manager import Task
from core.resource_manager import VirtualMachine
from core.scheduler_engine import ScheduleAssignment


def schedule_round_robin(
    tasks: List[Task],
    vms: List[VirtualMachine],
    **kwargs: Any,
) -> List[ScheduleAssignment]:
    """
    Schedule tasks using Round Robin policy.

    Tasks are sorted by arrival time. VMs are selected cyclically:
    task 0 -> VM 0, task 1 -> VM 1, ... task n -> VM (n mod num_vms).
    Only VMs that can accommodate the task's CPU requirement are considered.

    Args:
        tasks: List of tasks to schedule.
        vms: List of available VMs.
        **kwargs: Ignored (for API compatibility).

    Returns:
        List of ScheduleAssignment objects.
    """
    if not tasks or not vms:
        return []

    sorted_tasks = sorted(tasks, key=lambda t: (t.arrival_time, t.task_id))

    # Filter VMs that can run at least some tasks (cpu_capacity >= min_cpu)
    valid_vms = [v for v in vms if v.cpu_capacity > 0]
    if not valid_vms:
        valid_vms = vms

    assignments: List[ScheduleAssignment] = []
    vm_availability: List[float] = [0.0] * len(valid_vms)
    next_vm_idx = 0

    for task in sorted_tasks:
        # Find next valid VM (with sufficient CPU)
        attempts = 0
        while attempts < len(valid_vms):
            vm_idx = next_vm_idx % len(valid_vms)
            next_vm_idx += 1
            attempts += 1

            vm = valid_vms[vm_idx]
            if task.cpu_required <= vm.cpu_capacity:
                start = max(task.arrival_time, vm_availability[vm_idx])
                completion_time = start + task.execution_time
                vm_availability[vm_idx] = completion_time

                assignments.append(
                    ScheduleAssignment(
                        task_id=task.task_id,
                        vm_id=vm.vm_id,
                        start_time=start,
                        completion_time=completion_time,
                    )
                )
                break
        else:
            # Fallback: use first VM that fits
            for idx, vm in enumerate(valid_vms):
                if task.cpu_required <= vm.cpu_capacity:
                    start = max(task.arrival_time, vm_availability[idx])
                    completion_time = start + task.execution_time
                    vm_availability[idx] = completion_time
                    assignments.append(
                        ScheduleAssignment(
                            task_id=task.task_id,
                            vm_id=vm.vm_id,
                            start_time=start,
                            completion_time=completion_time,
                        )
                    )
                    break

    return assignments


class RoundRobinScheduler:
    """Round Robin scheduling algorithm - entry point for experiments."""

    @staticmethod
    def schedule(
        tasks: List[Task],
        vms: List[VirtualMachine],
        **kwargs: Any,
    ) -> List[ScheduleAssignment]:
        """Schedule tasks using Round Robin policy."""
        return schedule_round_robin(tasks, vms, **kwargs)
