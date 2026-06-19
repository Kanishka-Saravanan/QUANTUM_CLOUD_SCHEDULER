"""
Scheduler Engine - Orchestrates Scheduling Decisions

Coordinates tasks, resources, and scheduling algorithms to produce
task-to-VM assignments and evaluate scheduling outcomes.
"""

from typing import List, Dict, Callable, Any
from dataclasses import dataclass

from .workload_manager import Task
from .resource_manager import VirtualMachine


@dataclass
class ScheduleAssignment:
    """Represents a single task-to-VM assignment."""

    task_id: int
    vm_id: int
    start_time: float
    completion_time: float

    def __repr__(self) -> str:
        return f"Task {self.task_id} -> VM {self.vm_id} [{self.start_time:.1f}-{self.completion_time:.1f}]"


@dataclass
class SchedulingResult:
    """Complete result of a scheduling run."""

    algorithm_name: str
    assignments: List[ScheduleAssignment]
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        return (
            f"SchedulingResult({self.algorithm_name}, "
            f"{len(self.assignments)} assignments)"
        )


class SchedulerEngine:
    """
    Main orchestrator for running scheduling algorithms.

    Takes tasks, VMs, and a scheduling function; produces assignments
    and supports evaluation of scheduling metrics.
    """

    def __init__(
        self,
        tasks: List[Task],
        vms: List[VirtualMachine],
    ):
        """
        Initialize the scheduler engine.

        Args:
            tasks: List of tasks to schedule.
            vms: List of available VMs.
        """
        self.tasks = tasks
        self.vms = vms

    def run(
        self,
        scheduler_fn: Callable[
            [List[Task], List[VirtualMachine]], List[ScheduleAssignment]
        ],
        algorithm_name: str = "Unknown",
        **kwargs: Any,
    ) -> SchedulingResult:
        """
        Execute a scheduling algorithm and return structured result.

        Args:
            scheduler_fn: Function (tasks, vms) -> List[ScheduleAssignment].
            algorithm_name: Name for reporting.
            **kwargs: Additional arguments passed to scheduler_fn.

        Returns:
            SchedulingResult with assignments and metadata.
        """
        assignments = scheduler_fn(self.tasks, self.vms, **kwargs)
        return SchedulingResult(
            algorithm_name=algorithm_name,
            assignments=assignments,
            metadata={"num_tasks": len(self.tasks), "num_vms": len(self.vms)},
        )
