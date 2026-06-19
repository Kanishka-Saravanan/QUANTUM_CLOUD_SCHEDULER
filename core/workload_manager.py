"""
Workload Manager - Synthetic Cloud Workload Generation

Generates realistic cloud task workloads with execution times, deadlines,
resource requirements, and priority levels for scheduling experiments.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Task:
    """Represents a single cloud task with scheduling-relevant attributes."""

    task_id: int
    execution_time: float
    deadline: float
    cpu_required: float
    memory_required: float
    priority: int
    arrival_time: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Task(id={self.task_id}, exec={self.execution_time:.1f}s, "
            f"deadline={self.deadline:.1f}, cpu={self.cpu_required})"
        )


class WorkloadManager:
    """
    Generates synthetic cloud workloads for scheduling experiments.

    Tasks are generated with realistic distributions for execution time,
    resource demands, and deadlines to simulate heterogeneous workloads.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the workload manager with optional random seed.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

    def generate_workload(
        self,
        num_tasks: int,
        exec_time_range: tuple = (1.0, 50.0),
        deadline_slack: float = 2.0,
        cpu_range: tuple = (0.1, 1.0),
        memory_range: tuple = (0.1, 1.0),
        num_priorities: int = 5,
    ) -> List[Task]:
        """
        Generate a synthetic workload of cloud tasks.

        Args:
            num_tasks: Number of tasks to generate.
            exec_time_range: (min, max) execution time in time units.
            deadline_slack: Multiplier for deadline = arrival + slack * exec_time.
            cpu_range: (min, max) CPU requirement as fraction of VM capacity.
            memory_range: (min, max) memory requirement as fraction.
            num_priorities: Number of priority levels (1=highest).

        Returns:
            List of Task objects.
        """
        tasks: List[Task] = []

        exec_times = self.rng.uniform(
            exec_time_range[0], exec_time_range[1], num_tasks
        )
        cpu_reqs = self.rng.uniform(cpu_range[0], cpu_range[1], num_tasks)
        memory_reqs = self.rng.uniform(memory_range[0], memory_range[1], num_tasks)
        priorities = self.rng.integers(1, num_priorities + 1, num_tasks)

        # Deadlines based on execution time with some randomness
        slack_factors = self.rng.uniform(1.2, deadline_slack, num_tasks)
        deadlines = exec_times * slack_factors

        # Optional: stagger arrival times
        arrival_times = self.rng.uniform(0, 10, num_tasks)

        for i in range(num_tasks):
            task = Task(
                task_id=i,
                execution_time=float(exec_times[i]),
                deadline=float(deadlines[i]),
                cpu_required=float(cpu_reqs[i]),
                memory_required=float(memory_reqs[i]),
                priority=int(priorities[i]),
                arrival_time=float(arrival_times[i]),
            )
            tasks.append(task)

        return tasks

    def to_dict_list(self, tasks: List[Task]) -> List[dict]:
        """Convert task list to list of dictionaries for pandas/JSON."""
        return [
            {
                "task_id": t.task_id,
                "execution_time": t.execution_time,
                "deadline": t.deadline,
                "cpu_required": t.cpu_required,
                "memory_required": t.memory_required,
                "priority": t.priority,
                "arrival_time": t.arrival_time,
            }
            for t in tasks
        ]
