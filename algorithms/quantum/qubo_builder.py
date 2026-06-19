"""
QUBO Builder - Formulates Cloud Scheduling as Quadratic Unconstrained Binary Optimization

Maps task-to-VM assignment to a QUBO problem minimizing:
1. Execution cost (total VM cost)
2. SLA penalty (tasks exceeding deadline)
3. Load balancing penalty (vm_load - average_load)^2
4. Assignment constraint: each task exactly one VM
"""

import numpy as np
from typing import List, Tuple, Optional

from qiskit_optimization import QuadraticProgram

from core.workload_manager import Task
from core.resource_manager import VirtualMachine


class QUBOBuilder:
    """
    Builds a QUBO representation of the cloud task scheduling problem.

    Binary variables x[i,j] = 1 indicate task i is assigned to VM j.
    The objective combines cost, load balance, and SLA penalty terms
    to encourage distribution across VMs and reduce SLA violations.
    """

    def __init__(
        self,
        cost_weight: float = 0.5,
        balance_weight: float = 2.0,
        sla_weight: float = 1.5,
    ):
        """
        Initialize QUBO builder with objective weights.

        Weights are tuned so load balancing dominates (to avoid
        all tasks on one VM) while still considering cost and SLA.

        Args:
            cost_weight: Weight for total scheduling cost.
            balance_weight: Weight for (vm_load - avg_load)^2 penalty.
            sla_weight: Weight for SLA violation penalty.
        """
        self.cost_weight = cost_weight
        self.balance_weight = balance_weight
        self.sla_weight = sla_weight

    def _get_var_names(self, qp: QuadraticProgram) -> set:
        """Get set of variable names in the QuadraticProgram."""
        return {v.name for v in qp.variables}

    def build(
        self,
        tasks: List[Task],
        vms: List[VirtualMachine],
        vm_base_loads: Optional[List[float]] = None,
    ) -> QuadraticProgram:
        """
        Build a QuadraticProgram (QUBO) for the scheduling problem.

        Args:
            tasks: List of tasks to schedule.
            vms: List of available VMs.
            vm_base_loads: Optional prior load on each VM (for batch processing).
                Length must equal len(vms). Defaults to zeros.

        Returns:
            QuadraticProgram instance.
        """
        n_tasks = len(tasks)
        n_vms = len(vms)

        if n_tasks == 0 or n_vms == 0:
            raise ValueError("Need at least one task and one VM.")

        if vm_base_loads is None:
            vm_base_loads = [0.0] * n_vms
        if len(vm_base_loads) != n_vms:
            vm_base_loads = [0.0] * n_vms

        qp = QuadraticProgram(name="cloud_scheduling_qubo")

        # Create binary variables: x_ij = 1 if task i assigned to VM j
        for i in range(n_tasks):
            for j in range(n_vms):
                if tasks[i].cpu_required <= vms[j].cpu_capacity:
                    qp.binary_var(name=f"x_{i}_{j}")

        var_names = self._get_var_names(qp)

        # Constraint: each task assigned to exactly one VM
        for i in range(n_tasks):
            vars_i = [
                f"x_{i}_{j}"
                for j in range(n_vms)
                if tasks[i].cpu_required <= vms[j].cpu_capacity
            ]
            if vars_i:
                qp.linear_constraint(
                    linear={name: 1.0 for name in vars_i},
                    sense="==",
                    rhs=1,
                    name=f"task_{i}_assign",
                )

        # Precompute task loads and total work
        exec_times = [tasks[i].execution_time for i in range(n_tasks)]
        total_batch_work = sum(exec_times)
        total_prior_load = sum(vm_base_loads)
        total_work = total_prior_load + total_batch_work
        avg_load = total_work / n_vms if n_vms > 0 else 0.0

        linear: dict = {}
        quadratic: dict = {}

        # 1) Execution cost: sum_ij x_ij * (exec_time_i * cost_j)
        for i in range(n_tasks):
            for j in range(n_vms):
                name = f"x_{i}_{j}"
                if name in var_names:
                    cost = exec_times[i] * vms[j].cost_per_time_unit
                    linear[name] = linear.get(name, 0) + self.cost_weight * cost

        # 2) Load balancing penalty: sum_j (load_j - avg_load)^2
        # load_j = base_load_j + sum_i x_ij * exec_time_i
        # (load_j - avg)^2 = load_j^2 - 2*avg*load_j + const
        # Minimizing sum_j load_j^2 - 2*avg*load_j (constant drops)
        # load_j^2 = (base_j + sum_i x_ij e_i)^2 = base_j^2 + 2*base_j*sum_i x_ij e_i + (sum_i x_ij e_i)^2
        # (sum_i x_ij e_i)^2 = sum_i x_ij e_i^2 + 2 sum_{i<k} x_ij x_kj e_i e_k
        for j in range(n_vms):
            base_j = vm_base_loads[j]
            vm_vars: List[Tuple[str, float]] = []
            for i in range(n_tasks):
                name = f"x_{i}_{j}"
                if name in var_names:
                    vm_vars.append((name, exec_times[i]))

            for name, e_i in vm_vars:
                # Linear from load_j^2: x_ij^2 * e_i^2 = x_ij * e_i^2
                linear[name] = linear.get(name, 0) + self.balance_weight * (e_i ** 2)
                # Linear from -2*avg*load_j: -2*avg*x_ij*e_i
                linear[name] = linear.get(name, 0) - self.balance_weight * 2 * avg_load * e_i
                # Linear from 2*base_j*sum_i x_ij e_i: 2*base_j*e_i*x_ij
                linear[name] = linear.get(name, 0) + self.balance_weight * 2 * base_j * e_i

            for idx_a, (name_a, e_a) in enumerate(vm_vars):
                for idx_b, (name_b, e_b) in enumerate(vm_vars):
                    if idx_a < idx_b:
                        key = (name_a, name_b)
                        quadratic[key] = (
                            quadratic.get(key, 0)
                            + self.balance_weight * 2 * e_a * e_b
                        )

        # 3) SLA penalty: penalize assigning task i to overloaded VM j when task is urgent
        # Urgency = 1 / (deadline + epsilon). Penalty: x_ij * (sum_{k!=i} x_kj * e_k) * urgency_i
        # Quadratic: x_ij * x_kj * e_k * urgency_i for k != i
        eps = 1e-6
        for i in range(n_tasks):
            urgency_i = 1.0 / (tasks[i].deadline + eps)
            for j in range(n_vms):
                name_i = f"x_{i}_{j}"
                if name_i not in var_names:
                    continue
                for k in range(n_tasks):
                    if k == i:
                        continue
                    name_k = f"x_{k}_{j}"
                    if name_k not in var_names:
                        continue
                    key = (min(name_i, name_k), max(name_i, name_k))
                    quadratic[key] = (
                        quadratic.get(key, 0)
                        + self.sla_weight * exec_times[k] * urgency_i * 0.5
                    )

        qp.minimize(linear=linear, quadratic=quadratic)
        return qp

    def variable_to_assignment(
        self,
        solution: dict,
        tasks: List[Task],
        vms: List[VirtualMachine],
    ) -> List[Tuple[int, int]]:
        """
        Map QUBO solution (variable assignments) to task-VM pairs.

        Args:
            solution: Dict mapping variable name to 0/1.
            tasks: Original task list.
            vms: Original VM list.

        Returns:
            List of (task_id, vm_id) tuples.
        """
        n_tasks = len(tasks)
        n_vms = len(vms)
        assignments: List[Tuple[int, int]] = []

        for i in range(n_tasks):
            assigned = False
            for j in range(n_vms):
                name = f"x_{i}_{j}"
                if solution.get(name, 0) == 1:
                    assignments.append((tasks[i].task_id, vms[j].vm_id))
                    assigned = True
                    break
            if not assigned:
                for j in range(n_vms):
                    if tasks[i].cpu_required <= vms[j].cpu_capacity:
                        assignments.append((tasks[i].task_id, vms[j].vm_id))
                        break

        return assignments
