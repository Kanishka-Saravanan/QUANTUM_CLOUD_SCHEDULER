"""
QAOA Optimizer - Quantum Approximate Optimization for Cloud Scheduling

Uses Qiskit QAOA with Aer simulator to solve the scheduling QUBO.
Implements batch processing so all tasks are optimized (in batches),
producing distributed assignments across VMs for better utilization
and load balance.
"""

import numpy as np
from typing import List, Tuple, Optional

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from core.workload_manager import Task
from core.resource_manager import VirtualMachine
from core.scheduler_engine import ScheduleAssignment

from .qubo_builder import QUBOBuilder


# Max variables for QAOA (simulator limit ~12-16 qubits)
MAX_QAOA_VARS = 12


def _get_sampler():
    """Get Sampler for QAOA - Aer if available, else base Sampler."""
    try:
        from qiskit_aer.primitives import Sampler
        return Sampler()
    except ImportError:
        pass
    try:
        from qiskit.primitives import Sampler
        return Sampler()
    except ImportError:
        return None


def _run_qaoa(
    qp: QuadraticProgram,
    reps: int,
    max_iter: int,
    seed: int,
) -> Optional[np.ndarray]:
    """
    Run QAOA on the QuadraticProgram and return the best solution vector.

    Uses COBYLA or SPSA for parameter optimization.

    Returns:
        Solution vector x (0/1) or None if QAOA fails.
    """
    try:
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA

        sampler = _get_sampler()
        if sampler is None:
            return None

        optimizer = COBYLA(maxiter=max_iter)
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps, seed_simulator=seed)
        meo = MinimumEigenOptimizer(qaoa)
        result = meo.solve(qp)

        if result.x is not None and len(result.x) > 0:
            return np.array([int(round(z)) for z in result.x])
    except Exception:
        pass
    return None


def _run_numpy_solver(qp: QuadraticProgram) -> Optional[np.ndarray]:
    """Run NumPyMinimumEigensolver as fallback (exact classical solver)."""
    try:
        from qiskit_algorithms import NumPyMinimumEigensolver
        meo = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        result = meo.solve(qp)
        if result.x is not None and len(result.x) > 0:
            return np.array([int(round(z)) for z in result.x])
    except Exception:
        pass
    return None


def _solve_qubo(
    qp: QuadraticProgram,
    use_qaoa: bool,
    reps: int,
    max_iter: int,
    seed: int,
) -> Optional[dict]:
    """
    Solve QUBO, trying QAOA first if requested, then NumPy fallback.

    Returns:
        Dict mapping variable name to 0/1, or None.
    """
    x = None
    if use_qaoa and qp.get_num_binary_vars() <= MAX_QAOA_VARS:
        x = _run_qaoa(qp, reps=reps, max_iter=max_iter, seed=seed)
    if x is None:
        x = _run_numpy_solver(qp)
    if x is None:
        return None

    var_names = [v.name for v in qp.variables]
    return dict(zip(var_names, x.tolist()))


def _assign_to_least_loaded(
    task: Task,
    vms: List[VirtualMachine],
    vm_loads: dict,
) -> int:
    """Return VM index with lowest current load that can run the task."""
    valid = [
        (j, vm_loads.get(vms[j].vm_id, 0.0))
        for j in range(len(vms))
        if task.cpu_required <= vms[j].cpu_capacity
    ]
    if not valid:
        return 0
    return min(valid, key=lambda x: x[1])[0]


class QAOAOptimizer:
    """
    QAOA-based optimizer for cloud task scheduling.

    Uses batch processing: tasks are processed in batches, each batch
    solved via QUBO (QAOA or exact solver). VM loads are updated between
    batches so the optimizer distributes tasks across VMs.
    """

    def __init__(
        self,
        num_layers: int = 2,
        max_iter: int = 100,
        seed: int = 42,
        batch_size: int = 4,
        use_qaoa: bool = True,
    ):
        """
        Initialize QAOA optimizer.

        Args:
            num_layers: QAOA reps (depth).
            max_iter: Classical optimizer iterations.
            seed: Random seed.
            batch_size: Max tasks per QUBO batch (controls variable count).
            use_qaoa: If True, use QAOA when problem is small; else NumPy only.
        """
        self.num_layers = num_layers
        self.max_iter = max_iter
        self.seed = seed
        self.batch_size = max(batch_size, 2)
        self.use_qaoa = use_qaoa
        self.qubo_builder = QUBOBuilder()

    def _create_batches(
        self,
        tasks: List[Task],
        vms: List[VirtualMachine],
    ) -> List[List[Task]]:
        """Create batches of tasks. Tasks sorted by arrival, then by deadline (urgent first)."""
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (t.arrival_time, t.deadline, t.task_id),
        )
        batches: List[List[Task]] = []
        for i in range(0, len(sorted_tasks), self.batch_size):
            batch = sorted_tasks[i : i + self.batch_size]
            if batch:
                batches.append(batch)
        return batches

    def optimize(
        self,
        tasks: List[Task],
        vms: List[VirtualMachine],
        reduce_for_simulator: bool = True,
    ) -> List[ScheduleAssignment]:
        """
        Run batch-based optimization: each batch solved via QUBO, with
        vm_base_loads updated from previous assignments. Ensures tasks
        are distributed across VMs for better utilization and makespan.

        Args:
            tasks: Tasks to schedule.
            vms: Available VMs.
            reduce_for_simulator: If True, limit batch size for QAOA; else full batches.

        Returns:
            List of ScheduleAssignment.
        """
        if not tasks or not vms:
            return []

        max_tasks_per_batch = self.batch_size
        if reduce_for_simulator and len(vms) > 0:
            # Keep n_tasks * n_vms <= MAX_QAOA_VARS. Use 4 VMs per batch.
            n_vms_target = min(len(vms), 4)
            max_tasks_per_batch = min(
                self.batch_size,
                max(2, MAX_QAOA_VARS // n_vms_target),
            )

        batches = []
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (t.arrival_time, t.deadline, t.task_id),
        )
        for i in range(0, len(sorted_tasks), max_tasks_per_batch):
            batch = sorted_tasks[i : i + max_tasks_per_batch]
            if batch:
                batches.append(batch)

        vm_loads: dict = {v.vm_id: 0.0 for v in vms}
        assignments: List[ScheduleAssignment] = []

        vm_rotation = 0
        for batch in batches:
            n_tasks_batch = len(batch)
            n_vms = len(vms)
            total_vars = n_tasks_batch * n_vms

            if total_vars > MAX_QAOA_VARS and reduce_for_simulator:
                n_vms_use = max(2, min(n_vms, MAX_QAOA_VARS // n_tasks_batch))
                # Rotate VM subset so all VMs get used across batches
                start = vm_rotation % n_vms
                indices = [(start + k) % n_vms for k in range(n_vms_use)]
                vms_batch = [vms[i] for i in indices]
                vm_rotation += 1
            else:
                vms_batch = vms

            vm_base = [vm_loads.get(v.vm_id, 0.0) for v in vms_batch]

            try:
                qp = self.qubo_builder.build(
                    batch,
                    vms_batch,
                    vm_base_loads=vm_base,
                )
            except ValueError:
                # Fallback: assign batch via load balancing
                for task in batch:
                    j = _assign_to_least_loaded(task, vms, vm_loads)
                    vm = vms[j]
                    start = max(vm_loads[vm.vm_id], task.arrival_time)
                    end = start + task.execution_time
                    vm_loads[vm.vm_id] = end
                    assignments.append(
                        ScheduleAssignment(
                            task_id=task.task_id,
                            vm_id=vm.vm_id,
                            start_time=start,
                            completion_time=end,
                        )
                    )
                continue

            solution = _solve_qubo(
                qp,
                use_qaoa=self.use_qaoa,
                reps=self.num_layers,
                max_iter=min(self.max_iter, 80),
                seed=self.seed,
            )

            if solution is None:
                for task in batch:
                    j = _assign_to_least_loaded(task, vms, vm_loads)
                    vm = vms[j]
                    start = max(vm_loads[vm.vm_id], task.arrival_time)
                    end = start + task.execution_time
                    vm_loads[vm.vm_id] = end
                    assignments.append(
                        ScheduleAssignment(
                            task_id=task.task_id,
                            vm_id=vm.vm_id,
                            start_time=start,
                            completion_time=end,
                        )
                    )
                continue

            batch_assigns = self.qubo_builder.variable_to_assignment(
                solution,
                batch,
                vms_batch,
            )

            for (tid, vid) in batch_assigns:
                task = next(t for t in batch if t.task_id == tid)
                vm = next(v for v in vms if v.vm_id == vid)
                start = max(vm_loads.get(vid, 0.0), task.arrival_time)
                end = start + task.execution_time
                vm_loads[vid] = end
                assignments.append(
                    ScheduleAssignment(
                        task_id=tid,
                        vm_id=vid,
                        start_time=start,
                        completion_time=end,
                    )
                )

        return assignments


def schedule_quantum(
    tasks: List[Task],
    vms: List[VirtualMachine],
    **kwargs,
) -> List[ScheduleAssignment]:
    """
    Schedule tasks using QAOA-based quantum optimization.

    Uses batch processing and load-balancing QUBO formulation.
    Compatible with experiment runner.
    """
    opt = QAOAOptimizer(
        num_layers=kwargs.get("num_layers", 2),
        max_iter=kwargs.get("max_iter", 80),
        seed=kwargs.get("seed", 42),
        batch_size=kwargs.get("batch_size", 6),
        use_qaoa=kwargs.get("use_qaoa", True),
    )
    return opt.optimize(
        tasks,
        vms,
        reduce_for_simulator=kwargs.get("reduce_for_simulator", True),
    )
