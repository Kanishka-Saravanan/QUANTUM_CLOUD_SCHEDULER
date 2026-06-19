"""
Resource Manager - Virtual Machine and Resource Modeling

Models a pool of heterogeneous VMs with CPU capacity, cost, and energy
consumption for cloud scheduling simulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class VirtualMachine:
    """Represents a single virtual machine with capacity and cost attributes."""

    vm_id: int
    cpu_capacity: float
    cost_per_time_unit: float
    energy_per_time_unit: float
    memory_capacity: float = 1.0

    def __repr__(self) -> str:
        return (
            f"VM(id={self.vm_id}, cpu={self.cpu_capacity:.1f}, "
            f"cost={self.cost_per_time_unit:.3f}/unit)"
        )


class ResourceManager:
    """
    Manages a pool of virtual machines for cloud scheduling.

    Supports heterogeneous VM types with different capacities and costs.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the resource manager.

        Args:
            seed: Random seed for reproducible VM generation.
        """
        self.rng = np.random.default_rng(seed)
        self._vms: List[VirtualMachine] = []

    def add_vm(self, vm: VirtualMachine) -> None:
        """Add a VM to the pool."""
        self._vms.append(vm)

    def create_default_pool(
        self,
        num_vms: int = 8,
        cpu_range: tuple = (1.0, 4.0),
        cost_range: tuple = (0.05, 0.25),
        energy_range: tuple = (0.5, 2.0),
    ) -> List[VirtualMachine]:
        """
        Create a default heterogeneous VM pool.

        Args:
            num_vms: Number of VMs to create.
            cpu_range: (min, max) CPU capacity.
            cost_range: (min, max) cost per time unit.
            energy_range: (min, max) energy consumption per time unit.

        Returns:
            List of created VirtualMachine instances.
        """
        self._vms.clear()

        cpus = self.rng.uniform(cpu_range[0], cpu_range[1], num_vms)
        costs = self.rng.uniform(cost_range[0], cost_range[1], num_vms)
        energies = self.rng.uniform(energy_range[0], energy_range[1], num_vms)

        for i in range(num_vms):
            vm = VirtualMachine(
                vm_id=i,
                cpu_capacity=float(cpus[i]),
                cost_per_time_unit=float(costs[i]),
                energy_per_time_unit=float(energies[i]),
                memory_capacity=1.0,
            )
            self._vms.append(vm)

        return self._vms

    def get_vms(self) -> List[VirtualMachine]:
        """Return the current VM pool."""
        return self._vms

    def get_vm_by_id(self, vm_id: int) -> VirtualMachine | None:
        """Get VM by ID."""
        for vm in self._vms:
            if vm.vm_id == vm_id:
                return vm
        return None

    def num_vms(self) -> int:
        """Return number of VMs in the pool."""
        return len(self._vms)
