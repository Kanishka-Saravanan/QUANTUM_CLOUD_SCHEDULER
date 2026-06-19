from sklearn.ensemble import GradientBoostingRegressor
from core.scheduler_engine import ScheduleAssignment


def train_ml_model(tasks, vms):
    X = []
    y = []

    for task in tasks:
        for vm in vms:

            features = [
                task.execution_time,
                task.cpu_required,
                task.memory_required,
                vm.cpu_capacity,
                vm.memory_capacity,
                vm.cost_per_time_unit
            ]

            # Estimated execution cost
            cost = task.execution_time / vm.cpu_capacity

            X.append(features)
            y.append(cost)

    model = GradientBoostingRegressor()
    model.fit(X, y)

    return model


def ml_schedule(tasks, vms, model):

    assignments = []
    vm_available_time = {vm.vm_id: 0 for vm in vms}

    for task in tasks:

        best_vm = None
        best_cost = float("inf")

        for vm in vms:

            features = [[
                task.execution_time,
                task.cpu_required,
                task.memory_required,
                vm.cpu_capacity,
                vm.memory_capacity,
                vm.cost_per_time_unit
            ]]

            predicted_cost = model.predict(features)[0]

            if predicted_cost < best_cost:
                best_cost = predicted_cost
                best_vm = vm

        # Scheduling logic
        start_time = vm_available_time[best_vm.vm_id]
        execution_time = task.execution_time / best_vm.cpu_capacity
        completion_time = start_time + execution_time

        vm_available_time[best_vm.vm_id] = completion_time

        assignments.append(
            ScheduleAssignment(
                task_id=task.task_id,
                vm_id=best_vm.vm_id,
                start_time=start_time,
                completion_time=completion_time
            )
        )

    return assignments