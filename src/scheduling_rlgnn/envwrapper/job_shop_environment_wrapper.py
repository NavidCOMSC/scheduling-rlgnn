import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional, Union


# Job Shop Lib imports
try:
    from job_shop_lib import JobShopInstance, Operation, Job, Machine
    from job_shop_lib.dispatching import DispatchingRule
    from job_shop_lib.solvers import Solver
except ImportError:
    print(
        "Warning: job_shop_lib not installed. Install with: pip install job-shop-lib"
    )

    # Mock classes for development
    class JobShopInstance:
        def __init__(self, jobs, machines):
            self.jobs = jobs
            self.machines = machines

    class Operation:
        def __init__(self, machine_id, processing_time):
            self.machine_id = machine_id
            self.processing_time = processing_time

    class Job:
        def __init__(self, operations):
            self.operations = operations

    class Machine:
        def __init__(self, machine_id):
            self.id = machine_id


class JobShopEnvironmentWrapper:
    # TODO: Add a new doscstring after the full class implementation
    """
    RLModule implementation for Job Shop Scheduling using Graph Neural Networks.
    Uses the new RLLib API stack.
    """

    def __init__(self, instance: JobShopInstance, max_steps: int = 1000):

        self.instance = instance
        self.max_steps = max_steps
        self.current_step = 0
        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation."""
        self.current_step = 0
        self.completed_operations = set()
        self.machine_schedules = {
            machine.id: [] for machine in self.instance.machines
        }
        self.current_time = 0

        return self._get_observation()

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation as graph representation."""
        return self._create_graph_observation()

    def _create_graph_observation(self) -> Dict[str, np.ndarray]:
        """Create graph-based observation of the current state."""
        # Create nodes for operations and machines
        operation_nodes = []
        machine_nodes = []

        # Node features for operations
        for job in self.instance.jobs:
            for operation in job.operations:
                features = [
                    operation.processing_time,
                    float(operation in self.completed_operations),
                    operation.machine_id,
                    # Add more relevant features
                ]
                operation_nodes.append(features)

        # Node features for machines
        for machine in self.instance.machines:
            workload = sum(
                op["end_time"] - op["start_time"]
                for op in self.machine_schedules[machine.id]
            )
            features = [
                machine.id,
                workload,
                len(self.machine_schedules[machine.id]),
                # Add more machine features
            ]
            machine_nodes.append(features)

        # Combine all nodes
        all_nodes = operation_nodes + machine_nodes
        node_features = np.array(all_nodes, dtype=np.float32)

        # Pad to fixed size for consistency
        max_nodes = 64  # Adjust based on your problem size
        if len(all_nodes) < max_nodes:
            padding = np.zeros(
                (max_nodes - len(all_nodes), node_features.shape[1])
            )
            node_features = np.vstack([node_features, padding])

        # Create action mask
        available_ops = self._get_available_operations()
        action_mask = np.zeros(max_nodes, dtype=bool)
        action_mask[: len(available_ops)] = True

        return {"obs": node_features.flatten(), "action_mask": action_mask}
