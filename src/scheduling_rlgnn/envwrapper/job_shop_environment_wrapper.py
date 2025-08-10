import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, List, Tuple

from job_shop_lib import JobShopInstance, Operation
from job_shop_lib.dispatching import Dispatcher
from job_shop_lib.dispatching.rules import EarliestDueDateRule


class JobShopEnvironmentWrapper:
    # TODO: Add a new doscstring after the full class implementation
    """
    RLModule implementation for Job Shop Scheduling using Graph Neural Networks.
    Uses the new RLLib API stack.
    """

    def __init__(self, instance: "JobShopInstance", max_steps: int = 1000):

        self.instance = instance
        self.max_steps = max_steps
        self.current_step = 0
        self.completed_operations: set["Operation"] = set()
        self.machine_schedules: Dict[int, List[Dict[str, Any]]] = {}
        self.current_time = 0
        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation."""
        self.current_step = 0
        self.completed_operations = set()
        self.machine_schedules = {
            machine_id: [] for machine_id in range(self.instance.num_machines)
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
            for operation in job:
                features = [
                    operation.duration,
                    float(operation in self.completed_operations),
                    operation.machine_id,
                    # Add more relevant features
                ]
                operation_nodes.append(features)

        # Node features for machines
        for machine_id in range(self.instance.num_machines):
            workload = sum(
                op["end_time"] - op["start_time"]
                for op in self.machine_schedules[machine_id]
            )
            features = [
                machine_id,
                workload,
                len(self.machine_schedules[machine_id]),
                # Add more machine features
            ]
            machine_nodes.append(features)

        # Combine all nodes
        all_nodes = operation_nodes + machine_nodes
        node_features = np.array(all_nodes, dtype=np.float32)

        # Pad to fixed size for consistency
        # TODO: investigate the initial required maximum number of nodes
        max_nodes = 64
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

    def _get_available_operations(self) -> List[Operation]:
        """Get list of operations that can be scheduled."""
        available = []

        for job in self.instance.jobs:
            for i, operation in enumerate(job):
                # Check if all previous operations in the job are completed
                prev_completed = all(
                    (job[j] in self.completed_operations) for j in range(i)
                )

                if (
                    prev_completed
                    and operation not in self.completed_operations
                ):
                    available.append(operation)
                    break  # Only first available operation per job

        return available

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute action and return next observation, reward, done flags, and info."""
        self.current_step += 1

        # Execute action using Job Shop Lib
        reward = self._execute_action(action)
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> float:
        """Execute the selected action and return the reward."""
        # Map action to job and operation
        available_ops = self._get_available_operations()

        if not available_ops or action >= len(available_ops):
            return -10.0  # Penalty for invalid action

        # Select operation based on action
        selected_operation = available_ops[action]

        # Schedule the operation
        machine_id = selected_operation.machine_id
        start_time = max(
            self.current_time,
            max(
                [op["end_time"] for op in self.machine_schedules[machine_id]],
                default=0,
            ),
        )
        end_time = start_time + selected_operation.duration

        # Add to machine schedule
        self.machine_schedules[machine_id].append(
            {
                "operation": selected_operation,
                "start_time": start_time,
                "end_time": end_time,
                "job_id": self._get_job_id(selected_operation),
            }
        )

        # Mark operation as completed
        self.completed_operations.add(selected_operation)

        # Update current time
        self.current_time = max(self.current_time, end_time)

        # Calculate reward (negative makespan improvement)
        reward = self._calculate_reward(
            selected_operation, start_time, end_time
        )

        return reward

    def _get_job_id(self, operation: Operation) -> int:
        """Get job ID for a given operation."""
        for job_id, job in enumerate(self.instance.jobs):
            if operation in job:
                return job_id
        return -1
