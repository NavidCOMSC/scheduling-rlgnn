import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, List, Tuple

from job_shop_lib import JobShopInstance, Operation

# from job_shop_lib.dispatching import Dispatcher
# from job_shop_lib.dispatching.rules import shortest_processing_time_rule


class JobShopEnvironmentWrapper:
    # TODO: Add a new doscstring after the full class implementation
    """
    RLModule implementation for Job Shop Scheduling using Graph Neural Networks.
    Uses the new RLLib API stack.
    Uses per-job and per-machine availability times instead of global current_time.
    """

    def __init__(self, instance: "JobShopInstance", max_steps: int = 1000):

        self.instance = instance
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation."""
        self.current_step = 0
        self.completed_operations = set()
        self.machine_schedules = {
            machine_id: [] for machine_id in range(self.instance.num_machines)
        }

        # Per-job availability times (when job is ready for next operation)
        self.job_available_time = [0.0] * len(self.instance.jobs)

        # Per-machine availability times (when machine becomes free)
        self.machine_available_time = [0.0] * self.instance.num_machines

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
                    # Add more relevant features (plausibility)
                ]
                operation_nodes.append(features)

        # Node features for machines
        for machine_id in range(self.instance.num_machines):
            features = [
                machine_id,
                self.machine_available_time[machine_id],
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

        for job_id, job in enumerate(self.instance.jobs):
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
        """Execute the given action and return reward."""

        # Get available operations
        available_ops = self._get_available_operations()

        if not available_ops or action >= len(available_ops):
            return -10.0

        selected_operation = available_ops[action]
        machine_id = selected_operation.machine_id
        job_id = self._get_job_id(selected_operation)

        # Calculate start time based on job and machine availability
        start_time = max(
            self.job_available_time[job_id],
            self.machine_available_time[machine_id],
        )
        end_time = start_time + selected_operation.duration

        # Update state
        self.machine_schedules[machine_id].append(
            {
                "operation": selected_operation,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

        # Update availability times
        self.job_available_time[job_id] = end_time
        self.machine_available_time[machine_id] = end_time

        self.completed_operations.add(selected_operation)

        return self._calculate_reward(selected_operation, end_time)

    def _get_job_id(self, operation: Operation) -> int:
        """Get job ID for a given operation."""
        for job_id, job in enumerate(self.instance.jobs):
            if operation in job:
                return job_id
        return -1

    def _calculate_reward(
        self, operation: Operation, end_time: float
    ) -> float:
        """Calculate reward for scheduling an operation."""
        # Simple reward: negative of completion time (encourages earlier completion)
        base_reward = -end_time

        # Check if this completes a job
        job_id = self._get_job_id(operation)
        job = self.instance.jobs[job_id]
        job_completed = all(op in self.completed_operations for op in job)

        if job_completed:
            base_reward += 10.0  # Bonus for job completion

        return base_reward

    def _is_terminated(self) -> bool:
        """Check if all operations are completed."""
        total_operations = sum(len(job) for job in self.instance.jobs)
        return len(self.completed_operations) == total_operations

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        makespan = (
            max(self.machine_available_time)
            if self.machine_available_time
            else 0
        )

        total_operations = sum(len(job) for job in self.instance.jobs)

        return {
            "makespan": makespan,
            "completed_operations": len(self.completed_operations),
            "total_operations": total_operations,
            "completion_rate": (
                len(self.completed_operations) / total_operations
                if total_operations > 0
                else 0
            ),
        }
