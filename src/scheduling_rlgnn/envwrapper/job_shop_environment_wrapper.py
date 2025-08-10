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
        # self.current_step = 0
        # self.completed_operations: set["Operation"] = set()
        # self.machine_schedules: Dict[int, List[Dict[str, Any]]] = {}
        # self.current_time = 0
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
        """implementation for processing the action to
        select the operation to schedule.
        """

        # Get available operations
        available_ops = self._get_available_operations()

        if not available_ops or action >= len(available_ops):
            return -10.0

        selected_operation = available_ops[action]
        machine_id = selected_operation.machine_id

        # Find earliest start time for this operation on the machine
        if self.machine_schedules[machine_id]:
            last_end_time = self.machine_schedules[machine_id][-1]["end_time"]
        else:
            last_end_time = 0

        # Start time must be >= max(global time, machine's last end time)
        start_time = max(self.current_time, last_end_time)
        end_time = start_time + selected_operation.duration

        # Update state
        self.machine_schedules[machine_id].append(
            {
                "operation": selected_operation,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

        self.completed_operations.add(selected_operation)
        self.current_time = max(
            self.current_time, end_time
        )  # Update global time

        return self._calculate_reward(selected_operation, start_time, end_time)

    def _get_job_id(self, operation: Operation) -> int:
        """Get job ID for a given operation."""
        for job_id, job in enumerate(self.instance.jobs):
            if operation in job:
                return job_id
        return -1

    def _calculate_reward(
        self, operation: Operation, start_time: float, end_time: float
    ) -> float:
        """Calculate reward for scheduling an operation."""
        # Simple reward: negative of completion time (encourages earlier completion)
        base_reward = -end_time

        # Bonus for completing operations without delay
        if start_time == self.current_time:
            base_reward += 1.0

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
        makespan = max(
            [
                max([op["end_time"] for op in schedule], default=0)
                for schedule in self.machine_schedules.values()
            ],
            default=0,
        )

        return {
            "makespan": makespan,
            "completed_operations": len(self.completed_operations),
            "total_operations": sum(len(job) for job in self.instance.jobs),
            "current_time": self.current_time,
            "completion_rate": len(self.completed_operations)
            / sum(len(job) for job in self.instance.jobs),
        }
