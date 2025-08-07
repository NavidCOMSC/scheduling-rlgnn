from dataclasses import dataclass
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from torch_geometric.data import Data, Batch
from typing import Any, Tuple
import numpy as np
import torch


@dataclass
class JobShopInstance:
    """Represents a Job Shop Scheduling instance"""

    num_jobs: int
    num_machines: int
    processing_times: np.ndarray  # shape: (num_jobs, num_machines)
    machine_sequences: np.ndarray  # shape: (num_jobs, num_machines)

    def __post_init__(self):
        assert self.processing_times.shape == (
            self.num_jobs,
            self.num_machines,
        )
        assert self.machine_sequences.shape == (
            self.num_jobs,
            self.num_machines,
        )


class JobShopEnvironment(Env):
    """
    JobShopEnvironment is a custom environment for Job Shop Scheduling using a
    graph-based representation learning. This environment is designed for reinforcement
    learning tasks, where jobs must be scheduled on machines to minimize makespan or
    other objectives. The environment models jobs and machines as nodes in a graph,
    with node features representing processing time, machine/job IDs, start/completion
    times, and status.

    Attributes:
        num_jobs (int): Number of jobs in the scheduling problem.
        num_machines (int): Number of machines available for processing jobs.
        max_time_steps (int): Maximum number of time steps per operation.
        action_space (Discrete): Action space representing job-machine assignments.
        observation_space (Box): Observation space encoding node features and global state.

    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()

        self.num_jobs = config.get("num_jobs", 3)
        self.num_machines = config.get("num_machines", 3)
        self.max_time_steps = config.get("max_time_steps", 100)

        # Create action and observation spaces
        self.action_space = Discrete(self.num_jobs * self.num_machines)

        # Observation space for node features
        max_nodes = self.num_jobs * self.num_machines + self.num_machines
        node_features = 6  # [processing_time, machine_id, job_id, start_time, completion_time, status]

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                max_nodes * node_features + 2,
            ),  # +2 for current_time and makespan
            dtype=np.float32,
        )

        self.reset()

    # TODO: This method should be adjusted to take the JSP instance
    def _generate_random_instance(self) -> JobShopInstance:
        """Generates a random Job Shop Scheduling instance."""
        processing_times = np.random.randint(
            1, 10, (self.num_jobs, self.num_machines)
        )

        # Generate random machine sequences for each job
        machine_sequences = np.array(
            [
                np.random.permutation(self.num_machines)
                for _ in range(self.num_jobs)
            ]
        )

        return JobShopInstance(
            num_jobs=self.num_jobs,
            num_machines=self.num_machines,
            processing_times=processing_times,
            machine_sequences=machine_sequences,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Generate new instance
        self.instance = self._generate_random_instance()

        # Reset scheduling state
        self.current_time = 0
        self.completed_operations = set()
        self.operation_start_times = {}
        self.operation_completion_times = {}
        self.machine_available_time = [0] * self.num_machines
        self.step_count = 0

        # Create graph representation
        graph_data = self._create_graph_representation()

        # Flatten observation
        obs = self._flatten_observation(graph_data)

        info = {"instance": self.instance}
        return obs, info

    def _create_graph_representation(self) -> Data:
        """Create PyG Data object representing the current scheduling state"""

        # Nodes: operations (job_id, operation_idx) + machines
        operation_nodes = []
        machine_nodes = []

        # Operation nodes
        for job_id in range(self.num_jobs):
            for op_idx in range(self.num_machines):
                machine_id = self.instance.machine_sequences[job_id, op_idx]
                processing_time = self.instance.processing_times[
                    job_id, op_idx
                ]

                # Get scheduling info
                start_time = self.operation_start_times.get(
                    (job_id, op_idx), -1
                )
                completion_time = self.operation_completion_times.get(
                    (job_id, op_idx), -1
                )
                status = (
                    1 if (job_id, op_idx) in self.completed_operations else 0
                )

                operation_nodes.append(
                    [
                        processing_time,
                        machine_id,
                        job_id,
                        start_time,
                        completion_time,
                        status,
                    ]
                )

        # Machine nodes
        for machine_id in range(self.num_machines):
            machine_availability = self.machine_available_time[machine_id]
            machine_nodes.append(
                [
                    0,  # processing_time (not applicable)
                    machine_id,
                    -1,  # job_id (not applicable)
                    machine_availability,
                    machine_availability,
                    1,  # always available
                ]
            )

        # Combine all nodes
        all_nodes = operation_nodes + machine_nodes
        node_features = torch.tensor(all_nodes, dtype=torch.float32)

        # Create edges
        edge_indices = []

        # Job precedence edges
        for job_id in range(self.num_jobs):
            for op_idx in range(self.num_machines - 1):
                current_op = job_id * self.num_machines + op_idx
                next_op = job_id * self.num_machines + op_idx + 1
                edge_indices.extend(
                    [[current_op, next_op], [next_op, current_op]]
                )

        # Machine assignment edges (operation to machine)
        for job_id in range(self.num_jobs):
            for op_idx in range(self.num_machines):
                op_node_idx = job_id * self.num_machines + op_idx
                machine_id = self.instance.machine_sequences[job_id, op_idx]
                machine_node_idx = (
                    self.num_jobs * self.num_machines + machine_id
                )
                edge_indices.extend(
                    [
                        [op_node_idx, machine_node_idx],
                        [machine_node_idx, op_node_idx],
                    ]
                )

        edge_index = (
            torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            if edge_indices
            else torch.empty((2, 0), dtype=torch.long)
        )

        return Data(x=node_features, edge_index=edge_index)
