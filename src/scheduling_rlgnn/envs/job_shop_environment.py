from dataclasses import dataclass
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from typing import Dict, Any, Tuple, Optional
import numpy as np


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

    def __init__(self, config: Dict[str, Any]):
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
