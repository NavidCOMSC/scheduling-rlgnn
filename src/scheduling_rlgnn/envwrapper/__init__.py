"""Environment wrapper module for Job Shop Scheduling with RL-GNN.

This module provides wrapper classes for interfacing between Job
Shop Scheduling instances and reinforcement learning algorithms
using graph neural networks.
"""

from ._job_shop_environment_wrapper import JobShopEnvironmentWrapper

__all__ = ["JobShopEnvironmentWrapper"]

# Version info
__version__ = "0.1.0"
