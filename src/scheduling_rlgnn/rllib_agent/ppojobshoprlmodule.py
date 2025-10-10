from typing import Any, Dict, Optional
import gymnasium as gym
import torch
import torch.nn as nn

from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

from scheduling_rlgnn.rllib_agent.mlpencoder import MLPEncoder
from scheduling_rlgnn.rllib_agent.policyhead import PolicyHead
from scheduling_rlgnn.rllib_agent.valuehead import ValueHead

torch, nn = try_import_torch()


class PPOJobShopRLModule(TorchRLModule):
    """
    PPO RLModule for Job Shop Scheduling with MultiJobShopGraphEnv.

    This module implements the actor-critic architecture required for PPO,
    with separate policy and value networks sharing a common encoder.

    """
