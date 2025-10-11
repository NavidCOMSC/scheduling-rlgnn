from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import gymnasium as gym

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


class PPOJobShopRLModule(TorchRLModule):
    """
    PPO RLModule for Job Shop Scheduling with MultiJobShopGraphEnv.

    This module implements the actor-critic architecture required for PPO,
    with separate policy and value networks sharing a common encoder.

    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the PPOJobShopRLModule.

        Args:
            observation_space: The observation space of the environment.
            action_space: The action space of the environment.
            model_config: Optional dictionary containing model configuration parameters.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            **kwargs,
        )

        @override(TorchRLModule)
        def setup(self):
            """
            Setup the neural network architecture.
            """
            # Get model configuration with defaults
            config = self.model_config or {}

            # Handle different observation space types
            if isinstance(self.observation_space, gym.spaces.Box):
                # Flatten the observation space
                input_dim = int(
                    torch.prod(torch.tensor(self.observation_space.shape))
                )
            elif isinstance(self.observation_space, gym.spaces.Dict):
                # For graph-based observations, we'll handle the node features
                # Assuming 'node_features' key exists in the observation dict
                if "node_features" in self.observation_space.spaces:
                    node_feat_space = self.observation_space.spaces[
                        "node_features"
                    ]
                    input_dim = int(
                        torch.prod(torch.tensor(node_feat_space.shape))
                    )
                else:
                    # Fallback: sum all feature dimensions
                    input_dim = sum(
                        int(torch.prod(torch.tensor(space.shape)))
                        for space in self.observation_space.spaces.values()
                        if isinstance(space, gym.spaces.Box)
                    )
            else:
                raise ValueError(
                    f"Unsupported observation space: {type(self.observation_space)}"
                )

            # Get action space dimension
            if isinstance(self.action_space, gym.spaces.Discrete):
                num_actions = self.action_space.n
            else:
                raise ValueError(
                    f"Unsupported action space: {type(self.action_space)}"
                )
