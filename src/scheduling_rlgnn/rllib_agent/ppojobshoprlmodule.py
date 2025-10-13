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
            num_actions = int(self.action_space.n)
        else:
            raise ValueError(
                f"Unsupported action space: {type(self.action_space)}"
            )

        # Encoder configuration
        hidden_dims = config.get("fcnet_hiddens", [256, 256])
        activation = config.get("fcnet_activation", "relu")

        # Shared encoder network
        self.encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )

        # Policy head (actor)
        self.policy_head = PolicyHead(
            input_dim=self.encoder.output_dim,
            num_actions=num_actions,
        )

        # Value head (critic)
        self.value_head = ValueHead(input_dim=self.encoder.output_dim)

    def _preprocess_observations(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess observations from the environment.

        Args:
            batch: Batch of observations

        Returns:
            Processed tensor ready for the encoder
        """
        obs = batch["obs"]

        if isinstance(self.observation_space, gym.spaces.Box):
            # Flatten if necessary
            if len(obs.shape) > 2:
                obs = obs.reshape(obs.shape[0], -1)
            return obs

        elif isinstance(self.observation_space, gym.spaces.Dict):
            # Handle dictionary observations (graph-based)
            if "node_features" in obs:
                # Use node features (may need aggregation for graph-level tasks)
                node_features = obs["node_features"]

                # If we have multiple nodes, we can use mean pooling
                if (
                    len(node_features.shape) == 3
                ):  # [batch, num_nodes, features]
                    # Mean pooling over nodes
                    return torch.mean(node_features, dim=1)
                else:
                    return node_features

            else:
                # Concatenate all features
                features = []
                for key in sorted(obs.keys()):
                    feat = obs[key]
                    if len(feat.shape) > 2:
                        feat = feat.reshape(feat.shape[0], -1)
                    features.append(feat)
                return torch.cat(features, dim=-1)

        return obs

    @override(TorchRLModule)
    def _forward_inference(
        self, batch: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass for inference (deployment/production).

        Args:
            batch: Input batch containing observations

        Returns:
            Dictionary containing action logits
        """

        # Preprocess observations
        obs_processed = self._preprocess_observations(batch)

        # Encode observations
        encoded = self.encoder(obs_processed)

        # Compute action logits
        action_logits = self.policy_head(encoded)

        return {"action_dist_inputs": action_logits}

    @override(TorchRLModule)
    def _forward_exploration(
        self, batch: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass for exploration (training data collection).

        Args:
            batch: Input batch containing observations

        Returns:
            Dictionary containing action logits
        """
        # For PPO, exploration is the same as inference
        return self._forward_inference(batch)

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for training (computing losses).

        Args:
            batch: Input batch containing observations

        Returns:
            Dictionary containing action logits and value predictions
        """

        # Preprocess observations
        obs_processed = self._preprocess_observations(batch)

        # Encode observations
        encoded = self.encoder(obs_processed)

        # Compute action logits
        action_logits = self.policy_head(encoded)

        # Compute value predictions
        values = self.value_head(encoded)

        return {
            "action_dist_inputs": action_logits,
            "values": values,
        }

    @override(RLModule)
    def get_train_action_dist_cls(self):
        """Return the action distribution class for training."""
        return TorchCategorical
