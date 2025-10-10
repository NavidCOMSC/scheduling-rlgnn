import torch.nn as nn
import torch


class PolicyHead(nn.Module):
    """
    Policy head class for outputting action logits from encoded features.
    """

    def __init__(self, input_dim: int, num_actions: int):
        """
        Initialize the policy head.

        Args:
            input_dim: Dimension of input features
            num_actions: Dimension of output action logits
        """
        super().__init__()
        self.policy_layer = nn.Linear(input_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy head.

        Args:
            x: Input features tensor

        Returns:
            Action logits tensor
        """
        return self.policy_layer(x)
