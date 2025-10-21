import torch.nn as nn
import torch


class ValueHead(nn.Module):
    """
    Value head class for outputting state value from encoded features.
    """

    def __init__(self, input_dim: int):
        """
        Initialize the value head.

        Args:
            input_dim: Dimension of input features
        """
        super().__init__()
        self.value_layer = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value head.

        Args:
            x: Input features tensor

        Returns:
            State value tensor
        """
        return self.value_layer(x).squeeze(-1)
