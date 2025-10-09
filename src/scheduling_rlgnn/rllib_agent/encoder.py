import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    Encoder class using a Multi-Layer Perceptron (MLP) architecture for the graph representation of the observations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: str = "relu",
    ):
        """
        Initialize the MLP encoder.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()

        activation_fn = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }[activation]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    activation_fn(),
                ]
            )
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x):
        """
        Forward pass through the MLP encoder.

        Args:
            x: Input features tensor

        Returns:
            Encoded features tensor
        """
        return self.network(x)
