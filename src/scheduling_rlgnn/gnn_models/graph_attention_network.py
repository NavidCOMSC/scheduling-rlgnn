import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    BatchNorm,
)


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for job shop scheduling.
    Uses multi-head attention to focus on relevant operations and machines.
    """

    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "relu",
        use_global_pool: bool = False,  # Add option for graph-level features
        pool_type: str = "mean",  # mean, max, or add
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_global_pool = use_global_pool

        # Input projections
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = (
            nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = (
                hidden_dim // num_heads if i < num_layers - 1 else hidden_dim
            )

            self.gat_layers.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=num_heads if i < num_layers - 1 else 1,
                    dropout=dropout,
                    edge_dim=hidden_dim if edge_dim > 0 else None,
                    concat=i < num_layers - 1,
                )
            )
            final_dim = out_dim * num_heads if i < num_layers - 1 else out_dim
            self.batch_norms.append(BatchNorm(final_dim))

        # Global pooling for graph-level representations
        if use_global_pool:
            if pool_type == "mean":
                self.global_pool = global_mean_pool
            elif pool_type == "max":
                self.global_pool = global_max_pool
            elif pool_type == "add":
                self.global_pool = global_add_pool
            else:
                self.global_pool = global_mean_pool

        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the Graph Attention Network.
        """

        # Project input features
        x = self.node_proj(x)
        if edge_attr is not None and self.edge_proj is not None:
            edge_attr = self.edge_proj(edge_attr)

        # Apply GAT layers
        for i, (gat_layer, batch_norm) in enumerate(
            zip(self.gat_layers, self.batch_norms)
        ):
            residual = x

            x = gat_layer(x, edge_index, edge_attr)
            x = batch_norm(x)

            # Apply activation except for the last layer
            if i < len(self.gat_layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if residual.size(-1) == x.size(-1):
                x = x + residual

        # Apply global pooling if requested
        if self.use_global_pool and batch is not None:
            x = self.global_pool(x, batch)

        return x
