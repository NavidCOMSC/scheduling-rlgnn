import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv,
    GATConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    BatchNorm,
)


class HeterogeneousGraphAttentionNetwork(nn.Module):
    """
    Heterogeneous Graph Attention Network for Job Shop Scheduling Problems.

    Handles different node types (operations, machines, jobs) and edge types
    (precedence, assignment, machine availability, etc.) with specialized
    attention mechanisms for each relationship type.
    """

    def __init__(
        self,
        # Node type configurations
        node_types: list[str] = ["operation", "machine", "job"],
        node_dims: dict[str, int] = {
            "operation": 64,
            "machine": 32,
            "job": 16,
        },
        # Edge type configurations
        edge_types: list[tuple[str, str, str]] = [
            ("operation", "precedence", "operation"),
            ("operation", "assigned_to", "machine"),
            ("machine", "can_process", "operation"),
            ("job", "contains", "operation"),
            ("operation", "belongs_to", "job"),
        ],
        edge_dims: dict[tuple[str, str, str], int] = {},
        # Model architecture
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "relu",
        # Pooling configurations
        use_global_pool: bool = False,
        pool_type: str = "mean",  # mean, max, or add
        pool_node_types: list[str] = ["operation"],  # Which node types to pool
        # Output configurations
        output_node_types: list[str] = [
            "operation"
        ],  # Which node types to output
    ):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_global_pool = use_global_pool
        self.pool_node_types = pool_node_types
        self.output_node_types = output_node_types

        # Initialize edge dimensions if not provided
        self.edge_dims = {}
        for edge_type in edge_types:
            if edge_type in edge_dims:
                self.edge_dims[edge_type] = edge_dims[edge_type]
            else:
                self.edge_dims[edge_type] = 32  # default edge dimension

        # Input projections for each node type
        self.node_projections = nn.ModuleDict()
        for node_type in node_types:
            input_dim = node_dims.get(node_type, 64)
            self.node_projections[node_type] = nn.Linear(input_dim, hidden_dim)

        # Edge projections for each edge type
        self.edge_projections = nn.ModuleDict()
        for edge_type in edge_types:
            edge_key = (
                f"{edge_type[0]}__to__{edge_type[2]}__via__{edge_type[1]}"
            )
            edge_dim = self.edge_dims[edge_type]
            if edge_dim > 0:
                self.edge_projections[edge_key] = nn.Linear(
                    edge_dim, hidden_dim
                )

        # Heterogeneous convolution layers
        self.hetero_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            # Create convolutions for each edge type
            conv_dict = {}
            for edge_type in edge_types:
                src_type, rel_type, dst_type = edge_type
                edge_key = f"{src_type}__to__{dst_type}__via__{rel_type}"

                in_dim = hidden_dim
                out_dim = (
                    hidden_dim // num_heads
                    if i < num_layers - 1
                    else hidden_dim
                )

                # Use edge features if available
                edge_dim = (
                    hidden_dim if self.edge_dims[edge_type] > 0 else None
                )

                conv_dict[edge_type] = GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=num_heads if i < num_layers - 1 else 1,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=i < num_layers - 1,
                )

            self.hetero_convs.append(HeteroConv(conv_dict, aggr="add"))

            # Batch normalization for each node type
            bn_dict = {}
            for node_type in node_types:
                final_dim = (
                    (hidden_dim // num_heads) * num_heads
                    if i < num_layers - 1
                    else hidden_dim
                )
                bn_dict[node_type] = BatchNorm(final_dim)
            self.batch_norms.append(nn.ModuleDict(bn_dict))

            # Global pooling functions
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

        # Output projections for specified node types
        self.output_projections = nn.ModuleDict()
        for node_type in output_node_types:
            self.output_projections[node_type] = nn.Linear(
                hidden_dim, hidden_dim
            )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
        batch_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Forward pass for the Heterogeneous Graph Attention Network.

        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            edge_attr_dict: Dictionary of edge attributes for each edge type
            batch_dict: Dictionary of batch indices for each node type

        Returns:
            Dictionary of node embeddings for each output node type, or
            pooled graph-level representation if use_global_pool is True
        """
        # Project input features for each node type
        for node_type in self.node_types:
            if node_type in x_dict:
                x_dict[node_type] = self.node_projections[node_type](
                    x_dict[node_type]
                )

        # Project edge features for each edge type
        if edge_attr_dict is not None:
            processed_edge_attr = {}
            for edge_type in self.edge_types:
                edge_key = (
                    f"{edge_type[0]}__to__{edge_type[2]}__via__{edge_type[1]}"
                )
                if (
                    edge_type in edge_attr_dict
                    and edge_key in self.edge_projections
                ):
                    processed_edge_attr[edge_type] = self.edge_projections[
                        edge_key
                    ](edge_attr_dict[edge_type])
        else:
            processed_edge_attr = None

        # Apply heterogeneous GAT layers
        for i, (hetero_conv, batch_norm_dict) in enumerate(
            zip(self.hetero_convs, self.batch_norms)
        ):
            # Store residual connections
            residual_dict = {
                node_type: x_dict[node_type].clone()
                for node_type in x_dict.keys()
            }

            # Apply heterogeneous convolution
            x_dict = hetero_conv(x_dict, edge_index_dict, processed_edge_attr)

            # Apply batch normalization and activation
            for node_type in x_dict.keys():
                x_dict[node_type] = getattr(batch_norm_dict, node_type)(
                    x_dict[node_type]
                )

                # Apply activation except for the last layer
                if i < len(self.hetero_convs) - 1:
                    x_dict[node_type] = self.activation(x_dict[node_type])
                    x_dict[node_type] = F.dropout(
                        x_dict[node_type],
                        p=self.dropout,
                        training=self.training,
                    )

                # Residual connection (if dimensions match)
                if node_type in residual_dict and residual_dict[
                    node_type
                ].size(-1) == x_dict[node_type].size(-1):
                    x_dict[node_type] = (
                        x_dict[node_type] + residual_dict[node_type]
                    )

        # Apply output projections
        output_dict = {}
        for node_type in self.output_node_types:
            if node_type in x_dict:
                output_dict[node_type] = self.output_projections[node_type](
                    x_dict[node_type]
                )

        # Apply global pooling if requested
        if self.use_global_pool and batch_dict is not None:
            pooled_features = []
            for node_type in self.pool_node_types:
                if node_type in output_dict and node_type in batch_dict:
                    pooled = self.global_pool(
                        output_dict[node_type], batch_dict[node_type]
                    )
                    pooled_features.append(pooled)

            if pooled_features:
                return torch.cat(pooled_features, dim=-1)
            else:
                return output_dict

        return output_dict

    def get_attention_weights(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
        layer_idx: int = -1,
    ) -> dict[tuple[str, str, str], torch.Tensor]:
        """
        Extract attention weights from a specific layer.

        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            edge_attr_dict: Dictionary of edge attributes for each edge type
            layer_idx: Index of layer to extract attention from
            (-1 for last layer)

        Returns:
            Dictionary of attention weights for each edge type
        """
        if layer_idx < 0:
            layer_idx = len(self.hetero_convs) + layer_idx

        if layer_idx >= len(self.hetero_convs) or layer_idx < 0:
            raise ValueError(
                f"Invalid layer_idx: {layer_idx}. Must be in range [0, "
                f"{len(self.hetero_convs)-1}]"
            )

        # Project input features for each node type
        current_x_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict:
                current_x_dict[node_type] = self.node_projections[node_type](
                    x_dict[node_type]
                )

        # Project edge features for each edge type
        processed_edge_attr = None
        if edge_attr_dict is not None:
            processed_edge_attr = {}
            for edge_type in self.edge_types:
                edge_key = (
                    f"{edge_type[0]}__to__{edge_type[2]}__via__{edge_type[1]}"
                )
                if (
                    edge_type in edge_attr_dict
                    and edge_key in self.edge_projections
                ):
                    processed_edge_attr[edge_type] = self.edge_projections[
                        edge_key
                    ](edge_attr_dict[edge_type])

        # Forward pass up to the target layer
        for i in range(layer_idx + 1):
            hetero_conv = self.hetero_convs[i]
            batch_norm_dict = self.batch_norms[i]

            # Store residual connections
            residual_dict = {
                node_type: current_x_dict[node_type].clone()
                for node_type in current_x_dict.keys()
            }

            if i == layer_idx:
                # Extract attention weights from the target layer
                attention_weights = {}

                # Apply each GAT conv individually to get attention weights
                for edge_type in self.edge_types:
                    if edge_type in edge_index_dict:
                        src_type, rel_type, dst_type = edge_type

                        # Get the specific GAT convolution for this edge type
                        gat_conv = None

                        # Access the ModuleDict
                        if hasattr(hetero_conv, "convs") and isinstance(
                            hetero_conv.convs, nn.ModuleDict
                        ):
                            # Try direct access with string key
                            edge_key = (
                                f"{src_type}__to__{dst_type}__via__{rel_type}"
                            )
                            if edge_key in hetero_conv.convs:
                                gat_conv = hetero_conv.convs[edge_key]
                            else:
                                # Try with other string representations
                                edge_key_variants = [
                                    f"{src_type}__{rel_type}__{dst_type}",
                                    f"({src_type}, {rel_type}, {dst_type})",
                                ]

                                for variant in edge_key_variants:
                                    if variant in hetero_conv.convs:
                                        gat_conv = hetero_conv.convs[variant]
                                        break

                        # Get source and destination node features
                        x_src = current_x_dict.get(src_type, None)
                        x_dst = current_x_dict.get(dst_type, None)

                        if (
                            gat_conv is not None
                            and x_src is not None
                            and x_dst is not None
                        ):
                            # Get edge indices and attributes
                            edge_index = edge_index_dict[edge_type]
                            edge_attr = (
                                processed_edge_attr.get(edge_type, None)
                                if processed_edge_attr
                                else None
                            )

                            # Create input tuple for GAT conv
                            if src_type == dst_type:
                                x_input = x_src
                            else:
                                x_input = (x_src, x_dst)

                            # Forward pass through GAT
                            # with return_attention_weights=True
                            try:
                                if edge_attr is not None:
                                    _, attention_weights[edge_type] = gat_conv(
                                        x_input,
                                        edge_index,
                                        edge_attr=edge_attr,
                                        return_attention_weights=True,
                                    )
                                else:
                                    _, attention_weights[edge_type] = gat_conv(
                                        x_input,
                                        edge_index,
                                        return_attention_weights=True,
                                    )
                            except (
                                TypeError,
                                ValueError,
                                RuntimeError,
                                AttributeError,
                            ):
                                # If attention extraction fails, store None
                                attention_weights[edge_type] = None
                        else:
                            attention_weights[edge_type] = None

                return attention_weights
            else:
                # Regular forward pass for layers before target
                current_x_dict = hetero_conv(
                    current_x_dict, edge_index_dict, processed_edge_attr
                )

                # Apply batch normalization and activation
                for node_type in current_x_dict.keys():
                    current_x_dict[node_type] = getattr(
                        batch_norm_dict, node_type
                    )(current_x_dict[node_type])

                    if i < len(self.hetero_convs) - 1:
                        current_x_dict[node_type] = self.activation(
                            current_x_dict[node_type]
                        )
                        current_x_dict[node_type] = F.dropout(
                            current_x_dict[node_type],
                            p=self.dropout,
                            training=self.training,
                        )

                    # Residual connection (if dimensions match)
                    if node_type in residual_dict and residual_dict[
                        node_type
                    ].size(-1) == current_x_dict[node_type].size(-1):
                        current_x_dict[node_type] = (
                            current_x_dict[node_type]
                            + residual_dict[node_type]
                        )

        return {}
