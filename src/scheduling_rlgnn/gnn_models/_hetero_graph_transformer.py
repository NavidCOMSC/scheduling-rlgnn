import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, TransformerConv, Linear
from torch_geometric.typing import Metadata
from typing import Any


class HeteroGraphTransformer(nn.Module):
    """
    Heterogeneous Graph Transformer architecture
    that handles multiple node and edge types using
    PyTorch Geometric's heterogeneous graph capabilities.
    """

    def __init__(
        self,
        metadata: Metadata,
        node_dims: dict[str, int],
        edge_dims: dict[tuple, int] | None = None,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_nodes: int = 100,
        aggr: str = "sum",
    ):
        """
        Initialize Heterogeneous Graph Transformer.

        Args:
            metadata: Graph metadata containing node and edge types
            node_dims: Dictionary mapping node types to their
            feature dimensions
            hidden_dim: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_nodes: Maximum number of nodes for positional encoding
            aggr: Aggregation method for heterogeneous convolutions
        """
        super().__init__()

        self.metadata = metadata
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.num_heads = num_heads

        # Determine edge_dims for TransformerConv
        self.edge_dims = edge_dims or {}
        self.use_edge_features = edge_dims is not None and len(edge_dims) > 0

        # Input projections for each node type
        self.input_projections = nn.ModuleDict()
        for node_type in self.node_types:
            input_dim = node_dims.get(node_type, hidden_dim)
            self.input_projections[node_type] = Linear(input_dim, hidden_dim)

        # Positional encodings for each node type
        self.pos_encodings = nn.ParameterDict()
        for node_type in self.node_types:
            self.pos_encodings[node_type] = nn.Parameter(
                torch.randn(max_nodes, hidden_dim)
            )

        # Edge Projections
        self.edge_projections = nn.ModuleDict()
        if self.use_edge_features:
            for edge_type in self.edge_types:
                if edge_type in self.edge_dims:
                    edge_key = f"{edge_type[0]}__to__{edge_type[2]}__via__{edge_type[1]}"
                    self.edge_projections[edge_key] = Linear(
                        self.edge_dims[edge_type], hidden_dim
                    )

        # Heterogeneous transformer layers
        self.hetero_convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                # src_type, _, dst_type = edge_type
                conv_dict[edge_type] = TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=hidden_dim if self.use_edge_features else None,
                    beta=True,
                    concat=False,
                )

            hetero_conv = HeteroConv(conv_dict, aggr=aggr)
            self.hetero_convs.append(hetero_conv)

        # Layer normalization for each node type and layer
        self.layer_norms = nn.ModuleDict()
        for node_type in self.node_types:
            for layer_idx in range(num_layers):
                self.layer_norms[f"{node_type}_{layer_idx}"] = nn.LayerNorm(
                    hidden_dim
                )

        # Feedforward networks for each node type
        self.feedforwards = nn.ModuleList()
        for _ in range(num_layers):
            ff_dict = nn.ModuleDict()
            for node_type in self.node_types:
                ff_dict[node_type] = nn.Sequential(
                    Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
            self.feedforwards.append(ff_dict)

        # Output projections for each node type
        self.output_projections = nn.ModuleDict()
        for node_type in self.node_types:
            self.output_projections[node_type] = Linear(hidden_dim, hidden_dim)

        # Type embeddings to distinguish different node types
        self.type_embeddings = nn.ParameterDict()
        for node_type in self.node_types:
            self.type_embeddings[node_type] = nn.Parameter(
                torch.randn(hidden_dim)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[Any, torch.Tensor],
        edge_attr_dict: dict[Any, torch.Tensor] | None = None,
        batch_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through Heterogeneous Graph Transformer.

        Args:
            x_dict: Node features for each node type
            edge_index_dict: Edge indices for each edge type
            edge_attr_dict: Edge attributes (optional)
            batch_dict: Batch indices for each node type (optional)

        Returns:
            Dictionary of transformed node embeddings for each node type
        """
        # Project input features and add type embeddings
        h_dict = {}
        for node_type, x in x_dict.items():
            h = self.input_projections[node_type](x)

            # Add type embeddings
            h = h + self.type_embeddings[node_type].unsqueeze(0)

            # Add positional encodings
            h = self._add_positional_encoding(h, node_type, batch_dict)

            h_dict[node_type] = self.dropout(h)

        # Apply transformer layers
        for layer_idx in range(len(self.hetero_convs)):
            # Heterogeneous convolution (attention)
            h_new_dict = self.hetero_convs[layer_idx](
                h_dict,
                edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )

            # Add residual connections and layer norm
            for node_type in self.node_types:
                if node_type in h_new_dict:
                    # Residual connection
                    h_residual = h_dict[node_type] + h_new_dict[node_type]
                    h_residual = self.layer_norms[f"{node_type}_{layer_idx}"](
                        h_residual
                    )
                    h_ff = getattr(self.feedforwards[layer_idx], node_type)(
                        h_residual
                    )

                    # Another residual connection
                    h_dict[node_type] = h_residual + h_ff
                    h_dict[node_type] = self.dropout(h_dict[node_type])

        # Apply output projections
        output_dict = {}
        for node_type, h in h_dict.items():
            output_dict[node_type] = self.output_projections[node_type](h)

        return output_dict

    def _add_positional_encoding(
        self,
        h: torch.Tensor,
        node_type: str,
        batch_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Add positional encoding to node features."""
        num_nodes = h.size(0)

        if batch_dict is not None and node_type in batch_dict:
            batch = batch_dict[node_type]
            batch_size = len(torch.unique(batch))
            nodes_per_batch = num_nodes // batch_size

            if nodes_per_batch <= self.max_nodes:
                pos_enc = self.pos_encodings[node_type][:nodes_per_batch]
                # Repeat for each graph in the batch
                pos_enc = pos_enc.unsqueeze(0).repeat(batch_size, 1, 1)
                pos_enc = pos_enc.view(-1, self.hidden_dim)
            else:
                # Interpolate for larger graphs
                pos_enc = (
                    F.interpolate(
                        self.pos_encodings[node_type]
                        .unsqueeze(0)
                        .transpose(0, 1),
                        size=nodes_per_batch,
                        mode="linear",
                        align_corners=False,
                    )
                    .transpose(0, 1)
                    .squeeze(0)
                )
                pos_enc = pos_enc.unsqueeze(0).repeat(batch_size, 1, 1)
                pos_enc = pos_enc.view(-1, self.hidden_dim)
        else:
            # Single graph case
            if num_nodes <= self.max_nodes:
                pos_enc = self.pos_encodings[node_type][:num_nodes]
            else:
                pos_enc = (
                    F.interpolate(
                        self.pos_encodings[node_type]
                        .unsqueeze(0)
                        .transpose(0, 1),
                        size=num_nodes,
                        mode="linear",
                        align_corners=False,
                    )
                    .transpose(0, 1)
                    .squeeze(0)
                )
        return h + pos_enc

    def get_attention_weights(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[Any, torch.Tensor],
        edge_attr_dict: dict[Any, torch.Tensor] | None = None,
        layer_idx: int = -1,
    ) -> dict[Any, torch.Tensor]:
        """
        Extract attention weights from a specific layer.

        Args:
            x_dict: Node features
            edge_index_dict: Edge indices
            edge_attr_dict: Edge attributes (optional)
            layer_idx: Layer index to extract weights from (-1 for last layer)

        Returns:
            Dictionary of attention weights for each edge type
        """
        if layer_idx == -1:
            layer_idx = len(self.hetero_convs) - 1

        # Forward pass up to the specified layer
        h_dict = {}
        for node_type, x in x_dict.items():
            h = self.input_projections[node_type](x)
            h = h + self.type_embeddings[node_type].unsqueeze(0)
            h_dict[node_type] = h

        for i in range(layer_idx + 1):
            if i == layer_idx:
                # Extract attention weights at this layer
                attention_weights = {}

                # Access the HeteroConv layer and its constituent convolutions
                hetero_conv_layer = self.hetero_convs[i]

                for edge_type in self.edge_types:
                    edge_type_str = "__".join(edge_type)
                    if hasattr(hetero_conv_layer.convs, edge_type_str):
                        conv = getattr(hetero_conv_layer.convs, edge_type_str)
                        src_type, _, dst_type = edge_type

                        if edge_type in edge_index_dict:
                            edge_index = edge_index_dict[edge_type]
                            edge_attr = (
                                edge_attr_dict.get(edge_type)
                                if edge_attr_dict
                                else None
                            )

                            # Get source and destination node features
                            x_src = h_dict[src_type]
                            x_dst = (
                                h_dict[dst_type]
                                if dst_type != src_type
                                else x_src
                            )

                            # Perform forward pass through
                            # the specific conv layer
                            with torch.no_grad():
                                # Set conv to return attention weights
                                conv.return_attention_weights = True
                                try:
                                    _, (_, alpha) = conv(
                                        (x_src, x_dst),
                                        edge_index,
                                        edge_attr=edge_attr,
                                        return_attention_weights=True,
                                    )
                                    attention_weights[edge_type] = alpha
                                except Exception as e:
                                    print(
                                        f"Error extracting attention"
                                        f" weights for {edge_type}: {e}"
                                    )
                                    # Fallback if attention extraction fails
                                    attention_weights[edge_type] = edge_index
                                finally:
                                    conv.return_attention_weights = False

                return attention_weights
            else:
                h_dict = self.hetero_convs[i](
                    h_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
                )

        return {}

    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
