import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTransformer(nn.Module):
    """
    Graph Transformer architecture that treats the graph as a sequence of nodes
    and applies transformer-style attention with positional encodings.
    """

    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_nodes: int = 100,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # Positional encoding for graph structure
        self.pos_encoding = nn.Parameter(torch.randn(max_nodes, hidden_dim))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Graph structure encoding
        self.struct_encoder = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through Graph Transformer."""

        batch_size = 1 if batch is None else len(torch.unique(batch))
        num_nodes = x.size(0) // batch_size

        # Project input features
        x = self.input_proj(x)

        # Add positional encoding
        if num_nodes <= self.max_nodes:
            pos_enc = (
                self.pos_encoding[:num_nodes]
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            x = x.view(batch_size, num_nodes, -1) + pos_enc
        else:
            # Interpolate positional encoding for larger graphs
            pos_enc = (
                F.interpolate(
                    self.pos_encoding.unsqueeze(0).transpose(1, 2),
                    size=num_nodes,
                    mode="linear",
                    align_corners=False,
                )
                .transpose(1, 2)
                .expand(batch_size, -1, -1)
            )
            x = x.view(batch_size, num_nodes, -1) + pos_enc

        # Encode graph structure information
        struct_info = self._encode_graph_structure(
            edge_index, (batch_size, num_nodes, x.size(-1))
        )
        x = x + struct_info

        # Apply dropout
        x = self.dropout(x)

        # Create attention mask based on graph connectivity
        attn_mask = self._create_attention_mask(
            edge_index, num_nodes, batch_size
        )

        # Apply transformer
        x = self.transformer(x, mask=attn_mask)

        # Reshape back to node-level representation
        x = x.view(-1, self.hidden_dim)

        return self.output_proj(x)

    def _encode_graph_structure(
        self, edge_index: torch.Tensor, shape: tuple[int, int, int]
    ) -> torch.Tensor:
        """Encode graph connectivity as structural features."""
        batch_size, num_nodes, hidden_dim = shape

        # Create adjacency matrix encoding
        adj_encoding = torch.zeros(
            batch_size, num_nodes, num_nodes, device=edge_index.device
        )

        # Fill adjacency matrix
        if batch_size == 1:
            adj_encoding[0, edge_index[0], edge_index[1]] = 1.0
        else:
            # Handle batched case
            for i in range(batch_size):
                batch_mask = (edge_index[0] // num_nodes) == i
                if batch_mask.any():
                    batch_edges = edge_index[:, batch_mask]
                    batch_edges = batch_edges % num_nodes  # Adjust indices
                    adj_encoding[i, batch_edges[0], batch_edges[1]] = 1.0

        # Convert adjacency to structural encoding
        struct_encoding = torch.matmul(
            adj_encoding,
            torch.randn(num_nodes, hidden_dim, device=edge_index.device),
        )

        return struct_encoding

    def _create_attention_mask(
        self, edge_index: torch.Tensor, num_nodes: int, batch_size: int
    ) -> torch.Tensor | None:
        """Create attention mask to respect graph structure."""
        # At the first instance allows for the full connected network
        # it can be modified to select specific edges or nodes

        return None
