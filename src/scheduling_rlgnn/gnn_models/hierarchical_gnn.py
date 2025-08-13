import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GraphConv,
    SAGEConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    BatchNorm,
    LayerNorm,
)
from typing import Dict, List, Optional, Tuple, Union


class HierarchicalGNN(nn.Module):
    """
    Hierarchical Graph Neural Network that captures both local and global patterns.
    Processes job-level and machine-level representations separately then combines them.
    """

    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 3,
        pooling_method: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pooling_method = pooling_method

        # Node encoders for different node types
        self.operation_encoder = nn.Linear(node_dim, hidden_dim)
        self.machine_encoder = nn.Linear(node_dim, hidden_dim)
        self.job_encoder = nn.Linear(hidden_dim, hidden_dim)

        # Edge encoder
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # Local GNN for operation-level interactions
        self.local_gnn = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Global GNN for job/machine-level interactions
        self.global_gnn = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Attention pooling
        if pooling_method == "attention":
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )

        # Cross-level interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Final projection
        self.final_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with hierarchical processing.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Graph connectivity.
            edge_attr (torch.Tensor | None, optional): Edge features. Defaults to None.
            batch (torch.Tensor | None, optional): Batch vector. Defaults to None.
            node_types (torch.Tensor | None, optional): Node types. Defaults to None.
        """

        if node_types is None:
            # Assume first half are operations, second half are machines
            num_nodes = x.size(0)
            node_types = torch.cat(
                [
                    torch.zeros(
                        num_nodes // 2, dtype=torch.long, device=x.device
                    ),
                    torch.ones(
                        num_nodes - num_nodes // 2,
                        dtype=torch.long,
                        device=x.device,
                    ),
                ]
            )

        # Encode different node types
        operation_mask = node_types == 0
        machine_mask = node_types == 1
        job_mask = node_types == 2

        x_encoded = torch.zeros_like(x[:, : self.hidden_dim])

        if operation_mask.any():
            x_encoded[operation_mask] = self.operation_encoder(
                x[operation_mask]
            )
        if machine_mask.any():
            x_encoded[machine_mask] = self.machine_encoder(x[machine_mask])
        if job_mask.any():
            x_encoded[job_mask] = self.job_encoder(x[job_mask])

        # Encode edges
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        # Local processing (operation level)
        x_local = x_encoded.clone()
        for gnn_layer in self.local_gnn:
            x_local = gnn_layer(x_local, edge_index, edge_attr)
            x_local = F.relu(x_local)
            x_local = F.dropout(
                x_local, p=self.dropout, training=self.training
            )

        # Global processing (higher level)
        x_global = self._create_global_graph(x_encoded, batch)

        # Combine local and global representations
        x_combined = self._combine_representations(x_local, x_global, batch)

        return x_combined

    def _create_global_graph(
        self, x: torch.Tensor, batch: torch.Tensor | None
    ) -> torch.Tensor:
        """Create global graph representation through pooling."""

        if batch is None:
            # Single graph case
            if self.pooling_method == "attention":
                x_pooled, _ = self.attention_pool(
                    x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0)
                )
                return x_pooled.squeeze(0)
            else:
                return global_mean_pool(
                    x,
                    torch.zeros(x.size(0), dtype=torch.long, device=x.device),
                )

        else:
            # Batched case
            if self.pooling_method == "attention":
                # Group by batch and apply attention pooling
                unique_batches = torch.unique(batch)
                pooled_representations = []

                for batch_idx in unique_batches:
                    mask = batch == batch_idx
                    batch_nodes = x[mask].unsqueeze(0)
                    pooled, _ = self.attention_pool(
                        batch_nodes, batch_nodes, batch_nodes
                    )
                    pooled_representations.append(pooled.squeeze(0))

                return torch.cat(pooled_representations, dim=0)
            else:
                # Group by batch and apply mean pooling
                return global_mean_pool(x, batch)

    def _combine_representations(
        self,
        x_local: torch.Tensor,
        x_global: torch.Tensor,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        """Combine local and global representations using cross-attention."""

        if batch is None:
            # Single graph case
            local_seq = x_local.unsqueeze(0)
            global_seq = x_global.unsqueeze(0)
        else:
            # Batched case - need to handle variable graph sizes
            unique_batches = torch.unique(batch)
            combined_representations = []

            for batch_idx in unique_batches:
                mask = batch == batch_idx
                local_batch = x_local[mask].unsqueeze(0)

                # Expand global representation to match local sequence length
                global_batch = (
                    x_global[batch_idx]
                    .unsqueeze(0)
                    .expand(1, local_batch.size(1), -1)
                )

                # Apply cross-attention
                attended_local, _ = self.cross_attention(
                    local_batch, global_batch, global_batch
                )

                # Combine via concatenation and projection
                combined = torch.cat(
                    [attended_local.squeeze(0), local_batch.squeeze(0)], dim=-1
                )
                combined = self.final_proj(combined)

                combined_representations.append(combined)

            # Concatenate all combined representations for the current batch
            return torch.cat(combined_representations, dim=0)

        # Single graph case
        attended_local, _ = self.cross_attention(
            local_seq, global_seq, global_seq
        )
        combined = torch.cat([attended_local.squeeze(0), x_local], dim=-1)
        return self.final_proj(combined)
