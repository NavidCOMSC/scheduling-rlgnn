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
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class MultiScaleGNN(nn.Module):
    """
    Multi-scale Graph Neural Network that processes the graph at different scales
    and combines information across scales.
    """

    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        hidden_dim: int = 128,
        num_scales: int = 3,
        num_layers_per_scale: int = 2,
        aggregation: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation

        # Input projections
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = (
            nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None
        )

        # Multi-scale GNN layers
        self.scale_networks = nn.ModuleList()

        for scale in range(num_scales):
            scale_layers = nn.ModuleList()

            for layer in range(num_layers_per_scale):
                if scale == 0:  # Finest scale - use standard convolution
                    scale_layers.append(GraphConv(hidden_dim, hidden_dim))
                elif scale == 1:  # Medium scale - use SAGE
                    scale_layers.append(SAGEConv(hidden_dim, hidden_dim))
                else:  # Coarsest scale - use GAT
                    scale_layers.append(
                        GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
                    )

            self.scale_networks.append(scale_layers)

        # Scale aggregation
        if aggregation == "attention":
            self.scale_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
        elif aggregation == "weighted":
            self.scale_weights = nn.Parameter(
                torch.ones(num_scales) / num_scales
            )

        # Final layers
        self.final_norm = LayerNorm(hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through multi-scale GNN."""

        # Project inputs
        x = self.node_proj(x)
        if edge_attr is not None and self.edge_proj is not None:
            edge_attr = self.edge_proj(edge_attr)

        # Process at each scale
        scale_outputs = []

        for scale, scale_layers in enumerate(self.scale_networks):
            x_scale = x

            # Apply coarsening for higher scales
            if scale > 0:
                x_scale, edge_index_scale = self._coarsen_graph(
                    x_scale, edge_index, scale
                )
            else:
                edge_index_scale = edge_index

            # Apply GNN layers for this scale
            for layer in scale_layers:
                if isinstance(layer, GATConv):
                    x_scale = layer(x_scale, edge_index_scale)
                else:
                    x_scale = layer(x_scale, edge_index_scale, edge_attr)

                x_scale = F.relu(x_scale)
                x_scale = F.dropout(
                    x_scale, p=self.dropout, training=self.training
                )

            # Upsample back to original resolution if needed
            if scale > 0:
                x_scale = self._upsample_to_original(x_scale, x.size(0))

            scale_outputs.append(x_scale)

        # Aggregate across scales
        if self.aggregation == "attention":
            x_combined = self._attention_aggregate(scale_outputs)
        elif self.aggregation == "weighted":
            x_combined = self._weighted_aggregate(scale_outputs)
        else:
            x_combined = torch.stack(scale_outputs, dim=0).mean(dim=0)

        # Final processing
        x_combined = self.final_norm(x_combined)
        x_combined = self.final_proj(x_combined)

        return x_combined

    def _coarsen_graph(
        self, x: torch.Tensor, edge_index: torch.Tensor, scale: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Coarsen graph for higher-scale processing."""
        coarsening_ratio = 2**scale
        num_nodes = x.size(0)

        # Simple node clustering by grouping consecutive nodes
        cluster_size = max(1, num_nodes // (num_nodes // coarsening_ratio))

        coarse_x = []
        coarse_edge_list = []
        for i in range(0, num_nodes, cluster_size):
            end_idx = min(i + cluster_size, num_nodes)
            cluster_nodes = torch.arange(i, end_idx, device=x.device)

            # Pool node features
            pooled_features = x[cluster_nodes].mean(dim=0)
            coarse_x.append(pooled_features)

        coarse_x = torch.stack(coarse_x)

        # Create coarse edges (simplified)
        num_coarse_nodes = coarse_x.size(0)
        coarse_edge_index = torch.combinations(
            torch.arange(num_coarse_nodes), r=2
        ).t()

        return coarse_x, coarse_edge_index

    def _upsample_to_original(
        self, x_coarse: torch.Tensor, target_size: int
    ) -> torch.Tensor:
        """Upsample coarse features back to original graph size."""
        coarse_size = x_coarse.size(0)

        if coarse_size == target_size:
            return x_coarse

        # Simple upsampling by repetition
        repeat_factor = target_size // coarse_size
        remainder = target_size % coarse_size

        upsampled = x_coarse.repeat_interleave(repeat_factor, dim=0)

        if remainder > 0:
            upsampled = torch.cat([upsampled, x_coarse[:remainder]], dim=0)

        return upsampled
