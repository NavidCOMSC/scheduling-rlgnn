import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GraphConv,
    SAGEConv,
    LayerNorm,
)


class MultiScaleGNN(nn.Module):
    """
    Multi-scale Graph Neural Network that processes the graph at
    different scales and combines information across scales.
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
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
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
                x_scale, edge_index_scale, batch_scale = self._coarsen_graph(
                    x_scale, edge_index, scale, batch
                )
            else:
                edge_index_scale = edge_index
                batch_scale = batch

            # Apply GNN layers for this scale
            if isinstance(scale_layers, nn.ModuleList):
                layers_iter = scale_layers
            else:
                layers_iter = [scale_layers]

            for layer in layers_iter:
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
                x_scale = self._upsample_to_original(
                    x_scale, x.size(0), batch, batch_scale
                )

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
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        scale: int,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Coarsen graph for higher-scale processing."""
        coarsening_ratio = 2**scale
        num_nodes = x.size(0)

        # Simple node clustering by grouping consecutive nodes
        cluster_size = max(1, num_nodes // (num_nodes // coarsening_ratio))

        coarse_x = []
        coarse_batch = []
        node_mapping = {}

        if batch is not None:
            # Handle batched graphs
            unique_batches = batch.unique()
            coarse_node_idx = 0

            for batch_id in unique_batches:
                batch_mask = batch == batch_id
                batch_nodes = torch.where(batch_mask)[0]
                batch_x = x[batch_mask]

                for i in range(0, len(batch_nodes), cluster_size):
                    end_idx = min(i + cluster_size, len(batch_nodes))
                    cluster_indices = batch_nodes[i:end_idx]

                    # Map original nodes to coarse node
                    for orig_idx in cluster_indices:
                        node_mapping[orig_idx.item()] = coarse_node_idx

                    # Pool node features
                    pooled_features = batch_x[i:end_idx].mean(dim=0)
                    coarse_x.append(pooled_features)
                    coarse_batch.append(batch_id)
                    coarse_node_idx += 1

            coarse_batch = torch.tensor(coarse_batch, device=x.device)
        else:
            # Handle single graph
            coarse_node_idx = 0
            for i in range(0, num_nodes, cluster_size):
                end_idx = min(i + cluster_size, num_nodes)
                cluster_indices = torch.arange(i, end_idx, device=x.device)

                # Map original nodes to coarse node
                for orig_idx in cluster_indices:
                    node_mapping[orig_idx.item()] = coarse_node_idx

                # Pool node features
                pooled_features = x[cluster_indices].mean(dim=0)
                coarse_x.append(pooled_features)
                coarse_node_idx += 1

        coarse_batch = None

        coarse_x = torch.stack(coarse_x)

        # Create coarse edges based on original edge connectivity
        coarse_edge_set = set()
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            coarse_src = node_mapping.get(src)
            coarse_dst = node_mapping.get(dst)

            if (
                coarse_src is not None
                and coarse_dst is not None
                and coarse_src != coarse_dst
            ):
                coarse_edge_set.add((coarse_src, coarse_dst))
                coarse_edge_set.add((coarse_dst, coarse_src))  # Undirected

        if coarse_edge_set:
            coarse_edges = list(coarse_edge_set)
            coarse_edge_index = torch.tensor(coarse_edges, device=x.device).t()
        else:
            # Create coarse edges (simplified graph)
            num_coarse_nodes = coarse_x.size(0)
            coarse_edge_index = torch.combinations(
                torch.arange(num_coarse_nodes, device=coarse_x.device), r=2
            ).t()
            # Make undirected
            coarse_edge_index = torch.cat(
                [coarse_edge_index, coarse_edge_index.flip(0)], dim=1
            )

        return coarse_x, coarse_edge_index, coarse_batch

    def _upsample_to_original(
        self,
        x_coarse: torch.Tensor,
        target_size: int,
        original_batch: torch.Tensor | None = None,
        coarse_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Upsample coarse features back to original graph size."""
        coarse_size = x_coarse.size(0)

        if coarse_size == target_size:
            return x_coarse

        if original_batch is not None and coarse_batch is not None:
            # Handle batched upsampling
            upsampled = torch.zeros(
                target_size, x_coarse.size(1), device=x_coarse.device
            )

            for batch_id in original_batch.unique():
                orig_mask = original_batch == batch_id
                coarse_mask = coarse_batch == batch_id

                orig_count = orig_mask.sum().item()
                coarse_count = coarse_mask.sum().item()

                if coarse_count > 0:
                    coarse_features = x_coarse[coarse_mask]
                    repeat_factor = orig_count // coarse_count
                    remainder = orig_count % coarse_count

                    upsampled_batch = coarse_features.repeat_interleave(
                        repeat_factor, dim=0
                    )
                    if remainder > 0:
                        upsampled_batch = torch.cat(
                            [upsampled_batch, coarse_features[:remainder]],
                            dim=0,
                        )

                    upsampled[orig_mask] = upsampled_batch

            return upsampled
        else:

            # Simple upsampling by repetition
            repeat_factor = target_size // coarse_size
            remainder = target_size % coarse_size

            upsampled = x_coarse.repeat_interleave(repeat_factor, dim=0)

            if remainder > 0:
                upsampled = torch.cat([upsampled, x_coarse[:remainder]], dim=0)

            return upsampled

    def _attention_aggregate(
        self, scale_outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        """Aggregate scale outputs using attention mechanism."""
        # Stack outputs along scale dimension
        stacked = torch.stack(scale_outputs, dim=1)

        # Apply self-attention across scales
        attended, _ = self.scale_attention(stacked, stacked, stacked)

        # Average across scales
        return attended.mean(dim=1)

    def _weighted_aggregate(
        self, scale_outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        """Aggregate scale outputs using learned weights."""
        weights = F.softmax(self.scale_weights, dim=0)

        weighted_sum = torch.zeros_like(scale_outputs[0])
        for i, output in enumerate(scale_outputs):
            weighted_sum += weights[i] * output

        return weighted_sum
