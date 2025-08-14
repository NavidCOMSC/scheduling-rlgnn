"""GNN model architectures."""

"""
This package provides various Graph Neural Network (GNN) model 
architectures for use in graph-based learning tasks.

Available architectures:
- GraphAttentionNetwork: Implements a GNN using attention 
mechanisms to weigh node neighbors.
- GraphTransformer: A transformer-based model adapted for 
graph-structured data.
- HierarchicalGNN: A hierarchical GNN that captures 
multi-level graph representations.
- MultiScaleGNN: A GNN architecture designed to process 
information at multiple graph scales.

These models can be imported directly from this package for 
use in graph learning pipelines.
"""

from ._graph_attention_network import GraphAttentionNetwork
from ._graph_transformer import GraphTransformer
from ._hierarchical_gnn import HierarchicalGNN
from ._multi_scale_gnn import MultiScaleGNN

__all__ = [
    "GraphAttentionNetwork",
    "GraphTransformer",
    "HierarchicalGNN",
    "MultiScaleGNN",
]
