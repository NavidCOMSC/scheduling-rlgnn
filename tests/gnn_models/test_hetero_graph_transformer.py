import pytest
import torch
from torch_geometric.data import HeteroData

from scheduling_rlgnn.gnn_models import HeteroGraphTransformer


@pytest.fixture
def hetero_data():
    """Create a minimal dummy graph transformer for testing"""
    data = HeteroData()

    # Node features with different dimensions
    data["operation"].x = torch.randn(3, 64)  # 3 operations
    data["machine"].x = torch.randn(2, 32)  # 2 machines
    data["job"].x = torch.randn(1, 16)  # 1 job

    # Edge indices
    data["operation", "precedence", "operation"].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    data["operation", "assigned_to", "machine"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 1]], dtype=torch.long
    )
    data["machine", "can_process", "operation"].edge_index = torch.tensor(
        [[0, 1, 1], [0, 1, 2]], dtype=torch.long
    )
    data["job", "contains", "operation"].edge_index = torch.tensor(
        [[0, 0, 0], [0, 1, 2]], dtype=torch.long
    )
    data["operation", "belongs_to", "job"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 0, 0]], dtype=torch.long
    )

    # Edge features (only for precedence edges)
    data["operation", "precedence", "operation"].edge_attr = torch.randn(2, 32)

    # Batch indices
    data["operation"].batch = torch.tensor([0, 0, 0])
    data["machine"].batch = torch.tensor([0, 0])
    data["job"].batch = torch.tensor([0])

    return data


def test_forward_pass(hetero_data):
    """Test forward pass returns correct output shapes."""
    metadata = hetero_data.metadata()
    node_dims = {"operation": 64, "machine": 32, "job": 16}
    edge_dims = {("operation", "precedence", "operation"): 32}

    model = HeteroGraphTransformer(
        metadata=metadata,
        node_dims=node_dims,
        edge_dims=edge_dims,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_nodes=100,
        aggr="sum",
    )

    # Extract data from HeteroData
    x_dict = {
        node_type: hetero_data[node_type].x
        for node_type in hetero_data.node_types
    }
    edge_index_dict = hetero_data.edge_index_dict

    # Project edge attributes to hidden dimension and
    # create edge attributes for ALL edge types
    edge_attr_dict = {}
    for edge_type in hetero_data.edge_types:
        if hasattr(hetero_data[edge_type], "edge_attr"):
            # Use existing edge attributes and project them
            edge_attr_dict[edge_type] = hetero_data[edge_type].edge_attr
        else:
            # Add dummy edge attributes for edge types without them
            num_edges = hetero_data[edge_type].edge_index.shape[1]
            edge_attr_dict[edge_type] = torch.ones(
                num_edges, edge_dims.get(edge_type, 32)  # Default dimension
            )

    batch_dict = {
        node_type: hetero_data[node_type].batch
        for node_type in hetero_data.node_types
    }

    # Forward pass
    output_dict = model(x_dict, edge_index_dict, edge_attr_dict, batch_dict)

    # Check output shapes for each node type
    assert output_dict["operation"].shape == (3, 128)
    assert output_dict["machine"].shape == (2, 128)
    assert output_dict["job"].shape == (1, 128)


def test_attention_weights(hetero_data):
    """Test attention weight extraction."""
    metadata = hetero_data.metadata()
    node_dims = {"operation": 64, "machine": 32, "job": 16}
    edge_dims = {("operation", "precedence", "operation"): 32}

    model = HeteroGraphTransformer(
        metadata=metadata,
        node_dims=node_dims,
        edge_dims=edge_dims,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_nodes=100,
    )
    # Extract data from HeteroData
    x_dict = {
        node_type: hetero_data[node_type].x
        for node_type in hetero_data.node_types
    }
    edge_index_dict = hetero_data.edge_index_dict

    # Get attention weights with edge attributes
    edge_attr_dict = {}
    for edge_type in hetero_data.edge_types:
        if hasattr(hetero_data[edge_type], "edge_attr"):
            edge_attr_dict[edge_type] = hetero_data[edge_type].edge_attr
        else:
            num_edges = hetero_data[edge_type].edge_index.shape[1]
            edge_attr_dict[edge_type] = torch.ones(
                num_edges, edge_dims.get(edge_type, 32)  # Default dimension
            )

    # Get attention weights
    attention_weights = model.get_attention_weights(
        x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict, layer_idx=-1
    )

    # Check if attention weights are returned for precedence edges
    precedence_key = ("operation", "precedence", "operation")
    assert precedence_key in attention_weights
    assert attention_weights[precedence_key] is not None

    # Adjust this assertion based on the actual return format
    weights = attention_weights[precedence_key]
    assert weights.numel() > 0  # Just check that we got some weights back
