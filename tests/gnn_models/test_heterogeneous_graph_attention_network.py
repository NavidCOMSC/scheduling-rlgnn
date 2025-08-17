import pytest
import torch
from torch_geometric.data import HeteroData

from src.scheduling_rlgnn.gnn_models._heterogeneous_graph_attention_network import (
    HeterogeneousGraphAttentionNetwork,
)


@pytest.fixture
def hetero_data():
    """Create a minimal heterogeneous graph for testing."""
    data = HeteroData()

    # Node features
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

    # Edge features
    data["operation", "precedence", "operation"].edge_attr = torch.randn(2, 32)

    # Batch indices (optional)
    data["operation"].batch = torch.tensor([0, 0, 0])
    data["machine"].batch = torch.tensor([0, 0])
    data["job"].batch = torch.tensor([0])

    return data


def test_forward_pass(hetero_data):
    """Test forward pass returns correct output shapes."""
    model = HeterogeneousGraphAttentionNetwork(
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        use_global_pool=True,
        pool_node_types=["operation"],
    )

    # Extract data from HeteroData
    x_dict = {
        node_type: hetero_data[node_type].x
        for node_type in hetero_data.node_types
    }
    edge_index_dict = hetero_data.edge_index_dict
    edge_attr_dict = hetero_data.edge_attr_dict
    batch_dict = {
        node_type: hetero_data[node_type].batch
        for node_type in hetero_data.node_types
    }

    # Forward pass
    output = model(x_dict, edge_index_dict, edge_attr_dict, batch_dict)

    # Check pooled output shape
    assert output.shape == (1, 128)  # Batch size 1, hidden_dim 128
