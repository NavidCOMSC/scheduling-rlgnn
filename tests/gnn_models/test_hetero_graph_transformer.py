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
