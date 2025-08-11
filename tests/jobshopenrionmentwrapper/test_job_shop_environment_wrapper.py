import pytest
import numpy as np
from job_shop_lib import JobShopInstance, Operation
from scheduling_rlgnn.envwrapper import JobShopEnvironmentWrapper


# Fixture for a minimal JobShopInstance
@pytest.fixture
def minimal_job_shop_instance():
    """Creates a minimal JobShopInstance with 2 jobs and 2 machines."""
    jobs = [
        [Operation(0, duration=3), Operation(1, duration=2)],
        [Operation(1, duration=2), Operation(0, duration=4)],
    ]
    return JobShopInstance(jobs=jobs, num_machines=2)


# Test for _create_graph_observation method
def test_create_graph_observation(minimal_job_shop_instance):
    """Tests that _create_graph_observation returns expected structure."""
    env = JobShopEnvironmentWrapper(minimal_job_shop_instance)
    obs_dict = env._create_graph_observation()

    # Check keys in the returned dictionary
    assert "obs" in obs_dict
    assert "action_mask" in obs_dict

    # Validate observation array properties
    obs_array = obs_dict["obs"]
    assert isinstance(obs_array, np.ndarray)
    assert obs_array.dtype == np.float32
    assert obs_array.shape == (64 * 4,)  # 64 nodes * 4 features

    # Validate action mask properties
    action_mask = obs_dict["action_mask"]
    assert isinstance(action_mask, np.ndarray)
    assert action_mask.dtype == bool
    assert action_mask.shape == (64,)

    # Initial state should have 2 available operations (first ops of each job)
    assert action_mask.sum() == 2
    assert action_mask[0] and action_mask[1]  # First two operations available
    assert not action_mask[2:].any()  # Rest should be False
