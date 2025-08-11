# import pytest
# import numpy as np
# from job_shop_lib import JobShopInstance, Operation
# from scheduling_rlgnn.envwrapper import JobShopEnvironmentWrapper


# # Fixture for a minimal JobShopInstance
# @pytest.fixture
# def minimal_job_shop_instance():
#     """Creates a minimal JobShopInstance with 2 jobs and
#     2 machines."""
#     jobs = [
#         [Operation(0, duration=3), Operation(1, duration=2)],
#         [Operation(1, duration=2), Operation(0, duration=4)],
#     ]
#     return JobShopInstance(jobs=jobs, num_machines=2)


# # Test for _create_graph_observation method
# def test_create_graph_observation(minimal_job_shop_instance):
#     """Tests that _create_graph_observation returns expected structure."""
#     env = JobShopEnvironmentWrapper(minimal_job_shop_instance)
#     obs_dict = env._create_graph_observation()

#     # Check keys in the returned dictionary
#     assert "obs" in obs_dict
#     assert "action_mask" in obs_dict

#     # Validate observation array properties
#     obs_array = obs_dict["obs"]
#     assert isinstance(obs_array, np.ndarray)
#     assert obs_array.dtype == np.float64
#     assert obs_array.shape == (64 * 3,)  # 64 nodes * 3 features

#     # Validate action mask properties
#     action_mask = obs_dict["action_mask"]
#     assert isinstance(action_mask, np.ndarray)
#     assert action_mask.dtype == bool
#     assert action_mask.shape == (64,)

#     # Initial state should have 2 available operations (first ops of each job)
#     assert action_mask.sum() == 2
#     assert action_mask[0] and action_mask[1]  # First two operations available
#     assert not action_mask[2:].any()  # Rest should be False


# # Test for step method
# def test_step_feasible_parameters(minimal_job_shop_instance):
#     """Tests that step returns feasible parameters with valid actions."""
#     env = JobShopEnvironmentWrapper(minimal_job_shop_instance)
#     env.reset()

#     # Take a step with a valid action (index 0)
#     obs, reward, terminated, truncated, info = env.step(0)

#     # Check parameter types and boundaries
#     assert isinstance(obs, dict)
#     assert "obs" in obs and "action_mask" in obs
#     assert isinstance(reward, float)
#     assert isinstance(terminated, bool)
#     assert isinstance(truncated, bool)
#     assert isinstance(info, dict)

#     # Check values are within expected ranges
#     assert not terminated  # Shouldn't be terminated after first step
#     assert not truncated  # Shouldn't be truncated at step 1
#     assert info["completed_operations"] == 1
#     assert info["completion_rate"] == 0.25  # 1/4 operations completed


# # Test for _execute_action method
# def test_execute_action_output(minimal_job_shop_instance):
#     """Tests _execute_action returns viable rewards and updates state."""
#     env = JobShopEnvironmentWrapper(minimal_job_shop_instance)
#     env.reset()

#     # Test valid action
#     reward = env._execute_action(0)
#     assert isinstance(reward, float)

#     # Check state was updated
#     assert len(env.completed_operations) == 1
#     assert env.job_available_time[0] > 0  # Job 0 availability time updated

#     # Test invalid action (out of bounds)
#     invalid_action_reward = env._execute_action(100)
#     assert invalid_action_reward == -10.0  # Expected penalty
