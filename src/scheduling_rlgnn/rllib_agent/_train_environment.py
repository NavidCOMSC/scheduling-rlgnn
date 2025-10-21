import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from scheduling_rlgnn.rllib_agent._nodeisolationenv import NodeIsolationEnv
from scheduling_rlgnn.rllib_agent._ppojobshoprlmodule import PPOJobShopRLModule


def test_environment():
    """Test the NodeIsolationEnv to verify it works correctly."""
    print("Testing NodeIsolationEnv...")
    env = NodeIsolationEnv(num_nodes=10, p=0.3)

    obs, info = env.reset(seed=42)
    print(f"Initial observation keys: {obs.keys()}")
    print(f"Node features shape: {obs['x'].shape}")
    print(f"Edge index shape: {obs['edge_index'].shape}")
    print(f"Action mask shape: {obs['action_mask'].shape}")
    print(f"Action space: {env.action_space}")

    # Test a few random steps
    for i in range(3):
        valid_actions = np.where(obs["action_mask"] == 1)[0]
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {i+1}: action={action}, reward={reward}, terminated={terminated}, truncated={truncated}"
        )

        if terminated or truncated:
            break

    env.close()
    print("NodeIsolationEnv test completed successfully.")
