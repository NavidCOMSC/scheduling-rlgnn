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


def train_ppo_node_isolation():
    """Train PPO on the NodeIsolationEnv using PPOJobShopRLModule."""

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register the custom environment
    def env_creator(config):
        return NodeIsolationEnv(
            num_nodes=config.get("num_nodes", 15), p=config.get("p", 0.25)
        )

    tune.register_env("NodeIsolationEnv", env_creator)

    # Create PPO configuration
    config = PPOConfig()
    config.environment(
        "NodeIsolationEnv-v0",
        env_config={"num_nodes": 15, "p": 0.25},
    )
    config.framework("torch")
    config.rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=PPOJobShopRLModule,
            model_config_dict={
                "fcnet_hiddens": [128, 128, 64],
                "fcnet_activation": "relu",
            },
        )
    )
    config.training(
        train_batch_size=2000,
        minibatch_size=128,
        num_sgd_iter=10,
        lr=3e-4,
        gamma=0.99,
    )
    config.rollouts(
        num_rollout_workers=2,
        num_envs_per_worker=1,
    )
    config.resources(
        num_gpus=0,
    )
    config.debugging(
        log_level="INFO",
    )

    # Create the PPO algorithm
    print("Building PPO algorithm...")
    ppo_algorithm = config.build()

    # training loop
    print("\nStarting training...")
    num_iterations = 20

    for i in range(num_iterations):
        result = ppo_algorithm.train()

        print(f"\n{'='*60}")
        print(f"Iteration {i+1}/{num_iterations}")
        print(f"{'='*60}")
        print(
            f"Episode reward mean: {result['env_runners']['episode_return_mean']:.2f}"
        )
        print(
            f"Episode reward min: {result['env_runners']['episode_return_min']:.2f}"
        )
        print(
            f"Episode reward max: {result['env_runners']['episode_return_max']:.2f}"
        )
        print(
            f"Episode length mean: {result['env_runners']['episode_len_mean']:.2f}"
        )
        print(f"Training iteration time: {result['time_total_s']:.2f}s")

        # Optionally save checkpoint
        if (i + 1) % 10 == 0:
            checkpoint_dir = ppo_algorithm.save()
            print(f"Checkpoint saved at: {checkpoint_dir}")

    print("\n" + "=" * 60)
    print("Training completed.")
    print("=" * 60)

    # cleanup
    ppo_algorithm.stop()
    ray.shutdown()


def evaluate_trained_agent(checkpoint_path=None):
    """
    Evaluate a trained agent (optional - if you have a checkpoint).

    Args:
        checkpoint_path: Path to the saved checkpoint
    """

    if checkpoint_path is None:
        print("No checkpoint path provided. Skipping evaluation.")
        return

    ray.init(ignore_reinit_error=True)

    # Register environment
    tune.register_env(
        "NodeIsolation-v0",
        lambda config: NodeIsolationEnv(num_nodes=15, p=0.25),
    )

    # Load the trained algorithm
    config = PPOConfig().environment(
        "NodeIsolation-v0",
    )
    ppo_algorithm = config.build()
    ppo_algorithm.restore(checkpoint_path)

    # Run evaluation episodes
    env = NodeIsolationEnv(num_nodes=15, p=0.25)
    num_eval_episodes = 10

    for episode in range(num_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        while not done:
            # Get action from policy
            action = ppo_algorithm.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        print(
            f"Episode {episode + 1}: Total Reward: {episode_reward:.2f}, Steps: {steps}"
        )

    env.close()
    ppo_algorithm.stop()
    ray.shutdown()
