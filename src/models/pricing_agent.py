from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_pricing_agent(env, total_steps=500000, save_path=None):
    """
    Train a reinforcement learning agent for dynamic pricing.

    Parameters:
    -----------
    env : gym.Env
        The environment to train in
    total_steps : int
        Number of timesteps to train for 
    save_path : str, optional
        Path to save the trained model to models/ride_sharing_pricing_model
    
    Returns:
    --------
    stable_baselines3.PPO
        Trained RL agent
    """
    # Create a custom neural network policy
    policy_kwargs = dict(
        net_arch = [dict(pi=[128, 128], vf=[128,128])]
    )

    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs/"
    )

    # Train the agent
    model.learn(total_timesteps=total_steps)

    # Save the trained model
    if save_path:
        model.save(save_path)
        print(f"Model saved to {save_path}")
    return model

def evaluate_agent(model, env, n_eval_episodes=10):
    """
    Evaluate a trained agent's performance.

    Parameters:
    -----------
    model : stable_baselines3.PPO
        Trained RL agent
    env : gym.Env
        Environment to evaluate in
    n_eval_episodes : int
        Number of episodes to evaluate

    Returns:
    --------
    dict
        Evaluation metrics
    """
    # Run evaluation episodes
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)

    # Collect detailed metrics
    episode_rewards = []
    episode_lengths = []
    price_histories = []
    demand_histories = []

    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0

        # Store price and demand for this episode
        price_history = []
        demand_history = []

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

            episode_reward += reward
            step += 1

            price_history.append(info['price_multiplier'])
            demand_history.append(info['actual_demand'])

        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        price_histories.append(price_history)
        demand_histories.append(demand_history)

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'price_histories': price_histories,
        'demand_histories': demand_histories
    }

def visualize_agent_behavior(evaluation_results, save_path=None):
    """
    Visualize the behavior of a trained agent.

    Parameters:
    -----------
    evaluation_results : dict
        Output from evaluate_agent function
    save_path : str, optional
        Path to save visualizations to /output/rl_evaluation
    """
    # Plot price strategies over time (first episode)
    plt.figure(figsize=(15, 5))
    
    hours = range(len(evaluation_results['price_histories'][0]))
    plt.plot(hours, evaluation_results['price_histories'][0], label='Price Multiplier')
    
    plt.title('RL Agent Pricing Strategy Over Time')
    plt.xlabel('Hour')
    plt.ylabel('Price Multiplier')
    plt.grid(alpha=0.3)

    plt.legend(loc='upper left')

    # Add demand on secondary axis
    ax2 = plt.gca().twinx()
    ax2.plot(hours, evaluation_results['demand_histories'][0], 'r--', label='Demand')
    ax2.set_ylabel('Demand')
    ax2.legend(loc = 'upper right')
    
    
    if save_path:
        plt.savefig(f"{save_path}/agent_behavior.png", dpi=300, bbox_inches='tight')
    
    plt.show()