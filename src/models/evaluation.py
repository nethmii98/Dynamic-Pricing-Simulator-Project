import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def define_pricing_strategies():
    """
    Define different pricing strategies for comparison.

    Returns:
    --------
    dict
        Dictionary mapping stratagy names to pricing functions
    """
    strategies = {
        "Constant": lambda h, d, dr, cr, wt: 1.0, # Fixed pricing (h=hour of day, d=Day of the week, dr=demand ratio, cr=competitor pricing ratio, wt=wait time for a ride)
        "Time-based": lambda h, d, dr, cr, wt: ( # Simple time of day based
            1.5 if h in [7,8,9,17,18,19] else # Rush hours
            1.3 if d >= 5 else # Weekends
            0.9 # Off peak
        ),

        "Surge": lambda h, d, dr, cr, wt: min(2.5, max(0.7, dr)), # Pure demand based
        "Competitive": lambda h, d, dr, cr, wt: min(2.0, max(0.8, cr)), # Match competitors
        "Wait-Based": lambda h, d, dr, cr, wt: min(2.0, max(0.8, wt/10)), # Based on wait times

    }

    return strategies

def add_ml_strategy(strategies, model):
    """
    Add a trained ML model to the strategies dictionary.

    Parameters:
    -----------
    strategies : dict
        Dictionary of pricing strategies
    model : stable_baseline3.PPO
        Trained RL agent

    Returns:
    --------
    dict
        Updated strategies dictionary

    """

    # Create wrapper function that calls the model
    def ml_strategy(h, d, dr, cr, wt):
        # Construct observation vector
        obs = np.array([h, d, dr, cr, wt, 0, 0]) # Last two values are placeholders
        action,_ =model.predict(obs)
        return action[0]
    
    # Add to strategies
    strategies["RL-Agent"] = ml_strategy

    return strategies

def run_ab_test(env, strategies, n_episodes=10):
    """
    Run A/B tests comparing different pricing strategies.

    Parameters:
    -----------
    env : gym.Env
        Simulation environment
    strategies : dict
        Dictionary mapping strategy names to pricing functions
    n_episodes : int
        Number of simulation episodes for each strategy

    Returns:
    --------
    pandas.DataFrame
        Results of the A/B test
    """
    results = []

    for strategy_name, strategy_fn in strategies.items():
        print(f"Testing strategy: {strategy_name}")

        for episode in range(n_episodes):
            # Reset environment
            obs = env.reset()
            done = False

            hourly_data = []

            # Run one episode
            while not done:
                # Extract state variables
                hour =obs[0]
                day = obs[1]
                demand_ratio = obs[2]
                competitor_ratio = obs[3]
                wait_time = obs[4]

                # Apply strategy to determine price
                if callable(strategy_fn):
                    action = [strategy_fn(hour, day, demand_ratio, competitor_ratio, wait_time)]
                else:
                    # For RL agent
                    action,_ = strategy_fn.predict(obs)

                # Take step in environment
                new_obs, reward, done, info = env.step(action) 

                # Record hourly data
                hourly_data.append({
                    'strategy': strategy_name,
                    'episode': episode,
                    'hour': int(hour),
                    'day': int(day),
                    'price_multiplier': float(info['price_multiplier']),
                    'demand': float(info['actual_demand']),
                    'revenue': float(reward),
                    'wait_time': float(wait_time)
                })

                # Update for next step
                obs = new_obs

            # Add episode summary
            episode_df = pd.DataFrame(hourly_data)
            episode_summary = {
                'strategy': strategy_name,
                'episode': episode,
                'total_revenue': episode_df['revenue'].sum(),
                'total_rides': episode_df['demand'].sum(),
                'avg_price_mult': episode_df['price_multiplier'].mean(),
                'max_price_mult': episode_df['price_multiplier'].max(),
                'min_price_mult': episode_df['price_multiplier'].min(),
                'price_variance': episode_df['price_multiplier'].var()
            }

            results.append(episode_summary)

            # Also keep the hourly data
            results.extend(hourly_data)
    
    
    return pd.DataFrame(results)

def analyze_ab_test_results(results_df):
    """
    Analyze the results of the A/B test.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results from run_ab_test function

    Returns:
    -------
    dict
        Dictionary with analysis results
    """

    # Filter out hourly data rows
    summary_df = results_df[results_df['hour'].isnull()]
    hourly_df = results_df[results_df['hour'].notnull()]

    # Overall strategy performance
    strategy_summary = summary_df.groupby('strategy').agg({
        'total_revenue': ['mean', 'std', 'min', 'max'],
        'total_rides': ['mean', 'std'],
        'avg_price_mult': 'mean',
        'price_variance': 'mean'
    })

    # Statistical significance testing
    # For each pair of strategies, run t-test on revenue
    strategies = summary_df['strategy'].unique()
    significance_tests = []

    for i, strat1 in enumerate(strategies):
        for strat2 in strategies[i+1:]:
            revenue1 = summary_df[summary_df['strategy'] == strat1]['total_revenue']
            revenue2 = summary_df[summary_df['strategy'] == strat2]['total_revenue']

            t_stat, p_value = stats.ttest_ind(revenue1, revenue2)

            significance_tests.append({
                'strategy1':strat1,
                'strategy2': strat2,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'better_strategy': strat1 if t_stat > 0 else strat2
            })

    significance_df = pd.DataFrame(significance_tests)

    # Time based analysis
    time_performance = hourly_df.groupby(['strategy','hour']).agg({
        'price_multiplier': 'mean',
        'demand': 'mean',
        'revenue': 'mean'
    }).reset_index()

    day_performance = hourly_df.groupby(['strategy', 'day']).agg({
        'price_multiplier': 'mean',
        'demand': 'mean',
        'revenue': 'mean'
    }).reset_index()

    return{
        'summary': strategy_summary,
        'significance': significance_df,
        'time_performance': time_performance,
        'day_performance': day_performance,
        'hourly_data': hourly_df
    }

def visualize_ab_test_results(analysis_results, save_dir=None):
    """
    Create visualizations of A/B test results.

    Parameters:
    -----------
    analysis_results : dict
        Output from analyze_ab_test_results function
    save_dir : str, optional
        Directory to save visualizations to ab_test
    """

    # revenue comparison bar chart
    plt.figure(figsize=(10,6))
    summary = analysis_results['summary']['total_revenue']['mean'].reset_index()
    summary.columns = ['strategy', 'mean_revenue']

    # Sort by revenue
    summary = summary.sort_values('mean_revenue', ascending=False)

    sns.barplot(x='strategy', y='mean_revenue', data=summary)
    plt.title('Average Revenue by Pricing Strategy')
    plt.xlabel('Strategy')
    plt.ylabel('Average Revenue')
    plt.xticks(rotation = 45)
    plt.grid(axis='y', alpha=0.3)

    if save_dir:
        plt.savefig(f"{save_dir}/revenue_comparison.png", dpi=300, bbox_inches='tight')

    plt.show()

    # Hourly price comparison
    plt.figure(figsize=(12,6))

    # Pivot data for easier plotting
    hourly_prices = analysis_results['time_performance'].pivot(
        index='hour',
        columns='strategy',
        values='price_multiplier'
    )

    # Plot each strategy
    for strategy in hourly_prices.columns:
        plt.plot(hourly_prices.index, hourly_prices[strategy], label=strategy)

    plt.title('Average Price Multiplier by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Price Multiplier')
    plt.legend()
    plt.grid(alpha=0.3)

    if save_dir:
        plt.savefig(f"{save_dir}/hourly_price_comparison.png", dpi=300, bbox_inches='tight')

    plt.show()
