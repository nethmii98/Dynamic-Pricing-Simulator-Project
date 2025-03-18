import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def estimate_price_elasticity(df):
    """
    Estimate price elasticity of demand from taxi data.

    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe with pricing features
    segment_columns : list, optional
        List of columns to segment the data by for elasticity calculation

    Returns:
    --------
    dict
        Dictionary containing elasticity estimates for different segments
    """
    # Define segments for elasticity calculation
    segments = {
        'overall': df,
        'peak' : df[df['is_rush_hour']],
        'offpeak' : df[~df['is_rush_hour'] & ~df['is_weekend']], # Since different travel behavior on weekend
        'weekend' : df[df['is_weekend']]
    }

    # Calculate elasticity for each segment
    elasticity_results = {}

    for segment_name,segment_data in segments.items():

        # Group by hour to gt average prices and demand
        hourly_data = segment_data.groupby('hour').agg({
            'fare_amount':'mean',
            'pickup_datetime':'count'
        }).rename(columns = {'pickup_datetime':'demand'}).reset_index()

        # Log transform for elasticity calculation
        hourly_data['log_price'] = np.log(hourly_data['fare_amount'])
        hourly_data['log_demand'] = np.log(hourly_data['demand'])

        # Simple OLS(Ordinary Least Squares) Regeression
        model = LinearRegression()
        X = hourly_data[['log_price']]
        y = hourly_data['log_demand']
        model.fit(X,y)

        # For statistical significance
        X_sm = sm.add_constant(X)
        sm_model = sm.OLS(y, X_sm).fit()

        elasticity_results[segment_name] = {
            'elasticity': model.coef_[0],
            'r2': model.score(X,y),
            'p_value': sm_model.pvalues[1],
            'std_error': sm_model.bse[1],
            'sample_size': len(hourly_data)

        }

    return elasticity_results

def calculate_optimal_prices(elasticity_results):
    """
    Calculate optimal prices based on edtimated elasticities.

    Parameters:
    -----------
    elasticity_results : dict
        Output from estimate_price_elasticity function

    Returns:
    --------
    dict
        Dictionary containing optimal price multipliers for each segment
    """
    optimal_prices = {}

    for segment, results in elasticity_results.items():
        elasticity = results['elasticity'] 

        # For elastic demand,optimal markup is |e|/(|e|-1) (goal is to profit maximization)
        if abs(elasticity) >1:
            markup = abs(elasticity)/(abs(elasticity)-1)
        else:
            # For inelastic demand, theory suggests high prices,
            # but practical considerations apply
            markup = 2.0 # Cap at 2x

        optimal_prices[segment] = {
            'elasticity': elasticity,
            'optimal_multiplier': markup,
            'is_elastic': abs(elasticity)>1
        }

    return optimal_prices

def visualize_elasticity(elasticity_results, save_dir = None):
    """
    Visulaize elasticity results.

    Parameters:
    -----------
    elasticity_results : dict
        Output from estimate_price_elasticity function

    save_dir : str, optional
        Directory to save visualizations to elasticity analysis folder 
    """

    # Bar chart of elasticity estimates
    plt.figure(figsize=(10,6))
    segments = list(elasticity_results.keys())
    elasticities = [elasticity_results[s]['elasticity'] for s in segments]

    bars = plt.bar(segments, elasticities)
    plt.axhline(y=-1, color='r', linestyle='--', alpha=0.7)
    plt.text(0, -1.05, 'Unit Elastic', color='r', ha='center')

    # Color bars based on elasticity
    for i, e in enumerate(elasticities):
        if e > -1: #Inelastic
            bars[i].set_color('green')
        else:
            bars[i].set_color('blue')

    plt.title('Price Elasticity of Demand by Segment')
    plt.xlabel('Segment')
    plt.ylabel('Elasticity')
    plt.grid(axis='y', alpha=0.3)

    if save_dir:
        plt.savefig(f"{save_dir}/elasticity_estimates.png", dpi=300,bbox_inches='tight')
    plt.show()