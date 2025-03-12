import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_demand_patterns(df):
    """
    Analyze temporal demand patterns from taxi data

    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe with time features

    Returns:
    --------
    dict
        Dictionary containing various demand pattern analyses
    """
    # Hourly demand by day of week
    hourly_demand = df.groupby(['day_of_week', 'hour']).agg({
        'fare_amount': 'mean',
        'trip_distance': 'mean',
        'passenger_count':'mean',
        'pickup_datetime': 'count'
    }).rename(columns={'pickup_datetime':'trip_count'}).reset_index()

    # Daily Demand
    daily_demand = df.groupby('day_of_week')['pickup_datetime'].count().reset_index(
        name='trip_count'
    )

    # Time of day demand
    time_of_day_demand = df.groupby('time_of_day')['pickup_datetime'].count().reset_index(
        name='trip_count'
    )

    # Demand by location
    location_demand = df.groupby('PULocationID')['pickup_datetime'].count().reset_index(
        name = 'trip_count'
    ).sort_values('trip_count', ascending=False)

    # Identify peak demand periods 
    peak_demand = hourly_demand.sort_values('trip_count', ascending=False).head(10)

    # Calculate demand variability
    hourly_stats = df.groupby(['day_of_week','hour'])['pickup_datetime'].count()
    hourly_variability = hourly_stats.groupby('hour').std().reset_index(name='demand_std')
    hourly_variability['demand_cv'] = hourly_variability['demand_std'] / hourly_stats.groupby('hour').mean().values

    return {
        'hourly_demand': hourly_demand,
        'daily_demand': daily_demand,
        'time_of_day_demand': time_of_day_demand,
        'location_demand': location_demand,
        'peak_demand': peak_demand,
        'hourly_variability': hourly_variability
    }

def visualize_demand_patterns(demand_data, save_dir=None):
    """
    Create visualizations of demand patterns.

    Parameters:
    -----------
    demand_data : dict
        Output from analyze_demand_patterns function
    save_dir : str, optional
        Directory to save visualizations to demand analysis folder
    """
    # Heatmap of hourly demand by day of week
    plt.figure(figsize=(12,6))
    pivot_data = demand_data['hourly_demand'].pivot(
        index='day_of_week',
        columns='hour',
        values='trip_count'
    )
    sns.heatmap(pivot_data, cmap='YlGnBu')
    plt.title('Taxi Demand by Hour and Day of Week')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week (0=Monday, 6=Sunday)')
    if save_dir:
        plt.savefig(f"{save_dir}/hourly_demand_heatmap.png", dpi=300, bbox_inches = 'tight')
    plt.show()

    # Bar chart of daily demand
    plt.figure(figsize=(10,5))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.barplot(x = 'day_of_week', y='trip_count', data=demand_data['daily_demand'])
    plt.title('Taxi Demand by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Trips')
    plt.xticks(range(7), days)
    if save_dir:
        plt.savefig(f"{save_dir}/daily_demand.png", dpi=300, bbox_inches= 'tight')
    plt.show()