import pandas as pd
import numpy as np

def add_time_features(df):
    """
    
    Add time-based features to the dataframe.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with 'pickup_datetime' column

    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional time features

    """

    # Extract date components
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek # 0=Monday, 6=Sunday
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year

    # Derived features
    df['is_weekend'] = df['day_of_week'] >= 5
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19])
    df['time_of_day'] = pd.cut(
        df['hour'],
        bins = [0, 6, 12, 18, 24],
        labels = ['night', 'morning', 'afternoon', 'evening'],
        right=False  #right endpoint of each interval is excluded from that bin
    )
    return df

def add_pricing_features(df):

    """
    Add features related to pricing and economics

    parameters:
    -----------
    df : pandas.DataFrame
        DataFramewith fare and trip information

    Returns:
    --------
    pandas.DataFrame
        FataFrame with additional pricing features
    """

    # Price per mile
    df['price_per_mile'] = df['fare_amount'] / df['trip_distance'].replace(0,np.nan)
    df['price_per_minute'] = df['fare_amount'] / df['trip_duration'].replace(0,np.nan)

    # Fill or drop NaN values
    df = df.dropna(subset=['price_per_mile', 'price_per_minute'])

    # Remove extreme outliersin derived metrics
    df = df[(df['price_per_mile'] < 50) & (df['price_per_mile']>1)]
    df = df[(df['price_per_minute'] < 10) & (df['price_per_minute']>0.1)]

    return df