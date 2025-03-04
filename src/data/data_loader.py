import pandas as pd
import numpy as np
from pathlib import Path

def load_taxi_data(filepath, sample_size=None):
   
    """
    Load and perform initial processing of NYC taxi trip data.

    parameters:
    
    -----------
    
    filepath : str
        Path to the parquet file containing taxi trip data
    sample_size : int, optional
        Number of rows to sample (for faster development)

    Returns :

    ---------

    pandas.DataFrame
        Processed dataframe with basic cleaning applied        

    """
    print(f"Loading data from {filepath}...")

    # Load Data
    df = pd.read_parquet(filepath)

    if sample_size:
        df = df.sample(sample_size, random_state=42)

    print(f"Loaded {len(df)} rows")

    # Basic Cleaning
    print("Performing basic cleaning...")

    # Convert timestamps
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # Remove outliers
    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]
    df = df[(df['fare_amount'] > 2.5) & (df['fare_amount'] < 200)]
    df = df[df['passenger_count'] > 0]

    # Calculate trip duration
    df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() /60

    # Remove unreasonable durations
    df = df[(df['trip_duration'] > 1) & (df['trip_duration'] < 180)]

    print(f"{len(df)} rows after cleaning")

    return df

def save_processed_data(df, filename, output_dir):
    """
    Save processed dataframe to the specified output directory.

    Parameter:
    ----------
    df : pandas.DataFrame
        DataFrame to save 
    filename :str
        Name for the output file
    output_dir : str
        Directory to save the file to
    """

    # Ensure directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine file format and save
    if filename.endswith('.csv'):
        df.to_csv(f"{output_dir}/{filename}",index = False)
    elif filename.endswith('.parquet'):
        df.to_parquet(f"{output_dir}/{filename}", index = False)
    else:
        df.to_csv(f"{output_dir}/{filename}.csv", index=False) 

    print(f"Saved processed data to {output_dir}/{filename}")                              
    return None