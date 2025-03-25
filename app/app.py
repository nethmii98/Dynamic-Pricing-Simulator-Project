#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dashboard application for the ride-sharing dynamic pricing simulator.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import project modules
from src.models.environment import RideSharingPricingEnv
from src.models.evaluation import analyze_ab_test_results
from src.visualization.dashboard import create_dashboard

def main():
    """Main function to run the dashboard application."""
    # Load AB test results
    ab_results_path = '../output/ab_test/ab_test_raw_results.csv'
    if not os.path.exists(ab_results_path):
        print(f"Error: AB test results not found at {ab_results_path}")
        print("Please run notebook 05_strategy_evaluation.ipynb first to generate results.")
        return
    
    ab_results = pd.read_csv(ab_results_path)
    analysis_results = analyze_ab_test_results(ab_results)
    
    # Load elasticity data
    elasticity_path = '../data/processed/elasticity_estimates.csv'
    if not os.path.exists(elasticity_path):
        print(f"Error: Elasticity data not found at {elasticity_path}")
        print("Please run notebook 03_elasticity_modeling.ipynb first to generate elasticity data.")
        return
    
    elasticity_df = pd.read_csv(elasticity_path)
    
    # Convert to dictionary format
    elasticity_data = {}
    for _, row in elasticity_df.iterrows():
        elasticity_data[row['segment']] = {
            'elasticity': row['elasticity'],
            'r2': row['r2'],
            'p_value': row['p_value'],
            'std_error': row['std_error']
        }
    
    # Create dashboard
    app = create_dashboard(None, analysis_results, elasticity_data)
    
    # Run the app
    print("Starting dashboard server...")
    app.run_server(debug=True, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    main()