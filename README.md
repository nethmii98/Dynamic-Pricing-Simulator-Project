# Dynamic Pricing Simulator for Ride-Sharing Services

This project implements a dynamic pricing simulator for ride-sharing services using reinforcement learning. The system analyzes real NYC taxi data to extract demand patterns and price elasticity, then trains a reinforcement learning agent to optimize pricing strategies.

## Project Overview 

The ride-sharing pricing simulator allows for:

1. **Data-driven analysis** of temporal demand patterns and price elasticity
2. **Reinforcement learning** to optimize dynamic pricing strategies
3. **A/B testing** to compare different pricing approaches
4. **Interactive visualization** of pricing strategies and their impacts

## Data Source

This project uses the [New York City Taxi and Limousine Commission (TLC) Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), which is freely available and contains detailed trip records from yellow and green taxis in NYC.

## Project Structure

Dynamic-Pricing-Simulator-Project/
│
├── data/                          # Data storage directory
│   ├── raw/                       # Original, immutable data
│   ├── processed/                 # Cleaned, transformed data
│   └── external/                  # External data sources
│
├── src/                           # Source code
│   ├── data/                      # Data processing scripts
│   ├── features/                  # Feature engineering code
│   ├── models/                    # Model training and evaluation
│   └── visualization/             # Visualization code
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_demand_analysis.ipynb
│   ├── 03_elasticity_modeling.ipynb
│   ├── 04_rl_training.ipynb
│   └── 05_strategy_evaluation.ipynb
││
├── models/                        # Saved model files
│
├── app/                           # Dashboard application
│
├── output/                        # Output figures and analysis
│
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation

## Getting Started

### Prerequisites

- Python 3.12.9
- Required packages listed in `requirements.txt`

### Installation

1. Clone this repository: 
git clone https://github.com/nethmii98/Dynamic-Pricing-Simulator-Project.git
cd Dynamic-Pricing-Simulator-Project

2. Create and activate a virtual environment:
python -m venv "Path\venv"
venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Download the NYC Taxi data:
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet

### Running the Project

The project workflow consists of several steps, implemented as Jupyter notebooks:

1. **Data Exploration**: Explore and understand the taxi trip data
jupyter notebook notebooks/01_data_exploration.ipynb

2. **Demand Analysis**: Analyze temporal demand patterns
jupyter notebook notebooks/02_demand_analysis.ipynb

3. **Elasticity Modeling**: Estimate price elasticity of demand
jupyter notebook notebooks/03_elasticity_modeling.ipynb

4. **RL Training**: Train reinforcement learning agent for pricing
jupyter notebook notebooks/04_rl_training.ipynb

5. **Strategy Evaluation**: Compare different pricing strategies
jupyter notebook notebooks/05_strategy_evaluation.ipynb

### Dashboard

To run the interactive dashboard:
cd app
python app.py

This will start the dashboard on http://localhost:8050.

## Key Components

### Demand Pattern Analysis

The project analyzes historical taxi data to understand demand patterns by:
- Hour of day
- Day of week
- Geographic locations
- Special conditions (weather, events)

### Price Elasticity Modeling

Price elasticity of demand is estimated for different time periods:
- Peak hours
- Off-peak hours
- Weekends
- Overall average

### Reinforcement Learning Environment

A custom OpenAI Gym environment simulates the ride-sharing market:
- State: time, day, demand level, competitor prices
- Action: price multiplier (0.7x - 2.5x base price)
- Reward: revenue generated

### Dynamic Pricing Strategies

Several pricing strategies are implemented and compared:
- Constant pricing
- Time-based pricing
- Surge pricing (demand-based)
- Competitive pricing
- RL-optimized pricing

## Results

The RL-based pricing strategy consistently outperforms baseline approaches:
- Revenue improvement: 580% compared to constant pricing
- Optimized price multipliers during peak demand periods
- Intelligent response to changing market conditions
- Projected annual revenue increase of $7 million

## Acknowledgments

- NYC Taxi and Limousine Commission for providing open data
- OpenAI Gym and Stable Baselines3 for RL implementation
- Dash and Plotly for interactive visualizations
