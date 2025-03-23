import gym
from gym import spaces
import numpy as np
import pandas as pd

class RideSharingPricingEnv(gym.Env):
    """
    Reinforcement Learning environment for ride-sharing pricing optimization.

    This environment simulates a ride sharing market where the agent sets prices,
    and demand responds according to estimated elasticities.

    """

    def __init__(self, demand_data, elasticity_data, config=None):
        """
        Initialize the environment.

        Parameters:
        -----------
        demand_data : pandas.DataFrame
            Data containing hourly demand patterns
        elasticity_data : dict
            Dictionary with elasticity estimates for different segments
        config : dict, optional
            Configuration parameters for the environment
        """
        
        super(RideSharingPricingEnv, self).__init__() # Without this some gym.Env features might not work properly.

        # Store input data
        self.demand_data = demand_data
        self.elasticity_data = elasticity_data

        # Default configuration
        self.config = {
            'base_price': 10.0,
            'simulation_length': 168, # One week in hours
            'demand_noise': 0.1, # Random noise in demand (standard deviation)
            'max_price_multiplier': 2.5,
            'min_price_multiplier' : 0.7,
        }

        # Update with provided config
        if config:
            self.config.update(config)

        # Define action space (continuous price multiplier) - > spaces. 
        # The action space represents all possible price multipliers
        self.action_space = spaces.Box(
            low = np.array([self.config['min_price_multiplier']]),
            high = np.array([self.config['max_price_multiplier']]),
            dtype = np.float32
        ) #The RL agent learns to set the price dynamically based on demand, competition, and events.


        # Define observation space 
        # The observation space contains all the variables that describe the current state of the ride-sharing market.
        # [hour, day, recent_demand_ratio, competitor_price_ratio
        # wait_time, is_special_event, week_of_month]
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0.5, 0, 0, 0]),
            high=np.array([23, 6, 2.0, 2.0, 30, 1, 4]),
            dtype=np.float32
        )

        # Initialize state
        self.reset()

    def reset(self):
        """Reset the environment to start a new episode."""
        # Start simulation at a random time
        self.current_hour = np.random.randint(0, 24)
        self.current_day = np.random.randint(0, 7)
        self.week_of_month = np.random.randint(0, 4)
        self.step_count = 0

        # State variables
        self.baseline_demand = self._get_baseline_demand()
        self.recent_demand_ratio = 1.0 # Actual/expected
        self.competitor_price_ratio = 1.0 # Our price/ competitor price
        self.recent_wait_time = 5.0 # Minutes
        self.is_special_event = 0 # Binary flag

        # Performance tracking
        self.total_revenue = 0
        self.total_rides = 0
        self.price_history = []
        self.demand_history = []

        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment.

        Parameters:
        -----------
        action : array-like
            Price multiplier action

        Returns:
        --------
        tuple
            (observation, reward, done, info)
        """

        # Extract price multiplier from action
        price_multiplier = float(action[0])

        # Get current price
        current_price = self.config['base_price'] * price_multiplier

        # Record price
        self.price_history.append(current_price)

        # Get baseline demand for this period
        baseline_demand = self._get_baseline_demand()
 
        # Get appropriate elasticity for current period
        elasticity = self._get_current_elasticity()

        # Apply elasticity to determine actual demand 
        # Add randomness to simulate market variations
        demand_noise = np.random.normal(1.0, self.config['demand_noise'])
        actual_demand = baseline_demand * (price_multiplier ** elasticity) * demand_noise

        # Apply special event effect if active
        if self.is_special_event:
            actual_demand *= 1.5 # 50% more demand during events

        # Calculate revenue
        revenue = actual_demand * current_price

        # Record demand
        self.demand_history.append(actual_demand)

        # Update state
        self.recent_demand_ratio = actual_demand / baseline_demand
        self.recent_wait_time = max(1, min(30, 5 * (actual_demand / baseline_demand))) # wait time adjusts between 1min - 30min

        # Simulate competitor response (simple model)
        if self.step_count % 4 == 0: # Competitors adjust every 4 hours (% finds the remainder)
            competitor_adjustment = np.random.normal(1.0, 0.05)
            competitor_price = self.config['base_price'] * competitor_adjustment
            self.competitor_price_ratio = current_price / competitor_price

        # Update time
        self.current_hour = (self.current_hour + 1) % 24
        if self.current_hour == 0:
            self.current_day = (self.current_day + 1) % 7
            if self.current_day == 0:
                self.week_of_month = (self.week_of_month + 1) % 4

        # Random chance of special event occuring
        if np.random.random() < 0.02: # 2% chance each hour
            self.is_special_event = 1
        else:
            self.is_special_event = 0

        # Update tracking variables
        self.total_revenue += revenue
        self.total_rides += actual_demand
        self.step_count += 1

        # Determine if episode is done
        done = self.step_count >= self.config['simulation_length']

        # Define reward (focus on revenue)
        reward = revenue

        # Add information for debugging/visualization
        info = {
            'price_multiplier': price_multiplier,
            'baseline_demand': baseline_demand,
            'actual_demand': actual_demand,
            'revenue': revenue,
            'wait_time': self.recent_wait_time,
            'elasticity': elasticity
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        """Construct the observation vector representing current state.""" 
        return np.array([
            self.current_hour,
            self.current_day,
            self.recent_demand_ratio,
            self.competitor_price_ratio,
            self.recent_wait_time,
            self.is_special_event,
            self.week_of_month
        ])
    
    def _get_baseline_demand(self):
        """Get expected baseline demand for current time period."""
        # Find demand for current hour and day
        mask = (self.demand_data['hour'] == self.current_hour) & (self.demand_data['day_of_week'] == self.current_day)
        if mask.any():
            return self.demand_data.loc[mask, 'trip_count'].values[0]
        else:
            return 100 # Default Value
        
    def _get_current_elasticity(self):
        """Determine appropriate elasticity for current time period."""
        # Check if current hour is peak
        if self.current_hour in [7, 8, 9, 17, 18, 19]:
            return self.elasticity_data['peak']['elasticity']
        # Check if weekend
        elif self.current_day >= 5:
            return self.elasticity_data['weekend']['elasticity']
        # Otherwise off-peak weekday
        else:
            return self.elasticity_data['offpeak']['elasticity']

