import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np

def create_dashboard(simulation_results, ab_test_results, elasticity_data):
    """
    Create an interactive dashboard for the pricing simulator.
    
    Parameters:
    -----------
    simulation_results : dict
        Results from a simulation run
    ab_test_results : dict
        Results from an A/B test
    elasticity_data : dict
        Elasticity estimates for different segments
        
    Returns:
    --------
    dash.Dash
        Dash application
    """

    # Create a simplified DataFrame specifically for the revenue chart
    revenue_data = pd.DataFrame({
        'strategy': ab_test_results['summary']['total_revenue']['mean'].index,
        'mean_revenue': ab_test_results['summary']['total_revenue']['mean'].values
    })

    rides_data = pd.DataFrame({
        'strategy': ab_test_results['summary']['total_rides']['mean'].index,
        'mean_rides': ab_test_results['summary']['total_rides']['mean'].values
    })
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Ride-Sharing Dynamic Pricing Simulator", className="mt-4"),
                html.P("Analyze and compare pricing strategies based on NYC Taxi data")
            ])
        ]),
        
        # Overview Metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Revenue Comparison"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='revenue-comparison',
                            figure=px.bar(
                                revenue_data,
                                x='strategy',
                                y='mean_revenue',
                                title="Average Revenue by Pricing Strategy",
                                labels={'mean_revenue': 'Mean Revenue'},
                                color='strategy'
                            )
                        )
                    ])
                ])
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Rides Comparison"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='rides-comparison',
                            figure=px.bar(
                                rides_data,
                                x='strategy',
                                y='mean_rides',
                                title="Average Rides by Pricing Strategy",
                                labels={'mean_rides': 'Mean Rides'},
                                color='strategy'
                            )
                        )
                    ])
                ])
            ], width=6)
        ]),
        
        # Elasticity Analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Demand vs Price Analysis"),
                    dbc.CardBody([
                        html.P("Select Time Period:"),
                        dcc.Dropdown(
                            id='elasticity-dropdown',
                            options=[
                                {'label': 'Overall', 'value': 'overall'},
                                {'label': 'Peak Hours', 'value': 'peak'},
                                {'label': 'Off-Peak Hours', 'value': 'offpeak'},
                                {'label': 'Weekend', 'value': 'weekend'}
                            ],
                            value='overall'
                        ),
                        dcc.Graph(id='elasticity-curve')
                    ])
                ])
            ])
        ]),
        
        # Pricing Strategy Visualization
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Hourly Pricing by Strategy"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='hourly-pricing',
                            figure=px.line(
                                ab_test_results['time_performance'],
                                x='hour',
                                y='price_multiplier',
                                color='strategy',
                                title="Average Price Multiplier by Hour of Day"
                            )
                        )
                    ])
                ])
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Daily Pricing by Strategy"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='daily-pricing',
                            figure=px.line(
                                ab_test_results['day_performance'],
                                x='day',
                                y='price_multiplier',
                                color='strategy',
                                title="Average Price Multiplier by Day of Week",
                                labels={'day': 'Day of Week (0=Monday)'}
                            )
                        )
                    ])
                ])
            ], width=6)
        ]),
        
        # Strategy Heatmap
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Pricing Strategy Heatmap"),
                    dbc.CardBody([
                        html.P("Select Strategy:"),
                        dcc.Dropdown(
                            id='strategy-dropdown',
                            options=[
                                {'label': s, 'value': s} for s in ab_test_results['hourly_data']['strategy'].unique()
                            ],
                            value=ab_test_results['hourly_data']['strategy'].unique()[0]
                        ),
                        dcc.Graph(id='strategy-heatmap')
                    ])
                ])
            ])
        ])
    ], fluid=True)
    
    # Add callbacks for interactivity
    @app.callback(
        Output('elasticity-curve', 'figure'),
        [Input('elasticity-dropdown', 'value')]
    )
    def update_elasticity_curve(selected_period):
        elasticity = elasticity_data[selected_period]['elasticity']
        
        # Generate data points for the curve
        price_range = np.linspace(0.7, 2.5, 50)
        demand_values = 100 * (price_range ** elasticity)
        revenue_values = demand_values * price_range
        
        # Create the figure
        fig = go.Figure()
        
        # Add demand curve
        fig.add_trace(go.Scatter(
            x=price_range, 
            y=demand_values,
            mode='lines',
            name='Relative Demand',
            line=dict(color='blue')
        ))
        
        # Add revenue curve
        fig.add_trace(go.Scatter(
            x=price_range, 
            y=revenue_values,
            mode='lines',
            name='Relative Revenue',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title=f"Price Elasticity Analysis for {selected_period.capitalize()} (e={elasticity:.2f})",
            xaxis=dict(title="Price Multiplier"),
            yaxis=dict(title="Relative Value"),
            legend=dict(x=0, y=1.1, orientation='h')
        )
        
        # Mark the optimal price point (where revenue is maximized)
        optimal_idx = np.argmax(revenue_values)
        optimal_price = price_range[optimal_idx]
        optimal_revenue = revenue_values[optimal_idx]
        
        fig.add_shape(
            type="line",
            x0=optimal_price,
            y0=0,
            x1=optimal_price,
            y1=optimal_revenue,
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig.add_annotation(
            x=optimal_price,
            y=optimal_revenue/2,
            text=f"Optimal Price: {optimal_price:.2f}x",
            showarrow=False,
            font=dict(color="red")
        )
        
        return fig
    
    @app.callback(
        Output('strategy-heatmap', 'figure'),
        [Input('strategy-dropdown', 'value')]
    )
    def update_strategy_heatmap(selected_strategy):
        # Filter data for selected strategy
        strategy_data = ab_test_results['hourly_data'][
            ab_test_results['hourly_data']['strategy'] == selected_strategy
        ]
        
        # Calculate average price multiplier for each hour-day combination
        pivot_data = strategy_data.groupby(['day', 'hour'])['price_multiplier'].mean().reset_index()
        
        # Create pivot table for heatmap
        heatmap_data = pivot_data.pivot(index='day', columns='hour', values='price_multiplier')
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Hour of Day", y="Day of Week", color="Price Multiplier"),
            title=f"Price Multipliers for {selected_strategy}",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        
        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            yaxis=dict(
                tickmode='linear',
                ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                tickvals=[0, 1, 2, 3, 4, 5, 6]
            )
        )
        
        return fig
    
    return app