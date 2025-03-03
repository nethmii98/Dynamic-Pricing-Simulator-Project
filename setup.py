from setuptools import find_packages, setup

setup(
    name="dynamic_pricing_simulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "gym>=0.26.0",
        "stable-baselines3>=2.0.0",
        "dash>=2.9.0",
        "dash-bootstrap-components>=1.4.0",
        "plotly>=5.14.0",
        "pyarrow>=12.0.0",
        "fastparquet>=2023.4.0",
    ],
)