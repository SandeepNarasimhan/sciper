import pandas as pd
import numpy as np
from data_drift import check_data_drift  # Assuming data_drift.py is in the same directory

# Create sample reference data
np.random.seed(42)
reference_data = pd.DataFrame({
    'numerical_col_1': np.random.normal(loc=0, scale=1, size=100),
    'numerical_col_2': np.random.rand(100) * 10,
    'categorical_col_1': np.random.choice(['A', 'B', 'C'], size=100, p=[0.6, 0.3, 0.1]),
    'categorical_col_2': np.random.choice(['X', 'Y'], size=100, p=[0.8, 0.2])
})

# Create sample new data with simulated drift
np.random.seed(123)
new_data = pd.DataFrame({
    'numerical_col_1': np.random.normal(loc=0.5, scale=1.2, size=100),  # Drift in mean and std
    'numerical_col_2': np.random.rand(100) * 12,                      # Drift in range
    'categorical_col_1': np.random.choice(['A', 'B', 'C', 'D'], size=100, p=[0.4, 0.3, 0.2, 0.1]), # Drift in distribution and new category
    'categorical_col_2': np.random.choice(['X', 'Y', 'Z'], size=100, p=[0.5, 0.3, 0.2])           # Drift in distribution and new category
})

print("Checking for data drift between reference and new data:")
check_data_drift(reference_data, new_data)