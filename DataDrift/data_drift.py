import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, jensenshannon
from skmultiflow.drift_detection import DDM, ADWIN


def calculate_statistical_distance(data1, data2, method='ks'):
    """
    Calculates statistical distance between two datasets for a given method.

    Args:
        data1 (pd.Series or np.ndarray): First dataset.
        data2 (pd.Series or np.ndarray): Second dataset.
        method (str): Statistical distance method ('ks' for Kolmogorov-Smirnov,
                      'js' for Jensen-Shannon).

    Returns:
        float: The calculated statistical distance.
    """
    if method == 'ks':
        return ks_2samp(data1, data2).statistic
    elif method == 'js':
        # For JS divergence, we need probability distributions.
        # We can use histograms to approximate this.
        # This is a basic implementation, more sophisticated methods might be needed
        # for accurate JS divergence on continuous data.
        bins = min(len(np.unique(data1)), len(np.unique(data2)), 50)
        hist1, _ = np.histogram(data1, bins=bins, density=True)
        hist2, _ = np.histogram(data2, bins=bins, density=True)
        # Ensure the histograms have the same length for JS calculation
        min_len = min(len(hist1), len(hist2))
        hist1 = hist1[:min_len]
        hist2 = hist2[:min_len]

        # Add a small epsilon to avoid log(0)
        hist1 = hist1 + 1e-10
        hist2 = hist2 + 1e-10

        return jensenshannon(hist1, hist2)
    else:
        raise ValueError("Invalid statistical distance method.")


def detect_drift_ddm(data1, data2, min_samples=30):
    """
    Detects drift using the DDM drift detection method.

    Args:
        data1 (pd.Series or np.ndarray): Reference dataset.
        data2 (pd.Series or np.ndarray): New incoming data.
        min_samples (int): Minimum number of samples before checking for drift.

    Returns:
        bool: True if drift is detected, False otherwise.
    """
    ddm = DDM()
    # DDM typically works on a stream of errors or changes,
    # here we simulate by comparing to a reference.
    # A more proper streaming implementation would feed data point by data point.
    # For this example, we'll use a simplified approach comparing distributions.
    # This is not the standard way DDM is used but demonstrates the concept.
    # A better approach for batch comparison would involve comparing error rates
    # of a model trained on data1 when predicting on data2.

    # For simplicity, we'll compare if mean of data2 is significantly different from data1
    # based on a threshold derived from DDM principles. This is a simplified example.
    mean_data1 = np.mean(data1)
    mean_data2 = np.mean(data2)
    std_data1 = np.std(data1)

    # This is a placeholder logic, a real DDM implementation needs a stream
    # and monitoring the error rate.
    if len(data2) < min_samples:
        return False

    # Simplified drift detection logic (not standard DDM usage)
    # If the mean of the new data is more than 3 standard deviations away from the
    # reference data mean, consider it drift for this example.
    if abs(mean_data2 - mean_data1) > 3 * std_data1:
        ddm.add_element(1) # Simulate an error/change
    else:
        ddm.add_element(0) # Simulate no error/change

    return ddm.detected_change()


def detect_drift_adwin(data1, data2, delta=0.002):
    """
    Detects drift using the ADWIN drift detection method.

    Args:
        data1 (pd.Series or np.ndarray): Reference dataset.
        data2 (pd.Series or np.ndarray): New incoming data.
        delta (float): Confidence value for ADWIN.

    Returns:
        bool: True if drift is detected, False otherwise.
    """
    adwin = ADWIN(delta=delta)
    # Similar to DDM, ADWIN works on a stream.
    # This is a simplified batch comparison demonstration.
    # A proper streaming implementation would feed data point by data point.

    # For simplicity, we'll compare if the distribution of data2 is
    # significantly different from data1 using ADWIN's internal checks.
    # This is not the standard way ADWIN is used but demonstrates the concept.

    # Feed elements from both distributions to ADWIN (simplified)
    # A more realistic use would be feeding a stream of data points
    # or error rates.
    combined_data = np.concatenate((data1, data2))
    for element in combined_data:
        adwin.add_element(element)
        if adwin.detected_change():
            return True
    return False


def plot_distribution(data1, data2, column_name):
    """
    Plots the distributions of a feature from two datasets.

    Args:
        data1 (pd.Series): Data from the first dataset.
        data2 (pd.Series): Data from the second dataset.
        column_name (str): Name of the feature.
    """
    plt.figure(figsize=(10, 6))
    if pd.api.types.is_numeric_dtype(data1) and pd.api.types.is_numeric_dtype(data2):
        plt.hist(data1, bins=30, alpha=0.5, label='Reference Data', density=True)
        plt.hist(data2, bins=30, alpha=0.5, label='New Data', density=True)
        plt.xlabel(column_name)
        plt.ylabel('Density')
        plt.title(f'Distribution Comparison for {column_name} (Numerical)')
    elif pd.api.types.is_categorical_dtype(data1) or pd.api.types.is_categorical_dtype(data2) or \
         pd.api.types.is_object_dtype(data1) or pd.api.types.is_object_dtype(data2):

        # Handle potential differences in categories between datasets
        all_categories = pd.concat([data1.astype(str), data2.astype(str)]).unique()
        count1 = data1.value_counts().reindex(all_categories, fill_value=0)
        count2 = data2.value_counts().reindex(all_categories, fill_value=0)

        width = 0.35
        x = np.arange(len(all_categories))

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, count1, width, label='Reference Data')
        rects2 = ax.bar(x + width/2, count2, width, label='New Data')

        ax.set_xlabel('Categories')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution Comparison for {column_name} (Categorical)')
        ax.set_xticks(x)
        ax.set_xticklabels(all_categories, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
    else:
        print(f"Warning: Cannot plot distribution for {column_name} due to unsupported data type.")
        plt.close() # Close the figure if not plotted
        return

    plt.legend()
    plt.show()


def check_data_drift(data_ref, data_new, ks_threshold=0.1, js_threshold=0.1, ddm_enabled=False, adwin_enabled=False):
    """
    Checks for data drift between a reference dataset and new data.

    Args:
        data_ref (pd.DataFrame): The reference dataset.
        data_new (pd.DataFrame): The new incoming data.
        ks_threshold (float): Threshold for Kolmogorov-Smirnov test statistic.
        js_threshold (float): Threshold for Jensen-Shannon distance.
        ddm_enabled (bool): Whether to use DDM for drift detection (simplified).
        adwin_enabled (bool): Whether to use ADWIN for drift detection (simplified).

    Returns:
        dict: A dictionary indicating drift status for each column.
    """
    drift_results = {}
    columns = data_ref.columns

    for col in columns:
        drift_results[col] = {
            'ks_drift': False,
            'js_drift': False,
            'ddm_drift': False,
            'adwin_drift': False,
            'message': 'No drift detected'
        }

        if col not in data_new.columns:
            drift_results[col]['message'] = f"Column '{col}' missing in new data."
            continue

        col_ref = data_ref[col]
        col_new = data_new[col]

        # Ensure data types are compatible for comparisons
        if col_ref.dtype != col_new.dtype:
            drift_results[col]['message'] = f"Data type mismatch for column '{col}'."
            plot_distribution(col_ref, col_new, col)
            continue

        # Handle numerical and categorical data separately for statistical tests
        if pd.api.types.is_numeric_dtype(col_ref):
            # Kolmogorov-Smirnov test
            ks_statistic = calculate_statistical_distance(col_ref, col_new, method='ks')
            if ks_statistic > ks_threshold:
                drift_results[col]['ks_drift'] = True
                drift_results[col]['message'] = f"KS drift detected (statistic: {ks_statistic:.4f})"

            # Jensen-Shannon distance
            try:
                js_distance = calculate_statistical_distance(col_ref, col_new, method='js')
                if js_distance > js_threshold:
                    drift_results[col]['js_drift'] = True
                    if 'drift detected' in drift_results[col]['message']:
                        drift_results[col]['message'] += f", JS drift detected (distance: {js_distance:.4f})"
                    else:
                        drift_results[col]['message'] = f"JS drift detected (distance: {js_distance:.4f})"
            except Exception as e:
                print(f"Warning: Could not calculate JS distance for numerical column {col}: {e}")


            # DDM and ADWIN (simplified application for batch)
            if ddm_enabled:
                if detect_drift_ddm(col_ref, col_new):
                    drift_results[col]['ddm_drift'] = True
                    if 'drift detected' in drift_results[col]['message']:
                        drift_results[col]['message'] += ", DDM drift detected"
                    else:
                        drift_results[col]['message'] = "DDM drift detected"

            if adwin_enabled:
                if detect_drift_adwin(col_ref, col_new):
                    drift_results[col]['adwin_drift'] = True
                    if 'drift detected' in drift_results[col]['message']:
                        drift_results[col]['message'] += ", ADWIN drift detected"
                    else:
                        drift_results[col]['message'] = "ADWIN drift detected"

        elif pd.api.types.is_categorical_dtype(col_ref) or pd.api.types.is_object_dtype(col_ref):
            # For categorical data, we can compare the distribution of categories
            # using methods like chi-squared test or simply comparing frequencies.
            # KS and JS are generally for numerical distributions.
            # We'll focus on visualization and potentially frequency comparison.

            # Basic frequency comparison
            freq_ref = col_ref.value_counts(normalize=True)
            freq_new = col_new.value_counts(normalize=True)

            # Simple check: if any category frequency differs by a large amount
            # This is a basic example, more sophisticated methods exist.
            merged_freq = pd.concat([freq_ref, freq_new], axis=1).fillna(0)
            merged_freq.columns = ['ref', 'new']
            if any(abs(merged_freq['ref'] - merged_freq['new']) > 0.1): # Example threshold
                 if 'No drift detected' in drift_results[col]['message']:
                      drift_results[col]['message'] = "Potential categorical drift detected (frequency change)"
                 else:
                      drift_results[col]['message'] += ", Potential categorical drift detected (frequency change)"


        else:
            drift_results[col]['message'] = f"Unsupported data type for column '{col}' for drift detection."

        # Plot distribution regardless of detected drift for visual inspection
        plot_distribution(col_ref, col_new, col)

    return drift_results

if __name__ == '__main__':
    # Example Usage
    # Create some sample data
    np.random.seed(42)
    data_ref = pd.DataFrame({
        'numerical_feature_1': np.random.normal(0, 1, 100),
        'numerical_feature_2': np.random.rand(100),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.3, 0.2])
    })

    # Create new data with some drift
    data_new = pd.DataFrame({
        'numerical_feature_1': np.random.normal(0.5, 1.2, 100), # Mean and std changed
        'numerical_feature_2': np.random.rand(100) + 0.2, # Shifted
        'categorical_feature': np.random.choice(['A', 'B', 'C'], 100, p=[0.2, 0.4, 0.4]) # Frequencies changed
    })

    print("Checking for data drift...")
    drift_status = check_data_drift(data_ref, data_new, ddm_enabled=True, adwin_enabled=True)

    for col, status in drift_status.items():
        print(f"Column '{col}': {status['message']}")

    # Example with no drift
    print("\nChecking for data drift (no drift expected)...")
    data_new_no_drift = pd.DataFrame({
        'numerical_feature_1': np.random.normal(0, 1, 100),
        'numerical_feature_2': np.random.rand(100),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.3, 0.2])
    })
    drift_status_no_drift = check_data_drift(data_ref, data_new_no_drift)

    for col, status in drift_status_no_cols:
        print(f"Column '{col}': {status['message']}")
