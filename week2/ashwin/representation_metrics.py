"""
Author: Ashwin Kalyan
Date: 2025-10-30
Organization: Computational Biology at Berkeley

This file computes Chi-square goodness-of-fit and KL divergence for superpopulation representation and generates the CSV chi_kl_results.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import chisquare, entropy

def load_data(path):
    """
    Load our data files
    
    Args:
        path: Path to CSV file
    
    Returns:
        DataFrame with sample data
    """
    return pd.read_csv(path)

def compute_observed_counts(df):
    """
    Compute observed superpopulation counts
    
    Args:
        df: Panel DataFrame
    
    Returns:
        Dictionary of superpopulation -> count
    """
    return df['super_pop'].value_counts().to_dict()

def compute_expected_counts(observed):
    """
    Compute expected counts assuming uniform distribution
    
    Args:
        observed: Dictionary of observed counts
    
    Returns:
        Dictionary of expected counts (uniform distribution)
    """
    
    total_samples = sum(observed.values())  # Use actual sum of observed
    pops = sorted(observed.keys())
    
    result = {}
    for pop in pops:
        expected_per_pop = total_samples / len(pops)
        result[pop] = expected_per_pop

    return result

def chi_square_gof(observed, expected):
    """
    Compute Chi-square goodness-of-fit test
    
    Args:
        observed: Observed counts per superpopulation
        expected: Expected counts per superpopulation
    
    Returns:
        Tuple of (chi2_statistic, p_value)
    """
    # Ensure same ordering
    pops = sorted(observed.keys())
    obs_array = np.array([observed[pop] for pop in pops])
    exp_array = np.array([expected[pop] for pop in pops])
    
    chi2_stat, p_value = chisquare(f_obs=obs_array, f_exp=exp_array)
    return chi2_stat, p_value

def kl_divergence(observed, expected):
    """
    Compute Kullback-Leibler divergence
    
    Args:
        observed: Observed counts per superpopulation
        expected: Expected counts per superpopulation
    
    Returns:
        KL divergence value
    """
    # Ensure same ordering
    pops = sorted(observed.keys())
    
    # Convert to probability distributions
    total_obs = sum(observed.values())
    total_exp = sum(expected.values())
    
    obs_prob = np.array([observed[pop] / total_obs for pop in pops])
    exp_prob = np.array([expected[pop] / total_exp for pop in pops])
    
    # Compute KL divergence: sum(P(x) * log(P(x) / Q(x)))
    kl_div = entropy(obs_prob, exp_prob)
    return kl_div

ps_df = load_data('population_summary.csv')
ps_observed = compute_observed_counts(ps_df)
ps_expected = compute_expected_counts(ps_observed)
ps_chi2 = chi_square_gof(ps_observed, ps_expected)
ps_kl = kl_divergence(ps_observed, ps_expected)

print("Data: \n\n" + str(ps_df) + "\n")
print("Chi-squared Goodness of Fit, P-value: " + str(ps_chi2) + "\nKL Divergence: " + str(ps_kl))

# Create results dataframe
results_data = {
    'Dataset': ['population_summary'],
    'Chi-squared Statistic': [ps_chi2[0]],
    'P-value': [ps_chi2[1]],
    'KL Divergence': [ps_kl]
}

results_df = pd.DataFrame(results_data)
results_df.to_csv('chi_kl_results.csv', index=False) # Append to CSV