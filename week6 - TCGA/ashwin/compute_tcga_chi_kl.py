"""
compute_tcga_chi_kl.py

Script to compute Chi-square test and KL divergence comparing TCGA to 1000G.

Author: Ashwin Kalyan
Date: 2025-12-05
Organization: Computational Biology at Berkeley

This script:
1. Loads TCGA population summary
2. Loads 1000G population summary
3. Computes Chi-square goodness-of-fit test (TCGA vs 1000G)
4. Computes KL divergence (TCGA vs 1000G)
5. Saves results to tcga_chi_kl.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import chisquare, entropy


def load_population_summary(file_path: str) -> pd.DataFrame:
    """
    Load population summary CSV file.
    
    Args:
        file_path: Path to population summary CSV
        
    Returns:
        DataFrame with population summary
    """
    return pd.read_csv(file_path)


def get_superpop_distribution(df: pd.DataFrame) -> dict:
    """
    Extract superpopulation distribution from population summary.
    
    Args:
        df: Population summary DataFrame
        
    Returns:
        Dictionary mapping superpopulation -> count
    """
    # Group by super_pop and sum counts
    superpop_counts = df.groupby('super_pop')['super_num_indiv'].sum().to_dict()
    return superpop_counts


def align_distributions(tcga_dist: dict, kg1k_dist: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two distributions to have the same superpopulations.
    
    Args:
        tcga_dist: TCGA superpopulation distribution
        kg1k_dist: 1000G superpopulation distribution
        
    Returns:
        Tuple of (tcga_array, kg1k_array) with aligned superpopulations
    """
    # Get all unique superpopulations from both
    all_superpops = sorted(set(list(tcga_dist.keys()) + list(kg1k_dist.keys())))
    
    tcga_array = np.array([tcga_dist.get(pop, 0) for pop in all_superpops])
    kg1k_array = np.array([kg1k_dist.get(pop, 0) for pop in all_superpops])
    
    return tcga_array, kg1k_array, all_superpops


def chi_square_gof(observed: np.ndarray, expected: np.ndarray) -> Tuple[float, float]:
    """
    Compute Chi-square goodness-of-fit test.
    
    Args:
        observed: Observed counts (TCGA)
        expected: Expected counts (1000G, normalized to TCGA total)
        
    Returns:
        Tuple of (chi2_statistic, p_value)
    """
    # Normalize expected to match observed total
    observed_total = np.sum(observed)
    expected_total = np.sum(expected)
    
    if expected_total > 0:
        expected_normalized = expected * (observed_total / expected_total)
    else:
        expected_normalized = expected
    
    # Filter: only include categories where expected > 0 (to avoid division by zero)
    # Also include categories where observed > 0 but expected = 0 (these contribute to chi-square)
    # For categories where both are 0, exclude them
    mask = expected_normalized > 0
    obs_filtered = observed[mask]
    exp_filtered = expected_normalized[mask]
    
    # Also add categories where observed > 0 but expected = 0
    # These represent populations in TCGA but not in 1000G
    additional_mask = (observed > 0) & (expected_normalized == 0)
    if np.any(additional_mask):
        obs_additional = observed[additional_mask]
        # For these, we can't compute chi-square contribution (division by zero)
        # So we'll use a very small expected value (1e-10) to allow calculation
        exp_additional = np.full(len(obs_additional), 1e-10)
        obs_filtered = np.concatenate([obs_filtered, obs_additional])
        exp_filtered = np.concatenate([exp_filtered, exp_additional])
    
    if len(obs_filtered) == 0:
        return 0.0, 1.0
    
    # Compute chi-square
    chi2_stat, p_value = chisquare(f_obs=obs_filtered, f_exp=exp_filtered)
    
    return chi2_stat, p_value


def kl_divergence(observed: np.ndarray, expected: np.ndarray) -> float:
    """
    Compute Kullback-Leibler divergence.
    
    KL(P||Q) = sum(P(x) * log(P(x) / Q(x)))
    where P is observed (TCGA) and Q is expected (1000G)
    
    Args:
        observed: Observed distribution (TCGA)
        expected: Expected/reference distribution (1000G)
        
    Returns:
        KL divergence value
    """
    # Convert to probability distributions
    observed_total = np.sum(observed)
    expected_total = np.sum(expected)
    
    if observed_total == 0 or expected_total == 0:
        return np.nan
    
    obs_prob = observed / observed_total
    exp_prob = expected / expected_total
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    obs_prob = obs_prob + epsilon
    exp_prob = exp_prob + epsilon
    # Renormalize
    obs_prob = obs_prob / np.sum(obs_prob)
    exp_prob = exp_prob / np.sum(exp_prob)
    
    # Compute KL divergence
    kl_div = entropy(obs_prob, exp_prob)
    
    return kl_div


def main(
    tcga_summary_file: str = 'tcga_population_summary.csv',
    kg1k_summary_file: str = 'population_summary.csv',
    output_file: str = 'tcga_chi_kl.csv'
):
    """
    Main execution function.
    
    Args:
        tcga_summary_file: Path to TCGA population summary CSV
        kg1k_summary_file: Path to 1000G population summary CSV
        output_file: Output CSV file path
    """
    print("Loading population summaries...")
    
    # Load TCGA population summary
    print(f"Loading TCGA summary from: {tcga_summary_file}")
    tcga_df = load_population_summary(tcga_summary_file)
    tcga_dist = get_superpop_distribution(tcga_df)
    
    print(f"TCGA superpopulations: {tcga_dist}")
    
    # Load 1000G population summary
    print(f"Loading 1000G summary from: {kg1k_summary_file}")
    kg1k_df = load_population_summary(kg1k_summary_file)
    kg1k_dist = get_superpop_distribution(kg1k_df)
    
    print(f"1000G superpopulations: {kg1k_dist}")
    
    # Align distributions
    tcga_array, kg1k_array, superpops = align_distributions(tcga_dist, kg1k_dist)
    
    print(f"\nAligned superpopulations: {superpops}")
    print(f"TCGA counts: {dict(zip(superpops, tcga_array))}")
    print(f"1000G counts: {dict(zip(superpops, kg1k_array))}")
    
    # Compute Chi-square test
    print("\nComputing Chi-square goodness-of-fit test...")
    chi2_stat, p_value = chi_square_gof(tcga_array, kg1k_array)
    
    print(f"Chi-squared statistic: {chi2_stat:.6f}")
    print(f"P-value: {p_value:.6f}")
    
    # Compute KL divergence
    print("\nComputing KL divergence...")
    kl_div = kl_divergence(tcga_array, kg1k_array)
    
    print(f"KL divergence (TCGA || 1000G): {kl_div:.6f}")
    
    # Create results dataframe
    results_data = {
        'Dataset': ['TCGA vs 1000G'],
        'Chi-squared Statistic': [chi2_stat],
        'P-value': [p_value],
        'KL Divergence': [kl_div]
    }
    
    results_df = pd.DataFrame(results_data)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")
    
    return results_df


if __name__ == "__main__":
    import sys
    
    tcga_file = sys.argv[1] if len(sys.argv) > 1 else 'tcga_population_summary.csv'
    kg1k_file = sys.argv[2] if len(sys.argv) > 2 else 'population_summary.csv'
    output_file = sys.argv[3] if len(sys.argv) > 3 else 'tcga_chi_kl.csv'
    
    results = main(tcga_file, kg1k_file, output_file)
    print("\nDone!")

