"""
run_pi_analysis.py

Script to compute nucleotide diversity (π) and heterozygosity (H_S) 
per superpopulation from VCF data.

Author: Ashwin Kalyan (Task Runner)
Date: 2025-11-08
Organization: Computational Biology at Berkeley

This script:
1. Loads VCF data and population mappings
2. Computes π (pi) per superpopulation using allel.sequence_diversity()
3. Computes H_S (observed heterozygosity) per sample and summarizes by superpopulation
4. Saves results to CSV with columns: superpopulation, pi, hs_mean, hs_sd, n_samples
"""

import allel
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Import from existing modules (Cyan's code)
from config import VCF_FILE, PANEL_FILE, CACHE_FILE
from data_loader import (
    load_vcf_with_cache,
    load_population_mapping,
    map_samples_to_populations
)
from population_genetics import (
    calculate_nucleotide_diversity_per_population,
    calculate_sequence_diversity_per_population
)


def calculate_heterozygosity_per_sample(genotypes: allel.GenotypeArray) -> np.ndarray:
    """
    Calculate observed heterozygosity for each sample across all variants.
    
    Formula: H_S = (number of heterozygous genotypes) / (total genotypes)
    
    Args:
        genotypes: GenotypeArray of shape (n_variants, n_samples, ploidy)
    
    Returns:
        Array of heterozygosity values per sample
    """
    # Get heterozygosity for each variant (returns shape: n_variants, n_samples)
    is_het = genotypes.is_het()
    
    # Calculate mean heterozygosity per sample across all variants
    hs_per_sample = np.mean(is_het, axis=0)
    
    return hs_per_sample


def summarize_hs_by_population(
    hs_per_sample: np.ndarray,
    sample_superpops: list,
    superpops: list
) -> pd.DataFrame:
    """
    Summarize heterozygosity statistics by superpopulation.
    
    Args:
        hs_per_sample: Array of H_S values per sample
        sample_superpops: List mapping each sample to its superpopulation
        superpops: List of unique superpopulations
    
    Returns:
        DataFrame with columns: superpopulation, hs_mean, hs_sd, n_samples
    """
    sample_superpops_array = np.array(sample_superpops)
    
    results = []
    for pop in superpops:
        mask = sample_superpops_array == pop
        pop_hs = hs_per_sample[mask]
        
        results.append({
            'superpopulation': pop,
            'hs_mean': np.mean(pop_hs),
            'hs_sd': np.std(pop_hs, ddof=1),  # Sample standard deviation
            'n_samples': len(pop_hs)
        })
    
    return pd.DataFrame(results)


def main():
    """Main execution function."""
    
    # Step 1: Load VCF data
    callset = load_vcf_with_cache(VCF_FILE, CACHE_FILE)
    
    # Extract genotype and position data
    genotypes = allel.GenotypeArray(callset["calldata/GT"])
    samples = callset["samples"]
    pos = callset["variants/POS"]
    
    # Step 2: Load population mapping
    panel = load_population_mapping(PANEL_FILE)
    sample_superpops, superpops, pop_indices = map_samples_to_populations(samples, panel)
    
    # Step 3: Calculate π (nucleotide diversity) per population
    pi_results = calculate_nucleotide_diversity_per_population(
        genotypes, sample_superpops, superpops
    )
    
    # Also calculate sequence diversity (normalized by sequence length)
    seq_div_results = calculate_sequence_diversity_per_population(
        genotypes, pos, sample_superpops, superpops
    )
    
    # Step 4: Calculate H_S (heterozygosity) per sample
    hs_per_sample = calculate_heterozygosity_per_sample(genotypes)
    hs_summary = summarize_hs_by_population(hs_per_sample, sample_superpops, superpops)
    
    # Step 5: Combine results and save to CSV
    # Create combined dataframe
    results_df = pd.DataFrame([
        {
            'superpopulation': pop,
            'pi': pi_results[pop],
            'sequence_diversity': seq_div_results[pop],
        }
        for pop in superpops
    ])
    
    # Merge with H_S summary
    results_df = results_df.merge(hs_summary, on='superpopulation')
    
    # Reorder columns
    results_df = results_df[[
        'superpopulation', 'pi', 'sequence_diversity', 
        'hs_mean', 'hs_sd', 'n_samples'
    ]]
    
    # Sort by pi (descending)
    results_df = results_df.sort_values('pi', ascending=False)
    
    # Save to CSV
    output_file = 'pi_values.csv'
    results_df.to_csv(output_file, index=False)
    
    return results_df


if __name__ == "__main__":
    results = main()
    sys.exit(0)