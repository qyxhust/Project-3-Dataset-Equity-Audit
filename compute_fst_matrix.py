"""
compute_fst_matrix.py

Script to compute pairwise F_ST between all superpopulation pairs
using Weir-Cockerham method.

Author: Ashwin Kalyan
Date: 2025-11-14
Organization: Computational Biology at Berkeley

This script:
1. Loads VCF data and population mappings
2. Computes pairwise F_ST for all superpopulation pairs
3. Creates a symmetric F_ST matrix
4. Saves results to fst_matrix.csv
"""

import allel
import pandas as pd
import numpy as np
import sys
from itertools import combinations
from pathlib import Path

# Import from existing modules
from config import VCF_FILE, PANEL_FILE, CACHE_FILE
from data_loader import (
    load_vcf_with_cache,
    load_population_mapping,
    map_samples_to_populations
)


def compute_pairwise_fst(
    genotypes: allel.GenotypeArray,
    pop_indices: dict,
    superpops: list,
    exclude_unknown: bool = True,
    max_variants: int = 50000
) -> dict:
    """
    Compute pairwise F_ST between all superpopulation pairs.
    
    Args:
        genotypes: GenotypeArray for all samples
        pop_indices: Dictionary mapping population to sample indices
        superpops: List of superpopulations
        exclude_unknown: Whether to exclude "Unknown" population
        max_variants: Maximum number of variants to use (for speed)
    
    Returns:
        Dictionary mapping (pop1, pop2) tuples to F_ST values
    """
    fst_results = {}
    
    # Filter populations if needed
    pops_to_compare = [p for p in superpops if p != "Unknown"] if exclude_unknown else superpops
    
    # Subsample variants if necessary for speed
    n_variants = genotypes.shape[0]
    if n_variants > max_variants:
        step = n_variants // max_variants
        genotypes = genotypes[::step]
    
    # Compute F_ST for all pairs
    for pop1, pop2 in combinations(pops_to_compare, 2):
        # Get sample indices for both populations
        subpops = [pop_indices[pop1], pop_indices[pop2]]
        
        # Compute Weir-Cockerham F_ST
        a, b, c = allel.weir_cockerham_fst(genotypes, subpops=subpops)
        
        # Calculate F_ST: sum(a) / (sum(a) + sum(b) + sum(c))
        fst = np.sum(a) / (np.sum(a) + np.sum(b) + np.sum(c))
        
        fst_results[(pop1, pop2)] = fst
    
    return fst_results


def create_fst_matrix(fst_results: dict, populations: list) -> pd.DataFrame:
    """
    Create a symmetric F_ST matrix from pairwise results.
    
    Args:
        fst_results: Dictionary of (pop1, pop2) -> F_ST value
        populations: List of population names
    
    Returns:
        DataFrame with F_ST matrix (symmetric, diagonal = 0)
    """
    n_pops = len(populations)
    matrix = np.zeros((n_pops, n_pops))
    
    pop_to_idx = {pop: i for i, pop in enumerate(populations)}
    
    # Fill matrix with F_ST values
    for (pop1, pop2), fst in fst_results.items():
        i = pop_to_idx[pop1]
        j = pop_to_idx[pop2]
        
        # Symmetric matrix
        matrix[i, j] = fst
        matrix[j, i] = fst
    
    # Create DataFrame
    df = pd.DataFrame(matrix, index=populations, columns=populations)
    
    return df


def main():
    """Main execution function."""
    
    # Load VCF data
    callset = load_vcf_with_cache(VCF_FILE, CACHE_FILE)
    genotypes = allel.GenotypeArray(callset["calldata/GT"])
    samples = callset["samples"]
    
    # Load population mapping
    panel = load_population_mapping(PANEL_FILE)
    sample_superpops, superpops, pop_indices = map_samples_to_populations(samples, panel)
    
    # Compute pairwise F_ST
    fst_results = compute_pairwise_fst(
        genotypes=genotypes,
        pop_indices=pop_indices,
        superpops=superpops,
        exclude_unknown=True
    )
    
    # Create F_ST matrix
    pops_analyzed = sorted([p for p in superpops if p != "Unknown"])
    fst_matrix = create_fst_matrix(fst_results, pops_analyzed)
    
    # Save to CSV
    output_file = 'fst_matrix.csv'
    fst_matrix.to_csv(output_file)
    
    return fst_matrix


if __name__ == "__main__":
    fst_matrix = main()
    sys.exit(0)