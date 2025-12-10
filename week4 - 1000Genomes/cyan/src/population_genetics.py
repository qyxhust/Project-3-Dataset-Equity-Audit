"""
Population genetics calculations.
Contains functions for nucleotide diversity, FST, heterozygosity, and sequence diversity.
"""

import allel
import numpy as np
from typing import Dict, Tuple, List
from itertools import combinations


def calculate_heterozygosity_observed(genotypes: allel.GenotypeArray) -> float:
    """
    Calculate mean observed heterozygosity across all variants.

    Args:
        genotypes: GenotypeArray for all samples

    Returns:
        Mean observed heterozygosity
    """
    ho = allel.heterozygosity_observed(genotypes)
    return np.mean(ho)


def calculate_nucleotide_diversity_per_population(
    genotypes: allel.GenotypeArray,
    sample_superpops: List[str],
    superpops: List[str]
) -> Dict[str, float]:
    """
    Calculate nucleotide diversity (π) for each population.

    Args:
        genotypes: GenotypeArray for all samples
        sample_superpops: List mapping each sample to its superpopulation
        superpops: List of unique superpopulations

    Returns:
        Dictionary mapping population to nucleotide diversity
    """
    sample_superpops_array = np.array(sample_superpops)
    pi_results = {}

    for pop in superpops:
        mask = sample_superpops_array == pop
        pop_genotypes = genotypes[:, mask, :]
        allele_count = pop_genotypes.count_alleles()
        pi = np.mean(allel.mean_pairwise_difference(ac=allele_count))
        pi_results[pop] = pi

    return pi_results


def calculate_pairwise_fst(
    genotypes: allel.GenotypeArray,
    pop_indices: Dict[str, np.ndarray],
    superpops: List[str],
    exclude_unknown: bool = True
) -> Dict[Tuple[str, str], float]:
    """
    Calculate pairwise FST between all population pairs.

    Args:
        genotypes: GenotypeArray for all samples
        pop_indices: Dictionary mapping population to sample indices
        superpops: List of superpopulations
        exclude_unknown: Whether to exclude "Unknown" population

    Returns:
        Dictionary mapping (pop1, pop2) tuples to FST values
    """
    fst_results = {}

    pops_to_compare = [p for p in superpops if p != "Unknown"] if exclude_unknown else superpops

    for pop1, pop2 in combinations(pops_to_compare, 2):
        subpops = [pop_indices[pop1], pop_indices[pop2]]
        a, b, c = allel.weir_cockerham_fst(genotypes, subpops=subpops)
        fst = np.sum(a) / (np.sum(a) + np.sum(b) + np.sum(c))
        fst_results[pop1, pop2] = fst

    return fst_results


def calculate_sequence_diversity_per_population(
    genotypes: allel.GenotypeArray,
    pos: np.ndarray,
    sample_superpops: List[str],
    superpops: List[str]
) -> Dict[str, float]:
    """
    Calculate sequence diversity for each population.

    Args:
        genotypes: GenotypeArray for all samples
        pos: Array of variant positions
        sample_superpops: List mapping each sample to its superpopulation
        superpops: List of unique superpopulations

    Returns:
        Dictionary mapping population to sequence diversity
    """
    sample_superpops_array = np.array(sample_superpops)
    diversity_results = {}

    for pop in superpops:
        mask = sample_superpops_array == pop
        pop_genotypes = genotypes[:, mask, :]
        allele_count = pop_genotypes.count_alleles()
        seq_div = allel.sequence_diversity(pos, ac=allele_count)
        diversity_results[pop] = seq_div

    return diversity_results
