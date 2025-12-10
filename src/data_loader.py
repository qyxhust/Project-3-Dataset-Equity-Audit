"""
Data loading utilities for VCF and panel files.
Handles caching and population mapping.
"""

import allel
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Tuple, List


def load_vcf_with_cache(vcf_file: str, cache_file: str) -> dict:
    """
    Load VCF data from file or cache.

    Args:
        vcf_file: Path to VCF file
        cache_file: Path to cache pickle file

    Returns:
        callset dictionary from allel.read_vcf
    """
    if os.path.exists(cache_file):
        print("Loading from cache...")
        with open(cache_file, "rb") as f:
            callset = pickle.load(f)
        print("Cache loaded!")
    else:
        print("Reading VCF... this might take a minute")
        callset = allel.read_vcf(vcf_file, fields=["variants/*", "calldata/GT", "samples"])
        print("Saving to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(callset, f)
        print("Cache saved!")

    return callset


def load_population_mapping(panel_file: str) -> pd.DataFrame:
    """
    Load panel file mapping samples to populations.

    Args:
        panel_file: Path to panel file (tab-separated)

    Returns:
        DataFrame with sample to population mapping
    """
    panel = pd.read_csv(panel_file, sep="\t")
    return panel


def map_samples_to_populations(samples: np.ndarray, panel: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Map VCF samples to superpopulations.

    Args:
        samples: Array of sample IDs from VCF
        panel: DataFrame with sample to population mapping

    Returns:
        Tuple of (sample_superpops list, sorted superpops list, pop_indices dict)
    """
    # Create sample ID to superpopulation mapping
    sample_to_superpop = dict(zip(panel["sample"], panel["super_pop"]))

    # Map VCF samples to superpopulations
    sample_superpops = [sample_to_superpop.get(s, "Unknown") for s in samples]
    superpops = sorted(set(sample_superpops))

    # Create indices mapping for each population
    sample_superpops_array = np.array(sample_superpops)
    pop_indices = {}
    for pop in superpops:
        mask = sample_superpops_array == pop
        pop_indices[pop] = np.where(mask)[0]

    return sample_superpops, superpops, pop_indices


def print_population_summary(sample_superpops: List[str], superpops: List[str]):
    """Print summary of populations found and sample counts."""
    print(f"\nSuperpopulations found: {superpops}")
    for pop in superpops:
        count = sample_superpops.count(pop)
        print(f"  {pop}: {count} samples")
