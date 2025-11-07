"""
Main VCF analysis script.
Orchestrates data loading, population genetics calculations, and result output.

Author: Ashwin Kalyan
Date: 2025-10-30
Organization: Computational Biology at Berkeley
"""

import allel
import pandas as pd
import numpy as np
import os
import time

from config import (
    VCF_FILE, PANEL_FILE, CACHE_FILE,
    NUCLEOTIDE_DIVERSITY_CSV, PAIRWISE_FST_CSV, SEQUENCE_DIVERGENCE_CSV
)
from data_loader import (
    load_vcf_with_cache,
    load_population_mapping,
    map_samples_to_populations,
    print_population_summary
)
from population_genetics import (
    calculate_heterozygosity_observed,
    calculate_nucleotide_diversity_per_population,
    calculate_pairwise_fst,
    calculate_sequence_diversity_per_population
)


def load_or_calculate_pi(pi_csv_file, genotypes, sample_superpops, superpops):
    """Load nucleotide diversity from CSV if exists, otherwise calculate."""
    if os.path.exists(pi_csv_file):
        pi_df = pd.read_csv(pi_csv_file)
        pi_results = {}
        for _, row in pi_df.iterrows():
            pi_results[row["Superpopulation"]] = row["Nucleotide_Diversity_pi"]
        print(f"Loaded nucleotide diversity from {pi_csv_file}")
    else:
        pi_results = calculate_nucleotide_diversity_per_population(
            genotypes, sample_superpops, superpops
        )
        pi_df = pd.DataFrame(
            [{"Superpopulation": pop, "Nucleotide_Diversity_pi": pi.item() if hasattr(pi, "item") else pi}
             for pop, pi in pi_results.items()]
        )
        pi_df.to_csv(pi_csv_file, index=False)
        print(f"Calculated and saved nucleotide diversity to {pi_csv_file}")

    return pi_results


def load_or_calculate_fst(fst_csv_file, genotypes, pop_indices, superpops):
    """Load FST from CSV if exists, otherwise calculate."""
    if os.path.exists(fst_csv_file):
        fst_df = pd.read_csv(fst_csv_file)
        fst_results = {}
        for _, row in fst_df.iterrows():
            fst_results[(row["Pop1"], row["Pop2"])] = row["Pairwise_FST"]
        print(f"Loaded FST from {fst_csv_file}")
    else:
        fst_results = calculate_pairwise_fst(genotypes, pop_indices, superpops)
        fst_rows = [{"Pop1": pop1, "Pop2": pop2, "Pairwise_FST": fst}
                    for (pop1, pop2), fst in fst_results.items()]
        fst_df = pd.DataFrame(fst_rows)
        fst_df.to_csv(fst_csv_file, index=False)
        print(f"Calculated and saved FST to {fst_csv_file}")

    return fst_results


def load_or_calculate_diversity(div_csv_file, genotypes, pos, sample_superpops, superpops):
    """Load sequence diversity from CSV if exists, otherwise calculate."""
    if os.path.exists(div_csv_file):
        div_df = pd.read_csv(div_csv_file)
        sequence_div = {}
        for _, row in div_df.iterrows():
            sequence_div[row["Superpopulation"]] = row["Sequence_Diversity"]
        print(f"Loaded sequence diversity from {div_csv_file}")
    else:
        sequence_div = calculate_sequence_diversity_per_population(
            genotypes, pos, sample_superpops, superpops
        )
        div_rows = [{"Superpopulation": pop, "Sequence_Diversity": div}
                    for pop, div in sequence_div.items()]
        div_df = pd.DataFrame(div_rows)
        div_df.to_csv(div_csv_file, index=False)
        print(f"Calculated and saved sequence diversity to {div_csv_file}")

    return sequence_div


def main():
    start_time = time.time()

    # Load VCF data
    callset = load_vcf_with_cache(VCF_FILE, CACHE_FILE)

    # Extract key data
    genotypes = allel.GenotypeArray(callset["calldata/GT"])
    samples = callset["samples"]
    pos = callset["variants/POS"]

    # Calculate overall heterozygosity
    mean_ho = calculate_heterozygosity_observed(genotypes)
    print(f"Mean heterozygosity observed: {mean_ho}")

    # Load population mapping
    panel = load_population_mapping(PANEL_FILE)
    sample_superpops, superpops, pop_indices = map_samples_to_populations(samples, panel)
    print_population_summary(sample_superpops, superpops)

    # Calculate or load population genetics metrics
    pi_results = load_or_calculate_pi(
        NUCLEOTIDE_DIVERSITY_CSV, genotypes, sample_superpops, superpops
    )

    fst_results = load_or_calculate_fst(
        PAIRWISE_FST_CSV, genotypes, pop_indices, superpops
    )

    sequence_div = load_or_calculate_diversity(
        SEQUENCE_DIVERGENCE_CSV, genotypes, pos, sample_superpops, superpops
    )

    # Print results
    print("\n" + "="*100)
    print("Mean Nucleotide diversity per superpopulation")
    for pop, pi in pi_results.items():
        pi_value = pi.item() if hasattr(pi, "item") else pi
        print(f"{pop}: π = {pi_value:.4f}")

    print("="*100)
    print("Pairwise FST results")
    for (pop1, pop2), fst in fst_results.items():
        print(f"Pairwise FST between {pop1} and {pop2}: {fst}")

    print("="*100)
    print("Sequence diversity per superpopulation")
    for pop, div in sequence_div.items():
        print(f"{pop}: {div}")

    time_diff = time.time() - start_time
    print(f"\nTotal time: {time_diff:.2f} seconds")


if __name__ == "__main__":
    main()
