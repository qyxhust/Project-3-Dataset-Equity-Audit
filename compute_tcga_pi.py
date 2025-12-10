"""
compute_tcga_pi.py

Script to compute π (nucleotide diversity) for TCGA per ancestry group.

Author: Ashwin Kalyan
Date: 2025-12-05
Organization: Computational Biology at Berkeley

This script:
1. Loads TCGA VCF data (or genotype data)
2. Loads TCGA population/ancestry mapping
3. Computes π (nucleotide diversity) per ancestry group
4. Computes sequence diversity and heterozygosity statistics
5. Saves results to tcga_pi_values.csv
"""

import allel
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Import from existing modules
from data_loader import (
    load_vcf_with_cache,
    load_population_mapping,
    map_samples_to_populations
)
from population_genetics import (
    calculate_nucleotide_diversity_per_population,
    calculate_sequence_diversity_per_population
)


def load_tcga_vcf(vcf_file: str, cache_file: str = None) -> dict:
    """
    Load TCGA VCF data.
    
    Args:
        vcf_file: Path to TCGA VCF file
        cache_file: Optional path to cache file
        
    Returns:
        callset dictionary from allel.read_vcf
    """
    if cache_file and Path(cache_file).exists():
        print("Loading from cache...")
        import pickle
        with open(cache_file, "rb") as f:
            callset = pickle.load(f)
        print("Cache loaded!")
    else:
        print("Reading TCGA VCF... this might take a minute")
        callset = allel.read_vcf(vcf_file, fields=["variants/*", "calldata/GT", "samples"])
        if cache_file:
            print("Saving to cache...")
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(callset, f)
            print("Cache saved!")
    
    return callset


def load_tcga_ancestry_mapping(
    metadata_file: str,
    sample_id_col: str = None,
    ancestry_col: str = None
) -> pd.DataFrame:
    """
    Load TCGA ancestry mapping from metadata.
    
    Creates a panel-like DataFrame mapping sample IDs to superpopulations.
    
    Args:
        metadata_file: Path to TCGA metadata file
        sample_id_col: Name of sample ID column (if None, will try to detect)
        ancestry_col: Name of ancestry column (if None, will try to detect)
        
    Returns:
        DataFrame with columns: sample, super_pop
    """
    # Load metadata
    if metadata_file.endswith('.tsv'):
        df = pd.read_csv(metadata_file, sep='\t')
    elif metadata_file.endswith('.csv'):
        df = pd.read_csv(metadata_file)
    elif metadata_file.endswith('.json'):
        df = pd.read_json(metadata_file)
    else:
        try:
            df = pd.read_csv(metadata_file, sep='\t')
        except:
            df = pd.read_csv(metadata_file)
    
    # Detect sample ID column
    if sample_id_col is None:
        possible_sample_cols = [
            'sample', 'Sample', 'SAMPLE', 'sample_id', 'Sample ID',
            'barcode', 'Barcode', 'BARCODE', 'case_id', 'Case ID',
            'submitter_id', 'Submitter ID', 'patient_id', 'Patient ID'
        ]
        for col in possible_sample_cols:
            if col in df.columns:
                sample_id_col = col
                break
        
        if sample_id_col is None:
            # Try to find column with 'sample' or 'id' in name
            for col in df.columns:
                if 'sample' in col.lower() or 'id' in col.lower():
                    sample_id_col = col
                    break
        
        if sample_id_col is None:
            raise ValueError(f"Could not find sample ID column. Available columns: {list(df.columns)}")
    
    # Detect ancestry column (reuse function from parse_tcga_metadata)
    if ancestry_col is None:
        possible_columns = [
            'ancestry', 'race', 'ethnicity', 'demographics.race',
            'demographics.ethnicity', 'demographic.race', 'demographic.ethnicity',
            'patient.race', 'patient.ethnicity', 'Race', 'Ethnicity',
            'Ancestry', 'RACE', 'ETHNICITY', 'ANCESTRY'
        ]
        
        for col in possible_columns:
            if col in df.columns:
                ancestry_col = col
                break
        
        if ancestry_col is None:
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['race', 'ethnicity', 'ancestry', 'demographic']):
                    ancestry_col = col
                    break
        
        if ancestry_col is None:
            raise ValueError(f"Could not find ancestry column. Available columns: {list(df.columns)}")
    
    # Map ancestry to superpopulations (reuse function from parse_tcga_metadata)
    def map_to_superpop(ancestry_value):
        if pd.isna(ancestry_value):
            return "Unknown"
        ancestry_str = str(ancestry_value).upper().strip()
        if any(term in ancestry_str for term in ['AFRICAN', 'BLACK', 'AFR']):
            return "AFR"
        if any(term in ancestry_str for term in ['ASIAN', 'EAS', 'EAST ASIAN']):
            return "EAS"
        if any(term in ancestry_str for term in ['SOUTH ASIAN', 'SAS', 'INDIAN', 'PAKISTANI']):
            return "SAS"
        if any(term in ancestry_str for term in ['WHITE', 'EUROPEAN', 'EUR', 'CAUCASIAN']):
            return "EUR"
        if any(term in ancestry_str for term in ['HISPANIC', 'LATINO', 'AMR', 'AMERICAN', 'NATIVE AMERICAN']):
            return "AMR"
        if any(term in ancestry_str for term in ['UNKNOWN', 'NOT REPORTED', 'OTHER', 'N/A', '']):
            return "Unknown"
        return "Unknown"
    
    # Create mapping DataFrame
    mapping_df = pd.DataFrame({
        'sample': df[sample_id_col],
        'super_pop': df[ancestry_col].apply(map_to_superpop)
    })
    
    return mapping_df


def calculate_heterozygosity_per_sample(genotypes: allel.GenotypeArray) -> np.ndarray:
    """
    Calculate observed heterozygosity for each sample across all variants.
    
    Args:
        genotypes: GenotypeArray of shape (n_variants, n_samples, ploidy)
    
    Returns:
        Array of heterozygosity values per sample
    """
    is_het = genotypes.is_het()
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
        
        if len(pop_hs) > 0:
            results.append({
                'superpopulation': pop,
                'hs_mean': np.mean(pop_hs),
                'hs_sd': np.std(pop_hs, ddof=1),
                'n_samples': len(pop_hs)
            })
        else:
            results.append({
                'superpopulation': pop,
                'hs_mean': np.nan,
                'hs_sd': np.nan,
                'n_samples': 0
            })
    
    return pd.DataFrame(results)


def map_tcga_samples_to_populations(
    samples: np.ndarray,
    mapping_df: pd.DataFrame
) -> Tuple[List[str], List[str], Dict[str, np.ndarray]]:
    """
    Map TCGA VCF samples to superpopulations using mapping DataFrame.
    
    Args:
        samples: Array of sample IDs from VCF
        mapping_df: DataFrame with sample to superpopulation mapping
        
    Returns:
        Tuple of (sample_superpops list, sorted superpops list, pop_indices dict)
    """
    # Create sample ID to superpopulation mapping
    sample_to_superpop = dict(zip(mapping_df["sample"], mapping_df["super_pop"]))
    
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


def main(
    vcf_file: str = None,
    metadata_file: str = None,
    cache_file: str = 'tcga_callset_cache.pkl',
    output_file: str = 'tcga_pi_values.csv'
):
    """
    Main execution function.
    
    Args:
        vcf_file: Path to TCGA VCF file
        metadata_file: Path to TCGA metadata file with ancestry information
        cache_file: Path to cache file for VCF data
        output_file: Output CSV file path
    """
    if vcf_file is None:
        raise ValueError("Please provide path to TCGA VCF file")
    
    if metadata_file is None:
        # Try to use existing tcga_population_summary.csv to get mapping
        # Or use a default metadata file
        possible_files = [
            'tcga_metadata.tsv',
            'tcga_metadata.csv',
            'TCGA_metadata.tsv',
            'TCGA_metadata.csv',
            'gdc_sample_sheet.tsv'
        ]
        
        metadata_file = None
        for f in possible_files:
            if Path(f).exists():
                metadata_file = f
                break
        
        if metadata_file is None:
            raise ValueError(
                "Please provide path to TCGA metadata file with ancestry information.\n"
                "Alternatively, ensure tcga_population_summary.csv exists and contains sample mappings."
            )
    
    print("Step 1: Loading TCGA VCF data...")
    callset = load_tcga_vcf(vcf_file, cache_file)
    
    # Extract genotype and position data
    genotypes = allel.GenotypeArray(callset["calldata/GT"])
    samples = callset["samples"]
    pos = callset["variants/POS"]
    
    print(f"Loaded {len(samples)} samples and {len(pos)} variants")
    
    print("\nStep 2: Loading TCGA ancestry mapping...")
    mapping_df = load_tcga_ancestry_mapping(metadata_file)
    sample_superpops, superpops, pop_indices = map_tcga_samples_to_populations(samples, mapping_df)
    
    print(f"Superpopulations found: {superpops}")
    for pop in superpops:
        count = sample_superpops.count(pop)
        print(f"  {pop}: {count} samples")
    
    print("\nStep 3: Calculating π (nucleotide diversity) per population...")
    pi_results = calculate_nucleotide_diversity_per_population(
        genotypes, sample_superpops, superpops
    )
    
    print("\nStep 4: Calculating sequence diversity per population...")
    seq_div_results = calculate_sequence_diversity_per_population(
        genotypes, pos, sample_superpops, superpops
    )
    
    print("\nStep 5: Calculating heterozygosity per sample...")
    hs_per_sample = calculate_heterozygosity_per_sample(genotypes)
    hs_summary = summarize_hs_by_population(hs_per_sample, sample_superpops, superpops)
    
    print("\nStep 6: Combining results...")
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
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")
    
    print("\nResults:")
    print(results_df.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    import sys
    
    vcf_file = sys.argv[1] if len(sys.argv) > 1 else None
    metadata_file = sys.argv[2] if len(sys.argv) > 2 else None
    output_file = sys.argv[3] if len(sys.argv) > 3 else 'tcga_pi_values.csv'
    
    results = main(vcf_file, metadata_file, output_file=output_file)
    print("\nDone!")

