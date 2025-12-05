"""
parse_tcga_metadata.py

Script to parse TCGA metadata and generate population summary with ancestry counts and percentages.

Author: Ashwin Kalyan
Date: 2025-12-05
Organization: Computational Biology at Berkeley

This script:
1. Loads TCGA metadata (from GDC portal or local file)
2. Extracts ancestry/population information
3. Computes counts and percentages by population and superpopulation
4. Saves results to tcga_population_summary.csv

TCGA metadata typically contains ancestry information in fields like:
- 'ancestry' or 'race' or 'ethnicity' or 'demographics.race'
- May need to map to standard superpopulations (AFR, AMR, EAS, EUR, SAS)
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple


def load_tcga_metadata(metadata_file: str) -> pd.DataFrame:
    """
    Load TCGA metadata file.
    
    Supports various formats:
    - TSV files from GDC portal
    - CSV files
    - JSON files
    - Excel files (.xlsx, .xls)
    
    Args:
        metadata_file: Path to TCGA metadata file
        
    Returns:
        DataFrame with TCGA metadata
    """
    # Try different file formats
    if metadata_file.endswith('.xlsx') or metadata_file.endswith('.xls'):
        df = pd.read_excel(metadata_file)
    elif metadata_file.endswith('.tsv'):
        df = pd.read_csv(metadata_file, sep='\t')
    elif metadata_file.endswith('.csv'):
        df = pd.read_csv(metadata_file)
    elif metadata_file.endswith('.json'):
        df = pd.read_json(metadata_file)
    else:
        # Try Excel first, then TSV, then CSV
        try:
            df = pd.read_excel(metadata_file)
        except:
            try:
                df = pd.read_csv(metadata_file, sep='\t')
            except:
                df = pd.read_csv(metadata_file)
    
    return df


def map_tcga_ancestry_to_superpop(ancestry_value: str) -> str:
    """
    Map TCGA ancestry/race/ethnicity values to standard superpopulations.
    
    TCGA typically uses self-reported race/ethnicity which needs to be mapped
    to genetic ancestry superpopulations (AFR, AMR, EAS, EUR, SAS).
    
    Args:
        ancestry_value: Raw ancestry/race/ethnicity value from TCGA
        
    Returns:
        Standard superpopulation code
    """
    if pd.isna(ancestry_value):
        return "Unknown"
    
    ancestry_str = str(ancestry_value).upper().strip()
    
    # Handle TCGA-specific values first
    if '[NOT AVAILABLE]' in ancestry_str or '[NOT EVALUATED]' in ancestry_str:
        return "Unknown"
    
    # Mapping based on common TCGA race/ethnicity categories
    # African American / Black -> AFR
    if any(term in ancestry_str for term in ['AFRICAN', 'BLACK', 'AFR']):
        return "AFR"
    
    # Asian -> EAS (TCGA typically has East Asian, defaulting to EAS)
    if any(term in ancestry_str for term in ['ASIAN', 'EAS', 'EAST ASIAN']):
        return "EAS"
    if any(term in ancestry_str for term in ['SOUTH ASIAN', 'SAS', 'INDIAN', 'PAKISTANI']):
        return "SAS"
    
    # White / European -> EUR
    if any(term in ancestry_str for term in ['WHITE', 'EUROPEAN', 'EUR', 'CAUCASIAN']):
        return "EUR"
    
    # Native American / Alaska Native / Pacific Islander -> AMR or EAS
    if any(term in ancestry_str for term in ['NATIVE AMERICAN', 'ALASKA NATIVE', 'AMERICAN INDIAN']):
        return "AMR"
    if any(term in ancestry_str for term in ['PACIFIC ISLANDER', 'HAWAIIAN']):
        return "EAS"  # Pacific Islanders often have East Asian ancestry
    
    # Hispanic / Latino -> AMR
    if any(term in ancestry_str for term in ['HISPANIC', 'LATINO', 'AMR']):
        return "AMR"
    
    # Unknown or other
    if any(term in ancestry_str for term in ['UNKNOWN', 'NOT REPORTED', 'OTHER', 'N/A', '']):
        return "Unknown"
    
    return "Unknown"


def extract_ancestry_column(df: pd.DataFrame) -> pd.Series:
    """
    Extract ancestry information from TCGA metadata.
    
    Tries multiple possible column names that might contain ancestry info.
    
    Args:
        df: TCGA metadata DataFrame
        
    Returns:
        Series with ancestry values
    """
    # Common column names for ancestry in TCGA metadata
    possible_columns = [
        'ancestry',
        'race',
        'ethnicity',
        'demographics.race',
        'demographics.ethnicity',
        'demographic.race',
        'demographic.ethnicity',
        'patient.race',
        'patient.ethnicity',
        'Race',
        'Ethnicity',
        'Ancestry',
        'RACE',
        'ETHNICITY',
        'ANCESTRY'
    ]
    
    for col in possible_columns:
        if col in df.columns:
            return df[col]
    
    # If no direct match, try to find columns containing these keywords
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['race', 'ethnicity', 'ancestry', 'demographic']):
            return df[col]
    
    raise ValueError(f"Could not find ancestry/race/ethnicity column in TCGA metadata. Available columns: {list(df.columns)}")


def create_population_summary(df: pd.DataFrame, ancestry_col: str = None) -> pd.DataFrame:
    """
    Create population summary with counts and percentages.
    
    Args:
        df: TCGA metadata DataFrame
        ancestry_col: Name of ancestry column (if None, will try to detect)
        
    Returns:
        DataFrame with population summary matching population_summary.csv format
    """
    # Extract ancestry information
    if ancestry_col is None:
        ancestry_series = extract_ancestry_column(df)
    else:
        ancestry_series = df[ancestry_col]
    
    # Map to superpopulations
    superpops = ancestry_series.apply(map_tcga_ancestry_to_superpop)
    
    # For TCGA, we typically don't have sub-populations like 1000G,
    # so we'll use the superpopulation as both population and super_pop
    # If you have more detailed ancestry info, you can modify this
    
    # Count by superpopulation
    superpop_counts = Counter(superpops)
    total_samples = len(superpops)
    
    # Create results list
    results = []
    row_num = 1
    
    # Group by superpopulation
    for superpop in sorted(superpop_counts.keys()):
        count = superpop_counts[superpop]
        percent = count / total_samples if total_samples > 0 else 0
        
        # For TCGA, population and super_pop are the same unless we have more detail
        results.append({
            'Population': superpop,  # Using superpop as population for TCGA
            'num_indiv': count,
            'percent': percent,
            'super_pop': superpop,
            'super_num_indiv': count,
            'super_percent': percent
        })
        row_num += 1
    
    # Create DataFrame
    summary_df = pd.DataFrame(results)
    
    # Reorder columns to match population_summary.csv format
    summary_df = summary_df[[
        'Population', 'num_indiv', 'percent',
        'super_pop', 'super_num_indiv', 'super_percent'
    ]]
    
    return summary_df


def main(metadata_file: str = None, output_file: str = 'tcga_population_summary.csv'):
    """
    Main execution function.
    
    Args:
        metadata_file: Path to TCGA metadata file (if None, will prompt or use default)
        output_file: Output CSV file path
    """
    if metadata_file is None:
        # Try common TCGA metadata file names
        import os
        possible_files = [
            'tcga_metadata.tsv',
            'tcga_metadata.csv',
            'TCGA_metadata.tsv',
            'TCGA_metadata.csv',
            'gdc_sample_sheet.tsv',
            'gdc_metadata.tsv'
        ]
        
        metadata_file = None
        for f in possible_files:
            if os.path.exists(f):
                metadata_file = f
                break
        
        if metadata_file is None:
            raise FileNotFoundError(
                "TCGA metadata file not found. Please provide path to metadata file.\n"
                "Expected formats: TSV, CSV, or JSON from GDC portal.\n"
                "File should contain ancestry/race/ethnicity information."
            )
    
    print(f"Loading TCGA metadata from: {metadata_file}")
    df = load_tcga_metadata(metadata_file)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    print("\nCreating population summary...")
    summary_df = create_population_summary(df)
    
    print("\nPopulation Summary:")
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved population summary to: {output_file}")
    
    return summary_df


if __name__ == "__main__":
    import sys
    
    metadata_file = sys.argv[1] if len(sys.argv) > 1 else None
    summary = main(metadata_file)
    print("\nDone!")

