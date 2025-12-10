"""
collect_tcga_published_data.py

Script to help collect and organize published TCGA genetic diversity statistics by ancestry group.

This script creates a structured CSV file from manually collected published data
about TCGA genetic diversity metrics (π, F_ST, heterozygosity, etc.) by ancestry.

Author: Ashwin Kalyan
Date: 2025-12-05
Organization: Computational Biology at Berkeley
"""

import pandas as pd
from typing import Dict, List, Optional


def create_tcga_pi_template() -> pd.DataFrame:
    """
    Create a template DataFrame for collecting TCGA π values by ancestry.
    
    Returns:
        DataFrame with columns matching pi_values.csv format
    """
    template = pd.DataFrame({
        'superpopulation': ['AFR', 'EUR', 'EAS', 'AMR', 'SAS', 'Unknown'],
        'pi': [None] * 6,
        'sequence_diversity': [None] * 6,
        'hs_mean': [None] * 6,
        'hs_sd': [None] * 6,
        'n_samples': [None] * 6,
        'source': [None] * 6,
        'notes': [None] * 6
    })
    return template


def add_published_data(
    df: pd.DataFrame,
    superpop: str,
    pi: Optional[float] = None,
    sequence_diversity: Optional[float] = None,
    hs_mean: Optional[float] = None,
    hs_sd: Optional[float] = None,
    n_samples: Optional[int] = None,
    source: str = "",
    notes: str = ""
) -> pd.DataFrame:
    """
    Add published data for a specific superpopulation.
    
    Args:
        df: Template DataFrame
        superpop: Superpopulation code (AFR, EUR, EAS, etc.)
        pi: Nucleotide diversity value
        sequence_diversity: Sequence diversity value
        hs_mean: Mean heterozygosity
        hs_sd: Standard deviation of heterozygosity
        n_samples: Number of samples
        source: Source paper/publication
        notes: Additional notes
        
    Returns:
        Updated DataFrame
    """
    mask = df['superpopulation'] == superpop
    if mask.any():
        if pi is not None:
            df.loc[mask, 'pi'] = pi
        if sequence_diversity is not None:
            df.loc[mask, 'sequence_diversity'] = sequence_diversity
        if hs_mean is not None:
            df.loc[mask, 'hs_mean'] = hs_mean
        if hs_sd is not None:
            df.loc[mask, 'hs_sd'] = hs_sd
        if n_samples is not None:
            df.loc[mask, 'n_samples'] = n_samples
        if source:
            df.loc[mask, 'source'] = source
        if notes:
            df.loc[mask, 'notes'] = notes
    
    return df


def main():
    """
    Main function to create and populate TCGA published data collection.
    
    Usage:
        1. Run this script to create template
        2. Manually add published data using add_published_data() or edit CSV directly
        3. Save as tcga_pi_values.csv (published data)
    """
    print("Creating TCGA published genetic diversity data collection template...")
    
    # Create template
    df = create_tcga_pi_template()
    
    # Example: Add data from a hypothetical paper
    # Uncomment and fill in when you find published data:
    """
    df = add_published_data(
        df, 
        superpop='AFR',
        pi=0.042,  # Example value - replace with actual published data
        hs_mean=0.041,
        n_samples=500,
        source="Carrot-Zhang et al. (2020)",
        notes="From Table X, chromosome 22 analysis"
    )
    """
    
    # Save template
    output_file = 'tcga_pi_values_published.csv'
    df.to_csv(output_file, index=False)
    print(f"\nTemplate saved to: {output_file}")
    print("\nInstructions:")
    print("1. Search for published TCGA genetic diversity papers")
    print("2. Extract π, F_ST, heterozygosity values by ancestry")
    print("3. Edit this CSV file or use add_published_data() function")
    print("4. Include source citations and notes")
    print("\nKey papers to check:")
    print("- Carrot-Zhang et al. (2020) - TCGA ancestry analysis")
    print("- TCGA Pan-Cancer Analysis papers")
    print("- TCGA Research Network publications")
    
    return df


if __name__ == "__main__":
    df = main()
    print("\nTemplate DataFrame:")
    print(df.to_string(index=False))

