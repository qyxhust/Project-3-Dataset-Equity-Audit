"""
Author: Ashwin Kalyan
Date: 2025-10-20
Organization: Computational Biology @ Berkeley

This file automates the computation of:
    - Population counts and percentages by ancestry group and superpopulation
    - Missing analysis for demographic fields
    - Shannon Diversity Index
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# files
INPUT_METADATA_FILE = "population_summary.csv"
OUTPUT_DIR = Path.home() / "Downloads"
POPULATION_SUMMARY_FILE = OUTPUT_DIR / "metrics_summary.csv"
MISSINGNESS_REPORT_FILE = OUTPUT_DIR / "missingness_report.csv"

# Column name mappings 
COLUMN_MAPPINGS = {
    'population': ['population', 'pop', 'Population', 'Pop'],
    'superpopulation': ['superpopulation', 'super_pop', 'Superpopulation', 'Super_pop'],
    'sex': ['sex', 'Sex', 'gender', 'Gender'],
    'age': ['age', 'Age'],
    'geography': ['geography', 'Geography', 'location', 'Location', 'region', 'Region']
}

SUPERPOP_MAP = {
    # East Asian (EAS)
    'CHB': 'EAS', 'JPT': 'EAS', 'CHS': 'EAS', 'CDX': 'EAS', 'KHV': 'EAS',
    # European (EUR)
    'CEU': 'EUR', 'TSI': 'EUR', 'GBR': 'EUR', 'FIN': 'EUR', 'IBS': 'EUR',
    # African (AFR)
    'YRI': 'AFR', 'LWK': 'AFR', 'GWD': 'AFR', 'MSL': 'AFR', 'ESN': 'AFR',
    # Admixed American (AMR)
    'ASW': 'AMR', 'ACB': 'AMR', 'MXL': 'AMR', 'PUR': 'AMR', 'CLM': 'AMR', 'PEL': 'AMR',
    # South Asian (SAS)
    'GIH': 'SAS', 'PJL': 'SAS', 'BEB': 'SAS', 'STU': 'SAS', 'ITU': 'SAS'
}

def standardize_column_names(df):
    """
    Standardize the column names based on COLUMN_MAPPINGS
    Returns df with standardized names and a mapping dictionary
    """
    col_map = {}

    for standard_name, variants in COLUMN_MAPPINGS.items():
        for col in df.columns:
            if col in variants:
                col_map[col] = standard_name
                break
    
    df_standardized = df.rename(columns=col_map)
    print(f"Standardized columns: {col_map}")
    return df_standardized, col_map

def load_metadata(file_path):
    """Load Bren's data"""

    print(f"\nLoading metadata from: {file_path}")

    for sep in [',', '\t', '|']:
        try:
            df = pd.read_csv(file_path, sep=sep, low_memory=False)

            if df.shape[1] > 1:
                print(f"Successfully loaded with separator: {repr(sep)}")
                print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
                break

        except Exception as e:
            continue

    else:
        raise ValueError("Could not load file with any common separator")
    
    # Standardize column names
    df, _ = standardize_column_names(df)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    return df

def compute_population_summary(df):
    """Compute population summary statistics"""

    # Add superpopulation if not present
    if 'superpopulation' not in df.columns and 'population' in df.columns:
        df['superpopulation'] = df['population'].map(SUPERPOP_MAP)
        print("Added superpopulation mapping")
    
    # Count by population (ancestry group)
    pop_counts = df['population'].value_counts().reset_index()
    pop_counts.columns = ['population', 'num_indiv']
    total_indiv = pop_counts['num_indiv'].sum()
    pop_counts['percent'] = (pop_counts['num_indiv'] / total_indiv * 100).round(2)
    
    # Count by superpopulation
    super_counts = df['superpopulation'].value_counts().reset_index()
    super_counts.columns = ['super_pop', 'super_num_indiv']
    super_counts['super_percent'] = (super_counts['super_num_indiv'] / total_indiv * 100).round(2)
    
    # Merge population and superpopulation data
    pop_counts['super_pop'] = pop_counts['population'].map(SUPERPOP_MAP)
    summary = pop_counts.merge(super_counts, on='super_pop', how='left')
    
    # Reorder columns for clarity
    summary = summary[['population', 'num_indiv', 'percent', 
                       'super_pop', 'super_num_indiv', 'super_percent']]
    
    print(f"\nTotal individuals: {total_indiv}")
    print(f"Number of populations: {len(pop_counts)}")
    print(f"Number of superpopulations: {len(super_counts)}")
    
    return summary

def compute_missingness(df):
    """Compute missing values for demographic fields"""

    demographic_fields = ['sex', 'age', 'geography']
    available_fields = [f for f in demographic_fields if f in df.columns]
    
    if not available_fields:
        print("WARNING: No demographic fields found in data")
        return pd.DataFrame()
    
    missingness_data = []
    total_records = len(df)
    
    for field in available_fields:
        missing_count = df[field].isna().sum()
        missing_percent = (missing_count / total_records * 100).round(2)
        present_count = total_records - missing_count
        
        missingness_data.append({
            'field': field,
            'total_records': total_records,
            'present_count': present_count,
            'missing_count': missing_count,
            'missing_percent': missing_percent
        })
        
        print(f"{field:12} - Missing: {missing_count:5} ({missing_percent:5.1f}%)")
    
    return pd.DataFrame(missingness_data)

def compute_shannon_diversity(df):
    """
    Calculate Shannon Diversity Index based on ancestry proportions (Cyan's task).
    
    H = -Σ(p_i * ln(p_i))
    where p_i is the proportion of individuals in group i
    """
    
    # Calculate proportions
    pop_counts = df['population'].value_counts()
    proportions = pop_counts / pop_counts.sum()
    
    # Shannon Index
    shannon_index = -np.sum(proportions * np.log(proportions))
    
    # Maximum possible diversity (if all groups were equal)
    max_diversity = np.log(len(proportions))
    
    # Evenness (how evenly distributed populations are)
    evenness = shannon_index / max_diversity
    
    print(f"\nShannon Diversity Index: {shannon_index:.4f}")
    print(f"Maximum possible diversity: {max_diversity:.4f}")
    print(f"Evenness: {evenness:.4f} (0=uneven, 1=perfectly even)")
    print(f"\nInterpretation:")
    print(f"  - Higher values indicate greater diversity")
    print(f"  - This dataset has {len(proportions)} distinct populations")
    print(f"  - Evenness of {evenness:.2%} means populations are {'relatively balanced' if evenness > 0.8 else 'somewhat imbalanced'}")
    
    return {
        'shannon_index': shannon_index,
        'max_diversity': max_diversity,
        'evenness': evenness,
        'num_populations': len(proportions)
    }

def save_missingness_report(missingness_df, shannon_dict, output_file):
    """
    Save missingness statistics and Shannon Index to CSV with interpretive notes.
    """
    print(f"\nSaving missingness report to: {output_file}")
    
    # Create report sections
    with open(output_file, 'w') as f:
        # Header
        f.write("# 1000 Genomes Missingness and Diversity Report\n")
        f.write("# Generated by: extract_metrics.py (Ashwin)\n\n")
        
        # Missingness section
        f.write("## MISSINGNESS ANALYSIS\n")
    
    # Append missingness data
    missingness_df.to_csv(output_file, mode='a', index=False)
    
    # Append Shannon diversity
    with open(output_file, 'a') as f:
        f.write("\n## SHANNON DIVERSITY INDEX\n")
        f.write("metric,value\n")
        f.write(f"shannon_index,{shannon_dict['shannon_index']:.4f}\n")
        f.write(f"max_diversity,{shannon_dict['max_diversity']:.4f}\n")
        f.write(f"evenness,{shannon_dict['evenness']:.4f}\n")
        f.write(f"num_populations,{shannon_dict['num_populations']}\n")
        
        f.write("\n## INTERPRETATION NOTES\n")
        f.write(f"# Shannon Index of {shannon_dict['shannon_index']:.4f} indicates ")
        if shannon_dict['shannon_index'] > 2.5:
            f.write("high genetic diversity across populations.\n")
        elif shannon_dict['shannon_index'] > 2.0:
            f.write("moderate-to-high diversity across populations.\n")
        else:
            f.write("moderate diversity with some dominant populations.\n")
        
        f.write(f"# Evenness of {shannon_dict['evenness']:.2%} suggests populations are ")
        if shannon_dict['evenness'] > 0.85:
            f.write("relatively balanced in sample size.\n")
        elif shannon_dict['evenness'] > 0.70:
            f.write("somewhat balanced but with noticeable size differences.\n")
        else:
            f.write("imbalanced, with some populations much larger than others.\n")

def main():
    """Main execution pipeline."""

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load data
    try:
        df = load_metadata(INPUT_METADATA_FILE)
    except FileNotFoundError:
        print(f"\nERROR: Input file '{INPUT_METADATA_FILE}' not found!")
        print("Please update INPUT_METADATA_FILE in the configuration section.")
        return
    except Exception as e:
        print(f"\nERROR loading data: {e}")
        return
    
    # Compute population summary (Bren's work)
    pop_summary = compute_population_summary(df)
    pop_summary.to_csv(POPULATION_SUMMARY_FILE, index=False)
    print(f"\n✓ Saved: {POPULATION_SUMMARY_FILE}")
    
    # Compute missingness (Cyan's work)
    missingness_df = compute_missingness(df)
    
    # Compute Shannon diversity (Cyan's work)
    shannon_dict = compute_shannon_diversity(df)
    
    # Save missingness report
    if not missingness_df.empty:
        save_missingness_report(missingness_df, shannon_dict, MISSINGNESS_REPORT_FILE)
        print(f"✓ Saved: {MISSINGNESS_REPORT_FILE}")
    
    # Summary statistics for final report
    print("\n" + "="*70)
    print("SUMMARY FOR FINAL REPORT")
    print("="*70)
    
    if 'super_pop' in pop_summary.columns:
        eur_pct = pop_summary[pop_summary['super_pop'] == 'EUR']['super_percent'].iloc[0]
        print(f"\n% European ancestry: {eur_pct:.1f}%")
        
        print("\nSuperpopulation distribution:")
        super_summary = pop_summary[['super_pop', 'super_percent']].drop_duplicates()
        for _, row in super_summary.iterrows():
            print(f"  {row['super_pop']}: {row['super_percent']:.1f}%")
        
        # Identify underrepresented groups (< 15%)
        underrep = super_summary[super_summary['super_percent'] < 15]
        if not underrep.empty:
            print(f"\nUnderrepresented groups (< 15%):")
            for _, row in underrep.iterrows():
                print(f"  {row['super_pop']}: {row['super_percent']:.1f}%")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nOutputs created:")
    print(f"  1. {POPULATION_SUMMARY_FILE}")
    print(f"  2. {MISSINGNESS_REPORT_FILE}")


if __name__ == "__main__":
    main()