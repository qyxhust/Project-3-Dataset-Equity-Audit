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
import logging
from typing import Dict, Tuple, Optional

# Configuration
CONFIG = {
    'input_file': Path.home() / "Downloads" / "population_summary.csv",
    'output_dir': Path.home() / "Downloads",
    'demographic_fields': ['sex', 'age', 'geography'],
    'underrepresentation_threshold': 10.0
}

# File paths
INPUT_METADATA_FILE = CONFIG['input_file']
OUTPUT_DIR = CONFIG['output_dir']
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def standardize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Standardize column names based on COLUMN_MAPPINGS."""
    col_map = {col: standard_name for standard_name, variants in COLUMN_MAPPINGS.items() 
               for col in df.columns if col in variants}
    
    df_standardized = df.rename(columns=col_map)
    logger.info(f"Standardized columns: {col_map}")
    return df_standardized, col_map

def load_metadata(file_path: str) -> pd.DataFrame:
    """Load and preprocess metadata file."""
    logger.info(f"Loading metadata from: {file_path}")

    for sep in [',', '\t', '|']:
        try:
            df = pd.read_csv(file_path, sep=sep, low_memory=False)
            if df.shape[1] > 1:
                logger.info(f"Successfully loaded with separator: {repr(sep)}")
                logger.info(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
                break
        except Exception:
            continue
    else:
        raise ValueError("Could not load file with any common separator")
    
    # Standardize column names and clean data
    df, _ = standardize_column_names(df)
    df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).apply(lambda x: x.str.strip())
    
    return df

def compute_population_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute population summary statistics."""
    # Add superpopulation if not present
    if 'superpopulation' not in df.columns and 'population' in df.columns:
        df = df.copy()
        df['superpopulation'] = df['population'].map(SUPERPOP_MAP)
        logger.info("Added superpopulation mapping")
    
    # Compute population counts and percentages
    pop_counts = df['population'].value_counts().reset_index()
    pop_counts.columns = ['population', 'num_indiv']
    total_indiv = pop_counts['num_indiv'].sum()
    pop_counts['percent'] = (pop_counts['num_indiv'] / total_indiv * 100).round(2)
    
    # Compute superpopulation counts
    super_counts = df['superpopulation'].value_counts().reset_index()
    super_counts.columns = ['super_pop', 'super_num_indiv']
    super_counts['super_percent'] = (super_counts['super_num_indiv'] / total_indiv * 100).round(2)
    
    # Merge and organize data
    pop_counts['super_pop'] = pop_counts['population'].map(SUPERPOP_MAP)
    summary = pop_counts.merge(super_counts, on='super_pop', how='left')
    summary = summary[['population', 'num_indiv', 'percent', 'super_pop', 'super_num_indiv', 'super_percent']]
    
    logger.info(f"Total individuals: {total_indiv:,}")
    logger.info(f"Number of populations: {len(pop_counts)}")
    logger.info(f"Number of superpopulations: {len(super_counts)}")
    
    return summary

def compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing values for demographic fields."""
    logger.info(f"Looking for demographic fields: {CONFIG['demographic_fields']}")
    logger.info(f"Available columns: {list(df.columns)}")
    
    available_fields = [f for f in CONFIG['demographic_fields'] if f in df.columns]
    logger.info(f"Found demographic fields: {available_fields}")
    
    if not available_fields:
        logger.warning("No demographic fields found in data")
        return pd.DataFrame()
    
    total_records = len(df)
    missingness_data = []
    
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
        
        logger.info(f"{field:12} - Missing: {missing_count:5,} ({missing_percent:5.1f}%)")
    
    return pd.DataFrame(missingness_data)

def compute_shannon_diversity(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate Shannon Diversity Index based on ancestry proportions."""
    # Calculate proportions
    pop_counts = df['population'].value_counts()
    proportions = pop_counts / pop_counts.sum()
    
    # Shannon Index: H = -Σ(p_i * ln(p_i))
    shannon_index = -np.sum(proportions * np.log(proportions))
    max_diversity = np.log(len(proportions))
    evenness = shannon_index / max_diversity
    
    logger.info(f"Shannon Diversity Index: {shannon_index:.4f}")
    logger.info(f"Maximum possible diversity: {max_diversity:.4f}")
    logger.info(f"Evenness: {evenness:.4f} (0=uneven, 1=perfectly even)")
    logger.info(f"Dataset has {len(proportions)} distinct populations")
    logger.info(f"Evenness of {evenness:.2%} means populations are {'relatively balanced' if evenness > 0.8 else 'somewhat imbalanced'}")
    
    return {
        'shannon_index': shannon_index,
        'max_diversity': max_diversity,
        'evenness': evenness,
        'num_populations': len(proportions)
    }

def save_missingness_report(missingness_df: pd.DataFrame, shannon_dict: Dict[str, float], output_file: Path) -> None:
    """Save missingness statistics and Shannon Index to CSV with interpretive notes."""
    logger.info(f"Saving missingness report to: {output_file}")
    
    with open(output_file, 'w') as f:
        # Header
        f.write("# 1000 Genomes Missingness and Diversity Report\n")
        f.write("# Generated by: extract_metrics.py (Ashwin)\n\n")
        f.write("## MISSINGNESS ANALYSIS\n")
        
        # Write missingness data as CSV
        f.write("field,total_records,present_count,missing_count,missing_percent\n")
        for _, row in missingness_df.iterrows():
            f.write(f"{row['field']},{row['total_records']},{row['present_count']},{row['missing_count']},{row['missing_percent']}\n")
        
        # Write Shannon diversity
        f.write("\n## SHANNON DIVERSITY INDEX\n")
        f.write("metric,value\n")
        for key, value in shannon_dict.items():
            f.write(f"{key},{value:.4f}\n")
        
        # Write interpretation notes
        f.write("\n## INTERPRETATION NOTES\n")
        si = shannon_dict['shannon_index']
        evenness = shannon_dict['evenness']
        
        diversity_level = "high" if si > 2.5 else "moderate-to-high" if si > 2.0 else "moderate"
        balance_level = "relatively balanced" if evenness > 0.85 else "somewhat balanced" if evenness > 0.70 else "imbalanced"
        
        f.write(f"# Shannon Index of {si:.4f} indicates {diversity_level} genetic diversity across populations.\n")
        f.write(f"# Evenness of {evenness:.2%} suggests populations are {balance_level} in sample size.\n")

def print_summary_report(pop_summary: pd.DataFrame) -> None:
    """Print professional summary report."""
    print("\n" + "="*70)
    print("📊 DATASET EQUITY AUDIT SUMMARY")
    print("="*70)
    
    if 'super_pop' in pop_summary.columns:
        # European ancestry percentage
        eur_data = pop_summary[pop_summary['super_pop'] == 'EUR']
        if not eur_data.empty and not pd.isna(eur_data['super_percent'].iloc[0]):
            eur_pct = eur_data['super_percent'].iloc[0]
            print(f"\n🇪🇺 European ancestry: {eur_pct:.1f}%")
        else:
            print(f"\n🇪🇺 European ancestry: N/A")
        
        # Superpopulation distribution
        print("\n📈 Superpopulation Distribution:")
        super_summary = pop_summary[['super_pop', 'super_percent']].drop_duplicates().sort_values('super_percent', ascending=False)
        for _, row in super_summary.iterrows():
            percent = row['super_percent']
            if pd.isna(percent):
                bar = ""
                percent_str = "N/A"
            else:
                bar = "█" * int(percent / 2)
                percent_str = f"{percent:5.1f}%"
            print(f"  {row['super_pop']:3}: {percent_str} {bar}")
        
        # Underrepresented groups
        underrep = super_summary[
            (super_summary['super_percent'] < CONFIG['underrepresentation_threshold']) & 
            (super_summary['super_percent'].notna())
        ]
        if not underrep.empty:
            print(f"\n⚠️  Underrepresented groups (< {CONFIG['underrepresentation_threshold']}%):")
            for _, row in underrep.iterrows():
                print(f"  • {row['super_pop']}: {row['super_percent']:.1f}%")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)

def main() -> None:
    """Main execution pipeline."""
    print("🚀 Starting Dataset Equity Audit Pipeline")
    print("="*50)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Load data
    try:
        df = load_metadata(INPUT_METADATA_FILE)
    except FileNotFoundError:
        logger.error(f"Input file '{INPUT_METADATA_FILE}' not found!")
        logger.error("Please update INPUT_METADATA_FILE in the configuration section.")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Process data
    logger.info("Computing population summary...")
    pop_summary = compute_population_summary(df)
    pop_summary.to_csv(POPULATION_SUMMARY_FILE, index=False)
    logger.info(f"✅ Saved: {POPULATION_SUMMARY_FILE}")
    
    logger.info("Computing missingness analysis...")
    missingness_df = compute_missingness(df)
    
    logger.info("Computing Shannon diversity...")
    shannon_dict = compute_shannon_diversity(df)
    
    # Save missingness report
    logger.info(f"Missingness dataframe shape: {missingness_df.shape}")
    logger.info(f"Missingness dataframe empty: {missingness_df.empty}")
    
    if not missingness_df.empty:
        save_missingness_report(missingness_df, shannon_dict, MISSINGNESS_REPORT_FILE)
        logger.info(f"✅ Saved: {MISSINGNESS_REPORT_FILE}")
    else:
        logger.warning("Missingness dataframe is empty, skipping report creation")
    
    # Print summary
    print_summary_report(pop_summary)
    
    print(f"\n📁 Outputs created:")
    print(f"   1. {POPULATION_SUMMARY_FILE}")
    print(f"   2. {MISSINGNESS_REPORT_FILE}")
    print("\n🎉 Analysis complete!")


if __name__ == "__main__":
    main()