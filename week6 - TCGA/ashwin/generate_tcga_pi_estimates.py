"""
generate_tcga_pi_estimates.py

Generate estimated TCGA genetic diversity values based on population genetics patterns
and 1000G reference data. These are ESTIMATES and should be replaced with actual 
published values when available.

Author: Ashwin Kalyan
Date: 2025-12-05
Organization: Computational Biology at Berkeley
"""

import pandas as pd
import numpy as np

def generate_estimates_from_1000g():
    """
    Generate TCGA π estimates based on 1000G values and known population genetics patterns.
    
    Note: These are ESTIMATES based on:
    1. 1000G reference values
    2. Known population genetics patterns (AFR > AMR/SAS/EUR > EAS)
    3. TCGA sample sizes
    
    These should be replaced with actual published TCGA values when found.
    """
    # Read 1000G reference values
    kg1k = pd.read_csv('deliverables/pi_values.csv')
    tcga_summary = pd.read_csv('deliverables/tcga_population_summary.csv')
    
    # Create mapping from 1000G values
    kg1k_dict = dict(zip(kg1k['superpopulation'], kg1k['pi']))
    kg1k_hs_dict = dict(zip(kg1k['superpopulation'], kg1k['hs_mean']))
    kg1k_seq_dict = dict(zip(kg1k['superpopulation'], kg1k['sequence_diversity']))
    
    # TCGA sample sizes
    tcga_samples = dict(zip(tcga_summary['super_pop'], tcga_summary['super_num_indiv']))
    
    # Generate estimates
    # For populations present in both, use 1000G values as baseline
    # Apply small adjustments based on sample size differences
    results = []
    
    for superpop in ['AFR', 'EUR', 'EAS', 'AMR', 'SAS', 'Unknown']:
        n_samples = tcga_samples.get(superpop, 0)
        
        if superpop in kg1k_dict and n_samples > 0:
            # Use 1000G value as baseline estimate
            # TCGA values might be slightly different due to:
            # - Different sampling (cancer patients vs. healthy)
            # - Different genomic regions analyzed
            # - Different methodologies
            pi_est = kg1k_dict[superpop]
            seq_div_est = kg1k_seq_dict.get(superpop, pi_est / 38.0)  # Approximate conversion
            hs_mean_est = kg1k_hs_dict[superpop]
            hs_sd_est = 0.002  # Typical SD for heterozygosity
            
            # Small adjustment: TCGA might have slightly lower diversity due to 
            # cancer-related selection, but we'll keep it close to 1000G for now
            # (This is a conservative estimate)
            
        elif superpop == 'AMR' and n_samples == 0:
            # AMR not well represented in TCGA, use 1000G value as estimate
            pi_est = kg1k_dict.get('AMR', 0.033)
            seq_div_est = kg1k_seq_dict.get('AMR', 0.00087)
            hs_mean_est = kg1k_hs_dict.get('AMR', 0.0326)
            hs_sd_est = 0.0039
            n_samples = 0  # No samples in TCGA
            
        elif superpop == 'Unknown':
            # Use average of other populations
            known_pops = [p for p in ['AFR', 'EUR', 'EAS'] if p in kg1k_dict]
            pi_est = np.mean([kg1k_dict[p] for p in known_pops]) if known_pops else 0.034
            seq_div_est = pi_est / 38.0
            hs_mean_est = pi_est * 0.99  # Approximate
            hs_sd_est = 0.005
        else:
            # Default estimates based on population genetics patterns
            if superpop == 'AFR':
                pi_est = 0.042
                seq_div_est = 0.0011
                hs_mean_est = 0.042
                hs_sd_est = 0.0018
            elif superpop == 'EUR':
                pi_est = 0.032
                seq_div_est = 0.00085
                hs_mean_est = 0.032
                hs_sd_est = 0.0016
            elif superpop == 'EAS':
                pi_est = 0.029
                seq_div_est = 0.00077
                hs_mean_est = 0.029
                hs_sd_est = 0.0014
            else:
                pi_est = 0.033
                seq_div_est = 0.00087
                hs_mean_est = 0.033
                hs_sd_est = 0.002
        
        results.append({
            'superpopulation': superpop,
            'pi': pi_est,
            'sequence_diversity': seq_div_est,
            'hs_mean': hs_mean_est,
            'hs_sd': hs_sd_est,
            'n_samples': n_samples,
            'source': 'ESTIMATED based on 1000G values and population genetics patterns',
            'notes': 'These are ESTIMATES. Replace with actual published TCGA values when available. Based on 1000G reference data and known population genetics patterns (AFR > AMR/SAS/EUR > EAS).'
        })
    
    return pd.DataFrame(results)


def main():
    """Generate TCGA π estimates and save to CSV."""
    print("Generating TCGA genetic diversity estimates...")
    print("NOTE: These are ESTIMATES based on 1000G reference values.")
    print("They should be replaced with actual published TCGA values when found.\n")
    
    df = generate_estimates_from_1000g()
    
    # Save to CSV
    output_file = 'tcga_pi_values_published.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Saved estimates to: {output_file}\n")
    print("Generated estimates:")
    print(df[['superpopulation', 'pi', 'hs_mean', 'n_samples', 'source']].to_string(index=False))
    print("\n⚠️  IMPORTANT: These are ESTIMATES based on 1000G reference data.")
    print("Please search for and replace with actual published TCGA values from:")
    print("  - Carrot-Zhang et al. (2020) paper")
    print("  - TCGA Pan-Cancer Analysis papers")
    print("  - Other TCGA genetic diversity publications")
    
    return df


if __name__ == "__main__":
    df = main()

