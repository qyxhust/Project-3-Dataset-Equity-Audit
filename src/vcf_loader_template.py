"""
Author: Ashwin Kalyan
Date: 2025-10-30
Organization: Computational Biology at Berkeley

This file Map sample IDs to superpopulations using the panel file, computes allele counts and allele frequencies per superpopulation and saves the data to allele_freqs.csv
"""

import allel
import pandas as pd
import numpy as np

# Configuration
VCF_FILE = "ALL.chr22.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf"
PANEL_FILE = "integrated_call_samples_v3.20130502.ALL.panel"  # 1000 genomes panel ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel
OUTPUT_FILE = "allele_freqs.csv"

# Load VCF data for chromosome 22
callset = allel.read_vcf(VCF_FILE, fields=['variants/*', 'calldata/GT', 'samples'])

# Extract key data
genotypes = allel.GenotypeArray(callset['calldata/GT'])
samples = callset['samples']

# Access variant fields directly from callset
chrom = callset['variants/CHROM']
pos = callset['variants/POS']
ref = callset['variants/REF']
alt = callset['variants/ALT']
variant_id = callset['variants/ID'] if 'variants/ID' in callset else None

# Load panel file mapping samples to populations
panel = pd.read_csv(PANEL_FILE, sep='\t')
# Create sample ID to superpopulation mapping
sample_to_superpop = dict(zip(panel['sample'], panel['super_pop']))
# Map VCF samples to superpopulations
sample_superpops = [sample_to_superpop.get(s, 'Unknown') for s in samples]
superpops = sorted(set(sample_superpops))
print(f"\nSuperpopulations found: {superpops}")

# Count samples per superpopulation
for pop in superpops:
    count = sample_superpops.count(pop)
    print(f"  {pop}: {count} samples")

results = []

for variant_idx in range(len(chrom)):
    # Get variant information
    variant_chrom = chrom[variant_idx]
    variant_pos = pos[variant_idx]
    variant_ref = ref[variant_idx]
    variant_alt = alt[variant_idx, 0]  # Taking first alternate allele
    variant_variant_id = variant_id[variant_idx] if variant_id is not None else f"{variant_chrom}:{variant_pos}"
    
    # Get genotypes for this variant
    gt = genotypes[variant_idx]
    
    # Compute frequencies for each superpopulation
    variant_data = {
        'CHROM': variant_chrom,
        'POS': variant_pos,
        'ID': variant_variant_id,
        'REF': variant_ref,
        'ALT': variant_alt
    }
    
    for pop in superpops:
        # Get indices for samples in this superpopulation
        pop_indices = [i for i, sp in enumerate(sample_superpops) if sp == pop]
        
        # Extract genotypes for this population
        pop_gt = gt.take(pop_indices, axis=0)
        
        # Count alleles (0=ref, 1=alt, -1=missing)
        # For phased/diploid data, flatten to count all alleles
        allele_counts = pop_gt.flatten()
        
        # Count reference and alternate alleles (excluding missing)
        valid_alleles = allele_counts[allele_counts >= 0]
        n_alleles = len(valid_alleles)
        
        if n_alleles > 0:
            n_alt = np.sum(valid_alleles)  # Count alternate alleles
            n_ref = n_alleles - n_alt
            alt_freq = n_alt / n_alleles
        else:
            n_ref = 0
            n_alt = 0
            alt_freq = np.nan
        
        # Store counts and frequency
        variant_data[f'{pop}_REF_COUNT'] = n_ref
        variant_data[f'{pop}_ALT_COUNT'] = n_alt
        variant_data[f'{pop}_ALT_FREQ'] = alt_freq
    
    results.append(variant_data)

# Create DataFrame and save
df = pd.DataFrame(results)

# Reorder columns: variant info first, then pop-specific columns
variant_cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT']
pop_cols = [col for col in df.columns if col not in variant_cols]
df = df[variant_cols + sorted(pop_cols)]

# Save to CSV
df.to_csv(OUTPUT_FILE, index=False)
print(f"Total variants: {len(df)}")
print("\nFirst few rows:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
for pop in superpops:
    freq_col = f'{pop}_ALT_FREQ'
    if freq_col in df.columns:
        mean_freq = df[freq_col].mean()
        print(f"  {pop} mean alternate allele frequency: {mean_freq:.4f}")