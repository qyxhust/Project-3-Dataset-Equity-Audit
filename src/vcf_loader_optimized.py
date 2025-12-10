"""
Author: Ashwin Kalyan
Date: 2025-10-30
Organization: Computational Biology at Berkeley

This file Map sample IDs to superpopulations using the panel file, computes allele counts and allele frequencies per superpopulation and saves the data to allele_freqs.csv
"""

import allel
import pandas as pd
import numpy as np
import pickle
import os
import time
from itertools import combinations

start_time = time.time()

# Configuration
CACHE_FILE = "callset_cache.pkl"
VCF_FILE = "ALL.chr22.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf"
PANEL_FILE = "integrated_call_samples_v3.20130502.ALL.panel"  # 1000 genomes panel ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel
OUTPUT_FILE = "allele_freqs_optimized.csv"

# Load VCF data for chromosome 22 from cache if available
if os.path.exists(CACHE_FILE):
    print("Loading from cache...")
    with open(CACHE_FILE, "rb") as f:
        callset = pickle.load(f)
    print("Cache loaded!")

else: 
    print("Reading VCF... this might take a minute")
    callset = allel.read_vcf(VCF_FILE, fields=["variants/*", "calldata/GT", "samples"])
    print("Saving to cache...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(callset, f)
    print("Cache saved!")
# Extract key data
genotypes = allel.GenotypeArray(callset["calldata/GT"])
samples = callset["samples"]

# Access variant fields directly from callset
chrom = callset["variants/CHROM"]
pos = callset["variants/POS"]
ref = callset["variants/REF"]
alt = callset["variants/ALT"]
variant_id = callset["variants/ID"] if "variants/ID" in callset else None

# Load panel file mapping samples to populations
panel = pd.read_csv(PANEL_FILE, sep="\t")
# Create sample ID to superpopulation mapping
sample_to_superpop = dict(zip(panel["sample"], panel["super_pop"]))
# Map VCF samples to superpopulations
sample_superpops = [sample_to_superpop.get(s, "Unknown") for s in samples]
superpops = sorted(set(sample_superpops))
print(f"\nSuperpopulations found: {superpops}")

# Count samples per superpopulation
for pop in superpops:
    count = sample_superpops.count(pop)
    print(f"  {pop}: {count} samples")

sample_superpops_array = np.array(sample_superpops)

pop_data = {}
pi_results = {}
pop_indices = {}# goal is to run sums once per population across all variants
for pop in superpops:
    # create mask to only get and sum alt alleles from samples that are == pop
    mask = sample_superpops_array == pop
    pop_genotypes = genotypes[:, mask, :]
    n_alt = np.sum(pop_genotypes == 1, axis=(1, 2))

    allele_count = pop_genotypes.count_alleles()
    pop_indices[pop] = np.where(mask)[0]
    pi = np.mean(allel.mean_pairwise_difference(ac=allele_count))
    pi_results[pop] = pi
    # find ref alleles
#     valid_mask = pop_genotypes >= 0
#     n_valid = np.sum(valid_mask, axis=(1, 2))

#     n_ref = n_valid - n_alt

#     # only calculate freq when n_valid > 0
#     alt_freq = np.where(n_valid > 0, n_alt / n_valid, np.nan)

#     pop_data[f"{pop}_REF_COUNT"] = n_ref
#     pop_data[f"{pop}_ALT_COUNT"] = n_alt
#     pop_data[f"{pop}_ALT_FREQ"] = alt_freq

# df = pd.DataFrame({
#     "CHROM": chrom,
#     "POS": pos,
#     "ID": variant_id if variant_id is not None else [f"{c}:{p}" for c, p in zip(chrom, pos)],
#     "REF": ref,
#     "ALT": alt[:, 0],
#     **pop_data
# })

# df.to_csv(OUTPUT_FILE, index=False)
# print(f"Total variants: {len(df)}")
# print("\nFirst few rows:")
# print(df.head())

# # Summary
# print("\nSummary stats:")
# for pop in superpops:
#     freq_col = f"{pop}_ALT_FREQ"
#     if freq_col in df.columns:
#         mean_freq = df[freq_col].mean()
#         print(f"    {pop} mean alternate allele freq: {mean_freq:.4f}")


# calculate fst 
fst_results = {}

for pop1, pop2 in combinations([p for p in superpops if p != "Unknown"], 2):
    subpops = [pop_indices[pop1], pop_indices[pop2]]
    a, b, c = allel.weir_cockerham_fst(genotypes, subpops=subpops)
    fst = np.sum(a) / (np.sum(a) + np.sum(b) + np.sum(c))
    fst_results[pop1, pop2] = fst


# Write pi_results (nucleotide diversity) to CSV only if file doesn't exist
pi_csv_file = "nucleotide_diversity_per_superpop.csv"
if not os.path.exists(pi_csv_file):
    pi_df = pd.DataFrame(
        [{"Superpopulation": pop, "Nucleotide_Diversity_pi": pi.item() if hasattr(pi, "item") else pi} for pop, pi in pi_results.items()]
    )
    pi_df.to_csv(pi_csv_file, index=False)

else:
    # Load existing pi_results from CSV
    pi_df = pd.read_csv(pi_csv_file)
    pi_results = {}
    for _, row in pi_df.iterrows():
        pi_results[row["Superpopulation"]] = row["Nucleotide_Diversity_pi"]

# Write fst_results (pairwise FST) to CSV only if file doesn't exist
fst_csv_file = "pairwise_fst_results.csv"
if not os.path.exists(fst_csv_file):
    fst_rows = []
    for (pop1, pop2), fst in fst_results.items():
        fst_rows.append({"Pop1": pop1, "Pop2": pop2, "Pairwise_FST": fst})
    fst_df = pd.DataFrame(fst_rows)
    fst_df.to_csv(fst_csv_file, index=False)

else:
    # Load existing fst_results from CSV
    fst_df = pd.read_csv(fst_csv_file)
    fst_results = {}
    for _, row in fst_df.iterrows():
        fst_results[(row["Pop1"], row["Pop2"])] = row["Pairwise_FST"]


# print results 
print("Mean Nucleotide diversity per superpopulation")
for pop, pi in pi_results.items():
    print(f"{pop}: π = {pi.item():.4f}")

print("="*100)

print("Pairwise FST results")
for (pop1, pop2), fst in fst_results.items():
    print(f"Pairwise FST between {pop1} and {pop2}: {fst}")


time_diff= time.time() - start_time
print(f"Total time: {time_diff}")