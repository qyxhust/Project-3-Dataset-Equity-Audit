"""
Configuration file for VCF analysis pipeline.
Contains all file paths and constants.
"""

# Input files
VCF_FILE = "/Users/ashwin/Desktop/GitHub/Project-3-Dataset-Equity-Audit/ALL.chr22.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf"
PANEL_FILE = "/Users/ashwin/Desktop/GitHub/Project-3-Dataset-Equity-Audit/integrated_call_samples_v3.20130502.ALL.panel"

# Cache files
CACHE_FILE = "callset_cache.pkl"

# Output files
ALLELE_FREQS_OUTPUT = "allele_freqs_optimized.csv"
NUCLEOTIDE_DIVERSITY_CSV = "nucleotide_diversity_per_superpop.csv"
PAIRWISE_FST_CSV = "pairwise_fst_results.csv"
SEQUENCE_DIVERGENCE_CSV = "sequence_divergence_results.csv"
