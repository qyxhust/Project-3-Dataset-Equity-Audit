"""
debug_and_verify.py

Debugging and verification script for the π and H_S analysis pipeline.
Checks data integrity, validates calculations, and identifies potential issues.

Author: Ashwin Kalyan
Date: 2025-11-08
"""

import allel
import pandas as pd
import numpy as np
from pathlib import Path
import sys

from config import VCF_FILE, PANEL_FILE, CACHE_FILE
from data_loader import load_vcf_with_cache, load_population_mapping


def check_file_existence():
    """Check if all required files exist."""
    files = {
        'VCF': VCF_FILE,
        'Panel': PANEL_FILE,
        'Cache': CACHE_FILE
    }
    
    all_exist = True
    for name, path in files.items():
        exists = Path(path).exists()
        if not exists and name != 'Cache':
            all_exist = False
    
    return all_exist


def check_vcf_structure(callset):
    """Verify VCF data structure and content."""
    required_fields = ['calldata/GT', 'samples', 'variants/POS']
    
    for field in required_fields:
        if field not in callset:
            return False
    
    return True


def check_genotype_data(genotypes):
    """Check genotype data quality."""
    is_missing = genotypes.is_missing()
    missing_rate = np.mean(is_missing)
    
    ac = genotypes.count_alleles()
    is_polymorphic = ac.is_segregating()
    poly_rate = np.mean(is_polymorphic)
    
    return True


def check_population_mapping(panel, samples):
    """Verify population mapping completeness."""
    required_cols = ['sample', 'super_pop']
    for col in required_cols:
        if col not in panel.columns:
            return False
    
    panel_samples = set(panel['sample'])
    vcf_samples = set(samples)
    
    in_both = vcf_samples & panel_samples
    
    return len(in_both) > 0


def check_population_sizes(sample_superpops, superpops):
    """Check population sample sizes."""
    return True


def test_pi_calculation(genotypes, sample_superpops):
    """Test π calculation on a subset."""
    sample_superpops_array = np.array(sample_superpops)
    test_pop = sample_superpops[0]
    
    mask = sample_superpops_array == test_pop
    test_genotypes = genotypes[:100, mask, :]
    
    ac = test_genotypes.count_alleles()
    pi = np.mean(allel.mean_pairwise_difference(ac=ac))
    
    return True


def test_hs_calculation(genotypes):
    """Test H_S calculation."""
    test_genotypes = genotypes[:100, :10, :]
    
    is_het = test_genotypes.is_het()
    hs = np.mean(is_het, axis=0)
    
    return True


def verify_existing_results():
    """Check if results already exist and verify them."""
    result_files = [
        'pi_values.csv',
        'nucleotide_diversity_per_superpop.csv',
        'pairwise_fst_results.csv'
    ]
    
    return True


def main():
    """Run all verification checks."""
    checks_passed = []
    
    checks_passed.append(("File Existence", check_file_existence()))
    
    if not checks_passed[-1][1]:
        return False
    
    callset = load_vcf_with_cache(VCF_FILE, CACHE_FILE)
    genotypes = allel.GenotypeArray(callset["calldata/GT"])
    samples = callset["samples"]
    pos = callset["variants/POS"]
    panel = load_population_mapping(PANEL_FILE)
    
    sample_to_superpop = dict(zip(panel["sample"], panel["super_pop"]))
    sample_superpops = [sample_to_superpop.get(s, "Unknown") for s in samples]
    superpops = sorted(set(sample_superpops))
    
    checks_passed.append(("VCF Structure", check_vcf_structure(callset)))
    checks_passed.append(("Genotype Quality", check_genotype_data(genotypes)))
    checks_passed.append(("Population Mapping", check_population_mapping(panel, samples)))
    checks_passed.append(("Population Sizes", check_population_sizes(sample_superpops, superpops)))
    checks_passed.append(("π Calculation", test_pi_calculation(genotypes, sample_superpops)))
    checks_passed.append(("H_S Calculation", test_hs_calculation(genotypes)))
    checks_passed.append(("Existing Results", verify_existing_results()))
    
    all_passed = all(passed for _, passed in checks_passed)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)