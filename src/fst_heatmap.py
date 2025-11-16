
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


def create_fst_matrix(fst_results: dict, populations: list) -> pd.DataFrame:
    """
    Create a symmetric F_ST matrix from pairwise results.

    Args:
        fst_results: Dictionary of (pop1, pop2) -> F_ST value
        populations: List of population names

    Returns:
        DataFrame with F_ST matrix (symmetric, diagonal = 0)
    """

    n_pops = len(populations)

    matrix = np.zeros((n_pops, n_pops))

    pop_to_idx = {pop: i for i, pop in enumerate(populations)}

    # Fill matrix with F_ST values

    for (pop1, pop2), fst in fst_results.items():
        i = pop_to_idx[pop1]

        j = pop_to_idx[pop2]

        # Symmetric matrix

        matrix[i, j] = fst

        matrix[j, i] = fst

    # Create DataFrame

    df = pd.DataFrame(matrix, index=populations, columns=populations)
    
    return df

def create_fst_heatmap(fst_matrix, superpops, output_file="fst_heatmap.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        fst_matrix,
        xticklabels=superpops,
        yticklabels=superpops,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "FST"}
    )


    plt.title("Pairwise FST Between Superpopulations", fontsize=14, fontweight="bold")
    plt.xlabel("Population", fontsize=12)
    plt.ylabel("Population", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved FST heatmap to file {output_file}")
    plt.close()

