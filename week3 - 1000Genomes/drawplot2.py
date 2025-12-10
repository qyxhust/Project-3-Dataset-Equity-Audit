import numpy as np
import matplotlib.pyplot as plt

txt_path = "methods_representation.txt"   

pi_dict = {}
fst_pairs = []

with open(txt_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 1) get pi
        if "π" in line and ":" in line and "Mean Nucleotide" not in line:
            left, right = line.split(":", 1)
            pop = left.strip()
            val_str = right.split("=")[-1].strip()
            pi_dict[pop] = float(val_str)

        # 2) get Fst
        if line.startswith("Pairwise FST between"):
            parts = line.split()
            pop1 = parts[3]
            pop2 = parts[5].rstrip(":")
            val = float(parts[-1])
            fst_pairs.append((pop1, pop2, val))


pi_pops = list(pi_dict.keys())
pi_vals = [pi_dict[p] for p in pi_pops]


fst_pops = sorted(list({p1 for p1,_,_ in fst_pairs} |
                       {p2 for _,p2,_ in fst_pairs}))
n = len(fst_pops)
fst_mat = np.zeros((n, n))

pop_idx = {p:i for i, p in enumerate(fst_pops)}
for p1, p2, v in fst_pairs:
    i, j = pop_idx[p1], pop_idx[p2]
    fst_mat[i, j] = v
    fst_mat[j, i] = v   # 对称

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# π
x = np.arange(len(pi_pops))
ax1.bar(x, pi_vals)
ax1.set_xticks(x)
ax1.set_xticklabels(pi_pops)
ax1.set_ylabel("Mean nucleotide diversity π")
ax1.set_title("Mean nucleotide diversity per superpopulation")

# Fst 
im = ax2.imshow(fst_mat, origin="upper", cmap="Blues")
ax2.set_xticks(np.arange(n))
ax2.set_xticklabels(fst_pops, rotation=45, ha="right")
ax2.set_yticks(np.arange(n))
ax2.set_yticklabels(fst_pops)
ax2.set_title("Pairwise Fst between superpopulations")


for i in range(n):
    for j in range(n):
        ax2.text(j, i, f"{fst_mat[i,j]:.2f}",
                 ha="center", va="center", fontsize=7)

fig.colorbar(im, ax=ax2, label="Fst")

plt.tight_layout()
plt.savefig("diversity_fst_visuals.png", dpi=300)
plt.show()
