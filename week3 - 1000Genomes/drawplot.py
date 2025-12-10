import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# superpopulation name
rawdata = pd.read_csv("population_summary.csv")
superpops = rawdata["super_pop"].tolist()
unique_superpops = list(dict.fromkeys(superpops))
print(unique_superpops)

# observed
observed = [1044, 661, 673, 670, 535]
print(observed)

# expected
expected = sum(observed) / len(observed)

chi2_stat = 0.9230769230769229
p_value   = 0.9212269650259751
kl_div    = 0.017038188811782193


observed = np.array(observed, dtype=float)
expected_arr = np.array([expected] * len(observed), dtype=float)

# Δ% = ((Observed - Expected) / Expected) * 100
delta_pct = (observed - expected_arr) / expected_arr * 100

x = np.arange(len(unique_superpops))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

#Observed vs Expected
bars_obs = ax.bar(x - width/2, observed, width, label="Observed")
bars_exp = ax.bar(x + width/2, expected_arr, width, label="Expected")

ax.set_xticks(x)
ax.set_xticklabels(unique_superpops)
ax.set_ylabel("Sample count")
ax.set_title("Observed vs Expected sample counts by superpopulation")
ax.legend(loc="upper right")      


offset = observed.max() * 0.02
for xo, obs, d in zip(x, observed, delta_pct):
    ax.text(
        xo - width/2,
        obs + offset,
        f"Δ={int(round(d))}%",   
        ha="center",
        va="bottom",
        fontsize=8,
        rotation=0
    )


text_str = (
    f"$\\chi^2$ = {chi2_stat:.3f}\n"
    f"p = {p_value:.3g}\n"
    f"KL = {kl_div:.4f}"
)

ax.text(0.98, 0.85, text_str,     # 注释往下放一点
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.2))

plt.tight_layout()
plt.savefig("week4_gap_visuals.png", dpi=300)


open