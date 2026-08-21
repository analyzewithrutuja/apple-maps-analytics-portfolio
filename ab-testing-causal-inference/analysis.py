import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nsw = pd.read_csv("nsw_experimental.csv")
cps = pd.read_csv("cps_comparison.csv")

covariates = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]

# =====================================================================
# PART 1 -- A/B TEST: analyze the REAL randomized experiment (NSW)
# =====================================================================
print("="*70)
print("PART 1: A/B TEST ANALYSIS (randomized job-training experiment)")
print("="*70)

treat = nsw[nsw["treat"] == 1]
control = nsw[nsw["treat"] == 0]
print(f"Treatment (job training): n={len(treat)}   Control: n={len(control)}")

# Sample Ratio Mismatch check (expect ~ observed split, sanity check on randomization)
print(f"\nTreatment split: {len(treat)/len(nsw):.1%} vs expected roughly balanced design")

# Covariate balance check -- did randomization actually balance the groups?
print("\n--- Covariate balance (treat vs control), should show NO significant differences ---")
for c in covariates:
    t_stat, p_val = stats.ttest_ind(treat[c], control[c])
    flag = "  <-- IMBALANCE" if p_val < 0.05 else ""
    print(f"{c:10s}  treat_mean={treat[c].mean():9.2f}  control_mean={control[c].mean():9.2f}  p={p_val:.3f}{flag}")

# Primary outcome test: re78 (1978 earnings, post-treatment)
t_stat, p_val = stats.ttest_ind(treat["re78"], control["re78"])
diff = treat["re78"].mean() - control["re78"].mean()
se = np.sqrt(treat["re78"].var()/len(treat) + control["re78"].var()/len(control))
ci_low, ci_high = diff - 1.96*se, diff + 1.96*se

print(f"\n--- Primary outcome: re78 (1978 earnings) ---")
print(f"Treatment mean: ${treat['re78'].mean():,.0f}")
print(f"Control mean:   ${control['re78'].mean():,.0f}")
print(f"Estimated effect (TRUE experimental benchmark): ${diff:,.0f}")
print(f"95% CI: [${ci_low:,.0f}, ${ci_high:,.0f}]")
print(f"t={t_stat:.2f}, p={p_val:.4f}  -> {'SIGNIFICANT' if p_val < 0.05 else 'not significant'} at alpha=0.05")

TRUE_EFFECT = diff  # this is our ground-truth benchmark for Part 2

# =====================================================================
# PART 2 -- CAUSAL INFERENCE: recover the effect from OBSERVATIONAL data
# =====================================================================
print("\n" + "="*70)
print("PART 2: CAUSAL INFERENCE (no randomized control group available)")
print("="*70)
print("Scenario: only the treated group is available: no randomized control.")
print("Must construct a comparison group from an observational survey (CPS) instead.")

obs = pd.concat([treat.assign(source="treated"), cps.assign(source="CPS")], ignore_index=True)
print(f"\nObservational sample: {len(treat)} treated vs {len(cps)} CPS non-participants")

# Naive (biased) estimate: just compare means, ignoring confounding
naive_diff = obs[obs.treat==1]["re78"].mean() - obs[obs.treat==0]["re78"].mean()
print(f"\n--- Naive comparison (no adjustment) ---")
print(f"Naive estimated effect: ${naive_diff:,.0f}")
print(f"True experimental benchmark: ${TRUE_EFFECT:,.0f}")
print(f"Bias: ${naive_diff - TRUE_EFFECT:,.0f}  ({'MASSIVELY' if abs(naive_diff-TRUE_EFFECT) > 5000 else ''} biased -- CPS is not a valid comparison group)")

# Propensity score matching: model P(treat=1 | covariates), match on it
X = obs[covariates]
y = obs["treat"]
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X, y)
obs["propensity"] = ps_model.predict_proba(X)[:, 1]

# Nearest-neighbor 1:1 matching on propensity score (with replacement)
treated_ps = obs[obs.treat==1][["propensity"]].values
control_pool = obs[obs.treat==0].reset_index(drop=True)
control_ps = control_pool[["propensity"]].values

matched_control_idx = []
for p in treated_ps:
    dists = np.abs(control_ps - p).flatten()
    matched_control_idx.append(np.argmin(dists))

matched_controls = control_pool.iloc[matched_control_idx]
psm_diff = obs[obs.treat==1]["re78"].mean() - matched_controls["re78"].mean()

print(f"\n--- Propensity Score Matching (1:1 nearest neighbor) ---")
print(f"PSM estimated effect: ${psm_diff:,.0f}")
print(f"True experimental benchmark: ${TRUE_EFFECT:,.0f}")
print(f"Remaining bias after PSM: ${psm_diff - TRUE_EFFECT:,.0f}")
print(f"Bias reduction vs naive: {(1 - abs(psm_diff-TRUE_EFFECT)/abs(naive_diff-TRUE_EFFECT))*100:.1f}%")

# Check overlap / common support
print(f"\nPropensity score overlap check:")
print(f"Treated PS range:  [{obs[obs.treat==1]['propensity'].min():.3f}, {obs[obs.treat==1]['propensity'].max():.3f}]")
print(f"CPS pool PS range: [{obs[obs.treat==0]['propensity'].min():.3f}, {obs[obs.treat==0]['propensity'].max():.3f}]")

# =====================================================================
# Plots
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(16,4.5))

methods = ["True Experimental\n(A/B Test)", "Naive Comparison\n(CPS, unadjusted)", "Propensity Score\nMatching"]
estimates = [TRUE_EFFECT, naive_diff, psm_diff]
colors = ["seagreen", "firebrick", "steelblue"]
axes[0].bar(methods, estimates, color=colors)
axes[0].axhline(TRUE_EFFECT, color="black", linestyle="--", linewidth=1, label="True effect")
axes[0].set_ylabel("Estimated Treatment Effect on 1978 Earnings ($)")
axes[0].set_title("Estimate vs. Ground Truth")
axes[0].legend()

axes[1].hist(obs[obs.treat==1]["propensity"], bins=30, alpha=0.6, label="Treated (job training)", color="seagreen")
axes[1].hist(obs[obs.treat==0]["propensity"], bins=30, alpha=0.6, label="CPS comparison pool", color="gray")
axes[1].set_xlabel("Propensity Score")
axes[1].set_title("Propensity Score Overlap")
axes[1].legend()

def std_mean_diff(a, b):
    # Standardized mean difference: (mean_a - mean_b) / pooled_std -- the standard "Love plot" metric
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0

bal_data = pd.DataFrame({
    "covariate": covariates,
    "treat_control_smd": [std_mean_diff(treat[c], control[c]) for c in covariates],
    "treat_cps_smd": [std_mean_diff(treat[c], cps[c]) for c in covariates],
})
x = np.arange(len(covariates))
axes[2].barh(x - 0.2, bal_data["treat_control_smd"], height=0.4, label="Treat vs Randomized Control", color="seagreen")
axes[2].barh(x + 0.2, bal_data["treat_cps_smd"], height=0.4, label="Treat vs CPS (unmatched)", color="firebrick")
axes[2].set_yticks(x)
axes[2].set_yticklabels(covariates)
axes[2].axvline(0, color="black", linewidth=0.8)
axes[2].axvline(0.1, color="gray", linestyle=":", linewidth=1)
axes[2].axvline(-0.1, color="gray", linestyle=":", linewidth=1, label="±0.1 balance threshold")
axes[2].set_xlabel("Standardized Mean Difference")
axes[2].set_title("Covariate Balance (Love Plot)\nrandomized vs observational")
axes[2].legend(fontsize=7, loc="lower right")

plt.tight_layout()
plt.savefig("ab_causal_results.png", dpi=130)
print("\nSaved ab_causal_results.png")

# Save results table
results = pd.DataFrame({
    "method": methods,
    "estimated_effect": estimates,
    "abs_bias_vs_true": [abs(e - TRUE_EFFECT) for e in estimates],
})
results.to_csv("results_summary.csv", index=False)
print(results.to_string())
