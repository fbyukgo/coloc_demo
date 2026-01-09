import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration & Defaults ---
DEFAULTS = {
    "maf": 0.15,
    "ld_decay": 0.1,
    "n_trait1": 100000,
    "beta_trait1": 0.3,
    "pos_trait1": 100,
    "n_trait2": 5000,
    "beta_trait2": 0.4,
    "pos_trait2": 100,
}

st.set_page_config(layout="wide")

# --- Simulation Logic ---
def simulate_locus(n_snps, pos1, pos2, n1, n2, b1, b2, maf, ld_decay):
    snps = np.arange(n_snps)

    def generate_statistics(causal_idx, true_beta, n_samples, allele_freq):
        # Standard Error (SE) calculation based on N and MAF
        # Formula: SE = 1 / sqrt(2 * N * MAF * (1 - MAF))
        se_val = 1 / np.sqrt(2 * n_samples * allele_freq * (1 - allele_freq))

        # Modeling the association signal across the locus via LD decay
        # Simulated as an exponential decay from the causal variant
        base_beta = np.zeros(n_snps)
        base_beta[causal_idx] = true_beta
        for i in range(n_snps):
            dist = abs(i - causal_idx)
            ld_factor = np.exp(-dist * ld_decay)
            base_beta[i] = base_beta[causal_idx] * ld_factor

        # Sampling distribution: Observed Beta = True Beta + Noise(0, SE)
        observed_beta = base_beta + np.random.normal(0, se_val, n_snps)
        observed_se = np.full(n_snps, se_val)
        return observed_beta, observed_se

    b1_obs, se1_obs = generate_statistics(pos1, b1, n1, maf)
    b2_obs, se2_obs = generate_statistics(pos2, b2, n2, maf)
    return snps, b1_obs, b2_obs, se1_obs, se2_obs

# --- Colocalization Engine ---
def compute_coloc_posteriors(b1, se1, b2, se2):
    # Log-space Approximate Bayes Factor (ABF) for numerical stability
    def get_log_abf(beta, se):
        v, w = se**2, 0.15**2 # Prior variance W fixed at 0.15^2
        r = w / (w + v)
        z_sq = (beta/se)**2
        return 0.5 * (np.log(1 - r) + (z_sq * r))

    labf1 = get_log_abf(b1, se1)
    labf2 = get_log_abf(b2, se2)

    # Prior probabilities
    # p1: variant causal for trait 1
    # p2: variant causal for trait 2
    # p12: variant causal for both (shared)
    lp1, lp2, lp12 = np.log(1e-4), np.log(1e-4), np.log(1e-5)

    def logsum(l_vec):
        m = np.max(l_vec)
        return m + np.log(np.sum(np.exp(l_vec - m)))

    # Aggregate log-probabilities for the five hypotheses
    lH0 = 0 # Null hypothesis
    lH1 = lp1 + logsum(labf1)
    lH2 = lp2 + logsum(labf2)
    lH3 = lp1 + lp2 + logsum(labf1[:, None] + labf2[None, :])
    lH4 = lp12 + logsum(labf1 + labf2)

    l_all = np.array([lH0, lH1, lH2, lH3, lH4])
    post_probs = np.exp(l_all - logsum(l_all))

    return {"H0": post_probs[0], "H1": post_probs[1],
            "H2": post_probs[2], "H3": post_probs[3], "H4": post_probs[4]}

# --- Sidebar Interface ---
with st.sidebar:
    st.header("Simulation Parameters")

    if st.button("Reset to Defaults"):
        for key, val in DEFAULTS.items():
            st.session_state[key] = val

    st.subheader("Population Genetics")
    maf = st.slider("Minor Allele Frequency (MAF)", 0.01, 0.50, key="maf", value=DEFAULTS["maf"])
    ld_decay = st.slider("LD Decay Rate", 0.01, 0.5, key="ld_decay", value=DEFAULTS["ld_decay"])

    st.divider()
    st.subheader("Trait 1 Statistics (GWAS)")
    n1 = st.number_input("Sample Size (N1)", 1000, 1000000, key="n1", value=DEFAULTS["n1"])
    beta1 = st.slider("Causal Effect (Beta1)", -1.0, 1.0, key="beta1", value=DEFAULTS["beta1"])
    pos1 = st.slider("Causal Position 1", 0, 200, key="pos1", value=DEFAULTS["pos1"])

    st.divider()
    st.subheader("Trait 2 Statistics (Molecular QTL)")
    n2 = st.number_input("Sample Size (N2)", 100, 100000, key="n2", value=DEFAULTS["n2"])
    beta2 = st.slider("Causal Effect (Beta2)", -1.0, 1.0, key="beta2", value=DEFAULTS["beta2"])
    pos2 = st.slider("Causal Position 2", 0, 200, key="pos2", value=DEFAULTS["pos2"])

# Execution
snps, b1_s, b2_s, se1_s, se2_s = simulate_locus(200, pos1, pos2, n1, n2, beta1, beta2, maf, ld_decay)
probs = compute_coloc_posteriors(b1_s, se1_s, b2_s, se2_s)

# --- Main Results Display ---
st.title("Bayesian Colocalization Simulation")
st.markdown("""
This tool simulates summary statistics for two traits to evaluate the posterior probability
of shared versus independent causal variants within a single genomic locus.
""")

col_plt, col_res = st.columns([2, 1])

with col_plt:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    # Approximation of -log10P for visualization purposes
    lp1 = (b1_s**2 / (se1_s**2 * 2)) / np.log(10)
    lp2 = (b2_s**2 / (se2_s**2 * 2)) / np.log(10)

    ax1.scatter(snps, lp1, c=lp1, cmap='winter', s=30, edgecolors='none')
    ax1.set_ylabel("Trait 1 (-log10P)")
    ax2.scatter(snps, lp2, c=lp2, cmap='autumn', s=30, edgecolors='none')
    ax2.set_ylabel("Trait 2 (-log10P)")
    ax2.set_xlabel("SNP Index")
    st.pyplot(fig)

with col_res:
    st.subheader("Posterior Probabilities")
    for h, p in probs.items():
        st.write(f"**{h}:** {p:.2%}")
        st.progress(float(np.clip(p, 0.0, 1.0)))

# --- Documentation ---
st.divider()
st.header("Statistical Documentation")

st.markdown("""
### Hypothesis Definitions
Colocalization analysis evaluates five mutually exclusive hypotheses regarding the causal architecture of a locus:
* **H0**: No causal variant for either trait.
* **H1**: Causal variant for Trait 1 only.
* **H2**: Causal variant for Trait 2 only.
* **H3**: Independent causal variants for each trait (Linkage Disequilibrium).
* **H4**: Shared causal variant for both traits (Colocalization).

### Key Determinants of H4
1. **Sample Size (N) and SE**: Precision is determined by $SE = 1/\sqrt{2N \cdot MAF(1-MAF)}$. If $N$ is insufficient, the Bayes Factor cannot overcome the prior penalty for $H4$ ($10^{-5}$), often resulting in a high $H0$ or $H1$ probability despite apparent peak overlap.
2. **Positional Congruence**: The core of the $H4$ calculation is $\sum (ABF_{i,1} \times ABF_{i,2})$. Even a minor shift in causal position between traits will increase the evidence for $H3$ as the product of Bayes Factors for the top SNPs will decrease.
3. **Directionality**: Colocalization as implemented via ABF is agnostic to the sign of the effect size ($\beta$). It identifies a shared causal *locus*, but does not infer the causal *direction* between traits.
""")
