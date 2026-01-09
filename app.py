import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. SESSION STATE INITIALIZATION ---
# This block must run before widgets to prevent KeyErrors
DEFAULTS = {
    "maf": 0.15, "ld": 0.1,
    "n1": 100000, "b1": 0.3, "p1": 100,
    "n2": 5000, "b2": 0.4, "p2": 100
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- 2. SIMULATION ENGINE ---
def run_simulation(n_snps, pos1, pos2, n1, n2, b1, b2, maf, ld_decay):
    snps = np.arange(n_snps)

    def generate_stats(causal_idx, true_beta, n_samples, allele_freq):
        # Variance of the estimator based on sample size and allele frequency
        # Formula derived from 2 * N * MAF * (1-MAF)
        se_val = 1 / np.sqrt(2 * n_samples * allele_freq * (1 - allele_freq))

        # Modeling the regional association tower via exponential LD decay
        base_beta = np.zeros(n_snps)
        base_beta[causal_idx] = true_beta
        for i in range(n_snps):
            dist = abs(i - causal_idx)
            ld_factor = np.exp(-dist * ld_decay)
            base_beta[i] = base_beta[causal_idx] * ld_factor

        # Observed Beta = True Effect + Sampling Error
        obs_beta = base_beta + np.random.normal(0, se_val, n_snps)
        obs_se = np.full(n_snps, se_val)
        return obs_beta, obs_se

    b1_s, se1_s = generate_stats(pos1, b1, n1, maf)
    b2_s, se2_s = generate_stats(pos2, b2, n2, maf)
    return snps, b1_s, b2_s, se1_s, se2_s

# --- 3. NUMERICALLY STABLE COLOC ENGINE ---
def compute_posteriors(b1, se1, b2, se2):
    # Log-transforming ABF prevents numerical overflow (NaNs)
    def get_log_abf(beta, se):
        v, w = se**2, 0.15**2
        r = w / (w + v)
        z_sq = (beta/se)**2
        return 0.5 * (np.log(1 - r) + (z_sq * r))

    l_abf1 = get_log_abf(b1, se1)
    l_abf2 = get_log_abf(b2, se2)

    # Priors for H1, H2 (1e-4) and H4 (1e-5)
    lp1, lp2, lp12 = np.log(1e-4), np.log(1e-4), np.log(1e-5)

    # Log-Sum-Exp Trick for computational stability
    def logsum(l_vec):
        m = np.max(l_vec)
        return m + np.log(np.sum(np.exp(l_vec - m)))

    lH0 = 0
    lH1 = lp1 + logsum(l_abf1)
    lH2 = lp2 + logsum(l_abf2)
    lH3 = lp1 + lp2 + logsum(l_abf1[:, None] + l_abf2[None, :])
    lH4 = lp12 + logsum(l_abf1 + l_abf2)

    l_all = np.array([lH0, lH1, lH2, lH3, lH4])
    probs = np.exp(l_all - logsum(l_all))
    return probs

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Simulation Settings")

    if st.button("Reset Parameters"):
        for key, val in DEFAULTS.items():
            st.session_state[key] = val
        st.rerun()

    st.subheader("Regional Context")
    maf = st.slider("Minor Allele Frequency (MAF)", 0.01, 0.5, key="maf")
    ld = st.slider("LD Decay Rate", 0.01, 0.5, key="ld")

    st.divider()
    st.subheader("Trait 1 (Large Cohort)")
    n1 = st.number_input("Sample Size (N1)", 1000, 1000000, key="n1")
    beta1 = st.slider("Causal Effect (Beta 1)", -1.0, 1.0, key="b1")
    pos1 = st.slider("Causal SNP Index (A)", 0, 200, key="p1")

    st.divider()
    st.subheader("Trait 2 (Small Cohort)")
    n2 = st.number_input("Sample Size (N2)", 100, 100000, key="n2")
    beta2 = st.slider("Causal Effect (Beta 2)", -1.0, 1.0, key="b2")
    pos2 = st.slider("Causal SNP Index (B)", 0, 200, key="p2")

# --- 5. EXECUTION & OUTPUT ---
snps, b1_obs, b2_obs, se1_obs, se2_obs = run_simulation(200, pos1, pos2, n1, n2, beta1, beta2, maf, ld)
post_probs = compute_posteriors(b1_obs, se1_obs, b2_obs, se2_obs)

st.title("Bayesian Colocalization Analysis")
st.markdown("Analysis of shared versus independent causal variants within a genomic locus.")

col_viz, col_data = st.columns([2, 1])

with col_viz:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    # Approximation of -log10P
    lp1 = (b1_obs**2 / (se1_obs**2 * 2)) / np.log(10)
    lp2 = (b2_obs**2 / (se2_obs**2 * 2)) / np.log(10)

    ax1.scatter(snps, lp1, c=lp1, cmap='winter', s=30)
    ax1.set_ylabel("Trait 1 (-log10P)")
    ax2.scatter(snps, lp2, c=lp2, cmap='autumn', s=30)
    ax2.set_ylabel("Trait 2 (-log10P)")
    ax2.set_xlabel("Genomic Position (SNP Index)")
    st.pyplot(fig)

with col_data:
    st.subheader("Posterior Probabilities")
    h_labels = ["H0 (Null)", "H1 (Trait 1 Only)", "H2 (Trait 2 Only)", "H3 (Linkage/Indep)", "H4 (Shared/Coloc)"]
    for i, p in enumerate(post_probs):
        # Clamping value to [0.0, 1.0] to satisfy st.progress()
        safe_p = float(np.clip(p, 0.0, 1.0))
        st.write(f"**{h_labels[i]}:** {safe_p:.2%}")
        st.progress(safe_p)

# --- 6. STATISTICAL DOCUMENTATION ---
st.divider()
st.header("Assay Mechanics")


st.markdown("""
### Variable Determinants
* **Sample Size (N):** Determines the Standard Error ($SE$). If $N$ is too small, the Bayesian engine cannot distinguish a true signal from background noise ($H0$).
* **Minor Allele Frequency (MAF):** Modulates statistical precision. Low-frequency variants require exponentially larger sample sizes to reach the same evidence threshold as common variants.
* **LD Decay:** Controls the regional correlation structure. Lower decay rates create broader association blocks, making it statistically difficult to distinguish $H3$ (Linkage) from $H4$ (Colocalization).

### Hypothesis Logic
The posterior for $H4$ (Colocalization) is derived from the product of the Approximate Bayes Factors across all SNPs in the region.
If the peak association for Trait 1 and Trait 2 occurs at the same SNP, the product of their individual Bayes Factors increases exponentially, driving the $H4$ posterior toward 1.0.
""")
