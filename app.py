Iimport streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Global Session State Management ---
# Robust initialization to prevent KeyErrors during widget interactions
DEFAULTS = {
    "maf": 0.15, "ld": 0.1,
    "n1": 100000, "b1": 0.3, "p1": 100,
    "n2": 5000, "b2": 0.4, "p2": 100
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- 2. Statistically Grounded Simulator ---
def run_simulation(n_snps, pos1, pos2, n1, n2, b1, b2, maf, ld_decay):
    snps = np.arange(n_snps)

    def generate_statistics(causal_idx, true_beta, n_samples, allele_freq):
        # Calculation of Standard Error (SE) based on sample size and MAF
        # Precision scales with sqrt(2 * N * MAF * (1-MAF))
        se_val = 1 / np.sqrt(2 * n_samples * allele_freq * (1 - allele_freq))

        # Exponential LD decay modeling regional association structure
        base_beta = np.zeros(n_snps)
        base_beta[causal_idx] = true_beta
        for i in range(n_snps):
            dist = abs(i - causal_idx)
            ld_factor = np.exp(-dist * ld_decay)
            base_beta[i] = base_beta[causal_idx] * ld_factor

        # Sampling distribution: Observed Beta = True Effect + Residual Noise
        obs_beta = base_beta + np.random.normal(0, se_val, n_snps)
        obs_se = np.full(n_snps, se_val)
        return obs_beta, obs_se

    # Trait statistics generated independently to evaluate linkage vs colocalization
    b1_s, se1_s = generate_statistics(pos1, b1, n1, maf)
    b2_s, se2_s = generate_statistics(pos2, b2, n2, maf)
    return snps, b1_s, b2_s, se1_s, se2_s

# --- 3. Numerically Stable Colocalization Engine ---
def compute_posteriors(b1, se1, b2, se2):
    # Log-space Approximate Bayes Factor (ABF) calculation to prevent overflow
    def get_log_abf(beta, se):
        v, w = se**2, 0.15**2 # Prior variance W fixed at 0.15^2
        r = w / (w + v)
        z_sq = (beta/se)**2
        return 0.5 * (np.log(1 - r) + (z_sq * r))

    l_abf1, l_abf2 = get_log_abf(b1, se1), get_log_abf(b2, se2)

    # Bayesian priors: p1, p2 (1e-4) and p12 (1e-5)
    lp1, lp2, lp12 = np.log(1e-4), np.log(1e-4), np.log(1e-5)

    # Log-Sum-Exp trick for computational stability in high-resolution datasets
    def logsum(l_vec):
        m = np.max(l_vec)
        return m + np.log(np.sum(np.exp(l_vec - m)))

    lH0 = 0
    lH1, lH2 = lp1 + logsum(l_abf1), lp2 + logsum(l_abf2)
    lH3 = lp1 + lp2 + logsum(l_abf1[:, None] + l_abf2[None, :])
    lH4 = lp12 + logsum(l_abf1 + l_abf2) # Integration across shared variants

    l_all = np.array([lH0, lH1, lH2, lH3, lH4])
    probs = np.exp(l_all - logsum(l_all))
    return probs

# --- 4. Sidebar Controls & Automated Scenarios ---
with st.sidebar:
    st.header("Assay Configuration")

    if st.button("Initialize Defaults"):
        for key, val in DEFAULTS.items(): st.session_state[key] = val
        st.rerun()

    st.divider()
    st.subheader("Statistical Scenarios")

    if st.button("Apply Case: Underpowered QTL"):
        st.session_state.p1, st.session_state.p2 = 100, 100
        st.session_state.n2, st.session_state.b2 = 250, 0.15
        st.rerun()

    if st.button("Apply Case: Strong Linkage (H3)"):
        st.session_state.p1, st.session_state.p2 = 98, 102
        st.session_state.ld, st.session_state.n2 = 0.05, 5000
        st.rerun()

    st.divider()
    st.subheader("Population Parameters")
    maf = st.slider("Minor Allele Frequency", 0.01, 0.5, key="maf")
    ld = st.slider("LD Decay Rate", 0.01, 0.5, key="ld")

    st.subheader("Simulated Statistics")
    n1 = st.number_input("N1 (Trait A)", 1000, 1000000, key="n1")
    b1 = st.slider("Beta A", -1.0, 1.0, key="b1")
    p1 = st.slider("Peak Index A", 0, 200, key="p1")

    n2 = st.number_input("N2 (Trait B)", 100, 100000, key="n2")
    b2 = st.slider("Beta B", -1.0, 1.0, key="b2")
    p2 = st.slider("Peak Index B", 0, 200, key="p2")

# --- 5. Execution & Visualization ---
snps, b1_o, b2_o, se1_o, se2_o = run_simulation(200, p1, p2, n1, n2, b1, b2, maf, ld)
results = compute_posteriors(b1_o, se1_o, b2_o, se2_o)

st.title("Bayesian Colocalization Simulation")
col_plt, col_res = st.columns([2, 1])

with col_plt:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    lp1, lp2 = (b1_o**2 / (se1_o**2 * 2)) / np.log(10), (b2_o**2 / (se2_o**2 * 2)) / np.log(10)
    ax1.scatter(snps, lp1, c=lp1, cmap='winter', s=30); ax1.set_ylabel("Trait A (-log10P)")
    ax2.scatter(snps, lp2, c=lp2, cmap='autumn', s=30); ax2.set_ylabel("Trait B (-log10P)")
    st.pyplot(fig)

with col_res:
    st.subheader("Posterior Distribution")
    h_labels = ["H0 (Null)", "H1 (Trait A Only)", "H2 (Trait B Only)", "H3 (Independent/LD)", "H4 (Colocalized)"]
    for i, p in enumerate(results):
        st.write(f"**{h_labels[i]}:** {float(np.clip(p, 0.0, 1.0)):.2%}")
        st.progress(float(np.clip(p, 0.0, 1.0)))

# --- 6. Technical Analysis ---
st.divider()
st.header("Assay Interpretation and Methodology")

st.markdown("""
### Simulation Parameters
* **Sample Size (N) and SE:** The standard error is a function of sample size and allele frequency ($SE \sim 1/\sqrt{2N \cdot MAF(1-MAF)}$). Insufficient power in either dataset increases the probability of $H_0$ or $H_1/H_2$ despite overlapping signals.
* **LD Decay:** Simulates regional linkage disequilibrium. Low decay rates generate broad association blocks, which can inflate $H_4$ probabilities if independent causal variants are located in close proximity.
* **Positional Dissonance:** The Bayesian posterior for $H_4$ is derived from the integration of evidence across the locus. If the causal variants differ (e.g., Index 100 vs 103), the product of Bayes Factors decreases, shifting probability to $H_3$ (Linkage).

### Controlled Scenarios
1. **The Underpowered QTL:** Observe how $H_1$ dominates when N2 is low, even when peaks are perfectly aligned at index 100. This demonstrates the influence of Bayesian priors ($10^{-5}$ penalty for $H_4$).
2. **The LD Blur:** Set LD decay to 0.01 and N2 to 500 while causal indices differ by 4 units. Note how $H_4$ may increase as the statistical resolution fails to distinguish the separate causal drivers.
3. **FTO Example Application:** Consider the obesity-associated *FTO* locus. Colocalization analysis is used to determine if the BMI GWAS signal shares a causal variant with the expression of nearby genes like *IRX3*. A high $H_4$ indicates that the BMI-associated variant is likely a regulator of *IRX3*.
""")
