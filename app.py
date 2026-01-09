import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. SESSION STATE INITIALIZATION ---
DEFAULTS = {
    "maf": 0.15, "ld": 0.1,
    "n1": 100000, "b1": 0.3, "p1": 100,
    "n2": 5000, "b2": 0.4, "p2": 100
}

# Ensure session state is established before widget creation to prevent KeyErrors
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- 2. SIMULATION ENGINE ---
def run_simulation(n_snps, pos1, pos2, n1, n2, b1, b2, maf, ld_decay):
    snps = np.arange(n_snps)

    def generate_stats(causal_idx, true_beta, n_samples, allele_freq):
        # Variance calculation incorporating N and MAF to model statistical power
        se_val = 1 / np.sqrt(2 * n_samples * allele_freq * (1 - allele_freq))

        # Exponential LD decay modeling regional association structure
        base_beta = np.zeros(n_snps)
        base_beta[causal_idx] = true_beta
        for i in range(n_snps):
            dist = abs(i - causal_idx)
            ld_factor = np.exp(-dist * ld_decay)
            base_beta[i] = base_beta[causal_idx] * ld_factor

        # Sampling: Observed = Truth + Noise
        obs_beta = base_beta + np.random.normal(0, se_val, n_snps)
        obs_se = np.full(n_snps, se_val)
        return obs_beta, obs_se

    b1_s, se1_s = generate_stats(pos1, b1, n1, maf)
    b2_s, se2_s = generate_stats(pos2, b2, n2, maf)
    return snps, b1_s, b2_s, se1_s, se2_s

# --- 3. NUMERICALLY STABLE COLOC ENGINE ---
def compute_posteriors(b1, se1, b2, se2):
    # Log-space math prevents NaN/Overflow errors during exponential calculations
    def get_log_abf(beta, se):
        v, w = se**2, 0.15**2
        r = w / (w + v)
        z_sq = (beta/se)**2
        return 0.5 * (np.log(1 - r) + (z_sq * r))

    l_abf1 = get_log_abf(b1, se1)
    l_abf2 = get_log_abf(b2, se2)

    # Standard Bayesian priors
    lp1, lp2, lp12 = np.log(1e-4), np.log(1e-4), np.log(1e-5)

    # Log-Sum-Exp trick for computational stability
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

    if st.button("Reset to Defaults"):
        for key, val in DEFAULTS.items():
            st.session_state[key] = val
        st.rerun()

    st.subheader("Population Genetics")
    maf = st.slider("Minor Allele Frequency (MAF)", 0.01, 0.5, key="maf")
    ld = st.slider("LD Decay Rate", 0.01, 0.5, key="ld")

    st.divider()
    st.subheader("Trait 1 (e.g., BMI GWAS)")
    n1 = st.number_input("Sample Size (N1)", 1000, 1000000, key="n1")
    beta1 = st.slider("Causal Effect (Beta 1)", -1.0, 1.0, key="b1")
    pos1 = st.slider("Causal SNP Index (A)", 0, 200, key="p1")

    st.divider()
    st.subheader("Trait 2 (e.g., Gene Expression)")
    n2 = st.number_input("Sample Size (N2)", 100, 100000, key="n2")
    beta2 = st.slider("Causal Effect (Beta 2)", -1.0, 1.0, key="b2")
    pos2 = st.slider("Causal SNP Index (B)", 0, 200, key="p2")

# --- 5. EXECUTION & OUTPUT ---
snps, b1_obs, b2_obs, se1_obs, se2_obs = run_simulation(200, pos1, pos2, n1, n2, beta1, beta2, maf, ld)
post_probs = compute_posteriors(b1_obs, se1_obs, b2_obs, se2_obs)

st.title("Bayesian Colocalization Simulation")
col_viz, col_data = st.columns([2, 1])

with col_viz:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    lp1 = (b1_obs**2 / (se1_obs**2 * 2)) / np.log(10)
    lp2 = (b2_obs**2 / (se2_obs**2 * 2)) / np.log(10)

    ax1.scatter(snps, lp1, c=lp1, cmap='winter', s=30)
    ax1.set_ylabel("Trait 1 (-log10P)")
    ax2.scatter(snps, lp2, c=lp2, cmap='autumn', s=30)
    ax2.set_ylabel("Trait 2 (-log10P)")
    ax2.set_xlabel("SNP Index")
    st.pyplot(fig)

with col_data:
    st.subheader("Posterior Probabilities")
    h_labels = ["H0 (Null)", "H1 (Trait 1 Only)", "H2 (Trait 2 Only)", "H3 (Linkage/Indep)", "H4 (Shared/Coloc)"]
    for i, p in enumerate(post_probs):
        safe_p = float(np.clip(p, 0.0, 1.0))
        st.write(f"**{h_labels[i]}:** {safe_p:.2%}")
        st.progress(safe_p)

# --- 6. SCIENTIFIC POINTERS & CHALLENGES ---
st.divider()
st.header("🔬 Experimental Challenges")

st.markdown("""
Use these exercises to understand the statistical thresholds of colocalization.

### Challenge 1: The Power Threshold
* **The Setup:** Keep **Causal SNP Index A & B** both at **100**. Set **N1** to **100,000**.
* **The Action:** Gradually decrease **N2** (Sample Size for Trait 2) from 5,000 down to 200.
* **The Observation:** Note that even though the peaks overlap perfectly, $H_4$ will eventually crash and $H_1$ will win.
* **The Lesson:** Bayesian evidence requires a minimum signal strength to overcome the $10^{-5}$ prior penalty for $H_4$.

### Challenge 2: Rare Variant Sensitivity
* **The Setup:** Return to defaults. Keep peaks aligned at **100**.
* **The Action:** Move the **MAF** slider from **0.15** down to **0.01**.
* **The Observation:** Watch the association towers become "noisier" and the $H_4$ probability drop.
* **The Lesson:** Rare variants have higher Standard Error ($SE$), requiring much larger sample sizes ($N$) to achieve the same colocalization confidence.

### Challenge 3: The Linkage Disequilibrium (LD) Trap
* **The Setup:** Align peaks at **100**. Set **LD Decay Rate** to **0.01** (creating very wide towers).
* **The Action:** Move **Causal SNP Index B** to **105**.
* **The Observation:** Despite the massive visual overlap between the two towers, $H_3$ (Linkage) will quickly dominate $H_4$.
* **The Lesson:** Coloc identifies that the *best* SNP for Trait A is not the *best* SNP for Trait B, effectively distinguishing independent signals even in dense LD blocks.

### Challenge 4: Directional Independence
* **The Setup:** Achieve a high $H_4$ (>90%).
* **The Action:** Change **Beta 1** from **0.3** to **-0.3**.
* **The Observation:** The $H_4$ probability remains exactly the same.
* **The Lesson:** Colocalization identifies shared genomic location; it is agnostic to whether the effect is protective or risk-increasing.
""")

st.divider()
st.header("Statistical Concepts")



st.markdown("""
### Hypotheses Overview
* **H0 (Null):** No association detected in either dataset.
* **H1/H2:** Association exists in only one of the two traits.
* **H3 (Linkage):** Both traits are associated, but they are driven by two distinct causal variants in the same locus.
* **H4 (Colocalization):** Both traits share a single causal variant.

### Example Application: BMI & FTO
A classic application is testing if a **BMI GWAS** signal at the *FTO* locus colocalizes with **Gene Expression** (eQTL) of nearby genes like *IRX3*.
If $H_4$ is high, it suggests the BMI-associated variant is functioning by regulating that specific gene's expression.
""")
