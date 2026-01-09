import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Locus Simulation Lab")

# --- 1. Simulation Logic ---
def run_simulation(n_snps, pos1, pos2, n1, n2, b1, b2, ld_decay):
    snps = np.arange(n_snps)
    def generate(causal_idx, true_beta, n_samples):
        # SE is derived from sample size
        se_val = 1 / np.sqrt(n_samples)
        # Base signal + LD decay
        base = np.zeros(n_snps)
        base[causal_idx] = true_beta
        for i in range(n_snps):
            dist = abs(i - causal_idx)
            ld = np.exp(-dist * ld_decay)
            base[i] = base[causal_idx] * ld
        # Add noise relative to sample size
        beta = base + np.random.normal(0, se_val, n_snps)
        se = np.full(n_snps, se_val)
        return beta, se

    beta1, se1 = generate(pos1, b1, n1)
    beta2, se2 = generate(pos2, b2, n2)
    return snps, beta1, beta2, se1, se2

# --- 2. Sidebar Configuration ---
st.sidebar.header("Experimental Setup")

st.sidebar.subheader("Dataset A (BMI GWAS)")
n1 = st.sidebar.number_input("Sample Size (N1)", 5000, 500000, 100000)
b1 = st.sidebar.slider("Effect Size (β1)", 0.0, 1.0, 0.3)
pos1 = st.sidebar.slider("Causal Position (Trait A)", 0, 200, 100)

st.sidebar.subheader("Dataset B (IRX3 eQTL)")
n2 = st.sidebar.number_input("Sample Size (N2)", 100, 50000, 1000)
b2 = st.sidebar.slider("Effect Size (β2)", 0.0, 1.0, 0.5)
pos2 = st.sidebar.slider("Causal Position (Trait B)", 0, 200, 100)

st.sidebar.subheader("Genomic Structure")
ld = st.sidebar.slider("LD Decay Rate", 0.01, 0.5, 0.1)

# --- 3. Coloc Engine ---
def compute_coloc(b1, se1, b2, se2):
    # Calculate ABFs
    def get_abf(beta, se):
        v, w = se**2, 0.15**2
        r = w / (w + v)
        z_sq = (beta/se)**2
        return np.sqrt(1-r) * np.exp(z_sq * r / 2)

    abf1, abf2 = get_abf(b1, se1), get_abf(b2, se2)
    p1, p2, p12 = 1e-4, 1e-4, 1e-5 # Standard priors
    s1, s2, s12 = np.sum(abf1), np.sum(abf2), np.sum(abf1 * abf2)

    # Hypothesis weights
    h0 = 1
    h1, h2 = p1 * s1, p2 * s2
    h3 = p1 * p2 * (s1 * s2 - s12)
    h4 = p12 * s12

    total = h0 + h1 + h2 + h3 + h4
    return {"H0": h0/total, "H1": h1/total, "H2": h2/total, "H3": h3/total, "H4": h4/total}

# --- 4. Execution & Visualization ---
snps, b1_sim, b2_sim, se1_sim, se2_sim = run_simulation(200, pos1, pos2, n1, n2, b1, b2, ld)
results = compute_coloc(b1_sim, se1_sim, b2_sim, se2_sim)

st.title("FTO-IRX3 Colocalization Lab")
st.write("Determine if the BMI association (Trait A) and IRX3 expression (Trait B) share a single causal variant.")

col1, col2 = st.columns([2, 1])

with col1:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    # Approximation of -log10(P)
    lp1 = (b1_sim**2 / (se1_sim**2 * 2)) / np.log(10)
    lp2 = (b2_sim**2 / (se2_sim**2 * 2)) / np.log(10)

    ax1.scatter(snps, lp1, c=lp1, cmap='winter', s=30)
    ax1.set_ylabel("BMI GWAS (-log10P)")
    ax2.scatter(snps, lp2, c=lp2, cmap='autumn', s=30)
    ax2.set_ylabel("IRX3 eQTL (-log10P)")
    st.pyplot(fig)

with col2:
    st.subheader("Posterior Probabilities")
    for h, p in results.items():
        st.write(f"**{h}:** {p:.1%}")
        st.progress(p)

    if results['H4'] > 0.8:
        st.success("Evidence supports a shared causal variant (H4).")
    elif results['H3'] > 0.8:
        st.warning("Evidence supports independent causal variants in linkage (H3).")

# --- 5. Interpretation Guide ---
st.divider()
st.header("Assay Interpretation")



st.markdown("""
### 1. The Linkage Trap (H3)
Try moving the **Trait B Peak** just 5–10 units away from Trait A. Even though the association towers overlap significantly, you will see $H3$ overtake $H4$.
* **Why?** The Bayesian model identifies that the variant with the strongest evidence for Trait A is not the variant with the strongest evidence for Trait B. This distinguishes "near misses" from true biological sharing.

### 2. Sample Size vs. Priors
If $H4$ is low despite perfectly aligned peaks, check your **Sample Sizes ($N$)**.
* **Why?** $SE$ is calculated as $1/\sqrt{N}$. If $N$ is too small, the standard error is large, leading to low Z-scores. The Bayesian model is "skeptical" by default—if the evidence isn't strong enough to overcome the $10^{-5}$ prior penalty for $H4$, it will default to $H0$ (no signal).

### 3. Blindness to Directionality
Colocalization is based on squared Z-scores.
* **Insight:** Flip the **BMI Effect Size ($\beta1$)** from 0.3 to -0.3. The $H4$ probability remains identical. Coloc confirms the *location* is shared, but it does not determine the *direction* of causality. Determining if "High expression = High BMI" vs "High expression = Low BMI" requires secondary analysis like Mendelian Randomization.
""")
