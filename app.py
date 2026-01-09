import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. DEFAULT CONFIGURATION ---
# We define defaults here so the Reset Button can reference them
DEFAULTS = {
    "maf": 0.15, "ld": 0.1,
    "n1": 100000, "b1": 0.3, "pos1": 100,
    "n2": 1000, "b2": 0.5, "pos2": 100
}

st.set_page_config(layout="wide", page_title="FTO-IRX3 Locus Lab")

# --- 2. THE SIMULATION ENGINE ---
def run_simulation(n_snps, pos1, pos2, n1, n2, b1, b2, maf, ld_decay):
    snps = np.arange(n_snps)

    def generate(causal_idx, true_beta, n_samples, allele_freq):
        # Statistical Power is a function of N and MAF
        # Common variants have more power for the same sample size
        se_val = 1 / np.sqrt(2 * n_samples * allele_freq * (1 - allele_freq))

        base = np.zeros(n_snps)
        base[causal_idx] = true_beta
        for i in range(n_snps):
            dist = abs(i - causal_idx)
            # Correlation (LD) decays exponentially from the causal SNP
            ld = np.exp(-dist * ld_decay)
            base[i] = base[causal_idx] * ld

        beta = base + np.random.normal(0, se_val, n_snps)
        se = np.full(n_snps, se_val)
        return beta, se

    beta1, se1 = generate(pos1, b1, n1, maf)
    beta2, se2 = generate(pos2, b2, n2, maf)
    return snps, beta1, beta2, se1, se2

# --- 3. THE COLOC ENGINE (NUMERICAL STABILITY) ---
def compute_coloc(b1, se1, b2, se2):
    # Log-transforming ABF prevents the NaN error seen in large datasets
    def get_log_abf(beta, se):
        v, w = se**2, 0.15**2
        r = w / (w + v)
        z_sq = (beta/se)**2
        return 0.5 * (np.log(1 - r) + (z_sq * r))

    labf1 = get_log_abf(b1, se1)
    labf2 = get_log_abf(b2, se2)

    lp1, lp2, lp12 = np.log(1e-4), np.log(1e-4), np.log(1e-5)

    def logsum(l_vec):
        m = np.max(l_vec)
        return m + np.log(np.sum(np.exp(l_vec - m)))

    # Hypothesis aggregation
    lH0 = 0
    lH1 = lp1 + logsum(labf1)
    lH2 = lp2 + logsum(labf2)
    lH3 = lp1 + lp2 + logsum(labf1[:, None] + labf2[None, :])
    lH4 = lp12 + logsum(labf1 + labf2)

    l_all = np.array([lH0, lH1, lH2, lH3, lH4])
    probs = np.exp(l_all - logsum(l_all))

    return {"H0": probs[0], "H1": probs[1], "H2": probs[2], "H3": probs[3], "H4": probs[4]}

# --- 4. SIDEBAR & RESET BUTTON ---
with st.sidebar:
    st.header("⚙️ Experimental Setup")

    # Reset Button
    if st.button("🔄 Reset Parameters"):
        for key, val in DEFAULTS.items():
            st.session_state[key] = val

    st.subheader("Population Parameters")
    maf = st.slider("Minor Allele Frequency (MAF)", 0.01, 0.50, key="maf", value=DEFAULTS["maf"])
    ld = st.slider("LD Decay Rate", 0.01, 0.5, key="ld", value=DEFAULTS["ld"])

    st.divider()
    st.subheader("BMI GWAS (Trait A)")
    n1 = st.number_input("Sample Size (N1)", 1000, 1000000, key="n1", value=DEFAULTS["n1"])
    b1 = st.slider("Effect Size (Beta 1)", -1.0, 1.0, key="b1", value=DEFAULTS["b1"])
    pos1 = st.slider("Peak Position A", 0, 200, key="pos1", value=DEFAULTS["pos1"])

    st.divider()
    st.subheader("IRX3 eQTL (Trait B)")
    n2 = st.number_input("Sample Size (N2)", 100, 100000, key="n2", value=DEFAULTS["n2"])
    b2 = st.slider("Effect Size (Beta 2)", -1.0, 1.0, key="b2", value=DEFAULTS["b2"])
    pos2 = st.slider("Peak Position B", 0, 200, key="pos2", value=DEFAULTS["pos2"])

# Execution
snps, b1_s, b2_s, se1_s, se2_s = run_simulation(200, pos1, pos2, n1, n2, b1, b2, maf, ld)
results = compute_coloc(b1_s, se1_s, b2_s, se2_s)

# --- 5. MAIN DISPLAY ---
st.title("FTO-IRX3 Locus Simulation Lab")
col_plot, col_res = st.columns([2, 1])

with col_plot:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    # Manhattan Proxy (-log10P)
    lp1 = (b1_s**2 / (se1_s**2 * 2)) / np.log(10)
    lp2 = (b2_s**2 / (se2_s**2 * 2)) / np.log(10)

    ax1.scatter(snps, lp1, c=lp1, cmap='winter', s=35, edgecolors='black', linewidth=0.2)
    ax1.set_ylabel("BMI GWAS (-log10P)")
    ax2.scatter(snps, lp2, c=lp2, cmap='autumn', s=35, edgecolors='black', linewidth=0.2)
    ax2.set_ylabel("IRX3 eQTL (-log10P)")
    st.pyplot(fig)

with col_res:
    st.subheader("Posteriors")
    for h, p in results.items():
        # Clamp value to [0,1] to satisfy st.progress()
        safe_p = float(np.clip(p, 0.0, 1.0))
        st.write(f"**{h}:** {safe_p:.1%}")
        st.progress(safe_p)

# --- 6. LAB MANUAL: WHY THE RESULT CHANGED ---
st.divider()
st.header("🔬 Lab Manual: Understanding the Sliders")



col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("### 1. Sample Size ($N$)")
    st.write("""
    **What it does:** Shrinks the Standard Error ($SE$).
    - $SE \approx 1/\sqrt{N}$. Larger samples make your peaks 'taller' and 'thinner'.
    - If $H_0$ is high despite a peak, your $N$ is too small for the model to overcome the prior penalty.
    """)

with col_b:
    st.markdown("### 2. LD Decay")
    st.write("""
    **What it does:** Controls the width of the association block.
    - **Low Decay:** High correlation between neighbors. The 'tower' is wide.
    - **Insight:** Broad blocks make it harder to distinguish between a shared SNP ($H_4$) and two independent SNPs ($H_3$).
    """)

with col_c:
    st.markdown("### 3. Allele Frequency (MAF)")
    st.write("""
    **What it does:** Modulates statistical precision.
    - Rare variants ($MAF < 0.05$) have high $SE$, making signals noisy.
    - Common variants ($MAF > 0.20$) allow for much more precise colocalization even with smaller sample sizes.
    """)

st.markdown("---")
st.info("**Key Insight for Students:** Colocalization is **direction-blind**. If you flip Beta 1 to negative, $H_4$ remains unchanged because the model only cares about the *location* of the association peak, not whether the effect is up or down.")
