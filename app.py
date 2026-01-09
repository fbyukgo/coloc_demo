import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CORE WRAPPER FUNCTION ---
def run_coloc_simulation(n_snps, causal_1, causal_2, p12_prior, noise_level=0.01):
    """
    Simulates GWAS and eQTL data and calculates coloc posterior probabilities.
    """
    # Simulate SNP indices
    snps = np.arange(n_snps)

    # Simulate Traits (Trait 1: eQTL, Trait 2: GWAS)
    def simulate_trait(causal_idx, effect_size):
        beta = np.random.normal(0, noise_level, n_snps)
        beta[causal_idx] = effect_size
        se = np.random.uniform(0.04, 0.06, n_snps)
        return beta, se

    b1, se1 = simulate_trait(causal_1, 0.5)
    b2, se2 = simulate_trait(causal_2, -0.4) # Opposite direction demo

    # Calculate ABFs (Wakefield's Formula)
    def get_abf(beta, se, prior_var=0.15**2):
        v = se**2
        z_sq = (beta / se)**2
        r = prior_var / (prior_var + v)
        return np.sqrt(1 - r) * np.exp((z_sq * r) / 2)

    abf1 = get_abf(b1, se1)
    abf2 = get_abf(b2, se2)

    # Coloc Hypothesis Math
    p1, p2 = 1e-4, 1e-4
    p12 = p12_prior # We tweak this in the dashboard

    s1, s2 = np.sum(abf1), np.sum(abf2)
    s12 = np.sum(abf1 * abf2)

    h0 = 1
    h1 = p1 * s1
    h2 = p2 * s2
    h3 = p1 * p2 * (s1 * s2 - s12)
    h4 = p12 * s12

    total = h0 + h1 + h2 + h3 + h4
    probs = {"H0": h0/total, "H1": h1/total, "H2": h2/total, "H3": h3/total, "H4": h4/total}

    return snps, b1, b2, probs

# --- 2. STREAMLIT UI ---
st.title("🧬 Coloc Sandbox: The ABF Engine")
st.sidebar.header("Tweak the Locus")

n_snps = st.sidebar.slider("Number of SNPs", 50, 500, 100)
c1 = st.sidebar.slider("Causal SNP (Trait 1)", 0, n_snps-1, int(n_snps/2))
c2 = st.sidebar.slider("Causal SNP (Trait 2)", 0, n_snps-1, int(n_snps/2))
p12_log = st.sidebar.slider("Prior for H4 (log10)", -10.0, -1.0, -5.0)

# Run logic
snps, b1, b2, probs = run_coloc_simulation(n_snps, c1, c2, 10**p12_log)

# --- 3. VISUALIZATION ---
st.subheader("Regional Association Plot")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

ax1.scatter(snps, -np.log10(np.exp(-b1**2/0.002)), c='blue', alpha=0.6) # Proxy for p-values
ax1.set_ylabel("Trait 1 Signal")
ax1.axvline(c1, color='blue', linestyle='--', alpha=0.3)

ax2.scatter(snps, -np.log10(np.exp(-b2**2/0.002)), c='red', alpha=0.6)
ax2.set_ylabel("Trait 2 Signal")
ax2.axvline(c2, color='red', linestyle='--', alpha=0.3)
ax2.set_xlabel("SNP Position")

plt.tight_layout()
st.pyplot(fig)

# --- 4. RESULTS TABLE ---
st.subheader("Posterior Probabilities")
res_df = pd.DataFrame([probs])
st.table(res_df.style.highlight_max(axis=1, color='lightgreen'))

if probs["H4"] > 0.8:
    st.success("✅ COLOCALIZATION: High evidence for a shared causal variant.")
elif probs["H3"] > 0.8:
    st.warning("⚠️ LINKAGE: Two distinct variants are driving the traits.")
else:
    st.info("No strong evidence for colocalization yet.")
