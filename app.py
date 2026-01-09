import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="FTO-IRX3 Locus Sandbox")

# --- 1. Simulation Engine ---
def simulate_locus(n_snps, causal_1, causal_2, power_1, power_2, ld_decay):
    snps = np.arange(n_snps)

    def generate_signal(causal_idx, scale, power):
        base = np.zeros(n_snps)
        base[causal_idx] = scale * power
        for i in range(n_snps):
            dist = abs(i - causal_idx)
            ld = np.exp(-dist * ld_decay)
            base[i] = base[causal_idx] * ld
        noise = np.random.normal(0, 0.05, n_snps)
        beta = base + noise
        se = np.random.uniform(0.04, 0.06, n_snps)
        return beta, se

    b1, se1 = generate_signal(causal_1, 0.75, power_1)
    b2, se2 = generate_signal(causal_2, 0.65, power_2)
    return snps, b1, b2, se1, se2

# --- 2. Interface ---
st.title("FTO Locus Sandbox")
st.markdown("""
**The Problem:** GWAS says the *FTO* region is associated with BMI. But is *FTO* actually the causal gene, or is it regulating something else like *IRX3*?
We use **coloc** to figure out if these two association towers are actually the same signal.
""")

with st.sidebar:
    st.header("Parameters")
    st.markdown("### Locus Architecture")
    ld_strength = st.select_slider("LD Decay (Block size)", options=[0.5, 0.2, 0.1, 0.05], value=0.1)

    st.markdown("### Causal Variants")
    c1 = st.slider("BMI GWAS Peak", 0, 200, 100)
    c2 = st.slider("IRX3 eQTL Peak", 0, 200, 100)

    st.markdown("### Data Quality")
    pow1 = st.slider("GWAS Power", 0.0, 1.0, 0.9)
    pow2 = st.slider("eQTL Power", 0.0, 1.0, 0.7)

# Run
snps, b1, b2, se1, se2 = simulate_locus(200, c1, c2, pow1, pow2, ld_strength)

# --- 3. The Math ---
def get_abf(beta, se):
    v = se**2
    w = 0.15**2
    r = w / (w + v)
    z_sq = (beta/se)**2
    return np.sqrt(1-r) * np.exp(z_sq * r / 2)

abf1 = get_abf(b1, se1)
abf2 = get_abf(b2, se2)

p1, p2, p12 = 1e-4, 1e-4, 1e-5
sum1, sum2, sum12 = np.sum(abf1), np.sum(abf2), np.sum(abf1 * abf2)

h0 = 1
h1 = p1 * sum1
h2 = p2 * sum2
h3 = p1 * p2 * (sum1 * sum2 - sum12)
h4 = p12 * sum12
total = h0 + h1 + h2 + h3 + h4
probs = {"H0": h0/total, "H1": h1/total, "H2": h2/total, "H3": h3/total, "H4": h4/total}

# --- 4. Plotting ---
col1, col2 = st.columns([2, 1])

with col1:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    lp1 = (b1**2 / (se1**2 * 2)) / np.log(10)
    lp2 = (b2**2 / (se2**2 * 2)) / np.log(10)

    ax1.scatter(snps, lp1, c=lp1, cmap='winter', s=40)
    ax1.set_ylabel("-log10(P) [BMI GWAS]")

    ax2.scatter(snps, lp2, c=lp2, cmap='autumn', s=40)
    ax2.set_ylabel("-log10(P) [IRX3 eQTL]")
    ax2.set_xlabel("Genomic Position")
    st.pyplot(fig)

with col2:
    st.subheader("Posteriors")
    for h, p in probs.items():
        st.write(f"**{h}:** {p:.2%}")
        st.progress(p)

    if probs['H4'] > 0.8:
        st.success("Colocalized. Looks like the same variant is driving both traits.")
    elif probs['H3'] > 0.8:
        st.warning("Linkage. Two different SNPs are at play here. Not a match.")

# --- 5. Reality Check Section ---
st.divider()
st.header("⚠️ The Reality Check: What Coloc Doesn't Tell You")

st.markdown("""
### 1. Directionality is Invisible
A high $H_4$ probability only tells you that the signals **share a variant**. It doesn't tell you if Trait A *causes* Trait B.
- **Scenario:** The variant could increase *IRX3* expression and increase BMI.
- **Scenario:** The variant could *decrease* *IRX3* expression and still increase BMI.
- **Coloc result:** Both scenarios give you the same high $H_4$. To find the direction, you'd need to look at the **Betas** (slopes) and do something like Mendelian Randomization.

### 2. The "One Causal Variant" Assumption
Coloc assumes there is **at most one** causal variant in the region for each trait.
- If your locus is messy and actually has 2 or 3 different variants driving the same gene, the standard ABF method can get confused.
- It might give you a low $H_4$ even if there is a shared signal hidden in the mess.

### 3. The $P_{12}$ Prior (The Skepticism Dial)
We set a "prior" ($p_{12} = 10^{-5}$) which is our way of telling the model how often we *expect* signals to be shared.
- If you change this to be very small, you are being a **skeptic**. The model will need a massive, perfect overlap to give you a high $H_4$.
- If you make it larger, you are being an **optimist**. The model will be more likely to claim colocalization even with noisier data.
""")
