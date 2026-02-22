import streamlit as st
import pandas as pd
import re
import plotly.graph_objects as go

PROTON = 1.007276
H2O = 18.01056
NH3 = 17.02655

# -----------------------------
# Standard Amino Acid Monoisotopic Masses
# -----------------------------
AA_MASS = {
    "A": 71.03711,
    "R": 156.10111,
    "N": 114.04293,
    "D": 115.02694,
    "C": 103.00919,
    "E": 129.04259,
    "Q": 128.05858,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "L": 113.08406,
    "K": 128.09496,
    "M": 131.04049,
    "F": 147.06841,
    "P": 97.05276,
    "S": 87.03203,
    "T": 101.04768,
    "W": 186.07931,
    "Y": 163.06333,
    "V": 99.06841,
}

# -----------------------------
# PTM Library
# -----------------------------
PTM_LIBRARY = {
    "Phospho": 79.966331,
    "Oxidation": 15.994915,
    "Acetyl (N-term)": 42.010565,
    "Amidation (C-term)": -0.984016,
}

# -----------------------------
# Sequence Parser
# -----------------------------
def parse_sequence(seq):
    return re.findall(r'[A-Z]', seq)

# -----------------------------
# Apply PTMs
# -----------------------------
def apply_ptms(masses, ptm_selection):
    for ptm in ptm_selection:
        if "N-term" in ptm:
            masses[0] += PTM_LIBRARY[ptm]
        elif "C-term" in ptm:
            masses[-1] += PTM_LIBRARY[ptm]
        else:
            # apply to all possible residues (simplified)
            for i in range(len(masses)):
                masses[i] += PTM_LIBRARY[ptm]
    return masses

# -----------------------------
# Fragment Engine
# -----------------------------
def generate_fragments(sequence, ptms):
    residues = parse_sequence(sequence)

    if not all(r in AA_MASS for r in residues):
        return None

    masses = [AA_MASS[r] for r in residues]
    masses = apply_ptms(masses, ptms)

    fragments = []

    # b-ions
    running = 0
    for i in range(len(masses)-1):
        running += masses[i]
        b_mass = running + PROTON
        fragments.append(("b"+str(i+1), b_mass))

        fragments.append(("b"+str(i+1)+"-H2O", b_mass - H2O))
        fragments.append(("b"+str(i+1)+"-NH3", b_mass - NH3))

    # y-ions
    running = 0
    for i in range(len(masses)-1):
        running += masses[-(i+1)]
        y_mass = running + PROTON + H2O
        fragments.append(("y"+str(i+1), y_mass))

        fragments.append(("y"+str(i+1)+"-H2O", y_mass - H2O))
        fragments.append(("y"+str(i+1)+"-NH3", y_mass - NH3))

    return fragments

# -----------------------------
# Charge State m/z
# -----------------------------
def compute_mz(mass, z):
    return (mass + z*PROTON) / z

# -----------------------------
# Build Table
# -----------------------------
def build_table(fragments):
    rows = []
    for name, mass in fragments:
        row = {
            "Fragment": name,
            "z=1": round(compute_mz(mass,1),4),
            "z=2": round(compute_mz(mass,2),4),
            "z=3": round(compute_mz(mass,3),4),
            "z=4": round(compute_mz(mass,4),4),
            "z=5": round(compute_mz(mass,5),4),
        }
        rows.append(row)
    return pd.DataFrame(rows)

# -----------------------------
# Spectrum Plot
# -----------------------------
def plot_spectrum(fragments):
    mz = [compute_mz(mass,1) for _, mass in fragments]
    intensity = [100 if "-" not in name else 40 for name,_ in fragments]
    labels = [name for name,_ in fragments]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=mz, y=intensity, text=labels))
    fig.update_layout(
        title="Simulated MS/MS Spectrum (1+)",
        xaxis_title="m/z",
        yaxis_title="Intensity",
        showlegend=False
    )
    return fig

# -----------------------------
# UI
# -----------------------------
st.title("🔬 Research-Grade Peptide MS/MS Engine")

sequence = st.text_input("Peptide Sequence (1-letter code)")

st.subheader("🧬 Select PTMs")
ptms = st.multiselect("Choose Modifications", list(PTM_LIBRARY.keys()))

if sequence:
    fragments = generate_fragments(sequence.upper(), ptms)

    if fragments:
        st.subheader("📊 Fragment Table")
        table = build_table(fragments)
        st.dataframe(table)

        st.download_button(
            "Download CSV",
            table.to_csv(index=False),
            "fragments.csv"
        )

        st.subheader("📈 Simulated Spectrum")
        fig = plot_spectrum(fragments)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Invalid sequence.")
