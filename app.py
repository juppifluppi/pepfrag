import streamlit as st
import pandas as pd
import re
import sqlite3
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Descriptors import ExactMolWt
from streamlit_ketcher import st_ketcher

# -----------------------------
# Constants
# -----------------------------
PROTON = 1.007276
H2O = 18.01056
NH3 = 17.02655
DB = "aa_database.db"

# -----------------------------
# Default AA Residue Masses
# -----------------------------
AA_MASS = {
    "A": 71.03711, "R": 156.10111, "N": 114.04293, "D": 115.02694,
    "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G": 57.02146,
    "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
    "M": 131.04049, "F": 147.06841, "P": 97.05276, "S": 87.03203,
    "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V": 99.06841,
}

# -----------------------------
# PTM Library
# -----------------------------
PTM_LIBRARY = {
    "Phospho (+79.9663)": 79.966331,
    "Oxidation (+15.9949)": 15.994915,
    "Acetyl N-term (+42.0106)": 42.010565,
    "Amidation C-term (-0.9840)": -0.984016,
}

# -----------------------------
# Database
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS custom_aa (
            code TEXT PRIMARY KEY,
            smiles TEXT,
            mass REAL
        )
    """)
    conn.commit()
    conn.close()

def load_custom_aa():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM custom_aa", conn)
    conn.close()
    return df

def save_custom_aa(code, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mass = ExactMolWt(mol)
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("REPLACE INTO custom_aa VALUES (?, ?, ?)",
                  (code.upper(), smiles, mass))
        conn.commit()
        conn.close()
        return True
    return False

# -----------------------------
# Parse Sequence (A or (XXX))
# -----------------------------
def parse_sequence(seq):
    tokens = re.findall(r'\([A-Za-z0-9]+\)|[A-Z]', seq)
    cleaned = []
    for t in tokens:
        if t.startswith("("):
            cleaned.append(t[1:-1])
        else:
            cleaned.append(t)
    return cleaned

# -----------------------------
# Mass Calculation
# -----------------------------
def calculate_peptide_mass(tokens, ptms, custom_df):
    masses = []
    for t in tokens:
        if t in AA_MASS:
            masses.append(AA_MASS[t])
        else:
            row = custom_df[custom_df["code"] == t]
            if row.empty:
                return None
            masses.append(row.iloc[0]["mass"])

    # Add PTMs
    for ptm in ptms:
        shift = PTM_LIBRARY[ptm]
        if "N-term" in ptm:
            masses[0] += shift
        elif "C-term" in ptm:
            masses[-1] += shift
        else:
            masses = [m + shift for m in masses]

    neutral = sum(masses) + H2O
    return neutral, masses

# -----------------------------
# Fragmentation
# -----------------------------
def generate_fragments(masses):
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

def compute_mz(mass, z):
    return (mass + z*PROTON) / z

def build_table(fragments):
    rows = []
    for name, mass in fragments:
        rows.append({
            "Fragment": name,
            "z=1": round(compute_mz(mass,1),4),
            "z=2": round(compute_mz(mass,2),4),
            "z=3": round(compute_mz(mass,3),4),
            "z=4": round(compute_mz(mass,4),4),
            "z=5": round(compute_mz(mass,5),4),
        })
    return pd.DataFrame(rows)

# -----------------------------
# Spectrum
# -----------------------------
def plot_spectrum(fragments):
    mz = [compute_mz(m,1) for _,m in fragments]
    intensity = [100 if "-" not in n else 40 for n,_ in fragments]
    labels = [n for n,_ in fragments]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=mz, y=intensity, text=labels))
    fig.update_layout(
        title="Simulated MS/MS Spectrum (1+)",
        xaxis_title="m/z",
        yaxis_title="Relative Intensity",
        showlegend=False
    )
    return fig

# -----------------------------
# Draw Peptide Backbone (Simplified)
# -----------------------------
def build_backbone_smiles(length):
    backbone = "N"
    for _ in range(length):
        backbone += "C(=O)N"
    backbone += "C(=O)O"
    return backbone

# -----------------------------
# UI
# -----------------------------
st.title("🔬 Advanced Peptide MS/MS Research Tool")

init_db()
custom_df = load_custom_aa()

sequence = st.text_input("Peptide Sequence (e.g., ACD(ORN)K)")
ptms = st.multiselect("Select PTMs", list(PTM_LIBRARY.keys()))

if sequence:
    tokens = parse_sequence(sequence.upper())
    result = calculate_peptide_mass(tokens, ptms, custom_df)

    if result:
        neutral, masses = result

        st.subheader("Precursor m/z ([M+zH]z+)")
        precursor_table = pd.DataFrame({
            "Charge": [1,2,3,4,5],
            "m/z": [round(compute_mz(neutral,z),4) for z in range(1,6)]
        })
        st.dataframe(precursor_table)

        fragments = generate_fragments(masses)

        st.subheader("b / y Fragment Table")
        frag_table = build_table(fragments)
        st.dataframe(frag_table)

        st.download_button(
            "Download Fragment CSV",
            frag_table.to_csv(index=False),
            "fragments.csv"
        )

        st.subheader("Simulated MS/MS Spectrum")
        st.plotly_chart(plot_spectrum(fragments), use_container_width=True)

        st.subheader("Full Peptide Structure (Backbone Representation)")
        backbone = build_backbone_smiles(len(tokens))
        mol = Chem.MolFromSmiles(backbone)
        img = Draw.MolToImage(mol, size=(600,200))
        st.image(img)

    else:
        st.error("Unknown residue detected.")

# -----------------------------
# Custom AA Section
# -----------------------------
st.sidebar.header("Add Custom Amino Acid (3-letter code)")
code = st.sidebar.text_input("Code (e.g., ORN)")
smiles = st_ketcher()

if st.sidebar.button("Save Custom AA"):
    if save_custom_aa(code, smiles):
        st.sidebar.success("Custom AA saved.")
    else:
        st.sidebar.error("Invalid structure.")
