import streamlit as st
import pandas as pd
import re
import sqlite3
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Descriptors import ExactMolWt
from streamlit_ketcher import st_ketcher

# =============================
# Constants
# =============================
PROTON = 1.007276
H2O = 18.01056
NH3 = 17.02655
DB = "aa_database.db"

# =============================
# Standard Residue Masses
# =============================
AA_MASS = {
    "A": 71.03711, "R": 156.10111, "N": 114.04293, "D": 115.02694,
    "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G": 57.02146,
    "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
    "M": 131.04049, "F": 147.06841, "P": 97.05276, "S": 87.03203,
    "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V": 99.06841,
}

# =============================
# Database
# =============================
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS custom_aa (
            code TEXT PRIMARY KEY,
            mass REAL
        )
    """)
    conn.commit()
    conn.close()

def load_custom():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM custom_aa", conn)
    conn.close()
    return df

def save_custom(code, mass):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("REPLACE INTO custom_aa VALUES (?, ?)",
              (code.upper(), mass))
    conn.commit()
    conn.close()

def delete_custom(code):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("DELETE FROM custom_aa WHERE code=?", (code,))
    conn.commit()
    conn.close()

# =============================
# Parse Sequence (A or (XXX))
# =============================
def parse_sequence(seq):
    tokens = re.findall(r'\([A-Za-z0-9]+\)|[A-Z]', seq)
    cleaned = []
    fasta_seq = ""

    for t in tokens:
        if t.startswith("("):
            cleaned.append(t[1:-1])
            fasta_seq += "X"  # placeholder for structure
        else:
            cleaned.append(t)
            fasta_seq += t

    return cleaned, fasta_seq

# =============================
# Fragmentation
# =============================
def generate_fragments(masses, include_losses):
    fragments = []

    running = 0
    for i in range(len(masses)-1):
        running += masses[i]
        b = running + PROTON
        fragments.append(("b"+str(i+1), b))
        if include_losses:
            fragments.append(("b"+str(i+1)+"-H2O", b-H2O))
            fragments.append(("b"+str(i+1)+"-NH3", b-NH3))

    running = 0
    for i in range(len(masses)-1):
        running += masses[-(i+1)]
        y = running + PROTON + H2O
        fragments.append(("y"+str(i+1), y))
        if include_losses:
            fragments.append(("y"+str(i+1)+"-H2O", y-H2O))
            fragments.append(("y"+str(i+1)+"-NH3", y-NH3))

    return fragments

def compute_mz(mass, z):
    return (mass + z*PROTON)/z

# =============================
# UI
# =============================
st.title("🔬 Peptide MS/MS Tool (Stable & Chemically Safe)")

init_db()
custom_df = load_custom()

# Sidebar
with st.sidebar.expander("Custom Amino Acid Manager", expanded=False):
    code = st.text_input("3-letter Code")
    mass = st.number_input("Residue Mass (monoisotopic)", format="%.6f")

    if st.button("Save Custom AA"):
        save_custom(code, mass)
        st.success("Saved.")

    st.subheader("Database")
    for _, row in custom_df.iterrows():
        col1, col2 = st.columns([4,1])
        col1.write(f"{row['code']} ({row['mass']})")
        if col2.button("Delete", key=row["code"]):
            delete_custom(row["code"])
            st.rerun()

# Main
sequence = st.text_input("Peptide Sequence (e.g., ACD(ORN)K)")
include_losses = st.checkbox("Include neutral losses (-H2O / -NH3)", value=True)

if sequence:
    tokens, fasta_seq = parse_sequence(sequence.upper())

    masses = []
    for t in tokens:
        if t in AA_MASS:
            masses.append(AA_MASS[t])
        else:
            row = custom_df[custom_df["code"] == t]
            if row.empty:
                st.error("Unknown residue.")
                st.stop()
            masses.append(row.iloc[0]["mass"])

    neutral_mass = sum(masses) + H2O

    st.subheader("Precursor m/z")
    st.dataframe(pd.DataFrame({
        "Charge":[1,2,3,4,5],
        "m/z":[round(compute_mz(neutral_mass,z),4) for z in range(1,6)]
    }))

    fragments = generate_fragments(masses, include_losses)

    st.subheader("Fragment Table")
    st.dataframe(pd.DataFrame([
        {
            "Fragment": name,
            "z=1": round(compute_mz(m,1),4),
            "z=2": round(compute_mz(m,2),4),
            "z=3": round(compute_mz(m,3),4),
            "z=4": round(compute_mz(m,4),4),
            "z=5": round(compute_mz(m,5),4),
        }
        for name,m in fragments
    ]))

    st.subheader("Simulated Spectrum")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[compute_mz(m,1) for _,m in fragments],
        y=[100 if "-" not in n else 40 for n,_ in fragments],
        text=[n for n,_ in fragments]
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Peptide Structure")
    mol = Chem.MolFromFASTA(fasta_seq)
    st.image(Draw.MolToImage(mol, size=(700,300)))
