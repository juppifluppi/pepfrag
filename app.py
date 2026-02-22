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
# Default Residues (Sidechains)
# =============================
DEFAULT_SIDECHAINS = {
    "A": "C",
    "R": "CCCNC(N)=N",
    "N": "CC(=O)N",
    "D": "CC(=O)O",
    "C": "CS",
    "E": "CCC(=O)O",
    "Q": "CCC(=O)N",
    "G": "[H]",
    "H": "Cc1c[nH]cn1",
    "I": "C(C)CC",
    "L": "CC(C)C",
    "K": "CCCCN",
    "M": "CCSC",
    "F": "Cc1ccccc1",
    "P": "CCC",
    "S": "CO",
    "T": "C(O)C",
    "W": "Cc1c[nH]c2ccccc12",
    "Y": "Cc1ccc(O)cc1",
    "V": "C(C)C",
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
            sidechain TEXT
        )
    """)
    conn.commit()
    conn.close()

def load_custom():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM custom_aa", conn)
    conn.close()
    return df

def save_custom(code, sidechain):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("REPLACE INTO custom_aa VALUES (?, ?)",
              (code.upper(), sidechain))
    conn.commit()
    conn.close()

def delete_custom(code):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("DELETE FROM custom_aa WHERE code=?", (code,))
    conn.commit()
    conn.close()

# =============================
# Build Residue from Sidechain
# =============================
def build_residue(sidechain_smiles):
    backbone = Chem.MolFromSmiles("N[C@@H]([*:1])C(=O)O")
    sidechain = Chem.MolFromSmiles(f"[*:1]{sidechain_smiles}")
    combo = Chem.CombineMols(backbone, sidechain)

    emol = Chem.EditableMol(combo)
    emol.AddBond(1, backbone.GetNumAtoms(), Chem.rdchem.BondType.SINGLE)
    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    return mol

# =============================
# Parser
# =============================
def parse_sequence(seq):
    tokens = re.findall(r'\([A-Za-z0-9]+\)|[A-Z]', seq)
    return [t[1:-1] if t.startswith("(") else t for t in tokens]

# =============================
# UI
# =============================
st.title("🔬 Peptide MS Tool (Sidechain-Based Custom AA)")

init_db()
custom_df = load_custom()

# Sidebar
with st.sidebar.expander("Custom Amino Acid Manager", expanded=False):

    st.markdown("Draw **sidechain only** (no backbone).")

    code = st.text_input("3-letter Code")
    sidechain = st_ketcher(height=250)

    if st.button("Save Custom AA"):
        save_custom(code, sidechain)
        st.success("Saved.")

    st.subheader("Database")

    for _, row in custom_df.iterrows():
        col1, col2 = st.columns([3,1])
        col1.markdown(f"**{row['code']}**  \n`{row['sidechain']}`")
        if col2.button("Delete", key=row["code"]):
            delete_custom(row["code"])
            st.experimental_rerun()

# Main
sequence = st.text_input("Peptide Sequence (e.g., ACD(ORN)K)")

if sequence:
    tokens = parse_sequence(sequence.upper())

    residues = []
    for t in tokens:
        if t in DEFAULT_SIDECHAINS:
            residues.append(build_residue(DEFAULT_SIDECHAINS[t]))
        else:
            row = custom_df[custom_df["code"] == t]
            if row.empty:
                st.error("Unknown residue.")
                st.stop()
            residues.append(build_residue(row.iloc[0]["sidechain"]))

    # Combine residues linearly (simplified)
    mol = residues[0]
    for r in residues[1:]:
        mol = Chem.CombineMols(mol, r)

    mass = ExactMolWt(mol)

    st.subheader("Precursor m/z")
    precursor = pd.DataFrame({
        "Charge":[1,2,3,4,5],
        "m/z":[round((mass+z*PROTON)/z,4) for z in range(1,6)]
    })
    st.dataframe(precursor)

    st.subheader("Full Peptide Structure")
    img = Draw.MolToImage(mol, size=(700,300))
    st.image(img)
