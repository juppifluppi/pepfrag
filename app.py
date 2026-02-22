import streamlit as st
import pandas as pd
import re
import sqlite3
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
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
# Default AA SMILES (L-form)
# -----------------------------
DEFAULT_AA = {
    "A": "N[C@@H](C)C(=O)O",
    "R": "N[C@@H](CCCNC(N)=N)C(=O)O",
    "N": "N[C@@H](CC(=O)N)C(=O)O",
    "D": "N[C@@H](CC(=O)O)C(=O)O",
    "C": "N[C@@H](CS)C(=O)O",
    "E": "N[C@@H](CCC(=O)O)C(=O)O",
    "Q": "N[C@@H](CCC(=O)N)C(=O)O",
    "G": "NCC(=O)O",
    "H": "N[C@@H](Cc1c[nH]cn1)C(=O)O",
    "I": "N[C@@H](C(C)CC)C(=O)O",
    "L": "N[C@@H](CC(C)C)C(=O)O",
    "K": "N[C@@H](CCCCN)C(=O)O",
    "M": "N[C@@H](CCSC)C(=O)O",
    "F": "N[C@@H](Cc1ccccc1)C(=O)O",
    "P": "N1[C@@H](CCC1)C(=O)O",
    "S": "N[C@@H](CO)C(=O)O",
    "T": "N[C@@H](C(O)C)C(=O)O",
    "W": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
    "Y": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "V": "N[C@@H](C(C)C)C(=O)O",
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
            smiles TEXT
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
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("REPLACE INTO custom_aa VALUES (?, ?)",
                  (code.upper(), smiles))
        conn.commit()
        conn.close()
        return True
    return False

# -----------------------------
# Sequence Parser
# -----------------------------
def parse_sequence(seq):
    tokens = re.findall(r'\([A-Za-z0-9]+\)|[A-Z]', seq)
    return [t[1:-1] if t.startswith("(") else t for t in tokens]

# -----------------------------
# Amide Coupling
# -----------------------------
amide_rxn = AllChem.ReactionFromSmarts(
    "[C:1](=O)[O;H].[N;H2:2]>>[C:1](=O)[N:2]"
)

def couple(mol1, mol2):
    products = amide_rxn.RunReactants((mol1, mol2))
    if products:
        return products[0][0]
    return None

# -----------------------------
# Build Peptide Molecule
# -----------------------------
def build_peptide(tokens, custom_df):
    mol = None
    for t in tokens:
        if t in DEFAULT_AA:
            smiles = DEFAULT_AA[t]
        else:
            row = custom_df[custom_df["code"] == t]
            if row.empty:
                return None
            smiles = row.iloc[0]["smiles"]

        aa_mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            mol = aa_mol
        else:
            mol = couple(mol, aa_mol)

    return mol

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔬 Advanced Peptide MS Research Tool")

init_db()
custom_df = load_custom_aa()

# Sidebar
st.sidebar.header("Add Custom Amino Acid (3-letter code)")
code = st.sidebar.text_input("Code (e.g., ORN)")
smiles = st_ketcher(height=300)

if st.sidebar.button("Save Custom AA"):
    if save_custom_aa(code, smiles):
        st.sidebar.success("Saved successfully.")
    else:
        st.sidebar.error("Invalid structure.")

st.sidebar.subheader("Custom AA Database")
st.sidebar.dataframe(custom_df)

# Main
sequence = st.text_input("Peptide Sequence (e.g., ACD(ORN)K)")

if sequence:
    tokens = parse_sequence(sequence.upper())
    mol = build_peptide(tokens, custom_df)

    if mol:
        mass = ExactMolWt(mol)

        st.subheader("Precursor m/z")
        precursor_df = pd.DataFrame({
            "Charge": [1,2,3,4,5],
            "m/z": [round((mass + z*PROTON)/z,4) for z in range(1,6)]
        })
        st.dataframe(precursor_df)

        st.subheader("Full Peptide Structure")
        img = Draw.MolToImage(mol, size=(700,300))
        st.image(img)

    else:
        st.error("Invalid sequence or unknown residue.")
