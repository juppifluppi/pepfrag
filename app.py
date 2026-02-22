import streamlit as st
import sqlite3
import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Descriptors import ExactMolWt
from streamlit_ketcher import st_ketcher

PROTON = 1.007276
DB_NAME = "amino_acids.db"

# -----------------------------
# Sequence Parser
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
# Database
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS amino_acids (
            code TEXT PRIMARY KEY,
            name TEXT,
            smiles TEXT
        )
    """)
    conn.commit()
    conn.close()

def load_amino_acids():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM amino_acids", conn)
    conn.close()
    return df

def save_amino_acid(code, name, smiles):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("REPLACE INTO amino_acids VALUES (?, ?, ?)",
              (code.upper(), name, smiles))
    conn.commit()
    conn.close()

# -----------------------------
# Amide Coupling Reaction
# -----------------------------
amide_rxn = AllChem.ReactionFromSmarts(
    "[C:1](=O)[O;H].[N;H2:2]>>[C:1](=O)[N:2]"
)

def couple_amino_acids(mol1, mol2):
    products = amide_rxn.RunReactants((mol1, mol2))
    if products:
        return products[0][0]
    return None

# -----------------------------
# Build Peptide
# -----------------------------
def build_peptide(tokens, aa_df):
    mol = None
    for token in tokens:
        row = aa_df[aa_df["code"] == token]
        if row.empty:
            return None
        aa_mol = Chem.MolFromSmiles(row.iloc[0]["smiles"])
        if mol is None:
            mol = aa_mol
        else:
            mol = couple_amino_acids(mol, aa_mol)
    return mol

# -----------------------------
# Fragment Generation
# -----------------------------
def generate_fragments(tokens, aa_df):
    fragments = []

    for i in range(1, len(tokens)):
        # b-ion
        b_tokens = tokens[:i]
        b_mol = build_peptide(b_tokens, aa_df)
        b_mass = ExactMolWt(b_mol)

        # y-ion
        y_tokens = tokens[i:]
        y_mol = build_peptide(y_tokens, aa_df)
        y_mass = ExactMolWt(y_mol)

        fragments.append(("b" + str(i), b_mass))
        fragments.append(("y" + str(len(tokens) - i), y_mass))

    return fragments

def compute_mz(mass, charge):
    return (mass + charge * PROTON) / charge

def build_fragment_table(fragments):
    rows = []
    for name, mass in fragments:
        row = {
            "Fragment": name,
            "z=1": round(compute_mz(mass, 1), 4),
            "z=2": round(compute_mz(mass, 2), 4),
            "z=3": round(compute_mz(mass, 3), 4),
            "z=4": round(compute_mz(mass, 4), 4),
            "z=5": round(compute_mz(mass, 5), 4),
        }
        rows.append(row)
    return pd.DataFrame(rows)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Peptide MS Fragmentation Tool")
st.markdown("Use format: `ACD(ORN)K`")

init_db()

# Sidebar: Custom AA
st.sidebar.header("Add Custom Amino Acid (3-letter code)")
code = st.sidebar.text_input("Code (e.g., ORN)")
name = st.sidebar.text_input("Name")
smiles = st_ketcher()

if st.sidebar.button("Save Custom AA"):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        save_amino_acid(code, name, smiles)
        st.sidebar.success("Saved successfully.")
    else:
        st.sidebar.error("Invalid structure.")

aa_df = load_amino_acids()

# Main sequence input
sequence = st.text_input("Peptide Sequence").upper()

if sequence:
    tokens = parse_sequence(sequence)
    mol = build_peptide(tokens, aa_df)

    if mol:
        neutral_mass = ExactMolWt(mol)

        st.subheader("Peptide Neutral Mass")
        st.write(round(neutral_mass, 4))

        st.subheader("Peptide Structure")
        img = Draw.MolToImage(mol)
        st.image(img)

        st.subheader("b / y Fragment Table (m/z)")
        fragments = generate_fragments(tokens, aa_df)
        frag_table = build_fragment_table(fragments)
        st.dataframe(frag_table)

        csv = frag_table.to_csv(index=False)
        st.download_button(
            "Download Fragment Table",
            csv,
            "fragments.csv",
            "text/csv"
        )
    else:
        st.error("Invalid sequence or undefined amino acid.")
