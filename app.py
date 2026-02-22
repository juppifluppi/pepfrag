import streamlit as st
import sqlite3
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Descriptors import ExactMolWt
import io

# -------------------------
# Constants
# -------------------------
PROTON = 1.007276
WATER = 18.01056

DB_NAME = "amino_acids.db"

# -------------------------
# Database Setup
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS amino_acids (
            code TEXT PRIMARY KEY,
            name TEXT,
            smiles TEXT,
            mass REAL
        )
    """)
    conn.commit()
    conn.close()

def add_amino_acid(code, name, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mass = ExactMolWt(mol)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("REPLACE INTO amino_acids VALUES (?, ?, ?, ?)",
              (code.upper(), name, smiles, mass))
    conn.commit()
    conn.close()
    return True

def load_amino_acids():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM amino_acids", conn)
    conn.close()
    return df

# -------------------------
# Default 20 Amino Acids
# -------------------------
def load_default_amino_acids():
    defaults = {
        "A": ("Alanine", "CC(N)C(=O)O"),
        "G": ("Glycine", "NCC(=O)O"),
        "S": ("Serine", "N[C@@H](CO)C(=O)O"),
        "P": ("Proline", "N1CCCC1C(=O)O"),
        "V": ("Valine", "CC(C)C(N)C(=O)O"),
        "T": ("Threonine", "CC(O)C(N)C(=O)O"),
        "C": ("Cysteine", "N[C@@H](CS)C(=O)O"),
        "L": ("Leucine", "CC(C)CC(N)C(=O)O"),
        "I": ("Isoleucine", "CC[C@H](C)[C@H](N)C(=O)O"),
        "N": ("Asparagine", "NC(=O)C(N)C(=O)O"),
        "D": ("Aspartic Acid", "NC(CC(=O)O)C(=O)O"),
        "Q": ("Glutamine", "NC(=O)CCC(N)C(=O)O"),
        "K": ("Lysine", "NCCCC[C@H](N)C(=O)O"),
        "E": ("Glutamic Acid", "NC(CCC(=O)O)C(=O)O"),
        "M": ("Methionine", "CSCCC(N)C(=O)O"),
        "H": ("Histidine", "NC(Cc1c[nH]cn1)C(=O)O"),
        "F": ("Phenylalanine", "NC(Cc1ccccc1)C(=O)O"),
        "R": ("Arginine", "NC(CCCNC(N)=N)C(=O)O"),
        "Y": ("Tyrosine", "NC(Cc1ccc(O)cc1)C(=O)O"),
        "W": ("Tryptophan", "NC(Cc1c[nH]c2ccccc12)C(=O)O"),
    }

    for code, (name, smiles) in defaults.items():
        add_amino_acid(code, name, smiles)

# -------------------------
# Peptide Mass Calculation
# -------------------------
def calculate_peptide(sequence, aa_df):
    total = 0
    for aa in sequence:
        row = aa_df[aa_df["code"] == aa]
        if row.empty:
            return None
        total += row.iloc[0]["mass"]

    neutral = total + WATER
    protonated = neutral + PROTON
    return neutral, protonated

def calculate_fragments(sequence, aa_df):
    fragments = []
    for i in range(1, len(sequence)):
        b_seq = sequence[:i]
        y_seq = sequence[i:]

        b_mass = calculate_peptide(b_seq, aa_df)[1]
        y_mass = calculate_peptide(y_seq, aa_df)[1]

        fragments.append(("b"+str(i), b_mass))
        fragments.append(("y"+str(len(sequence)-i), y_mass))

    return pd.DataFrame(fragments, columns=["Fragment", "m/z (1+)"])

# -------------------------
# Build Peptide SMILES
# -------------------------
def sequence_to_smiles(sequence, aa_df):
    smiles_list = []
    for aa in sequence:
        row = aa_df[aa_df["code"] == aa]
        if row.empty:
            return None
        smiles_list.append(row.iloc[0]["smiles"])

    return ".".join(smiles_list)  # Simplified concatenation

# -------------------------
# Streamlit UI
# -------------------------
st.title("Peptide MS Fractionation Tool")

init_db()
load_default_amino_acids()
aa_df = load_amino_acids()

# Sidebar: Add Custom Amino Acid
st.sidebar.header("Add Custom Amino Acid")

code = st.sidebar.text_input("1-letter Code")
name = st.sidebar.text_input("Name")
smiles = st.sidebar.text_input("SMILES")

if st.sidebar.button("Add / Update"):
    if add_amino_acid(code, name, smiles):
        st.sidebar.success("Added successfully!")
    else:
        st.sidebar.error("Invalid SMILES")

st.sidebar.write("Current Amino Acids:")
st.sidebar.dataframe(aa_df[["code", "name"]])

# Main App
sequence = st.text_input("Enter Peptide Sequence").upper()

charge = st.slider("Charge State (z)", 1, 5, 1)

if sequence:
    result = calculate_peptide(sequence, aa_df)
    if result is None:
        st.error("Invalid amino acid in sequence.")
    else:
        neutral, protonated = result

        mz = (neutral + charge * PROTON) / charge

        st.subheader("Mass Results")
        st.write("Neutral Mass:", round(neutral, 4))
        st.write("[M+H]+:", round(protonated, 4))
        st.write("m/z (z={})".format(charge), round(mz, 4))
        st.write("Mol/2:", round(protonated/2, 4))
        st.write("Mol/3:", round(protonated/3, 4))

        # Fragments
        st.subheader("b/y Ion Series (1+)")
        frag_df = calculate_fragments(sequence, aa_df)
        st.dataframe(frag_df)

        # CSV Export
        csv = frag_df.to_csv(index=False)
        st.download_button(
            label="Download Fragment Table CSV",
            data=csv,
            file_name="fragments.csv",
            mime="text/csv"
        )

        # Structure Rendering
        st.subheader("Structure (Simplified View)")
        smiles = sequence_to_smiles(sequence, aa_df)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol)
                st.image(img)
