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
# Default Sidechains
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
# Parser
# =============================
def parse_sequence(seq):
    tokens = re.findall(r'\([A-Za-z0-9]+\)|[A-Z]', seq)
    return [t[1:-1] if t.startswith("(") else t for t in tokens]

# =============================
# Build Single Residue
# =============================
def build_residue(sidechain_smiles):
    backbone = Chem.MolFromSmiles("N[C@@H](*)C(=O)O")
    sidechain = Chem.MolFromSmiles(sidechain_smiles)

    rw = Chem.RWMol(backbone)
    star_idx = None

    for atom in rw.GetAtoms():
        if atom.GetSymbol() == "*":
            star_idx = atom.GetIdx()
            break

    offset = rw.GetNumAtoms()
    rw.InsertMol(sidechain)

    rw.AddBond(star_idx, offset, Chem.rdchem.BondType.SINGLE)
    rw.RemoveAtom(star_idx)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol

# =============================
# Couple Residues (Amide Bond)
# =============================
def couple_residues(m1, m2):
    rw = Chem.RWMol(Chem.CombineMols(m1, m2))
    Chem.SanitizeMol(rw)

    # Find C-terminal carbonyl carbon of m1
    carbonyl = None
    for atom in rw.GetAtoms():
        if atom.GetSymbol() == "C":
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == "O":
                    carbonyl = atom.GetIdx()
                    break

    # Find N-terminal nitrogen of m2
    nitrogen = None
    for atom in rw.GetAtoms():
        if atom.GetSymbol() == "N":
            nitrogen = atom.GetIdx()
            break

    rw.AddBond(carbonyl, nitrogen, Chem.rdchem.BondType.SINGLE)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol

# =============================
# Fragmentation (Mass-based)
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
st.title("🔬 Peptide MS/MS Research Tool")

init_db()
custom_df = load_custom()

with st.sidebar.expander("Custom Amino Acid Manager", expanded=False):
    st.markdown("Draw **sidechain only** (no backbone)")
    code = st.text_input("3-letter Code")
    sidechain = st_ketcher(height=250)

    if st.button("Save Custom AA"):
        save_custom(code, sidechain)
        st.success("Saved.")

    st.subheader("Database")
    for _, row in custom_df.iterrows():
        col1, col2 = st.columns([4,1])
        col1.markdown(f"**{row['code']}**  `{row['sidechain']}`")
        if col2.button("Delete", key=row["code"]):
            delete_custom(row["code"])
            st.rerun()

sequence = st.text_input("Peptide Sequence (e.g., ACD(ORN)K)")
include_losses = st.checkbox("Include neutral losses", value=True)

if sequence:
    tokens = parse_sequence(sequence.upper())
    residues = []
    masses = []

    for t in tokens:
        if t in DEFAULT_SIDECHAINS:
            sc = DEFAULT_SIDECHAINS[t]
        else:
            row = custom_df[custom_df["code"] == t]
            if row.empty:
                st.error("Unknown residue.")
                st.stop()
            sc = row.iloc[0]["sidechain"]

        res = build_residue(sc)
        residues.append(res)
        masses.append(ExactMolWt(res) - H2O)

    peptide = residues[0]
    for r in residues[1:]:
        peptide = couple_residues(peptide, r)

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

    st.subheader("Spectrum")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[compute_mz(m,1) for _,m in fragments],
        y=[100 if "-" not in n else 40 for n,_ in fragments],
        text=[n for n,_ in fragments]
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Peptide Structure")
    st.image(Draw.MolToImage(peptide, size=(700,300)))
