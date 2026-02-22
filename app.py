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

PTM_LIBRARY = {
    "Phospho (S,T,Y)": 79.966331,
    "Oxidation (M)": 15.994915,
    "Acetyl (Protein N-term)": 42.010565,
    "Carbamidomethyl (C)": 57.021464,
}

# =============================
# Database
# =============================
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("PRAGMA table_info(custom_aa)")
    columns = [col[1] for col in c.fetchall()]
    if columns and ("description" not in columns):
        c.execute("DROP TABLE IF EXISTS custom_aa")

    c.execute("""
        CREATE TABLE IF NOT EXISTS custom_aa (
            code TEXT PRIMARY KEY,
            smiles TEXT,
            mass REAL,
            description TEXT
        )
    """)
    conn.commit()
    conn.close()

def load_custom():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM custom_aa", conn)
    conn.close()
    return df

def save_custom(code, structure, description):
    if not code or not structure:
        return False

    mol = None

    if isinstance(structure, dict):
        smiles = structure.get("smiles", "")
        molfile = structure.get("molfile", "")
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None and molfile:
            mol = Chem.MolFromMolBlock(molfile)

    elif isinstance(structure, str):
        mol = Chem.MolFromMolBlock(structure)
        if mol is None:
            mol = Chem.MolFromSmiles(structure)

    if mol is None:
        return False

    try:
        Chem.SanitizeMol(mol)
        mass = ExactMolWt(mol)
        smiles = Chem.MolToSmiles(mol)

        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute(
            "REPLACE INTO custom_aa VALUES (?, ?, ?, ?)",
            (code.upper(), smiles, mass, description)
        )
        conn.commit()
        conn.close()
        return True
    except:
        return False

def delete_custom(code):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("DELETE FROM custom_aa WHERE code=?", (code,))
    conn.commit()
    conn.close()

# =============================
# Sequence Parser
# =============================
def parse_sequence(seq):
    tokens = re.findall(r'\([A-Za-z0-9]+\)|[A-Z]', seq)
    return [t[1:-1] if t.startswith("(") else t for t in tokens]

# =============================
# Visual Peptide Builder
# =============================
# =============================
# Visual Peptide Builder
# =============================
def build_visual_peptide(tokens, custom_df):

    backbone = Chem.RWMol()
    prev_c = None

    for t in tokens:

        # Backbone atoms
        n = backbone.AddAtom(Chem.Atom("N"))
        ca = backbone.AddAtom(Chem.Atom("C"))
        c = backbone.AddAtom(Chem.Atom("C"))
        o = backbone.AddAtom(Chem.Atom("O"))

        backbone.AddBond(n, ca, Chem.rdchem.BondType.SINGLE)
        backbone.AddBond(ca, c, Chem.rdchem.BondType.SINGLE)
        backbone.AddBond(c, o, Chem.rdchem.BondType.DOUBLE)

        if prev_c is not None:
            backbone.AddBond(prev_c, n, Chem.rdchem.BondType.SINGLE)

        prev_c = c

        if t in AA_MASS:
            # Standard residue → simple methyl sidechain
            sc = backbone.AddAtom(Chem.Atom("C"))
            backbone.AddBond(ca, sc, Chem.rdchem.BondType.SINGLE)

        else:
            # Custom residue → cyclobutane marker
            marker = Chem.MolFromSmiles("C1CCC1")
            offset = backbone.GetNumAtoms()
            backbone.InsertMol(marker)
            backbone.AddBond(ca, offset, Chem.rdchem.BondType.SINGLE)

    mol = backbone.GetMol()
    Chem.SanitizeMol(mol)
    return mol

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
# App Start
# =============================
st.title("PepFrag peptide MS tool")

init_db()
custom_df = load_custom()

# -----------------------------
# Add Custom AA
# -----------------------------

with st.expander("Draw and save custom amino acid", expanded=False):
    
    st.write("After drawing and providing a 3-letter code, click on 'Apply' and then on 'Save AA' to permanently add a custom AA to the database. Adding a description helps when searching the database.")
    
    code = st.text_input("3-letter code")
    description = st.text_input("Description (optional)")
    structure = st_ketcher(height=500)

    if st.button("Save AA"):
        if save_custom(code, structure, description):
            st.success("Custom amino acid saved.")
            st.rerun()
        else:
            st.error("Invalid structure or no structure detected.")

# -----------------------------
# Sidebar Database
# -----------------------------
with st.sidebar:
    st.subheader("Custom AA database")

    # --- Search ---
    search_term = st.text_input("Search custom AA")

    if search_term:
        filtered_df = custom_df[
            custom_df["code"].str.contains(search_term, case=False, na=False) |
            custom_df["description"].fillna("").str.contains(search_term, case=False, na=False)
        ]
    else:
        filtered_df = custom_df

    st.write(f"Showing {len(filtered_df)} entries")

    # --- Display entries ---
    for _, row in filtered_df.iterrows():
        st.markdown(f"### {row['code']}")
        st.write(f"Mass: {round(row['mass'],4)}")

        if row["description"]:
            st.write(row["description"])

        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            st.image(Draw.MolToImage(mol, size=(250,150)))

        if st.button("Delete", key=row["code"]):
            delete_custom(row["code"])
            st.rerun()

# -----------------------------
# Peptide Analysis
# -----------------------------
sequence = st.text_input("Peptide sequence (use 1-letter codes for standard and 3-letter codes for custom AAs from the database, e.g. ACD(ORN)K)")
include_losses = st.checkbox("Include neutral losses (-H2O / -NH3)", value=True)

st.write("PTM toggles:")
selected_ptms = [ptm for ptm in PTM_LIBRARY if st.checkbox(ptm)]

if sequence:
    tokens = parse_sequence(sequence.upper())

    masses = []
    for t in tokens:
        if t in AA_MASS:
            masses.append(AA_MASS[t])
        else:
            row = custom_df[custom_df["code"] == t]
            if row.empty:
                st.error(f"Unknown residue: {t}")
                st.stop()
            masses.append(row.iloc[0]["mass"])

    total_ptm_shift = sum(PTM_LIBRARY[p] for p in selected_ptms)
    neutral_mass = sum(masses) + H2O + total_ptm_shift

    st.subheader("Precursor m/z")
    st.dataframe(pd.DataFrame({
        "Charge":[1,2,3,4,5],
        "m/z":[round(compute_mz(neutral_mass,z),4) for z in range(1,6)]
    }))

    fragments = generate_fragments(masses, include_losses)

    st.subheader("Fragment table")
    st.dataframe(pd.DataFrame([
        {
            "Fragment": name,
            "z=1": round(compute_mz(m + total_ptm_shift,1),4),
            "z=2": round(compute_mz(m + total_ptm_shift,2),4),
            "z=3": round(compute_mz(m + total_ptm_shift,3),4),
            "z=4": round(compute_mz(m + total_ptm_shift,4),4),
            "z=5": round(compute_mz(m + total_ptm_shift,5),4),
        }
        for name,m in fragments
    ]))

    st.subheader("MS/MS spectrum")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[compute_mz(m + total_ptm_shift,1) for _,m in fragments],
        y=[100 if "-" not in n else 40 for n,_ in fragments],
        text=[n for n,_ in fragments]
    ))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Peptide structure")
    st.write("Custom AAs are shown as cyclobutane sidechains")
    visual_mol = build_visual_peptide(tokens, custom_df)
    st.image(Draw.MolToImage(visual_mol, size=(700,300)))
