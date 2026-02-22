import streamlit as st
import pandas as pd
import re
import sqlite3
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Descriptors import ExactMolWt

# =============================
# Constants
# =============================
PROTON = 1.007276
H2O = 18.01056
NH3 = 17.02655
DB = "aa_database.db"

# =============================
# Default Sidechains (no dummy)
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

    c.execute("PRAGMA table_info(custom_aa)")
    columns = [col[1] for col in c.fetchall()]
    if "sidechain" not in columns:
        c.execute("DROP TABLE IF EXISTS custom_aa")

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
# Build Residue Safely
# =============================
def build_residue(sidechain_smiles):
    backbone = Chem.MolFromSmiles("N[C@@H]([*:1])C(=O)O")

    # attach dummy atom to first atom of sidechain
    sidechain = Chem.MolFromSmiles(sidechain_smiles)
    rw = Chem.RWMol(sidechain)
    dummy = Chem.Atom("*")
    dummy_idx = rw.AddAtom(dummy)
    rw.AddBond(dummy_idx, 0, Chem.rdchem.BondType.SINGLE)
    sidechain = rw.GetMol()

    # reaction to connect dummy atoms
    rxn = AllChem.ReactionFromSmarts("[*:1]-[*:2].[*:2]-[*:1]>>[*:1]-[*:1]")
    products = rxn.RunReactants((backbone, sidechain))

    if not products:
        return None

    mol = products[0][0]
    Chem.SanitizeMol(mol)
    return mol

# =============================
# Peptide Builder
# =============================
amide_rxn = AllChem.ReactionFromSmarts(
    "[C:1](=O)[O;H].[N;H2:2]>>[C:1](=O)[N:2]"
)

def couple(m1, m2):
    products = amide_rxn.RunReactants((m1, m2))
    if products:
        return products[0][0]
    return None

def build_peptide(tokens, custom_df):
    residues = []
    masses = []

    for t in tokens:
        if t in DEFAULT_SIDECHAINS:
            sidechain = DEFAULT_SIDECHAINS[t]
        else:
            row = custom_df[custom_df["code"] == t]
            if row.empty:
                return None, None
            sidechain = row.iloc[0]["sidechain"]

        res = build_residue(sidechain)
        if res is None:
            return None, None

        residues.append(res)
        masses.append(ExactMolWt(res) - H2O)

    mol = residues[0]
    for r in residues[1:]:
        mol = couple(mol, r)

    return mol, masses

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

# =============================
# UI
# =============================
st.title("🔬 Peptide MS Tool (Stable Version)")

init_db()
custom_df = load_custom()

with st.sidebar.expander("Custom Amino Acid Manager", expanded=False):
    code = st.text_input("3-letter Code")
    sidechain = st.text_input("Sidechain SMILES (no backbone)")
    if st.button("Save Custom AA"):
        save_custom(code, sidechain)
        st.success("Saved.")

    for _, row in custom_df.iterrows():
        col1, col2 = st.columns([4,1])
        col1.markdown(f"**{row['code']}**  `{row['sidechain']}`")
        if col2.button("Delete", key=row["code"]):
            delete_custom(row["code"])
            st.rerun()

sequence = st.text_input("Peptide Sequence (e.g., ACD(ORN)K)")
include_losses = st.checkbox("Include neutral losses (-H2O / -NH3)", value=True)

if sequence:
    tokens = parse_sequence(sequence.upper())
    mol, masses = build_peptide(tokens, custom_df)

    if mol:
        neutral_mass = sum(masses) + H2O

        st.subheader("Precursor m/z")
        precursor_df = pd.DataFrame({
            "Charge":[1,2,3,4,5],
            "m/z":[round(compute_mz(neutral_mass,z),4) for z in range(1,6)]
        })
        st.dataframe(precursor_df)

        fragments = generate_fragments(masses, include_losses)

        st.subheader("Fragment Table")
        st.dataframe(build_table(fragments))

        st.subheader("Simulated MS/MS Spectrum")
        st.plotly_chart(plot_spectrum(fragments), use_container_width=True)

        st.subheader("Full Peptide Structure")
        img = Draw.MolToImage(mol, size=(700,300))
        st.image(img)

    else:
        st.error("Invalid sidechain or sequence.")
