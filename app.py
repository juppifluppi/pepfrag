import streamlit as st
import pandas as pd
import re
import sqlite3
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
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
# Default Amino Acids (L-form)
# =============================
DEFAULT_AA = {
    "A": ("N[C@@H](C)C(=O)O", 71.03711),
    "R": ("N[C@@H](CCCNC(N)=N)C(=O)O", 156.10111),
    "N": ("N[C@@H](CC(=O)N)C(=O)O", 114.04293),
    "D": ("N[C@@H](CC(=O)O)C(=O)O", 115.02694),
    "C": ("N[C@@H](CS)C(=O)O", 103.00919),
    "E": ("N[C@@H](CCC(=O)O)C(=O)O", 129.04259),
    "Q": ("N[C@@H](CCC(=O)N)C(=O)O", 128.05858),
    "G": ("NCC(=O)O", 57.02146),
    "H": ("N[C@@H](Cc1c[nH]cn1)C(=O)O", 137.05891),
    "I": ("N[C@@H](C(C)CC)C(=O)O", 113.08406),
    "L": ("N[C@@H](CC(C)C)C(=O)O", 113.08406),
    "K": ("N[C@@H](CCCCN)C(=O)O", 128.09496),
    "M": ("N[C@@H](CCSC)C(=O)O", 131.04049),
    "F": ("N[C@@H](Cc1ccccc1)C(=O)O", 147.06841),
    "P": ("N1[C@@H](CCC1)C(=O)O", 97.05276),
    "S": ("N[C@@H](CO)C(=O)O", 87.03203),
    "T": ("N[C@@H](C(O)C)C(=O)O", 101.04768),
    "W": ("N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O", 186.07931),
    "Y": ("N[C@@H](Cc1ccc(O)cc1)C(=O)O", 163.06333),
    "V": ("N[C@@H](C(C)C)C(=O)O", 99.06841),
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

# =============================
# Parser
# =============================
def parse_sequence(seq):
    tokens = re.findall(r'\([A-Za-z0-9]+\)|[A-Z]', seq)
    return [t[1:-1] if t.startswith("(") else t for t in tokens]

# =============================
# Amide Coupling Reaction
# =============================
amide_rxn = AllChem.ReactionFromSmarts(
    "[C:1](=O)[O;H].[N;H2:2]>>[C:1](=O)[N:2]"
)

def couple(m1, m2):
    products = amide_rxn.RunReactants((m1, m2))
    if products:
        return products[0][0]
    return None

# =============================
# Build Peptide Molecule
# =============================
def build_peptide(tokens, custom_df):
    mol = None
    masses = []

    for t in tokens:
        if t in DEFAULT_AA:
            smiles, mass = DEFAULT_AA[t]
        else:
            row = custom_df[custom_df["code"] == t]
            if row.empty:
                return None, None
            smiles = row.iloc[0]["smiles"]
            mass = row.iloc[0]["mass"]

        aa_mol = Chem.MolFromSmiles(smiles)
        masses.append(mass)

        if mol is None:
            mol = aa_mol
        else:
            mol = couple(mol, aa_mol)

    return mol, masses

# =============================
# Fragmentation
# =============================
def generate_fragments(masses):
    fragments = []

    # b-ions
    running = 0
    for i in range(len(masses)-1):
        running += masses[i]
        b = running + PROTON
        fragments.append(("b"+str(i+1), b))
        fragments.append(("b"+str(i+1)+"-H2O", b - H2O))
        fragments.append(("b"+str(i+1)+"-NH3", b - NH3))

    # y-ions
    running = 0
    for i in range(len(masses)-1):
        running += masses[-(i+1)]
        y = running + PROTON + H2O
        fragments.append(("y"+str(i+1), y))
        fragments.append(("y"+str(i+1)+"-H2O", y - H2O))
        fragments.append(("y"+str(i+1)+"-NH3", y - NH3))

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

# =============================
# Spectrum
# =============================
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
st.title("🔬 Publication-Level Peptide MS/MS Tool")

init_db()
custom_df = load_custom_aa()

# Sidebar: Custom AA
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

# Main sequence input
sequence = st.text_input("Peptide Sequence (e.g., ACD(ORN)K)")

if sequence:
    tokens = parse_sequence(sequence.upper())
    mol, masses = build_peptide(tokens, custom_df)

    if mol:
        neutral_mass = sum(masses) + H2O

        st.subheader("Precursor m/z ([M+zH]z+)")
        precursor_df = pd.DataFrame({
            "Charge": [1,2,3,4,5],
            "m/z": [round(compute_mz(neutral_mass,z),4) for z in range(1,6)]
        })
        st.dataframe(precursor_df)

        st.subheader("b / y Fragment Table")
        fragments = generate_fragments(masses)
        frag_table = build_table(fragments)
        st.dataframe(frag_table)

        st.download_button(
            "Download Fragment CSV",
            frag_table.to_csv(index=False),
            "fragments.csv"
        )

        st.subheader("Simulated MS/MS Spectrum")
        st.plotly_chart(plot_spectrum(fragments), use_container_width=True)

        st.subheader("Full Peptide Structure")
        img = Draw.MolToImage(mol, size=(700,300))
        st.image(img)

    else:
        st.error("Invalid sequence or unknown residue.")
