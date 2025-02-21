import py3Dmol
import streamlit as st
from stmol import showmol
import numpy as np
import pandas as pd
import datetime
import copy
# from st_aggrid import AgGrid

from app_funcs import * 

website_name = "unt-predicting-ir-raman-bands"
st.set_page_config(page_title=website_name)

# @title Open .xyz file with cartesian coordinates
fold_name = "Molecules"
file_paths = []
    
file_vars = {"CO2": "CO2", 
             "BH3": "BH3", 
             "SO2": "SO2", 
             "CH4": "CH4",
             "NH3": "NH3", 
             "H2O": "H2O", 
             "PF5": "PF5", 
             "SF4": "SF4", 
             "ClF3": "ClF3",
             "XeF2": "XeF2",
             "FeCl6": "FeCl6",
             "ClF5": "ClF5",
             "PtCl4": "PtCl4"
             }
file_list = list(file_vars)

for i in range(len(file_vars)):
    file_list[i] = open_file(file_list[i], fold_name)
    
for f in file_vars.values():
    file_paths.append(f"{fold_name}\\{f}.xyz")

vsepr_molecule_sheet = "VSEPR_Mol"
character_table_path = r"charactertables_v4.xlsx"
point_group_path = r"point_groups.xlsx"
point_group_geometry_sheet = "Geometry_PG"
point_group_symmetry_sheet = "Symmetry_PG"


point_group_geometry_df = pd.read_excel(point_group_path, sheet_name = point_group_geometry_sheet)
point_group_symmetry_df = pd.read_excel(point_group_path, sheet_name = point_group_symmetry_sheet)
vsepr_mol_df = pd.read_excel(point_group_path, sheet_name = vsepr_molecule_sheet, index_col=None, header = 0)

# Create an empty list to store the result
symmetry_ops = {}

# Iterate over the rows of the DataFrame
for _, row in point_group_symmetry_df.iterrows():
    label = row.iloc[0]  # Get the label from the first column
    items = row.iloc[1].split(',')  # Split the second column by comma to get the items
    symmetry_ops[label] = items

temp = symmetry_ops["Oh"]

# Create the list of dictionaries
groups = point_group_geometry_df.to_dict('records')

st.title('Predicting IR and Raman Bands')

geometry_options = [group["Geometry"] for group in groups]
selected_geometry = geometry_options[0]

selected_geometry =  st.sidebar.selectbox("Select Geometry", geometry_options)

color_styles = ['spectrum', 'element', 'chain', 'custom']

# List of available styles
styles = ['stick', 'line', 'sphere', 'cartoon']

version_dictionary = {
    "Selected Stretching Vibration": "UG", 
    "Full Vibration": "G"
}

# Dictionary mapping color styles to color options
color_options = {
    'spectrum': 'spectrum',
    'element': 'element',
    'chain': 'chain',
    'custom': 'red'  # You can change this to any custom color
}

# st.markdown("""
#     <style>
#     @media (max-width: 600px) {
#         .stApp {
#             transform: scale(0.5);
#             transform-origin: top left;
#         }
#     }
#     </style>
# """, unsafe_allow_html=True)

    
def geometry_change():
    # Create a selectbox for color style
    # Get the current date and time
    # now = datetime.datetime.now()

    # # Format the date and time
    # formatted_datetime = now.strftime("%m-%d-%Y %H:%M:%S")
    # print(f"NEW RUN ({formatted_datetime})---------------------------------------------------")
    
    selected_color_style = st.sidebar.selectbox('Select Color Style', color_styles)

    # Create a selectbox for style
    selected_style = st.sidebar.selectbox('Select Style', styles)
    
    versions = ["Selected Stretching Vibration", "Full Vibration"]
    selected_version = st.sidebar.selectbox("Select Version", versions)
    
    current_version = version_dictionary[selected_version]
    st.markdown(f"Selected version: {current_version}")
    
    idx2 = [group["Geometry"] for group in groups].index(selected_geometry)
    
    point_group = groups[idx2]['Point Group']
    geometry = groups[idx2]['Geometry']
    point_group_formatted = format_subscript_latex(point_group)
    
    st.markdown(f"VSEPR: ${format_vsepr_latex(geometry)}$")
    st.markdown(f"Point group:  $ {point_group_formatted}$")

    name_atoms = file_matrix(file_list[idx2])
    atom_names = name_atoms[:, 0]
    molecule_name = vsepr_mol_df[vsepr_mol_df["VSEPR"] == f"{geometry}"]["Molecule"].iat[0]
    molecule_name = f"${format_molecules(molecule_name)}$"
    st.markdown(f"Example Molecule: {molecule_name}")
        
    atoms = np.array(name_atoms[:,1:])
    atoms = atoms.astype(float)
    natoms = int(np.shape(atoms)[0])
    
    atoms_ug = np.delete(atoms, (0), axis=0)
    natoms_ug = int(np.shape(atoms_ug)[0])
    
    print(f"number of atoms: {natoms_ug} \natoms size og {np.size(atoms)} atoms size ug {np.size(atoms_ug)}")
    if(current_version == "UG"):
        st.markdown(f"Unmoved arrows: {natoms_ug}")
    elif(current_version == "G"):
        st.markdown(f"Number of atoms: {natoms}")
    ## REMOVE LATER
    # st.markdown(f"Atoms: {atoms}")
    # st.markdown(f"Atoms UG: {atoms_ug}")
    
    
    unmoved_atoms = apply_symm(point_group, natoms, atoms)
    ## REMOVE LATER
    # st.markdown(f"Unmoved atoms: {unmoved_atoms}")
    unmoved_atoms_ug = apply_symm(point_group, natoms_ug, atoms_ug)
    ## REMOVE LATER
    # st.markdown(f"Unmoved atoms UG: {unmoved_atoms_ug}")
    char_table_raw_df = pd.read_excel(character_table_path, point_group)

    symmetry_coefficients, order, atomic_contribution_symm = get_data_from_point_group(point_group, char_table_raw_df)
    st.markdown(f"order: {order}")
    
    unmoved_atoms_df = pd.DataFrame([unmoved_atoms], columns = char_table_raw_df.columns[1:-3])
    unmoved_atoms_df_ug = pd.DataFrame([unmoved_atoms_ug], columns = char_table_raw_df.columns[1:-3])
    unmoved_atoms_latex = [f"${format_char_table_header(col)}$" for col in unmoved_atoms_df]
    unmoved_atoms_latex_ug = [f"${format_char_table_header(col)}$" for col in unmoved_atoms_df_ug]
    unmoved_atoms_df.columns = unmoved_atoms_latex
    unmoved_atoms_df_ug.columns = unmoved_atoms_latex_ug
    
    view = py3Dmol.view(data=file_list[idx2])
    # Apply the selected color style and style
    view.setStyle({selected_style: {'color': color_options[selected_color_style]}})
    view.zoomTo()
    showmol(view, height=500, width=800)
    
    latex_title = [f"${format_char_table_header(col)}$" for col in char_table_raw_df]
    
    char_table_raw_df.iloc[:, 0] = char_table_raw_df.iloc[:, 0].apply(format_subscript_latex) 
    char_table_raw_df.iloc[:, 0] = char_table_raw_df.iloc[:, 0].apply(lambda x: f"${x}$" if isinstance(x, str) else x)

    char_table_latex_df = char_table_raw_df
    char_table_latex_df.columns = latex_title
    
    char_table_latex_df["$Quadratic$"] = format_superscripts_from_df_column(char_table_latex_df, "$Quadratic$")
    char_table_latex_df["$Rotational$"] = format_rotational_latex(char_table_latex_df, "$Rotational$")
    char_table_latex_df_no_na = char_table_raw_df[:-2].fillna("")
    
    table_markdown = char_table_latex_df_no_na.to_markdown(index = False)
    st.markdown(f"## Character table")

    st.markdown(table_markdown)
    
    gamma_total_label = r"$\Gamma_{total}$"
    st.markdown(f"### {gamma_total_label}")
    
    # st.markdown(f"### Atomic contributions of symmetry")
    gamma_total_table = char_table_raw_df.iloc[-1:, :-3]
    gamma_total_table_ug = char_table_raw_df.iloc[[-2], :-3]
    # st.markdown(f"{np.size(char_table_raw_df.iloc[[-2], :-3])}")

    gamma_total_table = gamma_total_table.reset_index(drop = True)
    gamma_total_table_ug = gamma_total_table_ug.reset_index(drop = True)
    unmoved_atoms_df = unmoved_atoms_df.reset_index(drop = True)
    unmoved_atoms_df_ug = unmoved_atoms_df_ug.reset_index(drop = True)
    gamma_total_table = pd.concat([gamma_total_table, unmoved_atoms_df])
    # st.markdown(f"Gamma total table temp: {gamma_total_table_ug}")

    gamma_total_table_ug = pd.concat([gamma_total_table_ug, unmoved_atoms_df_ug])

    gamma_xyz_label = r"$\Gamma_{xyz}$"
    number_unmoved_label = "unmoved"
    gamma_xyz = pd.DataFrame([gamma_total_table.iloc[0, 1:] * gamma_total_table.iloc[1, 1:]])
    gamma_xyz_ug = pd.DataFrame([gamma_total_table_ug.iloc[0, 1:] * gamma_total_table_ug.iloc[1, 1:]])
    gamma_total_table = pd.concat([gamma_total_table, gamma_xyz])
    gamma_total_table_ug = pd.concat([gamma_total_table_ug, gamma_xyz_ug])
    
    gamma_total_table.iat[0,0] = f"{gamma_xyz_label}"
    gamma_total_table.iat[1,0] = f"$\#_{{{number_unmoved_label}}}$"
    gamma_total_table.iat[2,0] = f"{gamma_total_label}"
    
    
    number_unmoved_label_ug = "unmovedArrows"
    gamma_total_table_ug.iat[0,0] = f"coefficients" #coefficients of each symmetry operation
    gamma_total_table_ug.iat[1,0] = f"$\#_{{{number_unmoved_label_ug}}}$"
    gamma_total_table_ug.iat[2,0] = f"{gamma_total_label}"
    
    # gamma_total_label_ug = r"$\Gamma_{total UG}$"
    if(current_version == "G"):
        st.markdown(gamma_total_table.to_markdown(index = False))
    elif(current_version == "UG"):    
        st.markdown(gamma_total_table_ug.to_markdown(index = False))
    # st.markdown(f"### {gamma_total_label_ug}")
    
    
    
    # print(f"og df: {char_table_raw_df}")
    irreducible_table = calculate_irreducible_representations(copy.copy(char_table_raw_df), order, unmoved_atoms, atomic_contribution_symm, symmetry_coefficients, "grad")
    # irreducible_table_v2 = calculate_irreducible_representations(char_table_raw_df, order, unmoved_atoms, atomic_contribution_symm, symmetry_coefficients, "grad")
    
    irreducible_table = irreducible_table[:-1]
    irreducible_table_no_na = irreducible_table.fillna("")
    irreducible_formula = f"{gamma_total_label} = {gamma_formula_notation(irreducible_table)}"
    if(current_version == "G"):
        st.markdown(f"### Breakdown of {gamma_total_label} into irreducible representations")
        st.markdown(irreducible_table_no_na.to_markdown(index = False))
        st.markdown(f"{irreducible_formula}")
    
    gamma_vib_label_ug = r"$\Gamma_{vibration}$"
    
    # print(f"uwu ug: {char_table_raw_df}")
    irreducible_table_ug = calculate_irreducible_representations(copy.copy(char_table_raw_df), order, unmoved_atoms_ug, atomic_contribution_symm, symmetry_coefficients, "undergrad")
    
    irreducible_table_ug = irreducible_table_ug[:-1]
    irreducible_table_no_na_ug = irreducible_table_ug.fillna("")
    irreducible_formula_ug = f"{gamma_total_label} = {gamma_formula_notation(irreducible_table_ug)}"
    
    if(current_version == "UG"):
        st.markdown(f"### Breakdown of {gamma_vib_label_ug} into irreducible representations")
        st.markdown(irreducible_table_no_na_ug.to_markdown(index = False))
        st.markdown(f"{irreducible_formula_ug}")
    
    reduced_rotation = irreducible_table[~irreducible_table['$Rotational$'].isnull()]
    #subtract the representation that corresponds to the rotational motion
    
    # Create a boolean mask for elements in 'col1' that don't contain 'Rz'
    if(point_group == "Dinf_h"):
        rz_mask = ~reduced_rotation['$Rotational$'].str.contains('$R_{z}$', na=False)
        reduced_rotation.loc[rz_mask, "M"] = reduced_rotation.loc[rz_mask, "M"].values - 1
    else:
        reduced_rotation.loc[:, "M"] = reduced_rotation.loc[:, "M"] - 1
    reduced_irreducible_rot = copy.copy(irreducible_table)
    reduced_irreducible_rot.loc[reduced_rotation.index, "M"] = reduced_rotation["M"]
    
    irreducible_table = reduced_irreducible_rot
    #subtract the representation that corresponds to the translational motion
    reduced_lin = irreducible_table[~irreducible_table['$Linear$'].isnull()]
    reduced_lin.loc[:, "M"] = reduced_lin.loc[:, "M"].values - 1
    reduced_irreducible_lin = copy.copy(irreducible_table)
    reduced_irreducible_lin.loc[reduced_lin.index, "M"] = reduced_lin["M"]
    irreducible_table = reduced_irreducible_lin
    vibration_rep = irreducible_table

    gamma_rotational_label = r"$\Gamma_{Rot(R_xR_yR_z)}$"

    gamma_linear_label = r"$\Gamma_{Trans(x,y,z)}$"
    gamma_vibrational_label = r"$\Gamma_{vib}$"
    
    if(current_version == "G"):
        st.markdown(f"#### {gamma_vibrational_label} = {gamma_total_label} - {gamma_linear_label} - {gamma_rotational_label}")
        st.markdown(reduced_irreducible_lin.fillna("").to_markdown(index = False))
        st.markdown(f"{gamma_vibrational_label} = {gamma_formula_notation(reduced_irreducible_lin)}")
    
    IR_active = vibration_rep[~vibration_rep["$Linear$"].isnull()]
    IR_active_reduced = IR_active.iloc[:, [0, -1]]
    IR_active_reduced = IR_active_reduced[IR_active_reduced.M > 0]
    IR_active_count = IR_active.iloc[:, -1].astype(int).sum()
    
    st.markdown(f"## IR Active Bands ")
    if(current_version == "G"):
        st.markdown(IR_active_reduced.fillna("").to_markdown(index = False))
        st.markdown(f"#### Number of IR bands: {IR_active_count}")
    
    IR_active_ug = irreducible_table_ug[~irreducible_table_ug["$Linear$"].isnull()]
    IR_active_reduced_ug = IR_active_ug.iloc[:, [0, -1]]
    IR_active_reduced_ug = IR_active_reduced_ug[IR_active_reduced_ug.M > 0]
    IR_active_count_ug = IR_active_ug.iloc[:, -1].astype(int).sum()
    # st.markdown(f"## IR Active Bands")
    
    if(current_version == "UG"):
        st.markdown(IR_active_reduced_ug.fillna("").to_markdown(index = False))
        st.markdown(f"#### Number of IR bands: {IR_active_count_ug}")
    
    Raman_active = vibration_rep.dropna(subset = ["$Quadratic$", "$Rotational$"], how = "all")
    # Raman_active = vibration_rep["$Quadratic$"].isna() | vibration_rep["$Rotational$"].isna()
    Raman_active_reduced = Raman_active.iloc[:, [0, -1]]
    Raman_active_reduced = Raman_active_reduced[Raman_active_reduced.M > 0]
     
    Raman_active_count = Raman_active.iloc[:, -1].astype(int).sum()
    
    st.markdown(f"## Raman Active Bands")
    if(current_version == "G"):
        st.markdown(Raman_active_reduced.fillna("").to_markdown(index = False))
        st.markdown(f"#### Number of Raman bands: {Raman_active_count}")
    
    Raman_active_ug = irreducible_table_ug[~irreducible_table_ug["$Quadratic$"].isnull()]
    # Raman_active = vibration_rep["$Quadratic$"].isna() | vibration_rep["$Rotational$"].isna()
    Raman_active_reduced_ug = Raman_active_ug.iloc[:, [0, -1]]
    Raman_active_reduced_ug = Raman_active_reduced_ug[Raman_active_reduced_ug.M > 0]
     
    Raman_active_count_ug = Raman_active_ug.iloc[:, -1].astype(int).sum()
    
    # st.markdown(f"## Raman Active Bands (UG)")
    if(current_version == "UG"):
        st.markdown(Raman_active_reduced_ug.fillna("").to_markdown(index = False))
        st.markdown(f"#### Number of Raman bands: {Raman_active_count_ug}")
    
geometry_change()    