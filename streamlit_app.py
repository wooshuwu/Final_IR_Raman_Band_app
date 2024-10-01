import py3Dmol
import streamlit as st
from stmol import showmol
import py3Dmol
import numpy as np
import pandas as pd
import datetime
import copy
# from st_aggrid import AgGrid

from app_funcs import * 



#open and store model files

# !git clone https://github.com/ignaciomigliaro/Group_theory_for_IR

# @title Open .xyz file with cartesian coordinates
fold_name = "C:\\Users\\artsy\\OneDrive - UNT System\\Research\\code2\\pythonExperiments\\Baba_apps\\Group_theory_for_IR\\Molecules"
file_paths = []

style_path = r"C:\Users\artsy\OneDrive - UNT System\Research\code2\pythonExperiments\Baba_apps\streamlit_styles.css"
with open(style_path, "r") as f:
    styles = f.read()
    
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
             "PtCL4": "PtCL4"
             }
file_list = list(file_vars)

for i in range(len(file_vars)):
    # print(f"{list(file_vars)[i]}")
    file_list[i] = open_file(file_list[i], fold_name)
    
for f in file_vars.values():
    file_paths.append(f"{fold_name}\\{f}.xyz")

vsepr_molecule_sheet = "VSEPR_Mol"
character_table_path = r"C:\Users\artsy\OneDrive - UNT System\Research\code2\pythonExperiments\Baba_apps\charactertables_v4.xlsx"
point_group_path = r"C:\Users\artsy\OneDrive - UNT System\Research\code2\pythonExperiments\Baba_apps\point_groups.xlsx"
point_group_geometry_sheet = "Geometry_PG"
point_group_symmetry_sheet = "Symmetry_PG"


point_group_geometry_df = pd.read_excel(point_group_path, sheet_name = point_group_geometry_sheet)
point_group_symmetry_df = pd.read_excel(point_group_path, sheet_name = point_group_symmetry_sheet)
vsepr_mol_df = pd.read_excel(point_group_path, sheet_name = vsepr_molecule_sheet, index_col=None, header = 0)

# Create an empty list to store the result
# symmetry_ops = []
symmetry_ops = {}

# Iterate over the rows of the DataFrame
for _, row in point_group_symmetry_df.iterrows():
    label = row[0]  # Get the label from the first column
    items = row[1].split(',')  # Split the second column by comma to get the items
    # symmetry_ops.append((label, items))  # Append a tuple of (label, items) to the result list
    symmetry_ops[label] = items

# Get the second elements (sublists) from the list of tuples
# sublists = [sublist for _, sublist in symmetry_ops]
# st.markdown(f"${symmetry_ops[1][1][3]}$")
temp = symmetry_ops["Oh"]
# st.markdown(f"symmetry: ${temp[-1]}$ {type(temp)}")

# Print the sublists
# for sublist in sublists:
#     # print(sublist)
#     st.markdown(f"${sublist}$")
    
# st.markdown(f"{symmetry_ops}")

# Create the list of dictionaries
groups = point_group_geometry_df.to_dict('records')

# Example usage
# for group in groups:
#     print(f"Geometry: {group['Geometry']}, Point Group: {group['Point Group']}")

st.title('Predicting IR and Raman Bands')

# style = st.sidebar.selectbox('Style', ['stick', 'line', 'sphere'])
# first_run = True
# change = 0

# def select_geometry_change_handler():
#     global first_run, change 
#     # idx2 = [group["Geometry"] for group in groups].index(selected_geometry)
#     # idx3 = 
#     # st.markdown(f"You selected: ${selected_geometry}$")
#     # if(first_run):
#     st.markdown(f"change: {change}")  
#     if(change == 0):
#         idx = 0
#         st.markdown(f"first run: {geometry_options[0]}")
#         first_run = False
#     else:
#         idx = [group["Geometry"] for group in groups].index(selected_geometry)
#         st.markdown(f"index (v2): {idx}")
#         st.markdown(f"NOT FIRST RUN")
#     change += 1 
#     st.markdown(f"change v2: {change}, index: {idx}")  
         
#     point_group = groups[idx]['Point Group']
#     geometry = groups[idx]['Geometry']
#     point_group_formatted = format_subscript_latex(point_group)

#     st.markdown(f"VSEPR (v2): ${format_vsepr_latex(geometry)}$")
#     st.markdown(f"Point group (v2):  $ {point_group_formatted}$")
    
    # color_styles = ['spectrum', 'element', 'chain', 'custom']

    # # List of available styles
    # styles = ['stick', 'line', 'sphere', 'cartoon']

    # # Dictionary mapping color styles to color options
    # color_options = {
    #     'spectrum': 'spectrum',
    #     'element': 'element',
    #     'chain': 'chain',
    #     'custom': 'red'  # You can change this to any custom color
    # }

    # # Create a selectbox for color style
    # selected_color_style = st.sidebar.selectbox('Select Color Style', color_styles)

    # # Create a selectbox for style
    # selected_style = st.sidebar.selectbox('Select Style', styles)

    # view = py3Dmol.view(data=file_list[idx])
    # # Apply the selected color style and style
    # view.setStyle({selected_style: {'color': color_options[selected_color_style]}})
    # view.zoomTo()

    # showmol(view, height=500, width=800)
    
    # st.markdown(f"Index plz: {selected_geometry}")
    # st.markdown(f"run: {change}")

    # print(f"Selection: {idx}")
    # st.markdown(f"Selection handler: {idx}")

# selected_geometry = groups["Geometry"][0] 

geometry_options = [group["Geometry"] for group in groups]
selected_geometry = geometry_options[0]
# idx = 0

# Selectbox for geometry
# selected_geometry =  st.sidebar.selectbox("Select Geometry", [group["Geometry"] for group in groups])
selected_geometry =  st.sidebar.selectbox("Select Geometry", geometry_options)
# selected_geometry_index = st.sidebar.selectbox("Select Geometry", [group["Geometry"] for group in groups], index=0, on_change=lambda value: select_geometry_change_handler(groups[value]['Geometry']))
# selected_geometry_index = st.sidebar.selectbox("Select Geometry", [group["Geometry"] for group in groups], on_change=lambda value: select_geometry_change_handler(geometry_options[value]))
# selected_geometry = st.sidebar.selectbox("Select Geometry", [group["Geometry"] for group in groups], index=0, on_change=lambda value: select_geometry_change_handler(groups[geometry_options.index(value)]))
# selected_geometry = st.sidebar.selectbox("Select Geometry", [group["Geometry"] for group in groups], index=0, on_change=select_geometry_change_handler(groups[[group["Geometry"] for group in groups].index(selected_geometry)]["Geometry"]))
# selected_geometry = st.sidebar.selectbox("Select Geometry", [group["Geometry"] for group in groups], index=0, on_change=select_geometry_change_handler(group["Geometry"][geometry_options.index(selected_geometry)]))
# selected_geometry = st.sidebar.selectbox("Select Geometry", [group["Geometry"] for group in groups], on_change=select_geometry_change_handler(geometry_options.index(selected_geometry)))
# selected_geometry = st.sidebar.selectbox("Select Geometry", [group["Geometry"] for group in groups], index=0, on_change=select_geometry_change_handler())
# selected_geometry = groups[selected_geometry_index]["Geometry"]

color_styles = ['spectrum', 'element', 'chain', 'custom']

# List of available styles
styles = ['stick', 'line', 'sphere', 'cartoon']

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
    now = datetime.datetime.now()

    # Format the date and time
    formatted_datetime = now.strftime("%m-%d-%Y %H:%M:%S")
    print(f"NEW RUN ({formatted_datetime})---------------------------------------------------")
    selected_color_style = st.sidebar.selectbox('Select Color Style', color_styles)

    # Create a selectbox for style
    selected_style = st.sidebar.selectbox('Select Style', styles)
    
    idx2 = [group["Geometry"] for group in groups].index(selected_geometry)
    # st.markdown(f"Index: {idx2}, {selected_geometry}")
    
    point_group = groups[idx2]['Point Group']
    geometry = groups[idx2]['Geometry']
    point_group_formatted = format_subscript_latex(point_group)
    
    st.markdown(f"VSEPR: ${format_vsepr_latex(geometry)}$")
    st.markdown(f"Point group:  $ {point_group_formatted}$")

    name_atoms = file_matrix(file_list[idx2])
    atom_names = name_atoms[:, 0]
    molecule_name = vsepr_mol_df[vsepr_mol_df["VSEPR"] == f"{geometry}"]["Molecule"].iat[0]
    molecule_name = f"${format_molecules(molecule_name)}$"
    st.markdown(f"Molecule name: {molecule_name}")
    # st.markdown(f"atom names: {atom_names}")
        
    atoms = np.array(name_atoms[:,1:])
    atoms = atoms.astype(float)
    natoms = int(np.shape(atoms)[0])
    st.markdown(f"Number of atoms: {natoms}")
    # st.markdown(f"Atom positions: {atoms}")
    
    unmoved_atoms = apply_symm(point_group, natoms, atoms)
    char_table_raw_df = pd.read_excel(character_table_path, point_group)
    
    # char_table_raw_df = pd.read_excel(character_table_path, point_group, index_col = 0)
    # char_table_raw_df = pd.read_excel(character_table_path, point_group, header = None)
    # char_table_raw_df.columns = char_table_raw_df.iloc[0, 1:]
    # char_table_raw_df = char_table_raw_df.set_index(0)
    # print(f"columns: {char_table_raw_df.columns}")
    # char_table_raw_df = char_table_raw_df.fillna("")
    symmetry_coefficients, order, atomic_contribution_symm = get_data_from_point_group(point_group, char_table_raw_df)
    st.markdown(f"order: {order}")
    # st.markdown(f"unmoved_atoms: {unmoved_atoms}")
    # st.markdown(f"symmetry_coefficients: {symmetry_coefficients}")
    # st.markdown(f"atomic contribution of symmetry: {atomic_contribution_symm}")
    
    unmoved_atoms_df = pd.DataFrame([unmoved_atoms], columns = char_table_raw_df.columns[1:-3])
    # print(f"{unmoved_atoms_df}")
    unmoved_atoms_latex = [f"${format_char_table_header(col)}$" for col in unmoved_atoms_df]
    unmoved_atoms_df.columns = unmoved_atoms_latex
     
    # print(f"{np.shape(char_table_raw_df.columns[1:-3])}")
    # print(f"{type(char_table_raw_df.columns[1:-3])}")
    
    view = py3Dmol.view(data=file_list[idx2])
    # Apply the selected color style and style
    view.setStyle({selected_style: {'color': color_options[selected_color_style]}})
    view.zoomTo()
    showmol(view, height=500, width=800)
    
    
    # latex_title = [f"${format_subscript_latex_title(col)}$" for col in char_table_raw_df]
    latex_title = [f"${format_char_table_header(col)}$" for col in char_table_raw_df]
    # st.markdown(f"{latex_title}")
    char_table_raw_df.iloc[:, 0] = char_table_raw_df.iloc[:, 0].apply(format_subscript_latex) 
    char_table_raw_df.iloc[:, 0] = char_table_raw_df.iloc[:, 0].apply(lambda x: f"${x}$" if isinstance(x, str) else x)
    # char_table_title_latex_df = pd.DataFrame(char_table_raw_df, columns = latex_title)
    # char_table_latex_df = char_table_raw_df
    # char_table_latex_df = char_table_raw_df.fillna("")
    char_table_latex_df = char_table_raw_df
    # char_table_latex_df = char_table_raw_df[:-2]
    # char_table_latex_df = char_table_raw_df.iloc[:-2]
    #hide the # and ## rows 
    # char_table_latex_df = char_table_latex_df.iloc[:-2]
    char_table_latex_df.columns = latex_title
    # char_table_latex_df = char_table_latex_df.iloc[:-2]
    
    char_table_latex_df["$Quadratic$"] = format_superscripts_from_df_column(char_table_latex_df, "$Quadratic$")
    char_table_latex_df["$Rotational$"] = format_rotational_latex(char_table_latex_df, "$Rotational$")
    # char_table_latex_df = char_table_latex_df[:-2]
    char_table_latex_df_no_na = char_table_raw_df[:-2].fillna("")
    # char_table_latex_df_no_na = char_table_raw_df.fillna("")
    
    # temp2 = char_table_title_latex_df["$Quadratic$"]
    # st.markdown(f"{temp2}")
    # st.markdown(f"$2^2$")
    # print(f"row names: {char_table_latex_df.index[-1]}")
    # char_table_latex_df = char_table_latex_df.rename(index = {-1: "Atomic contribution"})
    # char_table_latex_df.loc["Atomic contribution"] = char_table_latex_df.loc["#"]
    # print(f"row names: {char_table_latex_df.index[-1]}")
    
    
    table_markdown = char_table_latex_df_no_na.to_markdown(index = False)
    # table_markdown = char_table_latex_df.to_html()
    st.markdown(f"## Character table")
    # st.markdown(f"<style>{styles}</style> {table_markdown}", unsafe_allow_html=True)
    # st.markdown(f"{table_markdown}")
    # st.markdown(table_markdown, unsafe_allow_html=True)
    st.markdown(table_markdown)
    
    gamma_total_label = r"$\Gamma_{total}$"
    st.markdown(f"### {gamma_total_label}")
    # st.markdown(unmoved_atoms_df.to_markdown(index = False))
    
    # st.markdown(f"### Symmetry coefficients")
    # symmetry_coefficients_table = char_table_raw_df.iloc[-2:-1, :-3]
    # st.markdown(symmetry_coefficients_table.to_markdown(index = False))
    
    # st.markdown(f"### Atomic contributions of symmetry")
    gamma_total_table = char_table_raw_df.iloc[-1:, :-3]
    
    # st.markdown(f"dimensions owo: {np.shape(unmoved_atoms_df)}")
    # st.markdown(f"{np.shape(gamma_total_table)}")
    # st.markdown(f"{unmoved_atoms_df[1:]}")

    gamma_total_table = gamma_total_table.reset_index(drop = True)
    unmoved_atoms_df = unmoved_atoms_df.reset_index(drop = True)
    gamma_total_table = pd.concat([gamma_total_table, unmoved_atoms_df])
    # gamma_total_table = gamma_total_table.reset_index(drop = True)
    gamma_xyz_label = r"$\Gamma_{xyz}$"
    number_unmoved_label = "unmoved"
    gamma_xyz = pd.DataFrame([gamma_total_table.iloc[0, 1:] * gamma_total_table.iloc[1, 1:]])
    gamma_total_table = pd.concat([gamma_total_table, gamma_xyz])
    
    gamma_total_table.iat[0,0] = f"{gamma_xyz_label}"
    gamma_total_table.iat[1,0] = f"$\#_{{{number_unmoved_label}}}$"
    gamma_total_table.iat[2,0] = f"{gamma_total_label}"
    st.markdown(gamma_total_table.to_markdown(index = False))
    
    
    st.markdown(f"### Breakdown of {gamma_total_label} into irreducible representations")
    irreducible_table = calculate_irreducible_representations(char_table_raw_df, order, unmoved_atoms, atomic_contribution_symm, symmetry_coefficients)
    
    irreducible_table = irreducible_table[:-1]
    irreducible_table_no_na = irreducible_table.fillna("")
    st.markdown(irreducible_table_no_na.to_markdown(index = False))
    
    irreducible_formula = f"{gamma_total_label} = {gamma_formula_notation(irreducible_table)}"
    st.markdown(f"{irreducible_formula}")
    
    # irreducible_table = calculate_irreducible_representations(char_table_raw_df, order_grad, unmoved_atoms, symmetry_coefficients_grad, symmetry_coefficients_ugrad)
    # irreducible_table_markdown = irreducible_table.to_markdown(index = False)
    # # st.table(irreducible_table)
    # st.markdown(f"## Previous")
    # st.markdown(irreducible_table_markdown)
    
    #SEPARATING LINEAR, ROTATIONAL, QUADRATIC
    # st.markdown(f"### Linear xyz rows")
    # linear_clean = char_table_raw_df[~char_table_raw_df['$Linear$'].isnull()]
    # linear_clean_no_na = linear_clean.fillna("")
    # # linear_clean_table_markdown = linear_clean_no_na.to_markdown(index = False)
    # st.markdown(linear_clean_no_na.to_markdown(index = False))
    
    # st.markdown(f"### Rotational xyz row")
    # rot_clean = char_table_raw_df[~char_table_raw_df['$Rotational$'].isnull()]
    # # rot_clean = rot_clean[rot_clean.M > 0]
    # # rot_clean = rot_clean[:-1]
    # rot_clean = rot_clean.fillna("")
    # rot_clean_table_markdown = rot_clean.to_markdown(index = False)
    # st.markdown(rot_clean_table_markdown)
    
    # st.markdown(f"### Quadratic rows")
    # quad_clean = char_table_raw_df[~char_table_raw_df['$Quadratic$'].isnull()]
    # # quad_clean = quad_clean[quad_clean.M > 0]
    # quad_clean = quad_clean[:-1]
    # quad_clean_no_na = quad_clean.fillna("")
    # # quad_clean_table_markdown = quad_clean.to_markdown(index = False)
    # st.markdown(quad_clean_no_na.to_markdown(index = False))
    
    reduced_rotation = irreducible_table[~irreducible_table['$Rotational$'].isnull()]
    #subtract the representation that corresponds to the rotational motion
    
    # for rotations in reduced_rotation:
    #     Rz_check = rotations == "Rz"
    #     if(not Rz_check):
    #         reduced_rotation.loc[:, "M"] = reduced_rotation.loc[:, "M"] - 1
    
    # Create a boolean mask for elements in 'col1' that don't contain 'Rz'
    if(point_group == "Dinf_h"):
        rz_mask = ~reduced_rotation['$Rotational$'].str.contains('$R_{z}$', na=False)
        reduced_rotation.loc[rz_mask, "M"] = reduced_rotation.loc[rz_mask, "M"].values - 1
    else:
        reduced_rotation.loc[:, "M"] = reduced_rotation.loc[:, "M"] - 1
    reduced_irreducible_rot = copy.copy(irreducible_table)
    reduced_irreducible_rot.loc[reduced_rotation.index, "M"] = reduced_rotation["M"]
    
    # # Create a style object
    # styled_df = reduced_irreducible_rot.style

    # # Define a function to color the cells red
    # def color_red(val):
    #     return f"color: red" if val == reduced_rotation["M"] else ""

    # # Apply the color function to the "M" column
    # styled_df = styled_df.applymap(color_red, subset=pd.IndexSlice[:, ["M"]])
    # st.dataframe(styled_df)
    
    irreducible_table = reduced_irreducible_rot
    #subtract the representation that corresponds to the translational motion
    reduced_lin = irreducible_table[~irreducible_table['$Linear$'].isnull()]
    reduced_lin.loc[:, "M"] = reduced_lin.loc[:, "M"].values - 1
    reduced_irreducible_lin = copy.copy(irreducible_table)
    reduced_irreducible_lin.loc[reduced_lin.index, "M"] = reduced_lin["M"]
    irreducible_table = reduced_irreducible_lin
    vibration_rep = irreducible_table

    
    # st.markdown(f"#### Rotational (M - 1)")
    # st.markdown(reduced_rotation.fillna("").to_markdown(index = False))
    gamma_rotational_label = r"$\Gamma_{Rot(R_xR_yR_z)}$"
    # st.markdown(f"#### {gamma_total_label} - {gamma_rotational_label}")
    # st.markdown(reduced_irreducible_rot.fillna("").to_markdown(index = False))
    # st.markdown(f"#### Linear (M - 1)")
    # st.markdown(reduced_lin.fillna("").to_markdown(index = False))
    gamma_linear_label = r"$\Gamma_{Trans(x,y,z)}$"
    gamma_vibrational_label = r"$\Gamma_{vib}$"
    st.markdown(f"#### {gamma_vibrational_label} = {gamma_total_label} - {gamma_linear_label} - {gamma_rotational_label}")
    st.markdown(reduced_irreducible_lin.fillna("").to_markdown(index = False))
    st.markdown(f"{gamma_vibrational_label} = {gamma_formula_notation(reduced_irreducible_lin)}")
    
    # st.markdown(f"## Vibrational modes ")
    # st.markdown(irreducible_table.fillna("").to_markdown(index = False))
    
    IR_active = vibration_rep[~vibration_rep["$Linear$"].isnull()]
    IR_active_reduced = IR_active.iloc[:, [0, -1]]
    IR_active_reduced = IR_active_reduced[IR_active_reduced.M > 0]
    IR_active_count = IR_active.iloc[:, -1].astype(int).sum()
    
    st.markdown(f"## IR Active Bands ")
    st.markdown(IR_active_reduced.fillna("").to_markdown(index = False))
    st.markdown(f"#### Number of IR bands: {IR_active_count}")
    
    Raman_active = vibration_rep.dropna(subset = ["$Quadratic$", "$Rotational$"], how = "all")
    # Raman_active = vibration_rep["$Quadratic$"].isna() | vibration_rep["$Rotational$"].isna()
    Raman_active_reduced = Raman_active.iloc[:, [0, -1]]
    Raman_active_reduced = Raman_active_reduced[Raman_active_reduced.M > 0]
     
    Raman_active_count = Raman_active.iloc[:, -1].astype(int).sum()
    
    st.markdown(f"## Raman Active Bands")
    st.markdown(Raman_active_reduced.fillna("").to_markdown(index = False))
    st.markdown(f"#### Number of Raman bands: {Raman_active_count}")

    # temp = ", ".join(f"$[{title}]$" for title in latex_title)
    # st.markdown(temp)
    # st.table(char_table_raw_df)
    # st.table(char_table_title_latex_df)

    
    # print(f"natoms: {self.natoms}")
    
    # st.markdown(f"VSEPR: ${format_vsepr_latex(geometry)}$")
    # st.markdown(f"Point group:  $ {point_group_formatted}$")

    
    
    # st.markdown(f"Number of atoms: {natoms}")
    # st.markdown(f"Unmoved atoms: {unmoved_atoms} ")
    # st.markdown(f"Coefficient of each symmetry operation: {symmetry_coefficients} ")
    # st.markdown(f"Group order: {order}")

    # showmol(view, height=500, width=800)
    
geometry_change()    


# Get the index of the selected geometry
# idx2 = [group["Geometry"] for group in groups].index(selected_geometry)
# st.markdown(f"Index: {idx2}, {selected_geometry}")


# selected_geometry.set_on_change(select_geometry_change_handler, groups[idx]["Geometry"])

# point_group_formatted = [f"{s[0]}_{{{s[1:]}}}" for s in groups['Point Group']]
# point_group_formatted = format_subscript_latex(groups[idx2]['Point Group'])

# st.markdown(f"VSEPR: ${format_vsepr_latex(groups[idx2]['Geometry'])}$")
# st.markdown(f"Point group:  $ {point_group_formatted}$")
# st.latex(fr"Point group latex: {point_group_formatted}")
# st.write(f"Point group latex: {{{point_group_formatted}}}")


# p.rotate(45, 'x', 5000)
# p.rotate(90, 'y', 2000)
# p    
# xyzview = py3Dmol.view(query='pdb:1UBQ')

# List of available color styles
# color_styles = ['spectrum', 'element', 'chain', 'custom']

# # List of available styles
# styles = ['stick', 'line', 'sphere', 'cartoon']

# # Dictionary mapping color styles to color options
# color_options = {
#     'spectrum': 'spectrum',
#     'element': 'element',
#     'chain': 'chain',
#     'custom': 'red'  # You can change this to any custom color
# }

# # Create a selectbox for color style
# selected_color_style = st.sidebar.selectbox('Select Color Style', color_styles)

# # Create a selectbox for style
# selected_style = st.sidebar.selectbox('Select Style', styles)

# view = py3Dmol.view(data=file_list[idx2])
# # Apply the selected color style and style
# view.setStyle({selected_style: {'color': color_options[selected_color_style]}})
# view.zoomTo()

# showmol(view, height=500, width=800)

# xyzview.setStyle({style:{'color':'spectrum'}})
# xyzview.zoomTo()



# showmol(xyzview, height=500, width=800)