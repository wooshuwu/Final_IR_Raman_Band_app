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
st.set_page_config(page_title=website_name, layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for responsiveness
# st.markdown("""
# <style>
#     .reportview-container .main .block-container {
#         max-width: 95%;
#         padding-top: 5rem;
#         padding-right: 1rem;
#         padding-left: 1rem;
#         padding-bottom: 5rem;
#     }
#     .stTable {
#         width: 100%;
#         overflow-x: auto;
#     }
# </style>
# """, unsafe_allow_html=True)

st.markdown("""
<style>
    @media (max-width: 600px) {
        .stApp {
            transform: scale(0.6);
            transform-origin: top left;
            width: 166.66%;
            height: 166.66%;
            position: absolute;
            top: 0;
            left: 0;
        }
    }
</style>
""", unsafe_allow_html=True)


# Sidebar
# st.sidebar.title("Controls")

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

#Sidebar content
color_styles = ['spectrum', 'element', 'chain', 'custom']

# List of available styles
styles = ['stick', 'line', 'sphere', 'cartoon']

version_dictionary = {
    "Selected Stretching Vibrations": "UG", 
    "Full Vibrations": "G"
}

versions = ["Selected Stretching Vibration", "Full Vibration"]

# Dictionary mapping color styles to color options
color_options = {
    'spectrum': 'spectrum',
    'element': 'element',
    'chain': 'chain',
    'custom': 'red'  # You can change this to any custom color
}

selected_version = st.sidebar.selectbox("Select Version", version_dictionary.keys())
selected_geometry =  st.sidebar.selectbox("Select Geometry", geometry_options)
selected_color_style = st.sidebar.selectbox('Select Color Style', color_styles)
selected_style = st.sidebar.selectbox('Select Style', styles)
current_version = version_dictionary[selected_version]
if(current_version == "G"):
    st.markdown("## Vibrational Modes ")
    st.markdown('<span style="font-size: 18px;"> In chemistry, the study of molecular vibrations plays a critical role in understanding the chemical and physical properties of molecules. The vibrational spectroscopy techniques of IR (infrared) and Raman spectroscopy are used extensively to study the vibrational properties of molecules. Group theory and symmetry principles provide a powerful framework for understanding the vibrational modes of molecules and predicting their corresponding IR and Raman spectral bands. </span>', unsafe_allow_html=True)
    st.markdown("""
        -	For a set of N atoms, there is 3N degrees of freedom in a 3-dimensional space
        -	3 of these correlate translational motion: Trans(XYZ)
        -	3 more correspond to rotational motion Rot(RxRyRz) and 2 in linear molecules.
        -	Thus, each molecule has 3N-6 (3N-5 for linear molecules) normal modes of vibration
    """)
elif(current_version == "UG"):
    st.markdown("## Selected Strectching Vibrational Modes  ")
    st.markdown('<span style="font-size: 18px;"> Think of stretching vibrations as molecular yoga poses! Similar to how various yoga postures stretch and flex various body regions, various stretching vibrations in molecules stretch and flex various chemical bonds. IR and Raman vibrational frequencies are important for characterizing functional groups and chemical properties. Scientists can use IR and Raman spectra to learn more about molecular yoga poses and properties, and possibly find a new way to stretch molecules. </span>', unsafe_allow_html=True)

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

# st.markdown("""
#     <style>
#     .sidebar .sidebar-content {
#         width: 300px;
#     }
#     @media (max-width: 500px) {
#         .sidebar .sidebar-content {
#             width: 100%;
#         }
#     }
#     </style>
#     """, unsafe_allow_html=True)
  
def geometry_change():
    # Get the current date and time
    # now = datetime.datetime.now() 

    # # Format the date and time
    # formatted_datetime = now.strftime("%m-%d-%Y %H:%M:%S")
    # print(f"NEW RUN ({formatted_datetime})---------------------------------------------------")
    
    
    # st.markdown(f"Selected version: {current_version}")
    
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
    
    # print(f"number of atoms: {natoms_ug} \natoms size og {np.size(atoms)} atoms size ug {np.size(atoms_ug)}")
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
    
    line_width = 3
    # Add the axes of symmetry
    view.addCylinder({
        'start': {'z': 0, 'y': 0, 'x': -line_width},
        'end': {'z': 0, 'y': 0, 'x': line_width},
        'radius': 0.05,
        'color': 'red'
    })
    view.addCylinder({
        'start': {'x': 0, 'z': 0, 'y': -line_width},
        'end': {'x': 0, 'z': 0, 'y': line_width},
        'radius': 0.05,
        'color': 'green'
    })
    view.addCylinder({
        'start': {'x': 0, 'y': 0, 'z': -line_width},
        'end': {'x': 0, 'y': 0, 'z': line_width},
        'radius': 0.05,
        'color': 'blue'
    })

    view.zoomTo()
    # axes_var = {"origin": {"x": 0, "y": 0, "z": 0},
    #             "axes": [{
    #             "start": {"x": -10, "y": 0, "z": 0},
    #             "end": {"x": 10, "y": 0, "z": 0},
    #             "radius": 0.1,
    #             "color": "red"
    # }]
    #                 }
    # view.addAxes(axes_var)
    # viewer_html = view.getHTML()
    showmol(view, height=500, width = 600) 
    
    # TO-DO: understand this code to actually properly scale putting them in a container
    # viewer_container = st.container()
    # st.markdown("""
    # <style>
    #     .viewer-container {
    #         width: 100%;
    #         padding-bottom: 75%; /* Adjust this value to change the aspect ratio */
    #         position: relative;
    #     }
    # </style>
    # """, unsafe_allow_html=True)

    # with viewer_container:
    #     st.markdown('<div class="viewer-container">', unsafe_allow_html=True)
    #     viewer = py3Dmol.view(query='pdb:1A2C')
    #     viewer.setStyle({'cartoon':{'color':'spectrum'}})
    #     showmol(viewer, height=500, width=600)
    #     st.markdown('</div>', unsafe_allow_html=True)
    
    latex_title = [f"${format_char_table_header(col)}$" for col in char_table_raw_df]
    
    char_table_raw_df.iloc[:, 0] = char_table_raw_df.iloc[:, 0].apply(format_subscript_latex) 
    char_table_raw_df.iloc[:, 0] = char_table_raw_df.iloc[:, 0].apply(lambda x: f"${x}$" if isinstance(x, str) else x)

    char_table_latex_df = char_table_raw_df
    char_table_latex_df.columns = latex_title
    
    char_table_latex_df["$Quadratic$"] = format_superscripts_from_df_column(char_table_latex_df, "$Quadratic$")
    char_table_latex_df["$Rotational$"] = format_rotational_latex(char_table_latex_df, "$Rotational$")
    char_table_latex_df_no_na = char_table_raw_df[:-2].fillna("")
    # symmetry_operations_latex = char_table_latex_df.iloc[:0, 1:-3]
    symmetry_operations_latex = char_table_latex_df.columns[1:-3]
    # symmetry_operations_latex.reset_index()
    # for op in symmetry_operations_latex:
    #     st.markdown(op)
    # # print(f"data: \n{symmetry_operations_latex}")
    # st.markdown(symmetry_operations_latex)
    
    # Create a container for the buttons and viewer
    container = st.container()
    # Assuming you have your DataFrame 'df' already loaded
    # button_columns = df.columns[:-3]  # Exclude the last 3 columns

    # Create buttons in the container
    # with container:
    #     # for col in symmetry_operations_latex:
    #     #     value = char_table_latex_df.iloc[0][col]
    #     #     print(f"value")
    #     #     if st.button(f"\({value}\)", key=f"{col}"):  # Wrap the value in LaTeX inline math delimiters
    #     #         # Button action here
    #     #         st.write(f"Button \({value}\) clicked")
    #     # Get the first row of the DataFrame, excluding the last 3 columns
    #     button_data = char_table_latex_df.iloc[0, 1:-3]
        
    #     # Create a horizontal layout for buttons
    #     cols = st.columns(len(button_data))
        
    #     for i, (col_name, value) in enumerate(button_data.items()):
    #         with cols[i]:
    #             # Use LaTeX rendering for button labels
    #             button_label = f"$${col_name}$$"
    #             # st.markdown(f"button label: {button_label}, colname: {col_name}")
    #             if st.button(button_label, key=f"button_{i}"):
    #                 # Handle button click
    #                 st.write(f"Clicked: {col_name}")
    #                 # Create a copy of the model for rotation
    #                 # rotated_view = view.clone()
    #                 view.animate({'axis': [1, 1, 0], 'angle': 45, 'step': 1, 'ms': 1000})
    #                 # view.rotate(45, {"x": 1, "y": 1, "z": 0})
    #                 # view.render()
    #     showmol(view, height=500, width=800)
        
    
    char_table_markdown = char_table_latex_df_no_na.to_markdown(index = False)
    st.markdown(f"## Character table")

    #TO-DO: figure out how to do dynamic scaling of tables
    # scaled_table = f"""
    # <div style="font-size: 0.8em;">

    # {table_markdown}

    # </div>
    # """
    # st.markdown("""
    # <style>
    #     .responsive-table {
    #         width: 100%;
    #         overflow-x: auto;
    #     }
    #     .responsive-table table {
    #         width: 100%;
    #         min-width: 400px;
    #     }
    # </style>
    # """, unsafe_allow_html=True)

    # responsive_table = f"""
    # <div class ="responsive-table">

    # {char_table_latex_df_no_na}

    # </div>
    # """

    # st.markdown(responsive_table, unsafe_allow_html=True)
    # st.dataframe(char_table_latex_df_no_na, use_container_width=True)

    # st.markdown(scaled_table)
    # st.markdown("""
    # <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
    # """, unsafe_allow_html=True)

    # html_content = f"""
    # <div style="font-size: 100%; transform-origin: 0 0; transform: scale(0.8);">

    # {table_markdown}

    # </div>
    # """
    # st.markdown(html_content, unsafe_allow_html=True) 

    char_table_var = f"""{char_table_markdown}"""

    st.markdown(char_table_var, unsafe_allow_html=True)
    if(current_version == "UG"):
        st.markdown("The number of irreducible representations is usually calculated by taking the sum of the products of the number of unmoved vectors, the coefficient of each symmetry operation, and the character of the irreducible representation and dividing the sum by the order of the group.")
    
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
        gamma_total_equation = r"$\Gamma_{tot} = \Gamma_{trans} + \Gamma_{vib} + \Gamma_{rot}$"
        st.markdown("#### Total reducible representation for all degrees of freedom")
        st.markdown(gamma_total_equation)
        st.markdown(gamma_total_table.to_markdown(index = False))
        st.markdown(f"{gamma_xyz_label} = Atomic contributions by symmetry operations to the reducible representations.")
        st.markdown(f"$\#_{{{number_unmoved_label_ug}}}$ = Number of unmoved atoms after symmetry operation.")
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
        st.markdown(f"Break down the total reducible representation into irreducible representations (M > 0) by multiplying with each character of the character table and dividing the sum by the order of the group. Then, subtract translational and Rotational to obtain Vibrational modes.")
        st.markdown(irreducible_table_no_na.to_markdown(index = False))
        st.markdown(f"{irreducible_formula}")
    
    gamma_vib_label_ug = r"$\Gamma_{vibration}$"
    
    # print(f"uwu ug: {char_table_raw_df}")
    irreducible_table_ug = calculate_irreducible_representations(copy.copy(char_table_raw_df), order, unmoved_atoms_ug, atomic_contribution_symm, symmetry_coefficients, "undergrad")
    
    irreducible_table_ug = irreducible_table_ug[:-1]
    irreducible_table_no_na_ug = irreducible_table_ug.fillna("")
    irreducible_formula_ug = f"{gamma_vib_label_ug} = {gamma_formula_notation(irreducible_table_ug)}"
    
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