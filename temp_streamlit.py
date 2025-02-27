import streamlit as st
import py3Dmol
from stmol import showmol
import streamlit.components.v1 as components
import pandas as pd
import json
# Set page config for better mobile experience
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Create sidebar
with st.sidebar:
    st.title("Sidebar")
    # Add sidebar controls here

# Main content
st.title("3D Molecule Viewer and Data")

# py3Dmol viewer
# viewer = py3Dmol.view(query='pdb:1A2C')
# viewer.setStyle({'cartoon': {'color': 'spectrum'}})
# viewer_html = viewer.render()
# Render py3Dmol viewer with responsive size
# viewer_html = f"""
# <div id="viewer" style="height: 400px; width: 100%;"></div>
# {viewer.render()}
# {screen_width_js}
# """
# components.html(viewer_html, height=450)

# Initialize session state for view reset
# if 'reset_view' not in st.session_state:
#     st.session_state.reset_view = False

# # Create a container for the viewer and button
# container = st.container()

# # Add a reset button above the viewer
# if container.button("Reset View"):
#     st.session_state.reset_view = True

# # Create the py3Dmol viewer
# view = py3Dmol.view(query='pdb:1ubq')
# view.setStyle({'cartoon': {'color': 'spectrum'}})

# # Reset the view if the button was clicked
# if st.session_state.reset_view:
#     view.zoomTo()
#     st.session_state.reset_view = False

# # Show the viewer in the container
# with container:
#     showmol(view, height=500, width=800)
# st.components.v1.html(viewer_html, height=400)

def create_viewer():
    view = py3Dmol.view(query='pdb:1ubq')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    return view
view = py3Dmol.view(query='pdb:1ubq')
view.setStyle({'cartoon': {'color': 'spectrum'}})
view.zoomTo()

def animate_rotation(view):
    view.animate({'axis': [0, 1, 0], 'angle': 180, 'step': 1}, 1000)

st.title("180-Degree Rotation Animation")

# if 'viewer' not in st.session_state:
#     st.session_state.viewer = create_viewer()
#     st.session_state.animate = False

if st.button("Rotate 180 Degrees"):
    st.session_state.animate = True

if st.session_state.animate:
    animate_rotation(st.session_state.viewer)
    st.session_state.animate = False

showmol(st.session_state.viewer, height=500, width=800)

###
# Create a DataFrame with rotation details
# df_rotations = pd.DataFrame({
#     'Label': [r'\alpha', r'\beta', r'\gamma', r'\delta'],
#     'Axis': ['x', 'y', 'z', 'xy'],
#     'Angle': [90, 180, 270, 360],
#     'Duration': [1000, 2000, 1500, 10000]
# })

# def create_buttons_and_rotations(df, view):
#     cols = st.columns(len(df))
#     for i, (_, row) in enumerate(df.iterrows()):
#         with cols[i]:
#             if st.button(f"$${row['Label']}$$", key=f"button_{i}"):
#                 animate_rotation(view, row)

# def animate_rotation(view, rotation_data):
#     print("owo")
#     axis = rotation_data['Axis']
#     angle = rotation_data['Angle']
#     duration = rotation_data['Duration']
    
#     if axis == 'xy':
#         axis = [1, 1, 0]
#     else:
#         axis = [1 if ax == axis else 0 for ax in ['x', 'y', 'z']]
    
#     view.animate({'axis': axis, 'angle': angle, 'step': 1}, duration)
#     print(f"uwu: {duration}")

# def create_viewer():
#     view = py3Dmol.view(query='pdb:1ubq')
#     view.setStyle({'cartoon': {'color': 'spectrum'}})
    
    
#     # Get the PDB data of the first model
#     # pdb_data = view.getModel().getPDB()
#     # # Serialize the PDB data
#     # serialized_data = json.dumps({"pdb_data": pdb_data})
#     # # Deserialize the data
#     # deserialized_data = json.loads(serialized_data)
#     # # Add a copy of the model using the PDB data
#     # view.addModel(deserialized_data["pdb_data"], "pdb")
#     view.addModel(query='pdb:1ubq', format='pdb')
#     view.setStyle({'model': -1}, {'cartoon': {'color': 'grey'}})
    
#     view.zoomTo()
#     return view

# # Create the viewer
# viewer = create_viewer()

# # Create a container for buttons and viewer
# container = st.container()

# # Add buttons and viewer to the container
# with container:
#     create_buttons_and_rotations(df_rotations, viewer)
#     showmol(viewer, height=500, width=800)
#     # viewer.animate({'axis': [1, 1, 0], 'angle': 45, 'step': 1, 'ms': 10000})


# Markdown table
st.markdown("""
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |
""")


# Custom CSS for responsive design
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stTable {
        width: 100%;
        overflow-x: auto;
    }
    @media (max-width: 768px) {
        .reportview-container .main .block-container {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)



# JavaScript to get screen width and adjust viewer size
screen_width_js = """
<script>
    var screenWidth = window.innerWidth;
    var viewerWidth = Math.min(screenWidth - 40, 600);
    var viewerHeight = Math.min(viewerWidth * 0.75, 400);
    document.getElementById('viewer').style.width = viewerWidth + 'px';
    document.getElementById('viewer').style.height = viewerHeight + 'px';
</script>
"""



st.markdown("""
<style>
    .table-container {
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
</style>
<div class="table-container">

| Column 1 | Column 2 | Column 3 | ... | Column 10 |
|----------|----------|----------|-----|-----------|
| Data 1   | Data 2   | Data 3   | ... | Data 10   |
| Data 11  | Data 12  | Data 13  | ... | Data 20   |

</div>
""", unsafe_allow_html=True)
