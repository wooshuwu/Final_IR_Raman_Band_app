import streamlit as st
import py3Dmol
import streamlit.components.v1 as components

# Set page config for better mobile experience
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Create sidebar
with st.sidebar:
    st.title("Sidebar")
    # Add sidebar controls here

# Main content
st.title("3D Molecule Viewer and Data")

# py3Dmol viewer
viewer = py3Dmol.view(query='pdb:1A2C')
viewer.setStyle({'cartoon': {'color': 'spectrum'}})
viewer_html = viewer.render()
# st.components.v1.html(viewer_html, height=400)

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

# Render py3Dmol viewer with responsive size
viewer_html = f"""
<div id="viewer" style="height: 400px; width: 100%;"></div>
{viewer.render()}
{screen_width_js}
"""
components.html(viewer_html, height=450)

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
