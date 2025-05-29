import streamlit as st
import sys
import os

# Must be the first Streamlit command
st.set_page_config(
    page_title="Brightside Route Optimizer",
    page_icon="🚗",
    initial_sidebar_state="expanded",
    menu_items={
        'About': '# Brightside Route Optimizer\nThis app helps optimize delivery routes for Brightside PWYC.'
    }
)

# Add the directory containing the page files to the Python path
# Assuming page files are in the same directory as app_main.py or a subdirectory
# Adjust the path if needed based on the actual file structure
sys.path.append(os.path.dirname(__file__))

# Import the page functions from other files
# Ensure the function names match what's defined in the page files
# In app_home_pg.py, the function is named `app_home_page`
# In app_add_members_pg.py, the function is named `app_add_members_page`
from app_home_pg import app_home_page
from app_add_members_pg import app_add_members_page

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Add sidebar navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Navigation</h1>", unsafe_allow_html=True)
    st.write("---")

    # Home button
    if st.button("Home", use_container_width=True):
        st.session_state.current_page = 'home'
        st.rerun()
    
    # Add Members button
    if st.button("Team Member Directory", use_container_width=True):
        st.session_state.current_page = 'add_members'
        st.rerun()

# Navigation
if st.session_state.current_page == 'home':
    app_home_page()
else:
    app_add_members_page()