# app_main.py

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

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Navigation Pages
home_page = st.Page("app_home_pg.py", title="Home", icon=":material/home:")
members_page = st.Page("app_add_members_pg.py", title="Team Member Directory", icon=":material/group:")
routes_page = st.Page("app_update_routes_pg.py", title="Update Routes", icon=":material/edit_road:")
collections_page = st.Page("app_log_collections_pg.py", title="Log Collections", icon=":material/payments:")

# Set up navigation with position="top" for horizontal nav bar
pg = st.navigation([home_page, members_page, routes_page, collections_page], position="top")

# Run the selected page
pg.run()
