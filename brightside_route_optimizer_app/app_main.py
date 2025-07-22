
import streamlit as st
import sys
import os
import requests

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

# Add the PDF download section to sidebar
st.sidebar.write("Testing Resources:")
with st.sidebar.expander("Example PDF"):
    st.write("If testing, you can use the PDF below on Step 2:")
    # Replace with your actual GitHub raw URL
    pdf_url = "https://raw.githubusercontent.com/andrewjohnson038/data-science-portfolio/stock_analysis_app_testing/brightside_route_optimizer_app/resources/routes%20January%2026%202025.pdf"
    try:
        st.download_button(
            label="Download Routes PDF",
            data=requests.get(pdf_url).content,
            file_name="brightside_fake_test_routes.pdf",
            mime="application/pdf"
        )
    except:
        st.error("Could not load test PDF")

# Set Navigation Pages
home_page = st.Page("app_home_pg.py", title="Home", icon=":material/home:")
members_page = st.Page("app_add_members_pg.py", title="Team Member Directory", icon=":material/group:")

# Set up navigation with position="top" for horizontal nav bar
pg = st.navigation([home_page, members_page], position="top")

# Run the selected page
pg.run()
