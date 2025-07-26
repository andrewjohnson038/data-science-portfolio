# Import libraries:
import streamlit as st

# Set Layout of App, Provide App Version and Provide Title
st.set_page_config()  # sets layout to wide

# Set Navigation Pages Across Files
home_pg = st.Page("home_pg.py", title="Home", icon=":material/home:")

# Set var for navigation method
pg = st.navigation([home_pg])

# run
pg.run()
