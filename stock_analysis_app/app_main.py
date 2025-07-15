# Import libraries:
import streamlit as st  # Import streamlit (app Framework)
import sys
import os

# Add the parent directory to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Layout of App, Provide App Version and Provide Title
st.set_page_config(layout='wide')  # sets layout to wide
st.sidebar.markdown(
    "<div style='text-align: center; padding: 20px;'>App Version: 1.6 &nbsp; <span "
    "style='color:#FF6347;'>&#x271A;</span></div>",
    unsafe_allow_html=True)  # adds App version and red medical cross icon with HTML & CSS code; nbsp adds a space

# Set Navigation Pages Across Files
home_page = st.Page("page_home.py", title="Home", icon=":material/home:")
grade_page = st.Page("page_grades.py", title="Stock Grades", icon=":material/analytics:")
chat_page = st.Page("page_chatbot.py", title="Chat Bot", icon=":material/robot:")
watchlist_page = st.Page("page_watchlist.py", title="Watch List", icon=":material/list:")
portfolio_page = st.Page("page_portfolio.py", title="Portfolio", icon=":material/folder:")

# Set var for navigation method
pg = st.navigation([home_page, grade_page, watchlist_page, portfolio_page, chat_page])

# run
pg.run()
