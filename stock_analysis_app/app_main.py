# Import libraries:
import streamlit as st  # Import streamlit (app Framework)
import sys
import os

# Add the parent directory to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Layout of App, Provide App Version and Provide Title
st.set_page_config(layout='wide')  # sets layout to wide
st.sidebar.markdown(
    "<div style='text-align: center; padding: 20px;'>App Version: 1.5 &nbsp; <span "
    "style='color:#FF6347;'>&#x271A;</span></div>",
    unsafe_allow_html=True)  # adds App version and red medical cross icon with HTML & CSS code; nbsp adds a space

# Set Navigation Pages Across Files
home_page = st.Page("page_home.py", title="Home", icon=":material/home:")
grade_page = st.Page("page_grades.py", title="Stock Grades", icon=":material/analytics:")
chat_page = st.Page("page_chatbot.py", title="Chat Bot", icon=":material/robot:")

# Set var for navigation method
pg = st.navigation([home_page, grade_page, chat_page])

# run
pg.run()





# ------------------------------------------------------------------


# Note: in order to view the app on your local host, you will need run the following code on your terminal: streamlit
# run [insert location of file here {ex: %/stockpredictorapp.py}] Note: this is the streamlit red color code: #FF6347

# Note: to load in historical data over a set time period for a ticker, you can call the load_price_hist_data
# static method data.load_price_hist_data(ticker, st_dt, end_dt) - st & end dt are optional else 10 yr period

# Note: to load in current ticker metrics, you can call the load_price_hist_data
# static method data.load_curr_stock_metrics(ticker)
