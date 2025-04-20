# This file is to hold all variables/constants throughout the app

# Import Python Packages
import streamlit as st
from datetime import datetime, timedelta


# --------------------------------------------------- Data: Set Time Parameters -----------------------------------------------------------------

# Create a class for the date variables
# Importing the date vars as a class will be cleaner than importing each as its own var
class DateVars:
    def __init__(self):
        # Initialize today's date
        self.today = datetime.today()

        # Initialize other date-related variables
        self.yesterday = self.today - timedelta(days=1)
        self.start_date = self.today.replace(year=self.today.year - 10)
        self.three_yrs_ago = self.today.replace(year=self.today.year - 3)
        self.one_yr_ago = self.today.replace(year=self.today.year - 1)
        self.current_yr = datetime.now().year
        self.last_yr = self.current_yr - 1
        self.current_mth = datetime.now().month
        self.current_day = datetime.now().day


# --------------------------------------------------- API Keys -----------------------------------------------------------------

# Alpha Vantage API
alpha_vantage_key = st.secrets.get("Alpha_Vantage_API_Key")
groq_key = st.secrets.get("Groq_API_Key")
