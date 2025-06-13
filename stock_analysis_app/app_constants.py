# This file is to hold all variables/constants throughout the app

# Import Python Packages
import streamlit as st
from datetime import datetime, timedelta
import os


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

# Check if running in GitHub Actions or locally
if os.getenv('GITHUB_ACTIONS'):
    # Running in GitHub Actions - use environment variables
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
else:
    # Alpha Vantage API
    ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY")
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
