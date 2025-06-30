# This file is to hold all variables/constants throughout the app

# Import Python Packages
import streamlit as st
from datetime import datetime, timedelta
import os


# --------------------------------------------------- Time Parameters -----------------------------------------------------------------

# Class for Re-Usable Date Vars
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


# --------------------------------------------------- Re-usable App Components -----------------------------------------------------------------

class GradeColors:

    # Dictionary mapping each grade to a pair of RGBA color values:
    # (background_color, outline_color)
    grade_color_map = {
        "S":  ("rgba(212, 175, 55, 0.85)", "rgba(212, 175, 55, 1.0)"),
        "A":  ("rgba(34, 139, 84, 0.85)", "rgba(34, 139, 84, 1.0)"),
        "A-": ("rgba(50, 160, 75, 0.85)", "rgba(50, 160, 75, 1.0)"),
        "B+": ("rgba(72, 201, 137, 0.85)", "rgba(72, 201, 137, 1.0)"),
        "B":  ("rgba(102, 205, 170, 0.85)", "rgba(102, 205, 170, 1.0)"),
        "B-": ("rgba(144, 238, 144, 0.85)", "rgba(144, 238, 144, 1.0)"),
        "C+": ("rgba(205, 220, 57, 0.85)", "rgba(205, 220, 57, 1.0)"),
        "C":  ("rgba(255, 215, 0, 0.85)", "rgba(255, 215, 0, 1.0)"),
        "C-": ("rgba(255, 165, 0, 0.85)", "rgba(255, 165, 0, 1.0)"),
        "D+": ("rgba(255, 140, 0, 0.85)", "rgba(255, 140, 0, 1.0)"),
        "D":  ("rgba(255, 80, 60, 0.85)", "rgba(255, 80, 60, 1.0)"),
        "D-": ("rgba(255, 99, 71, 0.85)", "rgba(255, 99, 71, 1.0)"),
        "F":  ("rgba(200, 30, 30, 0.85)", "rgba(200, 30, 30, 1.0)"),
    }

    # Fallback color used when the grade is not recognized
    default_color = ("rgba(128, 128, 128, 0.5)", "rgba(128, 128, 128, 1.0)")

    @staticmethod
    def compare_performance(grade):
        """
        Given a letter grade (e.g., "A", "B+", "F"), returns a tuple:
        (background_color_rgba, outline_color_rgba)

        If the grade is not found in the mapping, returns a default gray color.
        This function is case-sensitive and expects valid grade keys.

        Example:
            GradeColors.compare_performance("B+")
            ➜ ("rgba(72, 201, 137, 0.85)", "rgba(72, 201, 137, 1.0)")
        """
        return GradeColors.grade_color_map.get(grade, GradeColors.default_color)


# Class for Re-Usable App Components
class ExtraComponents:

    # component for sidebar drop-downs with notes
    @staticmethod
    def get_sidebar_notes():

        # Drop-downs with Notes on Sidebar:
        st.sidebar.header("Financial Ratio Notes")
        with st.sidebar.expander("MARKET VALUE RATIOS"):
            st.write("---Measure the current market price relative to its value---")
            st.write('<span style="color: lightcoral;">PE Ratio: [Market Price per Share / EPS]</span>',
                     unsafe_allow_html=True)  # adding the additional HTML code allows us to change the text color in the write statement
            st.markdown(
                "[AVG PE Ratio by Sector](https://fullratio.com/pe-ratio-by-industry)")  # Insert a link on the sidebar to avg PE ratio by sector
            st.write(
                "Ratio Notes: PE should be evaluated and compared to competitors within the sector. A PE over the avg "
                "industry PE might indicate that the stock is selling at a premium, but may also indicate higher expected "
                "growth/trade volume; A lower PE may indicate that the stock is selling at a discount, but may also indicate "
                "low growth/trade volume.")
            st.write('<span style="color: lightcoral;">PEG Ratio: [PE / EPS Growth Rate]</span>', unsafe_allow_html=True)
            st.write("Ratio Notes: PEG > 1 = Likely overvalued || PEG < 1 = Likely undervalued")
            st.write(
                '<span style="color: lightcoral;">Price-to-Book Ratio: [Market Price per Share / Book Value Per Share]</span>',
                unsafe_allow_html=True)
            st.write(
                "Ratio Notes: PB > 1 = Indicates stock might be overvalued copared to its assets || PB < 1 = Indicates stock "
                "might be undervalued copared to its assets || Typically not a good indicator for companies with intangible "
                "assets, such as tech companies.")
            st.write('<span style="color: lightcoral;">Price-to-Sales Ratio: [Market Cap / Revenue]</span>',
                     unsafe_allow_html=True)
            st.write(
                "Ratio Notes: 2-1 = Good || Below 1 = Better || Lower = Indicates the company is generating more revenue for "
                "every dollar investors have put into the company.")

        with st.sidebar.expander("PROFITABILITY RATIOS"):
            st.write("---Measure the combined effects of liquidity, asset mgmt, and debt on operating results---")
            st.write('<span style="color: lightcoral;">ROE (Return on Equity): [Net Income / Common Equity]</span>',
                     unsafe_allow_html=True)
            st.write("Ratio Notes: Measures total return on investment | Compare to the stock's sector | Bigger = Better")

        with st.sidebar.expander("LIQUIDITY RATIOS"):
            st.write("---Measure the ability to meet current liabilities in the short term (Bigger = Better)---")
            st.write('<span style="color: lightcoral;">Current Ratio: [Current Assets / Current Liabilities]</span>',
                     unsafe_allow_html=True)
            st.write(
                "Ratio Notes: Close to or over 1 = Good || Over 1 means the company is covering its bills due within a one "
                "year period")
            st.write(
                '<span style="color: lightcoral;">Quick Ratio: [(Current Assets - Inventory) / Current Liabilities]</span>',
                unsafe_allow_html=True)
            st.write(
                "Ratio Notes: Close to or over 1 = Good || Over 1 means the company is able to cover its bills due within a "
                "one year period w/ liquid cash")

        with st.sidebar.expander("ASSET MANAGEMENT RATIOS"):
            st.write("---Measure how effectively assets are being managed---")
            st.write('<span style="color: lightcoral;">Dividend Yield: [DPS / SP]</span>', unsafe_allow_html=True)
            st.write(
                "Ratio Notes: For low growth stocks, should be higher and should look for consistent div growth over time- "
                "with signs of consistenly steady financials (able to pay debts consistently; want to see the company is "
                "managing its money well) || For growth stocks, typically lower, but if a stock shows high growth over time "
                "w/ a dividend yield that continues to remain the same or grow over time, this is a good sign (good to "
                "compare with their Current Ratio)")

        with st.sidebar.expander("DEBT MANAGEMENT RATIOS"):
            st.write("---Measure how well debts are being managed---")
            st.write('<span style="color: lightcoral;">Debt-to-Equity: [Total Liabilities / Total Shareholder Equity]</span>',
                     unsafe_allow_html=True)
            st.write(
                "Ratio Notes: A good D/E ratio will vary by sector & company. Typically a 1.0-1.5 ratio is ideal. The main "
                "thing to look for is that if the company is leveraging debt is that it has enough liquidity and consistent "
                "return to pay off those debts. Leveraging debt (when managed well) can be a good indicator that a growth "
                "company is leveraging debt in order to re-invest and grow faster, which is typically a good sign that the "
                "company is strategically well managed.")

        with st.sidebar.expander("PERFORMANCE/RISK RATIOS"):
            st.write("---Measures performance in the market against a certain benchmark---")
            st.write('<span style="color: lightcoral;">Beta:</span>', unsafe_allow_html=True)
            st.write("Ratio Notes: Beta measures the volatility of an investment relative to the overall market or benchmark "
                     "index. Beta > 1 = more volatile; Beta < 1 = less volatile.")
            st.write('<span style="color: lightcoral;">Sharpe Ratio: [(Return - RFR) / SD of Returns]</span>', unsafe_allow_html=True)
            st.write("Ratio Notes: Sharpe Ratio measures the level of adjusted-risk to return of an investment against the "
                     "current risk-free rate. The higher the ratio, the better overall return the asset provides against the "
                     "level of risk taken investing into the asset. A Sharpe Ratio > 1 = good; > 2 = very good.")



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
