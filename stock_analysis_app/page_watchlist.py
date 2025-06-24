import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime
import boto3

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app methods
from stock_analysis_app.app_constants import DateVars
from stock_analysis_app.app_data import AppData
from stock_analysis_app.app_animations import CSSAnimations
from stock_analysis_app.app_stock_grading_model import StockGradeModel

# Instantiate any imported classes here:
dv = DateVars()
animation = CSSAnimations()
data = AppData()
model = StockGradeModel()

# Provide Page Title
st.markdown(
    f"""
                    <div style="display: flex; justify-content: center; align-items: center;">
                        <h1 style="font-size: 32px; margin: 0;">Watch List</h1>
                    </div>
                    """,
    unsafe_allow_html=True
)

# Add a divider
st.write("---")

bucket_name = 'stock-ticker-data-bucket'  # S3 bucket name
csv_path = 'ticker_watchlist.csv'  # name of object in S3 bucket

# wrap grades_df in session state so data pull doesn't reload when navigating pages
if "watchlist_df" not in st.session_state:
    st.session_state["watchlist_df"] = data.load_csv_from_s3(bucket_name, csv_path)

# assign to session state var
watchlist_df = st.session_state["watchlist_df"]

# Check if the watchlist has been updated
if st.session_state.get("watchlist_updated", False):
    # Reload data
    watchlist_df = data.load_csv_from_s3('stock-ticker-data-bucket', 'ticker_watchlist.csv')

    # Reset the flag so it doesn't reload unnecessarily
    st.session_state.watchlist_updated = False
else:
    # Load once or from cache
    watchlist_df = data.load_csv_from_s3('stock-ticker-data-bucket', 'ticker_watchlist.csv')

# container title
st.write("Ticker Model Grades / Scores:")

# Create Container
sh_wl = st.container(border=True)

with sh_wl:
    st.dataframe(watchlist_df, use_container_width=True, hide_index=True)


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
