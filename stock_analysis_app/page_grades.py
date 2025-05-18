import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime
import boto3
from io import StringIO

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
                        <h1 style="font-size: 32px; margin: 0;">Stock Grades</h1>
                    </div>
                    """,
    unsafe_allow_html=True
)

# Add a divider
st.write("---")

# Create Container
sh_g = st.container()


# Function to Load CSV from S3 using credentials
def load_csv_from_s3(bucket_name, file_key):
    """
    Load the CSV file from S3 into a pandas DataFrame.

    Parameters:
    bucket_name (str): The name of the S3 bucket.
    file_key (str): The S3 file key (path).

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    # Retrieve credentials from Streamlit secrets
    aws_access_key_id = st.secrets["aws_access_key_id"]
    aws_secret_access_key = st.secrets["aws_secret_access_key"]
    aws_region = st.secrets["aws_region"]

    # Create an S3 client using the credentials from secrets
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )

    # Get the object from S3
    obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    data = obj['Body'].read().decode('utf-8')  # Read the content of the CSV
    df = pd.read_csv(StringIO(data))  # Convert the content into a pandas DataFrame
    return df

# Replace with your actual S3 bucket and file key
bucket_name = 'stock-ticker-data-bucket'  # S3 bucket name
file_key = 'ticker_grades_output.csv'  # name of object in S3 bucket

# Load the grades data from the S3 bucket
grades_df = load_csv_from_s3(bucket_name, file_key)

# Ticker Search Section
sh_g.write("Ticker Grade Search:")

# Write Grades Search to App
with sh_g.container():

    # Add a section for ticker grades
    def add_ticker_grades_section(grades_df):
        """
        Add a ticker grades section to your Streamlit app using the provided DataFrame.

        Parameters:
        grades_df (DataFrame): DataFrame containing stock grades and other data
        """

        # Check if grades_df is empty
        if grades_df.empty:
            st.error("No grades data available.")
            return

        # Add filters
        col1, col2, col3, col4 = st.columns(4)

        with col2:
            # Filter by grade
            all_grades = ['All'] + sorted(grades_df['Grade'].unique().tolist())
            selected_grade = st.selectbox("Filter by Grade", all_grades)

        with col3:
            # Filter by industry
            all_industries = ['All'] + sorted(grades_df['Industry'].unique().tolist())
            selected_industry = st.selectbox("Filter by Industry", all_industries)

        with col4:
            # Sort options
            sort_options = ['Grade (Best First)', 'Grade (Worst First)', 'Score (High to Low)', 'Ticker (A-Z)']
            sort_selection = st.selectbox("Sort by", sort_options)

        with col1:
            ticker_search = st.text_input("Search Ticker (e.g. AAPL)", value="")

        # Apply filters
        filtered_df = grades_df.copy()

        if selected_grade != 'All':
            filtered_df = filtered_df[filtered_df['Grade'] == selected_grade]

        if selected_industry != 'All':
            filtered_df = filtered_df[filtered_df['Industry'] == selected_industry]

        if ticker_search.strip():  # removes spaces
            filtered_df = filtered_df[filtered_df['Ticker'].str.contains(ticker_search.strip(), case=False)]

        # Apply sorting
        if sort_selection == 'Grade (Best First)':
            grade_order = {'S': 0, 'A': 1, 'A-': 2, 'B+': 3, 'B': 4, 'B-': 5, 'C+': 6, 'C': 7, 'C-': 8,
                           'D+': 9, 'D': 10, 'D-': 11, 'F': 12, 'Error': 13}
            filtered_df['grade_order'] = filtered_df['Grade'].map(grade_order)
            filtered_df = filtered_df.sort_values('grade_order')
            filtered_df = filtered_df.drop(columns=['grade_order'])
        elif sort_selection == 'Grade (Worst First)':
            grade_order = {'S': 0, 'A': 1, 'A-': 2, 'B+': 3, 'B': 4, 'B-': 5, 'C+': 6, 'C': 7, 'C-': 8,
                           'D+': 9, 'D': 10, 'D-': 11, 'F': 12, 'Error': 13}
            filtered_df['grade_order'] = filtered_df['Grade'].map(grade_order)
            filtered_df = filtered_df.sort_values('grade_order', ascending=False)
            filtered_df = filtered_df.drop(columns=['grade_order'])
        elif sort_selection == 'Score (High to Low)':
            filtered_df = filtered_df.sort_values('Score', ascending=False)
        else:  # Ticker (A-Z)
            filtered_df = filtered_df.sort_values('Ticker')

        st.write("---")
        # Define a function to color the grades
        def color_grades(val):
            grade_colors = {
                "S": "background-color: rgba(255, 215, 0, 0.5)",  # Gold
                "A": "background-color: rgba(34, 139, 34, 0.5)",  # Forest green
                "A-": "background-color: rgba(50, 205, 50, 0.5)",  # Lime green
                "B+": "background-color: rgba(60, 179, 113, 0.5)",  # Medium sea green
                "B": "background-color: rgba(102, 205, 170, 0.5)",  # Medium aquamarine
                "B-": "background-color: rgba(152, 251, 152, 0.5)",  # Pale green
                "C+": "background-color: rgba(173, 255, 47, 0.5)",  # Green yellow
                "C": "background-color: rgba(252, 226, 5, 0.5)",  # Bumblebee
                "C-": "background-color: rgba(255, 165, 0, 0.5)",  # Orange
                "D+": "background-color: rgba(255, 140, 0, 0.5)",  # Dark orange
                "D": "background-color: rgba(255, 69, 0, 0.5)",  # Orange red
                "D-": "background-color: rgba(255, 99, 71, 0.5)",  # Tomato
                "F": "background-color: rgba(255, 0, 0, 0.5)",  # Red
                "Error": "background-color: rgba(128, 128, 128, 0.5)"  # Gray for errors
            }

            if val in grade_colors:
                return grade_colors[val]
            return ""

        st.write("Ticker Model Grades / Scores:")
        with st.container(border=True):
            # Display dataframe of ticker grade data
            st.dataframe(filtered_df.style.applymap(color_grades, subset=['Grade']), use_container_width=True)

        st.write("Grade Distributions:")
        with st.container(border=True):
            # Show grade distribution of tickers
            grade_counts = grades_df['Grade'].value_counts().sort_index()
            st.bar_chart(grade_counts)

    # Write Custom Model Grade to App
    add_ticker_grades_section(grades_df)


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

# ---- TEST BLOCK ----
if __name__ == "__main__":

    print(grades_df.head())
