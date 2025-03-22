# uncomment pip install if the libraries are not installed
# pip install pandas
# pip install streamlit
# pip install plotly
# install fbprophet
# pip install yfinance

# Import the libraries we will use:
# import Python's datetime library (used to manipulate date and time object data)
from datetime import datetime, timedelta
# import streamlit library (open source library for specfic to data science to quickly build an application front end for the data with minimal python code)
import streamlit as st
# import yfinance library (used to retrieve financial data from yahoo finance)
import yfinance as yf
# import live pricing from yahoo finance
# import prophet library (prophet is an open source time series analysis module we will use with plotly to analyze and predict our stock data)
from prophet import Prophet
from prophet.plot import plot_plotly
# import plotly graph_objs module (used to plot our time series data within visuals)
from plotly import graph_objs as go
# import pandas library for dfs
import pandas as pd
# import numpy lib
import numpy as np
# import matplotlib lib
import matplotlib.pyplot as plt
# import Huggingface transformers model for gpt chat bot
from groq import Groq
# import lib for applying exponential smoothing line
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# import Beautiful Soup for Web Scraping
from bs4 import BeautifulSoup

import requests
import json
import sys

# for alpha vantage api
alpha_vantage_key = st.secrets.get("Alpha_Vantage_API_Key")


# Note: in order to view the app on your local host, you will need run the following code on your terminal: streamlit run [insert location of file here {ex: %/stockpredictorapp.py}]
# Note: this is the streamlit red color code: #FF6347



# Note: stock_data var = historical data of the stock over 10 years / stock_info var = stock metrics as of current date
# Note: to load in historical data over different time periods, you can call the load_data function that's set






# ////////////////////////////////////////////////// Configure App Layout /////////////////////////////////////////////////////////////////////////


# Set Layout of App, Provide App Version and Provide Title
st.set_page_config(layout='wide')  # sets layout to wide
st.sidebar.markdown("<div style='text-align: center; padding: 20px;'>App Version: 1.3.7.3 &nbsp; <span style='color:#FF6347;'>&#x271A;</span></div>", unsafe_allow_html=True) # adds App version and red medical cross icon with HTML & CSS code; nbsp adds a space
st.sidebar.header('Choose Stock & Forecast Range')  # provide sidebar header


# ////////////////////////////////////////////////// Configure App Layout /////////////////////////////////////////////////////////////////////////




















# /////////////////////////////////////////////  Pull in our data and set up datasets for use in app  ////////////////////////////////////////////////////////


# --------------------------------------------------- Data: Set Time Parameters -----------------------------------------------------------------

print("\n Today Date:")
today = datetime.today()  # retrieve today's data in sme format as above. date.today() retrieves the current date; apartof datetime module in Python. .strftime("%y-%m-%d") converts the date object to a string with the above format
print(today)

print("\n Yesterday Date:")
yesterday = today - timedelta(days=1)  # if we need to write code for the day before today, can use this
print(yesterday)

print("\n 10 Yrs Ago Date:")
start_date = today.replace(year=today.year - 10)  # retrieve data starting on this date 10 years ago
print(start_date)

print("\n 10 Yrs Ago Date:")
start_date_three_y_ago = today.replace(year=today.year - 3)  # retrieve data starting on this date 10 years ago
print(start_date)

# Get Current Year #
Current_Year = datetime.now().year
print("\n CURRENT Year:")
print(Current_Year)

# Calculate Previous Year #
Last_Year = Current_Year - 1
print("\n LAST Year:")
print(Last_Year)

# Get Current Month #
Current_Month = datetime.now().month
print("\n CURRENT MONTH:")
print(Current_Month)

# Get Current Day #
print("\n CURRENT DAY:")
Current_Day = datetime.now().day
print(Current_Day)

# Get Day a Year Ago From Today (YOY)
print("\n YOY:")
one_year_ago = today - timedelta(days=365)
print(one_year_ago)

# Get three years from today
three_years_ago = today - timedelta(days=3*365)

# --------------------------------------------------- Data: Set Time Parameters -----------------------------------------------------------------


# --------------------------------------------------- Animation Variables -----------------------------------------------------------------

# Function to generate the CSS for the cogwheel with adjustable size
def cog_wheel_css(size=30):
    return f"""
    <style>
    /* CSS for cog wheel animation */
    @keyframes rotate {{
      from {{
        transform: rotate(0deg);
      }}
      to {{
        transform: rotate(360deg);
      }}
    }}
    
    .cog-container {{
      display: flex;
      justify-content: center;
      align-items: center;
      height: 50%;
    }}
    
    .cog {{
      width: {size}px;  /* Adjustable size */
      height: {size}px; /* Adjustable size */
      border-radius: 50%;
      border: 5px solid transparent;
      border-top-color: #FF6347; /* Set to Streamlit's red color */
      animation: rotate 1s linear infinite;
    }}
    </style>
    """

# Function to generate the HTML for the cogwheel
def cog_html():
    return """
    <div class="cog-container">
      <div class="cog"></div>
    </div>
    """

# Create warning sign animation for error handling
def create_warning_animation(size_factor=1.0):
    """
    Creates a minimal CSS/HTML animation of an exclamation point warning sign with Streamlit red colors.

    Parameters:
    size_factor (float): Multiplier for the size of the animation. Default is 1.0.
                         Values larger than 1.0 make the animation bigger, smaller than 1.0 make it smaller.

    Returns:
    tuple: A tuple containing (css_code, html_code) that can be used with st.markdown()
    """
    # Calculate size-adjusted values
    base_size = 120 * size_factor
    triangle_size = 100 * size_factor  # Keeping variable name but using for exclamation
    exclamation_height = 60 * size_factor
    exclamation_width = 12 * size_factor
    exclamation_dot_size = 12 * size_factor
    pulse_size = 90 * size_factor  # Smaller to prevent overlap
    font_size = 18 * size_factor

    # Define the CSS with size-adjusted values
    warning_animation_css = f"""
    <style>
        .warning-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 20px auto;
            text-align: center;
            max-width: 100%;
        }}
        
        .warning-sign {{
            position: relative;
            width: {base_size}px;
            height: {base_size}px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .exclamation-container {{
            position: relative;
            width: {exclamation_width * 2}px;
            height: {exclamation_height + exclamation_dot_size + 5 * size_factor}px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 2;
        }}
        
        .exclamation {{
            width: {exclamation_width}px;
            height: {exclamation_height}px;
            background-color: #FF4B5C; /* Streamlit Red */
            border-radius: {4 * size_factor}px;
        }}
        
        .exclamation-dot {{
            width: {exclamation_dot_size}px;
            height: {exclamation_dot_size}px;
            background-color: #FF4B5C; /* Streamlit Red */
            border-radius: 50%;
            margin-top: {5 * size_factor}px;
        }}
        
        .pulse {{
            position: absolute;
            border: {3 * size_factor}px solid rgba(255, 75, 92, 0.5); /* Streamlit Red with opacity */
            width: {pulse_size}px;
            height: {pulse_size}px;
            border-radius: 50%;
            animation: pulse 1.5s ease-out infinite;
            opacity: 0;
            z-index: 1;
        }}
        
        @keyframes pulse {{
            0% {{
                transform: scale(0.8);
                opacity: 0.6;
            }}
            100% {{
                transform: scale(1.5);
                opacity: 0;
            }}
        }}
        
        .blink {{
            animation: blink 1s ease-in-out infinite alternate;
        }}
        
        @keyframes blink {{
            0% {{
                opacity: 1;
            }}
            100% {{
                opacity: 0.5;
            }}
        }}
        
        .error-message {{
            font-size: {font_size}px;
            font-weight: 600;
            line-height: 1.5;
            color: #FF4B5C; /* Streamlit Red */
            max-width: {600 * size_factor}px;
            text-align: center;
        }}
    </style>
    """

    # Define the HTML
    warning_animation_html = """
    <div class="warning-container">
        <div class="warning-sign">
            <div class="pulse"></div>
            <div class="exclamation-container blink">
                <div class="exclamation"></div>
                <div class="exclamation-dot"></div>
            </div>
        </div>
    </div>
    """

    return warning_animation_css, warning_animation_html


# --------------------------------------------------- Animation Variables -----------------------------------------------------------------









# ------------------------------------------ Data: Load Ticker Volume/Pricing/Ratios (Historic & Current) ---------------------------------------

@st.cache_data()
def load_data(ticker, st_dt=start_date, end_dt=today):  # Define the "load data" function
    stock_data = yf.download(ticker, start=st_dt, end=end_dt)  # Fetch data from Yahoo Finance

    # Check the index structure before resetting
    print("Current stock_data index:")
    print(stock_data.index)

    # Reset the index to remove any MultiIndex structure (if exists)
    stock_data.reset_index(inplace=True)  # This will reset all levels of the index

    # Flatten MultiIndex columns if they exist
    stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in stock_data.columns]

    # Now you should have a simpler DataFrame with the 'Date', 'Close', 'High', 'Low', etc.

    # Fill in missing dates if any
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])  # Ensure 'Date' is a datetime object
    date_range = pd.date_range(start=stock_data['Date'].min(), end=stock_data['Date'].max())  # Complete date range
    date_range_df = pd.DataFrame({'Date': date_range})  # Create a DataFrame with the full date range

    # Merge and fill missing data with the last available values
    stock_data = pd.merge(date_range_df, stock_data, on='Date', how='left').ffill()  # Merge and forward fill

    # Convert 'Date' column to datetime (if necessary)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Calculate the daily price change (Close - previous day's Close)
    stock_data['Price Change'] = stock_data['Close'].diff()

    # Replace NaN in 'Price Change' with 0
    stock_data['Price Change'].fillna(0, inplace=True)

    # Calculate the percentage change in price
    stock_data['Percentage Change'] = (stock_data['Price Change'] / stock_data['Close'].shift(1)) * 100

    # Replace NaN in 'Price Change' with 0
    stock_data['Percentage Change'].fillna(0, inplace=True)

    return stock_data  # Return the cleaned and filled data


# retrieve nasdaq tickers:
nasdaq_ticker_json_link = 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.json' # create variable for link to github with daily updated tickers from nasdaq
nasdaq_stock_tickers = pd.read_json(nasdaq_ticker_json_link,  typ='series')  # read the json file in pandas as a series since there is no column title
nasdaq_stocks = nasdaq_stock_tickers.tolist()  # convert set to list

# retrieve nyse tickers:
nyse_ticker_json_link = 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.json'  # create variable for link to github with daily updated tickers from nyse
nyse_stock_tickers = pd.read_json(nyse_ticker_json_link, typ='series')  # read the json file in pandas as a series since there is no column title
nyse_stocks = nyse_stock_tickers.tolist()  # convert set to list

# Combine NASDAQ and NYSE tickers
combined_tickers = nasdaq_stocks + nyse_stocks

# Convert to a set to remove duplicates and then back to a list
distinct_tickers = list(set(combined_tickers))

# Remove abnormal tickers containing "^" from list and sort alphabetically
filtered_tickers = [ticker for ticker in distinct_tickers if "^" not in ticker]  # assigns each item as "ticker" and loops through each item to remove those with an "^" sign
filtered_tickers = sorted(filtered_tickers)  # sorts tickers alphabetically

# create variable for our cleaned ticker list
stocks = filtered_tickers

# create a dropdown box for our stock options (First is the dd title, second is our dd options)
selected_stock = st.sidebar.selectbox("Select Stock:", stocks)

# Error handling: # Fetch Stock Info and handle errors
try:
    stock_info = yf.Ticker(selected_stock).info  # Try to fetch stock information

    # Check if the response is valid (not empty)
    if not stock_info or "error" in stock_info:
        raise ValueError("Received an empty or invalid response.")

# Handle errors related to the request, invalid JSON, or empty response
except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:
    st.markdown(cog_wheel_css, unsafe_allow_html=True)  # Include cogwheel CSS

    # Make sure cog_html is a string, not a function
    cog_html_str = """
    <div class="cogwheel">
        <i class="fas fa-cog fa-spin fa-3x"></i>
    </div>
    """

    st.markdown(f"""
    <div id="error-message">
        <div style="display: flex; align-items: center;">
            <!-- Insert the cogwheel animation here -->
            {cog_html_str}
            <span style="font-size: 18px; font-weight: 600; line-height: 1.5;">
                An error occurred while fetching the data. This is likely due from updates to the data source API/package. 
                Hold on tight - app maintenance is in place!
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stop execution of any further script if the error occurs
    sys.exit()

# Retrieve additional metrics from Yahoo Finance API
company_name = stock_info['longName']

# Valuation Info
pe_ratio = stock_info.get('trailingPE', None)  # PE Ratio (Price to Earnings)
peg_ratio = stock_info.get('pegRatio', None)  # PEG Ratio (Price to Earnings Growth)

# Function to get PEG ratio from Alpha Vantage (Usually missing from yfinance)
def get_peg_from_alpha_vantage(ticker):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",  # This is the correct function for overview data
        "symbol": ticker,
        "apikey": alpha_vantage_key  # Your API key
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Return PEG Ratio if available, otherwise None
    return data.get("PEGRatio", None)

# If PEG ratio is missing, pull it from Alpha Vantage
if peg_ratio is None:
    peg_ratio = get_peg_from_alpha_vantage(selected_stock)  # note: metric comes as string from alphavantage
else:
    None

price_to_book = stock_info.get('priceToBook', None)  # Price to Book Ratio (Stock Price / Book Value per Share)
price_to_sales = stock_info.get('priceToSalesTrailing12Months', None)  # Price to Sales Ratio (Market Price per Share / Revenue per Share)

# Profitability / Liquidity / Health
quick_ratio = stock_info.get('quickRatio', None)  # Quick Ratio (Current Assets excluding Inventory / Current Liabilities)
current_ratio = stock_info.get('currentRatio', None)  # Current Ratio (Current Assets / Current Liabilities)
roe = stock_info.get('returnOnEquity', None)  # Return on Equity (Net Income / Shareholder's Equity)
debt_to_equity = stock_info.get('debtToEquity', None)  # Debt-to-Equity Ratio (Total Liabilities / Shareholders' Equity)
# if statement to divide doe by 100 without an error
if debt_to_equity is not None:
    debt_to_equity = debt_to_equity / 100
else:
    # Set to a default value if missing
    debt_to_equity = None
gross_profit = stock_info.get('grossProfits', None)  # Gross Profit (Revenue - Cost of Goods Sold)
net_income = stock_info.get('netIncomeToCommon', None)  # Net Income Available to Common Shareholders
net_profit_margin = stock_info.get('profitMargins', None)  # Net Profit Margin (Net Income / Revenue)
yoy_revenue_growth = stock_info.get('revenueGrowth', None)  # get YOY revenue growth

# Market Ownership
shares_outstanding = stock_info.get('sharesOutstanding', None)  # Number of shares outstanding
enterprise_value = stock_info.get('enterpriseValue', None)  # Enterprise Value (Total Value of the Company)

# Dividend Info
dividend_yield = stock_info.get('dividendYield', None)  # Dividend Yield (Annual Dividend / Stock Price)
dividend_rate = stock_info.get('dividendRate', None)  # Annual Dividend Payment Per Share

# Market Data Info
previous_close = stock_info.get('previousClose', None)  # Previous Day's Closing Price
regular_market_price = stock_info.get('regularMarketPrice', None)  # Current Market Price
regular_market_day_low = stock_info.get('regularMarketDayLow', None)  # Low Price for the Current Day
regular_market_day_high = stock_info.get('regularMarketDayHigh', None)  # High Price for the Current Day
regular_market_volume = stock_info.get('regularMarketVolume', None)  # Trading Volume for the Current Day
regular_market_open = stock_info.get('regularMarketOpen', None)  # Opening Price for the Current Day
day_range = stock_info.get('dayRange', None)  # Range between Low and High Prices for the Day
fifty_two_week_range = stock_info.get('fiftyTwoWeekRange', None)  # 52-week Price Range
industry = stock_info.get('industry', None)  # Industry of the stock
sector = stock_info.get('sector', None)  # Sector of the stock
market_cap = stock_info.get('marketCap', None)  # Market Capitalization
beta = stock_info.get('beta', None)  # Stock Beta (volatility)
earnings_date = stock_info.get('earningsDate', None)  # Next earnings date

# ESG Score
esg_score = stock_info.get('esgScore', None)  # ESG score for the stock (if available)

# Analyst Summary
analyst_recommendation_summary = stock_info.get('recommendationKey', 'No recommendation available')  # Get the recommendation summary

# Get the cash flow statement
cash_flow = yf.Ticker(selected_stock).cashflow

# Extract the operating cash flow for the most recent year and the previous year
try:
    # Try to fetch the operating cash flow for the most recent year (current year) and previous year
    operating_cash_flow_current_year = cash_flow.loc['Operating Cash Flow'][0]  # Latest year (most recent)
    operating_cash_flow_previous_year = cash_flow.loc['Operating Cash Flow'][1]  # Previous year
except (KeyError, IndexError):
    # If data is not available, set both values to 0
    operating_cash_flow_current_year = 0
    operating_cash_flow_previous_year = 0

# Calculate YoY Operating Cash Flow Growth
if operating_cash_flow_previous_year == 0:
    yoy_ocfg_growth = 0  # Or 0, or some other default value
else:
    yoy_ocfg_growth = ((operating_cash_flow_current_year - operating_cash_flow_previous_year) / operating_cash_flow_previous_year)

# Create a DataFrame to store the ratios for the selected ticker
stock_ratios = pd.DataFrame({
    'Company Name': [company_name],
    'PE Ratio': [pe_ratio],
    'PEG Ratio': [peg_ratio],
    'Price-to-Book': [price_to_book],
    'Debt-to-Equity': [debt_to_equity],
    'Dividend Yield': [dividend_yield],  # Convert to percentage if available
    'Price-to-Sales': [price_to_sales],
    'ROE': [roe],
    'Quick Ratio': [quick_ratio],
    'Current Ratio': [current_ratio],
    'Industry': [industry],
    'Sector': [sector],
    'Market Cap': [market_cap],
    'Beta': [beta],
    'Earnings Date': [earnings_date],
    'Shares Outstanding': [shares_outstanding],
    'Gross Profit': [gross_profit],
    'Net Income': [net_income],
    'Net Profit Margin': [net_profit_margin],
    'Enterprise Value': [enterprise_value],
    'Dividend Rate': [dividend_rate],
    'Previous Close': [previous_close],
    'Regular Market Price': [regular_market_price],
    'Regular Market Day Low': [regular_market_day_low],
    'Regular Market Day High': [regular_market_day_high],
    'Regular Market Volume': [regular_market_volume],
    'Regular Market Open': [regular_market_open],
    'Day Range': [day_range],
    '52-Week Range': [fifty_two_week_range],
    'YOY Revenue Growth': [yoy_revenue_growth],
    'ESG Score': [esg_score],
    'YOY Operating Cash Flow Growth': [yoy_ocfg_growth]
})

selected_industry = industry

print("stock_ratios df")
print(stock_ratios)

# Load the stock data based on the ticker selected in front end
stock_data = load_data(selected_stock)  # loads the selected ticker data (selected_stock)

print('todays stock info (stock_data df)')
print(stock_data)

# Provide load context and load ticker data
data_load_state = st.text("loading data...")  # displays the following text when loading the data
data_load_state.empty()  # changes the load text to done when loaded

# Add function to get latest close price for close price field, assign var:
# this will allow us to use this var for calculations later in the app if we don't have a current mkt price
# Define A function to get the latest close price and its date:
# Reset index to remove the MultiIndex and keep 'Date' as a column


def last_close_price_field(stock_data):
    # Remove rows where Close is NaN or None
    data_valid = stock_data.dropna(subset=['Close'])
    # Sort data by Date in descending order
    data_sorted = data_valid.sort_values('Date', ascending=False)
    # Get the latest valid close price and date
    if not data_sorted.empty:
        latest_close_date = data_sorted.iloc[0]['Date']
        latest_close_price = data_sorted.iloc[0]['Close']
        return latest_close_price, latest_close_date
    else:
        return None, None

# Create Variable for Last Close Price / Last Close Date:
last_close_price, last_close_date = last_close_price_field(stock_data)

# Remove the seconds time stamp from the date (the 00:00:00 formatted stamp)
last_close_date = last_close_date.strftime('%Y-%m-%d')

# Create a var for current price that retrieves the last close price if the market price is unavailable:
Current_Price = last_close_price

# Round Current Price to 2 Decimal Points:
Current_Price = round(Current_Price, 2)
print(Current_Price)


# --------------- sharpe ratio
# Add Sharpe Ratio field using historical data for risk assessment
stock_data['Returns'] = stock_data['Close'].pct_change()  # pandas pct change method

# Get Risk Free Rate using Ticker symbol for the 10-year U.S. Treasury Yield (^TNX)
risk_free_rate_ticker = '^TNX'

# Fetch data for the 10-year U.S. Treasury Yield
rfr_data = yf.Ticker(risk_free_rate_ticker)

# Get the latest closing price (which represents the yield)
rfr_latest_yield = rfr_data.history(period="1d")['Close'].iloc[-1]

# Convert the percentage to a decimal by dividing by 100, and round to 4 decimal places
rfr_latest_yield = round(rfr_latest_yield / 100, 4)

# Risk Free Rate (1 Yr current)
risk_free_rate = rfr_latest_yield / 252  # approximate to two decimal points

# Calculate excess returns (stock returns - risk-free rate)
stock_data['Excess_Returns'] = stock_data['Returns'] - risk_free_rate

# Calculate the annualized Sharpe ratio (assuming 252 trading days in a year)
sharpe_ratio = np.sqrt(252) * stock_data['Excess_Returns'].mean() / stock_data['Excess_Returns'].std()

# Round to two decimal points
sharpe_ratio = round(sharpe_ratio, 2)
# --------------- sharpe ratio





# --------------- Retrieve Yahoo Finance Analyst Ratings / Grades
# Error handling: # Fetch Analyst Info and handle errors
try:
    stock_analyst_info = yf.Ticker(selected_stock).analyst_price_targets  # Try to fetch stock information

    # Check if the response is valid (not empty)
    if not stock_analyst_info or "error" in stock_analyst_info:  # comes through as a list
        raise ValueError("Received an empty or invalid response.")

# Handle errors related to the request, invalid JSON, or empty response
except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:

    # Use streamlit error handling message
    st.error(f"Oops... A large portion of the data is missing for this ticker. Probably not worth analysing :)")

    # Get the animation components with a custom size (e.g., 1.2x normal size)
    css, html = create_warning_animation(size_factor=2)

    # Add CSS for centering the animation
    centered_css = css + """
    <style>
    .warning-animation-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    </style>
    """

    # Wrap the HTML in a div with the centering class
    centered_html = f"""
    <div class="warning-animation-container">
        {html}
    </div>
    """

    # Display the error animation
    st.markdown(centered_css, unsafe_allow_html=True)
    st.markdown(centered_html, unsafe_allow_html=True)

    # Stop execution of any further script if the error occurs
    st.stop()

# Convert the dictionary into a DataFrame
stock_analyst_info_df = pd.DataFrame(list(stock_analyst_info.items()), columns=["Metric", "Value"])
# --------------- Retrieve Yahoo Finance Analyst Ratings / Grades





# --------------- Retrieve Rating Agency Ratings / Grades
# Error handling: Fetch Analyst Info and handle errors
try:
    # Fetch analyst recommendations
    agency_analyst_info = yf.Ticker(selected_stock).get_recommendations()

    # Check if DataFrame is valid and not empty
    if agency_analyst_info.empty:  # comes through as a df when the method is called
        raise ValueError("Received an empty or invalid response.")

# Handle errors related to the request, invalid JSON, or empty response
except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:

    # Use streamlit error handling message
    st.error(f"Oops... A large portion of the data is missing for this ticker. Probably not worth analysing :)")

    # Get the animation components with a custom size (e.g., 1.2x normal size)
    css, html = create_warning_animation(size_factor=2)

    # Add CSS for centering the animation
    centered_css = css + """
    <style>
    .warning-animation-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    </style>
    """

    # Wrap the HTML in a div with the centering class
    centered_html = f"""
    <div class="warning-animation-container">
        {html}
    </div>
    """

    # Display the error animation
    st.markdown(centered_css, unsafe_allow_html=True)
    st.markdown(centered_html, unsafe_allow_html=True)

    # Stop execution of any further script if the error occurs
    st.stop()
# --------------- Retrieve Rating Agency Ratings / Grades


# ------------------------------------------ Data: Load Ticker Volume/Pricing/Ratios (Historic & Current) ---------------------------------------










# ------------------------------------------ Data: Load Copy of Trade History for the SPY to use for benchmarking ---------------------------------------

# For Benchmarking Purposes, create a separate DF for SPY
SPY_data = load_data("SPY")  # loads the selected ticker data (selected_stock)

# ------------------------------------------ Data: Load Copy of Trade History for the SPY to use for benchmarking ---------------------------------------








# -------------------------------- Data: Add a yearly data dataframe to capture price by year on today's date over last 10 years ---------------------------------------

# Get data from each available day of the current month through all years
print("\n Current Month Yearly Historical Data:")
ct_month_yrly_data = stock_data[stock_data['Date'].dt.month == Current_Month]  # filters stock_data df to this month throughout all years
print(ct_month_yrly_data)  # check the df is pulling the correct data

# Get data from each available day of the current day through the all years
print("\n Current Day Yearly Historical Data:")
yearly_data = ct_month_yrly_data[ct_month_yrly_data['Date'].dt.day == Current_Day]  # filters stock_data df to this day throughout all years
print(yearly_data)  # check the df is pulling the correct data

# Add column with Trend indicators to the yearly_data table
# Make a copy of the DataFrame and reset its index so we can do a loop 1-10 for the indicators
yearly_data = yearly_data.copy().reset_index(drop=True)
print(yearly_data)

# Add a "indicator" column and add with empty strings
yearly_data['Trend'] = ''

# Iterate through each row starting from the second row
for i in range(1, len(yearly_data)):
    # Get the closing price of the current row and the previous row
    current_close = yearly_data.at[i, 'Close']   # close price at current row
    previous_close = yearly_data.at[i - 1, 'Close']   # close price at row before the previous

    # Compare current close with previous close
    if current_close > previous_close:
        yearly_data.at[i, 'Trend'] = '↑'  # Up arrow
    elif current_close < previous_close:
        yearly_data.at[i, 'Trend'] = '↓'  # Down arrow
    else:
        yearly_data.at[i, 'Trend'] = '■'  # Square

# Set the indicator for the first row as yellow square since there is no previous row
yearly_data.at[0, 'Trend'] = '■'

# check index reset and indicators added properly:
print("\n Yearly History Table W/ Indicators:")
print(yearly_data)


# -------------------------------- Data: Add a yearly data dataframe to capture price by year on today's date over last 10 years ---------------------------------------






# ------------------------------------------------- Data: Add a data dataframe for moving averages --------------------------------------------------------

# Add a df for moving average data
moving_average_data = stock_data.copy()

# Calculate the 50-day and 200-day SMAs for the new DataFrame
moving_average_data['50_day_SMA'] = moving_average_data['Close'].rolling(window=50, min_periods=1).mean()
moving_average_data['200_day_SMA'] = moving_average_data['Close'].rolling(window=200, min_periods=1).mean()

# Debugging - verify columns print
print(moving_average_data.columns)
print(moving_average_data[['Date', 'Close', '50_day_SMA', '200_day_SMA']].tail())  # Check last couple rows

# Create function to apply exponential smoothing line to future graphs

def apply_exponential_smoothing(data, smoothing_level=0.001):  # .2 = default alpha
    """
    Apply Simple Exponential Smoothing to the stock closing prices.

    :param data: Stock data
    :param alpha: Smoothing factor (0 < alpha <= 1) -> closer to 0 = more weight on older data. Will have a greater smoothing effect
    - -> closer to 1 = will have less of a smoothing effect. Will follow the actual close price of the stock more closely, resulting in less of a smoothing effect
    :return: Smoothed values

    heuristic -> straightforward, rule based approach. Takes the first close price and then uses the alpha as the smoothed price moving forward
    """
    # Initialize the model using the hueristic method
    es_model = SimpleExpSmoothing(data['Close'], initialization_method="heuristic")

    # Fit the model with a specified smoothing level
    es_model_fit = es_model.fit(smoothing_level=smoothing_level, optimized=False)

    return es_model_fit.fittedvalues  # Returns the smoothed data based on set alpha

# ------------------------------------------------- Data: Add a data dataframe for moving averages --------------------------------------------------------















# -------------------------------- Data: Calculate and Create Variables for Historical / Risk Models ---------------------------------------

# ---------- RSI
rsi_data = stock_data.copy()
def calculate_rsi(rsi_data, window=14):

    # Calculate the difference in price from previous day
    delta = rsi_data['Close'].diff()

    # Get the positive gains (where delta is positive) and negative losses (where delta is negative)
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()  # Rolling mean of gains
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()  # Rolling mean of losses

    # Calculate the Relative Strength (RS)
    rs = gain / loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Create Variable for RSI Values calling the RSI calculation function
rsi_values = calculate_rsi(rsi_data)

# Get the most recent RSI value (latest day)
latest_rsi_value = rsi_values.iloc[-1]  # This is the RSI for the most recent day
# ---------- RSI


# ---------- MACD
# Calculate MACD and Signal line to provide insight on short term trend momentum and strength
def calculate_macd(df, fast=12, slow=26, signal=9):

    # Calculate the Fast and Slow Exponential Moving Averages
    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()

    # MACD Line = Fast EMA - Slow EMA
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']

    # Signal Line = 9-period EMA of MACD
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()

    # Histogram = MACD - Signal
    df['Histogram'] = df['MACD'] - df['Signal']

    return df

# Apply the MACD calculation to the stock data
macd_data = calculate_macd(stock_data)

# Filter MACD df to include only the last year of data
macd_data = macd_data[moving_average_data['Date'] >= one_year_ago]
# ---------- MACD

# ---------- SMA
# Calculate Simple Moving Average Variables
price = moving_average_data['Close'].iloc[-1]  # Get the latest closing price
sma_50 = moving_average_data['50_day_SMA'].iloc[-1]  # Latest 50-day SMA
sma_200 = moving_average_data['200_day_SMA'].iloc[-1]  # Latest 200-day SMA

# Calculate the difference between 50-day and 200-day SMA
sma_price_difference = sma_50 - sma_200

# Calculate the percentage difference between the 50-day and 200-day SMA
sma_percentage_difference = (sma_price_difference / sma_200) * 100
# ---------- SMA

# ---------- Monte Carlo Simulation
# create df for monte carlo simulation data set
mc_sim_data = stock_data.copy()

# Set parameters for simulation
mc_simulations_num = 1000  # Number of simulated paths
mc_days_num = 252  # Number of days to simulate (e.g., one year of trading days)

# Calculate daily returns (log returns)
mc_sim_data['Log Return'] = np.log(mc_sim_data['Close'] / mc_sim_data['Close'].shift(1))

# Drop any missing values
mc_sim_data = mc_sim_data.dropna()

# Display the first few log returns
mc_sim_data[['Close', 'Log Return']].head()

# Calculate the mean and volatility from the historical data
mc_mean_return = mc_sim_data['Log Return'].mean()
mc_volatility = mc_sim_data['Log Return'].std()

# Simulate future price paths
mc_simulations = np.zeros((mc_simulations_num, mc_days_num))  # Shape should be (10000, 252)
last_price = mc_sim_data['Close'].iloc[-1]  # Use the last closing price as the starting point

# Ensure the simulation loop works without dimension issues
for i in range(mc_simulations_num):

    # Generate random returns for each day in the simulation
    mc_random_returns = np.random.normal(mc_mean_return, mc_volatility, mc_days_num)

    # Generate the simulated price path using compound returns
    mc_price_path = last_price * np.exp(np.cumsum(mc_random_returns))  # price_path shape will be (252,)

    # Assign the simulated price path to the simulations array
    mc_simulations[i, :] = mc_price_path  # Assign to the i-th row of the simulations array

# Convert the simulations to a DataFrame for easier visualization
mc_simulated_prices = pd.DataFrame(mc_simulations.T, columns=[f'Simulation {i+1}' for i in range(mc_simulations_num)])

# Check a sample of simulated paths
print(mc_simulated_prices.head())

# Debug checks
print(mc_price_path.shape)  # should print (252,) each time
print(mc_simulations.shape)  # should print (10000, 252)

# Calculate the 5th, 50th, and 95th percentiles of the simulated paths
mc_percentile_5 = mc_simulated_prices.quantile(0.05, axis=1)
mc_percentile_50 = mc_simulated_prices.quantile(0.5, axis=1)
mc_percentile_95 = mc_simulated_prices.quantile(0.95, axis=1)
# ---------- Monte Carlo Simulation


# ---------- Value at Risk (VAR)
# Date Index:
# The date index of the stock_data df is not in proper format to use the pandas resample() method.
# Need to reset to datetime index

# create a df for VaR data
var_data = stock_data.copy()

# Ensure the 'Date' column is in datetime format (if it isn't already)
var_data['Date'] = pd.to_datetime(var_data['Date'], errors='coerce')

# Set the 'Date' column as the index of the DataFrame
var_data.set_index('Date', inplace=True)

# Define function to calculate Historical VaR at 95% confidence level
def calculate_historical_VaR(var_data, time_window='daily'):
    """
    Calculates the historical Value at Risk (VaR) at 95% confidence for a given time window (daily, monthly, yearly).

    Parameters:
    - stock_data: pandas DataFrame with historical stock data.
    - time_window: str, the time window for VaR calculation ('daily', 'monthly', or 'yearly').

    Returns:
    - VaR value as a percentage and in dollars.

    - resample() -> pandas method that allows for quickly changing the frequency of data using a letter
                'D': Day
                'W': Week
                'M': Month
                'A': Year
                'H': Hour
                'Q': Quarter
                'B': Business day (excludes weekends)

                ex:
                df.resample(rule, how='mean', fill_method='ffill')

                The errors='coerce' argument ensures that invalid or incorrect date formats are turned
                into NaT (Not a Time) instead of raising errors.
    """

    # Calculate daily returns using adjusted closing price based on time window
    if time_window == 'daily':
        var_data['Return'] = var_data['Close'].pct_change()
    elif time_window == 'monthly':
        var_data['Return'] = var_data['Close'].resample('ME').ffill().pct_change()
    elif time_window == 'yearly':
        var_data['Return'] = var_data['Close'].resample('YE').ffill().pct_change()
    else:
        raise ValueError("Invalid time window. Choose from 'daily', 'monthly', or 'yearly'.")

    # Sort in ascending order
    sorted_returns = var_data['Return'].sort_values()

    # Calculate the 5th percentile (for 95% confidence level)
    hist_VaR_95 = sorted_returns.quantile(0.05)

    # Calculate the VaR in dollars
    current_adj_price = var_data['Close'].iloc[-1]
    hist_VaR_95_dollars = hist_VaR_95 * current_adj_price

    # Return both VaR in percentage and dollars
    return hist_VaR_95, hist_VaR_95_dollars

# Function to calculate Daily VaR
def calculate_daily_VaR(var_data):
    return calculate_historical_VaR(var_data, time_window='daily')

# Function to calculate Monthly VaR
def calculate_monthly_VaR(var_data):
    return calculate_historical_VaR(var_data, time_window='monthly')

# Function to calculate Yearly VaR
def calculate_yearly_VaR(var_data):
    return calculate_historical_VaR(var_data, time_window='yearly')

# Calculate hist VaR for daily, monthly, and yearly
hist_daily_VaR_95, hist_VaR_95_dollars = calculate_daily_VaR(var_data)
hist_monthly_VaR_95, hist_monthly_VaR_95_dollars = calculate_monthly_VaR(var_data)
hist_yearly_VaR_95, hist_yearly_VaR_95_dollars = calculate_yearly_VaR(var_data)
# ---------- Value at Risk (VAR)


# # ---------------- Get Industry Averages
# URL of the webpages containing ratios by industry
pe_url = "https://fullratio.com/pe-ratio-by-industry"
roe_url = "https://fullratio.com/roe-by-industry"

# Add user-agent header to mimic a browser (websites often block scraping)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Initialize an empty list to store the extracted data
pe_ind_avg_data = []
roe_data = []

# Function to find the table on the page
def find_table(soup):
    # Try to find any table first
    table = soup.find('table')

    if table is None:
        # If no table is found, try other ways to locate it
        tables = soup.find_all('table')
        if tables:
            table = tables[0]  # Use the first table found
        else:
            # Look for divs that might contain table data
            table_container = soup.find('div', class_='table-responsive')
            if table_container:
                table = table_container.find('table')

    return table

# First scrape P/E ratios
try:
    # Send a GET request to the P/E URL with headers
    response = requests.get(pe_url, headers=headers)
    response.raise_for_status()  # Check if the request was successful

    # Parse the content of the page with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table containing the P/E ratio data
    table = find_table(soup)

    if table is None:
        st.error("Could not find the P/E ratio table on the webpage. The website structure might have changed.")
    else:
        # Extract the table rows from the table
        rows = table.find_all('tr')

        if len(rows) <= 1:
            st.warning("Found a P/E table but it contains insufficient data.")
        else:
            # Get the headers to determine column indices
            headers_row = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]

            # Find the indices for industry and PE ratio columns
            industry_idx = next((i for i, h in enumerate(headers_row) if 'industry' in h.lower() or 'sector' in h.lower()), 0)
            pe_idx = next((i for i, h in enumerate(headers_row) if 'p/e' in h.lower() or 'pe' in h.lower()), 1)

            # Loop through each row, extract the industry and P/E ratio, and store it
            for row in rows[1:]:  # Skip the first row, which contains column headers
                cols = row.find_all('td')

                if len(cols) > max(industry_idx, pe_idx):
                    industry = cols[industry_idx].get_text(strip=True)
                    pe_ratio_text = cols[pe_idx].get_text(strip=True)

                    # Clean the PE ratio - remove any non-numeric characters except decimal point
                    pe_ratio_text = ''.join(c for c in pe_ratio_text if c.isdigit() or c == '.')

                    # Add to data list (convert P/E to float, handle missing values)
                    try:
                        pe_ratio = float(pe_ratio_text) if pe_ratio_text else None
                    except ValueError:
                        pe_ratio = None

                    pe_ind_avg_data.append({
                        'Industry': industry,
                        'Average P/E Ratio': pe_ratio
                    })

            # Create a pandas DataFrame to organize the P/E data
            industry_avg_df = pd.DataFrame(pe_ind_avg_data)

except requests.exceptions.RequestException as e:
    st.error(f"Error fetching P/E data: {e}")
    industry_avg_df = pd.DataFrame(columns=['Industry', 'Average P/E Ratio'])
except Exception as e:
    st.error(f"Error processing P/E data: {e}")
    industry_avg_df = pd.DataFrame(columns=['Industry', 'Average P/E Ratio'])

# Now scrape ROE data
try:
    # Send a GET request to the ROE URL with headers
    response = requests.get(roe_url, headers=headers)
    response.raise_for_status()  # Check if the request was successful

    # Parse the content of the page with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table containing the ROE data
    table = find_table(soup)

    if table is None:
        st.error("Could not find the ROE table on the webpage. The website structure might have changed.")
    else:
        # Extract the table rows from the table
        rows = table.find_all('tr')

        if len(rows) <= 1:
            st.warning("Found an ROE table but it contains insufficient data.")
        else:
            # Get the headers to determine column indices
            headers_row = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]

            # Find the indices for industry and ROE columns
            industry_idx = next((i for i, h in enumerate(headers_row) if 'industry' in h.lower() or 'sector' in h.lower()), 0)
            roe_idx = next((i for i, h in enumerate(headers_row) if 'roe' in h.lower() or 'return on equity' in h.lower()), 1)

            # Loop through each row, extract the industry and ROE, and store it
            for row in rows[1:]:  # Skip the first row, which contains column headers
                cols = row.find_all('td')

                if len(cols) > max(industry_idx, roe_idx):
                    industry = cols[industry_idx].get_text(strip=True)
                    roe_text = cols[roe_idx].get_text(strip=True)

                    # Clean the ROE value - remove any non-numeric characters except decimal point
                    # Some ROE values might be percentages, so also remove % signs
                    roe_text = roe_text.replace('%', '')
                    roe_text = ''.join(c for c in roe_text if c.isdigit() or c == '.' or c == '-')

                    # Add to data list (convert ROE to float, handle missing values)
                    try:
                        roe_value = float(roe_text) if roe_text else None
                    except ValueError:
                        roe_value = None

                    roe_data.append({
                        'Industry': industry,
                        'Average ROE': roe_value
                    })

            # Create a pandas DataFrame to organize the ROE data
            roe_df = pd.DataFrame(roe_data)

            # Merge the P/E and ROE DataFrames on the Industry column
            if not industry_avg_df.empty and not roe_df.empty:
                industry_avg_df = pd.merge(industry_avg_df, roe_df, on='Industry', how='outer')
            elif not roe_df.empty:
                industry_avg_df = roe_df

except requests.exceptions.RequestException as e:
    st.error(f"Error fetching ROE data: {e}")
except Exception as e:
    st.error(f"Error processing ROE data: {e}")
# ---------------- Get Industry Averages

# -------------------------------- Data: Calculate and Create Variables for Historical / Risk Models ---------------------------------------









# ------------------------------------------------------ Data: Forecast Model (Forecasted Data) --------------------------------------------------------------------

# Note: (For Our Time Series Analysis, We will use the Prophet Time series Model from META)

# Create a DataFrame for training our time-series model:
df_train = stock_data[['Date', 'Close']] # create a df (df_train) to train our data set (using date and close price)
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"}) # create a dictionary to rename the columns (must rename columns for META Prophet to read data [in documentation]. Documentation Link: https://facebook.github.io/prophet/docs/quick_start.html#python-api)

# Function to train the model and generate forecasts
@st.cache_data()
def train_model(df_train):
    # Fit Prophet model
    m = Prophet()
    m.fit(df_train)

    # Return the trained model
    return m

# Assign Variable to trained model data (the cached function)
trained_model = train_model(df_train)
print(trained_model)

# Create an interactive year range slider and set our forecast range period:
forecasted_year_range = st.sidebar.slider("Choose a Forecast Range (Years):", 1, 10) # creates a slider for forecast years. 1 and 10 are the year range
period = forecasted_year_range * 365 # stores full period of days in our year range. Will use this later in the app workflow
future = trained_model.make_future_dataframe(periods=period) # Create a df that goes into the futre; make prediction on a dataframe with a column ds containing the dates for which a prediction is to be made
forecast = trained_model.predict(future) # forecast future stock prices for our model. Will return predictions in forecast data

# Reorder columns with yhat as the second column
forecast = forecast[['ds', 'yhat'] + [col for col in forecast.columns if col not in ['ds', 'yhat']]]

# Apply the lower bound to the forecasted prices, ensuring they don't go below 0
forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0.01, x))  # Set all negative values to 1 cent

print(forecast)
print(f"forecast year range {forecasted_year_range}")
# ------------------------------------------------------ Data: Forecast Model (Forecasted Data) --------------------------------------------------------------------








# ------------------------------------------------------ Data: Custom Stock Grade Model --------------------------------------------------------------------

# Get Missing Metrics For Model



# Define a function to calculate grades for stocks based on defined metrics
def calculate_grades(stock_symbol):

    # Initialize scores
    model_pe_score = 0
    model_roe_score = 0
    model_volume_score = 0
    model_dividend_score = 0
    model_current_ratio_score = 0
    model_debt_to_equity_score = 0
    model_net_profit_margin_score = 0
    model_sharpe_score = 0
    model_prophet_sarima_score = 0
    model_stock_price_vs_sp500_score = 0
    model_analyst_rating_score = 0
    model_operating_cash_flow_growth_score = 0
    model_peg_ratio_score = 0
    model_var_score = 0
    model_monte_carlo_score = 0
    model_rsi_sma_score = 0

    # ------- Forecast Percent Diff Variable
    # Set Up Any Remaining Variables
    model_chosen_prophet_price = forecast['yhat'].iloc[-1]  # forecast['yhat'].iloc[-1] retrieves the forecasted value

    # Get Forecasted X Year $ Difference & Percentage:
    model_prophet_forecast_difference = model_chosen_prophet_price - Current_Price
    model_prophet_forecast_difference = round(model_prophet_forecast_difference, 2)
    model_prophet_forecast_percent_difference = (model_prophet_forecast_difference/Current_Price)
    # ------- Forecast Percent Diff Variable

    # ------- SPY vs Selected Ticker Compare
    spy_compare_df = load_data('SPY', start_date_three_y_ago, today)
    selected_ticker_compare = load_data(selected_stock, start_date_three_y_ago, today)

    # Function to compare performance
    def compare_performance(stock_data=selected_ticker_compare, SPY_data=spy_compare_df):
        # Merge the stock and SPY data on Date
        comparison_df = pd.merge(stock_data[['Date', 'Percentage Change']],
                                 SPY_data[['Date', 'Percentage Change']],
                                 on='Date',
                                 suffixes=('_stock', '_SPY'))

        # Compare if the stock outperforms SPY on each day
        comparison_df['Outperforms SPY'] = comparison_df['Percentage Change_stock'] > comparison_df['Percentage Change_SPY']

        # Calculate the percentage of days the stock outperforms SPY
        spy_outperform_percentage = comparison_df['Outperforms SPY'].mean()

        return spy_outperform_percentage

    # Call the Outperform % to retrieve
    spy_outperform_percentage = compare_performance()
    # ------- SPY vs Selected Ticker Compare

    # ------- Monte Carlo Compare
    # Get percentage of simulations over latest day close price for all simulations

    # Extract the final price from each simulation (the last price in each row)
    final_prices = mc_simulations[:, -1]  # The last column of each simulation path

    # Compare the final price to the current day's closing price (last_price)
    simulations_above_current_price = final_prices > last_price

    # Calculate the percentage of simulations that end above the current price
    percentage_above_current = (simulations_above_current_price.sum() / mc_simulations_num) * 100

    # Get percentage of simulations that end 9% are higher from the latest day close price
    price_threshold = last_price * 1.09

    # Compare the final price to the current day's closing price (last_price)
    simulations_above_9_percent = final_prices > price_threshold

    # Calculate the percentage of simulations that meet the condition
    percentage_above_9_percent = (simulations_above_9_percent.sum() / mc_simulations_num) * 100
    # ------- Monte Carlo Compare

    # ------- Merged Avg DFs
    # Merge the two DataFrames on the 'Industry' column
    industry_avg_merged_df = pd.merge(stock_ratios, industry_avg_df, on='Industry', how='left')

    # Reduce fields to company, industry and it's averages
    merged_df = industry_avg_merged_df[['Company Name', 'Industry', 'Average P/E Ratio', 'Average ROE']]

    # Check if the row is empty
    if not merged_df.empty:
        # Get the scalar values from the row
        selected_stock_industry_avg_pe = merged_df['Average P/E Ratio'].iloc[0]
        selected_stock_industry_avg_roe = merged_df['Average ROE'].iloc[0]
    else:
        # If the row is empty, assign default values
        selected_stock_industry_avg_pe = 25
        selected_stock_industry_avg_roe = 10  # comes through as if a percent
    # ------- Merged Avg DFs

    # PE Ratio / YOY Growth (13%)
    if pe_ratio is not None and yoy_revenue_growth is not None:
        if pe_ratio < selected_stock_industry_avg_pe and yoy_revenue_growth > 0.19:
            model_pe_score = 0.13  # +13% for good growth
        elif pe_ratio < selected_stock_industry_avg_pe and 0.12 <= yoy_revenue_growth <= 0.18:
            model_pe_score = 0.08  # +8% for decent growth
        elif pe_ratio < selected_stock_industry_avg_pe:
            model_pe_score = 0.05  # +5% for below-average PE
        else:
            model_pe_score = 0  # 0% for high PE ratio compared to industry

    # PEG Ratio Score (5%) -- validated
    if peg_ratio is None or peg_ratio == '':
        model_peg_ratio_score = 0.05  # Default score if missing
    else:
        try:
            peg_ratio_float = float(peg_ratio)
            if peg_ratio_float < 1:
                model_peg_ratio_score = 0.05  # +5% for low PEG ratio
            elif peg_ratio_float > 1:
                model_peg_ratio_score = -0.02  # -2% for high PEG ratio
            else:
                model_peg_ratio_score = 0  # 0% for neutral PEG ratio
        except ValueError:
            # Handle case where peg_ratio can't be converted to a float
            model_peg_ratio_score = 0.05  # Default score in case of invalid value

    # ROE Score (5%)
    if roe is not None:
        if roe > selected_stock_industry_avg_roe:
            model_roe_score = 0.05  # +5% if ROE is above sector average
        else:
            model_roe_score = 0  # 0% for low ROE compared to sector average

    # Volume Score (3%) -- validated
    if regular_market_volume is not None:
        if regular_market_volume >= 500000:
            model_volume_score = 0.03  # +3% for high volume
        elif regular_market_volume < 500000:
            model_volume_score = -0.02  # -2% for low volume

    # Dividend Yield Score (3%) -- validated
    if dividend_yield is not None:
        if dividend_yield > 2:
            model_dividend_score = 0.03  # +3% for above sector average
        elif dividend_yield < 2:
            model_dividend_score = 0

    # Current Ratio Score (4%) -- validated
    if current_ratio is not None:
        if current_ratio > 1.2:
            model_current_ratio_score = 0.04  # +4% for good liquidity
        elif 1.0 <= current_ratio <= 1.99:
            model_current_ratio_score = 0.025  # +2.5% for acceptable liquidity
        elif 0.9 <= current_ratio < 1.0:
            model_current_ratio_score = 0.015  # +1.5% for below-average liquidity
        elif current_ratio < 0.5:
            model_current_ratio_score = -0.02  # -2% for poor liquidity

    # Debt-to-Equity Ratio Score (4%) -- validated
    if debt_to_equity is not None:
        if .9 <= debt_to_equity <= 1.4:
            model_debt_to_equity_score = 0.04  # +4%
        elif 1.41 <= debt_to_equity <= 1.89:
            model_debt_to_equity_score = 0  # 0%
        elif debt_to_equity > 1.9:
            model_debt_to_equity_score = -0.02  # -2%
        elif 0.33 <= debt_to_equity <= 0.99:
            model_debt_to_equity_score = 0.02  # +2%
        else:
            model_debt_to_equity_score = -.01  # -1%

    # Operating Cash Flow Growth (4%) -- validated
    if yoy_ocfg_growth > 0.10:
        model_operating_cash_flow_growth_score = 0.04  # +4% for high cash flow growth
    else:
        model_operating_cash_flow_growth_score = 0  # 0% for low or no growth

    # Net Profit Margin Score (3%) -- validated
    if net_profit_margin is not None:
        if net_profit_margin >= 0.40:
            model_net_profit_margin_score = 0.04  # +4% for high profitability
        elif net_profit_margin >= 0.20:
            model_net_profit_margin_score = 0.03  # +3% for moderate profitability
        elif net_profit_margin >= 0.15:
            model_net_profit_margin_score = 0.02  # +2% for reasonable profitability
        elif net_profit_margin >= 0.10:
            model_net_profit_margin_score = 0.01  # +1% for acceptable profitability
        elif net_profit_margin < 0:
            model_net_profit_margin_score = -0.02  # -2% for negative profitability

    # Sharpe Ratio Score (4%) -- validated
    if sharpe_ratio is not None:
        if sharpe_ratio >= 1.5:
            model_sharpe_score = 0.04  # +4% for strong Sharpe ratio
        elif sharpe_ratio >= 1:
            model_sharpe_score = 0.02  # +2% for decent Sharpe ratio
        elif sharpe_ratio < 0.5:
            model_sharpe_score = -0.04  # -4% for poor Sharpe ratio

    # Prophet SARIMA Model (2%)
    if model_prophet_forecast_percent_difference is not None:
        if model_prophet_forecast_percent_difference >= 0.09:
            model_prophet_sarima_score = 0.02  # +2% for good forecast change
        else:
            model_prophet_sarima_score = 0  # 0% for no significant forecast change

    # Stock Price vs S&P 500 Score over 3 years (5%) -- validated
    if spy_outperform_percentage >= 0.90:
        model_stock_price_vs_sp500_score = 0.05  # +5% for outperforming S&P500
    else:
        model_stock_price_vs_sp500_score = 0  # 0% if not outperforming

    # Analyst Ratings Score (3%) -- validated
    if analyst_recommendation_summary is not None:
        if analyst_recommendation_summary == "strong_buy":
            model_analyst_rating_score = 0.025  # +2.5% for positive analyst ratings
        elif analyst_recommendation_summary == "outperform":
            model_analyst_rating_score = 0.03  # +3% for positive analyst ratings
        elif analyst_recommendation_summary == "buy":
            model_analyst_rating_score = 0.02  # +1.5% for negative analyst ratings
        elif analyst_recommendation_summary == "sell":
            model_analyst_rating_score = -0.02  # -1.5% for negative analyst ratings
        elif analyst_recommendation_summary == "strong_sell":
            model_analyst_rating_score = -0.02  # -1.5% for negative analyst ratings
        else:
            model_analyst_rating_score = 0  # 0% for neutral analyst ratings

    # VaR Score (3%) -- validated
    if hist_yearly_VaR_95 is not None:
        # If hist_yearly_VaR_95 is a percentage (e.g., -0.05 for 5% loss)
        if hist_yearly_VaR_95 > -0.10:  # Less than 10% loss
            model_var_score = 0.03  # +3% for low VaR
        elif -0.20 <= hist_yearly_VaR_95 <= -0.10:  # Between 10% and 20% loss
            model_var_score = 0.02  # +2% for moderate VaR
        elif hist_yearly_VaR_95 < -0.45:  # More than 45% loss
            model_var_score = -0.03  # -3% for high VaR
        else:
            model_var_score = 0  # 0% for neutral VaR

    # Monte Carlo Score (5%) -- compare % of simulations above current price and 9% inc from current price
    if percentage_above_current >= 0.9 and percentage_above_9_percent >= 0.5:
        model_monte_carlo_score = 0.05  # +5% for good simulation
    elif percentage_above_current >= 0.65 and percentage_above_9_percent >= 0.35:
        model_monte_carlo_score = 0.02  # +2% for decent simulation
    elif percentage_above_current <= 0.65 and percentage_above_9_percent <= 0.35:
        model_monte_carlo_score = -0.02  # -2% for poor simulation
    else:
        model_monte_carlo_score = 0  # 0% for neutral simulation

    # RSI/SMA Score (2%) -- validated
    if latest_rsi_value < 30 and sma_percentage_difference > 5:
        model_rsi_sma_score = 0.01  # +1% for good RSI and SMA
    elif latest_rsi_value > 70 and sma_percentage_difference < 5:
        model_rsi_sma_score = 0  # -1% for bad RSI and SMA
    else:
        model_rsi_sma_score = 0  # 0% for neutral RSI and SMA

    # Calculate model_score
    model_score = (
            model_pe_score + model_roe_score + model_volume_score + model_dividend_score +
            model_current_ratio_score + model_debt_to_equity_score + model_net_profit_margin_score +
            model_sharpe_score + model_prophet_sarima_score + model_stock_price_vs_sp500_score +
            model_analyst_rating_score + model_operating_cash_flow_growth_score +
            model_peg_ratio_score + model_var_score + model_monte_carlo_score + model_rsi_sma_score
    )

    # Calculate base points (remaining points to make total 1)
    base_points = .4  # Adjust base points to fill up to 1
    total_score = model_score + base_points  # Add the adjusted base points

    # Determine grade and color based on total score
    if total_score >= 0.97:
        grade = "S"
        grade_color_background = "rgba(255, 215, 0, 0.5)"  # Gold for S with 50% transparency
        grade_color_outline = "rgba(255, 215, 0, 0.7)"  # Gold for S with 70% transparency
    elif total_score >= 0.94:
        grade = "A"
        grade_color_background = "rgba(34, 139, 34, 0.5)"  # Forest green for A with 50% transparency
        grade_color_outline = "rgba(34, 139, 34, 0.7)"  # Forest green for A with 70% transparency
    elif total_score >= 0.90:
        grade = "A-"
        grade_color_background = "rgba(50, 205, 50, 0.5)"  # Lime green for A- with 50% transparency
        grade_color_outline = "rgba(50, 205, 50, 0.7)"  # Lime green for A- with 70% transparency
    elif total_score >= 0.87:
        grade = "B+"
        grade_color_background = "rgba(60, 179, 113, 0.5)"  # Medium sea green for B+ with 50% transparency
        grade_color_outline = "rgba(60, 179, 113, 0.7)"  # Medium sea green for B+ with 70% transparency
    elif total_score >= 0.84:
        grade = "B"
        grade_color_background = "rgba(102, 205, 170, 0.5)"  # Medium aquamarine for B with 50% transparency
        grade_color_outline = "rgba(102, 205, 170, 0.7)"  # Medium aquamarine for B with 70% transparency
    elif total_score >= 0.80:
        grade = "B-"
        grade_color_background = "rgba(152, 251, 152, 0.5)"  # Pale green for B- with 50% transparency
        grade_color_outline = "rgba(152, 251, 152, 0.7)"  # Pale green for B- with 70% transparency
    elif total_score >= 0.77:
        grade = "C+"
        grade_color_background = "rgba(173, 255, 47, 0.5)"  # Green yellow for C+ with 50% transparency
        grade_color_outline = "rgba(173, 255, 47, 0.7)"  # Green yellow for C+ with 70% transparency
    elif total_score >= 0.74:
        grade = "C"
        grade_color_background = "rgba(252, 226, 5, 0.5)"  # Bumblebee for C with 50% transparency
        grade_color_outline = "rgba(252, 226, 5, 0.7)"  # Bumblebee for C with 70% transparency
    elif total_score >= 0.70:
        grade = "C-"
        grade_color_background = "rgba(255, 165, 0, 0.5)"  # Orange for C- with 50% transparency
        grade_color_outline = "rgba(255, 165, 0, 0.7)"  # Orange for C- with 70% transparency
    elif total_score >= 0.67:
        grade = "D+"
        grade_color_background = "rgba(255, 140, 0, 0.5)"  # Dark orange for D+ with 50% transparency
        grade_color_outline = "rgba(255, 140, 0, 0.7)"  # Dark orange for D+ with 70% transparency
    elif total_score >= 0.64:
        grade = "D"
        grade_color_background = "rgba(255, 69, 0, 0.5)"  # Orange red for D with 50% transparency
        grade_color_outline = "rgba(255, 69, 0, 0.7)"  # Orange red for D with 70% transparency
    elif total_score >= 0.60:
        grade = "D-"
        grade_color_background = "rgba(255, 99, 71, 0.5)"  # Tomato for D- with 50% transparency
        grade_color_outline = "rgba(255, 99, 71, 0.7)"  # Tomato for D- with 70% transparency
    else:
        grade = "F"
        grade_color_background = "rgba(255, 0, 0, 0.5)"  # Red for F with 50% transparency
        grade_color_outline = "rgba(255, 0, 0, 0.7)"  # Red for F with 70% transparency

    return total_score, grade, grade_color_background, grade_color_outline

# Calculate grades for the stock
score, grade, grade_color_background, grade_color_outline = calculate_grades(selected_stock)

if score is not None:
    # Display the grade in a rounded box with the grade color
    st.markdown(f"""
        <div style="background-color:{grade_color_background}; 
                    color:white; 
                    font-size:20px; 
                    font-weight:bold; 
                    padding:10px 20px; 
                    border-radius:15px; 
                    border: 1px solid {grade_color_outline};
                    display:inline-block;">
            Model Grade: {grade}
        </div>
    """, unsafe_allow_html=True)
else:
    st.write(f"Error calculating grades for {selected_stock}")
# ------------------------------------------------------ Data: Custom Stock Grade Model --------------------------------------------------------------------


# /////////////////////////////////////////////  Pull in our data and set up datasets for use in app  ////////////////////////////////////////////////////////






















# //////////////////////////////////////////// Sidebar: Add Elements to Sidebar in App ////////////////////////////////////////////////////////////

# Note: Ticker selector was already added in section B as it is being used as a filter for our data
# Note: Company name added to sidebar already in section B

# ------------------------------------------- Sidebar: Add Dropdowns containing notes on metrics -----------------------------------------------

# Within the app, I want to have notes on each equation as well as what metric range I tend to look for within the sidebar
st.sidebar.header("Financial Ratio Notes")
with st.sidebar.expander("MARKET VALUE RATIOS"):
    st.write("---Measure the current market price relative to its value---")
    st.write('<span style="color: lightcoral;">PE Ratio: [Market Price per Share / EPS]</span>', unsafe_allow_html=True) # adding the additional HTML code allows us to change the text color in the write statement
    st.markdown("[AVG PE Ratio by Sector](https://fullratio.com/pe-ratio-by-industry)") # Insert a link on the sidebar to avg PE ratio by sector
    st.write("Ratio Notes: PE should be evaluated and compared to competitors within the sector. A PE over the avg industry PE might indicate that the stock is selling at a premium, but may also indicate higher expected growth/trade volume; A lower PE may indicate that the stock is selling at a discount, but may also indicate low growth/trade volume.")
    st.write('<span style="color: lightcoral;">PEG Ratio: [PE / EPS Growth Rate]</span>', unsafe_allow_html=True)
    st.write("Ratio Notes: PEG > 1 = Likely overvalued || PEG < 1 = Likely undervalued")
    st.write('<span style="color: lightcoral;">Price-to-Book Ratio: [Market Price per Share / Book Value Per Share]</span>', unsafe_allow_html=True)
    st.write("Ratio Notes: PB > 1 = Indicates stock might be overvalued copared to its assets || PB < 1 = Indicates stock might be undervalued copared to its assets || Typically not a good indicator for companies with intangible assets, such as tech companies.")
    st.write('<span style="color: lightcoral;">Price-to-Sales Ratio: [Market Cap / Revenue]</span>', unsafe_allow_html=True)
    st.write("Ratio Notes: 2-1 = Good || Below 1 = Better || Lower = Indicates the company is generating more revenue for every dollar investors have put into the company.")

with st.sidebar.expander("PROFITABILITY RATIOS"):
    st.write("---Measure the combined effects of liquidity, asset mgmt, and debt on operating results---")
    st.write('<span style="color: lightcoral;">ROE (Return on Equity): [Net Income / Common Equity]</span>', unsafe_allow_html=True)
    st.write("Ratio Notes: Measures total return on investment | Compare to the stock's sector | Bigger = Better")

with st.sidebar.expander("LIQUIDITY RATIOS"):
    st.write("---Measure the ability to meet current liabilities in the short term (Bigger = Better)---")
    st.write('<span style="color: lightcoral;">Current Ratio: [Current Assets / Current Liabilities]</span>', unsafe_allow_html=True)
    st.write("Ratio Notes: Close to or over 1 = Good || Over 1 means the company is covering its bills due within a one year period")
    st.write('<span style="color: lightcoral;">Quick Ratio: [(Current Assets - Inventory) / Current Liabilities]</span>', unsafe_allow_html=True)
    st.write("Ratio Notes: Close to or over 1 = Good || Over 1 means the company is able to cover its bills due within a one year period w/ liquid cash")

with st.sidebar.expander("ASSET MANAGEMENT RATIOS"):
    st.write("---Measure how effectively assets are being managed---")
    st.write('<span style="color: lightcoral;">Dividend Yield: [DPS / SP]</span>', unsafe_allow_html=True)
    st.write("Ratio Notes: For low growth stocks, should be higher and should look for consistent div growth over time- with signs of consistenly steady financials (able to pay debts consistently; want to see the company is managing its money well) || For growth stocks, typically lower, but if a stock shows high growth over time w/ a dividend yield that continues to remain the same or grow over time, this is a good sign (good to compare with their Current Ratio)")

with st.sidebar.expander("DEBT MANAGEMENT RATIOS"):
    st.write("---Measure how well debts are being managed---")
    st.write('<span style="color: lightcoral;">Debt-to-Equity: [Total Liabilities / Total Shareholder Equity]</span>', unsafe_allow_html=True)
    st.write("Ratio Notes: A good D/E ratio will vary by sector & company. Typically a 1.0-1.5 ratio is ideal. The main thing to look for is that if the company is leveraging debt is that it has enough liquidity and consistent return to pay off those debts. Leveraging debt (when managed well) can be a good indicator that a growth company is leveraging debt in order to re-invest and grow faster, which is typically a good sign that the company is strategically well managed.")

# ------------------------------------------- Sidebar: Add Dropdowns containing notes on metrics -----------------------------------------------




# //////////////////////////////////////////// Sidebar: Add Elements to Sidebar in App ////////////////////////////////////////////////////////////












# ///////////////////////////////////////////////////////////// Home Page //////////////////////////////////////////////////////////////////////////

# (1.1): Show stock current stock details
NA = "no data available for this stock"  # assign variable so can use in elseif statements when there is no data

# Add Company Name:
# Create a container for the company name:
cn_c = st.container()
# Check if data is available for the field:
with cn_c:
    if company_name.strip():  # if the field is not missing (use strip() for strings instead of .empty since it is not an object in the df in this situation)
        # Write the value to the app for today in KPI format if the data is available
        # Custom CSS for styling the kpis below with a translucent grey border
        st.markdown(
            f"""
                <div style="display: flex; justify-content: center; align-items: center;">
                    <h1 style="font-size: 32px; margin: 0;">{company_name}</h1>
                </div>
                """,
            unsafe_allow_html=True
        )
    else:
        # Write the data is not available for the field if missing:
        st.warning("Company Name: Not Available")

# Add a Line Seperator under the company name
st.markdown("""---""")

# ------------------------------------------------------ Home Page: KPISs -------------------------------------------------------------------

# Create a container for our KPIs:
kpi_c = st.container()

# Define columns for the KPIs
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = kpi_c.columns(4)

# Define the CSS for positive and negative trends
positive_icon = '<span style="color: green;">&#9650;</span>'
negative_icon = '<span style="color: red;">&#9660;</span>'
neutral_icon = '<span style="color: gray;">&#8212;</span>'

# --------------------------------------------------- Add KPI for Current Price --------------------------------------------------------------
with kpi_col1:
    kpi_col1 = st.container(border=True)
    with kpi_col1:
        # Give title for 1 year forecasted price:
        kpi_col1.write("Current Stock Price:")

        # Forecasted Price Difference in 1 year ($ Difference + Current Price & % Difference Concatenated)
        Current_Price_kpi = ("$" + str(round(Current_Price, 2)))

        # Write 1 Year Forecast Diff to sidebar
        if Current_Price_kpi is not None:
            # Write the value to the app for today in KPI format if the data is available
            kpi_col1.markdown(Current_Price_kpi, unsafe_allow_html=True)
        else:
            kpi_col1.warning(f"Current Price: Data Not Available")
# --------------------------------------------------- Add KPI for Current Price --------------------------------------------------------------

# ------------------------- Add KPI for YOY change in price from last year to this year from yesterday ---------------------------------------
with kpi_col2:
    kpi_col2 = st.container(border=True)
    with kpi_col2:
        # Get the index of the last row (current year)
        previous_year_index = yearly_data.index[-1]

        # Retrieve the 'Close' price of the previous year dynamically
        try:
            price_year_ago = yearly_data[yearly_data['Date'].dt.year == Last_Year]['Close'].iloc[0]
        except IndexError:
            # Handle case where if there is an index error (no data available for the previous year)
            # it takes the current price instead
            price_year_ago = yearly_data[yearly_data['Date'].dt.year == Current_Year]['Close'].iloc[0]

        # Get change # & percentage:
        print("YOY Price Change")
        print(f"Current Price: {Current_Price}")
        print(f"Price Year Ago: {price_year_ago}")
        YOY_difference = Current_Price - price_year_ago
        YOY_difference_number = round(YOY_difference, 2)    # round the number to two decimal points
        print(f"YOY Diff: {YOY_difference_number}")
        YOY_difference_percentage = (YOY_difference/price_year_ago)
        YOY_difference_percentage = round(YOY_difference_percentage * 100)
        print(f"YOY Diff: {YOY_difference_percentage}%")

        # Create a trend icon for if the YOY price is positive or negative:
        if YOY_difference_percentage > 0:
            trend_icon = positive_icon  # Use positive trend icon
        elif YOY_difference_percentage < 0:
            trend_icon = negative_icon  # Use negative trend icon
        else:
            trend_icon = neutral_icon  # Neutral trend icon if the difference is 0

        # Give title for YOY trend metric
        kpi_col2.write("YOY Price Change:")

        # YOY Price Difference in 1 year ($ & % Difference Concatenated)
        YOY_Price_Change = ("$" + str(YOY_difference_number) + " | " + str(YOY_difference_percentage) + "% " + trend_icon)

        # Write 1 Year Forecast Diff to Home Page
        if YOY_Price_Change is not None:
            # Write the value to the app for today in KPI format if the data is available
            kpi_col2.markdown(YOY_Price_Change, unsafe_allow_html=True)
        else:
            kpi_col2.warning(f"YOY Price Change: Data Not Available")
# ------------------------- Add KPI for YOY change in price from last year to this year from yesterday ---------------------------------------

# ------------------------------ Add KPI for Avg YOY price change over last 10 years from yesterday -----------------------------------------
with kpi_col3:
    kpi_col3 = st.container(border=True)
    with kpi_col3:
        # Extract year from the date
        yearly_data['Year'] = pd.to_datetime(yearly_data['Date']).dt.year

        # Group data by year and calculate average closing price for each year
        yearly_avg_close_price = yearly_data.groupby('Year')['Close'].mean()

        # Get the avg dollar amount change over the last 10 years available:
        # Print the number of total rows to get the number of years:
        total_rows = yearly_data.shape[0]
        # Calculate the dollar amount.
        avg_price_chg_dollar_amt = round(yearly_avg_close_price.mean() / (total_rows), 2)

        # Calculate year-over-year percentage change in average closing price
        yoy_avg_close_price_change_by_year = yearly_avg_close_price.pct_change() * 100
        yoy_avg_close_price_change = round(yoy_avg_close_price_change_by_year.mean(),2)

        # Print the result
        print("Year-over-Year Average Closing Price Change by Year:")
        print(yoy_avg_close_price_change_by_year)
        print("Year-over-Year Average Closing Price Change:")
        print(f"{yoy_avg_close_price_change}%")

        # Create a trend icon for if the YOY price is positive or negative:
        if yoy_avg_close_price_change > 0:
            trend_icon = positive_icon  # Use positive trend icon
        elif yoy_avg_close_price_change < 0:
            trend_icon = negative_icon  # Use negative trend icon
        else:
            trend_icon = neutral_icon  # Neutral trend icon if the difference is 0

        # Write avg price change % over last 10 yrs to sidebar
        kpi_col3.write("AVG YOY Price Change (Last 10 Yrs):")
        yoy_avg_close_price_change = ("$" + str(avg_price_chg_dollar_amt) + " | " + str(yoy_avg_close_price_change) + "% " + trend_icon)

        # Write to Home Page
        if yoy_avg_close_price_change is not None:
            # Write the value to the app for today in KPI format if the data is available
            kpi_col3.markdown(yoy_avg_close_price_change, unsafe_allow_html=True)
        else:
            kpi_col3.warning(f"AVG YOY Price Change (Last 10 Yrs): Data Not Available")
# ------------------------------ Add KPI for Avg YOY price change over last 10 years from yesterday -----------------------------------------

# -----------------------------------  Add KPI for Forecasted Price Based on Forecast Slider ------------------------------------------------
with kpi_col4:
    kpi_col4 = st.container(border=True)
    with kpi_col4:
        # chooses the seasonal trend price (yhat) at the forecasted year dynamically with the sidebar
        chosen_forecasted_price = forecast['yhat'].iloc[-1]  # forecast['yhat'].iloc[-1] retrieves the forecasted value
        print("Price Forecasted in Chosen Forecast Yr", chosen_forecasted_price)

        # Get Forecasted X Year $ Difference & Percentage:
        trend_difference = chosen_forecasted_price - Current_Price
        trend_difference_number = round(trend_difference, 2)  # round the number to two decimal points
        trend_difference_percentage = (trend_difference/Current_Price)
        trend_difference_percentage = round(trend_difference_percentage * 100)

        # Print the result
        print("Difference in trend between next year and today (dollars):", trend_difference_number)
        print("Difference in trend between next year and today (% return):", trend_difference_percentage)

        # Create a trend icon for if the forecasted price is positive or negative
        if trend_difference_percentage > 0:
            trend_icon = positive_icon  # Use positive trend icon
        elif trend_difference_percentage < 0:
            trend_icon = negative_icon  # Use negative trend icon
        else:
            trend_icon = neutral_icon  # Neutral trend icon if the difference is 0

        # Give title for 1 year forecasted price:
        chosen_forecasted_year = str(int(Current_Year) + int(forecasted_year_range))
        kpi_col4.write(f" {chosen_forecasted_year} Price Forecast:")

        # Forecasted Price Difference in 1 year ($ Difference + Current Price & % Difference Concatenated)
        chosen_forecasted_price_kpi = ("$" + str(round(chosen_forecasted_price, 2)) + " | " + str(trend_difference_percentage) + "% " + trend_icon)

        # Write 1 Year Forecast Diff to sidebar
        if chosen_forecasted_price_kpi is not None:
            # Write the value to the app for today in KPI format if the data is available
            kpi_col4.markdown(chosen_forecasted_price_kpi, unsafe_allow_html=True)
        else:
            kpi_col4.warning(f"1 Year Forecast: Data Not Available")
# -----------------------------------  Add KPI for Forecasted Price Based on Forecast Slider ------------------------------------------------

# ------------------------------------------------------ Home Page: KPISs -------------------------------------------------------------------






# --------------------------------------------- Home Page: Historical/Current Data Visuals --------------------------------------------------------

# Create a tab list to navigate to different sections on our home page:
home_tab1, home_tab2, home_tab3, home_tab4 = st.tabs(["Historical/Current Data", "Forecasted Data", "Stock Grades", "Chat-Bot"])

with home_tab1:
    # create variable for a container to put our stock detail section in:
    sh_c = st.container()  # add a fixed height of 800 px and add border to container

    # define columns for the stock_details container (sd_c)
    sh_col1, sh_col2 = sh_c.columns([4, 6])  # Use ratios to make control area width

    # write container to app with columns for our stock current details and history data:
    with sh_c:

        # Within sh_col1, create two columns:
        with sh_col1:
            sh_col1.write('Key Metrics:')
            sh_col1 = st.container(border=True, height=489)
            with sh_col1:
                sh_col1_1, sh_col1_2 = sh_col1.columns(2)

                # run if statement to check there is data available today first:
                if not stock_data.empty:

                    # Check if data is available for the field today
                    if regular_market_price is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_1.metric(label="Current Price:", value="$" + str(round(regular_market_price, 2)))  # round # to 2 decimal points and make a string
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_1.warning(f"Current Price: {NA}")

                    # Check if data is available for the field today
                    if regular_market_open is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_1.metric(label="Today's Open Price:", value="$" + str(round(regular_market_open, 2)))  # round # to 2 decimal points and make a string
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_1.warning(f"Today's Open Price: {NA}")

                    # Add Latest Close Price:
                    # Note: we already set a var for latest close price in the load data section of the app
                    if last_close_price is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_2.metric(label=f"Last Close Price ({last_close_date}):", value="$" + str(round(last_close_price, 2)))
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_2.warning(f"Last Close Price: {NA}")

                    # Add Trade Volume:
                    if regular_market_volume is not None:
                        regular_market_volume = "{:,}".format(round(regular_market_volume, 2))  # adds commas
                        sh_col1_2.metric(label=f"Today's Trade Volume:", value=str(regular_market_volume))

                    # If variable from info is missing, check if we can get from history table
                    elif regular_market_volume is None:

                        # create variable to pull latest vol from history table
                        today_vol = stock_data.loc[stock_data['Date'] == today.strftime('%Y-%m-%d'), 'Volume']
                        today_vol = today_vol.iloc[0]

                        # Format trade volume number with commas for thousands separator
                        today_vol = "{:,.2f}".format(regular_market_volume)

                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_2.metric(label="Today's Trade Volume:", value=str(today_vol))

                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_2.warning(f"Today's Trade Volume: {NA}")

                    # Create a variable for PE ratio & extract the data for today if available:
                    today_pe = pe_ratio

                    # Check if data is available for the field today
                    if today_pe is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_1.metric(label="PE:", value=str(round(today_pe, 2)))
                        print(f"PE Ratio today: {str(round(today_pe, 2))}")
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_1.warning(f"PE: {NA}")
                        print(f"PE: {NA}")

                    # Create a variable for PEG ratio & extract the data for today if available:
                    today_peg = peg_ratio

                    # Check if data is available for the field today
                    if today_peg is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_1.metric(label="PEG:", value=today_peg)
                        print(f"PEG Ratio today: {str(today_peg)}")
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_1.warning(f"PEG: {NA}")
                        print(f"PEG: {NA}")

                    # Create a variable for Price-to-Book ratio & extract the data for today if available:
                    today_PB_ratio = price_to_book

                    # Check if data is available for the field today
                    if today_PB_ratio is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_1.metric(label="Price-to-Book:", value=str(round(today_PB_ratio, 2)))
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_1.warning(f"Price-to-Book: {NA}")

                    # Create a variable for Debt-to-Equity ratio & extract the data for today if available:
                    today_DE_ratio = debt_to_equity

                    # Check if data is available for the field today
                    if today_DE_ratio is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_1.metric(label="Debt-to-Equity:", value=str(round(today_DE_ratio*100, 2)) + '%')
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_1.warning(f"Debt-to-Equity: {NA}")

                    # Create a variable for Dividend Yield ratio & extract the data for today if available:
                    today_DY_ratio = dividend_yield

                    # Check if data is available for the field today
                    if today_DY_ratio is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_1.metric(label="Dividend Yield:", value=str(round(today_DY_ratio, 2)) + "%")
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_1.warning(f"Dividend Yield: {NA}")

                    # Add Net Profit Margin
                    if net_profit_margin is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_1.metric(label="Net Profit Margin:", value=str(round(net_profit_margin * 100, 2)) + "%")
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_1.warning(f"Net Profit Margin: {NA}")

                    # Check if data is available for the field today
                    if beta is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_1.metric(label="Beta:", value=str(round(beta, 2)))
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_1.warning(f"Beta: {NA}")

                    # Create a variable for Price-to-Sales ratio & extract the data for today if available:
                    today_PS_ratio = price_to_sales
                    # Check if data is available for the field today
                    if today_PS_ratio is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_2.metric(label="Price-to-Sales:", value=str(round(today_PS_ratio, 2)))
                    else:
                        sh_col1_2.warning(f"Price-to-Sales: {NA}")

                    # Create a variable for ROE ratio & extract the data for today if available:
                    today_ROE_ratio = roe

                    # Add Return On Equity
                    if today_ROE_ratio is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_2.metric(label="ROE:", value=str(round(today_ROE_ratio, 2)))
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_2.warning(f"ROE Ratio: {NA}")

                    # Create a variable for Current Ratio & extract the data for today if available:
                    today_CR_ratio = current_ratio

                    # Add Current Ratio
                    if today_CR_ratio is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_2.metric(label="Current Ratio:", value=str(round(today_CR_ratio, 2)))
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_2.warning(f"Current Ratio: {NA}")

                    # Create a variable for Quick Ratio & extract the data for today if available:
                    today_QR_ratio = quick_ratio

                    # Add Quick Ratio
                    if today_QR_ratio is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_2.metric(label="Quick Ratio:", value=str(round(today_QR_ratio, 2)))
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_2.warning(f"Quick Ratio: {NA}")

                    # Get Dividend Rate
                    if dividend_rate is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_2.metric(label="Dividend Rate (Annual):", value='$' + str(round(dividend_rate, 2)))
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_2.warning(f"Dividend Rate (Annual): {NA}")

                    # Add YOY Operational Growth:
                    if yoy_ocfg_growth is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_2.metric(label=f"YOY OCF Growth:", value=round(yoy_ocfg_growth, 2))
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_2.warning(f"YOY OCF Growth: {NA}")

                    # Add Sharpe Ratio:
                    if sharpe_ratio is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1_2.metric(label=f"Sharpe Ratio:", value=str(sharpe_ratio))
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1_2.warning(f"Sharpe Ratio: {NA}")

                    # Add 52 Week Price Range:
                    if fifty_two_week_range is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1.metric(label=f"52 Week Range:", value="$" + str(fifty_two_week_range))
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1.warning(f"52 Week Range: {NA}")

                    # Add Enterprise Value Range:
                    if enterprise_value is not None:
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1.metric(label=f"Enterprise Value:", value="$" + "{:,.0f}".format(round(enterprise_value, 0)))
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1.warning(f"Enterprise Value: {NA}")

                    # Add Analyst Recommendation:
                    if analyst_recommendation_summary is not None:
                        # Capitalize the first letter of the value
                        capitalized_value = str(analyst_recommendation_summary).capitalize()
                        # Write the value to the app for today in KPI format if the data is available
                        sh_col1.metric(label=f"Analyst Rating:", value=capitalized_value)
                    else:
                        # Write the data is not available for the field if missing
                        sh_col1.warning(f"Analyst Rating: {NA}")

                # if there is no data available at all for today, print no data available
                else:
                    st.warning("Stock data for this ticker is missing.")

        with sh_col2:
            # Add Title
            Price_History_Tbl_Title = "Price History (Limit Last 10 Years):"
            sh_col2.write(Price_History_Tbl_Title)

            # Create a Time Series Visual for our Data in column 2 of the sh container:
            with st.container(border=True):

                # Apply an exponential smoothing line to the graph starting - create smoothed pricing variable
                smoothed_prices = apply_exponential_smoothing(stock_data, smoothing_level=0.002)  # can change alpha

                # Add a variable for trend color
                # Calculate the trend direction based on the last two smoothed values
                trend_direction = "up" if smoothed_prices.iloc[-1] > smoothed_prices.iloc[0] else "down"

                # Define color based on trend direction
                trend_color = 'rgba(0, 177, 64, .8)' if trend_direction == "up" else 'rgba(244, 67, 54, 0.8)'

                # Define color based on trend direction
                trend_fill = 'rgba(0, 177, 64, 0.2)' if trend_direction == "up" else 'rgba(244, 67, 54, 0.2)'

                def plot_raw_data():  # define a function for our plotted data (fig stands for figure)
                    fig = go.Figure()  # create a plotly graph object.
                    fig.add_trace(go.Scatter(
                        x=stock_data['Date'],
                        y=stock_data['Close'],
                        name='Price',
                        fill='tozeroy',  # adds color fill below trace
                        line=dict(color=trend_color),  # give line color based on smoothing trend line
                        fillcolor=trend_fill  # Light green with transparency
                    ))

                    # Trace the exponential smoothing line to the graph
                    fig.add_trace(go.Scatter(
                        x=stock_data['Date'],
                        y=smoothed_prices,
                        name='Smoothing (a = .002)',
                        line=dict(color='yellow', width=1.5, dash='dash')  # Red dashed line for smoothed prices
                    ))

                    # Trace a line for the S&P 500 (SPY) as a benchmark to the selected stock
                    # First, need to normalize the SPY data to match the stock's initial price
                    stock_initial_price = stock_data['Close'].iloc[0]
                    spy_initial_price = SPY_data['Close'].iloc[0]

                    # Scale the SPY data to the stock's initial price
                    normalized_spy = (SPY_data['Close'] / spy_initial_price) * stock_initial_price

                    # Trace to graph
                    fig.add_trace(go.Scatter(
                        x=SPY_data['Date'],
                        y=normalized_spy,
                        name='S&P 500',
                        line=dict(color='gray', width=1.5, dash='dash')  # Red dashed line for smoothed prices
                    ))

                    # Update layout
                    fig.layout.update(xaxis_rangeslider_visible=True,template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)  # writes the graph to app and fixes width to the container width
                plot_raw_data()

        # Create a Dropdown for stock desc - Get desc from yfinance
        stock_description = stock_info.get('longBusinessSummary', 'No description available')

        # Function to split the description into paragraphs every 3 sentences
        def split_into_paragraphs(text, sentence_count=4):
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            paragraphs = [". ".join(sentences[i:i + sentence_count]) + '.' for i in range(0, len(sentences), sentence_count)]
            return paragraphs

        # Get the paragraphs by splitting the stock description into chunks of 3 sentences
        paragraphs = split_into_paragraphs(stock_description, 4)

        # Wrap in dropdown visual
        with st.expander(f"{company_name} Overview - {sector} / {selected_industry}"):
            for paragraph in paragraphs:
                st.markdown(paragraph)

        # Price Trend Section (Long Term Price Trend)
        sh_c.write("Price Trend History - Long Term (Limit 10 Years):")

        # Add a Yearly Data Trend visual to container:
        with sh_c.container(border=True):

            # Apply HTML formatting for "Price Trend" column to add as a column in the df
            def highlight_trend(val):
                if val == '↑':
                    color = 'green'
                elif val == '↓':
                    color = 'red'
                elif val == '■':
                    color = 'grey'
                else:
                    color = 'black'
                return f'color: {color}'

            # Drop columns not needed for the visual
            yearly_data = yearly_data.drop(columns=['Year', 'Returns', 'Excess_Returns'])

            # order yearly_data by date
            yearly_data = yearly_data.sort_values(by='Date', ascending=False)

            # Convert 'Date' column to datetime format
            yearly_data['Date'] = pd.to_datetime(yearly_data['Date'])

            # Format 'Date' column to remove the time component
            yearly_data['Date'] = yearly_data['Date'].dt.strftime('%Y-%m-%d')

            # Move the Trend column to the first column
            yearly_data_columns = ['Trend'] + [col for col in yearly_data.columns if col != 'Trend']  # variable for columns
            yearly_data = yearly_data[yearly_data_columns]

            print(f"styled yearly data df {yearly_data}")

            # Apply the style function above to the "Trend" column
            styled_df = yearly_data.style.map(highlight_trend, subset=['Trend'])

            # Assign again as yearly_df
            yearly_data = styled_df

            # write a table of the past data within the last 10 years on this date
            st.dataframe(yearly_data, hide_index=True, use_container_width=True)

            # Price Trend Section (Long Term Price Trend)
            sh_c.write("Price Trend History - Short Term:")

            # Add Short Term Trend visuals to ST Trend Section; split into two columns in app:
            sh_col1, sh_col2 = sh_c.columns([6, 4])  # Use ratios to make control area width

            with sh_col1.container(border=True):
                def plot_mov_avg_data():  # define a function for our plotted data (fig stands for figure)

                    # Filter moving average df to include only the last year of data
                    one_yr_moving_average_data = moving_average_data[moving_average_data['Date'] >= one_year_ago]

                    fig = go.Figure()  # create a plotly graph object.
                    fig.add_trace(go.Scatter(
                        x=one_yr_moving_average_data['Date'],
                        y=one_yr_moving_average_data['Close'],
                        name='Price',
                        fill='tozeroy',  # adds color fill below trace
                        line=dict(color='grey')
                    ))

                    # Trace for the 200-day SMA
                    fig.add_trace(go.Scatter(
                        x=one_yr_moving_average_data['Date'],
                        y=one_yr_moving_average_data['200_day_SMA'],
                        name='200-Day SMA',
                        line=dict(color='orange', width=1),
                        opacity=.8
                    ))

                    # Split the 50-day SMA into two parts: Above the 200-day SMA and Below the 200-day SMA
                    above_50_sma = one_yr_moving_average_data[one_yr_moving_average_data['50_day_SMA'] > one_yr_moving_average_data['200_day_SMA']]
                    below_50_sma = one_yr_moving_average_data[one_yr_moving_average_data['50_day_SMA'] < one_yr_moving_average_data['200_day_SMA']]

                    # Trace for the 50-day SMA when above the 200-day SMA (green)
                    fig.add_trace(go.Scatter(
                        x=above_50_sma['Date'],
                        y=above_50_sma['50_day_SMA'],
                        name='50-Day SMA (Above 200)',
                        line=dict(color='#00B140', width=2, dash='dash'),
                        opacity=1
                    ))

                    # Trace for the 50-day SMA when below the 200-day SMA (red)
                    fig.add_trace(go.Scatter(
                        x=below_50_sma['Date'],
                        y=below_50_sma['50_day_SMA'],
                        name='50-Day SMA (Below 200)',
                        line=dict(color='red', width=2, dash='dash'),
                        opacity=1
                    ))

                    # Update layout
                    fig.layout.update(title='Moving Averages - One Year', xaxis_rangeslider_visible=True, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)  # writes the graph to app and fixes width to the container width
                plot_mov_avg_data()

                # kpi for Present day SMA indicator
                with sh_col1.container(border=True):

                    # Logic to decide if it's a good buy or sell based on the crossover
                    if sma_percentage_difference > 5:
                        signal = "Bullish Momentum"
                        color = "rgba(0, 177, 64, 0.6)"  # Green color for Buy
                        text_color = "white"  # Setting as white but leaving a variable if want to change in the future
                        action = "BUY"
                        indicator = f"{sma_percentage_difference:.2f}% Price Differential*"
                    if sma_percentage_difference < -5:
                        signal = "Bearish Momentum"
                        color = "rgba(244, 67, 54, 0.6)"  # Red color for Sell
                        text_color = "white"  # Setting as white but leaving a variable if want to change in the future
                        action = "SELL"
                        indicator = f"{sma_percentage_difference:.2f}% Price Differential*"
                    else:
                        signal = "Neutral Momentum"
                        color = "rgba(255, 255, 0, 0.6)"  # Yellow color for Neutral
                        text_color = "white"  # Setting as white but leaving a variable if want to change in the future
                        action = "HOLD"
                        indicator = f"{sma_percentage_difference:.2f}% Price Differential*"

                    # Provide KPI title
                    st.markdown("<p style='margin: 0; padding: 0; font-size: 14px; '>Latest 50-Day & 200-Day SMA Price Differential:</p>", unsafe_allow_html=True)  # writing with html removes extra spacing between lines

                    # Write SMA Momentum Indicator to App

                    # CSS styling notes for markdown below:
                    # - display: flex - Creates a flexible container
                    # - flex-direction: row - Arranges items horizontally
                    # - align-items: center - Vertically centers the items
                    # - flex-wrap: nowrap - Prevents items from wrapping to next line
                    # - width: 100% - Uses full width of container
                    # - overflow: hidden - Hides any content that overflows
                    # - font-size: clamp() - Responsive font sizing that scales between minimum and maximum values
                    # - white-space: nowrap - Prevents text from wrapping
                    # - text-overflow: ellipsis - Shows ellipsis (...) for truncated text
                    # - max-width: 80% - Limits width to prevent overflow on small screens

                    st.markdown(f"""
                        <div style='
                            display: flex; 
                            flex-direction: row; 
                            align-items: center;
                            flex-wrap: nowrap;
                            width: 100%;
                            overflow: hidden;
                        '>
                            <div style='
                                font-size: clamp(24px, 8vw, 48px);
                                margin: 0; 
                                padding: 0;
                                white-space: nowrap;
                            '>${sma_price_difference:.2f}</div>
                            <div style='
                                background-color: {color}; 
                                color: {text_color}; 
                                padding: 6px 8px; 
                                border-radius: 20px; 
                                font-size: clamp(9px, 2.5vw, 12px);
                                font-weight: bold; 
                                text-align: center; 
                                margin-left: 10px;
                                white-space: nowrap;
                                overflow: hidden;
                                text-overflow: ellipsis;
                                max-width: 80%;
                            '>
                                {signal}: {indicator}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            # Create a visual for RSI data over 14-day window
            with sh_col2.container(border=True):

                # Write a buy, hold, sell momentum "if" logic based on rsi
                if latest_rsi_value > 70:
                    signal = "Currently Overbought - Consider Sell*"
                    color = "rgba(244, 67, 54, 0.6)"  # Sell signal in red
                    text_color = "white"  # Setting as white but leaving a variable if want to change in the future
                elif latest_rsi_value < 30:
                    signal = "Currently Oversold - Consider Buy*"
                    color = "rgba(0, 177, 64, 0.6)"  # Buy signal in green
                    text_color = "white"  # Setting as white but leaving a variable if want to change in the future
                else:
                    signal = "Neutral Momentum - Consider Holding*"
                    color = "rgba(255, 255, 0, 0.6)"  # Neutral signal in yellow
                    text_color = "white"  # Setting as white but leaving a variable if want to change in the future

                # Provide visual title
                st.markdown("<p style='margin: 0; padding: 0; font-size: 14px; '>RSI Value (14-Day Window):</p>", unsafe_allow_html=True)  # writing with html removes extra spacing between lines

                # Write value and buy/hold/sell indicator to app
                st.markdown(f"""
                    <div style='
                        display: flex; 
                        flex-direction: row; 
                        align-items: center;
                        flex-wrap: nowrap;
                        width: 100%;
                        overflow: hidden;
                    '>
                        <div style='
                            font-size: clamp(24px, 8vw, 48px);
                            margin: 0; 
                            padding: 0;
                            white-space: nowrap;
                        '>{latest_rsi_value:.2f}</div>
                        <div style='
                            background-color: {color}; 
                            color: {text_color}; 
                            padding: 6px 8px; 
                            border-radius: 20px; 
                            font-size: clamp(9px, 2.5vw, 12px);
                            font-weight: bold; 
                            text-align: center; 
                            margin-left: 10px;
                            white-space: nowrap;
                            overflow: hidden;
                            text-overflow: ellipsis;
                            max-width: 80%;
                        '>
                            {signal}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Create Container for MACD visual
            with sh_col2.container(border=True):

                # Plot MACD chart with Signal and Histogram
                with st.container():

                    # MACD + Signal Line Plot
                    fig = go.Figure()

                    # Add MACD Line
                    fig.add_trace(go.Scatter(
                        x=macd_data['Date'],
                        y=macd_data['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='rgba(255, 255, 0, 0.6)')))  # yellow

                    # Add Signal Line
                    fig.add_trace(go.Scatter(
                        x=macd_data['Date'],
                        y=macd_data['Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='rgba(244, 67, 54, 0.6)')))  # red

                    # Add Histogram (Difference between MACD and Signal)
                    fig.add_trace(go.Bar(
                        x=macd_data['Date'],
                        y=macd_data['Histogram'],
                        name='Histogram',
                        marker=dict(color='rgba(128, 128, 128, 0.6)'),
                    ))

                    # Update layout with titles and axis labels
                    fig.update_layout(
                        title='MACD - YoY',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        template='plotly_white',
                        barmode='relative',
                        showlegend=True
                    )

                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)

        # Add a container for short-term trend section footnote expander:
        with sh_c.container():

            # write expander to app
            with st.expander("Leveraging Short Term Models for Short-Term Buy/Sell/Hold Actions*"):  # add footnote drop-down
                st.markdown("""
                <span style="color:lightcoral; font-weight:bold;">**Relative Strength Index (RSI)**</span> is a momentum oscillator that provides a range from 0-100 to indicate if the asset is under or overpriced based on its speed and change of price movements.
                
                Interpret the RSI Ranges can be interpreted with the following logic:
                - <span style="color:lightcoral; font-weight:bold;">**RSI > 70**</span> = Likely Overbought (Sell Signal): The maximum expected loss for one day.
                - <span style="color:lightcoral; font-weight:bold;">**RSI ≈ 50**</span> = Neutral State: The maximum expected loss for one month.
                - <span style="color:lightcoral; font-weight:bold;">**RSI < 30**</span> = Likely Oversold (Buy Signal): The maximum expected loss for one year.
        
                <span style="color:lightcoral; font-weight:bold;">**Example**</span>:
                - If Meta's RIS is under 20, this indicates a more-extreme likelihood of market oversell compared to if the RSI were at 40, indicating market activity is in a neutral state
                """, unsafe_allow_html=True)

        # Add Risk Section:
        sh_c.write("Risk Assessment:")

        # Create new Container for VaR Risk metrics:
        with sh_c.container(border=True):

            # Header for VAR risk section
            st.markdown("<h4 style='text-align: center;'>Historical VaR - Model Metrics</h3>", unsafe_allow_html=True)

            # Write to container horizontally in 3 columns:
            risk_col1, risk_col2, risk_col3 = st.columns(3)

            with risk_col1:
                st.metric(label="1-Day VaR at 95% Confidence", value=f"{hist_daily_VaR_95 * 100:.2f}%", delta=f"${hist_VaR_95_dollars:.2f}", delta_color="inverse")

            with risk_col2:
                st.metric(label="1-Month VaR at 95% Confidence", value=f"{hist_monthly_VaR_95 * 100:.2f}%", delta=f"${hist_monthly_VaR_95_dollars:.2f}", delta_color="inverse")

            with risk_col3:
                st.metric(label="1-Year VaR at 95% Confidence", value=f"{hist_yearly_VaR_95 * 100:.2f}%", delta=f"${hist_yearly_VaR_95_dollars:.2f}", delta_color="inverse")

            # VAR Explanation
            with st.expander("Value at Risk (VaR)*"):
                st.markdown("""
                <span style="color:lightcoral; font-weight:bold;">**Value at Risk (VaR)**</span> is a measure used to assess the risk of loss on an investment. It indicates the maximum potential loss over a specified time horizon (daily, monthly, or yearly) at a given confidence level (e.g., 95%).
        
                - <span style="color:lightcoral; font-weight:bold;">**Daily VaR**</span>: The maximum expected loss for one day.
                - <span style="color:lightcoral; font-weight:bold;">**Monthly VaR**</span>: The maximum expected loss for one month.
                - <span style="color:lightcoral; font-weight:bold;">**Yearly VaR**</span>: The maximum expected loss for one year.
        
                <span style="color:lightcoral; font-weight:bold;">**How it's Calculated in this Model**</span>:
                The VaR is calculated by looking at the historical adj closing price data over the last 10 years (end YTD) and finding the point at which the worst 5% of returns fall. For example, if the VaR at a 95% confidence level is 3%, it means there is a 95% chance that the loss will not exceed 3% on the given time horizon.
        
                <span style="color:lightcoral; font-weight:bold;">**Example**</span>:
                - If the 1-day VaR is 5%, it means there is a 95% chance the value of the asset will not drop more than 5% in one day.
                """, unsafe_allow_html=True)

        # Create new Container for Monte Carlo Simulation Container:
        with sh_c.container(border=True):

            # Header for VAR risk section
            st.markdown("<h4 style='text-align: center;'>Monte Carlo Simulation - 1 Year Trading Period</h3>", unsafe_allow_html=True)

            # Plot the simulation and percentiles together
            plt.figure(figsize=(10, 4))

            # Set color map
            mc_colormap = plt.cm.plasma  # sets to "plasma" color map for variable
            mc_colors = [mc_colormap(i / mc_simulations_num) for i in range(mc_simulations_num)]  # applies color map to iterations

            # Plot all simulated paths
            for i in range(mc_simulations_num):
                plt.plot(mc_simulations[i], color=mc_colors[i], alpha=0.1)

            # Plot the percentiles on top
            plt.plot(mc_percentile_5, label="5th Percentile", color='red', linestyle='--', linewidth=1)
            plt.plot(mc_percentile_50, label="50th Percentile (Median)", color='white', linestyle='--', linewidth=1)
            plt.plot(mc_percentile_95, label="95th Percentile", color='green', linestyle='--', linewidth=1)

            # plot labels
            plt.xlabel('Days', fontsize=6, family='sans-serif', color=(1, 1, 1, 0.7))  # 50% transparent white; couldn't use "alpha"
            plt.ylabel('Price', fontsize=6, family='sans-serif', color=(1, 1, 1, 0.7))  # 50% transparent white
            plt.legend(fontsize=6, frameon=False, labelcolor='white')

            # Set the background to transparent (so it matches app background)
            plt.gcf().patch.set_facecolor('none')  # Transparent figure background
            plt.gca().patch.set_facecolor('none')  # Transparent axes background

            # Change the ticks color to 50% transparent white (use RGBA format for color)
            plt.tick_params(axis='both', colors=(1, 1, 1, 0.5))  # 50% transparent white for ticks

            # Change the outer grid and spines to transparent
            for spine in plt.gca().spines.values():
                spine.set_edgecolor('none')
                # spine.set_linewidth(0.5)
                # spine.set_alpha(0.5)

            # Adjust the layout to fit within the Streamlit container
            plt.tight_layout()

            # Display the combined plot in Streamlit
            st.pyplot(plt)

            # Monte Carlo Explanation
            with st.expander("Monte Carlo*"):
                st.markdown("""
                <span style="color:lightcoral; font-weight:bold;">**Monte Carlo**</span> is a forecast simulation used to assess the volatility of an investment's probabilistic future price paths over a set time period (the simulation in app uses 252 days for trading days within a year) based on historical price movements.

                <span style="color:lightcoral; font-weight:bold;">**How it's Calculated in this Model**</span>:
                The simulation generates 1000 price paths over a full year trading period using a 10 year (or earliest available under 10 years) daily ticker price history set and calculates the 5, 50, and 95 confidence intervals to compare against.

                - <span style="color:lightcoral; font-weight:bold;">**95th Percentile**</span>: 95% of the simulated price paths fell under the 95th percentile path; Only 5% of the simulations were above.
                - <span style="color:lightcoral; font-weight:bold;">**50th Percentile**</span>: 50% of the simulated price paths fell under the 50th percentile path. 50% were above. This is the median path and most likely outcome.
                - <span style="color:lightcoral; font-weight:bold;">**5th Percentile**</span>: Only 5% of the simulated price paths fell under the 5th percentile path. 95% of the simulations were above.
                """, unsafe_allow_html=True)

        st.write("Industry Averages:")
        with sh_c.container(border=True):
            # Add DF with industry averages
            if not industry_avg_df.empty:
                st.dataframe(industry_avg_df, hide_index=True, use_container_width=True)






# --------------------------------------------- Home Page: Historical/Current Data Visuals --------------------------------------------------------





















# --------------------------------------------- Home Page: Add Stock Price Forecasting Visuals --------------------------------------------------------

# Add forecasted data visuals to our second tab
with home_tab2:

    # Create forecast container for our forecast section and columns:
    fs_c = st.container()

    # Add Visuals into fs_c container:
    with fs_c:

        # Create two columns for fs_c
        fs_c_col1, fs_c_col2 = fs_c.columns([7, 3])  # Use ratios to make control area width

        # Write Title for Forecast Graph in col1:
        fs_c_col1.write("Forecast Graph:")

        # Write Forecast Graph Container in col1:
        fs_graph_c = fs_c_col1.container(border=True)
        with fs_graph_c:

            # Plot the forecasted future data using the prophet model within a forecasted visual:
            fig1 = plot_plotly(trained_model, forecast)  # plot visual (plotly needs the model and forecast to plot)

            # Remove extra spacing at the Y-axis
            fig1.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),  # Adjust margin to remove any extra spacing in the container
                xaxis_title='Date',
                yaxis_title='Price'
            )

            # Plot the Forecast graph
            fs_graph_c.plotly_chart(fig1, use_container_width=True)

        # Write Title for Forecasted Price Metric in col2:
        fs_c_col2.write("Forecasted Price:")

        # Create a container for metrics in col2
        fs_price_metric = fs_c_col2.container()
        with fs_price_metric:

            # create columns for each metric in the fs_price_metric container
            fs_price_metric_col1, fs_price_metric_col2 = fs_price_metric.columns([1, 1])

            # Write metric 1 container
            fs_price_metric1 = fs_price_metric_col1.container(border=True)
            with fs_price_metric1:

                # Recreate KPI 4 as streamlit metric
                fs_price_metric1.metric(f"Forecast Year: {chosen_forecasted_year}", f"${str(round(chosen_forecasted_price, 2))}", f"{trend_difference_percentage}%")

            # Write metric 2 container
            fs_price_metric2 = fs_price_metric_col2.container(border=True)
            with fs_price_metric2:

                # Get the average YOY forecasted price change
                yoy_avg_fr_price_change = round((int(chosen_forecasted_price) - int(Current_Price)) / (int(chosen_forecasted_year) - int(Current_Year)), 2)

                # Get the Average Change %
                yoy_avg_fr_price_change_pct = round((yoy_avg_fr_price_change / Current_Price) * 100, 0)

                # Add a metric for avg forecast price change from current price YOY dynamically with sidebar
                fs_price_metric2.metric(f"YOY Change Average:", f"${str(round(yoy_avg_fr_price_change, 2))}", f"{yoy_avg_fr_price_change_pct}%")



        # Write Title for Forecast Components in col2:
        fs_c_col2.write("Forecast Components:")

        # Write Forecast Components Container in col2
        fs_components_c = fs_c_col2.container(border=True, height=456)
        with fs_components_c:
            # Plot the Prophet forecast components:
            fig2 = trained_model.plot_components(forecast) # write component visuals to label fig2.
            fs_components_c.pyplot(fig2) # write figure 2 to the app

            # Notes:
            # By calling m.plot_components(forecast), we're instructing prophet to generate a plot showing the components of the forecast. These components include trends, seasonalities, and holiday effect
            # The result of m.plot_components(forecast) is a set of plots visualizing these components

        # Write Title for Forecast Tail:
        fs_c.write("Price Forecast Overview:")

        # Write Forecast Overview Container:
        fs_overview_c = fs_c.container(border=True)
        with fs_overview_c:
            # PLot our forecasted information in tabular format:
            fs_overview_c.dataframe(forecast.tail(1), hide_index=True)


# --------------------------------------------- Home Page: Add Stock Price Forecasting Visuals --------------------------------------------------------



# ///////////////////////////////////////////////////////////// Home Page //////////////////////////////////////////////////////////////////////////




















# ///////////////////////////////////////////////////////////// Stock Grader Page //////////////////////////////////////////////////////////////////////////

with home_tab3:

    # Create container variable for the grades tab
    sh_g = st.container()

    # Title for Custom Model Section
    sh_g.write("Custom Grade Model:")

    # Write Custom Model Grade to App
    with sh_g.container(border=True):

        # Title to Add the Model Build is in Progress:
        st.markdown("<h2 style='text-align: center;'>Custom Grade Model Build in Progress...</h2>", unsafe_allow_html=True)
        st.header("")
        st.header("")

        # Container for Stock Grades Page:
        sg_c = st.container(height=500, border=False)

        # Add Animations for WIP:

        # Display the cog wheel spinner with an adjusted larger cogwheel size
        adjusted_cog_size = 150  # Adjust the size to 100px (larger cogwheel)
        sg_c.markdown(cog_wheel_css(adjusted_cog_size) + cog_html(), unsafe_allow_html=True)

    # Title for Yahoo Finance Analyst Section
    sh_g.write("Analyst Grades / Predictions:")

    # Write Analyst Grades to App
    with sh_g.container(border=True):

        # ----------------- Yahoo Finance Grades
        # Display the DataFrame in Streamlit
        st.write("Year End Price Predictions:")
        st.dataframe(stock_analyst_info_df, hide_index=True, use_container_width=True)
        # ----------------- Yahoo Finance Grades

        # ----------------- Agency Grades
        # Display the DataFrame in Streamlit
        st.write("Analyst Recommendations:")
        agency_analyst_info_df = agency_analyst_info
        st.dataframe(agency_analyst_info_df, hide_index=True, use_container_width=True)
        # ----------------- Agency Grades





# ///////////////////////////////////////////////////////////// Stock Grader Page //////////////////////////////////////////////////////////////////////////





# ///////////////////////////////////////////////////////////// Chat Bot Page //////////////////////////////////////////////////////////////////////////

# create tab for an AI ChatBot
with home_tab4:

    # Create groq client
    client = Groq(api_key=st.secrets.get("Groq_API_Key"))

    # Session State
    if "default_model" not in st.session_state:
        st.session_state["default_model"] = "llama3-8b-8192"

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hi there! Any stock-related questions? Drop it below :)"}]

    print(st.session_state)

    # Display the messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input for user message
    if prompt := st.chat_input():
        # append message to message collection
        st.session_state.messages.append({"role": "user", "content": prompt})

        # display the new message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display the assistant response from the model
        with st.chat_message("assistant"):
            # place-holder for the response text
            response_text = st.empty()

            # Call the Groq API
            completion = client.chat.completions.create(
                model=st.session_state.default_model,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True
            )

            full_response = ""

            for chunk in completion:
                full_response += chunk.choices[0].delta.content or ""
                response_text.markdown(full_response)

            # add full response to the messages
            st.session_state.messages.append({"role": "assistant", "content": full_response})
