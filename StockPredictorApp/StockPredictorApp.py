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
from yahoo_fin import stock_info as si
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

import requests
import json
import sys


# Note: in order to view the app on your local host, you will need run the following code on your terminal: streamlit run [insert location of file here {ex: %/stockpredictorapp.py}]
# Note: this is the streamlit red color code: #FF6347










# ////////////////////////////////////////////////// Configure App Layout /////////////////////////////////////////////////////////////////////////


# Set Layout of App, Provide App Version and Provide Title
st.set_page_config(layout='wide')  # sets layout to wide
st.sidebar.markdown("<div style='text-align: center; padding: 20px;'>App Version: 1.3.5 &nbsp; <span style='color:#FF6347;'>&#x271A;</span></div>", unsafe_allow_html=True) # adds App version and red medical cross icon with HTML & CSS code; nbsp adds a space
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

# --------------------------------------------------- Data: Set Time Parameters -----------------------------------------------------------------













# ------------------------------------------ Data: Load Ticker Volume/Pricing/Ratios (Historic & Current) ---------------------------------------

@st.cache_data()
def load_data(ticker):  # Define the "load data" function
    stock_data = yf.download(ticker, start=start_date, end=today)  # Fetch data from Yahoo Finance

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

# Fetch Stock Info and handle errors
try:
    stock_info = yf.Ticker(selected_stock).info  # Try to fetch stock information

    # Check if the response is valid (not empty)
    if not stock_info or "error" in stock_info:
        raise ValueError("Received an empty or invalid response.")

# Handle errors related to the request, invalid JSON, or empty response
except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:

    st.markdown("""
        <style>
            /* Keyframes for rotating the error icon */
            @keyframes rotate {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
    
            /* Keyframes for sliding the error box in */
            @keyframes slideIn {
                0% { transform: translateY(-100%); }
            100% { transform: translateY(0); }
            }
    
            #error-message {
                background: #f8d7da; 
                color: #721c24; 
                padding: 20px 30px; 
                border-radius: 12px; 
                font-family: 'Roboto', sans-serif; 
                box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.1); 
                display: flex; 
                align-items: center; 
                justify-content: space-between; 
                position: relative; 
                animation: slideIn 0.5s ease-out;
            }
    
            #error-message .error-icon {
                font-size: 30px;
                margin-right: 30px;
                animation: rotate 3s infinite linear;
            }
    
            #error-message .close-btn {
                background: none; 
                border: none; 
                font-size: 24px; 
                color: #721c24; 
                cursor: pointer; 
                font-weight: bold; 
                transition: color 0.3s, transform 0.3s;
            }
    
            #error-message .close-btn:hover {
                color: #e63946;
                transform: scale(1.1);
            }
        </style>
    
        <div id="error-message">
            <div style="display: flex; align-items: center;">
                <span class="error-icon">X</span>
                <span style="font-size: 18px; font-weight: 600; line-height: 1.5;">
                    An error occurred while fetching the data. This is likely due from updates to the data source api/package. Hold on tight - app maintenance is in place!
                </span>
        </div>
    """, unsafe_allow_html=True)

    # Stop execution of any further script if the error occurs
    sys.exit()

# Retrieve additional metrics from yahoo finance api
company_name = stock_info['longName']
pe_ratio = stock_info.get('trailingPE', None)
peg_ratio = stock_info.get('pegRatio', None)
price_to_book = stock_info.get('priceToBook', None)
debt_to_equity = stock_info.get('debtToEquity', None)
dividend_yield = stock_info.get('dividendYield', None)
price_to_sales = stock_info.get('priceToSalesTrailing12Months', None)
roe = stock_info.get('returnOnEquity', None)
quick_ratio = stock_info.get('quickRatio', None)
current_ratio = stock_info.get('currentRatio', None)
industry = stock_info.get('industry', None)
sector = stock_info.get('sector', None)
market_cap = stock_info.get('marketCap', None)  # Market Capitalization
beta = stock_info.get('beta', None)  # Stock Beta (volatility)
earnings_date = stock_info.get('earningsDate', None)  # Next earnings date
shares_outstanding = stock_info.get('sharesOutstanding', None)  # Number of shares outstanding
last_dividend = stock_info.get('lastDividendValue', None)  # Last dividend value

# Create a DataFrame to store the ratios for the selected ticker
stock_ratios = pd.DataFrame({
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
    'Last Dividend': [last_dividend]
})

print("todays stock metrics")
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

# ------------------------------------------ Data: Load Ticker Volume/Pricing/Ratios (Historic & Current) ---------------------------------------














# -------------------------------- Data: Add a yearly data dataframe to capture price by year on today's date over last 10 years ---------------------------------------

# Get data from each available day of the current month through all years
print("\n Current Month Yearly Historical Data:")
ct_month_yrly_data = stock_data[stock_data['Date'].dt.month == Current_Month] # filters stock_data df to this month throughout all years
print(ct_month_yrly_data) # check the df is pulling the correct data

# Get data from each available day of the current day through the all years
print("\n Current Day Yearly Historical Data:")
yearly_data = ct_month_yrly_data[ct_month_yrly_data['Date'].dt.day == Current_Day] # filters stock_data df to this day throughout all years
print(yearly_data) # check the df is pulling the correct data

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

# ---------------------------------


# For Benchmarking Purposes, create a separate DF for SPY
SPY_data = load_data("SPY")  # loads the selected ticker data (selected_stock)


# -------------------------------- Data: Add a yearly data dataframe to capture price by year on today's date over last 10 years ---------------------------------------














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

print(forecast)
print(f"forecast year range {forecasted_year_range}")
# ------------------------------------------------------ Data: Forecast Model (Forecasted Data) --------------------------------------------------------------------


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
        trend_difference_number = round(trend_difference, 2)    # round the number to two decimal points
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
                sh_col1.write('Stock Metrics:')
                sh_col1 = st.container(border=True, height=489)
                with sh_col1:
                    sh_col1_1, sh_col1_2 = sh_col1.columns(2)

                    # run if statement to check there is data available today first:
                    if not stock_data.empty:

                        # Create a variable for Open Price & extract the data for today if available:
                        today_open = stock_data.loc[stock_data['Date'] == today.strftime('%Y-%m-%d'), 'Open']

                        # Check if data is available for the field today
                        if not today_open.empty:
                            today_open = today_open.iloc[0]
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_1.metric(label="Today's Open Price:", value="$" + str(round(today_open, 2))) # round # to 2 decimal points and make a string
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_1.warning(f"Today's Open Price: {NA}")


                        # Add Latest Close Price:
                        # Note: we already set a var for latest close price in the load data section of the app
                        if last_close_price is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_2.metric(label=f"Last Close Price (as of {last_close_date}):", value="$" + str(round(last_close_price, 2)))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_2.warning(f"Last Close Price: {NA}")


                        # Create a variable for Trade Volume & extract the data for today if available:
                        today_vol = stock_data.loc[stock_data['Date'] == today.strftime('%Y-%m-%d'), 'Volume']

                        # Check if data is available for the field today and format with commas
                        if not today_vol.empty:
                            today_vol = today_vol.iloc[0]
                            # Format trade volume number with commas for thousands separator
                            today_vol = "{:,.2f}".format(today_vol)
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
                            print(f"PEG Ratio today: {str(round(today_peg, 2))}")
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
                            sh_col1_1.metric(label="Debt-to-Equity:", value=str(round(today_DE_ratio, 2)))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_1.warning(f"Debt-to-Equity: {NA}")


                        # Create a variable for Dividend Yield ratio & extract the data for today if available:
                        today_DY_ratio = dividend_yield
                        # Check if data is available for the field today
                        if today_DY_ratio is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_1.metric(label="Dividend Yield:", value=str(round(today_DY_ratio * 100, 4)) + "%")
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_1.warning(f"Dividend Yield: {NA}")


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
                        # Check if data is available for the field today
                        if today_ROE_ratio is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_2.metric(label="ROE:", value=str(round(today_ROE_ratio, 2)))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_2.warning(f"ROE Ratio: {NA}")


                        # Create a variable for Current Ratio & extract the data for today if available:
                        today_CR_ratio = current_ratio
                        # Check if data is available for the field today
                        if today_CR_ratio is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_2.metric(label="Current Ratio:", value=str(round(today_CR_ratio, 2)))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_2.warning(f"Current Ratio: {NA}")


                        # Create a variable for Quick Ratio & extract the data for today if available:
                        today_QR_ratio = quick_ratio
                        # Check if data is available for the field today
                        if today_QR_ratio is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_2.metric(label="Quick Ratio:", value=str(round(today_QR_ratio, 2)))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_2.warning(f"Quick Ratio: {NA}")


                        # Create a variable for Quick Ratio & extract the data for today if available:
                        # Check if data is available for the field today
                        if sector is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1.metric(label="Sector:", value=str(sector))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1.warning(f"Sector: {NA}")


                        # Create a variable for Quick Ratio & extract the data for today if available:
                        # Check if data is available for the field today
                        if industry is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1.metric(label="Industry:", value=str(industry))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1.warning(f"Industry: {NA}")


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

        # Price Trend Section (Long Term Price Trend)
        sh_c.write("Price Trend History- Long Term (Limit 10 Years):")

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

            # Drop the 'Year' column
            yearly_data = yearly_data.drop(columns=['Year'])

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
                    fig.layout.update(xaxis_rangeslider_visible=True, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)  # writes the graph to app and fixes width to the container width
                plot_mov_avg_data()

                # kpi for Present day SMA indicator
                with sh_col1.container(border=True):

                    # Get latest day price and SMA metrics
                    price = moving_average_data['Close'].iloc[-1]  # Get the latest closing price
                    sma_50 = moving_average_data['50_day_SMA'].iloc[-1]  # Latest 50-day SMA
                    sma_200 = moving_average_data['200_day_SMA'].iloc[-1]  # Latest 200-day SMA

                    # Calculate the difference between 50-day and 200-day SMA
                    sma_price_difference = sma_50 - sma_200

                    # Calculate the percentage difference between the 50-day and 200-day SMA
                    sma_percentage_difference = (sma_price_difference / sma_200) * 100

                    # Logic to decide if it's a good buy or sell based on the crossover
                    if sma_percentage_difference > 5:
                        signal = "Bullish Trend Signal"
                        color = "rgba(0, 177, 64, 0.6)"  # Green color for Buy
                        text_color = "white"  # Setting as white but leaving a variable if want to change in the future
                        action = "BUY"
                        indicator = f"{sma_percentage_difference:.2f}% Price Differential*"
                    if sma_percentage_difference < -5:
                        signal = "Bearish Trend Signal"
                        color = "rgba(244, 67, 54, 0.6)"  # Red color for Sell
                        text_color = "white"  # Setting as white but leaving a variable if want to change in the future
                        action = "SELL"
                        indicator = f"{sma_percentage_difference:.2f}% Price Differential*"
                    else:
                        signal = "Neutral Market Momentum"
                        color = "rgba(255, 255, 0, 0.6)"  # Yellow color for Neutral
                        text_color = "white"  # Setting as white but leaving a variable if want to change in the future
                        action = "HOLD"
                        indicator = f"{sma_percentage_difference:.2f}% Price Differential*"

                    # Provide KPI title
                    st.markdown("<p style='margin: 0; padding: 0; font-size: 14px; '>Latest 50-Day & 200-Day SMA Price Differential:</p>", unsafe_allow_html=True)  # writing with html removes extra spacing between lines

                    # Write KPI data to app
                    st.markdown(f"""
                        <div style='
                            display: inline-block; 
                            font-size: 48px; 
                            margin: 0; 
                            padding: 0; 
                            vertical-align: middle;'>
                            ${sma_price_difference:.2f}
                        </div>
                        <div style='
                            display: inline-block; 
                            background-color: {color}; 
                            color: {text_color}; 
                            padding: 10px 20px; 
                            border-radius: 30px; 
                            font-size: 12px; 
                            font-weight: bold; 
                            text-align: center; 
                            margin-left: 20px;
                            vertical-align: middle;
                        '>
                            {signal}: {indicator}
                        </div>
                    """, unsafe_allow_html=True)

            # Create a visual for RSI data over 14-day window
            with sh_col2.container(border=True):
                rsi_data = stock_data
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

                rsi_values = calculate_rsi(rsi_data)

                # Get the most recent RSI value (latest day)
                latest_rsi_value = rsi_values.iloc[-1]  # This is the RSI for the most recent day

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
                        display: inline-block; 
                        font-size: 48px; 
                        margin: 0; 
                        padding: 0;
                        vertical-align: middle;
                    '>{latest_rsi_value:.2f}</div>
                    <div style='
                        display: inline-block; 
                        background-color: {color}; 
                        color: {text_color}; 
                        padding: 10px 20px; 
                        border-radius: 30px; 
                        font-size: 12px; 
                        font-weight: bold; 
                        text-align: center; 
                        margin-left: 20px;
                        vertical-align: middle;
                    '>
                        {signal}
                    </div>
                """, unsafe_allow_html=True)

            # Create Container for MACD visual
            with sh_col2.container(border=True):
                # Calculate MACD and Signal line to provide additional insight on short term trend momentum and strength
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

                # Plot the MACD chart with Signal and Histogram
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
        with sh_c.container(border=True):

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

            print(stock_data.columns)
            print(stock_data.index)

            # Date Index:
            # The date index of the stock_data df is not in proper format to use the pandas resample() method.
            # Need to reset to datetime index

            # Ensure the 'Date' column is in datetime format (if it isn't already)
            stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')

            # Set the 'Date' column as the index of the DataFrame
            stock_data.set_index('Date', inplace=True)

            # Define function to calculate Historical VaR at 95% confidence level
            def calculate_historical_VaR(stock_data, time_window='daily'):
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
                    stock_data['Return'] = stock_data['Close'].pct_change()
                elif time_window == 'monthly':
                    stock_data['Return'] = stock_data['Close'].resample('ME').ffill().pct_change()
                elif time_window == 'yearly':
                    stock_data['Return'] = stock_data['Close'].resample('YE').ffill().pct_change()
                else:
                    raise ValueError("Invalid time window. Choose from 'daily', 'monthly', or 'yearly'.")

                # Sort in ascending order
                sorted_returns = stock_data['Return'].sort_values()

                # Calculate the 5th percentile (for 95% confidence level)
                hist_VaR_95 = sorted_returns.quantile(0.05)

                # Calculate the VaR in dollars
                current_adj_price = stock_data['Close'].iloc[-1]
                hist_VaR_95_dollars = hist_VaR_95 * current_adj_price

                # Return both VaR in percentage and dollars
                return hist_VaR_95, hist_VaR_95_dollars

            # Function to calculate Daily VaR
            def calculate_daily_VaR(stock_data):
                return calculate_historical_VaR(stock_data, time_window='daily')

            # Function to calculate Monthly VaR
            def calculate_monthly_VaR(stock_data):
                return calculate_historical_VaR(stock_data, time_window='monthly')

            # Function to calculate Yearly VaR
            def calculate_yearly_VaR(stock_data):
                return calculate_historical_VaR(stock_data, time_window='yearly')

            # Calculate hist VaR for daily, monthly, and yearly
            hist_daily_VaR_95, hist_VaR_95_dollars = calculate_daily_VaR(stock_data)
            hist_monthly_VaR_95, hist_monthly_VaR_95_dollars = calculate_monthly_VaR(stock_data)
            hist_yearly_VaR_95, hist_yearly_VaR_95_dollars = calculate_yearly_VaR(stock_data)

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

            # Set parameters for simulation
            mc_simulations_num = 1000  # Number of simulated paths
            mc_days_num = 252  # Number of days to simulate (e.g., one year of trading days)

            # Calculate daily returns (log returns)
            stock_data['Log Return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))

            # Drop any missing values
            stock_data = stock_data.dropna()

            # Display the first few log returns
            stock_data[['Close', 'Log Return']].head()

            # Calculate the mean and volatility from the historical data
            mean_return = stock_data['Log Return'].mean()
            volatility = stock_data['Log Return'].std()

            # Simulate future price paths
            mc_simulations = np.zeros((mc_simulations_num, mc_days_num))  # Shape should be (10000, 252)
            last_price = stock_data['Close'].iloc[-1]  # Use the last closing price as the starting point

            # Ensure the simulation loop works without dimension issues
            for i in range(mc_simulations_num):

                # Generate random returns for each day in the simulation
                mc_random_returns = np.random.normal(mean_return, volatility, mc_days_num)

                # Generate the simulated price path using compound returns
                price_path = last_price * np.exp(np.cumsum(mc_random_returns))  # price_path shape will be (252,)

                # Assign the simulated price path to the simulations array
                mc_simulations[i, :] = price_path  # Assign to the i-th row of the simulations array

            # Convert the simulations to a DataFrame for easier visualization
            mc_simulated_prices = pd.DataFrame(mc_simulations.T, columns=[f'Simulation {i+1}' for i in range(mc_simulations_num)])

            # Check a sample of simulated paths
            print(mc_simulated_prices.head())

            # Debug checks
            print(price_path.shape)  # should print (252,) each time
            print(mc_simulations.shape)  # should print (10000, 252)

            # Calculate the 5th, 50th, and 95th percentiles of the simulated paths
            percentile_5 = mc_simulated_prices.quantile(0.05, axis=1)
            percentile_50 = mc_simulated_prices.quantile(0.5, axis=1)
            percentile_95 = mc_simulated_prices.quantile(0.95, axis=1)

            # Plot the simulation and percentiles together
            plt.figure(figsize=(10, 4))

            # Set color map
            mc_colormap = plt.cm.plasma  # sets to "plasma" color map for variable
            mc_colors = [mc_colormap(i / mc_simulations_num) for i in range(mc_simulations_num)]  # applies color map to iterations

            # Plot all simulated paths
            for i in range(mc_simulations_num):
                plt.plot(mc_simulations[i], color=mc_colors[i], alpha=0.1)

            # Plot the percentiles on top
            plt.plot(percentile_5, label="5th Percentile", color='red', linestyle='--', linewidth=1)
            plt.plot(percentile_50, label="50th Percentile (Median)", color='white', linestyle='--', linewidth=1)
            plt.plot(percentile_95, label="95th Percentile", color='green', linestyle='--', linewidth=1)

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

    # Title to Add the Model Build is in Progress:
    st.markdown("<h2 style='text-align: center;'>Model Build in Progress...</h2>", unsafe_allow_html=True)
    st.header("")
    st.header("")

    # Container for Stock Grades Page:
    sg_c = st.container(height=500, border=False)

    # Add Animations for WIP:
    # CSS for cog wheel animation
    cog_wheel_css = """
    <style>
    /* CSS for cog wheel animation */
    @keyframes rotate {
      from {
        transform: rotate(0deg);
      }
      to {
        transform: rotate(360deg);
      }
    }
    
    .cog-container {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 50%;
    }
    
    .cog {
      width: 200px;
      height: 200px;
      border-radius: 50%;
      border: 5px solid transparent;
      border-top-color: #FF6347; /* Set to Streamlit's red color */
      animation: rotate 1s linear infinite;
    }
    </style>
    """

    # HTML for cog wheel animation
    cog_html = """
    <div class="cog-container">
      <div class="cog"></div>
    </div>
    """

    # Display the animated spinner
    sg_c.markdown(cog_wheel_css + cog_html, unsafe_allow_html=True)





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
