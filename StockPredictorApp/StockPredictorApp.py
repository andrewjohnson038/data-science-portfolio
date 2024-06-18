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
import yahoo_fin.stock_info
from streamlit_option_menu import option_menu
# import yfinance library (used to retrieve financial data from yahoo finance)
import yfinance as yf
# import prophet library (prophet is an open source time series analysis module we will use with plotly to analyze and predict our stock data)
from prophet import Prophet
from prophet.plot import plot_plotly
# import plotly graph_objs module (used to plot our time series data within visuals)
from plotly import graph_objs as go
# import pandas library for dfs
import pandas as pd


# Note: in order to view the app on your local host, you will need run the following code on your terminal: streamlit run [insert location of file here {ex: %/stockpredictorapp.py}]
# Note: this is the streamlit red color code: #FF6347









# //////////////////////////////////////////// SECTION A: Configure Layout for our App ////////////////////////////////////////////////////////////


# (0.1): Set Layout of App, Provide App Version and Provide Title
st.set_page_config(layout='wide', initial_sidebar_state='expanded')  # sets layout to wide; has sidebar expanded at start
st.sidebar.markdown("<div align='center'>App Version: 1.2 &nbsp; <span style='color:#FF6347;'>&#x271A;</span></div>", unsafe_allow_html=True) # adds App version and red medical cross icon with HTML & CSS code; nbsp adds a space
st.sidebar.header('Choose Stock & Forecast Range')  # provide sidebar header

# (0.2): Add a Nav Bar using Streamlit-Option-Menu library

# Create variable for the container:
nav_bar = st.container()

# Within the "navbar" container, add an option menu using Streamlit-Option-Menu library as a navbar:
with nav_bar:
    nav_bar_options = option_menu(
        menu_title=None,  # remove menu title
        options=["Home", "Stock Grades"],  # option names on the menu
        icons=["house", "pencil-square"],  # uses bootstrap icons
        default_index=0,  # sets the first page (Home) as the default
        orientation="horizontal",  # sets the option menu as
    )

# //////////////////////////////////////////// SECTION A: Configure Layout for our App ////////////////////////////////////////////////////////////



















# ////////////////////////////////////////// Section B: Pull in our data and set up datasets for use in app  ////////////////////////////////////////////////////////


# --------------------------------------------------- Data: Set Time Parameters -----------------------------------------------------------------

print("\n Today Date:")
today = datetime.today() # retrieve today's data in sme format as above. date.today() retrieves the current date; apartof datetime module in Python. .strftime("%y-%m-%d") converts the date object to a string with the above format
print(today)

print("\n Yesterday Date:")
yesterday = today - timedelta(days=1) # if we need to write code for the day before today, can use this
print(yesterday)

print("\n 10 Yrs Ago Date:")
start_date = today.replace(year=today.year - 10) # retrieve data starting on this date 10 years ago
print(start_date)

# Get Current Year #
Current_Year = datetime.now().year
print("\n CURRENT Year:")
print(Current_Year)

# Get Current Month #
Current_Month = datetime.now().month
print("\n CURRENT MONTH:")
print(Current_Month)

# Get Current Day #
print("\n CURRENT DAY:")
Current_Day = datetime.now().day
print(Current_Day)

# --------------------------------------------------- Data: Set Time Parameters -----------------------------------------------------------------













# ------------------------------------------ Data: Load Ticker Volume/Pricing/Ratios (Historic & Current) ---------------------------------------

@st.cache_data # caache data to improve load speeds in app
def load_data(ticker): # define our "load data" function. The function "load_data" takes one parameter, "ticker", as a string representing the ticker symbol of the stock.
    stock_data = yf.download(ticker, start=start_date, end=today) # Fetch ticker data from yahoo finance within our date range. returns data in a pandas df (data = df)
    stock_data.reset_index(inplace=True) # this will reset the index to pull the data starting on today's date in the first column

    # Fill in any missing dates with the most recent date's data
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])  # Convert 'Date' column to datetime
    date_range = pd.date_range(start=stock_data['Date'].min(), end=stock_data['Date'].max())  # Create a complete date range within the fetched data
    date_range_df = pd.DataFrame({'Date': date_range})  # Convert date range to DataFrame
    stock_data = pd.merge(date_range_df, stock_data, on='Date', how='left').ffill()  # Merge date range with original DataFrame, filling missing values with the last available data

    return stock_data # return data with the range and ticker params to df "stock_data"


# retrieve nasdaq tickers:
nasdaq_ticker_json_link = 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.json' # create variable for link to github with daily updated tickers from nasdaq
nasdaq_stock_tickers = pd.read_json(nasdaq_ticker_json_link,  typ='series') # read the json file in pandas as a series since there is no column title
nasdaq_stocks = nasdaq_stock_tickers.tolist() # convert set to list

# retrieve nyse tickers:
nyse_ticker_json_link = 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.json'  # create variable for link to github with daily updated tickers from nyse
nyse_stock_tickers = pd.read_json(nyse_ticker_json_link, typ='series')  # read the json file in pandas as a series since there is no column title
nyse_stocks = nyse_stock_tickers.tolist() # convert set to list

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

# Fetch Additional Ticker Metrics/Info
stock_info = yf.Ticker(selected_stock).info  # fetches data based on the ticker selected
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

# Create a DataFrame to store the ratios if needed
stock_ratios = pd.DataFrame({
    'PE Ratio': [pe_ratio],
    'PEG Ratio': [peg_ratio],
    'Price-to-Book': [price_to_book],
    'Debt-to-Equity': [debt_to_equity],
    'Dividend Yield': [dividend_yield],
    'Price-to-Sales': [price_to_sales],
    'ROE': [roe],
    'Quick Ratio': [quick_ratio],
    'Current Ratio': [current_ratio],
    'Industry': [industry],
    'Sector': [sector]
})

print("todays stock metrics")
print(stock_ratios)

# Load the data with the stock list
stock_data = load_data(selected_stock) # loads the selected ticker data (selected_stock)
print('todays stock info')
print(stock_data)

# Provide load context and load ticker data
data_load_state = st.text("loading data...") # displays the following text when loading the data
data_load_state.empty() # changes the load text to done when loaded

# Add function to retreive latest close price for close price field, assign var:
# this will allow us to use this var for calculations later in the app if we don't have a current mkt price
# Define A function to get the latest close price and its date:

@st.cache_data()
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

# Retrieve the Current Stock Price, Assign Var & Add to Sidebar:
Current_Mkt_Price = yahoo_fin.stock_info.get_live_price(selected_stock)  # pull price based on the stock selected

# Create a var for current price that retrieves the last close price if the market price is unavailable:
if Current_Mkt_Price is not None:
    Current_Price = Current_Mkt_Price
else:
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
print(forecast)
# ------------------------------------------------------ Data: Forecast Model (Forecasted Data) --------------------------------------------------------------------


# ////////////////////////////////////////// Section B: Pull in our data and set up datasets for use in app  ////////////////////////////////////////////////////////






















# //////////////////////////////////////////// Sidebar: Add Elements to Sidebar in App ////////////////////////////////////////////////////////////

# Note: Ticker selector was already added in section B as it is being used as a filter for our data
# Note: Company name added to sidebar already in section B






# Write the Current Price & Company Name to Sidebar:
if Current_Price is not None:
    st.sidebar.write(f"{company_name} | ${Current_Price}")
else:
    st.sidebar.write("")











# ----------------------------------------------------- Sidebar: Add KPISs on Change %'s -----------------------------------------------------------

# Add KPI for forecasted price change in 1 year from today:

# Get forecast data from only this month
print("\n f_cm: data on today's month YOY (trend = price at date):")
f_cm = forecast[forecast['ds'].dt.month == Current_Month] # filters stock_data df to this month throughout all years; using Current Month variable from above
print(f_cm)

# Get forecast data from only this day; Note: Need to get the lastest market date since that will be the last available stock pricing if we are unable to use the market price (on days when mkt close):
# Get the last row of the DataFrame
latest_mkt_date = f_cm.iloc[-1]

# Get the date from the last row
latest_mkt_date = latest_mkt_date['ds'].date() # get full date
latest_mktdate_day = latest_mkt_date.day # get just the day (#) of the latest mkt date recorded
print(f" Latest Mkt Date {latest_mkt_date}")
print(f" Latest Mkt Date {latest_mktdate_day}")

print("\n f_cd: data on today's date YOY (trend = price at date):")
f_cd = f_cm[f_cm['ds'].dt.day == latest_mktdate_day] # filters stock_data df to latest mkt date throughout all years
print(f_cd)

# Get only year after this year
print("\n f_fy: Price Predicted in 1 year")
today_year = datetime.now().year

f_fy = f_cd[f_cd['ds'].dt.year > today_year] # filters stock_data df to this day 1 year ahead
print(f_fy)

# Get only today from forecasted data
print("\n f_cy: Price of Stock Today")
today_year = datetime.now().year
f_cy = f_cd[f_cd['ds'].dt.year == today_year] # filters stock_data df to this day throughout all years
print(f_cy[['ds', 'trend']])

# Need to subtract "trend" from next year and this year:
# Extract trend values
print("Price Today", Current_Price)

forecasted_price_next_year = f_fy['trend'].values[0]
print("Price Forecasted in One Year", forecasted_price_next_year)

# Get Forecasted 1 Year $ Difference & Percentage:
trend_difference = forecasted_price_next_year - Current_Price
trend_difference_number = round(trend_difference, 2)    # round the number to two decimal points
trend_difference_percentage = (trend_difference/Current_Price)
trend_difference_percentage = round(trend_difference_percentage * 100)

# Print the result
print("Difference in trend between next year and today (dollars):", trend_difference_number)
print("Difference in trend between next year and today (% return):", trend_difference_percentage)

# Define the CSS for positive and negative trends
positive_icon = '<span style="color: green;">&#9650;</span>'
negative_icon = '<span style="color: red;">&#9660;</span>'
neutral_icon = '<span style="color: gray;">&#8212;</span>'

# Create a trend icon for if the forecasted price is positive or negative
if trend_difference_percentage > 0:
    trend_icon = positive_icon  # Use positive trend icon
elif trend_difference_percentage < 0:
    trend_icon = negative_icon  # Use negative trend icon
else:
    trend_icon = neutral_icon  # Neutral trend icon if the difference is 0

# Give title for 1 year forecasted price:
st.sidebar.subheader("1 Year Price Forecast:")

# Get Forecasted 1 Year Price:
one_yr_forecasted_price = Current_Price + trend_difference_number
print(f"One Year Forecasted Price: {one_yr_forecasted_price}")

# Forecasted Price Difference in 1 year ($ Difference + Current Price & % Difference Concatenated)
Yr_Price_Forecast = ("$" + str(round(one_yr_forecasted_price, 2)) + " | " + str(trend_difference_percentage) + "% " + trend_icon)

# Write 1 Year Forecast Diff to sidebar
if Yr_Price_Forecast is not None:
    # Write the value to the app for today in KPI format if the data is available
    st.sidebar.markdown(Yr_Price_Forecast, unsafe_allow_html=True)
else:
    st.warning(f"1 Year Forecast: Data Not Available")

# Add KPI for YOY change in price from last year to this year from yesterday:

# We already have a variable for the price today, now we must create a variable to get the price a year ago from yesterday:
print("\n p_ly: Price of Stock a Year Ago Today")
today_last_year = today_year-1
p_ly = f_cd[f_cd['ds'].dt.year == today_last_year] # filters stock_data df to this day throughout all years
print(p_ly[['ds', 'trend']])

price_year_ago = p_ly['trend'].values[0] #values[0] grabs the first value

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
st.sidebar.subheader("YOY Price Change:")

# YOY Price Difference in 1 year ($ & % Difference Concatenated)
YOY_Price_Change = ("$" + str(YOY_difference_number) + " | " + str(YOY_difference_percentage) + "% " + trend_icon)

# Write 1 Year Forecast Diff to sidebar
if YOY_Price_Change is not None:
    # Write the value to the app for today in KPI format if the data is available
    st.sidebar.markdown(YOY_Price_Change, unsafe_allow_html=True)
else:
    st.warning(f"YOY Price Change: Data Not Available")

# Add KPI for Avg YOY price change over last 10 years from yesterday:

# Ensure DataFrame is sorted by date
Price_L10yrs = yearly_data.sort_values(by='Date')

# Extract year from the date
Price_L10yrs['Year'] = pd.to_datetime(Price_L10yrs['Date']).dt.year

# Group data by year and calculate average closing price for each year
yearly_avg_close_price = Price_L10yrs.groupby('Year')['Close'].mean()

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
st.sidebar.subheader("AVG YOY Price Change (Last 10 Yrs):")
st.sidebar.markdown(str(yoy_avg_close_price_change) + "% " + trend_icon, unsafe_allow_html=True)


# ----------------------------------------------------- Sidebar: Add KPISs on Change %'s -----------------------------------------------------------












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



# --------------------------------------------- Home Page: Historical/Current Data Visuals --------------------------------------------------------

# assign this section to show in app when the "Home" Icon is selected in the Nav Bar
if nav_bar_options == "Home":

    # (1.1): Show stock current stock details
    NA = "no data available for this stock"  # assign variable so can use in elseif statements when there is no data

    # Add Company Name:
    # Create a container for the company name so we can add a border around it:
    cn_c = st.container(border=True)
    # Check if data is available for the field:
    with cn_c:
        if company_name.strip():  # if the field is not missing (use strip() for strings instead of .empty since it is not an object in the df in this situation)
            # Write the value to the app for today in KPI format if the data is available
            # Custom CSS for styling the kpis below with a translucent grey border
            st.markdown(f"<h3 style='text-align: center;'>{company_name}</h3>", unsafe_allow_html=True)
        else:
            # Write the data is not available for the field if missing:
            st.warning(f"Company Name: Not Available")

    # Create a tab for our historical/current data and one for our forecasted data for the ticker selected
    home_tab1, home_tab2 = st.tabs(["Current/Historical Data", "Forecasted Data"])

    with home_tab1:
        # create variable for a container to put our stock detail section in:
        sh_c = st.container(height = 700, border=True)  # add a fixed height of 800 px and add border to container

        # define columns for the stock_details container (sd_c)
        sh_col1, sh_col2 = sh_c.columns(2)

        # write container to app with columns for our stock current details and history data:
        with sh_c:

            # Within sh_col1, create two columns:
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
                        sh_col1_1.metric(label="Dividend Yield:", value=str(round(today_DY_ratio, 2)) + "%")
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
                # (1.2): Create a Time Series Visual for our Data in column 2 of the sh container:
                Price_History_Tbl_Title = "Price History (Last 10 Years):"
                sh_col2.write(Price_History_Tbl_Title)
                def plot_raw_data(): # define a function for our plotted data (fig stands for figure)
                    fig = go.Figure() # create a plotly graph object.
                    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name='Price', line=dict(color='#0072B2')))
                    fig.layout.update(xaxis_rangeslider_visible=True) # add axis slider to graph to adjust time of visual
                    sh_col2.plotly_chart(fig, use_container_width=True) # writes the graph to app and fixes width to the container width
                plot_raw_data()

                # (1.3): Add Yearly Data Table to column 2 of our sh container:
                # write a table of the past data within the last 10 years on this date
                sh_col2.write("Ticker Price History (Last 10 Years):") # provide title for raw data visual
                sh_col2.dataframe(yearly_data.sort_values(by='Date', ascending=False), hide_index=True)

# --------------------------------------------- Home Page: Historical/Current Data Visuals --------------------------------------------------------





















# --------------------------------------------- Home Page: Add Stock Price Forecasting Visuals --------------------------------------------------------

    # Add forecasted data visuals to our second tab
    with home_tab2:

        # create container for forecast section and columns:
        fs_c = st.container(height = 700, border=True) # puts the data below in a container with a border
        fs_col1, fs_col2 = fs_c.columns(2)  # put 2 columns in our fs_c container

        # Provide title for column 1 & 2 of forecast container:
        with fs_c:
            fs_col1.write("Price Forecast:")
            fs_col2.write("Forecast Components:")

        with fs_c:

            # PLot our forecasted future data in tabular format:
            fs_col1.dataframe(forecast.tail(1), hide_index=True)

            # Plot the forecasted future data using the prophet model within a forecasted visual:
            fig1 = plot_plotly(trained_model, forecast) # plot data on visual (plotly needs the model and forecast to plot)
            fs_col1.plotly_chart(fig1, use_container_width=True, use_container_height=True) # write the chart of the plotly figure (figure 1 = fig1) in app

            # Plot the Prophet forecast components:
            fig2 = trained_model.plot_components(forecast) # write component visuals to label fig2.
            fs_col2.pyplot(fig2) # write figure 2 to the app
            fs_col2.write('Provided above are forecast components within the Prophet model regarding the (1) trend component, (2) seasonality component, and (3) Holiday Effects that describe different aspects of the time series data*') # add a text label for the forecasted components

            # Notes:
            # By calling m.plot_components(forecast), we're instructing prophet to generate a plot showing the components of the forecast. These components include trends, seasonalities, and holiday effect
            # The result of m.plot_components(forecast) is a set of plots visualizing these components


# --------------------------------------------- Home Page: Add Stock Price Forecasting Visuals --------------------------------------------------------



# ///////////////////////////////////////////////////////////// Home Page //////////////////////////////////////////////////////////////////////////




















# ///////////////////////////////////////////////////////////// Stock Grader Page //////////////////////////////////////////////////////////////////////////




if nav_bar_options == "Stock Grades":

    # Title to Add the Model Build is in Progress:
    st.markdown("<h2 style='text-align: center;'>Model Build in Progress...</h2>", unsafe_allow_html=True)
    st.header("")
    st.header("")

    # Container for Stock Grades Page:
    sg_c = st.container(height = 500, border=False)

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
