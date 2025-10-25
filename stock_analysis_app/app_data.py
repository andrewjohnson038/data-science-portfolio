# This file is to hold all datasets / data load functions used within the main and grading model file

# Import python packages
import pandas as pd
import yfinance as yf
import streamlit as st
import numpy as np
import requests
from statsmodels.tsa.holtwinters import SimpleExpSmoothing  # import lib for applying exponential smoothing line
from bs4 import BeautifulSoup  # Import Beautiful Soup for Web Scraping
from prophet import Prophet  # Import Prophet (META Time Series Model for Forecasting)
import boto3  # AWS client
from io import StringIO

# Import app methods
from stock_analysis_app.app_constants import DateVars
from stock_analysis_app.app_constants import ALPHA_VANTAGE_API_KEY
from stock_analysis_app.app_animations import CSSAnimations

from datetime import datetime, timedelta, time
import time as tm

# Instantiate any imported classes here:
dv = DateVars()
animation = CSSAnimations()

# --------------------------------------------------- Data: Create Class for ticker data -----------------------------------------------------------------


# Class for data dfs/variables used across the app
class AppData:

    # set timeline for data caching dependent on market hours
    @staticmethod
    def get_market_cache_ttl():
        now = datetime.now()
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        # Weekend (Saturday=5, Sunday=6)
        if weekday >= 5:
            return 86400  # 24 hours on weekends

        # Weekday - check market hours
        market_open = time(9, 30)   # 9:30 AM EST
        market_close = time(16, 0)  # 4:00 PM EST

        if market_open <= now.time() <= market_close:
            return 600   # 10 minutes during market hours
        else:
            return 3600  # 1 hour after market close

    # Method to retrieve a full list of filtered tickers
    @staticmethod
    @st.cache_data(ttl=get_market_cache_ttl())  # cache data based on market hours
    def filtered_tickers():
        """Fetch and filter NASDAQ and NYSE stock tickers."""
        # Retrieve NASDAQ tickers (pull from open source github repo):
        nasdaq_ticker_json_link = 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.json'
        nasdaq_stock_tickers = pd.read_json(nasdaq_ticker_json_link,
                                            typ='series')  # Read the JSON file into a pandas series
        nasdaq_stocks = nasdaq_stock_tickers.tolist()  # Convert the series to a list

        # Retrieve NYSE tickers (pull from open source github repo):
        nyse_ticker_json_link = 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.json'
        nyse_stock_tickers = pd.read_json(nyse_ticker_json_link,
                                          typ='series')  # Read the JSON file into a pandas series
        nyse_stocks = nyse_stock_tickers.tolist()  # Convert the series to a list

        # Combine NASDAQ and NYSE tickers
        combined_tickers = nasdaq_stocks + nyse_stocks

        # Remove duplicates by converting to a set and then back to a list
        distinct_tickers = list(set(combined_tickers))

        # Remove abnormal tickers containing "^" from the list
        filtered_tickers = [ticker for ticker in distinct_tickers if "^" not in ticker]

        # Sort the tickers alphabetically
        filtered_tickers = sorted(filtered_tickers)

        return filtered_tickers  # Return the filtered and sorted list of tickers

    # load price history function
    @staticmethod
    @st.cache_data(ttl=get_market_cache_ttl())  # cache data based on market hours
    def load_price_hist_data(ticker, start_date=None, end_date=None):

        # Convert string dates to datetime objects if provided
        if isinstance(start_date, str):
            try:
                start_date = pd.to_datetime(start_date)
            except:
                st.error(f"Invalid start date format: {start_date}")
                return pd.DataFrame()

        if isinstance(end_date, str):
            try:
                end_date = pd.to_datetime(end_date)
            except:
                st.error(f"Invalid end date format: {end_date}")
                return pd.DataFrame()

        # Set default end_date as today if not provided
        if end_date is None:
            end_date = datetime.today().date()

        # Set default start_date as 10 years before end_date if not provided
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 10)

        # Ensure dates are properly formatted for yfinance
        start_date_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date
        end_date_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date

        # Download data with string-formatted dates
        stock_price_history_df = yf.download(ticker, start=start_date_str, end=end_date_str, auto_adjust=True)

        if stock_price_history_df.empty:
            st.error(f"No data returned for ticker '{ticker}' in the selected date range.")
            return pd.DataFrame()

        # Reset index to convert Date from index to column
        stock_price_history_df = stock_price_history_df.reset_index()

        # Handle potential MultiIndex columns (happens with adjusted data)
        stock_price_history_df.columns = [col[0] if isinstance(col, tuple) else col for col in stock_price_history_df.columns]

        # Ensure 'Date' is datetime and drop rows with missing dates
        stock_price_history_df['Date'] = pd.to_datetime(stock_price_history_df['Date'], errors='coerce')
        stock_price_history_df = stock_price_history_df.dropna(subset=['Date'])

        if stock_price_history_df.empty:
            st.warning(f"No valid dates found for '{ticker}'.")
            return pd.DataFrame()

        # Validate dates before building the full date range
        min_date = stock_price_history_df['Date'].min()
        max_date = stock_price_history_df['Date'].max()

        if pd.isna(min_date) or pd.isna(max_date):
            st.warning(f"Could not determine valid date range for '{ticker}'.")
            return pd.DataFrame()

        # Create full date range and fill missing dates
        date_range = pd.date_range(start=min_date, end=max_date, freq='B')  # Business days only
        date_range_df = pd.DataFrame({'Date': date_range})

        # Merge with proper datetime format
        stock_price_history_df = pd.merge(date_range_df, stock_price_history_df, on='Date', how='left')

        # Forward fill missing values
        stock_price_history_df = stock_price_history_df.ffill()

        # Calculate price change and percent change
        stock_price_history_df['Price Change'] = stock_price_history_df['Close'].diff().fillna(0)

        # Avoid division by zero
        prev_close = stock_price_history_df['Close'].shift(1)
        stock_price_history_df['Percentage Change'] = (
                                                              stock_price_history_df['Price Change'] / prev_close.replace(0, np.nan)
                                                      ) * 100
        stock_price_history_df['Percentage Change'] = stock_price_history_df['Percentage Change'].fillna(0)

        return stock_price_history_df

    # Method to Retrieve the Latest Close Price and Date as Fields
    @staticmethod
    @st.cache_data(ttl=get_market_cache_ttl())  # cache data based on market hours
    def get_last_close_price_date(stock_data_df):
        # Remove rows where Close is NaN or None
        data_valid = stock_data_df.dropna(subset=['Close'])
        # Sort data by Date in descending order
        data_sorted = data_valid.sort_values('Date', ascending=False)
        # Get the latest valid close price and date
        if not data_sorted.empty:
            latest_close_date = data_sorted.iloc[0]['Date']  # get the most recent date at index 0
            latest_close_price = data_sorted.iloc[0]['Close']  # get the most recent close at index 0
            return latest_close_price, latest_close_date
        else:
            return None, None

    # Method to pull latest date stock metrics from yfinance / alpha vantage
    @staticmethod
    @st.cache_data(ttl=get_market_cache_ttl())  # cache data based on market hours
    def load_curr_stock_metrics(ticker="AAPL"):  # defaults ticker to apple if one isn't provided
        """
        Fetch stock data and handle errors gracefully.
        """

        # Check if data is available
        try:
            stock_info = yf.Ticker(ticker).info

            # If info dict is empty or contains an error key, warn but don't stop
            if not stock_info or "error" in stock_info:
                st.warning(f"No valid data found for ticker `{ticker}`. It may be delisted or unavailable.")
                stock_info = None  # Set to None for downstream logic

        except Exception as e:
            error_message = str(e)

            # Handle rate limit errors more seriously
            if "rate limit" in error_message.lower() or "too many requests" in error_message.lower():
                st.write(" ")
                st.write(" ")
                st.markdown(f"""
                    <div id="error-message">
                        <div style="display: flex; align-items: center;">
                            {animation.warning_animation(3)}
                            <span style="font-size: 18px; font-weight: 600; line-height: 1.5;">
                                Rate limit error while fetching data for `{ticker}`.<br>
                                <code>{error_message}</code>
                            </span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.stop()
            else:
                # Log non-critical errors without stopping the app
                st.warning(f"An error occurred while fetching data for `{ticker}`: {error_message}")
                st.stop()

        # Retrieve additional metrics from Yahoo Finance API
        company_name = stock_info.get('longName', None)

        # Valuation Info
        pe_ratio = stock_info.get('trailingPE', None)
        peg_ratio = stock_info.get('pegRatio', None)

        # Function to manually calculate PEG ratio using PE and EPS growth
        def calculate_peg_manually(ticker):
            """
            Manually calculate PEG ratio if not available from Yahoo Finance
            PEG = PE Ratio / EPS Growth Rate (%)
            """
            try:
                # Create ticker object and get info
                ticker_obj = yf.Ticker(ticker)
                stock_info = ticker_obj.info

                # Get current PE ratio
                pe_ratio

                if not pe_ratio:
                    return None

                # Method 1: Use earnings growth estimate if available
                earnings_growth = stock_info.get('earningsGrowth')
                if earnings_growth and earnings_growth > 0:
                    # Convert to percentage and calculate PEG
                    eps_growth_rate = earnings_growth * 100
                    return pe_ratio / eps_growth_rate

                # Method 2: Calculate from historical financials if growth estimate not available
                try:
                    financials = ticker_obj.financials

                    if 'Basic EPS' in financials.index:
                        eps_data = financials.loc['Basic EPS'].dropna()

                        if len(eps_data) >= 2:
                            # Calculate annual growth rate
                            years = len(eps_data) - 1
                            oldest_eps = eps_data.iloc[-1]  # Oldest data
                            newest_eps = eps_data.iloc[0]   # Most recent data

                            # Calculate compound annual growth rate (CAGR)
                            if oldest_eps > 0:
                                eps_cagr = ((newest_eps / oldest_eps) ** (1/years) - 1) * 100

                                if eps_cagr > 0:  # Only calculate PEG if growth is positive
                                    return pe_ratio / eps_cagr
                except:
                    pass

                # Method 3: Use quarterly EPS data as last resort
                try:
                    quarterly = ticker_obj.quarterly_financials
                    if 'Basic EPS' in quarterly.index:
                        quarterly_eps = quarterly.loc['Basic EPS'].dropna()

                        if len(quarterly_eps) >= 8:  # Need at least 8 quarters
                            # Compare most recent 4 quarters vs previous 4 quarters
                            recent_4q_eps = quarterly_eps.iloc[:4].sum()
                            previous_4q_eps = quarterly_eps.iloc[4:8].sum()

                            if previous_4q_eps > 0:
                                quarterly_growth = ((recent_4q_eps / previous_4q_eps) - 1) * 100

                                if quarterly_growth > 0:
                                    return pe_ratio / quarterly_growth
                except:
                    pass

                return None

            except Exception as e:
                return None

        # Function to get PEG ratio from Alpha Vantage (Usually missing from yfinance)
        def get_peg_from_alpha_vantage(ticker):
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "OVERVIEW",  # Correct function for overview data
                "symbol": ticker,
                "apikey": ALPHA_VANTAGE_API_KEY  # Make sure to replace this with your actual API key
            }

            response = requests.get(url, params=params)
            data = response.json()

            return data.get("PEGRatio", None)

        # If PEG ratio is missing, try to calculate it manually first
        if peg_ratio is None:
            peg_ratio = calculate_peg_manually(ticker)
            if peg_ratio is not None:
                peg_ratio = round(peg_ratio, 2)

        # If PEG ratio is missing, pull it from Alpha Vantage
        if peg_ratio is None:
            peg_ratio = get_peg_from_alpha_vantage(ticker)

        # Collect other financial metrics
        price_to_book = stock_info.get('priceToBook', None)
        price_to_sales = stock_info.get('priceToSalesTrailing12Months', None)
        quick_ratio = stock_info.get('quickRatio', None)
        current_ratio = stock_info.get('currentRatio', None)
        roe = stock_info.get('returnOnEquity', None)
        debt_to_equity = stock_info.get('debtToEquity', None)
        if debt_to_equity is not None:
            debt_to_equity = debt_to_equity / 100
        else:
            debt_to_equity = None

        gross_profit = stock_info.get('grossProfits', None)
        net_income = stock_info.get('netIncomeToCommon', None)
        net_profit_margin = stock_info.get('profitMargins', None)
        yoy_revenue_growth = stock_info.get('revenueGrowth', None)

        # Market Ownership
        shares_outstanding = stock_info.get('sharesOutstanding', None)
        enterprise_value = stock_info.get('enterpriseValue', None)

        # Dividend Info
        dividend_yield = stock_info.get('dividendYield', None)
        dividend_rate = stock_info.get('dividendRate', None)

        # Market Data Info
        previous_close = stock_info.get('previousClose', None)
        regular_market_price = stock_info.get('regularMarketPrice', None)
        regular_market_day_low = stock_info.get('regularMarketDayLow', None)
        regular_market_day_high = stock_info.get('regularMarketDayHigh', None)
        regular_market_volume = stock_info.get('regularMarketVolume', None)
        regular_market_open = stock_info.get('regularMarketOpen', None)
        day_range = stock_info.get('dayRange', None)
        fifty_two_week_range = stock_info.get('fiftyTwoWeekRange', None)
        industry = stock_info.get('industry', None)
        sector = stock_info.get('sector', None)
        market_cap = stock_info.get('marketCap', None)
        beta = stock_info.get('beta', None)
        earnings_date = stock_info.get('earningsDate', None)
        business_summary = stock_info.get('longBusinessSummary', 'No description available')

        # Add fallback logic for previous close if comes in null
        if previous_close is None:
            # Try to use regular market price as fallback
            previous_close = regular_market_price
            if previous_close is None:
                previous_close = stock_info.get('regularMarketOpen', None)

        # ESG Score
        esg_score = stock_info.get('esgScore', None)

        # Analyst Summary
        analyst_recommendation_summary = stock_info.get('recommendationKey', 'No recommendation available')

        # Get the cash flow statement
        cash_flow = yf.Ticker(ticker).cashflow

        # Extract operating cash flow
        try:
            operating_cash_flow_current_year = cash_flow.loc['Operating Cash Flow'].iloc[0]
            operating_cash_flow_previous_year = cash_flow.loc['Operating Cash Flow'].iloc[1]
        except (KeyError, IndexError):
            operating_cash_flow_current_year = 0
            operating_cash_flow_previous_year = 0

        # Calculate YoY Operating Cash Flow Growth
        if operating_cash_flow_previous_year == 0:
            yoy_ocfg_growth = 0
        else:
            yoy_ocfg_growth = ((operating_cash_flow_current_year - operating_cash_flow_previous_year) /
                               operating_cash_flow_previous_year)

        # Create a DataFrame to store the stock ratios
        # Create a DataFrame to store the stock ratios
        stock_ratios_df = pd.DataFrame({
            'Company Name': [company_name],  # The official name of the company

            # Valuation Ratios
            'PE Ratio': [pe_ratio],  # Price-to-Earnings ratio: how much investors are paying per $1 of earnings
            'PEG Ratio': [peg_ratio],  # PE ratio adjusted for growth: lower = better growth for price
            'Price-to-Book': [price_to_book],  # Compares stock price to book value per share

            # Debt & Leverage
            'Debt-to-Equity': [debt_to_equity],  # Measures company leverage: total debt vs shareholder equity

            # Dividends
            'Dividend Yield': [dividend_yield],  # Annual dividend as a percentage of stock price
            'Dividend Rate': [dividend_rate],  # Dollar amount of dividend paid per share annually

            # Profitability & Revenue
            'Price-to-Sales': [price_to_sales],  # Market cap divided by total revenue: valuation vs sales
            'ROE': [roe],  # Return on Equity: net income / shareholder equity, shows profitability
            'Gross Profit': [gross_profit],  # Revenue minus cost of goods sold (COGS)
            'Net Income': [net_income],  # Bottom-line profit after all expenses
            'Net Profit Margin': [net_profit_margin],  # Net income as % of revenue
            'Business Description': [business_summary],  # Description of the business

            # Liquidity
            'Quick Ratio': [quick_ratio],  # Can the company meet short-term obligations without inventory?
            'Current Ratio': [current_ratio],  # Includes inventory: current assets / current liabilities

            # Market Data
            'Market Cap': [market_cap],  # Total market value of all outstanding shares
            'Enterprise Value': [enterprise_value],  # Market cap + debt - cash; true company valuation
            'Beta': [beta],  # Volatility vs market: >1 = more volatile, <1 = less volatile

            # Trading Info
            'Previous Close': [previous_close],  # Price at market close the day before
            'Regular Market Price': [regular_market_price],  # Current or most recent price
            'Regular Market Day Low': [regular_market_day_low],  # Lowest price during today's session
            'Regular Market Day High': [regular_market_day_high],  # Highest price during today's session
            'Regular Market Volume': [regular_market_volume],  # Shares traded today
            'Regular Market Open': [regular_market_open],  # Opening price for the current day
            'Day Range': [day_range],  # Range between today's high and low
            '52-Week Range': [fifty_two_week_range],  # Low and high over the past year

            # Growth Metrics
            'YOY Revenue Growth': [yoy_revenue_growth],  # Year-over-year increase in revenue
            'YOY Operating Cash Flow Growth': [yoy_ocfg_growth],  # Year-over-year change in cash flow from operations

            # ESG / Ratings
            'ESG Score': [esg_score],  # Environmental, Social, Governance score: sustainability metric
            'Analyst Recommendation': [analyst_recommendation_summary],  # Analyst consensus (e.g., Buy, Hold)

            # Company Info
            'Earnings Date': [earnings_date],  # Next expected earnings report date
            'Shares Outstanding': [shares_outstanding],  # Total number of shares issued
            'Industry': [industry],  # Industry classification
            'Sector': [sector],  # Sector classification (broader than industry)
        })

        return stock_ratios_df  # Return the DataFrame containing the stock ratios

    # Method to pull the Sharpe Ratio - this uses the load price history data
    @staticmethod
    @st.cache_data(ttl=get_market_cache_ttl())  # cache data based on market time (5 mins during market hours)
    def calculate_sharpe_ratio(stock_data_df):
        stock_data_df['Returns'] = stock_data_df['Close'].pct_change()  # pandas pct change method

        # Get Risk-Free Rate using Ticker symbol for the 10-year U.S. Treasury Yield (^TNX)
        risk_free_rate_ticker = '^TNX'

        # Fetch data for the 10-year U.S. Treasury Yield
        rfr_data = yf.Ticker(risk_free_rate_ticker)

        # Get the latest closing price (which represents the yield)
        rfr_latest_yield = rfr_data.history(period="7d")['Close'].iloc[-1]  # get latest available price in last 7 days

        # Convert the percentage to a decimal by dividing by 100, and round to 4 decimal places
        rfr_latest_yield = round(rfr_latest_yield / 100, 4)

        # Risk Free Rate (1 Yr current)
        risk_free_rate = rfr_latest_yield / 252  # approximate to two decimal points

        # Calculate excess returns (stock returns - risk-free rate)
        stock_data_df['Excess_Returns'] = stock_data_df['Returns'] - risk_free_rate

        # Calculate the annualized Sharpe ratio (assuming 252 trading days in a year)
        sharpe_ratio = np.sqrt(252) * stock_data_df['Excess_Returns'].mean() / stock_data_df['Excess_Returns'].std()

        # Round to two decimal points
        sharpe_ratio = round(sharpe_ratio, 2)

        return sharpe_ratio

    # Method to fetch analyst ratings from yahoo finance and dconvert to a df.
    @staticmethod
    @st.cache_data(ttl=86400)  # cache data for a day
    def fetch_yf_analyst_price_targets(ticker):
        # Fetch Analyst Price Target Data
        analyst_price_targets = yf.Ticker(ticker).analyst_price_targets  # Returns a dictionary of targets

        # Check if the response is valid (not empty)
        if not analyst_price_targets:
            # Use Streamlit warning if no data is available
            # st.warning(f"No analyst price target data available for {ticker}.")
            return None  # Return None if no data is available

        # Convert the yfinance stock analyst dictionary into a DataFrame
        analyst_price_targets_df = pd.DataFrame(list(analyst_price_targets.items()), columns=["Metric", "Value"])

        return analyst_price_targets_df

    # Method to fetch analyst ratings from yahoo finance and dconvert to a df.
    @staticmethod
    @st.cache_data(ttl=86400)  # cache data for a day
    def fetch_yf_analyst_recommendations(ticker):
        # Fetch analyst recommendations
        analyst_buy_sell_recommendations_df = yf.Ticker(ticker).get_recommendations()  # Comes through as a df when the method is called

        # Check if DataFrame is valid and not empty
        if analyst_buy_sell_recommendations_df is None or analyst_buy_sell_recommendations_df.empty:  # Use .empty to check if an empty df was returned and not just none

            # # Use Streamlit warning if no data is available
            # st.warning(f"No analyst recommendation data available for {ticker}.")
            return None  # Return None if no data is available

        return analyst_buy_sell_recommendations_df

    # Create a df for moving average data to run SMA simulation
    @staticmethod
    @st.cache_data(ttl=86400)  # cache data for a day
    def get_simple_moving_avg_data_df(stock_data_df):
        """Calculate 50-day and 200-day Simple Moving Averages (SMAs)"""
        # Make a copy of the original dataframe to avoid modifying it
        moving_avg_data_df = stock_data_df.copy()

        # Calculate the 50-day and 200-day SMAs
        moving_avg_data_df['50_day_SMA'] = moving_avg_data_df['Close'].rolling(window=50, min_periods=1).mean()
        moving_avg_data_df['200_day_SMA'] = moving_avg_data_df['Close'].rolling(window=200, min_periods=1).mean()

        return moving_avg_data_df

    # Create a df with exponential smoothing for price history smoothed line tracing
    @staticmethod
    @st.cache_data(ttl=3600)  # cache data for an hour
    def apply_exponential_smoothing(stock_data_df, smoothing_level=0.001):  # .2 = default alpha
        """
        Apply Simple Exponential Smoothing to the stock closing prices.

        :param stock_data_df: price history dataset
        :param alpha: Smoothing factor (0 < alpha <= 1) -> closer to 0 = more weight on older data. Will have a greater smoothing effect
        - -> closer to 1 = will have less of a smoothing effect. Will follow the actual close price of the stock more closely, resulting in less of a smoothing effect
        :return: Smoothed values

        heuristic -> straightforward, rule based approach. Takes the first close price and then uses the alpha as the smoothed price moving forward
        """

        # Initialize the model using the hueristic method
        es_model = SimpleExpSmoothing(stock_data_df['Close'], initialization_method="heuristic")

        # Fit the model with a specified smoothing level
        es_model_fit = es_model.fit(smoothing_level=smoothing_level, optimized=False)

        return es_model_fit.fittedvalues  # Returns the smoothed data based on set alpha

    # Create a df to get the latest RSI value by calculating the RSI by day
    @staticmethod
    @st.cache_data(ttl=86400)  # cache data for a day
    def get_latest_rsi(stock_data_df, window=14):
        # Make a copy to avoid modifying the original DataFrame
        rsi_data = stock_data_df.copy()

        # Calculate the difference in price from the previous day
        delta = rsi_data['Close'].diff()

        # Calculate gains and losses
        gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window, min_periods=1).mean()

        # Calculate Relative Strength (RS) and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        latest_rsi_value = rsi.iloc[-1]  # Get the latest RSI value (last value)

        # Return the latest RSI value (last value)
        return latest_rsi_value

    # Create a df to run MACD simulations on
    @staticmethod
    @st.cache_data(ttl=86400)  # cache data for a day
    # Calculate MACD and Signal line to provide insight on short term trend momentum and strength
    def get_macd_df(stock_data_df, fast=12, slow=26, signal=9):

        # Make a copy to avoid modifying the original DataFrame
        macd_data_df = stock_data_df.copy()

        # Calculate the Fast and Slow Exponential Moving Averages
        macd_data_df['EMA_fast'] = macd_data_df['Close'].ewm(span=fast, adjust=False).mean()
        macd_data_df['EMA_slow'] = macd_data_df['Close'].ewm(span=slow, adjust=False).mean()

        # MACD Line = Fast EMA - Slow EMA
        macd_data_df['MACD'] = macd_data_df['EMA_fast'] - macd_data_df['EMA_slow']

        # Signal Line = 9-period EMA of MACD
        macd_data_df['Signal'] = macd_data_df['MACD'].ewm(span=signal, adjust=False).mean()

        # Histogram = MACD - Signal
        macd_data_df['Histogram'] = macd_data_df['MACD'] - macd_data_df['Signal']

        # Filter MACD df to include only the last year of data
        macd_data_df = macd_data_df[macd_data_df['Date'] >= dv.one_yr_ago]

        return macd_data_df

    # Create a df for Monte Carlo Simulation
    @staticmethod
    @st.cache_data(ttl=86400)  # cache data for a day
    def get_monte_carlo_df(stock_data_df, mc_simulations_num=1000, mc_days_num=252):
        # Make a copy to avoid modifying the original DataFrame
        mc_sim_df = stock_data_df.copy()

        # Calculate daily returns (log returns)
        mc_sim_df['Log Return'] = np.log(mc_sim_df['Close'] / mc_sim_df['Close'].shift(1))

        # Drop any missing values
        mc_sim_df = mc_sim_df.dropna()

        # Display the first few log returns
        mc_sim_df[['Close', 'Log Return']].head()

        # Calculate the mean and volatility from the historical data
        mc_mean_return = mc_sim_df['Log Return'].mean()
        mc_volatility = mc_sim_df['Log Return'].std()

        # Simulate future price paths
        mc_simulations = np.zeros((mc_simulations_num, mc_days_num))  # Shape should be (10000, 252)
        last_price = mc_sim_df['Close'].iloc[-1]  # Use the last closing price as the starting point

        # Ensure the simulation loop works without dimension issues
        for i in range(mc_simulations_num):

            # Generate random returns for each day in the simulation
            mc_random_returns = np.random.normal(mc_mean_return, mc_volatility, mc_days_num)

            # Generate the simulated price path using compound returns
            mc_price_path = last_price * np.exp(np.cumsum(mc_random_returns))  # price_path shape will be (252,)

            # Assign the simulated price path to the simulations array
            mc_simulations[i, :] = mc_price_path  # Assign to the i-th row of the simulations array

        # Convert the simulations to a DataFrame for visualization
        mc_simulation_df = pd.DataFrame(mc_simulations.T, columns=[f'Simulation {i+1}' for i in range(mc_simulations_num)])

        return mc_simulation_df

    # Calculate the VaR
    @staticmethod
    @st.cache_data(ttl=86400)  # cache data for a day
    # Define function to calculate Historical VaR at 95% confidence level
    def calculate_historical_VaR(stock_data_df, time_window='daily'):
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
        # Date Index:
        # The date index of the stock_data_df (price history ds pulled from yfinance)
        # Will not be in proper format to use the pandas resample() method.
        # Need to reset to datetime index first

        # create a df for VaR data
        var_data_df = stock_data_df.copy()

        # Ensure the 'Date' column is in datetime format (if it isn't already)
        var_data_df['Date'] = pd.to_datetime(var_data_df['Date'], errors='coerce')

        # Set the 'Date' column as the index of the DataFrame
        var_data_df.set_index('Date', inplace=True)

        # Calculate daily returns using adjusted closing price based on time window
        if time_window == 'daily':
            var_data_df['Return'] = var_data_df['Close'].pct_change()
        elif time_window == 'monthly':
            var_data_df['Return'] = var_data_df['Close'].resample('ME').ffill().pct_change()
        elif time_window == 'yearly':
            var_data_df['Return'] = var_data_df['Close'].resample('YE').ffill().pct_change()
        else:
            raise ValueError("Invalid time window. Choose from 'daily', 'monthly', or 'yearly'.")

        # Sort in ascending order
        sorted_returns = var_data_df['Return'].sort_values()

        # Calculate the 5th percentile (for 95% confidence level)
        hist_VaR_95 = sorted_returns.quantile(0.05)

        # Calculate the VaR in dollars
        current_adj_price = var_data_df['Close'].iloc[-1]
        hist_VaR_95_dollars = hist_VaR_95 * current_adj_price

        # Return both VaR in percentage and dollars
        return hist_VaR_95, hist_VaR_95_dollars

    # Get Industry Averages Method
    @staticmethod
    @st.cache_data(ttl=86400)  # cache data for a day
    def get_industry_averages_df():

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
                    industry_idx = next(
                        (i for i, h in enumerate(headers_row) if 'industry' in h.lower() or 'sector' in h.lower()), 0)
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
                    industry_idx = next(
                        (i for i, h in enumerate(headers_row) if 'industry' in h.lower() or 'sector' in h.lower()), 0)
                    roe_idx = next(
                        (i for i, h in enumerate(headers_row) if 'roe' in h.lower() or 'return on equity' in h.lower()), 1)

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

        return industry_avg_df

    # Get Forecast Data Method (w/ META Prophet Model)
    @staticmethod
    @st.cache_data(ttl=86400)  # cache data for a day
    def get_forecasted_data_df(stock_data_df, forecasted_year_range):
        """
        Note: (For Our Time Series Analysis, We will use the Prophet Time series Model from META)

        Parameters:
        - stock_data_df: pandas DataFrame with historical stock data.
        - time_window: year range for the time series model (In main add year range slider to make dynamic).

        Returns:
        - Data Frame with Forecasted Data
        """
        # Note: (For Our Time Series Analysis, We will use the Prophet Time series Model from META)

        # Make a copy to avoid modifying the original DataFrame
        df_train = stock_data_df.copy()

        # Create a DataFrame for training our time-series model:
        df_train = df_train[['Date', 'Close']]  # create a df (df_train) to train our data set (using date and close price)
        df_train = df_train.rename(columns={"Date": "ds",
                                            "Close": "y"})  # create a dictionary to rename the columns (must rename columns for META Prophet to read data [in documentation]. Documentation Link: https://facebook.github.io/prophet/docs/quick_start.html#python-api)

        # Function to train the model and generate forecasts
        def train_model(df_train):
            # Fit Prophet model
            m = Prophet()
            m.fit(df_train)

            # Return the trained model
            return m

        # Assign Variable to trained model data (the cached function)
        trained_model = train_model(df_train)

        # Create an interactive year range slider and set our forecast range period:
        forecasted_year_range = forecasted_year_range

        # Period = total days of the year range
        period = forecasted_year_range * 365

        # Create df with future periods
        future = trained_model.make_future_dataframe(
            periods=period)

        # Forecast future price predictions with trained model into df
        forecast = trained_model.predict(
            future)

        # Reorder columns with yhat as the second column
        forecast_df = forecast[['ds', 'yhat'] + [col for col in forecast.columns if col not in ['ds', 'yhat']]]

        # Apply the lower bound to the forecasted prices, ensuring they don't go below 0
        forecast_df['yhat'] = forecast['yhat'].apply(lambda x: max(0.01, x))  # Set all negative values to 1 cent

        return trained_model, forecast_df

    @staticmethod
    # Function to Load CSV from S3 using credentials
    def load_csv_from_s3(bucket_name, csv_path):
        """
        Load the CSV file from S3 into a pandas DataFrame.

        Parameters:
        bucket_name (str): The name of the S3 bucket.
        file_key (str): The S3 file key (path).

        Returns:
        pd.DataFrame: The loaded DataFrame.
        """
        # Retrieve credentials from Streamlit secrets
        aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
        aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
        aws_region = st.secrets["AWS_REGION"]

        # Create an S3 client using the credentials from secrets
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )

        # Get the object from S3
        obj = s3_client.get_object(Bucket=bucket_name, Key=csv_path)
        data = obj['Body'].read().decode('utf-8')  # Read the content of the CSV
        df = pd.read_csv(StringIO(data))  # Convert the content into a pandas DataFrame
        return df

    # method to upload ticker to watchlist csv file
    @staticmethod
    def upsert_watchlist_to_s3_csv(ticker: str, industry: str, current_price: float, bucket_name, csv_path):
        """
        Upsert a ticker to watchlist CSV file in S3.
        If the ticker is not in the CSV, append it with current price and timestamp.

        Args:
            ticker (str): Stock ticker symbol
            current_price (float): Current market price of the stock
            bucket_name (str): S3 bucket name
            csv_path (str): Path to CSV file in S3
            industry (str): ticker industry
        """
        # Load AWS credentials from Streamlit secrets
        aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
        aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
        aws_region = st.secrets["AWS_REGION"]

        # Create S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

        # for success / fail message
        message_container = st.empty()

        # Try to load the existing CSV
        try:
            obj = s3.get_object(Bucket=bucket_name, Key=csv_path)
            df = pd.read_csv(obj["Body"])
        except s3.exceptions.NoSuchKey:
            # File doesn't exist — create a new DataFrame with proper columns
            df = pd.DataFrame(columns=["Ticker", "Price_When_Added", "Date_Added"])

        # Append if not already present
        if ticker not in df["Ticker"].values:
            # Get current date (today's date)
            current_date = pd.Timestamp.now().strftime("%Y-%m-%d")

            # Create new row with ticker, price, and timestamp
            new_row = {
                "Ticker": ticker,
                "Industry": industry,
                "Price_When_Added": current_price,
                "Date_Added": current_date
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # Convert to CSV and upload
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            s3.put_object(Bucket=bucket_name, Key=csv_path, Body=csv_buffer.getvalue())

            # Show success message temporarily
            message_container.success(f"Ticker '{ticker}' added to watchlist at ${current_price:.2f}.")
        else:
            message_container.info(f"Ticker '{ticker}' is already in the watchlist.")

        # Wait 1 second, then clear the message
        tm.sleep(2)
        message_container.empty()

    # method to upload ticker to portfolio list csv file
    @staticmethod
    def upsert_ticker_to_s3_portfolio_csv(ticker: str, industry: str, current_price: float, amount: float, bucket_name, csv_path):
        """
        Upsert a ticker to portfolio list CSV file in S3.
        If the ticker is not in the CSV, append it with current price and timestamp.

        Args:
            ticker (str): Stock ticker symbol
            current_price (float): Current market price of the stock
            amount (float): amount of shares
            bucket_name (str): S3 bucket name
            csv_path (str): Path to CSV file in S3
            industry (str): ticker industry
        """
        # Load AWS credentials from Streamlit secrets
        aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
        aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
        aws_region = st.secrets["AWS_REGION"]

        # Create S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

        # for success / fail message
        message_container = st.empty()

        # Try to load the existing CSV
        try:
            obj = s3.get_object(Bucket=bucket_name, Key=csv_path)
            df = pd.read_csv(obj["Body"])
        except s3.exceptions.NoSuchKey:
            # File doesn't exist — create a new DataFrame with proper columns
            df = pd.DataFrame(columns=["Ticker", "Price_When_Added", "Date_Added"])

        # Append if not already present
        if ticker not in df["Ticker"].values:
            # Get current date (today's date)
            current_date = pd.Timestamp.now().strftime("%Y-%m-%d")

            # Create new row with ticker, price, and timestamp
            new_row = {
                "Ticker": ticker,
                "Industry": industry,
                "Price_When_Added": current_price,
                "Amount": amount,
                "Date_Added": current_date
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # Convert to CSV and upload
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            s3.put_object(Bucket=bucket_name, Key=csv_path, Body=csv_buffer.getvalue())

            # Show success message temporarily
            message_container.success(f"{ticker} added to portfolio with {amount} shares.")
        else:
            # Ticker exists - add to existing amount
            ticker_index = df[df["Ticker"] == ticker].index[0]
            current_amount = df.loc[ticker_index, "Amount"]
            new_total_amount = current_amount + amount
            df.loc[ticker_index, "Amount"] = new_total_amount

            # Convert to CSV and upload
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            s3.put_object(Bucket=bucket_name, Key=csv_path, Body=csv_buffer.getvalue())  # upload new amount

            # Show success message with added amount
            message_container.success(f"{amount} more shares added to {ticker} holdings.")

        # Wait 1 second, then clear the message
        tm.sleep(2)
        message_container.empty()

    @staticmethod
    def remove_ticker_from_csv(ticker: str, bucket_name, csv_path):
        """
        Remove a ticker from CSV stored in S3.

        Args:
            ticker (str): Stock ticker to remove.
            bucket_name (str): Name of the S3 bucket.
            csv_path (str): Path to the watchlist CSV in the S3 bucket.
        """

        # Load AWS credentials from Streamlit secrets
        aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
        aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
        aws_region = st.secrets["AWS_REGION"]

        # Create S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

        message_container = st.empty()

        try:
            # Load existing CSV from S3
            obj = s3.get_object(Bucket=bucket_name, Key=csv_path)
            df = pd.read_csv(obj["Body"])

            # Check if ticker is in the watchlist
            if ticker in df["Ticker"].values:
                df = df[df["Ticker"] != ticker]  # Remove the ticker row

                # Upload updated CSV back to S3
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                s3.put_object(Bucket=bucket_name, Key=csv_path, Body=csv_buffer.getvalue())

                message_container.error(f"Ticker '{ticker}' removed from the watchlist.")
            else:
                message_container.info(f"Ticker '{ticker}' not found in the watchlist.")

        except s3.exceptions.NoSuchKey:
            message_container.warning("Watchlist file not found in S3.")
        except Exception as e:
            message_container.error(f"Error removing ticker: {str(e)}")

        # Wait 2 second, then clear the message
        time.sleep(2)
        message_container.empty()


# ---- TEST BLOCK ----
if __name__ == "__main__":

    # Test the load_price_hist_data method and use for other tests using price history dfs
    print("\nTesting load_price_hist_data method...")
    price_history = AppData.load_price_hist_data('AAPL', '2023-01-01', '2023-02-01')
    print(f"Price history for AAPL:\n{price_history.head()}")

    # Test the filtered_tickers method
    print("Testing filtered_tickers method...")
    tickers = AppData.filtered_tickers()
    print(f"Filtered tickers: {tickers[:10]}")  # Print first 10 tickers

    # Test the last_close_price_date method
    print("Testing last_close_price_date method...")
    latest_close_price, latest_close_date = AppData.get_last_close_price_date(price_history)
    print(f"Latest Close Price: {latest_close_price}, Latest Close Date: {latest_close_date}")

    # Test the load_curr_stock_metrics method
    print("\nTesting load_curr_stock_metrics method...")
    stock_metrics = AppData.load_curr_stock_metrics('AAPL')  # Default to AAPL if no ticker is provided
    print(f"Stock metrics for AAPL:\n{stock_metrics}")
    for column in stock_metrics.columns:
        values = stock_metrics[column].tolist()
        print(f"{column}: {values}")

    # Test the calculate_sharpe_ratio method
    print("\nTesting calculate_sharpe_ratio method...")
    df = pd.DataFrame(price_history)
    sharpe_ratio = AppData.calculate_sharpe_ratio(df)
    print(f"Calculated Sharpe Ratio: {sharpe_ratio}")

    # Test the fetch_yf_analyst_price_targets method
    print("\nTesting fetch_analyst_data method...")
    analyst_targets_test_df = AppData.fetch_yf_analyst_price_targets('AAPL')  # Using AAPL ticker
    if analyst_targets_test_df is not None:
        print(f"Analyst data for AAPL:\n{analyst_targets_test_df.head()}")
    else:
        print("No analyst data available.")

    # Test the fetch_yf_analyst_recommendations method
    print("\nTesting fetch_analyst_data method...")
    analyst_rec_test_df = AppData.fetch_yf_analyst_recommendations('AAPL')  # Using AAPL ticker
    if analyst_rec_test_df is not None:
        print(f"Analyst data for AAPL:\n{analyst_rec_test_df.head()}")
    else:
        print("No analyst data available.")

    # Test the Moving Average Data Set
    print("\nTesting get moving average ds method...")
    moving_average_test_df = AppData.get_simple_moving_avg_data_df(price_history)
    print(moving_average_test_df.columns)
    print(moving_average_test_df[['Date', 'Close', '50_day_SMA', '200_day_SMA']].tail())  # Check last couple rows

    # Test Monte Carlo Simulation DS and Vars
    print("\nTesting get Monte Carlo ds method...")
    # Set parameters for simulation
    mc_test_df = AppData.get_monte_carlo_df(price_history)
    print(mc_test_df.head())  # Check a sample of simulated paths
    print(mc_test_df.shape)  # should print (10000 = # of sims, 252 = # of days)

    # Test retrieving VaR variables
    print("\nTesting calculating the VaR...")
    var_percent, var_dollars = AppData.calculate_historical_VaR(price_history, time_window='daily')
    # Display the results
    print(f"\nHistorical VaR (95% confidence):")
    print(f"- As Percentage: {var_percent:.2%}")
    print(f"- As Dollar Amount: ${var_dollars:,.2f}")

    # Test the get industry averages method
    print("\nTesting get industry averages method...")
    industry_avg_df = AppData.get_industry_averages_df()
    print(industry_avg_df.columns)
    print(industry_avg_df)
    print(industry_avg_df.to_string())

    # Test the get forecasted data method
    print("\nTesting get forecasted data method...")
    forecast_range_test = 5  # test for 5 year range
    trained_model_test, forecast_df_test = AppData.get_forecasted_data_df(price_history, forecast_range_test)
    # Print the trained model
    print("\nTrained Model")
    print("Model:", trained_model_test)  # Object info
    print("Seasonalities:", trained_model_test.seasonalities)  # See seasonality
    print("Changepoints:", trained_model_test.changepoints)  # Access change points
    print("Params:", trained_model_test.params)  # See model parameters
    # Print the forecast_df
    print("\nForecast Data DF...")
    print(forecast_df_test.columns)
    print(forecast_df_test.head())
