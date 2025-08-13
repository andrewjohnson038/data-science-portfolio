# Import libraries:
import streamlit as st  # Import streamlit (app Framework)
from prophet.plot import plot_plotly  # Plotly package for plotting Prophet Model
from plotly import graph_objs as go  # Import plotly for time series visuals
import pandas as pd  # Import Pandas for df / data wrangling
import matplotlib.pyplot as plt  # Import MatPlotLib for Monte Carlo sim
import seaborn as sns

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app methods
from stock_analysis_app.app_constants import DateVars
from stock_analysis_app.app_data import AppData
from stock_analysis_app.app_animations import CSSAnimations
from stock_analysis_app.app_stock_grading_model import StockGradeModel
from stock_analysis_app.app_constants import ExtraComponents


# ---------------- SET UP DATA VARS/MODULES ----------------
# Instantiate any imported classes here:
dv = DateVars()
animation = CSSAnimations()
data = AppData()
model = StockGradeModel()
ec = ExtraComponents()

# Global Data Vars
st_dt = dv.start_date
end_dt = dv.today


# --------- additional data loaders
def get_stock_data(selected_stock: str, st_dt, end_dt):
    return data.load_price_hist_data(selected_stock, st_dt, end_dt)


def get_stock_metrics(selected_stock: str):
    return data.load_curr_stock_metrics(selected_stock)


def get_analyst_data(selected_stock: str):
    return data.fetch_yf_analyst_price_targets(selected_stock), data.fetch_yf_analyst_recommendations(selected_stock)


def get_forecast_data(price_history_df, forecast_years):
    return data.get_forecasted_data_df(price_history_df, forecast_years)


# ---------------- SESSION STATE: SET TICKER DD ----------------
# Initialize session state for selected ticker
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None

# Create Var for filtered list of tickers that will be used throughout the app
ticker_list = data.filtered_tickers()

# Set default index
if st.session_state.current_ticker and st.session_state.current_ticker in ticker_list:
    default_index = ticker_list.index(st.session_state.current_ticker)
else:
    default_index = 0

# Create a dropdown box - use key to persist selection
selected_stock = st.sidebar.selectbox(
    "Select Stock:",
    ticker_list,
    index=default_index,
    key="ticker_dropdown"
)

# Update session state to keep selected stock
st.session_state.current_ticker = selected_stock

# ---------------- SESSION STATE: SET WatchList & Portfolio Buttons ----------------

# ---------------- FETCH SHARED STOCK DATA ----------------
# (Only need to call once if same function used)
selected_stock_metrics_df = get_stock_metrics(selected_stock)

# Extract shared values
stock_price = selected_stock_metrics_df['Regular Market Price'].values[0]
stock_industry = selected_stock_metrics_df['Industry'].values[0]

# ---------------- BUTTONS IN COLUMNS ----------------

add_con = st.container()

with add_con:
    left_col, right_col = st.columns([4, 4, 4, 4, 4, 4, 4, 4, 4])

    with left_col:
        # Add to Watch List
        if st.button("Add to Watchlist"):
            data.upsert_watchlist_to_s3_csv(
                selected_stock,
                stock_industry,
                stock_price,
                'stock-ticker-data-bucket',
                'ticker_watchlist.csv'
            )
            st.session_state.watchlist_updated = True

    with right_col:
        # Show "Add to Portfolio" button only if not already in input mode
        if not st.session_state.get("show_portfolio_list_amount_input", False):
            if st.button("Add to Portfolio"):
                st.session_state.show_portfolio_list_amount_input = True

        # Show amount input and confirm button only if input mode is active
        if st.session_state.get("show_portfolio_list_amount_input", False):
            input_col, confirm_col = st.columns([1, 1])

            with input_col:
                amount = st.number_input(
                    "Shares",
                    min_value=0,
                    step=1,
                    key="portfolio_list_amount"
                )

            with confirm_col:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Add"):
                    if amount > 0:  # Only proceed if amount is valid
                        # Store the action to be executed outside the column
                        st.session_state.execute_portfolio_add = True
                        st.session_state.portfolio_add_amount = amount

        # Execute the portfolio addition outside of the columns for full width display
        if st.session_state.get("execute_portfolio_add", False):
            data.upsert_ticker_to_s3_portfolio_csv(
                selected_stock,
                stock_industry,
                stock_price,
                st.session_state.portfolio_add_amount,
                'stock-ticker-data-bucket',
                'portfolio_ticker_list.csv'
            )

            # Clean up session state
            st.session_state.show_portfolio_list_amount_input = False
            st.session_state.portfolio_updated = True
            st.session_state.execute_portfolio_add = False
            if 'portfolio_add_amount' in st.session_state:
                del st.session_state.portfolio_add_amount

            st.rerun()


# ---------------- SIDEBAR CONTENT: Forecast Slider ----------------
# Create an interactive year range slider and set our forecast range period:
forecasted_year_range_slider = st.sidebar.slider("Choose a Forecast Range (Years):", 1,
                                                 10)  # creates a slider for forecast years. 1 and 10 are the year range


# ---------------- SIDEBAR CONTENT: GET NOTES ----------------
ec.get_sidebar_notes()


# ---------------- PAGE CONTENT: RENDER MAIN SCREEN ----------------
# wrap home page render based on ticker selected
def render_home_page_data(selected_stock: str):

    # ------------------- Data: Load Ticker Volume/Pricing/Ratios (Historic & Current) ------------------------

    # Load the stock data based on the ticker selected in front end
    selected_stock_price_history_df = get_stock_data(selected_stock, st_dt, end_dt)

    # Fetch stock data for the selected stock
    selected_stock_metrics_df = get_stock_metrics(selected_stock)

    # ------- Create Selected Stock Metrics as Vars to reference in app
    # Add all data elements from the stock metrics df as their own string vars:
    selected_stock_company_name = selected_stock_metrics_df['Company Name'].values[0]  # str
    selected_stock_pe_ratio = selected_stock_metrics_df['PE Ratio'].values[0]  # float or None
    selected_stock_peg_ratio = selected_stock_metrics_df['PEG Ratio'].values[
        0]  # float or str (if pulled from Alpha Vantage)
    selected_stock_price_to_book = selected_stock_metrics_df['Price-to-Book'].values[0]  # float or None
    selected_stock_price_to_sales = selected_stock_metrics_df['Price-to-Sales'].values[0]  # float or None
    selected_stock_quick_ratio = selected_stock_metrics_df['Quick Ratio'].values[0]  # float or None
    selected_stock_current_ratio = selected_stock_metrics_df['Current Ratio'].values[0]  # float or None
    selected_stock_roe = selected_stock_metrics_df['ROE'].values[0]  # float (as decimal, e.g., 0.23 = 23%) or None
    selected_stock_debt_to_equity = selected_stock_metrics_df['Debt-to-Equity'].values[0]  # float or None
    selected_stock_gross_profit = selected_stock_metrics_df['Gross Profit'].values[0]  # int (USD) or None
    selected_stock_net_income = selected_stock_metrics_df['Net Income'].values[0]  # int (USD) or None
    selected_stock_net_profit_margin = selected_stock_metrics_df['Net Profit Margin'].values[
        0]  # float (as decimal) or None
    selected_stock_yoy_revenue_growth = selected_stock_metrics_df['YOY Revenue Growth'].values[
        0]  # float (as decimal) or None
    selected_stock_yoy_ocfg_growth = selected_stock_metrics_df['YOY Operating Cash Flow Growth'].values[
        0]  # float (as decimal) or None
    selected_stock_shares_outstanding = selected_stock_metrics_df['Shares Outstanding'].values[0]  # int or None
    selected_stock_enterprise_value = selected_stock_metrics_df['Enterprise Value'].values[0]  # int (USD) or None
    selected_stock_market_cap = selected_stock_metrics_df['Market Cap'].values[0]  # int (USD) or None
    selected_stock_dividend_yield = selected_stock_metrics_df['Dividend Yield'].values[0]  # float (as decimal) or None
    selected_stock_dividend_rate = selected_stock_metrics_df['Dividend Rate'].values[0]  # float (USD per share) or None
    selected_stock_previous_close = selected_stock_metrics_df['Previous Close'].values[0]  # float
    selected_stock_regular_market_price = selected_stock_metrics_df['Regular Market Price'].values[0]  # float
    selected_stock_regular_market_day_low = selected_stock_metrics_df['Regular Market Day Low'].values[0]  # float
    selected_stock_regular_market_day_high = selected_stock_metrics_df['Regular Market Day High'].values[0]  # float
    selected_stock_regular_market_volume = selected_stock_metrics_df['Regular Market Volume'].values[0]  # int
    selected_stock_regular_market_open = selected_stock_metrics_df['Regular Market Open'].values[0]  # float
    selected_stock_day_range = selected_stock_metrics_df['Day Range'].values[0]  # str (e.g., "123.45 - 129.56")
    selected_stock_fifty_two_week_range = selected_stock_metrics_df['52-Week Range'].values[
        0]  # str (e.g., "88.00 - 180.00")
    selected_stock_esg_score = selected_stock_metrics_df['ESG Score'].values[0]  # float or None
    selected_stock_analyst_recommendation_summary = selected_stock_metrics_df['Analyst Recommendation'].values[0]  # str
    selected_stock_earnings_date = selected_stock_metrics_df['Earnings Date'].values[
        0]  # list or pandas.Timestamp (yfinance often returns a list of earnings dates)
    selected_stock_industry = selected_stock_metrics_df['Industry'].values[0]  # str
    selected_stock_sector = selected_stock_metrics_df['Sector'].values[0]  # str
    selected_stock_beta = selected_stock_metrics_df['Beta'].values[0]  # float or None
    selected_stock_business_description = selected_stock_metrics_df['Business Description'].values[0]  # string

    # Provide load context message while data is being fetched
    data_load_state = st.text("loading data...")  # Displays the following text when loading the data
    data_load_state.empty()  # Clears the loading message once the python interpreter hits it (once the df above is loaded)

    # Create Variables for Last Close Price / Last Close Date for the selected stock:
    last_close_price, last_close_date = data.get_last_close_price_date(selected_stock_price_history_df)

    # Remove the seconds time stamp from the date (the 00:00:00 formatted stamp)
    last_close_date = last_close_date.strftime('%Y-%m-%d')

    # Create a var for current price that retrieves the last close price if the market price is unavailable:
    last_close_price = last_close_price

    # Round last close price to 2 Decimal Points:
    last_close_price = round(last_close_price, 2)

    # Calculate Sharpe Ratio for the selected ticker
    selected_stock_sharpe_ratio = data.calculate_sharpe_ratio(selected_stock_price_history_df)

    # Fetch Analyst Price Target Data for the selected ticker from yf
    selected_stock_analyst_targets_df, selected_stock_analyst_recommendations_df = get_analyst_data(selected_stock)

    # For Benchmarking Purposes, create a separate DF for SPY
    SPY_data = get_stock_data("SPY", st_dt, end_dt)  # loads the selected ticker data (selected_stock)

    # -------------------------------- Data: Add a yearly data dataframe to capture price by year on today's date over last 10 years ---------------------------------------

    # Create a copy of selected_stock_price_history_df
    selected_stock_price_history_df_copy = selected_stock_price_history_df.copy()

    # Ensure 'Date' column is datetime
    selected_stock_price_history_df_copy['Date'] = pd.to_datetime(selected_stock_price_history_df_copy['Date'])

    # Filter just the current month across all years
    ct_month_yrly_data = selected_stock_price_history_df_copy[
        selected_stock_price_history_df_copy['Date'].dt.month == dv.current_mth
        ]

    # Build yearly snapshot by going from current year backwards (e.g., 2025 to 2015)
    year_range = list(range(dv.current_yr, dv.current_yr - 10, -1))
    yearly_data = pd.DataFrame()

    for year in year_range:
        year_subset = ct_month_yrly_data[ct_month_yrly_data['Date'].dt.year == year]

        if year_subset.empty:
            continue

        try:
            target_date = pd.Timestamp(year=year, month=dv.current_mth, day=dv.current_day)
        except ValueError:
            continue  # skip Feb 30, etc.

        # Get the closest date <= target
        valid_rows = year_subset[year_subset['Date'] <= target_date]

        if valid_rows.empty:
            continue  # No earlier date in that month for that year

        closest_row = valid_rows.sort_values('Date', ascending=False).iloc[0]
        yearly_data = pd.concat([yearly_data, pd.DataFrame([closest_row])])

    # Reset and sort by date ascending before adding trends
    yearly_data = yearly_data.sort_values('Date').reset_index(drop=True)
    yearly_data['Trend'] = ''

    for i in range(1, len(yearly_data)):
        current_close = yearly_data.at[i, 'Close']
        previous_close = yearly_data.at[i - 1, 'Close']

        if current_close > previous_close:
            yearly_data.at[i, 'Trend'] = '↑'
        elif current_close < previous_close:
            yearly_data.at[i, 'Trend'] = '↓'
        else:
            yearly_data.at[i, 'Trend'] = '■'

    if not yearly_data.empty:
        yearly_data.at[0, 'Trend'] = '■'

    # Change the Price Change and percentage change columns to change from day to day to year over year
    # Drop columns 'Price Change' and 'Percentage Change' safely
    yearly_data = yearly_data.drop(columns=['Price Change', 'Percentage Change'], errors='ignore')

    # Sort by Date (ascending), if not already sorted
    yearly_data = yearly_data.sort_values('Date').reset_index(drop=True)

    # Add columns for Year-over-Year (YOY) price change and YOY % change
    yearly_data['Price Change'] = yearly_data['Close'].diff()  # Price difference between current and previous year
    yearly_data['Percentage Change'] = (yearly_data['Price Change'] / yearly_data['Close'].shift(
        1)) * 100  # Percentage change

    # Optionally, you can round the YOY values to make it more readable
    yearly_data['Price Change'] = yearly_data['Price Change'].round(2)
    yearly_data['Percentage Change'] = yearly_data['Percentage Change'].round(2)

    # The final dataframe now has the YOY changes calculated and added as the last columns

    # Check if we have at least 2 years of data
    unique_years = yearly_data['Date'].dt.year.nunique()
    if unique_years < 2:
        st.error(
            "This ticker doesn't have enough historical data or metrics available to run analysis 😕\n\nPlease try another ticker.")
        st.markdown(animation.warning_animation(2), unsafe_allow_html=True)
        st.stop()

    # -------------------------------- Data: Calculate and Create Variables for Simulations / Risk Models ---------------------------------------

    # ---------- Gather SMA - Data Set / Vars For Selected Stock
    # Retrieve Data Set for SMA
    selected_stock_moving_average_data = data.get_simple_moving_avg_data_df(selected_stock_price_history_df)

    # Calculate Simple Moving Average Variables
    selected_stock_sma_price = selected_stock_moving_average_data['Close'].iloc[-1]  # Get the latest closing price
    selected_stock_sma_50 = selected_stock_moving_average_data['50_day_SMA'].iloc[-1]  # Latest 50-day SMA
    selected_stock_sma_200 = selected_stock_moving_average_data['200_day_SMA'].iloc[-1]  # Latest 200-day SMA

    # Calculate the difference between 50-day and 200-day SMA
    selected_stock_sma_price_difference = selected_stock_sma_50 - selected_stock_sma_200

    # Calculate the percentage difference between the 50-day and 200-day SMA
    selected_stock_sma_percentage_difference = (selected_stock_sma_price_difference / selected_stock_sma_200) * 100

    # ---------- Gather Monte Carlo Simulation Data / Vars for selected stock
    selected_stock_mc_sim_df = data.get_monte_carlo_df(selected_stock_price_history_df)

    # Calculate the 5th, 50th, and 95th percentiles of the simulated paths
    mc_percentile_5 = selected_stock_mc_sim_df.quantile(0.05, axis=1)
    mc_percentile_50 = selected_stock_mc_sim_df.quantile(0.5, axis=1)
    mc_percentile_95 = selected_stock_mc_sim_df.quantile(0.95, axis=1)

    # ---------- Gather Value at Risk (VAR) Data / Vars for selected stock
    # Calculate hist VaR for daily, monthly, and yearly
    selected_stock_hist_daily_VaR_95, selected_stock_hist_daily_VaR_95_dollars = AppData.calculate_historical_VaR(
        selected_stock_price_history_df,
        time_window='daily')
    selected_stock_hist_monthly_VaR_95, selected_stock_hist_monthly_VaR_95_dollars = AppData.calculate_historical_VaR(
        selected_stock_price_history_df,
        time_window='monthly')
    selected_stock_hist_yearly_VaR_95, selected_stock_hist_yearly_VaR_95_dollars = AppData.calculate_historical_VaR(
        selected_stock_price_history_df,
        time_window='yearly')
    # ---------------- Get latest RSI Score for selected stock
    selected_stock_rsi_score = data.get_latest_rsi(selected_stock_price_history_df)

    # ---------------- Get MACD Data for selected stock
    selected_stock_macd_df = data.get_macd_df(selected_stock_price_history_df)

    # ---------------- Get Industry Averages
    industry_avg_df = data.get_industry_averages_df()

    # ---------------- Get Forecasted Data for selected stock
    # Create Forecasted Year Range Var
    selected_stock_forecasted_year_range = forecasted_year_range_slider  # will use the app slider to work dynamically

    # Call forecasted data df method to bring in forecasted data
    selected_stock_trained_model, selected_stock_forecasted_df = get_forecast_data(
        selected_stock_price_history_df, selected_stock_forecasted_year_range)

    # ------------------------------------------------------ Data: Custom Stock Grade Model --------------------------------------------------------------------

    # Calculate grades for the selected stock
    selected_stock_score, selected_stock_grade, selected_stock_grade_color_background, selected_stock_grade_color_outline, selected_stock_score_details = model.calculate_grades(
        selected_stock, selected_stock_metrics_df, selected_stock_forecasted_df, industry_avg_df,
        selected_stock_mc_sim_df, selected_stock_sharpe_ratio,
        selected_stock_hist_yearly_VaR_95, selected_stock_rsi_score, selected_stock_sma_percentage_difference)

    # ///////////////////////////////////////////////////////////// Home Tab //////////////////////////////////////////////////////////////////////////

    NA = "no data available for this stock"  # assign variable so can use in elseif statements when there is no data

    if selected_stock_score is not None and selected_stock_company_name and selected_stock_company_name.strip():
        st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <h1 style="font-size: 32px; margin: 0;">{selected_stock_company_name}</h1>
                <div style="background-color:{selected_stock_grade_color_background}; color:white; font-size:20px; font-weight:bold; padding:10px 20px; border-radius:15px; border: 1px solid {selected_stock_grade_color_outline};">
                    Model Grade: {selected_stock_grade}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Company Name or Grade: Not Available")

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
            Current_Price_kpi = ("$" + str(round(selected_stock_regular_market_price, 2)))

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
            def get_closest_price_before_date(df, year, target_month, target_day):
                try:
                    target_date = pd.Timestamp(year=year, month=target_month, day=target_day)
                except ValueError:
                    return None  # Invalid date like Feb 30

                year_data = df[df['Date'].dt.year == year]
                year_data = year_data[year_data['Date'] <= target_date]

                if not year_data.empty:
                    closest_row = year_data.sort_values('Date', ascending=False).iloc[0]
                    return closest_row['Close']
                else:
                    return None

            # Get fallback price from last year and current year
            price_last_year = get_closest_price_before_date(selected_stock_price_history_df, dv.last_yr, dv.current_mth,
                                                            dv.current_day)
            price_current_year = get_closest_price_before_date(selected_stock_price_history_df, dv.current_yr,
                                                               dv.current_mth, dv.current_day)

            # Decide what to use for YOY comparison
            price_year_ago = price_last_year or price_current_year

            if price_year_ago is not None:
                YOY_difference = selected_stock_regular_market_price - price_year_ago
                YOY_difference_number = round(YOY_difference, 2)
                YOY_difference_percentage = round((YOY_difference / price_year_ago) * 100)

                if YOY_difference_percentage > 0:
                    trend_icon = positive_icon
                elif YOY_difference_percentage < 0:
                    trend_icon = negative_icon
                else:
                    trend_icon = neutral_icon

                YOY_Price_Change = f"${YOY_difference_number} | {YOY_difference_percentage}% {trend_icon}"

                kpi_col2.write("YOY Price Change:")
                kpi_col2.markdown(YOY_Price_Change, unsafe_allow_html=True)

            else:
                kpi_col2.warning("YOY Price Change: No valid historical price found.")
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
            yoy_avg_close_price_change = round(yoy_avg_close_price_change_by_year.mean(), 2)

            # Create a trend icon for if the YOY price is positive or negative:
            if yoy_avg_close_price_change > 0:
                trend_icon = positive_icon  # Use positive trend icon
            elif yoy_avg_close_price_change < 0:
                trend_icon = negative_icon  # Use negative trend icon
            else:
                trend_icon = neutral_icon  # Neutral trend icon if the difference is 0

            # Write avg price change % over last 10 yrs to sidebar
            kpi_col3.write("AVG YOY $ Change (Last 10 Yrs):")
            yoy_avg_close_price_change = (
                    "$" + str(avg_price_chg_dollar_amt) + " | " + str(yoy_avg_close_price_change) + "% " + trend_icon)

            # Write to Home Page
            if yoy_avg_close_price_change is not None:
                # Write the value to the app for today in KPI format if the data is available
                kpi_col3.markdown(yoy_avg_close_price_change, unsafe_allow_html=True)
            else:
                kpi_col3.warning(f"AVG YOY $ Change (Last 10 Yrs): Data Not Available")
    # ------------------------------ Add KPI for Avg YOY price change over last 10 years from yesterday -----------------------------------------

    # -----------------------------------  Add KPI for Forecasted Price Based on Forecast Slider ------------------------------------------------
    with kpi_col4:
        kpi_col4 = st.container(border=True)
        with kpi_col4:
            # chooses the seasonal trend price (yhat) at the forecasted year dynamically with the sidebar
            chosen_forecasted_price = selected_stock_forecasted_df['yhat'].iloc[
                -1]  # forecast['yhat'].iloc[-1] retrieves the forecasted value
            print("Price Forecasted in Chosen Forecast Yr", chosen_forecasted_price)

            # Get Forecasted X Year $ Difference & Percentage:
            trend_difference = chosen_forecasted_price - selected_stock_regular_market_price
            trend_difference_number = round(trend_difference, 2)  # round the number to two decimal points
            trend_difference_percentage = (trend_difference / selected_stock_regular_market_price)
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
            chosen_forecasted_year = str(int(dv.current_yr) + int(selected_stock_forecasted_year_range))
            kpi_col4.write(f" {chosen_forecasted_year} Price Forecast:")

            # Forecasted Price Difference in 1 year ($ Difference + Current Price & % Difference Concatenated)
            chosen_forecasted_price_kpi = ("$" + str(round(chosen_forecasted_price, 2)) + " | " + str(
                trend_difference_percentage) + "% " + trend_icon)

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
    home_tab1, home_tab2, home_tab3 = st.tabs(
        ["Historical/Current Data", "Forecasted Data", "Model/Analyst Grades"])

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
                sh_col1 = st.container(border=True, height=465)
                with sh_col1:
                    sh_col1_1, sh_col1_2 = sh_col1.columns(2)

                    # run if statement to check there is data available today first:
                    if not selected_stock_price_history_df.empty:

                        # Check if data is available for the field today
                        if selected_stock_regular_market_price is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_1.metric(label="Current Price:", value="$" + str(
                                round(selected_stock_regular_market_price,
                                      2)))  # round # to 2 decimal points and make a string
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_1.warning(f"Current Price: {NA}")

                        # Check if data is available for the field today
                        if selected_stock_regular_market_open is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_1.metric(label="Open Price:", value="$" + str(
                                round(selected_stock_regular_market_open,
                                      2)))  # round # to 2 decimal points and make a string
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_1.warning(f"Open Price: {NA}")

                        # Add Latest Close Price:
                        # Note: we already set a var for latest close price in the load data section of the app
                        if last_close_price is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_2.metric(label=f"Last Close Price ({last_close_date}):",
                                             value="$" + str(round(last_close_price, 2)))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_2.warning(f"Last Close Price: {NA}")

                        # Add Trade Volume:
                        if selected_stock_regular_market_volume is not None:
                            selected_stock_regular_market_volume = "{:,}".format(
                                round(selected_stock_regular_market_volume, 2))  # adds commas
                            sh_col1_2.metric(label=f"Today's Trade Volume:",
                                             value=str(selected_stock_regular_market_volume))

                        # If variable from info is missing, check if we can get from history table
                        elif selected_stock_regular_market_volume is None:

                            # create variable to pull latest vol from history table
                            today_vol = selected_stock_price_history_df.loc[
                                selected_stock_price_history_df['Date'] == dv.today.strftime('%Y-%m-%d'), 'Volume']
                            today_vol = today_vol.iloc[0]

                            # Format trade volume number with commas for thousands separator
                            today_vol = "{:,.2f}".format(selected_stock_regular_market_volume)

                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_2.metric(label="Today's Trade Volume:", value=str(today_vol))

                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_2.warning(f"Today's Trade Volume: {NA}")

                        # Check if data is available for the field today
                        if selected_stock_pe_ratio is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_1.metric(label="PE:", value=str(round(selected_stock_pe_ratio, 2)))
                            print(f"PE Ratio today: {str(round(selected_stock_pe_ratio, 2))}")
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_1.warning(f"PE: {NA}")
                            print(f"PE: {NA}")

                        # Check if data is available for the field today
                        if selected_stock_peg_ratio is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_1.metric(label="PEG:", value=selected_stock_peg_ratio)
                            print(f"PEG Ratio today: {str(selected_stock_peg_ratio)}")
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_1.warning(f"PEG: {NA}")
                            print(f"PEG: {NA}")

                        # Check if data is available for the field today
                        if selected_stock_price_to_book is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_1.metric(label="Price-to-Book:", value=str(round(selected_stock_price_to_book, 2)))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_1.warning(f"Price-to-Book: {NA}")

                        # Check if data is available for the field today
                        if selected_stock_debt_to_equity is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_1.metric(label="Debt-to-Equity:",
                                             value=str(round(selected_stock_debt_to_equity * 100, 2)) + '%')
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_1.warning(f"Debt-to-Equity: {NA}")

                        # Check if data is available for the field today
                        if selected_stock_dividend_yield is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_1.metric(label="Dividend Yield:",
                                             value=str(round(selected_stock_dividend_yield, 2)) + "%")
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_1.warning(f"Dividend Yield: {NA}")

                        # Add Net Profit Margin
                        if selected_stock_net_profit_margin is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_1.metric(label="Net Profit Margin:",
                                             value=str(round(selected_stock_net_profit_margin * 100, 2)) + "%")
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_1.warning(f"Net Profit Margin: {NA}")

                        # Check if data is available for the field today
                        if selected_stock_beta is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_1.metric(label="Beta:", value=str(round(selected_stock_beta, 2)))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_1.warning(f"Beta: {NA}")

                        # Check if data is available for the field today
                        if selected_stock_price_to_sales is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_2.metric(label="Price-to-Sales:",
                                             value=str(round(selected_stock_price_to_sales, 2)))
                        else:
                            sh_col1_2.warning(f"Price-to-Sales: {NA}")

                        # Check if data is available for the field today
                        if selected_stock_roe is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_2.metric(label="ROE:", value=str(round(selected_stock_roe, 2)))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_2.warning(f"ROE Ratio: {NA}")

                        # Check if data is available for the field today
                        if selected_stock_current_ratio is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_2.metric(label="Current Ratio:", value=str(round(selected_stock_current_ratio, 2)))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_2.warning(f"Current Ratio: {NA}")

                        # Check if data is available for the field today
                        if selected_stock_quick_ratio is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_2.metric(label="Quick Ratio:", value=str(round(selected_stock_quick_ratio, 2)))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_2.warning(f"Quick Ratio: {NA}")

                        # Check if data is available for the field today
                        if selected_stock_dividend_rate is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_2.metric(label="Dividend Rate (Annual):",
                                             value='$' + str(round(selected_stock_dividend_rate, 2)))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_2.warning(f"Dividend Rate (Annual): {NA}")

                        # Check if data is available for the field today
                        if selected_stock_yoy_ocfg_growth is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_2.metric(label=f"YOY OCF Growth:", value=round(selected_stock_yoy_ocfg_growth, 2))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_2.warning(f"YOY OCF Growth: {NA}")

                        # Add Sharpe Ratio:
                        if selected_stock_sharpe_ratio is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1_2.metric(label=f"Sharpe Ratio:", value=str(selected_stock_sharpe_ratio))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1_2.warning(f"Sharpe Ratio: {NA}")

                        # Add 52 Week Price Range:
                        if selected_stock_fifty_two_week_range is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1.metric(label=f"52 Week Range:",
                                           value="$" + str(selected_stock_fifty_two_week_range))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1.warning(f"52 Week Range: {NA}")

                        # Add Enterprise Value Range:
                        if selected_stock_enterprise_value is not None:
                            # Write the value to the app for today in KPI format if the data is available
                            sh_col1.metric(label=f"Enterprise Value:",
                                           value="$" + "{:,.0f}".format(round(selected_stock_enterprise_value, 0)))
                        else:
                            # Write the data is not available for the field if missing
                            sh_col1.warning(f"Enterprise Value: {NA}")

                        # Add Analyst Recommendation:
                        if selected_stock_analyst_recommendation_summary is not None:
                            # Capitalize the first letter of the value
                            capitalized_value = str(selected_stock_analyst_recommendation_summary).capitalize()
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
                with st.container(border=True, height=465):
                    # Apply an exponential smoothing line to the graph starting - create smoothed pricing variable
                    smoothed_prices = data.apply_exponential_smoothing(selected_stock_price_history_df,
                                                                       smoothing_level=0.002)  # can change alpha

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
                            x=selected_stock_price_history_df['Date'],
                            y=selected_stock_price_history_df['Close'],
                            name='Price',
                            fill='tozeroy',  # adds color fill below trace
                            line=dict(color=trend_color),  # give line color based on smoothing trend line
                            fillcolor=trend_fill  # Light green with transparency
                        ))

                        # Trace the exponential smoothing line to the graph
                        fig.add_trace(go.Scatter(
                            x=selected_stock_price_history_df['Date'],
                            y=smoothed_prices,
                            name='Smoothing (a = .002)',
                            line=dict(color='yellow', width=1.5, dash='dash')  # Red dashed line for smoothed prices
                        ))

                        # Get stock data date range
                        ss_stock_start_date = selected_stock_price_history_df['Date'].min()
                        ss_stock_end_date = selected_stock_price_history_df['Date'].max()

                        # Trace a line for the S&P 500 (SPY) as a benchmark to the selected stock
                        # First, need to normalize the SPY data to match the stock's initial price
                        stock_initial_price = selected_stock_price_history_df['Close'].iloc[0]
                        spy_initial_price = SPY_data['Close'].iloc[0]

                        # Scale the SPY data to the stock's initial price
                        normalized_spy = (SPY_data['Close'] / spy_initial_price) * stock_initial_price

                        # Trace SPY to graph - only show full SPY if stock data goes back far enough
                        if ss_stock_start_date >= SPY_data['Date'].min():  # only show if have more than 3640 days data
                            fig.add_trace(go.Scatter(
                                x=SPY_data['Date'],
                                y=normalized_spy,
                                name='S&P 500',
                                line=dict(color='gray', width=1.5, dash='dash')  # Red dashed line for smoothed prices
                            ))

                        # Filter data for one year Spy normalized to selected stock
                        one_year_ago = dv.one_yr_ago
                        SPY_one_year = SPY_data[SPY_data['Date'] >= one_year_ago].copy()
                        stock_one_year = selected_stock_price_history_df[selected_stock_price_history_df['Date'] >= one_year_ago].copy()

                        # Get the stock's price from one year ago (starting point for normalization)
                        stock_one_year_initial = stock_one_year['Close'].iloc[0]
                        spy_one_year_initial = SPY_one_year['Close'].iloc[0]

                        # Normalize SPY to match the stock's price from one year ago
                        normalized_spy_one_year = (SPY_one_year['Close'] / spy_one_year_initial) * stock_one_year_initial

                        # Add one-year SPY trace
                        if ss_stock_start_date <= SPY_one_year['Date'].min():
                            fig.add_trace(go.Scatter(
                                x=SPY_one_year['Date'],
                                y=normalized_spy_one_year,
                                name='S&P 500 (1Y)',
                                line=dict(color='violet', width=2, dash='dot')
                            ))

                        # Filter data for one year Spy normalized to selected stock
                        three_year_ago = dv.three_yrs_ago
                        SPY_three_year = SPY_data[SPY_data['Date'] >= three_year_ago].copy()
                        stock_three_year = selected_stock_price_history_df[selected_stock_price_history_df['Date'] >= three_year_ago].copy()

                        # Get the stock's price from one year ago (starting point for normalization)
                        stock_three_year_initial = stock_three_year['Close'].iloc[0]
                        spy_three_year_initial = SPY_three_year['Close'].iloc[0]

                        # Normalize SPY to match the stock's price from one year ago
                        normalized_spy_three_year = (SPY_three_year['Close'] / spy_three_year_initial) * stock_three_year_initial

                        # Add three-year SPY trace
                        if ss_stock_start_date <= SPY_three_year['Date'].min():
                            fig.add_trace(go.Scatter(
                                x=SPY_three_year['Date'],
                                y=normalized_spy_three_year,
                                name='S&P 500 (3Y)',
                                line=dict(color='lightblue', width=2, dash='dot')
                            ))

                        # Update layout with time range selector buttons
                        fig.update_layout(
                            xaxis=dict(
                                rangeslider=dict(visible=True),
                                rangeselector=dict(
                                    buttons=list([
                                        dict(count=1, label="1W", step="day", stepmode="backward"),
                                        dict(count=6, label="6M", step="month", stepmode="backward"),
                                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                                        dict(step="all", label="All")
                                    ])
                                )
                            ),
                            template='plotly_white'
                        )

                        st.plotly_chart(fig,
                                        use_container_width=True)  # writes the graph to app and fixes width to the container width

                    plot_raw_data()

            # Get Business Description and add to dropdown
            # Function to split the description into paragraphs every 3 sentences
            def split_into_paragraphs(text, sentence_count=4):
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                paragraphs = [". ".join(sentences[i:i + sentence_count]) + '.' for i in
                              range(0, len(sentences), sentence_count)]
                return paragraphs

            # Get the paragraphs by splitting the stock description into chunks of 3 sentences
            paragraphs = split_into_paragraphs(selected_stock_business_description, 4)

            # Wrap in dropdown visual
            with st.expander(
                    f"{selected_stock_company_name} Overview - {selected_stock_sector} / {selected_stock_industry}"):
                for paragraph in paragraphs:
                    st.markdown(paragraph)

            # Price Trend Section (Long Term Price Trend)
            sh_c.write("Price Trend History - Long Term (Limit 10 Years):")

            # Add a Yearly Data Trend visual to container:
            with sh_c.container(border=True):

                # Apply HTML formatting for "Price Trend" column to add as a column in the df
                def highlight_trend(row):
                    trend = row['Trend']
                    if trend == '↑':
                        color = '#388E3C'
                    elif trend == '↓':
                        color = '#D32F2F'
                    elif trend == '■':
                        color = 'grey'
                    else:
                        color = 'black'
                    # Apply color only to specific columns
                    return [
                        f'color: {color}' if col in ['Trend', 'Price Change', 'Percentage Change'] else ''
                        for col in row.index
                    ]

                # Drop columns not needed for the visual
                yearly_data = yearly_data.drop(columns=['Year', 'Returns', 'Excess_Returns'], errors='ignore')

                # order yearly_data by date
                yearly_data = yearly_data.sort_values(by='Date', ascending=False)

                # Convert 'Date' column to datetime format
                yearly_data['Date'] = pd.to_datetime(yearly_data['Date'])

                # Format 'Date' column to remove the time component
                yearly_data['Date'] = yearly_data['Date'].dt.strftime('%Y-%m-%d')

                # Move the Trend column to the first column
                yearly_data_columns = ['Trend'] + [col for col in yearly_data.columns if
                                                   col != 'Trend']  # variable for columns
                yearly_data = yearly_data[yearly_data_columns]

                styled_df = yearly_data.style \
                    .apply(highlight_trend, axis=1) \
                    .format({
                    'Close': '{:.2f}',
                    'High': '{:.2f}',
                    'Low': '{:.2f}',
                    'Open': '{:.2f}',
                    'Volume': '{:.0f}',
                    'Price Change': '${:.2f}',
                    'Percentage Change': '{:.2f}%'
                })

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
                        one_yr_moving_average_data = selected_stock_moving_average_data[
                            selected_stock_moving_average_data['Date'] >= dv.one_yr_ago]

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
                        above_50_sma = one_yr_moving_average_data[
                            one_yr_moving_average_data['50_day_SMA'] > one_yr_moving_average_data['200_day_SMA']]
                        below_50_sma = one_yr_moving_average_data[
                            one_yr_moving_average_data['50_day_SMA'] < one_yr_moving_average_data['200_day_SMA']]

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
                        fig.layout.update(title='Moving Averages - One Year', xaxis_rangeslider_visible=True,
                                          template='plotly_white')
                        st.plotly_chart(fig,
                                        use_container_width=True)  # writes the graph to app and fixes width to the container width

                    # run function
                    plot_mov_avg_data()

                    # kpi for Present day SMA indicator
                    with sh_col1.container(border=True):

                        # Logic to decide if it's a good buy or sell based on the crossover
                        if selected_stock_sma_percentage_difference > 5:
                            signal = "Bullish Momentum"
                            color = "#388E3C"  # Green color for Buy
                            text_color = "white"  # Setting as white but leaving a variable if want to change in the future
                            action = "BUY"
                            indicator = f"{selected_stock_sma_percentage_difference:.2f}% Price Differential*"
                        if selected_stock_sma_percentage_difference < -5:
                            signal = "Bearish Momentum"
                            color = "#D32F2F"  # Red color for Sell
                            text_color = "white"  # Setting as white but leaving a variable if want to change in the future
                            action = "SELL"
                            indicator = f"{selected_stock_sma_percentage_difference:.2f}% Price Differential*"
                        else:
                            signal = "Neutral Momentum"
                            color = "#C79200"  # Yellow color for Neutral
                            text_color = "white"  # Setting as white but leaving a variable if want to change in the future
                            action = "HOLD"
                            indicator = f"{selected_stock_sma_percentage_difference:.2f}% Price Differential*"

                        # Provide KPI title
                        st.markdown(
                            "<p style='margin: 0; padding: 0; font-size: 14px; '>Latest 50-Day & 200-Day SMA Price Differential:</p>",
                            unsafe_allow_html=True)  # writing with html removes extra spacing between lines

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
                                '>${selected_stock_sma_price_difference:.2f}</div>
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
                    if selected_stock_rsi_score > 70:
                        signal = "Currently Overbought - Consider Sell*"
                        color = "#D32F2F"  # Sell signal in red
                        text_color = "white"  # Setting as white but leaving a variable if want to change in the future
                    elif selected_stock_rsi_score < 30:
                        signal = "Currently Oversold - Consider Buy*"
                        color = "#388E3C"  # Buy signal in green
                        text_color = "white"  # Setting as white but leaving a variable if want to change in the future
                    else:
                        signal = "Neutral Momentum - Consider Holding*"
                        color = "#C79200"  # Neutral signal in yellow
                        text_color = "white"  # Setting as white but leaving a variable if want to change in the future

                    # Provide visual title
                    st.markdown("<p style='margin: 0; padding: 0; font-size: 14px; '>RSI Value (14-Day Window):</p>",
                                unsafe_allow_html=True)  # writing with html removes extra spacing between lines

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
                            '>{selected_stock_rsi_score:.2f}</div>
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
                            x=selected_stock_macd_df['Date'],
                            y=selected_stock_macd_df['MACD'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='rgba(240, 194, 0, 0.8)')))  # yellow

                        # Add Signal Line
                        fig.add_trace(go.Scatter(
                            x=selected_stock_macd_df['Date'],
                            y=selected_stock_macd_df['Signal'],
                            mode='lines',
                            name='Signal',
                            line=dict(color='rgba(200, 55, 45, 0.8)')))  # red

                        # Add Histogram (Difference between MACD and Signal)
                        fig.add_trace(go.Bar(
                            x=selected_stock_macd_df['Date'],
                            y=selected_stock_macd_df['Histogram'],
                            name='Histogram',
                            marker=dict(color='rgba(180, 180, 180, 1)'),
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

                # Expander with description tabs inside
                with st.expander("Leveraging Short Term Models for Short-Term Buy/Sell/Hold Actions*"):
                    tab1, tab2, tab3 = st.tabs(["RSI", "SMA", "MACD"])

                    # RSI Tab
                    with tab1:
                        st.markdown("""
                        <span style="color:lightcoral; font-weight:bold;">**Relative Strength Index (RSI)**</span> is a momentum oscillator that provides a range from 0-100 to indicate if the asset is under or overpriced based on its speed and change of price movements.
                        
                        RSI Ranges can be interpreted with the following logic:
                        - <span style="color:lightcoral; font-weight:bold;">**RSI > 70**</span> = Likely Overbought (Sell Signal): The maximum expected loss for one day.
                        - <span style="color:lightcoral; font-weight:bold;">**RSI ≈ 50**</span> = Neutral State: The maximum expected loss for one month.
                        - <span style="color:lightcoral; font-weight:bold;">**RSI < 30**</span> = Likely Oversold (Buy Signal): The maximum expected loss for one year.
                
                        <span style="color:lightcoral; font-weight:bold;">**Example:**</span>
                        - If Meta's RSI is under 20, this indicates a more-extreme likelihood of market oversell compared to if the RSI were at 40, indicating market activity is in a neutral state.
                        """, unsafe_allow_html=True)

                    # SMA Tab
                    with tab2:
                        st.markdown(""" 
                        <span style="color:lightcoral; font-weight:bold;">**Simple Moving Average (SMA)**</span> is a statistical calculation that smooths the given range of price data to help identify price trends/cycles. Can leverage the long-term and short-term SMAs together as follows:
                        
                        - <span style="color:lightcoral; font-weight:bold;">**200-Day SMA**</span> -> Use as long-term trend indicator to benchmark against the 50-day SMA indicator.
                        - <span style="color:lightcoral; font-weight:bold;">**50-Day SMA > 200-Day SMA**</span> -> ST Bullish Signal: Referred to as a "Golden Cross" - Indicates the price has trended above its LT SMA and has upward momentum.
                        - <span style="color:lightcoral; font-weight:bold;">**50-Day SMA < 200-Day SMA**</span> -> ST Bearish Signal: Referred to as a "Death Cross" - Indicates the price has trended below its LT SMA and has downward momentum.
                        
                        <span style="color:lightcoral; font-weight:bold;">**Key Decision Points:**</span>
                        - <span style="color:lightcoral; font-weight:bold;">**Buy Signal (Green)**</span> -> When 50-Day SMA crosses above 200-Day SMA (line turns green) + price is above both SMAs
                        - <span style="color:lightcoral; font-weight:bold;">**Sell Signal (Red)**</span> -> When 50-Day SMA crosses below 200-Day SMA (line turns red) + price is below both SMAs
                        - <span style="color:lightcoral; font-weight:bold;">**Support/Resistance**</span> -> Price often bounces off SMA lines - watch for rejections or breaks
                        - <span style="color:lightcoral; font-weight:bold;">**Trend Strength**</span> -> Wider gap between SMAs = stronger trend; converging SMAs = weakening trend
                        
                        <span style="color:lightcoral; font-weight:bold;">**Position Sizing Indicator:**</span>
                        - <span style="color:lightcoral; font-weight:bold;">**>5% Differential**</span> -> Strong bullish momentum (BUY signal)
                        - <span style="color:lightcoral; font-weight:bold;">**<-5% Differential**</span> -> Strong bearish momentum (SELL signal)  
                        - <span style="color:lightcoral; font-weight:bold;">**Between -5% & 5%**</span> -> Neutral momentum (HOLD signal)
                        
                        <span style="color:lightcoral; font-weight:bold;">**Example:**</span> 
                        - If the price is above both the 50-day and 200-day SMAs, but it drops and then finds support at the 50-day SMA, it could signal that the short-term trend remains intact.
                        - Look for volume confirmation when SMAs cross - higher volume makes the signal more reliable.
                        """, unsafe_allow_html=True)

                    # MACD Tab
                    with tab3:
                        st.markdown("""
                        <span style="color:lightcoral; font-weight:bold;">**MACD (Moving Average Convergence Divergence)**</span> is a trend analysis indicator used to visualize momentum of a stock and buy/sell signals based on two staggered exponential moving averages (EMAs). The function in this application uses a 26-day EMA for the long and a 12-day EMA for the short trend line. The signal line uses a 9-day EMA which helps provide a clearer indication for buy/sell opportunity in the current short-term market state.
                        
                        - <span style="color:lightcoral; font-weight:bold;">**MACD Line (Yellow)**</span> -> The difference between the 12-day and 26-day EMA (12-Day EMA - 26-day EMA). A positive MACD value (e.g., +4) means the 12-day EMA is above the 26-day EMA, suggesting that an uptrend (bullish momentum) is currently present in the market. Conversely, if the 12-day is below the 26-day, this would indicate the ticker is in a downtrend (bearish momentum). The Y-axis is the variation in price between the two averages. The higher the number, the stronger the momentum.
                         
                        <span style="color:lightcoral; font-weight:bold;">**Note:**</span> You can also use the length of fluctuation as a benchmark to predict that the stock might be nearing a reversal in momentum (e.g. ticker typically caps at 4; if it is currently at a difference of 3, this could mean it may be near a reversal). So in theory, a good entry point would be when the stock is nearing the end of a downtrend and is gearing for reversal.
                        - <span style="color:lightcoral; font-weight:bold;">**Signal Line (Red)**</span> -> The 9-day signal line used as a benchmark against the MACD line. 
                        - <span style="color:lightcoral; font-weight:bold;">**Histogram (Grey)**</span> -> The histogram is used in addition to the MACD line to visually show when the ticker is indicating bullish vs bearish momentum by utilizing the Signal line against it. When the MACD line is above the Signal line, the histogram traces positive or above the 0-axis (bullish), and when the MACD line is below the Signal line, the histogram falls negative or below the 0-axis (bearish).
                
                        <span style="color:lightcoral; font-weight:bold;">**What the Point Differential Actually Means**</span> <span style="color:white;">-></span> The MACD value represents the **actual dollar difference** between the 12-day and 26-day EMAs. If MACD shows +15, this means the 12-day EMA is literally $15 higher than the 26-day EMA in terms of stock price.
    
                        <span style="color:lightcoral; font-weight:bold;">**High-Priced Stocks (e.g., META at $500+):**</span>
                        - <span style="color:lightcoral; font-weight:bold;">**Range 20-30**</span> <span style="color:white;">-></span> Strong momentum - $20-30 price gap between EMAs indicates significant trend strength
                        - <span style="color:lightcoral; font-weight:bold;">**Range 10-20**</span> <span style="color:white;">-></span> Moderate momentum - $10-20 gap shows normal trending behavior  
                        - <span style="color:lightcoral; font-weight:bold;">**Range 0-10**</span> <span style="color:white;">-></span> Weak momentum - Small dollar gap indicates consolidation or sideways movement
                        - <span style="color:lightcoral; font-weight:bold;">**Example**</span> <span style="color:white;">-></span> META showing MACD at +25 means the 12-day EMA is $25 above the 26-day EMA (very strong bullish momentum)
                        
                        <span style="color:lightcoral; font-weight:bold;">**Mid-Priced Stocks (e.g., AAPL at $180):**</span>
                        - <span style="color:lightcoral; font-weight:bold;">**Range 3-8**</span> <span style="color:white;">-></span> Strong momentum - $3-8 price gap is significant for this price level
                        - <span style="color:lightcoral; font-weight:bold;">**Range 1-3**</span> <span style="color:white;">-></span> Moderate momentum - $1-3 gap shows normal movement
                        - <span style="color:lightcoral; font-weight:bold;">**Range 0-1**</span> <span style="color:white;">-></span> Weak momentum - Small dollar gap indicates weak trend
                        - <span style="color:lightcoral; font-weight:bold;">**Example**</span> <span style="color:white;">-></span> AAPL showing MACD at +5 means the 12-day EMA is $5 above the 26-day EMA (strong bullish momentum for AAPL's price range)
                        
                        <span style="color:lightcoral; font-weight:bold;">**Key Decision Points:**</span>
                        - <span style="color:lightcoral; font-weight:bold;">**Buy Signal**</span> <span style="color:white;">-></span> MACD line crosses above Signal line (histogram turns positive) + MACD is recovering from negative territory
                        - <span style="color:lightcoral; font-weight:bold;">**Sell Signal**</span> <span style="color:white;">-></span> MACD line crosses below Signal line (histogram turns negative) + MACD is declining from positive territory
                        - <span style="color:lightcoral; font-weight:bold;">**Momentum Strength**</span> <span style="color:white;">-></span> Higher absolute dollar values = stronger trends (but watch for exhaustion at historical extremes)
                        - <span style="color:lightcoral; font-weight:bold;">**Divergence**</span> <span style="color:white;">-></span> Price making new highs while MACD makes lower highs = potential reversal warning
                        
                        <span style="color:lightcoral; font-weight:bold;">**Real-World Example - META:**</span>
                        - <span style="color:lightcoral; font-weight:bold;">**Scenario**</span> <span style="color:white;">-></span> META trading at $520, MACD at +15, Signal line at +12, Histogram positive
                        - <span style="color:lightcoral; font-weight:bold;">**Analysis**</span> <span style="color:white;">-></span> The $15 dollar gap between EMAs shows strong bullish momentum, MACD above Signal line confirms uptrend
                        - <span style="color:lightcoral; font-weight:bold;">**Action**</span> <span style="color:white;">-></span> If META's historical MACD range is -30 to +30, current +15 shows moderate-strong bullish momentum with room to run
                        - <span style="color:lightcoral; font-weight:bold;">**Warning**</span> <span style="color:white;">-></span> If MACD reaches +28 (near historical high), watch for potential reversal or consolidation
                        """, unsafe_allow_html=True)

            # Add Risk Section:
            sh_c.write("Risk Assessment:")

            # Create new Container for VaR Risk metrics:
            with sh_c.container(border=True):

                # Header for VAR risk section
                st.markdown("<h4 style='text-align: center;'>Historical VaR - Model Metrics</h3>",
                            unsafe_allow_html=True)

                # Write to container horizontally in 3 columns:
                risk_col1, risk_col2, risk_col3 = st.columns(3)

                with risk_col1:
                    st.metric(label="1-Day VaR at 95% Confidence",
                              value=f"{selected_stock_hist_daily_VaR_95 * 100:.2f}%",
                              delta=f"${selected_stock_hist_daily_VaR_95_dollars:.2f}", delta_color="inverse")

                with risk_col2:
                    st.metric(label="1-Month VaR at 95% Confidence",
                              value=f"{selected_stock_hist_monthly_VaR_95 * 100:.2f}%",
                              delta=f"${selected_stock_hist_monthly_VaR_95_dollars:.2f}", delta_color="inverse")

                with risk_col3:
                    st.metric(label="1-Year VaR at 95% Confidence",
                              value=f"{selected_stock_hist_yearly_VaR_95 * 100:.2f}%",
                              delta=f"${selected_stock_hist_yearly_VaR_95_dollars:.2f}", delta_color="inverse")

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
                st.markdown("<h4 style='text-align: center;'>Monte Carlo Simulation - 1 Year Trading Period</h3>",
                            unsafe_allow_html=True)

                # Define the number of days (assuming your simulation is over 252 trading days, adjust if needed)
                num_days = selected_stock_mc_sim_df.shape[
                    0]  # This is the number of rows in your DataFrame, assuming it's daily data

                # Generate the range of days for the x-axis (e.g., 1 to num_days)
                x_axis = list(range(num_days))

                # Initialize the figure
                fig = go.Figure()

                # Set color map (replicate the "plasma" color map from matplotlib)
                mc_colormap = plt.cm.plasma  # Set to "plasma" colormap
                mc_colors = [mc_colormap(i / len(selected_stock_mc_sim_df.columns)) for i in
                             range(len(selected_stock_mc_sim_df.columns))]

                # Plot the Monte Carlo simulations (with the faded color)
                for i in range(selected_stock_mc_sim_df.shape[1]):
                    fig.add_trace(go.Scatter(
                        x=x_axis,  # Use the generated x-axis
                        y=selected_stock_mc_sim_df.iloc[:, i],  # Simulation data
                        mode='lines',
                        line=dict(
                            color=f'rgba({int(mc_colors[i][0] * 255)}, {int(mc_colors[i][1] * 255)}, {int(mc_colors[i][2] * 255)}, 0.1)'),
                        # Faded color using rgba
                        showlegend=False
                    ))

                # Add the percentiles (5th, Median, 95th percentiles)
                fig.add_trace(go.Scatter(
                    x=x_axis,  # Use the generated x-axis
                    y=mc_percentile_5,
                    name="5th Percentile",
                    line=dict(color='red', dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=x_axis,  # Use the generated x-axis
                    y=mc_percentile_50,
                    name="Median",
                    line=dict(color='#FFE87C', dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=x_axis,  # Use the generated x-axis
                    y=mc_percentile_95,
                    name="95th Percentile",
                    line=dict(color='green', dash='dash')
                ))

                # Layout adjustments
                fig.update_layout(
                    # title="Monte Carlo Simulation - 1 Year Trading Period",
                    xaxis_title="Days",
                    yaxis_title="Price",
                    template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
                    # Adjusting for dark/light mode
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255, 255, 255, 0.3)' if st.get_option(
                            "theme.base") == "dark" else 'rgba(0, 0, 0, 0.1)',  # Adjust grid color based on theme
                        zerolinecolor='rgba(255, 255, 255, 0.5)' if st.get_option(
                            "theme.base") == "dark" else 'rgba(0, 0, 0, 0.3)',  # Zero line color
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255, 255, 255, 0.3)' if st.get_option(
                            "theme.base") == "dark" else 'rgba(0, 0, 0, 0.1)',  # Adjust grid color based on theme
                        zerolinecolor='rgba(255, 255, 255, 0.5)' if st.get_option(
                            "theme.base") == "dark" else 'rgba(0, 0, 0, 0.3)',  # Zero line color
                    ),
                    legend=dict(font=dict(size=10))
                )

                # Display the Plotly chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)

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
            with sh_c.container(border=False):
                # Add DF with industry averages
                if not industry_avg_df.empty:

                        # Get industry avg df
                        ind_avg_pe = industry_avg_df.loc[industry_avg_df['Industry'] == selected_stock_industry, 'Average P/E Ratio'].values[0]
                        ind_avg_roe = industry_avg_df.loc[industry_avg_df['Industry'] == selected_stock_industry, 'Average ROE'].values[0]

                        # Create four columns for the card layout
                        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

                        kpi_height = 135  # set reusable height

                        # Add kpi
                        with col1:
                            try:
                                # get delta
                                col_4_price_delta = selected_stock_pe_ratio - ind_avg_pe

                                with st.container(height=kpi_height):
                                    st.metric(
                                        label="Current PE",
                                        value=f"{selected_stock_pe_ratio:.2f}",
                                        delta=f"{col_4_price_delta:.2f} from Industry Avg",
                                        delta_color="inverse"  # Invert the color logic
                                    )
                            except:
                                with st.container(height=kpi_height):
                                    st.warning(f"PE Data Not Available")
                        # Add kpi
                        with col2:
                            try:
                                # get delta
                                col_5_price_delta = selected_stock_roe - ind_avg_roe

                                with st.container(height=kpi_height):
                                    st.metric(
                                        label="Current ROE",
                                        value=f"{selected_stock_roe:.2f}",
                                        delta=f"{col_5_price_delta:.2f} from Industry Avg"
                                    )
                            except:
                                with st.container(height=kpi_height):
                                    st.warning(f"ROE Data Not Available")
                else:
                    st.warning(f"Industry Average Data Not Available")

    # ///////////////////////////////////////////////////////////// Forecast Tab //////////////////////////////////////////////////////////////////////////

    # Add forecasted data visuals to our second tab
    with home_tab2:
        # Create forecast container for our forecast section and columns:
        fs_c = st.container()

        # Add Visuals into fs_c container:
        with fs_c:
            # Create two columns for fs_c
            fs_c_col1, fs_c_col2 = fs_c.columns([8, 3])  # Use ratios to make control area width

            # Write Title for Forecast Graph in col1:
            fs_c_col1.write("Forecast Graph:")

            # Write Forecast Graph Container in col1:
            fs_graph_c = fs_c_col1.container(border=True, height=635)
            with fs_graph_c:
                # Plot the forecasted future data using the prophet model within a forecasted visual:
                fig1 = plot_plotly(selected_stock_trained_model,
                                   selected_stock_forecasted_df)  # plot visual (plotly needs the model and forecast to plot)

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

                # Write metric 1 container
                fs_price_metric1 = fs_price_metric.container(border=True)
                with fs_price_metric1:
                    # Recreate KPI 4 as streamlit metric
                    fs_price_metric1.metric(f"Forecast Year: {chosen_forecasted_year}",
                                            f"${str(round(chosen_forecasted_price, 2))}",
                                            f"{trend_difference_percentage:.0f}%")

                # Write metric 2 container
                fs_price_metric2 = fs_price_metric.container(border=True)
                with fs_price_metric2:
                    # Get the average YOY forecasted price change
                    yoy_avg_fr_price_change = round(
                        (int(chosen_forecasted_price) - int(selected_stock_regular_market_price)) / (
                                int(chosen_forecasted_year) - int(dv.current_yr)), 2)

                    # Get the Average Change %
                    yoy_avg_fr_price_change_pct = round(
                        (yoy_avg_fr_price_change / selected_stock_regular_market_price) * 100, 0)

                    # Add a metric for avg forecast price change from current price YOY dynamically with sidebar
                    fs_price_metric2.metric(f"YOY $ Change Average:", f"${str(round(yoy_avg_fr_price_change, 2))}",
                                            f"{yoy_avg_fr_price_change_pct:.0f}%")

            # Write Title for Forecast Components in col2:
            fs_c_col2.write("Forecast Components:")

            # Write Forecast Components Container in col2
            fs_components_c = fs_c_col2.container(border=True, height=307)
            with fs_components_c:
                # Plot the Prophet forecast components:
                fig2 = selected_stock_trained_model.plot_components(
                    selected_stock_forecasted_df)  # write component visuals to label fig2.
                fs_components_c.pyplot(fig2)  # write figure 2 to the app

                # Notes:
                # By calling m.plot_components(forecast), we're instructing prophet to generate a plot showing the components of the forecast. These components include trends, seasonalities, and holiday effect
                # The result of m.plot_components(forecast) is a set of plots visualizing these components

            # Write Title for Forecast Tail:
            fs_c.write("Price Forecast Overview:")

            # Write Forecast Overview Container:
            fs_overview_c = fs_c.container(border=True)
            with fs_overview_c:
                # PLot our forecasted information in tabular format:
                fs_overview_c.dataframe(selected_stock_forecasted_df.tail(1), hide_index=True)

    # ///////////////////////////////////////////////////////////// Stock Grades Tab //////////////////////////////////////////////////////////////////////////

    # Create DataFrame from the dictionary
    selected_stock_score_details_df = pd.DataFrame.from_dict(selected_stock_score_details, orient='index')

    with home_tab3:
        # Create container variable for the grades tab
        sh_g = st.container()

        # Ticker Analyst #s Section
        sh_g.write("Model Grade Summary:")

        # Write Analyst Grades to App
        with sh_g.container(border=True):

            # Get grades from AWS
            grades_bucket_name = 'stock-ticker-data-bucket'  # S3 bucket name
            grades_csv_path = 'ticker_grades_output.csv'  # name of object in S3 bucket

            grades_df = data.load_csv_from_s3(grades_bucket_name, grades_csv_path)

            # Add KPIs at the top
            total_score_row = selected_stock_score_details_df.loc["Total Score"]

            # Multiply by 100 for viewing
            total_score = total_score_row['score'] * 100
            total_max_score = total_score_row['max'] * 100

            # Display the stacked bar chart in Streamlit
            st.write("Score Summary:")

            # Define grade mapping using midpoint of each range
            grade_mapping = {
                'S': 13,
                'A': 12,
                'A-': 11,
                'B+': 10,
                'B': 9,
                'B-': 8,
                'C+': 7,
                'C': 6,
                'C-': 5,
                'D+': 4,
                'D': 3,
                'D-': 2,
                'F': 1
            }

            # Convert grades to numeric values
            grades_df['Grade_Numeric'] = grades_df['Grade'].map(grade_mapping)

            # selected stock grade as numeric with mapping
            ss_grade_numeric = grade_mapping.get(selected_stock_grade, 0)  # defaults to 0 if grade not found

            # Calculate averages
            avg_grade_score = round(grades_df['Score'].mean(), 1) * 100
            avg_grade = round(grades_df['Grade_Numeric'].mean(), 0)

            # Calculate score difference for delta
            grade_score_difference = total_score - avg_grade_score

            # Calculate score difference for delta
            grade_difference = ss_grade_numeric - avg_grade

            with st.container():
                col1, col2, col3 = st.columns([2, 2, 6])

                with col1:
                    with st.container(border=True, height=135):
                        st.metric("Score Achieved",
                                  f"{total_score:.1f} / {total_max_score:.0f}",
                                  delta=f"{grade_score_difference:+.1f} From Avg Score")
                with col2:
                    with st.container(border=True, height=135):
                        st.metric("Model Grade",
                                  f"{selected_stock_grade}",
                                  delta=f"{grade_difference:+.0f} From Avg Grade")
                with col3:
                    with st.container(border=True, height=135):
                        # Define grade order from best to worst
                        grade_order = ['S', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F']

                        # Reverse it so lowest grade (F) has 0
                        grade_order_reversed = list(reversed(grade_order))

                        # Create mapping
                        grade_to_num = {grade: i for i, grade in enumerate(grade_order_reversed)}
                        # num_to_grade = {i: grade for grade, i in grade_to_num.items()}

                        # Map grades in your DataFrame
                        grades_df['grade_numeric'] = grades_df['Grade'].map(grade_to_num)

                        # Get numeric value of select stock grade
                        ss_grade_numeric = grade_to_num[selected_stock_grade]

                        # Get current theme colors
                        text_color = st.get_option("theme.textColor") or "#262730"
                        background_color = st.get_option("theme.backgroundColor") or "#FFFFFF"
                        # primary_color = st.get_option("theme.primaryColor") or "#FF6B6B"

                        # Create the plot with transparent background
                        fig, ax = plt.subplots(figsize=(15, 1))
                        fig.patch.set_facecolor(background_color)  # Set figure background to theme color
                        ax.patch.set_facecolor(background_color)   # Set axes background to theme color

                        # Set text colors to match theme
                        ax.tick_params(colors=text_color)
                        ax.xaxis.label.set_color(text_color)
                        ax.yaxis.label.set_color(text_color)
                        ax.title.set_color(text_color)

                        # Create horizontal box plot with theme-aware colors
                        box_plot = sns.boxplot(
                            x=grades_df['grade_numeric'],
                            ax=ax,
                            color='#FFCCCC',  # salmon
                            fliersize=3,
                            orient='h'
                        )

                        # set outside border to white
                        for spine in ax.spines.values():
                            spine.set_color(text_color)  # Change to white/theme color

                        # Add a vertical yellow line for selected stock's grade
                        ax.axvline(ss_grade_numeric, color='#DAA520', linestyle='--', linewidth=2,
                                   label=f'Selected Stock')

                        # Set up the x-axis
                        ax.set_xticks(range(len(grade_order)))
                        ax.set_xticklabels(grade_order_reversed)  # Use reversed order for proper display
                        ax.set_xlabel("Stock Grades")
                        ax.set_ylabel("")

                        # Remove y-axis ticks since we don't need them
                        ax.set_yticks([])

                        # Add title
                        ax.set_title("Grade Distribution of All Stocks", fontsize=12, pad=20)

                        # Add legend with theme colors
                        legend = ax.legend(loc='upper right', fontsize='small')
                        legend.get_frame().set_facecolor('none')
                        legend.get_frame().set_alpha(0)
                        for text in legend.get_texts():
                            text.set_color(text_color)

                        # Display the plot
                        st.pyplot(fig)

            # Show bar chart against max possible scores
            st.write("Score Breakdown:")

            with st.container(border=True):

                # Prepare data for the bar chart with line overlay
                chart_data = []

                # loop through each metric
                for index, row in selected_stock_score_details_df.iterrows():
                    # Skip the "Total Score" and "Base Points" rows
                    if index in ["Total Score", "Base Points"]:
                        continue

                    score = row['score'] * 100
                    max_score = row['max'] * 100
                    value = row['value']

                    # Add data for the chart
                    chart_data.append({
                        'Metric': index,
                        'Achieved Score': score,
                        'Max Score': max_score,
                        'Value': str(round(value, 2)) if isinstance(value, (int, float)) else str(value)
                    })

                # Create DataFrame for plotting and sort by max score (descending)
                chart_df = pd.DataFrame(chart_data)
                chart_df = chart_df.sort_values('Max Score', ascending=False)

                # Create the combined bar and line chart using Plotly Graph Objects
                fig = go.Figure()

                # Add bar chart for achieved scores
                fig.add_trace(go.Bar(
                    x=chart_df['Metric'],
                    y=chart_df['Achieved Score'],
                    name='Achieved Score',
                    marker_color='#2E8B57',  # Sea Green
                    customdata=chart_df['Value'],
                    hovertemplate='<b>%{x}</b><br>' +
                                  'Achieved Score: %{y}<br>' +
                                  'Value: %{customdata}<br>' +
                                  '<extra></extra>'
                ))

                # Add line chart for max scores
                fig.add_trace(go.Scatter(
                    x=chart_df['Metric'],
                    y=chart_df['Max Score'],
                    mode='markers',
                    name='Max Possible',
                    marker=dict(size=8, color="#DAA520"),
                    hovertemplate='<b>%{x}</b><br>' +
                                  'Max Possible: %{y}<br>' +
                                  '<extra></extra>'
                ))

                # Add horizontal benchmark line for each marker
                for i, row in chart_df.iterrows():
                    fig.add_hline(
                        y=row['Max Score'],
                        line=dict(color="#DAA520", width=1, dash="dash"),
                        opacity=0.5,
                        layer="below"  # puts lines behind other traces
                    )

                # Update layout
                fig.update_layout(
                    # title='Score Breakdown by Metric',
                    xaxis_title="Metrics",
                    yaxis_title="Score",
                    showlegend=True,
                    height=400,
                    hovermode='x unified'
                )

                # Rotate x-axis labels for better readability
                fig.update_xaxes(tickangle=45)

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)

        # Ticker Analyst #s Section
        sh_g.write("Yahoo Finance Analyst Summary:")

        # Write Analyst Grades to App
        with sh_g.container(border=True):

            col1, col2 = st.columns([4,3])

            with col1:
                # ----------------- Yahoo Finance Grades
                # Display the DataFrame in Streamlit
                st.write("Year End Price Predictions:")
                with st.container(border=True):
                    # Check if DataFrame is valid and not empty
                    if selected_stock_analyst_targets_df is None or selected_stock_analyst_targets_df.empty:
                        # Use .empty to check if an empty df was returned and not just none
                        # Use Streamlit warning if no data is available
                        st.warning(f"No analyst price targets available for {selected_stock}.")
                    else:
                        # Create a horizontal bar chart for price targets
                        # Check if data has Metric/Value structure or direct columns
                        if 'Metric' in selected_stock_analyst_targets_df.columns and 'Value' in selected_stock_analyst_targets_df.columns:
                            # Convert Metric/Value structure to dictionary
                            targets = dict(zip(selected_stock_analyst_targets_df['Metric'], selected_stock_analyst_targets_df['Value']))
                        else:
                            # Use direct column structure
                            targets = selected_stock_analyst_targets_df.iloc[0]

                        # Extract available price data
                        price_data = []
                        colors = []

                        if 'current' in targets:
                            price_data.append(('Current', float(targets['current'])))
                            colors.append('#495057')

                        if 'low' in targets:
                            price_data.append(('Low Target', float(targets['low'])))
                            colors.append('#E74C3C')

                        if 'median' in targets:
                            price_data.append(('Median Target', float(targets['median'])))
                            colors.append('#DAA520')

                        if 'high' in targets:
                            price_data.append(('High Target', float(targets['high'])))
                            colors.append('#27AE60')

                            if price_data:
                                # Create horizontal bar chart
                                fig_bar = go.Figure()

                                labels = [item[0] for item in price_data]
                                values = [item[1] for item in price_data]

                                fig_bar.add_trace(go.Bar(
                                    x=values,
                                    y=labels,
                                    orientation='h',
                                    marker_color=colors,
                                    text=[f'${val:.2f}' for val in values],
                                    textposition='auto'
                                ))

                                fig_bar.update_layout(
                                    # title="Price Target Range",
                                    xaxis_title="Price ($)",
                                    height=400,
                                    showlegend=False,
                                    margin=dict(l=20, r=20, t=40, b=20)
                                )

                                # Add vertical line for current price if available
                                if 'current' in targets:
                                    fig_bar.add_vline(
                                        x=targets['median'],
                                        line_dash="dash",
                                        line_color="#DAA520",  # yellow
                                    )

                                st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.dataframe(selected_stock_analyst_targets_df, hide_index=True, use_container_width=True)

            with col2:

                # Widget Title
                st.write("Analyst Recommendations:")
                with st.container(border=True):
                    # ----------------- Agency Grades
                    # Check if DataFrame is valid and not empty
                    if selected_stock_analyst_recommendations_df is None or selected_stock_analyst_recommendations_df.empty:
                        st.warning(f"No analyst recommendation data available for {selected_stock}.")
                    else:
                        # Create a donut chart for recommendations
                        if 'strongBuy' in selected_stock_analyst_recommendations_df.columns:
                            recommendations = selected_stock_analyst_recommendations_df.iloc[0]

                            labels = []
                            values = []
                            colors = ['#00CC96', '#00AA66', '#FFA500', '#FF6B6B', '#DC143C']

                            rec_mapping = {
                                'strongBuy': 'Strong Buy',
                                'buy': 'Buy',
                                'hold': 'Hold',
                                'sell': 'Sell',
                                'strongSell': 'Strong Sell'
                            }

                            for key, label in rec_mapping.items():
                                if key in recommendations and recommendations[key] > 0:
                                    labels.append(label)
                                    values.append(recommendations[key])

                            if labels and values:
                                fig_donut = go.Figure(data=[go.Pie(
                                    labels=labels,
                                    values=values,
                                    hole=0.4,
                                    marker_colors=colors[:len(labels)],
                                    textinfo='label+percent',
                                    textposition='outside'
                                )])

                                fig_donut.update_layout(
                                    showlegend=True,
                                    height=400,
                                    annotations=[dict(text=f'Total<br>{sum(values)}', x=0.5, y=0.5, font_size=20, showarrow=False)]
                                )

                                st.plotly_chart(fig_donut, use_container_width=True)

                            else:
                                st.dataframe(selected_stock_analyst_recommendations_df, hide_index=True, use_container_width=True)
                        else:
                            st.dataframe(selected_stock_analyst_recommendations_df, hide_index=True, use_container_width=True)


# ---------------- PAGE CONTENT: RUN MAIN SCREEN ----------------
render_home_page_data(selected_stock)
