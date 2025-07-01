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
from stock_analysis_app.app_constants import ExtraComponents
from stock_analysis_app.app_constants import GradeColors

# ---------------- SET UP DATA/MODULES ----------------
# Instantiate any imported classes here:
dv = DateVars()
animation = CSSAnimations()
data = AppData()
model = StockGradeModel()
ec = ExtraComponents()
gc = GradeColors()

# Set data vars
aws_bucket = 'stock-ticker-data-bucket'  # S3 bucket name
wl_csv = 'ticker_watchlist.csv'  # name of object in S3 bucket


# ---------------- SIDEBAR CONTENT: GET NOTES ----------------
ec.get_sidebar_notes()


# ---------------- SESSION STATE: SET WATCHLIST ----------------
# wrap grades_df in session state so data pull doesn't reload when navigating pages
if "watchlist_df" not in st.session_state:
    st.session_state["watchlist_df"] = data.load_csv_from_s3(aws_bucket, wl_csv)

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


# ---------------- PAGE CONTENT: TITLE ----------------
# Markdown title
st.markdown(
    f"""
    <div style="display: flex; justify-content: center; align-items: center;">
        <h1 style="font-size: 32px; margin: 0;">Watch List</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Divider
st.write("---")


# ---------------- PAGE CONTENT: Search Bar ----------------

# Add columns to break up page spacing
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Ticker Search with session key
    wl_ticker_search = st.text_input(
        "Search Ticker (e.g. AAPL)",
        key="wl_ticker_search"  # key Uniquely identify and persist a widget’s value in st.session_state
    )

    if wl_ticker_search.strip():  # removes spaces
        watchlist_df = watchlist_df[watchlist_df['Ticker'].str.contains(wl_ticker_search.strip(), case=False)]


# ---------------- PAGE CONTENT: WATCHLIST CARDS ----------------
# container title
# st.write("Watchlist:")

# Check if watchlist has data
if not watchlist_df.empty:

    # Create cards for each stock
    for index, row in watchlist_df.iterrows():
        ticker = row['Ticker']  # ticker per looped ticker
        industry = row['Industry']  # industry per looped ticker

        # Get current stock data
        try:
            # Fetch stock data for the looped ticker
            stock_metrics_df = data.load_curr_stock_metrics(ticker)

            # Get stock grades df
            stock_grade_df = data.load_csv_from_s3('stock-ticker-data-bucket', 'ticker_grades_output.csv')

            # Get industry avg df
            industry_avg_df = data.get_industry_averages_df()

            # Data Vars
            stock_name = stock_metrics_df['Company Name'].values[0]
            current_price = stock_metrics_df['Regular Market Price'].values[0]
            current_pe = stock_metrics_df['PE Ratio'].values[0]
            current_roe = stock_metrics_df['ROE'].values[0]
            # fiftytwo_week_range = stock_metrics_df['52-Week Range'].values[0]  # str (e.g., "88.00 - 180.00")
            stock_grade = stock_grade_df.loc[stock_grade_df['Ticker'] == ticker, 'Grade'].values[0]
            date_added = watchlist_df.loc[watchlist_df['Ticker'] == ticker, 'Date_Added'].values[0]
            price_when_added = watchlist_df.loc[watchlist_df['Ticker'] == ticker, 'Price_When_Added'].values[0]
            ind_avg_pe = industry_avg_df.loc[industry_avg_df['Industry'] == industry, 'Average P/E Ratio'].values[0]
            ind_avg_roe = industry_avg_df.loc[industry_avg_df['Industry'] == industry, 'Average ROE'].values[0]

            # Get days since added
            date_added_dt = datetime.strptime(date_added, "%Y-%m-%d")  # convert from str to datetime
            days_since_added = (dv.today - date_added_dt).days

            # Get Grade Colors
            background_color, outline_color = gc.compare_performance(stock_grade)

            # Create bordered container for each stock
            with st.container(border=True):
                # Create four columns for the card layout
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

                with col1:
                    # create 2 cols within col
                    subcol1, subcol2 = st.columns([2, 2])

                    # with sub col1
                    with subcol1:
                        st.subheader(ticker)
                        st.caption("Symbol")

                        # Add remove button
                        if st.button("Remove", key=f"remove_{ticker}"):  # give each button a key in loop
                            data.remove_from_watchlist(ticker, aws_bucket, wl_csv)
                            st.rerun()  # refresh UI

                    # Add CSS of grade to card in sub col2
                    with subcol2:
                        st.markdown(
                            f"""
                            <div style='text-align: left; padding-top: 10px; padding-bottom: 20px;'>
                                <div style="
                                    display: inline-block;
                                    padding: 20px;
                                    border-radius: 8px;
                                    font-weight: bold;
                                    font-size: 18px;
                                    color: white;
                                    background-color: {background_color};
                                    border: 2px solid {outline_color};
                                ">
                                    {stock_grade}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                kpi_height = 130  # set reusable height

                # Add kpi
                with col2:
                    with st.container(height=kpi_height):
                        st.metric(
                            label="Date Added",
                            value=f"{date_added}",
                            delta=f"{days_since_added} Days Since Added"
                        )

                    # Divider
                    st.markdown("---")

                # Add kpi
                with col3:

                    # get delta
                    col_3_price_delta = (current_price - price_when_added) / price_when_added
                    col_3_price_delta = f"{col_3_price_delta:.1%}"  # format as percentage str to 1 decimal

                    with st.container(height=kpi_height):
                        st.metric(
                            label="Current Price",
                            value=f"${current_price:.2f}",
                            delta=f"{col_3_price_delta} from Added Date"
                        )

                    # Divider
                    st.markdown("---")

                # Add kpi
                with col4:

                    # get delta
                    col_4_price_delta = current_pe - ind_avg_pe

                    with st.container(height=kpi_height):
                        st.metric(
                            label="Current PE",
                            value=f"{current_pe:.2f}",
                            delta=f"{col_4_price_delta:.2f} from Industry Avg",
                            delta_color="inverse"  # Invert the color logic
                        )

                    # Divider
                    st.markdown("---")

                # Add kpi
                with col5:

                    # get delta
                    col_5_price_delta = current_roe - ind_avg_roe

                    with st.container(height=kpi_height):
                        st.metric(
                            label="Current ROE",
                            value=f"{current_roe:.2f}",
                            delta=f"{col_5_price_delta:.2f} from Industry Avg"
                        )

                    # Divider
                    st.markdown("---")

                # Add expandable section for stock metrics
                with st.expander(f"{stock_name} ({ticker}): Metric Summary", expanded=False):

                    # pivot flattened table to show columns as metric and value
                    pivoted_metrics_df = stock_metrics_df.melt(var_name="Metric", value_name="Value")

                    # Display metrics in a clean table
                    st.dataframe(
                        pivoted_metrics_df,
                        use_container_width=True,
                        hide_index=True,
                    )

        # exception handling
        except Exception as e:
            # Error handling for individual stocks
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    # Add remove button
                    if st.button("Remove", key=f"error_remove_{ticker}"):  # give each button a key in loop
                        data.remove_from_watchlist(ticker, aws_bucket, wl_csv)
                        st.rerun()  # refresh UI
                with col2:
                    st.subheader(ticker)
                with col3:
                    st.error(f"Unable to load data for {ticker}")
else:
    # Empty watchlist message
    with st.container(border=True):
        st.info("Watchlist is empty. Add some tickers to get started!")
