import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime
import boto3
import plotly.express as px
from plotly import graph_objs as go  # Import plotly for time series visuals

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
pf_csv = 'portfolio_ticker_list.csv'  # name of object in S3 bucket


# ---------------- SIDEBAR CONTENT: GET NOTES ----------------
ec.get_sidebar_notes()


# ---------------- SESSION STATE: SET PORTFOLIO LIST ----------------
# wrap grades_df in session state so data pull doesn't reload when navigating pages
if "portfolio_df" not in st.session_state:
    st.session_state["portfolio_df"] = data.load_csv_from_s3(aws_bucket, pf_csv)

# assign to session state var
portfolio_df = st.session_state["portfolio_df"]

# Check if the portfolio list has been updated
if st.session_state.get("portfolio_updated", False):
    # Reload data
    portfolio_df = data.load_csv_from_s3(aws_bucket, pf_csv)

    # Reset the flag so it doesn't reload unnecessarily
    st.session_state.portfolio_updated = False
else:
    # Load once or from cache
    portfolio_df = data.load_csv_from_s3(aws_bucket, pf_csv)

if portfolio_df.empty:

    # Empty watchlist message
    with st.container(border=False):
        st.info("Portfolio is empty. Add some tickers to get started!")
        st.stop()

# ---------------- DATA: GET ADDITIONAL DATA ----------------

# Get additional data if not empty
else:
    # Prepare containers for new columns
    stock_names = []
    current_prices = []

    # Create cards for each stock
    for index, row in portfolio_df.iterrows():
        ticker = row['Ticker']  # ticker per looped ticker
        industry = row['Industry']  # industry per looped ticker

        # Get current stock data
        try:
            # Fetch stock data for the looped ticker
            stock_metrics_df = data.load_curr_stock_metrics(ticker)

            # Data Vars
            stock_name = (
                stock_metrics_df['Company Name'].values[0]
                if 'Company Name' in stock_metrics_df.columns and not stock_metrics_df['Company Name'].empty
                else "N/A"
            )

            current_price = (
                stock_metrics_df['Regular Market Price'].values[0]
                if 'Regular Market Price' in stock_metrics_df.columns and not stock_metrics_df['Regular Market Price'].empty
                else None
            )

            # Add to lists
            stock_names.append(stock_name)
            current_prices.append(current_price)

        # exception handling
        except Exception as e:
            # Error handling for individual stocks
            st.error(f"Unable to load additional data for {ticker}")

    # Update the portfolio DataFrame with new columns
    portfolio_df['Stock_Name'] = stock_names
    portfolio_df['Current_Price'] = current_prices

    # Calculate total value at time of purchase
    portfolio_df['Total_Value_When_Added'] = (portfolio_df['Price_When_Added'] * portfolio_df['Amount']).round(2)

    # Calculate total current value
    portfolio_df['Total_Current_Value'] = (portfolio_df['Current_Price'] * portfolio_df['Amount']).round(2)

    # Calculate percentage change in value
    portfolio_df['Value_%_Change'] = (
            ((portfolio_df['Total_Current_Value'] - portfolio_df['Total_Value_When_Added']) / portfolio_df['Total_Value_When_Added']) * 100
    ).round(2)

    # Calculate total earnings (profit or loss in dollars)
    portfolio_df['Total_Earnings'] = (portfolio_df['Total_Current_Value'] - portfolio_df['Total_Value_When_Added']).round(2)

# ---------------- PAGE CONTENT: TITLE ----------------
# Markdown title
st.markdown(
    f"""
    <div style="display: flex; justify-content: center; align-items: center;">
        <h1 style="font-size: 32px; margin: 0;">Portfolio</h1>
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
    pf_ticker_search = st.text_input(
        "Search Ticker (e.g. AAPL)",
        key="pf_ticker_search"  # key Uniquely identify and persist a widget’s value in st.session_state
    )

    if pf_ticker_search.strip():  # removes spaces
        portfolio_df = portfolio_df[portfolio_df['Ticker'].str.contains(pf_ticker_search.strip(), case=False)]


# ---------------- PAGE CONTENT: PORTFOLIO SUMMARY ----------------
# container title
st.write("Portfolio Summary:")

with st.container(border=True):

    # Add columns to break up page spacing
    col1, col2 = st.columns([1, 2])

    # Pie Chart with Holdings
    with col1:
        # Create custom hover text (right aligned)
        portfolio_df['Hover_Text'] = (
                "Ticker: " + portfolio_df['Ticker'] + "<br>" +
                "Shares: " + portfolio_df['Amount'].astype(str) + "<br>" +
                "Total Value: $" + portfolio_df['Total_Current_Value'].round(2).astype(str) + "<br>" +
                "Date Bought: " + portfolio_df['Date_Added']
        )

        # chart color sequence
        colors = px.colors.qualitative.Set2

        # Create pie chart
        fig = px.pie(
            portfolio_df,
            names='Ticker',
            values='Total_Current_Value',
            custom_data=['Hover_Text'],
            color_discrete_sequence=colors  # Apply red sequence
        )

        # Update tooltip
        fig.update_traces(
            hovertemplate='%{customdata[0]}<extra></extra>'
        )
        # %{customdata[0]} pulls the first item from the custom_data list
        # In this case, it displays the 'hover_text' column content
        # <extra></extra> removes the default "trace name" or legend info
        # that would normally appear in the tooltip (e.g., "● TickerName")

        # Display chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    # DataFrame Summary
    with col2:

        st.write("")

        # Reorder and rename columns for display
        display_df = portfolio_df[[
            'Ticker',
            'Industry',
            'Date_Added',
            'Amount',
            'Price_When_Added',
            'Current_Price',
            'Total_Current_Value',
            'Value_%_Change'   # Add earnings percent here
        ]].rename(columns={
            'Amount': 'Shares',
            'Price_When_Added': 'Buy Price',
            'Current_Price': 'Current Price',
            'Total_Current_Value': 'Total Share Value',
            'Value_%_Change': 'Earnings %',
            'Date_Added': 'Date Added'
        })

        # Calculate totals for the portfolio
        total_shares = portfolio_df['Amount'].sum()
        total_current_value = portfolio_df['Total_Current_Value'].sum()
        total_original_value = (portfolio_df['Amount'] * portfolio_df['Price_When_Added']).sum()
        overall_earnings_percent = ((total_current_value - total_original_value) / total_original_value) * 100
        total_earnings_dollar = total_current_value - total_original_value

        # Create three equal columns for metrics
        col1, col2 = st.columns([1, 2])

        with col1:
            with st.container(border=True, height=127):
                st.metric(
                    label="Total Portfolio Value",
                    value=f"${total_current_value:,.2f}",
                    delta=f"{overall_earnings_percent:.2f}%",
                )

        with col2:
            with st.container(border=True, height=127):
                # Calculate the total gain/loss for the entire portfolio
                total_gain_loss = total_current_value - total_original_value

                # Create a simple timeline based on dates in your portfolio
                dates = sorted(portfolio_df['Date_Added'].unique())

                # For each date, calculate cumulative investment and current value
                timeline_data = []
                for date in dates:
                    # Get all investments up to this date
                    investments_to_date = portfolio_df[portfolio_df['Date_Added'] <= date]

                    # Calculate original investment to this date
                    original_to_date = (investments_to_date['Amount'] * investments_to_date['Price_When_Added']).sum()

                    # Calculate current value to this date
                    current_to_date = investments_to_date['Total_Current_Value'].sum()

                    # Calculate gain/loss
                    gain_loss = current_to_date - original_to_date

                    timeline_data.append({
                        'Date': date,
                        'Gain_Loss': gain_loss
                    })

                timeline_df = pd.DataFrame(timeline_data)

                # Determine trend direction based on final gain/loss
                trend_direction = "up" if timeline_df['Gain_Loss'].iloc[-1] > 0 else "down"
                trend_color = 'rgba(0, 177, 64, .8)' if trend_direction == "up" else 'rgba(244, 67, 54, 0.8)'
                trend_fill = 'rgba(0, 177, 64, 0.2)' if trend_direction == "up" else 'rgba(244, 67, 54, 0.2)'

                def plot_portfolio_growth():
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=timeline_df['Date'],
                        y=timeline_df['Gain_Loss'],
                        name='Portfolio Gain/Loss',
                        fill='tozeroy',
                        line=dict(color=trend_color),
                        fillcolor=trend_fill
                    ))

                    fig.layout.update(
                        template='plotly_white',
                        height=105,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(
                            showgrid=False,
                            showticklabels=False,
                            zeroline=False
                        ),
                        yaxis=dict(
                            showgrid=False,
                            showticklabels=True,
                            zeroline=True
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                plot_portfolio_growth()

        st.write("---")

        # Sort display_df by 'Current Value' descending
        display_df_sorted = display_df.sort_values(by='Total Share Value', ascending=False)

        # Display sorted dataframe without index
        st.dataframe(display_df_sorted, hide_index=True)


# ---------------- PAGE CONTENT: PORTFOLIO LIST CARDS ----------------

# Title
st.write("Ticker Breakdown:")

# Check if portfolio list has data
if not portfolio_df.empty:

    # Create cards for each stock
    for index, row in portfolio_df.iterrows():
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
            date_added = portfolio_df.loc[portfolio_df['Ticker'] == ticker, 'Date_Added'].values[0]
            price_when_added = portfolio_df.loc[portfolio_df['Ticker'] == ticker, 'Price_When_Added'].values[0]
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
                            data.remove_ticker_from_csv(ticker, aws_bucket, pf_csv)
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

                kpi_height = 133  # set reusable height

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
                        data.remove_ticker_from_csv(ticker, aws_bucket, pf_csv)
                        st.rerun()  # refresh UI
                with col2:
                    st.subheader(ticker)
                with col3:
                    st.error(f"Unable to load data for {ticker}")
