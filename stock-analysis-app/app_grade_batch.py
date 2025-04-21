# Import necessary libraries
import streamlit as st
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

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

# for alpha vantage api
alpha_vantage_key = st.secrets.get("Alpha_Vantage_API_Key")

# Class for data dfs/variables used across the app
class GradeBatchMethods:

    # Create a Batch Process Method to load tickers to a dataframe in App
    @staticmethod
    @st.cache_data(ttl=3600 * 24 * 7)  # Cache data for 7 days (1 week)
    def batch_process_tickers(ticker_list, batch_size=400, rate_limit_delay=15):
        """
        Process tickers in batches to stay under YFinance rate limits.

        Parameters:
        ticker_list (list): List of ticker symbols to process
        batch_size (int): Maximum number of API calls per batch (default 250 to be safe)
        rate_limit_delay (int): Minutes to wait between batches

        Returns:
        DataFrame with ticker grades
        """
        results = []
        batch_date = datetime.now().strftime("%Y-%m-%d")

        # Split tickers into batches
        for i in range(0, len(ticker_list), batch_size):
            batch = ticker_list[i:i+batch_size]

            # Initialize progress bar
            progress = st.progress(0)
            batch_length = len(batch)

            # Process each ticker in the current batch
            for j, ticker in enumerate(batch):
                try:
                    # Set variables needed for calculate_grades
                    test_price_hist_df = data.load_price_hist_data(ticker)
                    test_stock_metrics_df = data.load_curr_stock_metrics(ticker)
                    test_move_avg_data_df = data.get_simple_moving_avg_data_df(test_price_hist_df)

                    # Get VaR
                    test_hist_yearly_VaR_95, hist_yearly_VaR_95_dollars = data.calculate_historical_VaR(test_price_hist_df,
                                                                                                        time_window='yearly')

                    # Get SMA percent difference
                    sma_50 = test_move_avg_data_df['50_day_SMA'].iloc[-1]  # Latest 50-day SMA
                    sma_200 = test_move_avg_data_df['200_day_SMA'].iloc[-1]  # Latest 200-day SMA

                    # Calculate the difference between 50-day and 200-day SMA
                    test_sma_price_difference = sma_50 - sma_200

                    # Calculate the percentage difference between the 50-day and 200-day SMA
                    test_sma_percentage_difference = (test_sma_price_difference / sma_200) * 100

                    # Set Up Test Variables
                    test_trained_model, test_forecasted_df = data.get_forecasted_data_df(test_price_hist_df, 5)  # 5 year forecast range
                    test_ind_avg_df = data.get_industry_averages_df()
                    test_mc_sim_df = data.get_monte_carlo_df(test_price_hist_df, 1000, 252)
                    test_sharpe_ratio = data.calculate_sharpe_ratio(test_price_hist_df)
                    test_var_score = test_hist_yearly_VaR_95
                    test_rsi_score = data.get_latest_rsi(test_price_hist_df)
                    test_sma_percent_diff = test_sma_percentage_difference

                    # Calculate grades for the selected stock
                    score, grade, grade_color_background, grade_color_outline, score_details = StockGradeModel.calculate_grades(
                        ticker, test_stock_metrics_df, test_forecasted_df, test_ind_avg_df, test_mc_sim_df, test_sharpe_ratio,
                        test_var_score, test_rsi_score, test_sma_percent_diff)

                    # Get industry from stock data
                    # Get industry from stock data
                    industry = "Unknown"
                    try:
                        # Access industry data from the DataFrame
                        if 'Industry' in test_stock_metrics_df.columns:
                            industry_data = test_stock_metrics_df['Industry'].iloc[0]
                        industry = industry_data if industry_data else "Unknown"

                    except:
                        pass

                    # Store results
                    results.append({
                        'Ticker': ticker,
                        'Grade': grade,
                        'Score': round(score, 3),
                        'Industry': industry,
                        'Update_Date': batch_date
                    })

                    # Update progress bar
                    progress.progress((i + j + 1) / len(ticker_list))

                    # Small delay between API calls to respect rate limits
                    time.sleep(0.5)

                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
                    results.append({
                        'Ticker': ticker,
                        'Grade': 'Error',
                        'Score': None,
                        'Industry': 'Unknown',
                        'Update_Date': batch_date
                    })

            # Wait to respect rate limits before the next batch
            if i + batch_size < len(ticker_list):
                print(f"Waiting {rate_limit_delay} minutes before next batch...")
                time.sleep(rate_limit_delay * 60)  # wait 15 minutes (60 * 15) until next batch load

        # Convert results to DataFrame
        ticker_grades_batch_df = pd.DataFrame(results)

        # Return Batch in dataframe
        return ticker_grades_batch_df

    # Create a Process Method to update the batch df in app
    @staticmethod
    def load_or_update_grades(ticker_grades_batch_df, ticker_list, force_update=False):
        """
        Load grades from data frame or update them if needed.

        Parameters:
        ticker_grades_batch_df (DataFrame): Dataframe of the batched tickers
        ticker_list (list): List of tickers to process
        force_update (bool): If True, force update regardless of last update date

        Returns:
        DataFrame with updated ticker grades on new batch date after load
        """

        # # Use pathlib to dynamically get the path relative to the script location
        # stock_grades_csv_path = Path(__file__).resolve().parent / "ticker_grades.csv"

        # Directly use the passed DataFrame
        update_needed = force_update

        if not update_needed and not ticker_grades_batch_df.empty:
            last_update = datetime.strptime(ticker_grades_batch_df['Update_Date'].iloc[0], "%Y-%m-%d")
            days_since_update = (datetime.now() - last_update).days

            # Update if more than 7 days since last update
            if days_since_update >= 7:
                update_needed = True
        else:
            update_needed = True

        if update_needed:
            st.info("Updating all stock grades - this will take several hours. The app will be unavailable during this time :).")
            grades_df = GradeBatchMethods.batch_process_tickers(ticker_list)

            # Return updated DataFrame
            return grades_df
        else:
            return ticker_grades_batch_df

    # Method that adds a grade section to the app
    @staticmethod
    def add_ticker_grades_section(ticker_list, container=None):
        """
        Add a ticker grades section to your Streamlit app

        Parameters:
        ticker_list (list): List of Tickers to process
        container (streamlit.container, optional): Container to render elements in. Defaults to None (global st).
        """
        # Use the passed container if provided, otherwise use global st
        st = container if container is not None else container

        # Create an empty DataFrame as fallback
        empty_df = pd.DataFrame(columns=['Ticker', 'Grade', 'Score', 'Industry', 'Update_Date'])

        # Try to load existing grades if available
        try:
            # This is a placeholder - in a real app, you'd load from a database or file
            existing_grades_df = empty_df

            # Check if update is needed (7 days passed)
            update_needed = False
            if not existing_grades_df.empty:
                last_update = datetime.strptime(existing_grades_df['Update_Date'].iloc[0], "%Y-%m-%d")
                days_since_update = (datetime.now() - last_update).days
                if days_since_update >= 7:
                    update_needed = True
                    st.warning(f"Grades are {days_since_update} days old. Consider updating for the latest data.")
            else:
                update_needed = True
                st.warning("No grade data found. Please update to generate stock grades.")

            # Add update button
            col1, col2 = st.columns([1, 3])
            with col1:
                update_clicked = st.button("Update Stock Grades", type="primary" if update_needed else "secondary")

            with col2:
                if not existing_grades_df.empty:
                    last_update = existing_grades_df['Update_Date'].iloc[0]
                    st.info(f"Grades last updated on: {last_update}")

            # If button is clicked, update the grades
            if update_clicked:
                with st.spinner("Updating stock grades... This may take several hours depending on the number of stocks."):
                    grades_df = GradeBatchMethods.batch_process_tickers(ticker_list)
                    st.success("Stock grades successfully updated!")
                    # Here you would typically save the updated grades to a database or file
            else:
                # Use existing grades if available and no update requested
                grades_df = existing_grades_df

        except Exception as e:
            st.error(f"Error loading grades: {str(e)}")
            grades_df = empty_df

        # Don't display anything else if we have no data and user hasn't clicked update
        if grades_df.empty:
            return

        # Add filters
        col1, col2, col3 = st.columns(3)

        with col1:
            # Filter by grade
            all_grades = ['All'] + sorted(grades_df['Grade'].unique().tolist())
            selected_grade = st.selectbox("Filter by Grade", all_grades)

        with col2:
            # Filter by industry
            all_industries = ['All'] + sorted(grades_df['Industry'].unique().tolist())
            selected_industry = st.selectbox("Filter by Industry", all_industries)

        with col3:
            # Sort options
            sort_options = ['Grade (Best First)', 'Grade (Worst First)', 'Score (High to Low)', 'Ticker (A-Z)']
            sort_selection = st.selectbox("Sort by", sort_options)

        # Apply filters
        filtered_df = grades_df.copy()

        if selected_grade != 'All':
            filtered_df = filtered_df[filtered_df['Grade'] == selected_grade]

        if selected_industry != 'All':
            filtered_df = filtered_df[filtered_df['Industry'] == selected_industry]

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

        # Display the styled dataframe
        st.dataframe(filtered_df.style.applymap(color_grades, subset=['Grade']), use_container_width=True)

        # Show grade distribution
        st.subheader("Grade Distribution")
        grade_counts = grades_df['Grade'].value_counts().sort_index()
        st.bar_chart(grade_counts)

# Add the ticker grades section to the app (use code below)
# GradeBatchMethods.add_ticker_grades_section(ticker_list)


# Any blocks written under here can use for testing directly from this file w/o importing the code below to the main
# (this is a built-in syntax of python)
if __name__ == "__main__":

    # Test Ticker List Batch
    test_ticker_list = [
        "AAPL", "MSFT", "AMZN"]

    # Run batch process directly
    print("Running test batch...")
    test_results_df = GradeBatchMethods.batch_process_tickers(test_ticker_list, batch_size=6, rate_limit_delay=0)

    # Print test results
    print(test_results_df)
