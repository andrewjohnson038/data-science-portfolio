# Import necessary libraries
import streamlit as st
import pandas as pd
import time
from datetime import datetime

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
