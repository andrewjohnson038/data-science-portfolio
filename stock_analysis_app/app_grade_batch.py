import pandas as pd
import time
import boto3
import io
from datetime import datetime
import streamlit as st
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app methods
from stock_analysis_app.app_constants import DateVars
from stock_analysis_app.app_data import AppData
from stock_analysis_app.app_stock_grading_model import StockGradeModel

# Instantiate necessary helper classes
dv = DateVars()
data = AppData()
model = StockGradeModel()


class GradeBatchMethods:

    # AWS S3 Configuration

    # Check if running in GitHub Actions or locally
    if os.getenv('GITHUB_ACTIONS'):
        # Running in GitHub Actions - use environment variables
        AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        AWS_REGION = os.getenv('AWS_REGION')
    else:
        # Running locally - use Streamlit secrets
        AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
        AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
        AWS_REGION = st.secrets["AWS_REGION"]

    # Create an S3 client using the credentials from Streamlit secrets
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    bucket_name = 'stock-ticker-data-bucket'  # S3 bucket name
    input_key = 'ticker_list_ref.csv'         # S3 object key for the input ticker list
    output_key = 'ticker_grades_output.csv'   # S3 object key for the output file

    # Batch Method to retrieve tickers based on csv ref in AWS
    @staticmethod
    def run_batch(batch_size=100, rate_limit_delay=5):
        """
        Executes the batch grading of stock tickers using a custom scoring model,
        writes the results to CSV, and uploads the file back to S3.
        """

        # STEP 1: Load the ticker list from S3
        ticker_csv_obj = GradeBatchMethods.s3.get_object(
            Bucket=GradeBatchMethods.bucket_name,
            Key=GradeBatchMethods.input_key
        )
        tickers_df = pd.read_csv(ticker_csv_obj['Body'])  # Read the S3 StreamingBody as a CSV
        ticker_list = tickers_df['Ticker'].dropna().tolist()  # Extract ticker column to list

        # List to store results
        results = []
        batch_date = datetime.now().strftime("%Y-%m-%d")  # Current date for tagging results

        # STEP 2: Process tickers in batches
        for i in range(0, len(ticker_list), batch_size):
            batch = ticker_list[i:i + batch_size]  # Subset of tickers for this batch

            for ticker in batch:
                try:
                    # Load price and financial data
                    price_hist_df = data.load_price_hist_data(ticker)
                    stock_metrics_df = data.load_curr_stock_metrics(ticker)
                    move_avg_df = data.get_simple_moving_avg_data_df(price_hist_df)

                    # Calculate moving average difference
                    sma_50 = move_avg_df['50_day_SMA'].iloc[-1]
                    sma_200 = move_avg_df['200_day_SMA'].iloc[-1]
                    sma_percent_diff = ((sma_50 - sma_200) / sma_200) * 100

                    # Risk and return metrics
                    var_95, _ = data.calculate_historical_VaR(price_hist_df, time_window='yearly')
                    sharpe_ratio = data.calculate_sharpe_ratio(price_hist_df)
                    rsi_score = data.get_latest_rsi(price_hist_df)

                    # Forecasts and simulations
                    _, forecasted_df = data.get_forecasted_data_df(price_hist_df, 1)  # 1 year of data, gets latest price in forecast range
                    ind_avg_df = data.get_industry_averages_df()
                    mc_sim_df = data.get_monte_carlo_df(price_hist_df, 1000, 252)

                    # Calculate grade and score
                    score, grade, _, _, _ = StockGradeModel.calculate_grades(
                        ticker, stock_metrics_df, forecasted_df, ind_avg_df,
                        mc_sim_df, sharpe_ratio, var_95, rsi_score, sma_percent_diff
                    )

                    # Get industry if available
                    industry = "Unknown"
                    try:
                        if 'Industry' in stock_metrics_df.columns:
                            industry_data = stock_metrics_df['Industry'].iloc[0]
                        industry = industry_data if industry_data else "Unknown"
                    except:
                        pass

                    # Append result
                    results.append({
                        'Ticker': ticker,
                        'Grade': grade,
                        'Score': round(score, 3),
                        'Industry': industry,
                        'Update_Date': batch_date
                    })

                    time.sleep(0.5)  # Gentle rate-limiting between tickers

                except Exception as e:
                    # Catch individual errors to avoid halting the batch
                    print(f"Error processing {ticker}: {str(e)}")
                    results.append({
                        'Ticker': ticker,
                        'Grade': 'Error',
                        'Score': None,
                        'Industry': 'Unknown',
                        'Update_Date': batch_date
                    })

            # Delay between batches to respect rate limits / API quotas
            if i + batch_size < len(ticker_list):
                print(f"Waiting {rate_limit_delay} minutes before next batch...")
                time.sleep(rate_limit_delay * 60)

        # STEP 3: Upload results to S3 as CSV
        results_df = pd.DataFrame(results)            # Convert list of results to DataFrame
        csv_buffer = io.StringIO()                    # In-memory text buffer
        results_df.to_csv(csv_buffer, index=False)    # Write DataFrame to CSV string

        # Upload the CSV to S3
        GradeBatchMethods.s3.put_object(
            Bucket=GradeBatchMethods.bucket_name,
            Key=GradeBatchMethods.output_key,
            Body=csv_buffer.getvalue()
        )

        print(f" Output uploaded to s3://{GradeBatchMethods.bucket_name}/{GradeBatchMethods.output_key}")

    # Create a Batch Process Method to load tickers from a list for test
    @staticmethod
    def batch_process_from_list_test(ticker_list, batch_size=10, rate_limit_delay=15):
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
                    test_trained_model, test_forecasted_df = data.get_forecasted_data_df(test_price_hist_df, 1)  # 1 year forecast range
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

        print(f":) Finished batch {i//batch_size + 1}: {len(batch)} tickers processed")

        # Return Batch in dataframe
        return ticker_grades_batch_df


# Run batch script (hash out if testing)
GradeBatchMethods.run_batch()


# ---- TEST BLOCK ----
if __name__ == "__main__":

    # Test Connection
    def test_aws_connection():
        """
        Tests the AWS connection by attempting to list S3 buckets using the credentials
        currently available in the environment (e.g., from environment variables,
        .streamlit/secrets.toml, or IAM roles if running on AWS).
        """
        try:
            # Attempt to list all S3 buckets using the current boto3 client
            response = GradeBatchMethods.s3.list_buckets()

            # If the request is successful, print the confirmation and list the bucket names
            print("✅ AWS connection successful!")
            print("Available buckets:")
            for bucket in response['Buckets']:
                print(f" - {bucket['Name']}")

        # Handle specific exception for failed upload attempts (not expected here but included for completeness)
        except boto3.exceptions.S3UploadFailedError as e:
            print(f"S3 upload failed: {str(e)}")

        # Handle missing credentials (likely cause of access issues)
        except boto3.exceptions.NoCredentialsError:
            print("❌ AWS credentials not found. Make sure they are set correctly in your environment or secrets.")

        # Catch all other exceptions and print the error
        except Exception as e:
            print(f"❌ Error connecting to AWS: {str(e)}")

    # Run AWS connection test
    print("🔌 Testing AWS connection...")
    test_aws_connection()

    # Run test batch using test from list method if wanting to test by list
    # test_ticker_list = ["AAPL", "MSFT", "AMZN"]
    # print("🧪 Running test batch...")
    # test_results_df = GradeBatchMethods.batch_process_from_list_test(
    #     test_ticker_list, batch_size=6, rate_limit_delay=0
    # )
    #
    # print(test_results_df)

    # Function to test batch to AWS limiting to default 5 row test batch
    def run_batch_test_limited_rows(row_limit=5):
        """
        Test wrapper to run run_batch() on only the first `row_limit` tickers
        by monkey-patching the S3 input with a limited CSV.
        """
        # Step 1: Load the real ticker list from S3
        original_get_object = GradeBatchMethods.s3.get_object  # Save original method
        ticker_csv_obj = original_get_object(
            Bucket=GradeBatchMethods.bucket_name,
            Key=GradeBatchMethods.input_key
        )
        tickers_df = pd.read_csv(ticker_csv_obj['Body'])

        # Step 2: Slice to the first `row_limit` rows
        limited_df = tickers_df.head(row_limit)

        # Step 3: Convert to in-memory CSV buffer
        csv_buffer = io.StringIO()
        limited_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # Reset pointer to start

        # Step 4: Monkey-patch the get_object method to return a limited CSV
        def mock_get_object(Bucket, Key):
            return {'Body': io.StringIO(csv_buffer.getvalue())}

        GradeBatchMethods.s3.get_object = mock_get_object  # Patch

        # Step 5: Call the original function (which now uses the mock input)
        try:
            GradeBatchMethods.run_batch()
        finally:
            # Step 6: Restore original method so other code is unaffected
            GradeBatchMethods.s3.get_object = original_get_object


    # Run the test batch
    # run_batch_test_limited_rows(5)
