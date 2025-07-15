import pandas as pd
import streamlit as st
import boto3  # AWS client
import io


# Function to check filtered list of tickers from repo
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


# run check
tickers = filtered_tickers()
# print(tickers)


# Function to update csv ref file in directory
def update_ticker_list_csv(bucket_name='stock-ticker-data-bucket', csv_path='ticker_list_ref.csv'):
    """Update the reference CSV by removing obsolete tickers but preserving extra columns.

    Args:
        bucket_name (str): S3 bucket name
        csv_path (str): Path to CSV file in S3
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

    # --- Step 1: Load the current reference CSV ---
    try:
        # Get the CSV file from S3
        response = s3.get_object(Bucket=bucket_name, Key=csv_path)
        csv_content = response['Body'].read()

        # Read CSV content into pandas DataFrame
        ref_df = pd.read_csv(io.BytesIO(csv_content))
        print(f"Successfully loaded {csv_path} from S3 bucket {bucket_name}")

    # Catches the specific error when the file doesn't exist in S3
    except s3.exceptions.NoSuchKey:
        print(f"File {csv_path} not found in S3 bucket {bucket_name}. Exiting update.")
        return

    # Catches any other error that might occur during the S3 operation
    except Exception as e:
        print(f"Error reading {csv_path} from S3: {str(e)}")
        return

    # --- Step 2: Fetch NASDAQ and NYSE tickers from GitHub repo ---
    nasdaq_link = 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.json'
    nyse_link = 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.json'

    # Read NASDAQ tickers from JSON into a list
    nasdaq = pd.read_json(nasdaq_link, typ='series').tolist()

    # Read NYSE tickers from JSON into a list
    nyse = pd.read_json(nyse_link, typ='series').tolist()

    # Combine both ticker lists into one
    combined = nasdaq + nyse

    # Convert to set to remove duplicates and filter out tickers with "^" (e.g., indices or non-standard tickers)
    valid_tickers = set(ticker for ticker in combined if "^" not in ticker)

    # --- Step 3: Filter the reference DataFrame ---
    # Keep only rows where the 'Ticker' value is still in the list of valid tickers
    updated_df = ref_df[ref_df['Ticker'].isin(valid_tickers)].copy()

    # --- Step 4: Upload the cleaned data back to S3 ---
    try:
        # Convert DataFrame to CSV string
        csv_buffer = io.StringIO()
        updated_df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()

        # Upload the updated CSV back to S3
        s3.put_object(
            Bucket=bucket_name,
            Key=csv_path,
            Body=csv_string,
            ContentType='text/csv'
        )

        print(f"Updated {csv_path}: {len(ref_df)} → {len(updated_df)} tickers retained.")
        print(f"Successfully uploaded updated CSV to S3 bucket {bucket_name}")

    except Exception as e:
        print(f"Error uploading updated CSV to S3: {str(e)}")
        return

# Run update:
update_ticker_list_csv()

# -------- Log -----------
# last update: 2367 tickers (5/19/25)
# Updated ticker_list_ref.csv: 2367 → 2343 tickers retained (7/14/25)
