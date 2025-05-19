import pandas as pd


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
def update_ticker_list_csv(csv_path='ticker_list_ref.csv'):
    """Update the reference CSV by removing obsolete tickers but preserving extra columns."""

    # --- Step 1: Load the current reference CSV ---
    try:
        ref_df = pd.read_csv(csv_path)  # Read the existing CSV containing tickers and additional metadata
    except FileNotFoundError:
        print(f"File {csv_path} not found. Exiting update.")  # Handle case where file doesn't exist
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

    # --- Step 4: Overwrite the CSV with the cleaned data ---
    updated_df.to_csv(csv_path, index=False)  # Save the updated DataFrame back to the same CSV file
    print(f"Updated {csv_path}: {len(ref_df)} → {len(updated_df)} tickers retained.")  # Log how many tickers were kept


# Run update:
update_ticker_list_csv()


# -------- Log -----------
# last update: 2367 tickers (5/19/25)
