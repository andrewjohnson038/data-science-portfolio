import pandas as pd

# Define each financial product that will be compared
data = {
    'Product ID': [f'P{i}' for i in range(1, 11)],  # Unique ID for each product
    'Financial Product': [
        'Individual Securities', 'Mutual Funds', 'ETFs',
        'Government Bonds', 'High Yield Savings Account',
        'Real Estate Investment Trusts (REITs)', 'Certificates of Deposit (CDs)',
        'Corporate Bonds', 'Dividend Stocks', 'Peer-to-Peer Lending'
    ],
    'Stability': [60, 70, 65, 90, 80, 75, 85, 70, 60, 50],
    'Risk': [70, 60, 55, 20, 30, 40, 25, 50, 65, 75],
    'Potential Earnings': [85, 70, 75, 40, 25, 60, 30, 55, 80, 65],
    'Tax Friendly': [50, 65, 60, 90, 70, 55, 75, 45, 50, 40],
    'Liquidity': [80, 75, 85, 30, 90, 60, 40, 55, 70, 75],
    'Diversification': [60, 80, 70, 10, 15, 75, 25, 55, 80, 45],
    'Management Fees': [50, 60, 55, 0, 0, 25, 0, 20, 35, 10],
    'Minimum Investment': [70, 60, 75, 90, 95, 80, 85, 70, 60, 50],
    'Historical Performance': [75, 70, 65, 50, 40, 80, 45, 55, 85, 65],
    'Ease of Access': [80, 75, 85, 90, 95, 70, 80, 75, 70, 60],
    'Inflation Hedge': [50, 40, 35, 70, 20, 60, 50, 30, 65, 55]
}

# Create a DataFrame for the data set
financial_df = pd.DataFrame(data)

# Add the average score for each product and add to the dataframe
score_columns = [
    'Stability', 'Risk', 'Potential Earnings',
    'Tax Friendly', 'Liquidity', 'Diversification',
    'Management Fees', 'Minimum Investment',
    'Historical Performance', 'Ease of Access',
    'Inflation Hedge']

financial_df['Avg Score'] = financial_df[score_columns].mean(axis=1)  # axis = 1 informs to calc by row, not column
financial_df['Avg Score'] = financial_df['Avg Score'].round(2)

# Set display option to show all rows and columns (# Out if you want to just print a summary of the data set)
pd.set_option('display.max_rows', None)  # None means no limit
pd.set_option('display.max_columns', None)  # None means no limit

# Check Results
print(financial_df)

# Write the DataFrame to an Excel file
file_path = './financial_products.xlsx'
financial_df.to_excel(file_path,
                      index=False,  # indicates not to write the index #s to excel
                      sheet_name='Financial Products')
