import pandas as pd

# Define each financial product that will be compared
data = {
    'ProductID': [f'P{i}' for i in range(1, 11)],  # Unique ID for each product
    'Product Name': [
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

financial_df['Product Avg Score'] = financial_df[score_columns].mean(axis=1)  # axis = 1 informs to calc by row, not column
financial_df['Product Avg Score'] = financial_df['Product Avg Score'].round(2)

# Set display option to show all rows and columns (# Out if you want to just print a summary of the data set)
pd.set_option('display.max_rows', None)  # None means no limit
pd.set_option('display.max_columns', None)  # None means no limit

# Check Results
print(financial_df)

# Calculate average scores
average_scores = financial_df.loc[:, 'Stability':'Inflation Hedge'].mean()

# Convert the Series to a DataFrame
metric_avg_df = average_scores.reset_index()

# Rename the columns correctly
metric_avg_df.columns = ['Metric', 'Average Score']

# Check Results
print(metric_avg_df)

# Add a df where financial products df is transposed (need for radar chart in Tableau)
melted_df = financial_df.melt(   # melt() needs a id_vars, var_name, & value_name argument
    id_vars=['ProductID', 'Product Name'],  # columns that will not be melted; id variables
                              var_name='Metric',  # new column that holds the var name (metric in this case)
                              value_name='Score'  # holds the values of the melted var (score in this case)
)  # Reshapes the DataFrame using the melt function

# Give df meaningful name
metric_scores_df = melted_df

# Check Results
print(metric_scores_df)

# Write the DataFrame to an Excel file
file_path = './financial_products.xlsx'

with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:  # 'a' for append excel sheet

    # Write the financial products DataFrame to a sheet
    financial_df.to_excel(writer, sheet_name='Financial Products', index=False)

    # Write the metric average scores DataFrame to another new sheet
    metric_avg_df.to_excel(writer, sheet_name='Metric Avg', index=False)

    # Write the metric scores transposed DataFrame to another new sheet
    metric_scores_df.to_excel(writer, sheet_name='Metric Scores', index=False)
