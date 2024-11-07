import kagglehub
import pandas as pd
import os

# Download latest version
kaggle_path = kagglehub.dataset_download("talhabu/us-regional-sales-data")
print("Path to dataset files:", kaggle_path)

# Retrieve the dataset for data wrangling
# List files in the directory to verify the contents
if os.path.isdir(kaggle_path):
    print(f"Listing contents of {kaggle_path}:")
    dataset_files = os.listdir(kaggle_path)

    # Find the CSV file in the directory
    csv_file = None
    for file in dataset_files:
        if file.endswith('.csv'):
            csv_file = os.path.join(kaggle_path, file)
            break

    if csv_file:
        print(f"CSV file found: {csv_file}")

        # Read the dataset into a pandas df
        orders_tbl = pd.read_csv(csv_file)

        # Construct a relative path based on the current working directory
        project_root = os.getcwd()  # Gets the current working directory
        sales_proj_csv = f'{project_root}/US_Regional_Sales_Data.csv'

        # add dataset back as a csv in the IDE dir to push to git for reference
        orders_tbl.to_csv(sales_proj_csv, encoding='utf-8', index=False)

        # Print the first few rows to verify
        print(orders_tbl.head())
    else:
        print("No CSV file found in the dataset directory.")
else:
    print(f"The path {kaggle_path} is not a directory.")

# View top 50 rows of the full dataset
print(orders_tbl.head(50).to_string())

# Convert df to excel for Tableau
orders_tbl.to_excel('US_Regional_Sales_Data.xlsx', sheet_name='orders_tbl')
