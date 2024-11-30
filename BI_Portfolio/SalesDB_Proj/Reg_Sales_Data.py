import kagglehub
import pandas as pd
import os
from faker import Faker
import random

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

# Validate Data Types
print(orders_tbl.dtypes)

# The unit price is in object type, need to change to float or will agg like a string
# Need to remove the commas first to convert to numeric or will get errors
orders_tbl['Unit Price'] = orders_tbl['Unit Price'].replace({',': ''}, regex=True)  # regex= replacement pattern
orders_tbl['Unit Cost'] = orders_tbl['Unit Cost'].replace({',': ''}, regex=True)

# Convert to numeric
orders_tbl['Unit Price'] = pd.to_numeric(orders_tbl['Unit Price'])
orders_tbl['Unit Cost'] = pd.to_numeric(orders_tbl['Unit Cost'])

# Re-Validate Data Types
print(orders_tbl.dtypes)

# Convert df to excel for Tableau
orders_tbl.to_excel('US_Regional_Sales_Data.xlsx', sheet_name='orders_tbl')

# Add in ref tables for customers, products, and sales teams
customers_tbl = orders_tbl[['_CustomerID']].drop_duplicates().reset_index(drop=True)  # drop=true creates a new index
products_tbl = orders_tbl[['_ProductID']].drop_duplicates().reset_index(drop=True)
salesteam_tbl = orders_tbl[['_SalesTeamID']].drop_duplicates().reset_index(drop=True)

# Initialize Faker instance
fake = Faker()

# Add in fake customer data with the Faker package
def gen_fake_customer_contact_data():
    return {
        'company_name': fake.company(),
        'email': fake.email(),
        'address': fake.address().replace('\n', ', '),  # Make address a single line
        'state': fake.state(),
        'zip_code': fake.zipcode()
    }

# Add in fake sales team data with the Faker package
def gen_fake_sales_team_data():
    return {
        'team_name': fake.last_name(),
        'members': [fake.name() for _ in range(4)]  # Generate 4 team member names
    }

# Add in fake product data with the Faker package
def gen_fake_product_data():
    product_adjectives = ["Wide", "Ultra", "Mega", "Pro", "Elite", "Max", "Sleek Ultra"]
    product_type = ["ePhone", "Pants", "Basketball", "Hat", "White T-Shirt", "Black E-pods", "Lulu Leggings"]

    adjective = random.choice(product_adjectives)  # Randomly choose an adjective
    type = random.choice(product_type)  # Randomly choose a product option

    return {
        'product_name': f"{adjective} {type}"
    }

# Apply the fake data generation function to each row in the customers_tbl
customer_fake_data = [gen_fake_customer_contact_data() for _ in range(len(customers_tbl))]
salesteam_fake_data = [gen_fake_sales_team_data() for _ in range(len(salesteam_tbl))]
products_fake_data = [gen_fake_product_data() for _ in range(len(products_tbl))]

# ^ loops through each row, len() counts the number of rows in the list object

# Convert the list of dictionaries to a DataFrame
customer_fake_df = pd.DataFrame(customer_fake_data)
salesteam_fake_df = pd.DataFrame(salesteam_fake_data)
products_fake_df = pd.DataFrame(products_fake_data)

# Concatenate the new data with the existing customers_tbl
customers_tbl = pd.concat([customers_tbl, customer_fake_df], axis=1)  # axis=1 adds new fields as columns, not additional rows
salesteam_tbl = pd.concat([salesteam_tbl, salesteam_fake_df], axis=1)
products_tbl = pd.concat([products_tbl, products_fake_df], axis=1)

# Display the new customers_tbl DataFrame
print(customers_tbl, salesteam_tbl, products_tbl)

# need to unflatten team members in the sales table
salesteam_tbl = salesteam_tbl.explode('members', ignore_index=True)

# validate transformation
print(salesteam_tbl)


# Add tables as new tab in Excel sheet
with pd.ExcelWriter('US_Regional_Sales_Data.xlsx', engine='openpyxl', mode='a') as writer:
    customers_tbl.to_excel(writer, sheet_name='customers_tbl', index=True)
    salesteam_tbl.to_excel(writer, sheet_name='salesteam_tbl', index=True)
    products_tbl.to_excel(writer, sheet_name='products_tbl', index=True)

# 'mode='a'' allows you to append to an existing Excel file without overwriting it.



# Create a view from the tables with total sales by state for a map figure in Tableau
salesbystate_vw = pd.merge(orders_tbl, customers_tbl, on='_CustomerID', how='left')

# Group by each state and aggregate the total sales
salesbystate_vw = salesbystate_vw.groupby('state')[('Unit Price')].sum().reset_index()

# Rename price column to total sales
salesbystate_vw.rename(columns={'Unit Price': 'total_sales'}, inplace=True)

# Validate new set
print(salesbystate_vw)

# There are some states missing from the list. Need to add them in with 0 total_sales
all_states = [  # create list with all states
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
    'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
    'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
    'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
    'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
    'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
]

# Create a list of the missing states by comparing to all states
existing_states = salesbystate_vw['state'].tolist()  # converts to list
missing_states = [state for state in all_states if state not in existing_states]  # takes state if not in the orig df

# Create a list for missing sales and assign 0 to each row
missing_sales = [0] * len(missing_states)  # len (length) of list is equal to # of missing states

# Add both lists into a df for the missing states
missing_states_temp_tbl = pd.DataFrame({'state': missing_states, 'total_sales': missing_sales})

# Append the missing rows to the original DataFrame (salesbystate_vw)
salesbystate_vw = pd.concat([salesbystate_vw, missing_states_temp_tbl], ignore_index=True)  # adds states after last row

# Reset the index and sort by the state names
salesbystate_vw = salesbystate_vw.sort_values(by='state').reset_index(drop=True)

# Validate changes
print(salesbystate_vw)

# Add salesbystate_vw as new tab in Excel sheet
with pd.ExcelWriter('US_Regional_Sales_Data.xlsx', engine='openpyxl', mode='a') as writer:
    salesbystate_vw.to_excel(writer, sheet_name='salesbystate_vw', index=True)
