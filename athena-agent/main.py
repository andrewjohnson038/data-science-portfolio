# Import libraries:
import streamlit as st
import config

# Set Layout of App, Provide App Version and Provide Title
st.set_page_config()  # sets layout to wide

# Set Navigation Pages Across Files
home_pg = st.Page("home_pg.py", title="Home", icon=":material/home:")

# Set var for navigation method
pg = st.navigation([home_pg])

# Sidebar with information
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>App Info</h1>", unsafe_allow_html=True)
    st.write("---")
    st.markdown(f"**Athena Database:** {config.ATHENA_DATABASE}")

    with st.sidebar.expander("Tables & Their Schemas"):
        st.markdown("""
                        **users**
                        - `user_id`: int (primary key)  
                        - `user_name`: string  
                        - `email`: string  
                        - `signup_date`: date (use `DATE 'YYYY-MM-DD'` for comparisons)  
                        - `country`: string  
                        
                        **sales**
                        - `record_id`: int (primary key)  
                        - `user_id`: int (foreign key to `users.user_id`)  
                        - `record_date`: date (use `DATE 'YYYY-MM-DD'` for comparisons)  
                        - `sale_amount_usd`: decimal(10,2)  
                        - `product_category`: string _(values: Software, Service, Hardware)_  
                        
                        **contracts**
                        - `record_id`: int (primary key)  
                        - `user_id`: int (foreign key to `users.user_id`)  
                        - `contract_date`: date (use `DATE 'YYYY-MM-DD'` for comparisons)  
                        - `contract_value_usd`: decimal(10,2)  
                        - `contract_type`: string  
                        
                        **invoices**
                        - `record_id`: int (primary key)  
                        - `user_id`: int (foreign key to `users.user_id`)  
                        - `invoice_date`: date (use `DATE 'YYYY-MM-DD'` for comparisons)  
                        - `invoice_amount_usd`: decimal(10,2)  
                        - `payment_status`: string  
                        """)

    st.write("---")

with st.sidebar.expander("Example Questions"):
    st.markdown("""
    - Show me top 5 users by contracts in June.
    - What are the total sales by country?
    - Give me sales trends by month.
    - Who are the highest performing sellers in total contracts in 2025 ytd?
    - Show me sales data for Q1 2025 aggregated by week.
    """)

with st.sidebar.expander("Setup Required"):
    st.markdown("""
    Before using this app (Repo Fork):
    1. Configure AWS credentials
    2. Update `config.py` with your settings & customized system prompt w/ table metadata
    3. Ensure proper IAM permissions
    4. Update table schema in config & JSON file
    """)

st.sidebar.warning("NOTE: DUMMY DATA ONLY CONTAINS 2024-2025 DATA (up to 07-26-2025). THE AGENT WILL KNOW THIS, "
           "BUT TRY TO BE DESCRIPTIVE WHERE POSSIBLE :)")

# run
pg.run()
