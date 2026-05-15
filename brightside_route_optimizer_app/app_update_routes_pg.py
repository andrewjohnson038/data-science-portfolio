# app_update_routes_pg.py

import streamlit as st
import pandas as pd
import boto3
from io import StringIO
import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging to file
logging.basicConfig(
    filename=f'logs/app_errors_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize session state for row deletion
if 'route_row_to_delete' not in st.session_state:
    st.session_state.route_row_to_delete = None

# AWS Configuration
aws_access_key_id = st.secrets.get("aws_access_key_id")
aws_secret_access_key = st.secrets.get("aws_secret_access_key")
aws_region = st.secrets.get("aws_region")

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# S3 bucket and file configuration
ROUTES_BUCKET = "brightside-route-optimizer"
ROUTES_FILE = "pwyc_delivery_routes_dummy_file.csv"

# Expected columns matching the routes CSV schema
ROUTES_COLUMNS = ['Name', 'Address', 'Delivery Instructions', 'Phone', 'Amount', 'Language', 'Notes']


def load_routes():
    """Load routes from S3 CSV file."""
    try:
        logger.info(f"Attempting to load routes from s3://{ROUTES_BUCKET}/{ROUTES_FILE}")
        response = s3.get_object(Bucket=ROUTES_BUCKET, Key=ROUTES_FILE)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        # Ensure all expected columns exist, add blanks if missing
        for col in ROUTES_COLUMNS:
            if col not in df.columns:
                df[col] = ''
        return df[ROUTES_COLUMNS]
    except Exception as e:
        logger.error(f"Failed to load routes from S3: {str(e)}")
        return pd.DataFrame(columns=ROUTES_COLUMNS)


def save_routes(df):
    """Save routes DataFrame to S3 CSV file."""
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=ROUTES_BUCKET,
            Key=ROUTES_FILE,
            Body=csv_buffer.getvalue()
        )
        logger.info(f"Successfully saved routes to s3://{ROUTES_BUCKET}/{ROUTES_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save routes to S3: {str(e)}")
        return False


# PAGE CONTENT
st.markdown("<h1 style='text-align: center;'>Update Routes</h1>", unsafe_allow_html=True)
st.write("---")

# --- ADD NEW ROUTE STOP ---
st.write("Add Route Stop:")

with st.form("add_route_form"):
    col1, col2 = st.columns(2)
    with col1:
        new_name = st.text_input("Name")
        new_address = st.text_input("Address")
        new_phone = st.text_input("Phone")
        new_amount = st.text_input("Amount ($)")
    with col2:
        new_instructions = st.text_area("Delivery Instructions", height=100)
        new_language = st.selectbox("Language", options=["", "English", "Spanish", "Other"])
        new_notes = st.text_input("Notes")

    submitted = st.form_submit_button("Add Stop", use_container_width=True)

    if submitted and new_address:
        current_df = load_routes()
        new_row = pd.DataFrame([{
            'Name': new_name,
            'Address': new_address,
            'Delivery Instructions': new_instructions,
            'Phone': new_phone,
            'Amount': new_amount,
            'Language': new_language,
            'Notes': new_notes
        }])
        current_df = pd.concat([current_df, new_row], ignore_index=True)
        if save_routes(current_df):
            st.success(f"Added stop: {new_address}")
            st.rerun()
        else:
            st.error("Failed to save. Please try again.")
    elif submitted and not new_address:
        st.warning("Address is required to add a stop.")

st.write("---")

# --- HANDLE DELETION (before loading display data) ---
if st.session_state.route_row_to_delete is not None:
    current_df = load_routes()
    idx = st.session_state.route_row_to_delete
    if idx < len(current_df):
        current_df = current_df.drop(idx).reset_index(drop=True)
        save_routes(current_df)
    st.session_state.route_row_to_delete = None
    st.rerun()

# --- CURRENT ROUTE STOPS ---
routes_df = load_routes()

st.write("Current Route Stops:")
st.caption(f"{len(routes_df)} stops loaded from S3")

if not routes_df.empty:
    with st.container(border=True):
        for idx, row in routes_df.iterrows():
            with st.expander(f"**{row['Name'] or 'Unknown'}** — {row['Address']}"):
                col1, col2 = st.columns(2)
                with col1:
                    edit_name = st.text_input("Name", value=str(row['Name']) if pd.notna(row['Name']) else '', key=f"name_{idx}")
                    edit_address = st.text_input("Address", value=str(row['Address']) if pd.notna(row['Address']) else '', key=f"addr_{idx}")
                    edit_phone = st.text_input("Phone", value=str(row['Phone']) if pd.notna(row['Phone']) else '', key=f"phone_{idx}")
                    edit_amount = st.text_input("Amount ($)", value=str(row['Amount']) if pd.notna(row['Amount']) else '', key=f"amount_{idx}")
                with col2:
                    edit_instructions = st.text_area("Delivery Instructions", value=str(row['Delivery Instructions']) if pd.notna(row['Delivery Instructions']) else '', key=f"instr_{idx}", height=100)
                    edit_language = st.selectbox(
                        "Language",
                        options=["", "English", "Spanish", "Other"],
                        index=["", "English", "Spanish", "Other"].index(row['Language']) if pd.notna(row['Language']) and row['Language'] in ["", "English", "Spanish", "Other"] else 0,
                        key=f"lang_{idx}"
                    )
                    edit_notes = st.text_input("Notes", value=str(row['Notes']) if pd.notna(row['Notes']) else '', key=f"notes_{idx}")

                btn_col1, btn_col2 = st.columns([3, 1])
                with btn_col1:
                    if st.button("Save Changes", key=f"save_{idx}"):
                        current_df = load_routes()
                        current_df.at[idx, 'Name'] = edit_name
                        current_df.at[idx, 'Address'] = edit_address
                        current_df.at[idx, 'Delivery Instructions'] = edit_instructions
                        current_df.at[idx, 'Phone'] = edit_phone
                        current_df.at[idx, 'Amount'] = edit_amount
                        current_df.at[idx, 'Language'] = edit_language
                        current_df.at[idx, 'Notes'] = edit_notes
                        if save_routes(current_df):
                            st.success("Stop updated!")
                            st.rerun()
                        else:
                            st.error("Failed to save. Please try again.")
                with btn_col2:
                    if st.button("Delete", key=f"delete_{idx}", type="secondary"):
                        st.session_state.route_row_to_delete = idx
                        st.rerun()
else:
    st.write("No route stops found. Add some above!")
