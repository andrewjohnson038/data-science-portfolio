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

# Initialize session state
if 'member_to_delete_index' not in st.session_state:
    st.session_state.member_to_delete_index = None

# AWS Configuration (assuming secrets are accessible)
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
TEAM_MEMBERS_BUCKET = "brightside-route-optimizer"
TEAM_MEMBERS_FILE = "brightside_team_members_dummy_file.csv"


def load_team_members():
    """Load team members from S3 CSV file."""
    try:
        logger.info(f"Attempting to load team members from s3://{TEAM_MEMBERS_BUCKET}/{TEAM_MEMBERS_FILE}")
        response = s3.get_object(Bucket=TEAM_MEMBERS_BUCKET, Key=TEAM_MEMBERS_FILE)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        return df
    except Exception as e:
        # Log the error to file
        logger.error(f"Failed to load team members from S3: {str(e)}")
        # Return empty DataFrame if S3 load fails
        return pd.DataFrame(columns=['name', 'email'])


def save_team_members(df):
    """Save team members DataFrame to S3 CSV file."""
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=TEAM_MEMBERS_BUCKET,
            Key=TEAM_MEMBERS_FILE,
            Body=csv_buffer.getvalue()
        )
        logger.info(f"Successfully saved team members to s3://{TEAM_MEMBERS_BUCKET}/{TEAM_MEMBERS_FILE}")
        return True
    except Exception as e:
        # Log the error to file
        logger.error(f"Failed to save team members to S3: {str(e)}")
        return False


# PAGE CONTENT
st.markdown("<h1 style='text-align: center;'>Add Team Members</h1>", unsafe_allow_html=True)
st.write("---")

# Add Section Title
st.write("Add Members:")

# Add new member form
with st.form("add_member_form"):
    new_name = st.text_input("Name")
    new_email = st.text_input("Email")
    submitted = st.form_submit_button("Add Member")

    if submitted and new_name and new_email:
        # Load fresh data from S3
        current_members_df = load_team_members()
        # Add new member to DataFrame
        new_member = pd.DataFrame({'name': [new_name], 'email': [new_email]})
        current_members_df = pd.concat([current_members_df, new_member], ignore_index=True)
        save_team_members(current_members_df)
        st.success(f"Added {new_name} to team members!")
        st.rerun()

# Handle member deletion FIRST (before loading for display)
if st.session_state.member_to_delete_index is not None:
    # Load fresh data from S3
    current_members_df = load_team_members()
    index_to_delete = st.session_state.member_to_delete_index
    if index_to_delete < len(current_members_df):
        current_members_df = current_members_df.drop(index_to_delete).reset_index(drop=True)
        save_team_members(current_members_df)
        st.session_state.member_to_delete_index = None
        st.rerun()

# Load current team members for display (AFTER operations)
current_members_df = load_team_members()

# Display current members with delete option
if not current_members_df.empty:
    st.write("Current Team Members:")
    with st.container(border=True):
        for idx, row in current_members_df.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{row['name']} ({row['email']})")
            with col2:
                if st.button("Delete", key=f"delete_{idx}"):
                    st.session_state.member_to_delete_index = idx
                    st.rerun()
else:
    st.write("No team members found. Add some above!")
