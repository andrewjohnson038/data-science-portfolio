import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime
import boto3
from io import StringIO

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app methods
from stock_analysis_app.app_constants import DateVars
from stock_analysis_app.app_data import AppData
from stock_analysis_app.app_animations import CSSAnimations
from stock_analysis_app.app_stock_grading_model import StockGradeModel
from stock_analysis_app.app_constants import ExtraComponents


# ---------------- SET UP DATA/MODULES ----------------
# Instantiate any imported classes here:
dv = DateVars()
animation = CSSAnimations()
data = AppData()
model = StockGradeModel()
ec = ExtraComponents()

# Set Data Vars
bucket_name = 'stock-ticker-data-bucket'  # S3 bucket name
csv_path = 'ticker_grades_output.csv'  # name of object in S3 bucket


# ---------------- SIDEBAR CONTENT: GET NOTES ----------------
ec.get_sidebar_notes()


# ---------------- SESSION STATE: SET GRADES DF ----------------
# wrap grades_df in session state so data pull doesn't reload when navigating pages
if "grades_df" not in st.session_state:
    st.session_state["grades_df"] = data.load_csv_from_s3(bucket_name, csv_path)

# assign to session state var
grades_df = st.session_state["grades_df"]


# ---------------- PAGE CONTENT: TITLE ----------------
# Provide Page Title
st.markdown(
    f"""
                    <div style="display: flex; justify-content: center; align-items: center;">
                        <h1 style="font-size: 32px; margin: 0;">Stock Grades</h1>
                    </div>
                    """,
    unsafe_allow_html=True
)

# Add a divider
st.write("---")

# Create Container
sh_g = st.container()


# ---------------- PAGE CONTENT: RENDER PAGE ----------------
# Ticker Search Section Title
sh_g.write("Ticker Grade Search:")

# Write Grades Search to App
with sh_g.container():

    # Add a section for ticker grades
    def add_ticker_grades_section(grades_df):
        """
        Add a ticker grades section to your Streamlit app using the provided DataFrame.

        Parameters:
        grades_df (DataFrame): DataFrame containing stock grades and other data
        """

        # Check if grades_df is empty
        if grades_df.empty:
            st.error("No grades data available.")
            return

        # Add filters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Ticker Search with session key
            ticker_search = st.text_input(
                "Search Ticker (e.g. AAPL)",
                key="ticker_search"  # key Uniquely identify and persist a widget’s value in st.session_state
            )

        with col2:
            # Grade Filter with session key
            all_grades = ['All'] + sorted(grades_df['Grade'].unique().tolist())
            selected_grade = st.selectbox(
                "Filter by Grade",
                options=all_grades,
                key="selected_grade"
            )

        with col3:
            # Industry Filter with session key
            all_industries = ['All'] + sorted(grades_df['Industry'].unique().tolist())
            selected_industry = st.selectbox(
                "Filter by Industry",
                options=all_industries,
                key="selected_industry"
            )

        with col4:
            # Sort Option with session key
            sort_grade_options = ['Score (High to Low)', 'Grade (Best First)', 'Ticker (A-Z)', 'Grade (Worst First)']
            sort_selection = st.selectbox(
                "Sort by",
                options=sort_grade_options,
                key="sort_grade_options"
            )

        # Apply filters
        filtered_df = grades_df.copy()

        if selected_grade != 'All':
            filtered_df = filtered_df[filtered_df['Grade'] == selected_grade]

        if selected_industry != 'All':
            filtered_df = filtered_df[filtered_df['Industry'] == selected_industry]

        if ticker_search.strip():  # removes spaces
            filtered_df = filtered_df[filtered_df['Ticker'].str.contains(ticker_search.strip(), case=False)]

        # Apply sorting
        if sort_selection == 'Grade (Best First)':
            grade_order = {'S': 0, 'A': 1, 'A-': 2, 'B+': 3, 'B': 4, 'B-': 5, 'C+': 6, 'C': 7, 'C-': 8,
                           'D+': 9, 'D': 10, 'D-': 11, 'F': 12, 'Error': 13}
            filtered_df['grade_order'] = filtered_df['Grade'].map(grade_order)
            filtered_df = filtered_df.sort_values('grade_order')
            filtered_df = filtered_df.drop(columns=['grade_order'])
        elif sort_selection == 'Grade (Worst First)':
            grade_order = {'S': 0, 'A': 1, 'A-': 2, 'B+': 3, 'B': 4, 'B-': 5, 'C+': 6, 'C': 7, 'C-': 8,
                           'D+': 9, 'D': 10, 'D-': 11, 'F': 12, 'Error': 13}
            filtered_df['grade_order'] = filtered_df['Grade'].map(grade_order)
            filtered_df = filtered_df.sort_values('grade_order', ascending=False)
            filtered_df = filtered_df.drop(columns=['grade_order'])
        elif sort_selection == 'Score (High to Low)':
            filtered_df = filtered_df.sort_values('Score', ascending=False)
        else:  # Ticker (A-Z)
            filtered_df = filtered_df.sort_values('Ticker')

        st.write("---")

        # Define a function to color the grades
        def color_grades(val):
            grade_colors = {
                "S": "background-color: rgba(255, 215, 0, 0.5)",  # Gold
                "A": "background-color: rgba(34, 139, 34, 0.5)",  # Forest green
                "A-": "background-color: rgba(50, 205, 50, 0.5)",  # Lime green
                "B+": "background-color: rgba(60, 179, 113, 0.5)",  # Medium sea green
                "B": "background-color: rgba(102, 205, 170, 0.5)",  # Medium aquamarine
                "B-": "background-color: rgba(152, 251, 152, 0.5)",  # Pale green
                "C+": "background-color: rgba(173, 255, 47, 0.5)",  # Green yellow
                "C": "background-color: rgba(252, 226, 5, 0.5)",  # Bumblebee
                "C-": "background-color: rgba(255, 165, 0, 0.5)",  # Orange
                "D+": "background-color: rgba(255, 140, 0, 0.5)",  # Dark orange
                "D": "background-color: rgba(255, 69, 0, 0.5)",  # Orange red
                "D-": "background-color: rgba(255, 99, 71, 0.5)",  # Tomato
                "F": "background-color: rgba(255, 0, 0, 0.5)",  # Red
                "Error": "background-color: rgba(128, 128, 128, 0.5)"  # Gray for errors
            }

            if val in grade_colors:
                return grade_colors[val]
            return ""

        st.write("Ticker Model Grades / Scores:")
        with st.container(border=True):
            # Style and format the df for visual
            ticker_grades_df = filtered_df.style \
                .format({'Score': '{:.2f}'}) \
                .map(color_grades, subset=['Grade'])  # map colors to grades column

            # Display the styled dataframe
            st.dataframe(ticker_grades_df, use_container_width=True)

        st.write("Grade Distributions:")
        with st.container(border=True):
            # Show grade distribution of tickers
            grade_counts = grades_df['Grade'].value_counts().sort_index()
            st.bar_chart(grade_counts)

    # Write Grades Section to App
    add_ticker_grades_section(grades_df)


# ---- TEST BLOCK ----
if __name__ == "__main__":

    print(grades_df.head())
