# Import libraries:
import streamlit as st  # Import streamlit (app Framework)
import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app methods
from stock_analysis_app.app_constants import DateVars
from stock_analysis_app.app_data import AppData
from stock_analysis_app.app_animations import CSSAnimations
from stock_analysis_app.app_stock_grading_model import StockGradeModel
from stock_analysis_app.app_grade_batch import GradeBatchMethods

# Instantiate any imported classes here:
dv = DateVars()
animation = CSSAnimations()
data = AppData()
model = StockGradeModel()
batch = GradeBatchMethods()

# Drop-downs with Notes on Sidebar:
st.sidebar.header("Financial Ratio Notes")
with st.sidebar.expander("MARKET VALUE RATIOS"):
    st.write("---Measure the current market price relative to its value---")
    st.write('<span style="color: lightcoral;">PE Ratio: [Market Price per Share / EPS]</span>',
             unsafe_allow_html=True)  # adding the additional HTML code allows us to change the text color in the write statement
    st.markdown(
        "[AVG PE Ratio by Sector](https://fullratio.com/pe-ratio-by-industry)")  # Insert a link on the sidebar to avg PE ratio by sector
    st.write(
        "Ratio Notes: PE should be evaluated and compared to competitors within the sector. A PE over the avg industry PE might indicate that the stock is selling at a premium, but may also indicate higher expected growth/trade volume; A lower PE may indicate that the stock is selling at a discount, but may also indicate low growth/trade volume.")
    st.write('<span style="color: lightcoral;">PEG Ratio: [PE / EPS Growth Rate]</span>', unsafe_allow_html=True)
    st.write("Ratio Notes: PEG > 1 = Likely overvalued || PEG < 1 = Likely undervalued")
    st.write(
        '<span style="color: lightcoral;">Price-to-Book Ratio: [Market Price per Share / Book Value Per Share]</span>',
        unsafe_allow_html=True)
    st.write(
        "Ratio Notes: PB > 1 = Indicates stock might be overvalued copared to its assets || PB < 1 = Indicates stock might be undervalued copared to its assets || Typically not a good indicator for companies with intangible assets, such as tech companies.")
    st.write('<span style="color: lightcoral;">Price-to-Sales Ratio: [Market Cap / Revenue]</span>',
             unsafe_allow_html=True)
    st.write(
        "Ratio Notes: 2-1 = Good || Below 1 = Better || Lower = Indicates the company is generating more revenue for every dollar investors have put into the company.")

with st.sidebar.expander("PROFITABILITY RATIOS"):
    st.write("---Measure the combined effects of liquidity, asset mgmt, and debt on operating results---")
    st.write('<span style="color: lightcoral;">ROE (Return on Equity): [Net Income / Common Equity]</span>',
             unsafe_allow_html=True)
    st.write("Ratio Notes: Measures total return on investment | Compare to the stock's sector | Bigger = Better")

with st.sidebar.expander("LIQUIDITY RATIOS"):
    st.write("---Measure the ability to meet current liabilities in the short term (Bigger = Better)---")
    st.write('<span style="color: lightcoral;">Current Ratio: [Current Assets / Current Liabilities]</span>',
             unsafe_allow_html=True)
    st.write(
        "Ratio Notes: Close to or over 1 = Good || Over 1 means the company is covering its bills due within a one year period")
    st.write(
        '<span style="color: lightcoral;">Quick Ratio: [(Current Assets - Inventory) / Current Liabilities]</span>',
        unsafe_allow_html=True)
    st.write(
        "Ratio Notes: Close to or over 1 = Good || Over 1 means the company is able to cover its bills due within a one year period w/ liquid cash")

with st.sidebar.expander("ASSET MANAGEMENT RATIOS"):
    st.write("---Measure how effectively assets are being managed---")
    st.write('<span style="color: lightcoral;">Dividend Yield: [DPS / SP]</span>', unsafe_allow_html=True)
    st.write(
        "Ratio Notes: For low growth stocks, should be higher and should look for consistent div growth over time- with signs of consistenly steady financials (able to pay debts consistently; want to see the company is managing its money well) || For growth stocks, typically lower, but if a stock shows high growth over time w/ a dividend yield that continues to remain the same or grow over time, this is a good sign (good to compare with their Current Ratio)")

with st.sidebar.expander("DEBT MANAGEMENT RATIOS"):
    st.write("---Measure how well debts are being managed---")
    st.write('<span style="color: lightcoral;">Debt-to-Equity: [Total Liabilities / Total Shareholder Equity]</span>',
             unsafe_allow_html=True)
    st.write(
        "Ratio Notes: A good D/E ratio will vary by sector & company. Typically a 1.0-1.5 ratio is ideal. The main thing to look for is that if the company is leveraging debt is that it has enough liquidity and consistent return to pay off those debts. Leveraging debt (when managed well) can be a good indicator that a growth company is leveraging debt in order to re-invest and grow faster, which is typically a good sign that the company is strategically well managed.")


# Create Container
sh_g = st.container()

# Set Up Date for Search
sample_ticker_list = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
ticker_list = data.filtered_tickers()

# Ticker Search Section
sh_g.write("Ticker Grade Search:")

# Write Grades Search to App
with sh_g.container():

    # Create a Process Method to update the batch df in app
    def load_or_update_grades(ticker_grades_batch_df, ticker_list, force_update=False):
        """
        Load grades from data frame or update them if needed.

        Parameters:
        ticker_grades_batch_df (DataFrame): Dataframe of the batched tickers
        ticker_list (list): List of tickers to process
        force_update (bool): If True, force update regardless of last update date

        Returns:
        DataFrame with updated ticker grades on new batch date after load
        """

        # Directly use the passed DataFrame
        update_needed = force_update

        if not update_needed and not ticker_grades_batch_df.empty:
            last_update = datetime.strptime(ticker_grades_batch_df['Update_Date'].iloc[0], "%Y-%m-%d")
            days_since_update = (datetime.now() - last_update).days

            # Update if more than 7 days since last update
            if days_since_update >= 7:
                update_needed = True
        else:
            update_needed = True

        if update_needed:
            st.info("Updating all stock grades - this will take several hours. The app will be unavailable during this time :).")
            grades_df = GradeBatchMethods.batch_process_tickers(ticker_list)

            # Return updated DataFrame
            return grades_df
        else:
            return ticker_grades_batch_df

    def add_ticker_grades_section(ticker_list):
        """
        Add a ticker grades section to your Streamlit app

        Parameters:
        ticker_list (list): List of Tickers to process
        """

        # Create an empty DataFrame as fallback
        empty_df = pd.DataFrame(columns=['Ticker', 'Grade', 'Score', 'Industry', 'Update_Date'])

        # Try to load existing grades if available
        try:
            if 'grades_df' not in st.session_state:
                st.session_state['grades_df'] = empty_df
            existing_grades_df = st.session_state['grades_df']

            # Check if update is needed (7 days passed)
            update_needed = False
            if not existing_grades_df.empty:
                last_update = datetime.strptime(existing_grades_df['Update_Date'].iloc[0], "%Y-%m-%d")
                days_since_update = (datetime.now() - last_update).days
                if days_since_update >= 7:
                    update_needed = True
                    st.warning(f"Grades are {days_since_update} days old. Consider updating for the latest data.")
            else:
                update_needed = True

            # Add update button
            col1, col2 = st.columns([1, 3])
            with col1:
                update_clicked = st.button("Update Stock Grades", type="secondary" if update_needed else "secondary")

            with col2:
                if not existing_grades_df.empty:
                    last_update = existing_grades_df['Update_Date'].iloc[0]
                    st.info(f"Grades last updated on: {last_update}")

            # If button is clicked, update the grades
            if update_clicked:
                with st.spinner("Updating stock grades..."):
                    grades_df = GradeBatchMethods.batch_process_tickers(ticker_list)
                    st.session_state['grades_df'] = grades_df
                    # Here you would typically save the updated grades to a database or file
            else:
                # Use existing grades if available and no update requested
                grades_df = existing_grades_df

        except Exception as e:
            st.error(f"Error loading grades: {str(e)}")
            grades_df = empty_df

        # Don't display anything else if we have no data and user hasn't clicked update
        if grades_df.empty:
            return

        # Add filters
        col1, col2, col3 = st.columns(3)

        with col1:
            # Filter by grade
            all_grades = ['All'] + sorted(grades_df['Grade'].unique().tolist())
            selected_grade = st.selectbox("Filter by Grade", all_grades)

        with col2:
            # Filter by industry
            all_industries = ['All'] + sorted(grades_df['Industry'].unique().tolist())
            selected_industry = st.selectbox("Filter by Industry", all_industries)

        with col3:
            # Sort options
            sort_options = ['Grade (Best First)', 'Grade (Worst First)', 'Score (High to Low)', 'Ticker (A-Z)']
            sort_selection = st.selectbox("Sort by", sort_options)

        # Apply filters
        filtered_df = grades_df.copy()

        if selected_grade != 'All':
            filtered_df = filtered_df[filtered_df['Grade'] == selected_grade]

        if selected_industry != 'All':
            filtered_df = filtered_df[filtered_df['Industry'] == selected_industry]

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

        # Display the styled dataframe
        st.dataframe(filtered_df.style.applymap(color_grades, subset=['Grade']), use_container_width=True)

        # Show grade distribution
        st.subheader("Grade Distribution")
        grade_counts = grades_df['Grade'].value_counts().sort_index()
        st.bar_chart(grade_counts)

    # Write Custom Model Grade to App
    add_ticker_grades_section(ticker_list)


# Any blocks written under here can use for testing directly from this file w/o importing the code below to the main
# (this is a built-in syntax of python)
if __name__ == "__main__":

    # Test Ticker List Pull
    print(ticker_list)
