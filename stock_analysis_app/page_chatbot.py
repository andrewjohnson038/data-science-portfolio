import pandas as pd
from stock_analysis_app.app_constants import GROQ_API_KEY
from stock_analysis_app.app_constants import ExtraComponents
from stock_analysis_app.app_data import AppData
import streamlit as st
from groq import Groq

# Initialize classes and API client
ec = ExtraComponents()
data = AppData()
client = Groq(api_key=GROQ_API_KEY)

# ---------------- SIDEBAR CONTENT ----------------
ec.get_sidebar_notes()

# Get current ticker from session state (set in home_page.py)
current_ticker = st.session_state.get('current_ticker', None)


# ---------------- CHATBOT TICKER SELECTION ----------------

# Use default 'META' if a ticker hasn't been selected yet
if not current_ticker:
    current_ticker = 'META'

# Get the same ticker list used in home page
ticker_list = data.filtered_tickers()

# Set default index for dropdown based on current app ticker
if current_ticker in ticker_list:
    default_index = ticker_list.index(current_ticker)
else:
    default_index = 0

# Create two columns: ticker select (left), success message (right)
col1, col2 = st.columns([1, 2])

with col1:
    # Create chatbot-specific ticker dropdown
    chatbot_ticker = st.selectbox(
        "Stock for Chatbot Analysis",
        ticker_list,
        index=default_index,
        key="chatbot_ticker_dropdown",
        help="Select any stock for detailed chatbot analysis"
    )

st.write("---")


# ---------------- CHAT SETUP ----------------
# Initialize chat model and message history
if "default_model" not in st.session_state:
    st.session_state["default_model"] = "llama3-8b-8192"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! Any stock-related questions? Drop it below :)\n\n(Note: I will "
                                         "use app-related data for the ticker selected above if specifically "
                                         "prompted)"}]


# ---------------- Formatting HELPER FUNCTIONS FOR FORMATTING ----------------
def format_number(value, decimal_places=2):
    """Format numbers with proper decimal places, handling None/NaN"""
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        num = float(value)
        if num == int(num):  # If it's a whole number, show no decimals
            return f"{int(num)}"
        return f"{num:.{decimal_places}f}"
    except (ValueError, TypeError):
        return "N/A"


def format_percentage(value, decimal_places=0):
    """Format percentages properly"""
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        num = float(value)
        return f"{num:.{decimal_places}f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_currency(value, decimal_places=2):
    """Format currency values"""
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        num = float(value)
        return f"${num:.{decimal_places}f}"
    except (ValueError, TypeError):
        return "N/A"


def format_large_number(value, suffix=""):
    """Format large numbers with commas"""
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        num = float(value)
        if num >= 1e9:
            return f"{num/1e9:.2f}B{suffix}"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M{suffix}"
        else:
            return f"{int(num):,}{suffix}"
    except (ValueError, TypeError):
        return "N/A"


# ---------------- STOCK CONTEXT FUNCTION ----------------
def get_stock_context(ticker):
    """
    Fetches comprehensive stock data to provide context for the AI chatbot.
    Returns formatted string with key financial metrics and company info.
    """
    if not ticker:
        return "No stock is currently selected."

    try:
        # Get stock data from your existing method
        df = data.load_curr_stock_metrics(ticker)

        if df.empty:
            return f"No data available for {ticker}"

        # Get the data row (should only be one row)
        row = df.iloc[0]

        # Build context string with key stock information
        # Using proper formatting functions for consistent display
        context = f"""
        Current Stock Analysis Context for {ticker}:
        
        COMPANY OVERVIEW:
        - Company: {row['Company Name'] if pd.notna(row['Company Name']) else 'N/A'}
        - Sector: {row['Sector'] if pd.notna(row['Sector']) else 'N/A'} | Industry: {row['Industry'] if pd.notna(row['Industry']) else 'N/A'}
        - Business: {str(row['Business Description'])[:300] + '...' if pd.notna(row['Business Description']) else 'N/A'}
        
        CURRENT TRADING DATA:
        - Current Price: {format_currency(row['Regular Market Price'])}
        - Previous Close: {format_currency(row['Previous Close'])}
        - Day Range: {row['Day Range'] if pd.notna(row['Day Range']) else 'N/A'}
        - 52-Week Range: {row['52-Week Range'] if pd.notna(row['52-Week Range']) else 'N/A'}
        - Volume: {format_large_number(row['Regular Market Volume'], ' shares')}
        - Market Cap: {format_large_number(row['Market Cap'])}
        
        VALUATION METRICS:
        - P/E Ratio: {format_number(row['PE Ratio'])}
        - PEG Ratio: {format_number(row['PEG Ratio'])}
        - Price-to-Book: {format_number(row['Price-to-Book'])}
        - Price-to-Sales: {format_number(row['Price-to-Sales'])}
        
        FINANCIAL HEALTH:
        - ROE: {format_percentage(row['ROE'])}
        - Net Profit Margin: {format_percentage(row['Net Profit Margin'])}
        - Debt-to-Equity: {format_number(row['Debt-to-Equity'])}
        - Current Ratio: {format_number(row['Current Ratio'])}
        - Quick Ratio: {format_number(row['Quick Ratio'])}
        
        DIVIDENDS & GROWTH:
        - Dividend Yield: {format_percentage(row['Dividend Yield'])}
        - Dividend Rate: {format_currency(row['Dividend Rate'])}
        - YOY Revenue Growth: {format_percentage(row['YOY Revenue Growth'])}
        - Analyst Recommendation: {row['Analyst Recommendation'] if pd.notna(row['Analyst Recommendation']) else 'N/A'}
        
        RISK & OTHER METRICS:
        - Beta: {format_number(row['Beta'])} (Market Volatility vs S&P 500)
        - ESG Score: {row['ESG Score'] if pd.notna(row['ESG Score']) else 'N/A'}
        - Next Earnings Date: {row['Earnings Date'] if pd.notna(row['Earnings Date']) else 'N/A'}
        """
        return context

    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"


# ---------------- CHAT ENHANCEMENT FUNCTION ----------------
def get_enhanced_prompt(user_prompt, ticker):
    """
    Adds stock context to user questions so AI can give informed responses.
    If no ticker is selected, returns the original prompt unchanged.
    """
    if ticker:
        stock_context = get_stock_context(ticker)
        enhanced_prompt = f"""
{stock_context}

User Question: {user_prompt}

You are a top-tier financial analyst that is data driven in decisions. Your focus is in analysing mid to long term growth stocks and emphasize stocks that are at value in the current market.
Please answer the user's question with reference to the current stock ({ticker}) when relevant. 

- PROVIDE DATA BACK IN CLEAR AND PRECISE FORMATTING THAT IS VISUALLY APPEALING.
- IF THE USER ASKS A GENERAL QUESTION, PLEASE ANSWER IT AS BEST AS POSSIBLE. THE USER CAN USE YOU FOR NON_RELATED QUESTIONS TO THE SELECTED STOCKS.
- IF THE QUESTIONS IS NOT ABOUT THE SELECTED STOCK, ANSWER IT, BUT LET THE USER KNOW IF THEY WANT MORE ACCURATE AND UP-TO-DATE INSIGHTS TO SELECT THE STOCK ABOVE.
- IF THE USER ASKS ABOUT A METRIC YOU DON'T HAVE, TRY TO CALCULATE IT USING THE DATA AVAILABLE. IF YOU CAN'T TELL THE USER WHAT THE DATA IS AS OF THE LATEST DATE YOU KNOW IT WAS AVAILABLE.
"""

        return enhanced_prompt
    else:
        return user_prompt


# ---------------- CHAT INTERFACE ----------------
# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle new user input
if prompt := st.chat_input():
    # Add user message to chat history (display version)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        # Prepare messages for API call
        # Use enhanced prompt (with stock context) for the latest user message
        # NOTE: Using chatbot_ticker from dropdown selection
        api_messages = []
        for i, msg in enumerate(st.session_state.messages):
            if i == len(st.session_state.messages) - 1:  # Latest message
                api_messages.append({"role": "user", "content": get_enhanced_prompt(prompt, chatbot_ticker)})
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        # Stream response from Groq API
        completion = client.chat.completions.create(
            model=st.session_state.default_model,
            messages=api_messages,
            stream=True
        )

        # Display streaming response
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response)

        # Add complete response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
