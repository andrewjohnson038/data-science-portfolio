# Import libraries:
import streamlit as st  # Import streamlit (app Framework)
from groq import Groq  # Import Huggingface transformers model for gpt chatbot from Groq

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app methods
from stock_analysis_app.app_constants import groq_key

# Create groq client
client = Groq(api_key=groq_key)

# Session State
if "default_model" not in st.session_state:
    st.session_state["default_model"] = "llama3-8b-8192"

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! Any stock-related questions? Drop it below :)"}]

# Display the messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input for user message
if prompt := st.chat_input():
    # append message to message collection
    st.session_state.messages.append({"role": "user", "content": prompt})

    # display the new message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display the assistant response from the model
    with st.chat_message("assistant"):
        # place-holder for the response text
        response_text = st.empty()

        # Call the Groq API
        completion = client.chat.completions.create(
            model=st.session_state.default_model,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True
        )

        full_response = ""

        for chunk in completion:
            full_response += chunk.choices[0].delta.content or ""
            response_text.markdown(full_response)

        # add full response to the messages
        st.session_state.messages.append({"role": "assistant", "content": full_response})


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
