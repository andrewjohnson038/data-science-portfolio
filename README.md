# Data Science Portfolio

Welcome to my Data Science Portfolio! This repo is where I keep a mix of personal and professional projects that highlight my experience with data analysis, data engineering, BI tools, and building data-driven applications.

Each folder in here is its own project, complete with code, documentation, and everything needed to explore it further. ReadMe files with more context will be placed in each sub-directory for the project.

---

## 📊 Projects

### 1. [Stock Analysis App](./stock_analysis_app)

Built with the **Streamlit** framework using **Python**, **CSS**, and **HTML**, this app features a custom **S-F grading model**, multiple **forecasting** and **trend analysis tools**, **web scraping**, and an integrated **generative AI chatbot**. Key libraries include **Pandas**, **NumPy**, **Plotly**, **BeautifulSoup**, and **MATLAB**.

The model data is retrieved through a scheduled batch process that loops through and does a lookup against a ticker list repo and iterates through spaced batch loops to adhere to **Yahoo Finance API** rate limits.

---

### 🔍 Techniques Used

- **Historical Price Plotting**  
  Uses **exponential smoothing** to identify historical trends.

- **Time-Series Forecasting**  
  Implements both **SARIMA** and **META Prophet** models for future price prediction.

- **Monte Carlo Simulation**  
  Generates a range of possible outcomes based on historical data and volatility.

- **Value at Risk (VaR) Assessment**  
  Evaluates potential downside risk at a given confidence level.

➡️ [Go to Project Directory](./stock_analysis_app)

---

### 2. [BI Portfolio](./bi-portfolio)

This subdirectory contains the **Python scripts** used to clean, transform, and prepare the data behind each **Tableau** report in my Tableau Public portfolio. It includes all the data wrangling and preprocessing steps that prepare the data for visualization.

The datasets utilized are either pulled from Kaggle or were generated with AI.

➡️ [Go to Project Directory](./bi-portfolio)

---

## 📁 Structure
