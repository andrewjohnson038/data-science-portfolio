# Data Science Portfolio

This repo contains a mix of personal projects that highlight experience with data analysis, data engineering, data wrangling, machine learning, ai optimization & app integration, etc. for app and BI-driven projects. 

Each sub-directory folder from the root is its own project, complete with its own code and documetnation. ReadMe files with more context can be referenced in each sub-directory for the project.

---

## 📊 Projects

### 1. [Stock Analysis App](./stock_analysis_app)

Built with the **Streamlit** framework using **Python**, **CSS**, and **HTML**, this app features a custom **S-F grading model**, multiple **forecasting** and **trend analysis tools**, **web scraping**, and an integrated **generative AI chatbot**. Key libraries include **Pandas**, **NumPy**, **Plotly**, **BeautifulSoup**, and **MATLAB**.

Data for the stock grades page is batched on a weekly schedule in **AWS**, using an **S3**, **EC2**, and **EventBridge Scheduler** stack.

The model data is retrieved through a scheduled batch process that loops through and does a lookup against a ticker list and iterates through spaced batch loops to adhere to **Yahoo Finance API** rate limits.


---

### 🔍 Techniques Used

- **Historical Price Plotting**  
  Uses **exponential smoothing** to identify historical trends.

- **Time-Series Forecasting**  
  Implements **META Prophet** model for future price prediction.

- **MACD (Moving Average Convergence Divergence)**  
  A trend-following momentum indicator used to determine the strength and direction of a trend.

- **SMA (Simple Moving Average)**  
  Used to smooth out price data to identify trends over a specific period.

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

Reference the ReadMe files located in the subdirectory folders for individual project structures.
