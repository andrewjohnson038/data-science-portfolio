# Data Science Portfolio

This repo contains a mix of personal projects that highlight experience with data analysis, data engineering, data wrangling, machine learning, ai optimization & app integration, etc. for app and BI-driven projects. 

Each sub-directory folder from the root is its own project, complete with its own code and documetnation. ReadMe files with more context can be referenced in each sub-directory for the project.

---

## Projects

### 1. [Stock Analysis App](./stock_analysis_app)

Built with the **Streamlit** framework using **Python**, **CSS**, and **HTML**, this app features a custom **S-F grading model**, multiple **forecasting** and **trend analysis tools**, **web scraping**, and an integrated **generative AI chatbot**. Key libraries include **Pandas**, **NumPy**, **Plotly**, **BeautifulSoup**, and **MATLAB**.

Data for the stock grades page is batched on a weekly schedule in **AWS**, using an **S3**, **EC2**, and **EventBridge Scheduler** stack.

The model data is retrieved through a scheduled batch process that loops through and does a lookup against a ticker list and iterates through spaced batch loops to adhere to **Yahoo Finance API** rate limits.


---

### Analysis Techniques Used
Multiple market, trend, and financial analysis techniques are used in addition to retrieved financial data to improve data insights for each ticker. These techniques include:

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

- **Custom Grading Model**  
  A custom S-F grading model is applied to each selected ticker that utilizes market data and analysis techniques above to give a comprehensive model grade based on customised parameters and weighting scales.

### Engineering Techniques Used
The App uses multiple engineering techniques to strategically cahce data, utilize streamlit's session state for upholding session data, 
batch scripting for retrieving weekly grades against custom grading model, and data caching to limit processing and improve transition speeds across pages.

1. **Batch Scripting**  

- Leverages **AWS (S3, EC2, EventBridge Stack)** for data storage and batch scheduling. Custom grade models are pulled on a weekly batch.
- **S3** for data storage.
- **EC2** for batch script hosting on virtual server.
- **EventBridge Scheduler** to schedule batch script in EC2 with cron job.

2. **Data Wrangling & Visualization**  
   
- **Pandas** for data wrangling between data prep and ETL for batch jobs.
- **Plotly & Matplotlib** for time series visualizations and visual trend tracing.

3. **Data APIs**  
  
- Utilizes **yfinance** & **Alpha Vantage** APIs for pulling in market data. Alpha Vantage is used as a back-up point if unreliable data elements pull through as null (e.g. yfinance lib not reliable for PEG ratio).

4. **Caching**  
  
- Cached sets on ticker retrieved to reduce rendering times between page switches. 
  
➡ [Go to Project Directory](./stock_analysis_app)

---

### 2. [Brightside Route Optimizer App](./brightside_route_optimizer_app)

A Streamlit application that optimizes delivery routes for Brightside's Fresh Produce PWYC (Pay What You Can) 
service in the Minnesota metro area, using a combination of machine learning clustering and real-time traffic data to create balanced, efficient routes based on number of team members volunterring to drive delivery routes that day.

---

### Analysis Techniques Used

The application uses a two-phase approach to optimize routes:

1. **Phase 1: Geographic Clustering (K-means)**
  - Uses latitude/longitude coordinates to create initial groups
  - Groups addresses that are geographically close to each other
  - Provides a quick first pass to get reasonable starting groups
  - Special case handling for specific addresses (e.g., '1920 4th Ave S')

2. **Phase 2: Drive-Time Optimization (Google Maps API)**
  - Takes the groups from Phase 1
  - Uses Google Directions API to find optimal driving order
  - Calculates actual drive times between addresses
  - Optimizes routes to minimize total drive time
  - Incorporates real-time traffic data

### Engineering Techniques Used

1. **Parallel Processing**
  - Uses ThreadPoolExecutor for parallel geocoding
  - Processes multiple addresses simultaneously
  - 5 worker threads for geocoding
  - 3 worker threads for route optimization

2. **Caching**
  - Uses streamlit cacheing for geographic coordinates (30-minute cache)
    - Prevents repeated API calls for same address

3. **API Optimization**
  - Retry logic for API calls (3 attempts)
  - 10-second timeout
  - Traffic-aware routing parameters

➡ [Go to Project Directory](./brightside_route_optimizer_app)

---

### 3. [BI Portfolio](./bi-portfolio)

This subdirectory contains the **Python scripts** used to clean, transform, and prepare the data behind each **Tableau** report in my Tableau Public portfolio. It includes all the data wrangling and preprocessing steps that prepare the data for visualization.

The datasets utilized are either pulled from Kaggle or were generated with AI.

➡ [Go to Project Directory](./bi-portfolio)

---

## Structure

Reference the ReadMe files located in the subdirectory folders for individual project structures.
