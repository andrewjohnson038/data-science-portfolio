# Business Intelligence Portfolio

Collection of ETL and data wrangling projects that prepare data for visulization in Tableau Public.

Tableau Public Link: https://public.tableau.com/app/profile/andrew.johnson1314/vizzes

## Projects Overview

### Spotify Analysis
`spotify-analysis-tableau-proj/`
- **Data Source**: Kaggle API: path = "nelgiriyewithana/top-spotify-songs-2023"
- **ETL Process**: 
  - Extracts user listening history and track metadata
  - Transforms raw JSON into structured tables
  - Outputs CSV files for Tableau dashboard
- **Key Metrics**: Listening patterns, genre preferences, artist analysis

### Regional Sales Dashboard
`regional-sales-dashboard-tableau-proj/`
- **Data Source**: Kaggle API: path = "talhabu/us-regional-sales-data"
- **ETL Process**:
  - Extracts sales transactions and regional data
  - Performs regional aggregations and time-based calculations
  - Outputs structured CSV for regional performance dashboard
- **Key Metrics**: Regional performance, sales trends, product categories

### Product Comparison
`product-compare-tableau-proj/`
- **Data Source**: AI dummy data generation 
- **ETL Process**:
  - Extracts product specifications and performance metrics
  - Normalizes data across different product categories
  - Outputs comparison-ready CSV for Tableau
- **Key Metrics**: Product specifications, performance comparisons, market positioning

## Common Features
- All projects follow similar ETL patterns:
  1. Data extraction from source
  2. Data cleaning and transformation
  3. Aggregation and calculation of metrics (Most of calculations done in Tableau)
  4. Output to CSV/excel format for Tableau consumption
- Each project includes:
  - Data wrangling in pandas
  - ETL from API and/or source file
  - Documentation of data transformations

## Output Format
- CSV files with data structured for optimal data modeling and/or vizulization in Tableau 