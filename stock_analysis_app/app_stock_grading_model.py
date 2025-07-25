# This file is for the custom grade model that is called from the main file of the app

# Import python packages
import pandas as pd
import yfinance as yf

# Import app methods
from stock_analysis_app.app_constants import DateVars
from stock_analysis_app.app_data import AppData

# Instantiate any imported classes here:
dv = DateVars()
data = AppData()


# Create stock grade model class
class StockGradeModel:

    # Method to retrieve a full list of filtered tickers
    @staticmethod
    def calculate_grades(ticker, stock_metrics_df, forecast_df, ind_avg_df, mc_sim_df, sharpe_ratio, var_score, rsi_score, sma_percent_diff):

        # Initialize scores
        model_pe_score = 0
        model_roe_score = 0
        model_volume_score = 0
        model_dividend_score = 0
        model_current_ratio_score = 0
        model_debt_to_equity_score = 0
        model_net_profit_margin_score = 0
        model_sharpe_score = 0
        model_prophet_sarima_score = 0
        model_stock_price_vs_sp500_score = 0
        model_analyst_rating_score = 0
        model_operating_cash_flow_growth_score = 0
        model_peg_ratio_score = 0
        model_var_score = 0
        model_monte_carlo_score = 0
        model_rsi_sma_score = 0

        # convert stock_merics_df to scalar values
        # (will eliminate having to call iloc across all variables called from df)
        stock_metrics_dict = stock_metrics_df.iloc[0].to_dict()

        # Set Up Generic Vars Here
        current_price = stock_metrics_dict['Regular Market Price']  # Most recent stock price
        last_price = stock_metrics_dict['Previous Close']  # Most recent stock price
        mc_simulations_num = len(mc_sim_df)  # Number of monte carlo simulations by row length (currently 1000 sims)

        # ------- Forecast Percent Diff Variable
        # Set Up Any Remaining Variables
        model_chosen_prophet_price = forecast_df['yhat'].iloc[
            -1]  # forecast['yhat'].iloc[-1] retrieves the forecasted value

        # Get Forecasted X Year $ Difference & Percentage:
        model_prophet_forecast_difference = model_chosen_prophet_price - current_price
        model_prophet_forecast_difference = round(model_prophet_forecast_difference, 2)
        model_prophet_forecast_percent_difference = (model_prophet_forecast_difference / current_price)

        # ------- SPY vs Selected Ticker Compare (3 years of data)
        spy_compare_df = yf.download('SPY', dv.three_yrs_ago, dv.today)
        selected_ticker_compare_df = yf.download(ticker, dv.three_yrs_ago, dv.today)  # Note: directly using _df suffix here

        # Reset index to make 'Date' a column if not already
        spy_compare_df = spy_compare_df.reset_index()
        selected_ticker_compare_df = selected_ticker_compare_df.reset_index()

        # Calculate daily percentage change for each
        spy_compare_df['Percentage Change'] = spy_compare_df['Close'].pct_change() * 100
        selected_ticker_compare_df['Percentage Change'] = selected_ticker_compare_df['Close'].pct_change() * 100

        # Function to compare performance
        def compare_performance(stock_data=selected_ticker_compare_df, spy_data=spy_compare_df):  # Match the variable name here
            # Add error handling and debugging
            try:
                # Verify dataframes have expected columns
                print(f"Stock data columns: {stock_data.columns.tolist()}")
                print(f"SPY data columns: {spy_data.columns.tolist()}")

                # Make sure 'Date' and 'Percentage Change' exist in both dataframes
                if 'Date' not in stock_data.columns or 'Percentage Change' not in stock_data.columns:
                    print("Missing required columns in stock data")
                    return 0.5  # Return a neutral value if data is missing

                # Merge the stock and SPY data on Date
                comparison_df = pd.merge(stock_data[['Date', 'Percentage Change']],
                                         spy_data[['Date', 'Percentage Change']],
                                         on='Date',
                                         suffixes=('_stock', '_SPY'))

                # Debug merged dataframe
                print(f"Merged columns: {comparison_df.columns.tolist()}")

                # Compare if the stock outperforms SPY on each day
                comparison_df['Outperforms SPY'] = comparison_df['Percentage Change_stock'] > comparison_df['Percentage Change_SPY']

                # Calculate the percentage of days the stock outperforms SPY
                spy_outperform_percentage = comparison_df['Outperforms SPY'].mean()
                return spy_outperform_percentage

            except Exception as e:
                print(f"Error in compare_performance: {str(e)}")
                return 0  # Return zero on error

        # Call the Outperform % to retrieve
        spy_outperform_percentage = compare_performance(selected_ticker_compare_df, spy_compare_df)

        # ------- Monte Carlo Compare
        # Get percentage of simulations over latest day close price for all simulations

        # Extract the final price from each simulation (the last price in each row)
        final_prices = mc_sim_df.iloc[:, -1]  # The last column of each simulation path

        # Compare the final price to the most recent day's closing price (last_price)
        simulations_above_current_price = final_prices > last_price

        # Calculate the percentage of simulations that end above the current price
        percentage_above_current = (simulations_above_current_price.sum() / mc_simulations_num) * 100

        # Get percentage of simulations that end 9% are higher from the latest day close price
        price_threshold = last_price * 1.09

        # Compare the final price to the current day's closing price (last_price)
        simulations_above_9_percent = final_prices > price_threshold

        # Calculate the percentage of simulations that meet the condition
        percentage_above_9_percent = (simulations_above_9_percent.sum() / mc_simulations_num) * 100

        # ------- Merged Avg DFs
        # Merge the two DataFrames on the 'Industry' column
        industry_avg_merged_df = pd.merge(stock_metrics_df, ind_avg_df, on='Industry', how='left')

        # Reduce fields to company, industry and it's averages
        merged_df = industry_avg_merged_df[['Company Name', 'Industry', 'Average P/E Ratio', 'Average ROE']]

        # Check if the row is empty
        if not merged_df.empty:
            # Get the scalar values from the row
            selected_stock_industry_avg_pe = merged_df['Average P/E Ratio'].iloc[0]
            selected_stock_industry_avg_roe = merged_df['Average ROE'].iloc[0]
        else:
            # If the row is empty, assign default values
            selected_stock_industry_avg_pe = 25
            selected_stock_industry_avg_roe = 10  # comes through as if a percent

        # PE Ratio / YOY Growth (13%)
        pe_ratio = stock_metrics_dict.get('PE Ratio')

        yoy_revenue_growth = stock_metrics_dict.get('YOY Revenue Growth')

        if pe_ratio is not None and yoy_revenue_growth is not None:
            if pe_ratio < selected_stock_industry_avg_pe and yoy_revenue_growth > 0.19:
                model_pe_score = 0.13  # +13% for good growth
            elif pe_ratio < selected_stock_industry_avg_pe and 0.12 <= yoy_revenue_growth <= 0.18:
                model_pe_score = 0.08  # +8% for decent growth
            elif pe_ratio < selected_stock_industry_avg_pe:
                model_pe_score = 0.05  # +5% for below-average PE
            else:
                model_pe_score = 0  # 0% for high PE ratio compared to industry

        # PEG Ratio Score (5%) -- validated
        peg_ratio = stock_metrics_dict.get('PEG Ratio')

        if peg_ratio is None or peg_ratio == '':
            model_peg_ratio_score = 0.05  # Default score if missing
        else:
            try:
                peg_ratio_float = float(peg_ratio)
                if peg_ratio_float < 1:
                    model_peg_ratio_score = 0.05  # +5% for low PEG ratio
                elif peg_ratio_float > 1:
                    model_peg_ratio_score = -0.02  # -2% for high PEG ratio
                else:
                    model_peg_ratio_score = 0  # 0% for neutral PEG ratio
            except ValueError:
                # Handle case where peg_ratio can't be converted to a float
                model_peg_ratio_score = 0.05  # Default score in case of invalid value

        # ROE Score (5%)
        roe = stock_metrics_dict.get('ROE')

        if roe is not None:
            if roe > selected_stock_industry_avg_roe:
                model_roe_score = 0.05  # +5% if ROE is above sector average
            else:
                model_roe_score = 0  # 0% for low ROE compared to sector average

        # Volume Score (3%) -- validated
        regular_market_volume = stock_metrics_dict.get('Regular Market Volume')

        if regular_market_volume is not None:
            if regular_market_volume >= 500000:
                model_volume_score = 0.03  # +3% for high volume
            elif regular_market_volume < 500000:
                model_volume_score = -0.02  # -2% for low volume

        # Dividend Yield Score (3%) -- validated
        dividend_yield = stock_metrics_dict.get('Dividend Yield')

        if dividend_yield is not None:
            if dividend_yield > 2:
                model_dividend_score = 0.03  # +3% for above sector average
            elif dividend_yield < 2:
                model_dividend_score = 0

        # Current Ratio Score (4%) -- validated
        current_ratio = stock_metrics_dict.get('Current Ratio')

        if current_ratio is not None:
            if current_ratio > 1.2:
                model_current_ratio_score = 0.04  # +4% for good liquidity
            elif 1.0 <= current_ratio <= 1.99:
                model_current_ratio_score = 0.025  # +2.5% for acceptable liquidity
            elif 0.9 <= current_ratio < 1.0:
                model_current_ratio_score = 0.015  # +1.5% for below-average liquidity
            elif current_ratio < 0.5:
                model_current_ratio_score = -0.02  # -2% for poor liquidity

        # Debt-to-Equity Ratio Score (4%) -- validated
        debt_to_equity = stock_metrics_dict.get('Debt-to-Equity')

        if debt_to_equity is not None:
            if .9 <= debt_to_equity <= 1.4:
                model_debt_to_equity_score = 0.04  # +4%
            elif 1.41 <= debt_to_equity <= 1.89:
                model_debt_to_equity_score = 0  # 0%
            elif debt_to_equity > 1.9:
                model_debt_to_equity_score = -0.02  # -2%
            elif 0.33 <= debt_to_equity <= 0.99:
                model_debt_to_equity_score = 0.02  # +2%
            else:
                model_debt_to_equity_score = -.01  # -1%

        # Operating Cash Flow Growth (4%) -- validated
        yoy_ocfg_growth = stock_metrics_dict.get('YOY Operating Cash Flow Growth')

        if yoy_ocfg_growth > 0.10:
            model_operating_cash_flow_growth_score = 0.04  # +4% for high cash flow growth
        else:
            model_operating_cash_flow_growth_score = 0  # 0% for low or no growth

        # Net Profit Margin Score (3%) -- validated
        net_profit_margin = stock_metrics_dict.get('Net Profit Margin')

        if net_profit_margin is not None:
            if net_profit_margin >= 0.40:
                model_net_profit_margin_score = 0.04  # +4% for high profitability
            elif net_profit_margin >= 0.20:
                model_net_profit_margin_score = 0.03  # +3% for moderate profitability
            elif net_profit_margin >= 0.15:
                model_net_profit_margin_score = 0.02  # +2% for reasonable profitability
            elif net_profit_margin >= 0.10:
                model_net_profit_margin_score = 0.01  # +1% for acceptable profitability
            elif net_profit_margin < 0:
                model_net_profit_margin_score = -0.02  # -2% for negative profitability

        # Sharpe Ratio Score (4%) -- validated
        if sharpe_ratio is not None:
            if sharpe_ratio >= 1.5:
                model_sharpe_score = 0.04  # +4% for strong Sharpe ratio
            elif sharpe_ratio >= 1:
                model_sharpe_score = 0.02  # +2% for decent Sharpe ratio
            elif sharpe_ratio < 0.5:
                model_sharpe_score = -0.04  # -4% for poor Sharpe ratio

        # Prophet SARIMA Model (2%)
        if model_prophet_forecast_percent_difference is not None:
            if model_prophet_forecast_percent_difference >= 0.09:
                model_prophet_sarima_score = 0.02  # +2% for good forecast change
            else:
                model_prophet_sarima_score = 0  # 0% for no significant forecast change

        # Stock Price vs S&P 500 Score over 3 years (5%) -- validated
        if spy_outperform_percentage >= 0.90:
            model_stock_price_vs_sp500_score = 0.05  # +5% for outperforming S&P500
        else:
            model_stock_price_vs_sp500_score = 0  # 0% if not outperforming

        # Analyst Ratings Score (3%) -- validated
        analyst_recommendation_summary = stock_metrics_dict.get('Analyst Recommendation')
        # Need to pull as lower case for compare logic -> if statement to check the value is not None before adding .lr
        if analyst_recommendation_summary:
            analyst_recommendation_summary = analyst_recommendation_summary.lower()  # change to lower case

        if analyst_recommendation_summary is not None:
            if analyst_recommendation_summary == "strong_buy":
                model_analyst_rating_score = 0.025  # +2.5% for positive analyst ratings
            elif analyst_recommendation_summary == "outperform":
                model_analyst_rating_score = 0.03  # +3% for positive analyst ratings
            elif analyst_recommendation_summary == "buy":
                model_analyst_rating_score = 0.02  # +1.5% for negative analyst ratings
            elif analyst_recommendation_summary == "sell":
                model_analyst_rating_score = -0.02  # -1.5% for negative analyst ratings
            elif analyst_recommendation_summary == "strong_sell":
                model_analyst_rating_score = -0.02  # -1.5% for negative analyst ratings
            else:
                model_analyst_rating_score = 0  # 0% for neutral analyst ratings

        # VaR Score (3%) - Use the Yearly VaR -- validated
        if var_score is not None:
            # If hist_yearly_VaR_95 is a percentage (e.g., -0.05 for 5% loss)
            if var_score > -0.10:  # Less than 10% loss
                model_var_score = 0.03  # +3% for low VaR
            elif -0.20 <= var_score <= -0.10:  # Between 10% and 20% loss
                model_var_score = 0.02  # +2% for moderate VaR
            elif var_score < -0.45:  # More than 45% loss
                model_var_score = -0.03  # -3% for high VaR
            else:
                model_var_score = 0  # 0% for neutral VaR

        # Monte Carlo Score (5%) -- compare % of simulations above current price and 9% inc from current price
        if percentage_above_current >= 0.9 and percentage_above_9_percent >= 0.5:
            model_monte_carlo_score = 0.05  # +5% for good simulation
        elif percentage_above_current >= 0.65 and percentage_above_9_percent >= 0.35:
            model_monte_carlo_score = 0.02  # +2% for decent simulation
        elif percentage_above_current <= 0.65 and percentage_above_9_percent <= 0.35:
            model_monte_carlo_score = -0.02  # -2% for poor simulation
        else:
            model_monte_carlo_score = 0  # 0% for neutral simulation

        # RSI/SMA Score (2%) -- validated
        if rsi_score < 30 and sma_percent_diff > 5:
            model_rsi_sma_score = 0.01  # +1% for good RSI and SMA
        elif rsi_score > 70 and sma_percent_diff < 5:
            model_rsi_sma_score = 0  # -1% for bad RSI and SMA
        else:
            model_rsi_sma_score = 0  # 0% for neutral RSI and SMA

        # Calculate model_score
        model_score = (
                model_pe_score + model_roe_score + model_volume_score + model_dividend_score +
                model_current_ratio_score + model_debt_to_equity_score + model_net_profit_margin_score +
                model_sharpe_score + model_prophet_sarima_score + model_stock_price_vs_sp500_score +
                model_analyst_rating_score + model_operating_cash_flow_growth_score +
                model_peg_ratio_score + model_var_score + model_monte_carlo_score + model_rsi_sma_score
        )

        # Calculate base points (remaining points to make total 1)
        base_points = .4  # Adjust base points to fill up to 1
        total_score = model_score + base_points  # Add the adjusted base points

        # Create a dictionary of score_details for the model
        score_details = {
            'PE Ratio': {
                'score': model_pe_score,
                'max': 0.13,
                'value': pe_ratio
            },
            'ROE': {
                'score': model_roe_score,
                'max': 0.05,
                'value': roe
            },
            'Volume': {
                'score': model_volume_score,
                'max': 0.03,
                'value': regular_market_volume
            },
            'Dividend Yield': {
                'score': model_dividend_score,
                'max': 0.03,
                'value': dividend_yield
            },
            'Current Ratio': {
                'score': model_current_ratio_score,
                'max': 0.04,
                'value': current_ratio
            },
            'Debt-to-Equity': {
                'score': model_debt_to_equity_score,
                'max': 0.04,
                'value': debt_to_equity
            },
            'Net Profit Margin': {
                'score': model_net_profit_margin_score,
                'max': 0.04,
                'value': net_profit_margin
            },
            'Sharpe Ratio': {
                'score': model_sharpe_score,
                'max': 0.04,
                'value': sharpe_ratio
            },
            'Forecast % Change': {
                'score': model_prophet_sarima_score,
                'max': 0.02,
                'value': model_prophet_forecast_percent_difference
            },
            'SP500 Outperformance %': {
                'score': model_stock_price_vs_sp500_score,
                'max': 0.05,
                'value': spy_outperform_percentage
            },
            'Analyst Rating': {
                'score': model_analyst_rating_score,
                'max': 0.03,
                'value': analyst_recommendation_summary
            },
            'OCF Growth': {
                'score': model_operating_cash_flow_growth_score,
                'max': 0.04,
                'value': yoy_ocfg_growth
            },
            'PEG Ratio': {
                'score': model_peg_ratio_score,
                'max': 0.05,
                'value': peg_ratio
            },
            'VaR (95%)': {
                'score': model_var_score,
                'max': 0.03,
                'value': var_score
            },
            'Monte Carlo % Above Current': {
                'score': model_monte_carlo_score,
                'max': 0.05,
                'value': percentage_above_current
            },
            'RSI/SMA Combo': {
                'score': model_rsi_sma_score,
                'max': 0.01,
                'value': f"RSI: {rsi_score}, SMA Diff: {sma_percent_diff}"
            },
            'Base Points': {
                'score': base_points,
                'max': 0.4,
                'value': 'Constant'
            },
            'Total Score': {
                'score': total_score,
                'max': 1.0,
                'value': 'Calculated'
            }
        }

        # Determine grade and color based on total score
        if total_score >= 0.97:
            grade = "S"
            grade_color_background = "rgba(212, 175, 55, 0.85)"  # Rich gold
            grade_color_outline = "rgba(212, 175, 55, 1.0)"

        elif total_score >= 0.94:
            grade = "A"
            grade_color_background = "rgba(34, 139, 84, 0.85)"  # Deep forest green
            grade_color_outline = "rgba(34, 139, 84, 1.0)"

        elif total_score >= 0.90:
            grade = "A-"
            grade_color_background = "rgba(50, 160, 75, 0.85)"  # Balanced green
            grade_color_outline = "rgba(50, 160, 75, 1.0)"

        elif total_score >= 0.87:
            grade = "B+"
            grade_color_background = "rgba(72, 201, 137, 0.85)"  # Jade green
            grade_color_outline = "rgba(72, 201, 137, 1.0)"

        elif total_score >= 0.84:
            grade = "B"
            grade_color_background = "rgba(102, 205, 170, 0.85)"  # Aquamarine
            grade_color_outline = "rgba(102, 205, 170, 1.0)"

        elif total_score >= 0.80:
            grade = "B-"
            grade_color_background = "rgba(144, 238, 144, 0.85)"  # Light green
            grade_color_outline = "rgba(144, 238, 144, 1.0)"

        elif total_score >= 0.77:
            grade = "C+"
            grade_color_background = "rgba(205, 220, 57, 0.85)"  # Olive yellow
            grade_color_outline = "rgba(205, 220, 57, 1.0)"

        elif total_score >= 0.74:
            grade = "C"
            grade_color_background = "rgba(255, 215, 0, 0.85)"  # Deeper yellow
            grade_color_outline = "rgba(255, 215, 0, 1.0)"

        elif total_score >= 0.70:
            grade = "C-"
            grade_color_background = "rgba(255, 165, 0, 0.85)"  # Bright orange
            grade_color_outline = "rgba(255, 165, 0, 1.0)"

        elif total_score >= 0.67:
            grade = "D+"
            grade_color_background = "rgba(255, 140, 0, 0.85)"  # Dark orange
            grade_color_outline = "rgba(255, 140, 0, 1.0)"

        elif total_score >= 0.64:
            grade = "D"
            grade_color_background = "rgba(255, 80, 60, 0.85)"  # Soft red-orange
            grade_color_outline = "rgba(255, 80, 60, 1.0)"

        elif total_score >= 0.60:
            grade = "D-"
            grade_color_background = "rgba(255, 99, 71, 0.85)"  # Tomato
            grade_color_outline = "rgba(255, 99, 71, 1.0)"

        else:
            grade = "F"
            grade_color_background = "rgba(200, 30, 30, 0.85)"  # Deeper red
            grade_color_outline = "rgba(200, 30, 30, 1.0)"

        return total_score, grade, grade_color_background, grade_color_outline, score_details


# ---- TEST BLOCK ----
if __name__ == "__main__":

    # Get Test Data
    test_ticker = "META"
    test_price_hist_df = data.load_price_hist_data(test_ticker)
    test_stock_metrics_df = data.load_curr_stock_metrics(test_ticker)
    test_move_avg_data_df = data.get_simple_moving_avg_data_df(test_price_hist_df)

    # Get VaR
    test_hist_yearly_VaR_95, hist_yearly_VaR_95_dollars = data.calculate_historical_VaR(test_price_hist_df,
                                                                                      time_window='yearly')

    # Get SMA percent difference
    # Calculate Simple Moving Average Variables
    sma_price = test_move_avg_data_df['Close'].iloc[-1]  # Get the latest closing price
    sma_50 = test_move_avg_data_df['50_day_SMA'].iloc[-1]  # Latest 50-day SMA
    sma_200 = test_move_avg_data_df['200_day_SMA'].iloc[-1]  # Latest 200-day SMA

    # Calculate the difference between 50-day and 200-day SMA
    test_sma_price_difference = sma_50 - sma_200

    # Calculate the percentage difference between the 50-day and 200-day SMA
    test_sma_percentage_difference = (test_sma_price_difference / sma_200) * 100

    # Set Up Test Variables
    test_trained_model, test_forecasted_df = data.get_forecasted_data_df(test_price_hist_df, 5)  # 5 year forecast range
    test_ind_avg_df = data.get_industry_averages_df()
    test_mc_sim_df = data.get_monte_carlo_df(test_price_hist_df, 1000, 252)
    test_sharpe_ratio = data.calculate_sharpe_ratio(test_price_hist_df)
    test_var_score = test_hist_yearly_VaR_95
    test_rsi_score = data.get_latest_rsi(test_price_hist_df)
    test_sma_percent_diff = test_sma_percentage_difference

    print("\nTesting Grade Model Method...")
    # Calculate grades for the selected stock
    score, grade, grade_color_background, grade_color_outline, score_details = StockGradeModel.calculate_grades(
        test_ticker, test_stock_metrics_df, test_forecasted_df, test_ind_avg_df, test_mc_sim_df, test_sharpe_ratio,
        test_var_score, test_rsi_score, test_sma_percent_diff)

    if score is not None:
        # Print Test Grade for Test Ticker
        print(f"Grade & Score for: {test_ticker}")
        print(f"Score: {score}f")
        print(f"Grade: {grade}")

        # Print the detailed score breakdown
        print("\nScore Breakdown:")
        for metric, data in score_details.items():
            print(f"{metric:30s} | Score: {data['score']:.2f} / {data['max']:.2f} | Value: {data['value']}")
            # The 30s and .4f are format specifiers used to control how the output is displayed.
            # 30s: This specifies that the metric (which is a string) should be printed with a width of 30 characters
            # .2f: This specifies formatting values to two decimal points as a float type.

    # Test in Streamlit (Uncomment)
#     if score is not None:
#     # Display the grade in a rounded box with the grade color
#     st.markdown(f"""
#         <div style="background-color:{grade_color_background};
#                     color:white;
#                     font-size:20px;
#                     font-weight:bold;
#                     padding:10px 20px;
#                     border-radius:15px;
#                     border: 1px solid {grade_color_outline};
#                     display:inline-block;">
#             Model Grade: {grade}
#         </div>
#     """, unsafe_allow_html=True)
# else:
#     st.write(f"Error calculating grades for {selected_stock}")
