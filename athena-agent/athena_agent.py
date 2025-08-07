# agent.py - AI Agent using Bedrock and Athena integration

import boto3
import pandas as pd
import json
import time
from typing import Tuple
import config
import streamlit as st
import plotly.express as px
import re


class AthenaChatbotAgent:
    def __init__(self):
        """
        Initialize the agent with AWS services

        Prerequisites:
        1. AWS credentials configured (via AWS CLI, environment variables, or IAM role)
        2. Proper IAM permissions for Bedrock and Athena
        3. S3 bucket for Athena results

        ***Think of AthenaChatbotAgent like a personal assistant you're hiring.***

        __init__ is like onboarding them — you give them a laptop (bedrock_client), access to your database (
        athena_client), and a filing cabinet (s3_client). self is just "this assistant" — so when you say
        self.bedrock_client, you're saying this assistant's Bedrock tool.
        """
        # Ensure AWS credentials are configured
        # You can configure AWS credentials using:
        # - AWS CLI: aws configure
        # - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        # - IAM role (if running on EC2)

        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
        )

        self.athena_client = boto3.client(
            'athena',
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
        )

        self.s3_client = boto3.client(
            's3',
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
        )

    def generate_analysis(self, user_question: str) -> dict:
        """
        Use Bedrock to convert natural language to SQL and provide analysis

        Args:
            user_question: User's natural language question

        Returns:
            Dictionary with sql_query, explanation, summary_format, chart_recommendation
        """
        try:

            # Clean the system prompt - remove extra newlines and format properly
            cleaned_system_prompt = config.SYSTEM_PROMPT.strip().replace('\n\n', ' ').replace('\n', ' ')

            # Prepare the prompt using Llama chat format
            full_prompt = f"System: {cleaned_system_prompt} User: {user_question} Assistant:"

            # Correct Llama request body format
            request_body = {
                "prompt": full_prompt,
                "max_gen_len": 500,
                "temperature": 0.1,
                "top_p": 0.9
            }

            # Call Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=config.BEDROCK_MODEL_ID,
                body=json.dumps(request_body),
                contentType='application/json'
            )

            # Parse response - KEY FIX: Llama uses 'generation' not 'content'
            response_body = json.loads(response['body'].read())
            ai_response = response_body['generation'].strip()

            # Try to parse as JSON
            try:
                analysis = json.loads(ai_response)
                return analysis
            except json.JSONDecodeError:
                # Fallback if AI doesn't return proper JSON
                return {
                    "sql_query": ai_response,
                    "explanation": "Analysis of your data request",
                    "summary_format": "Results displayed in tabular format",
                    "chart_recommendation": "bar"
                }

        except Exception as e:
            raise Exception(f"Error generating analysis: {str(e)}")

    def execute_athena_query(self, sql_query: str) -> Tuple[pd.DataFrame, str]:
        """
        Execute SQL query in Athena and return results

        Args:
            sql_query: SQL query to execute

        Returns:
            Tuple of (DataFrame with results, execution_id)
        """
        try:
            # Start query execution
            response = self.athena_client.start_query_execution(
                QueryString=sql_query,
                QueryExecutionContext={'Database': config.ATHENA_DATABASE},
                ResultConfiguration={
                    'OutputLocation': config.ATHENA_S3_OUTPUT_LOCATION
                },
                WorkGroup=config.ATHENA_WORKGROUP
            )

            execution_id = response['QueryExecutionId']

            # Wait for query completion
            self._wait_for_query_completion(execution_id)

            # Get results
            results = self.athena_client.get_query_results(QueryExecutionId=execution_id)

            # Convert to DataFrame
            df = self._results_to_dataframe(results)

            return df, execution_id

        except Exception as e:
            raise Exception(f"Error executing Athena query: {str(e)}")

    def _wait_for_query_completion(self, execution_id: str, max_wait_time: int = 60):
        """Wait for Athena query to complete"""
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            response = self.athena_client.get_query_execution(QueryExecutionId=execution_id)
            status = response['QueryExecution']['Status']['State']

            if status == 'SUCCEEDED':
                return
            elif status in ['FAILED', 'CANCELLED']:
                error_msg = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                raise Exception(f"Query failed: {error_msg}")

            time.sleep(2)

        raise Exception("Query timed out")

    # note -> use static method when it doesn't need access to instance data (self) or class data (cls)
    @staticmethod
    def _results_to_dataframe(results: dict) -> pd.DataFrame:
        """Convert Athena results to pandas DataFrame"""
        try:
            rows = results['ResultSet']['Rows']

            if not rows:
                return pd.DataFrame()

            # Extract column names from first row
            columns = [col['VarCharValue'] for col in rows[0]['Data']]

            # Extract data rows (skip header)
            data = []
            for row in rows[1:]:
                row_data = []
                for col in row['Data']:
                    # Handle different data types
                    if 'VarCharValue' in col:
                        row_data.append(col['VarCharValue'])
                    else:
                        row_data.append(None)
                data.append(row_data)

            return pd.DataFrame(data, columns=columns)

        except Exception as e:
            print(f"Error converting Athena results to DataFrame: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def create_chart(self, df: pd.DataFrame, question: str):
        """
        Create dynamic charts based on AI analysis of data characteristics and question type
        """
        if df.empty:
            st.warning("No data to visualize")
            return

        # Clean and prepare data
        df_clean = df.copy()
        for col in df_clean.columns:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
            except:
                pass

        # Get data characteristics
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=['object', 'string']).columns.tolist()

        # Get AI recommendation for chart type
        try:
            chart_prompt = f"""
            You are a data visualization expert. Based on this analysis request: "{question}"
            
            Data characteristics:
            - Rows: {len(df_clean)}
            - Numeric columns: {numeric_cols}
            - Categorical columns: {categorical_cols}
            - Sample data: {df_clean.head(3).to_dict('records') if not df_clean.empty else 'No data'}
            
            Chart selection guidelines:
            {config.CHARTS_GUIDELINE}
            
            IMPORTANT: If the user specifically asks for a chart type (like "bar chart", "line chart", etc.), honor that request.
            
            Return ONLY one word: bar, line, pie, histogram, scatter, or none
            """

            request_body = {
                "prompt": f"System: {chart_prompt} Assistant:",
                "max_gen_len": 50,
                "temperature": 0.1,
                "top_p": 0.9
            }

            response = self.bedrock_client.invoke_model(
                modelId=config.BEDROCK_MODEL_ID,
                body=json.dumps(request_body),
                contentType='application/json'
            )

            response_body = json.loads(response['body'].read())
            chart_type = response_body['generation'].strip().lower()

            # Override AI recommendation if user specifically requests a chart type
            question_lower = question.lower()
            if "bar chart" in question_lower or "bar graph" in question_lower:
                chart_type = "bar"
            elif "line chart" in question_lower or "line graph" in question_lower:
                chart_type = "line"
            elif "pie chart" in question_lower or "pie graph" in question_lower:
                chart_type = "pie"
            elif "scatter plot" in question_lower or "scatter chart" in question_lower:
                chart_type = "scatter"
            elif "histogram" in question_lower:
                chart_type = "histogram"

        except Exception as e:
            # Check if user requested specific chart type as fallback
            question_lower = question.lower()
            if "bar chart" in question_lower or "bar graph" in question_lower:
                chart_type = "bar"
            elif "line chart" in question_lower or "line graph" in question_lower:
                chart_type = "line"
            else:
                chart_type = "bar"  # Default fallback

        # Identify date columns
        date_cols = []
        for col in df_clean.columns:
            if any(date_word in col.lower() for date_word in ['date', 'time', 'month', 'year', 'quarter']):
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col])
                    date_cols.append(col)
                    if col in categorical_cols:
                        categorical_cols.remove(col)
                except:
                    pass

        try:
            # Create chart based on AI recommendation
            if chart_type == "line" and date_cols and numeric_cols:
                df_sorted = df_clean.sort_values(date_cols[0])
                fig = px.line(df_sorted, x=date_cols[0], y=numeric_cols[0],
                              title=f"{numeric_cols[0].replace('_', ' ').title()} Over Time", markers=True)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{hash(question)}")

            elif chart_type == "pie" and categorical_cols and numeric_cols and len(df_clean) <= 8:
                fig = px.pie(df_clean, values=numeric_cols[0], names=categorical_cols[0],
                             title=f"{numeric_cols[0].replace('_', ' ').title()} Distribution")
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{hash(question)}")

            elif chart_type == "bar" and categorical_cols and numeric_cols:
                # Determine if horizontal or vertical based on category names length
                avg_name_length = df_clean[categorical_cols[0]].astype(str).str.len().mean()
                if avg_name_length > 15 or len(df_clean) > 10:
                    # Horizontal for long names or many categories
                    df_sorted = df_clean.sort_values(by=numeric_cols[0], ascending=True)
                    fig = px.bar(df_sorted, x=numeric_cols[0], y=categorical_cols[0], orientation='h',
                                 title=f"{numeric_cols[0].replace('_', ' ').title()} by {categorical_cols[0].replace('_', ' ').title()}")
                else:
                    # Vertical for short names
                    df_sorted = df_clean.sort_values(by=numeric_cols[0], ascending=False)
                    fig = px.bar(df_sorted, x=categorical_cols[0], y=numeric_cols[0],
                                 title=f"{numeric_cols[0].replace('_', ' ').title()} by {categorical_cols[0].replace('_', ' ').title()}")
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{hash(question)}")

            elif chart_type == "histogram" and numeric_cols:
                fig = px.histogram(df_clean, x=numeric_cols[0], nbins=min(20, len(df_clean)//5 + 5),
                                   title=f"Distribution of {numeric_cols[0].replace('_', ' ').title()}")
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{hash(question)}")

            elif chart_type == "scatter" and len(numeric_cols) >= 2:
                color_col = categorical_cols[0] if categorical_cols else None
                fig = px.scatter(df_clean, x=numeric_cols[0], y=numeric_cols[1], color=color_col,
                                 title=f"{numeric_cols[1].replace('_', ' ').title()} vs {numeric_cols[0].replace('_', ' ').title()}")
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{hash(question)}")

            elif chart_type == "none":
                st.info("Data is best viewed in table format")
                st.dataframe(df_clean, use_container_width=True)

            else:  # Default to bar chart - ensure we always create a chart if possible
                if categorical_cols and numeric_cols:
                    df_sorted = df_clean.sort_values(by=numeric_cols[0], ascending=False)
                    fig = px.bar(df_sorted, x=categorical_cols[0], y=numeric_cols[0],
                                 title=f"{numeric_cols[0].replace('_', ' ').title()} by {categorical_cols[0].replace('_', ' ').title()}")
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{hash(question)}")
                elif numeric_cols and len(numeric_cols) >= 2:
                    # If no categorical columns, create scatter plot with numeric columns
                    fig = px.scatter(df_clean, x=numeric_cols[0], y=numeric_cols[1],
                                     title=f"{numeric_cols[1].replace('_', ' ').title()} vs {numeric_cols[0].replace('_', ' ').title()}")
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{hash(question)}")
                elif numeric_cols:
                    # Single numeric column - create histogram
                    fig = px.histogram(df_clean, x=numeric_cols[0], nbins=min(20, len(df_clean)//5 + 5),
                                       title=f"Distribution of {numeric_cols[0].replace('_', ' ').title()}")
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{hash(question)}")
                else:
                    st.dataframe(df_clean, use_container_width=True)

        except Exception as e:
            st.error(f"Visualization error: {e}")
            st.dataframe(df_clean, use_container_width=True)

        st.caption(f"Total records: {len(df_clean)}")

    def generate_insights(self, df: pd.DataFrame, question: str) -> str:
        """
        Generate AI-powered actionable insights as if from a data analyst
        """
        if df.empty:
            return "No data available for analysis."

        try:
            # Prepare data summary for AI
            numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

            # Get basic stats
            data_summary = {
                "total_records": len(df),
                "columns": list(df.columns),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
            }

            # Add statistical summary for numeric columns
            if numeric_cols:
                for col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    data_summary[f"{col}_stats"] = {
                        "mean": float(df[col].mean()) if not df[col].isna().all() else 0,
                        "max": float(df[col].max()) if not df[col].isna().all() else 0,
                        "min": float(df[col].min()) if not df[col].isna().all() else 0
                    }

            # Add top categories for categorical columns
            if categorical_cols:
                for col in categorical_cols[:1]:  # Limit to first categorical column
                    top_categories = df[col].value_counts().head(3).to_dict()
                    data_summary[f"{col}_top_values"] = top_categories

            # Sample of actual data
            sample_data = df.head(3).to_dict('records') if not df.empty else []

            insight_prompt = f"""
            You are a senior data analyst providing actionable business insights.
            
            User Question: "{question}"
            
            Data Analysis Results:
            {json.dumps(data_summary, indent=2)}
            
            Sample Data:
            {json.dumps(sample_data, indent=2)}
            
            {config.EXPLANATION_GUIDELINE}
            """

            request_body = {
                "prompt": f"System: {insight_prompt} Assistant:",
                "max_gen_len": 400,
                "temperature": 0.1,  # Lower temperature for more consistent formatting
                "top_p": 0.8          # Lower top_p for more focused responses
            }

            response = self.bedrock_client.invoke_model(
                modelId=config.BEDROCK_MODEL_ID,
                body=json.dumps(request_body),
                contentType='application/json'
            )

            response_body = json.loads(response['body'].read())
            ai_insights = response_body['generation'].strip()

            return ai_insights

        except Exception as e:
            # Fallback to basic insights if AI fails
            return f"Not enough information to do a reasonable analysis."

    @staticmethod
    def athena_create_chart_debug(df: pd.DataFrame, question: str):
        """
        Debugs Athena to chart creation. Athena often returns numeric data as strings.
        Replace main function with this to show if this is an issue.
        """
        if df.empty:
            st.warning("No data to visualize")
            return

        # DEBUG: Add this section to see what's happening
        st.write("DEBUG INFO:")
        st.write(f"DataFrame shape: {df.shape}")
        st.write(f"DataFrame dtypes:\n{df.dtypes}")
        st.write(f"DataFrame head:\n{df.head()}")

        # Convert numeric columns that might be strings
        for col in df.columns:
            if col != df.columns[0]:  # Skip the first column (usually names/categories)
                try:
                    # Try to convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass

        st.write(f"After conversion - DataFrame dtypes:\n{df.dtypes}")

        # Get data characteristics
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        st.write(f"Numeric columns: {numeric_cols}")
        st.write(f"Categorical columns: {categorical_cols}")
        st.write(f"Date columns: {date_cols}")

        # Convert date-like string columns to datetime if possible
        for col in df.columns:
            if 'date' in col.lower() and col not in date_cols:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                    if col in categorical_cols:
                        categorical_cols.remove(col)
                except:
                    pass

        question_lower = question.lower()
        st.write(f"Question: {question_lower}")

        # For your specific case - "top 5 users by contracts" should trigger this
        if any(word in question_lower for word in ['top', 'highest', 'largest', 'most', 'best']) and len(df) <= 20:
            st.write("Matched 'top' condition")
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                st.write(f"Using categorical: {cat_col}, numeric: {num_col}")

                # Create horizontal bar chart for better readability of names
                chart_data = df.set_index(cat_col)[num_col]
                st.bar_chart(chart_data, horizontal=True)
                st.caption(f"📊 {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}")
                return  # Early return after successful chart
            else:
                st.write("Not enough categorical or numeric columns")

        # Add a fallback for your specific data structure
        st.write("Trying fallback chart...")
        if len(df.columns) >= 2:
            # Assume first column is categorical, second is numeric
            cat_col = df.columns[0]
            num_col = df.columns[1]

            # Force convert the numeric column
            try:
                df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
                chart_data = df.set_index(cat_col)[num_col]
                st.bar_chart(chart_data, horizontal=True)
                st.caption(f"📊 {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}")
                return
            except Exception as e:
                st.write(f"Fallback failed: {e}")

        # Final fallback - show the data table
        st.dataframe(df, use_container_width=True)
        st.caption("📋 Data results")

        # Always show row count
        st.caption(f"Total records: {len(df)}")

    def process_question(self, user_question: str) -> dict:
        """
        Main method to process user question and return comprehensive results

        Args:
            user_question: User's natural language question

        Returns:
            Dictionary containing analysis, sql_query, data, summary_stats, and any errors
        """
        result = {
            'analysis': {},
            'sql_query': '',
            'data': pd.DataFrame(),
            'error': None
        }

        try:
            # Generate comprehensive analysis from natural language
            analysis = self.generate_analysis(user_question)
            result['analysis'] = analysis
            result['sql_query'] = analysis.get('sql_query', '')

            # Execute query in Athena
            data, execution_id = self.execute_athena_query(result['sql_query'])
            result['data'] = data
            result['execution_id'] = execution_id

        except Exception as e:
            result['error'] = str(e)

        return result


# ---------------------- Test Functions -----------------------
def test_generate_analysis():
    """Test function to verify generate_analysis works"""
    print("Testing generate_analysis method...")

    try:
        # Create an instance of your agent
        agent = AthenaChatbotAgent()
        print("✅ Agent initialized successfully")

        # Test with a simple question
        test_question = "Show me the top 5 users by total sales"
        print(f"Testing question: {test_question}")

        # Call the method
        result = agent.generate_analysis(test_question)
        print("✅ generate_analysis completed")

        # Print the results
        print("\nRESULTS:")
        print("=" * 50)

        if isinstance(result, dict):
            for key, value in result.items():
                print(f"{key}: {value}")
        else:
            print(f"Unexpected result type: {type(result)}")
            print(f"Result: {result}")

        print("=" * 50)

        # Validate expected keys
        expected_keys = ['sql_query', 'explanation']
        missing_keys = [key for key in expected_keys if key not in result]

        if missing_keys:
            print(f"⚠️ Missing expected keys: {missing_keys}")
        else:
            print("✅ All expected keys present")

        # Check if SQL query looks valid
        sql_query = result.get('sql_query', '')
        if sql_query and ('SELECT' in sql_query.upper() or 'select' in sql_query):
            print("✅ SQL query appears to be valid")
        else:
            print(f"⚠️ SQL query might be invalid: {sql_query[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_athena_chatbot_agent():
    """Comprehensive test function to verify all AthenaChatbotAgent methods work"""
    print("Starting comprehensive AthenaChatbotAgent test...")
    print("=" * 60)

    try:
        # Test 1: Agent Initialization
        print("\n1: Testing Agent Initialization...")
        agent = AthenaChatbotAgent()
        print("✅ Agent initialized successfully")

        # Test 2: Generate Analysis
        print("\n2: Testing generate_analysis method...")
        test_question = "Show me the top 5 users by total sales"
        print(f"🤔 Testing question: {test_question}")

        analysis_result = agent.generate_analysis(test_question)
        print("✅ generate_analysis completed")

        # Print analysis results
        print("\n📋 ANALYSIS RESULTS:")
        print("-" * 40)
        if isinstance(analysis_result, dict):
            for key, value in analysis_result.items():
                print(f"{key}: {value}")
        else:
            print(f"Unexpected result type: {type(analysis_result)}")
            print(f"Result: {analysis_result}")

        # Validate expected keys (updated for new structure)
        expected_keys = ['sql_query', 'explanation']
        missing_keys = [key for key in expected_keys if key not in analysis_result]
        if missing_keys:
            print(f"⚠️ Missing expected keys: {missing_keys}")
        else:
            print("✅ All expected analysis keys present")

        # Check if SQL query looks valid
        sql_query = analysis_result.get('sql_query', '')
        if sql_query and ('SELECT' in sql_query.upper() or 'select' in sql_query):
            print("✅ SQL query appears to be valid")
        else:
            print(f"⚠️ SQL query might be invalid: {sql_query[:100]}...")

        print("-" * 40)

        # Test 3: Execute Athena Query (if we have a valid SQL query)
        if sql_query:
            print("\n3: Testing execute_athena_query method...")
            try:
                df, execution_id = agent.execute_athena_query(sql_query)
                print(f"✅ Athena query executed successfully")
                print(f"📊 Query Execution ID: {execution_id}")
                print(f"📈 Data shape: {df.shape}")
                print(f"📝 Column names: {list(df.columns)}")

                # Show first few rows if data exists
                if not df.empty:
                    print(f"📋 First 3 rows:")
                    print(df.head(3).to_string())
                else:
                    print("⚠️ Query returned no data")

            except Exception as e:
                print(f"❌ Athena query failed: {e}")
                df = None
                execution_id = None
        else:
            print("\n3️⃣ Skipping Athena query test - no valid SQL query")
            df = None
            execution_id = None

        # Test 4: Create Chart (replaced summary stats)
        print("\n4: Testing create_chart method...")
        if df is not None and not df.empty:
            try:
                print("✅ Chart creation test - displaying chart...")
                print("📊 CHART OUTPUT:")
                print("-" * 40)
                # Note: create_chart displays directly, so we just call it
                agent.create_chart(df, test_question)
                print("-" * 40)
                print("✅ Chart created and displayed successfully")
            except Exception as e:
                print(f"❌ Chart creation failed: {e}")
        else:
            print("⚠️ Skipping chart test - no data available")
            # Test with empty DataFrame
            try:
                empty_df = pd.DataFrame()
                agent.create_chart(empty_df, "test question")
                print("✅ Empty DataFrame chart test passed")
            except Exception as e:
                print(f"❌ Empty DataFrame chart test failed: {e}")

        # Test 5: Full Process Question Method
        print("\n5: Testing process_question method (full workflow)...")
        try:
            full_result = agent.process_question(test_question)
            print("✅ process_question completed successfully")

            print("\nFULL PROCESS RESULTS:")
            print("-" * 50)
            print(f"Analysis keys: {list(full_result.get('analysis', {}).keys())}")
            print(f"SQL Query length: {len(full_result.get('sql_query', ''))}")
            print(f"Data shape: {full_result.get('data', pd.DataFrame()).shape}")
            print(f"Error: {full_result.get('error')}")
            print(f"Execution ID: {full_result.get('execution_id', 'N/A')}")

            # Test chart creation with full result data
            if not full_result.get('data', pd.DataFrame()).empty:
                print("\n📊 Testing chart with full result data:")
                try:
                    agent.create_chart(full_result['data'], test_question)
                    print("✅ Full result chart created successfully")
                except Exception as e:
                    print(f"❌ Full result chart failed: {e}")

            if full_result.get('error'):
                print(f"⚠️ Process completed with error: {full_result['error']}")
            else:
                print("✅ Full process completed without errors")
            print("-" * 50)

        except Exception as e:
            print(f"❌ process_question failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 6: Edge Cases
        print("\n6: Testing edge cases...")

        # Test with empty question
        try:
            empty_result = agent.generate_analysis("")
            print("✅ Empty question test passed")
        except Exception as e:
            print(f"⚠️ Empty question test failed: {e}")

        # Test with complex question
        try:
            complex_question = "What's the average sales by region for users who joined in the last 6 months, broken down by product category?"
            complex_result = agent.generate_analysis(complex_question)
            print("✅ Complex question test passed")
            print(f"Complex SQL preview: {complex_result.get('sql_query', '')[:100]}...")
        except Exception as e:
            print(f"⚠️ Complex question test failed: {e}")

        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEST COMPLETED!")
        print("✅ Check the results above for any issues")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Test failed with critical error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_static_methods():
    """Test static methods independently"""
    print("\n🔧 Testing static methods independently...")

    # Test _results_to_dataframe with mock data
    mock_athena_results = {
        'ResultSet': {
            'Rows': [
                {'Data': [{'VarCharValue': 'user_name'}, {'VarCharValue': 'total_sales'}]},
                {'Data': [{'VarCharValue': 'John'}, {'VarCharValue': '1000'}]},
                {'Data': [{'VarCharValue': 'Jane'}, {'VarCharValue': '1500'}]}
            ]
        }
    }

    try:
        df = AthenaChatbotAgent._results_to_dataframe(mock_athena_results)
        print("✅ _results_to_dataframe test passed")
        print(f"📊 Mock data shape: {df.shape}")
        print(f"📝 Mock data columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ _results_to_dataframe test failed: {e}")

    # Test create_chart with sample data
    try:
        import pandas as pd
        sample_data = pd.DataFrame({
            'user_name': ['John', 'Jane', 'Bob'],
            'total_sales': [1000, 1500, 800],
            'region': ['North', 'South', 'North']
        })

        print("✅ create_chart test with sample data:")
        print("📊 SAMPLE CHART OUTPUT:")
        print("-" * 30)
        AthenaChatbotAgent.create_chart(sample_data, "show me the top users by sales")
        print("-" * 30)
        print("✅ create_chart test passed")
    except Exception as e:
        print(f"❌ create_chart test failed: {e}")

    # Test different chart types with various questions
    try:
        print("\nTesting different chart scenarios...")

        # Time series data
        time_data = pd.DataFrame({
            'contract_date': pd.date_range('2023-01-01', periods=6, freq='M'),
            'total_revenue': [10000, 15000, 12000, 18000, 20000, 25000]
        })
        print("📅 Time series chart test:")
        AthenaChatbotAgent.create_chart(time_data, "show revenue over time")

        # Single metric data
        single_metric = pd.DataFrame({
            'total_revenue': [150000]
        })
        print("\nSingle metric test:")
        AthenaChatbotAgent.create_chart(single_metric, "what is the total revenue")

        print("✅ All chart scenario tests passed")

    except Exception as e:
        print(f"❌ Chart scenario tests failed: {e}")


# ------------ test ------------
if __name__ == "__main__":

    # Test inserts
    print(f"Invoking model ID: {config.BEDROCK_MODEL_ID}")  # check if model pulls

    # Clean the system prompt - remove extra newlines and format properly
    test_cleaned_system_prompt = config.SYSTEM_PROMPT.strip().replace('\n\n', ' ').replace('\n', ' ')
    print(test_cleaned_system_prompt)  # check what system prompt looks like when formatted to single string

    # full prompt
    user_question = "test question"
    # Prepare the prompt using Llama chat format
    test_full_prompt = f"System: {test_cleaned_system_prompt}\n\nUser: {user_question}\n\nAssistant:"
    print(f" Test Full Prompt Output\n {test_full_prompt}")

    # ------------- function tests ------------------
    # Test json generation
    test_generate_analysis()

    # Main comprehensive test
    success = test_athena_chatbot_agent()

    # Static methods test
    test_static_methods()

    print(f"\nOverall test result: {'PASSED' if success else 'FAILED'}")
