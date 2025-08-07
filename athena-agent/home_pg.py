# home_pg.py - Streamlit chatbot interface for Athena queries

import streamlit as st
import pandas as pd
from athena_agent import AthenaChatbotAgent
import config
import sys


# Initialize the agent
# @st.cache_resource
def initialize_agent():
    """Initialize the chatbot agent (cached for performance)"""
    return AthenaChatbotAgent()


def home_pg():
    """Main Streamlit application"""

    # Header
    st.markdown("<h1 style='text-align: center;'>AWS Athena AI Agent</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Ask questions about your data in natural language!</p>", unsafe_allow_html=True)

    st.write("---")

    # Before running, ensure you have:
    # 1. AWS credentials configured
    # 2. Updated config.py with your settings
    # 3. Proper IAM permissions for Bedrock and Athena

    # Initialize agent
    try:
        agent = initialize_agent()
    except Exception as e:
        st.error(f"Failed to initialize agent. Please check your AWS configuration: {str(e)}")
        return

    # Initialize chat history with Ari's greeting
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! I'm Ari, your AI-powered SQL assistant. What data would you like me to retrieve for you?"
            }
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display explanation if it exists
            if "explanation" in message and message["explanation"]:
                st.info(f"📊 **Analysis:** {message['explanation']}")

            # Display SQL query if it exists
            if "sql_query" in message and message["sql_query"]:
                with st.expander("📝 SQL Query Used"):
                    st.code(message["sql_query"], language="sql")

            # Display data visualization if it exists
            if "data" in message and not message["data"].empty:
                # Create chart based on original question and data
                original_question = message.get("original_question", message.get("content", ""))

                with st.expander("📈 Data Visualization", expanded=False):
                    agent.create_chart(message["data"], original_question)

            # Display insights if they exist (ADD THIS SECTION)
            if "insights" in message and message["insights"]:
                with st.expander("🧠 Intelligent Insights", expanded=False):
                    st.markdown(message["insights"])

            # Display data if it exists
            if "data" in message and not message["data"].empty:
                st.dataframe(message["data"], use_container_width=True)

    # Chat input
    if prompt := st.chat_input("e.g., 'Show me top 5 users by sales in June'"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the question
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question and querying the database..."):
                # Use the agent's process_question method (from AthenaChatbotAgent)
                result = agent.process_question(prompt)

            # Handle errors
            if result['error']:
                error_message = f"Sorry, I am unable to answer that question:/ Try asking again."
                error_message_dd = f"{result['error']}"
                st.markdown(error_message)
                with st.expander("Error Details"):
                    st.text(error_message_dd)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                return

            # Display results
            if result['data'].empty:
                response = "I executed the query but didn't find any matching data."
                st.markdown(response)

                # Still show the explanation and SQL
                if result['analysis'].get('explanation'):
                    st.info(f"📊 **Analysis:** {result['analysis']['explanation']}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "explanation": result['analysis'].get('explanation', ''),
                    "sql_query": result['sql_query'],
                    "data": result['data'],
                })
            else:
                # Success message with explanation
                response = f"I found {len(result['data'])} results for your question."
                st.markdown(response)

                # Show AI explanation
                if result['analysis'].get('explanation'):
                    st.info(f"📊 **Analysis:** {result['analysis']['explanation']}")

                # Show SQL query
                with st.expander("📝 SQL Query Used"):
                    st.code(result['sql_query'], language="sql")

                # CREATE AND SHOW CHART IMMEDIATELY (ADD THIS SECTION)
                with st.expander("📈 Data Visualization", expanded=False):
                    agent.create_chart(result['data'], prompt)

                # CREATE INSIGHTS SECTION (ADD THIS)
                with st.expander("🧠 Intelligent Insights", expanded=False):
                    insights = agent.generate_insights(result['data'], prompt)
                    st.markdown(insights)

                # Show data of SQL Query in DF form
                st.dataframe(result['data'], use_container_width=True)

                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "original_question": prompt,
                    "explanation": result['analysis'].get('explanation', ''),
                    "sql_query": result['sql_query'],
                    "data": result['data'],
                    "insights": insights
                })

    # Show clear chat button only if there are user messages (more than just the initial greeting)
    user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
    if user_messages:
        # Create a horizontal row just above the chat input
        col1, col2 = st.columns([4, 1])

        with col1:
            st.write(" ")

        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                # Reset to initial state with Ari's greeting
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Hi! I'm Ari, your AI-powered SQL assistant. What data would you like me to retrieve for you?"
                    }
                ]
                st.rerun()


# ------------ Test Functions ------------
def test_streamlit_integration():
    """Test the Streamlit integration with the AthenaChatbotAgent"""
    print("Testing Streamlit Integration with AthenaChatbotAgent...")
    print("=" * 60)

    try:
        # Test 1: Agent Initialization (same as Streamlit does)
        print("\n1: Testing Agent Initialization (Streamlit way)...")
        agent = initialize_agent()
        print("✅ Agent initialized successfully via Streamlit cache")

        # Test 2: Test the exact flow Streamlit uses
        print("\n2: Testing Streamlit Question Processing Flow...")
        test_question = "Show me the top 5 users by total sales"
        print(f"Testing question: {test_question}")

        # This is exactly what Streamlit calls
        result = agent.process_question(test_question)
        print("✅ agent.process_question() completed")

        # Print the exact result structure Streamlit expects
        print("\nSTREAMLIT RESULT STRUCTURE:")
        print("-" * 50)
        print(f"Keys in result: {list(result.keys())}")
        print(f"Error: {result.get('error')}")
        print(f"Analysis keys: {list(result.get('analysis', {}).keys())}")
        print(f"SQL Query length: {len(result.get('sql_query', ''))}")
        print(f"Data shape: {result.get('data', pd.DataFrame()).shape}")
        print(f"Execution ID: {result.get('execution_id', 'N/A')}")

        if result.get('error'):
            print(f"❌ ERROR FOUND: {result['error']}")
            print("🔍 This is the error causing your Streamlit issue!")
            return False
        else:
            print("✅ No errors in process_question")

        # Test 4: Test config access (what Streamlit sidebar uses)
        print("\n4: Testing Config Access...")
        try:
            database = config.ATHENA_DATABASE
            region = config.AWS_REGION
            print(f"✅ Database: {database}")
            print(f"✅ Region: {region}")
        except Exception as e:
            print(f"❌ Config access failed: {e}")

        # Test 5: Test edge cases that might break Streamlit
        print("\n5: Testing Edge Cases...")

        # Empty question
        try:
            empty_result = agent.process_question("")
            if empty_result.get('error'):
                print(f"✅ Empty question handled: {empty_result['error']}")
            else:
                print("✅ Empty question processed without error")
        except Exception as e:
            print(f"❌ Empty question test failed: {e}")

        # Very long question
        try:
            long_question = "Show me " + "the top users " * 50
            long_result = agent.process_question(long_question)
            if long_result.get('error'):
                print(f"✅ Long question handled: {long_result['error'][:100]}...")
            else:
                print("✅ Long question processed")
        except Exception as e:
            print(f"❌ Long question test failed: {e}")

        print("\n" + "=" * 60)
        print("STREAMLIT INTEGRATION TEST COMPLETED!")
        print("✅ Check above for any errors that match your Streamlit issue")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Critical test failure: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_direct_method_calls():
    """Test calling agent methods directly to isolate issues"""
    print("\nTesting Direct Method Calls...")
    print("-" * 40)

    try:
        # Create agent directly (not cached)
        print("Creating agent directly...")
        agent = AthenaChatbotAgent()

        # Test each method individually
        test_question = "Show me the top 5 users by total sales"

        print("\nTesting generate_analysis...")
        analysis = agent.generate_analysis(test_question)
        print(f"✅ Analysis completed: {list(analysis.keys())}")

        if 'sql_query' in analysis:
            print(f"\nTesting execute_athena_query...")
            data, exec_id = agent.execute_athena_query(analysis['sql_query'])
            print(f"✅ Athena query completed: {data.shape}")

            print(f"\nTesting create_chart...")
            chart = agent.create_chart(data, test_question)
            print(f"✅ Summary stats completed: {list(chart.keys())}")

        print("✅ All direct method calls successful")

    except Exception as e:
        print(f"❌ Direct method call failed: {e}")
        import traceback
        traceback.print_exc()


def test_config_validation():
    """Test configuration values"""
    print("\nTesting Configuration...")
    print("-" * 30)

    required_configs = [
        'AWS_REGION', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
        'BEDROCK_MODEL_ID', 'ATHENA_DATABASE', 'ATHENA_S3_OUTPUT_LOCATION',
        'ATHENA_WORKGROUP', 'SYSTEM_PROMPT'
    ]

    for config_name in required_configs:
        try:
            value = getattr(config, config_name, None)
            if value:
                # Hide sensitive values
                if 'KEY' in config_name:
                    display_value = f"{value[:8]}***"
                else:
                    display_value = str(value)[:50]
                print(f"✅ {config_name}: {display_value}")
            else:
                print(f"❌ {config_name}: NOT SET")
        except Exception as e:
            print(f"❌ {config_name}: ERROR - {e}")


# ------------ Run App or Tests ------------
if __name__ == "__main__":

    # to run the test functions, run ***python main.py test*** in terminal at athena-agent root
    # Check if we're running tests
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("RUNNING STREAMLIT TESTS...")
        print("=" * 60)

        # Run configuration test
        test_config_validation()

        # Run direct method tests
        test_direct_method_calls()

        # Run streamlit integration test
        success = test_streamlit_integration()

        print(f"\nTest Result: {'PASSED' if success else 'FAILED'}")

    else:
        # Run normal Streamlit app
        home_pg()
