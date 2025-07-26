def test_athena_setup():
    """Test Athena database setup and table structure"""
    print("🔍 DEBUGGING ATHENA SETUP...")
    print("=" * 50)

    try:
        from athena_agent import AthenaChatbotAgent
        import config

        agent = AthenaChatbotAgent()

        # Test 1: List all databases
        print("\n1️⃣ Testing Database Access...")
        try:
            list_databases_query = "SHOW DATABASES"
            df, exec_id = agent.execute_athena_query(list_databases_query)
            print("✅ Available databases:")
            print(df.to_string() if not df.empty else "No databases found")
        except Exception as e:
            print(f"❌ Cannot list databases: {e}")

        # Test 2: List tables in your configured database
        print(f"\n2️⃣ Testing Tables in Database '{config.ATHENA_DATABASE}'...")
        try:
            list_tables_query = "SHOW TABLES"
            df, exec_id = agent.execute_athena_query(list_tables_query)
            print("✅ Available tables:")
            print(df.to_string() if not df.empty else "No tables found")
        except Exception as e:
            print(f"❌ Cannot list tables: {e}")

        # Test 3: Check if specific tables exist
        print("\n3️⃣ Testing Specific Table Access...")
        test_tables = ['users', 'sales', 'user', 'sale', 'customers', 'orders', 'transactions']

        for table in test_tables:
            try:
                describe_query = f"DESCRIBE {table}"
                df, exec_id = agent.execute_athena_query(describe_query)
                print(f"✅ Table '{table}' exists with columns:")
                print(df.to_string())
                print("-" * 30)
            except Exception as e:
                print(f"❌ Table '{table}' not found: {str(e)[:100]}...")

        # Test 4: Try a simple SELECT to see what works
        print("\n4️⃣ Testing Simple Queries...")
        simple_queries = [
            "SELECT 1 as test_column",
            "SELECT current_timestamp",
            "SELECT * FROM information_schema.tables LIMIT 5"
        ]

        for query in simple_queries:
            try:
                df, exec_id = agent.execute_athena_query(query)
                print(f"✅ Query works: {query}")
                print(f"   Result shape: {df.shape}")
            except Exception as e:
                print(f"❌ Query failed: {query}")
                print(f"   Error: {str(e)[:100]}...")

        # Test 5: Show the exact query that's failing
        print("\n5️⃣ Testing the Failing Query...")
        try:
            test_question = "show me top 5 sellers in month of July by total contracts"
            analysis = agent.generate_analysis(test_question)
            failing_query = analysis.get('sql_query', '')

            print(f"🔍 Generated SQL Query:")
            print("-" * 40)
            print(failing_query)
            print("-" * 40)

            # Try to execute it
            try:
                df, exec_id = agent.execute_athena_query(failing_query)
                print("✅ Query executed successfully!")
                print(f"Result shape: {df.shape}")
            except Exception as e:
                print(f"❌ This is the failing query!")
                print(f"Error: {e}")

                # Let's try to fix it by suggesting alternatives
                print("\n🔧 Suggested fixes:")
                print("1. Check if table names are correct")
                print("2. Check if columns exist")
                print("3. Update your SYSTEM_PROMPT with correct table schema")

        except Exception as e:
            print(f"❌ Cannot generate analysis: {e}")

        print("\n" + "=" * 50)
        print("🎯 DIAGNOSIS COMPLETE!")
        print("Check above for missing tables/databases")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Critical error in Athena setup test: {e}")
        import traceback
        traceback.print_exc()


def test_config_tables():
    """Check what tables are configured in your system prompt"""
    print("\n📋 CHECKING CONFIGURED TABLE SCHEMA...")
    print("-" * 40)

    try:
        import config
        system_prompt = config.SYSTEM_PROMPT

        print("System prompt content:")
        print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)

        # Look for table names in the prompt
        common_table_words = ['table', 'schema', 'users', 'sales', 'orders', 'customers']
        for word in common_table_words:
            if word.lower() in system_prompt.lower():
                print(f"✅ Found '{word}' in system prompt")
            else:
                print(f"❌ '{word}' not found in system prompt")

    except Exception as e:
        print(f"❌ Cannot check config: {e}")


if __name__ == "__main__":
    test_config_tables()
    test_athena_setup()
