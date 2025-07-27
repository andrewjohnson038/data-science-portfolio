# config.py - Configuration file for the Athena Chatbot
import streamlit as st

# AWS Configuration
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = st.secrets.get("AWS_REGION")

# Bedrock Configuration
# Available anthropic options: anthropic.claude-3-sonnet-20240229-v1:0, anthropic.claude-3-haiku-20240307-v1:0
BEDROCK_MODEL_ID = "meta.llama3-70b-instruct-v1:0"

# Athena Database Configuration
ATHENA_DATABASE = "dummy_sales_data"

# Athena Query Output Location
ATHENA_S3_OUTPUT_LOCATION = "s3://tt-athena-results-bucket/queries"

# Workgroup: optional
ATHENA_WORKGROUP = "primary"

# This helps the AI generate better SQL queries
TABLE_SCHEMA = """
{
  "users": {
    "user_id": "int (primary key)",
    "user_name": "string", 
    "email": "string",
    "signup_date": "date (use DATE 'YYYY-MM-DD' for comparisons)",
    "country": "string"
  },
  "sales": {
    "record_id": "int (primary key)",
    "user_id": "int (foreign key to users.user_id)",
    "record_date": "date (use DATE 'YYYY-MM-DD' for comparisons)", 
    "sale_amount_usd": "decimal(10,2)",
    "product_category": "string"
  },
  "contracts": {
    "record_id": "int (primary key)",
    "user_id": "int (foreign key to users.user_id)",
    "contract_date": "date (use DATE 'YYYY-MM-DD' for comparisons)",
    "contract_value_usd": "decimal(10,2)",
    "contract_type": "string"
  },
  "invoices": {
    "record_id": "int (primary key)", 
    "user_id": "int (foreign key to users.user_id)",
    "invoice_date": "date (use DATE 'YYYY-MM-DD' for comparisons)",
    "invoice_amount_usd": "decimal(10,2)",
    "payment_status": "string"
  }
}
"""

# System prompt for the AI agent
# Add this to your config.py SYSTEM_PROMPT

SYSTEM_PROMPT = """
You are a data analyst ai assistant name Ari. You help Convert natural language questions into Athena-compatible SQL (Presto dialect) and return results with insights.

Database: dummy_sales_data

DATA CONTEXT:
- all data covers only: June 2025 to July 2025 (current data range)
- if user asks for top 5 "advisor", "seller", "wholesaler", etc, they mean the users in the users table
- When users ask about "recent" data, use 2025 dates
- When users ask about "last month" or "this month", interpret based on July 2025 being current
- Available product categories: Software, Service, Hardware

IMPORTANT SQL RULES:
1. For date comparisons, always use DATE 'YYYY-MM-DD' format
2. For date columns, use proper date functions like DATE_TRUNC, EXTRACT
3. Never compare date columns directly to strings
4. Use CAST() when necessary for type conversions

Table schema:
{
  "users": {
    "user_id": "int (primary key)",
    "user_name": "string", 
    "email": "string",
    "signup_date": "date (use DATE 'YYYY-MM-DD' for comparisons)",
    "country": "string"
  },
  "sales": {
    "record_id": "int (primary key)",
    "user_id": "int (foreign key to users.user_id)",
    "record_date": "date (use DATE 'YYYY-MM-DD' for comparisons)", 
    "sale_amount_usd": "decimal(10,2)",
    "product_category": "string (values: Software, Service, Hardware)"
  },
  "contracts": {
    "record_id": "int (primary key)",
    "user_id": "int (foreign key to users.user_id)",
    "contract_date": "date (use DATE 'YYYY-MM-DD' for comparisons)",
    "contract_value_usd": "decimal(10,2)",
    "contract_type": "string"
  },
  "invoices": {
    "record_id": "int (primary key)", 
    "user_id": "int (foreign key to users.user_id)",
    "invoice_date": "date (use DATE 'YYYY-MM-DD' for comparisons)",
    "invoice_amount_usd": "decimal(10,2)",
    "payment_status": "string"
  }
}

EXAMPLE CORRECT DATE QUERIES:
- WHERE record_date >= DATE '2025-01-01' (for 2025 data)
- WHERE record_date BETWEEN DATE '2025-06-01' AND DATE '2025-06-30' (June 2025)
- WHERE EXTRACT(YEAR FROM record_date) = 2025
- WHERE DATE_TRUNC('month', record_date) = DATE '2025-07-01' (July 2025)
- WHERE user_id = 12 (integer, no quotes)
- WHERE sale_amount_usd > 1000.00 (decimal, no quotes)
- WHERE product_category = 'Software' (string, with quotes)
- WHERE record_date >= DATE '2025-06-01' (date literal)
- WHERE user_id IN (1, 2, 3) (integer list, no quotes)
- WHERE product_category IN ('Software', 'Hardware') (string list, with quotes)
- dollar amounts or ids will be integers, others will be string and date will be '2025-01-01' type format

Return response as JSON with:
{
  "sql_query": "valid SQL query",
  "explanation": "brief explanation of the query",
}

If the user asks for aggregated data by user, use the user name in the data chart and not the user ID.
"""


# Higher Level System prompt for the AI agent that would use more tokens
HIGHER_LEVEL_SYSTEM_PROMPT = f"""
You are a helpful data analyst assistant named Ari. Your job is to convert natural language questions into SQL queries for Amazon Athena and provide comprehensive analysis.

Available database: {ATHENA_DATABASE}
Available tables and schema: {TABLE_SCHEMA}

When a user asks a question, you need to provide a JSON response with the following structure:
{{
    "sql_query": "SELECT * FROM table...",
    "explanation": "Brief explanation of what this query does and what the user can expect to see",
}}

Guidelines for SQL:
- Use proper SQL syntax for Athena (Presto SQL)
- For date comparisons, assume current year if not specified
- For "top N" requests, use ORDER BY and LIMIT
- Always use table aliases for clarity
- Always limit results to reasonable numbers (use LIMIT clause)

Guidelines for explanations:
- Keep explanations concise but informative
- Explain what business insights the query will reveal
- Mention any assumptions you made about the request

Guidelines for chart recommendations:
- "bar" for top/bottom comparisons, categories
- "line" for trends over time
- "pie" for part-to-whole relationships (max 8 categories)
- "histogram" for distributions
- "scatter" for correlations
- "none" if data isn't suitable for visualization

Return only valid JSON, no additional text or markdown formatting.
"""

# Bedrock Config
#     Step 1: Go to Bedrock Console
#
#     Open AWS Bedrock Console: https://console.aws.amazon.com/bedrock/
#     Make sure you're in the correct region (same region as your code - probably us-east-1)
#
#     Step 2: Request Model Access
#
#     Click "Model access" in the left sidebar
#     Click "Request model access" or "Manage model access"
#     Find "Anthropic" section
#     Check the box next to "Claude 3 Sonnet" (or whatever Claude model you're using)
#     Click "Request model access"
#
#     Step 3: Wait for Approval
#
#     Access is usually granted instantly for most models
#     You'll see a green "Access granted" status when it's ready
#     Refresh the page if needed
#
#     Step 4: Check Your Model ID
#     Make sure your config.py has the correct model ID:

# schema in json form
TABLE_SCHEMA = {
    "users": {
        "description": "Contains user profile information with unique identifiers and contact details.",
        "columns": {
            "user_id": "Unique identifier for each user.",
            "user_name": "Full name of the user.",
            "email": "User's email address.",
            "signup_date": "Date when the user signed up.",
            "country": "User's country of residence."
        }
    },
    "contracts": {
        "description": "Contains contract transaction details linked to users. Each record represents a single contract signed by a user on a specific date.",
        "columns": {
            "record_id": "Unique identifier for each contract record.",
            "user_id": "Foreign key linking to users.user_id.",
            "record_date": "Date the contract was signed or recorded.",
            "contract_value_usd": "Total value of the contract in USD.",
            "contract_type": "Type of contract, e.g., Annual, Monthly, or One-Time."
        }
    },
    "sales": {
        "description": "Contains sales transactions linked to users, detailing the amount and product category.",
        "columns": {
            "record_id": "Unique identifier for each sales transaction.",
            "user_id": "Foreign key linking to users.user_id.",
            "record_date": "Date the sale was made.",
            "sale_amount_usd": "Amount of the sale in USD.",
            "product_category": "Category of the product sold."
        }
    }
}
# "invoices": {
#     "description": "Contains invoice records issued to users. Each record represents a single invoice with associated payment status.",
#     "columns": {
#         "record_id": "Unique identifier for each invoice record.",
#         "user_id": "Foreign key linking to users.user_id.",
#         "record_date": "Date the invoice was issued.",
#         "invoice_amount_usd": "Total amount billed in USD on the invoice.",
#         "status": "Current status of the invoice, e.g., Paid, Unpaid, Overdue."
#     }
# },
# "leads": {
#     "description": "Contains lead information generated by or linked to users, with details on source and value.",
#     "columns": {
#         "record_id": "Unique identifier for each lead record.",
#         "user_id": "Foreign key linking to users.user_id.",
#         "record_date": "Date the lead was created or recorded.",
#         "lead_source": "Source from which the lead originated, e.g., referral, ad campaign.",
#         "lead_value_usd": "Estimated monetary value of the lead in USD."
#     }
# },
# "renewals": {
#     "description": "Contains contract renewal records associated with users, detailing term length and renewal value.",
#     "columns": {
#         "record_id": "Unique identifier for each renewal record.",
#         "user_id": "Foreign key linking to users.user_id.",
#         "record_date": "Date the renewal was processed.",
#         "renewal_term_months": "Length of the renewal term in months.",
#         "renewal_value_usd": "Monetary value of the renewal in USD."
#     }
# },

# -------------------- Testing --------------------------
if __name__ == "__main__":
    # Test each part separately
    print("Testing ATHENA_DATABASE:", ATHENA_DATABASE)
    print("Testing TABLE_SCHEMA:", repr(TABLE_SCHEMA))  # repr() shows hidden characters
    print("Testing SYSTEM_PROMPT creation...")

    try:
        test_prompt = f"Database: {ATHENA_DATABASE}\nSchema: {TABLE_SCHEMA}"
        print("✅ F-string works")
    except Exception as e:
        print(f"❌ F-string error: {e}")
