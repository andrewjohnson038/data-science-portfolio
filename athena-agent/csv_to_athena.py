import boto3
import pandas as pd
from typing import Dict, List
import config


class CSVToAthenaConverter:
    def __init__(self, bucket_name: str, database_name: str, region: str):
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

        self.bucket_name = bucket_name
        self.database_name = database_name
        self.region = region

    def upload_csv_to_s3(self, csv_file_path: str, table_name: str) -> str:
        """Upload CSV file to S3 and return the S3 location"""
        s3_key = f"{table_name}/data.csv"

        try:
            self.s3_client.upload_file(csv_file_path, self.bucket_name, s3_key)
            s3_location = f"s3://{self.bucket_name}/{table_name}/"
            print(f"Uploaded {csv_file_path} to {s3_location}")
            return s3_location
        except Exception as e:
            print(f"Error uploading {csv_file_path}: {e}")
            return None

    def infer_schema_from_csv(self, csv_file_path: str) -> List[Dict]:
        """Read CSV and infer column types"""
        df = pd.read_csv(csv_file_path, nrows=1000)  # Sample first 1000 rows

        schema = []
        for col_name, dtype in df.dtypes.items():
            # Clean column name (remove spaces, special chars)
            clean_name = col_name.replace(' ', '_').replace('-', '_').lower()

            # Map pandas dtypes to Athena types
            if pd.api.types.is_integer_dtype(dtype):
                athena_type = 'INT'
            elif pd.api.types.is_float_dtype(dtype):
                athena_type = 'DOUBLE'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                athena_type = 'TIMESTAMP'
            else:
                athena_type = 'STRING'

            schema.append({
                'name': clean_name,
                'type': athena_type,
                'original_name': col_name
            })

        return schema

    def create_athena_table(self, table_name: str, schema: List[Dict], s3_location: str) -> bool:
        """Create Athena table with inferred schema"""

        # Build column definitions
        columns = []
        for col in schema:
            columns.append(f"{col['name']} {col['type']}")

        columns_str = ',\n    '.join(columns)

        create_table_query = f"""
        CREATE EXTERNAL TABLE {self.database_name}.{table_name} (
            {columns_str}
        )
        ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
        WITH SERDEPROPERTIES (
            'serialization.format' = ',',
            'field.delim' = ','
        ) 
        LOCATION '{s3_location}'
        TBLPROPERTIES (
            'has_encrypted_data'='false',
            'skip.header.line.count'='1'
        );
        """

        try:
            response = self.athena_client.start_query_execution(
                QueryString=create_table_query,
                QueryExecutionContext={'Database': self.database_name},
                ResultConfiguration={
                    'OutputLocation': f's3://{self.bucket_name}/athena-results/'
                }
            )

            print(f"Created table {table_name} in Athena")
            print(f"Query execution ID: {response['QueryExecutionId']}")
            return True

        except Exception as e:
            print(f"Error creating table {table_name}: {e}")
            return False

    def process_csv_files(self, csv_files: Dict[str, str]):
        """
        Process multiple CSV files

        Args:
            csv_files: Dict mapping table_name -> csv_file_path
        """
        for table_name, csv_path in csv_files.items():
            print(f"\nProcessing {table_name}...")

            # 1. Upload to S3
            s3_location = self.upload_csv_to_s3(csv_path, table_name)
            if not s3_location:
                continue

            # 2. Infer schema
            schema = self.infer_schema_from_csv(csv_path)
            print(f"Inferred schema: {[col['name'] + ':' + col['type'] for col in schema]}")

            # 3. Create Athena table
            success = self.create_athena_table(table_name, schema, s3_location)

            if success:
                print(f"✅ Successfully created table: {table_name}")
            else:
                print(f"❌ Failed to create table: {table_name}")


# Usage example:
if __name__ == "__main__":
    # Configure your settings
    BUCKET_NAME = "tt-athena-results-bucket"  # change to your bucket name
    DATABASE_NAME = "dummy_sales_data"  # change to your database name
    REGION = config.AWS_REGION  # change to your region

    # Map CSV files to table names; get path to data
    csv_files = {
        "sales_data": "dummy_data/sales.csv",
        "customer_data": "dummy_data/customers.csv",
        "product_data": "dummy_data/products.csv",
        "orders_data": "dummy_data/orders.csv",
        "inventory_data": "dummy_data/inventory.csv"
    }

    # Create converter and process files
    converter = CSVToAthenaConverter(BUCKET_NAME, DATABASE_NAME, REGION)
    converter.process_csv_files(csv_files)


# Adding Manually in Athena once CSV is in example

# -> contracts csv

        # -- Create contracts table in dummy_sales_data database
        # CREATE EXTERNAL TABLE dummy_sales_data.contracts (
        #     record_id STRING,
        # user_id STRING,
        # record_date DATE,
        # contract_value_usd DECIMAL(15,2),
        # contract_type STRING
        # )
        # ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
        # WITH SERDEPROPERTIES (
        #     'serialization.format' = ',',
        # 'field.delim' = ','
        # )
        # LOCATION 's3://sales-data-dummy-athena-bucket/contracts/'
        # TBLPROPERTIES (
        #     'has_encrypted_data'='false',
        # 'skip.header.line.count'='1'
        # );

# -> invoices csv
#
        # -- Create invoices table in dummy_sales_data database
        # CREATE EXTERNAL TABLE dummy_sales_data.invoices (
        #     record_id INT,
        # user_id INT,
        # record_date DATE,
        # invoice_amount_usd DECIMAL(10,2),
        # status STRING
        # )
        # ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
        # WITH SERDEPROPERTIES (
        #     'serialization.format' = ',',
        # 'field.delim' = ','
        # )
        # LOCATION 's3://sales-data-dummy-athena-bucket/invoices/'
        # TBLPROPERTIES (
        #     'has_encrypted_data'='false',
        # 'skip.header.line.count'='1'
        # );
