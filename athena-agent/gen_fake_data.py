import os
import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

# Setup
fake = Faker()
output_dir = os.path.expanduser("dummy_data/")
os.makedirs(output_dir, exist_ok=True)

# Generate users.csv (30 unique users)
user_ids = list(range(1, 31))
users_data = [
    {
        "user_id": uid,
        "user_name": fake.name(),
        "email": fake.email(),
        "signup_date": fake.date_between(start_date="-1y", end_date="-3mo"),
        "country": fake.country()
    }
    for uid in user_ids
]
users_df = pd.DataFrame(users_data)
users_df.to_csv(os.path.join(output_dir, "users.csv"), index=False)


# Function to generate 100 rows of sales-related data from 2024–2025 up until today
def generate_sales_data(table_name, user_ids):
    start_date = datetime(2024, 1, 1)
    end_date = datetime.today()  # Up to today in 2025
    total_days = (end_date - start_date).days

    data = []
    for i in range(100):
        user_id = random.choice(user_ids)
        random_days = random.randint(0, total_days)
        record_date = start_date + timedelta(days=random_days)

        entry = {
            "record_id": i + 1,
            "user_id": user_id,
            "record_date": record_date.strftime("%Y-%m-%d"),
        }

        if table_name == "contracts":
            entry.update({
                "contract_value_usd": round(random.uniform(1000, 10000), 2),
                "contract_type": random.choice(["Annual", "Monthly", "One-Time"])
            })
        elif table_name == "sales":
            entry.update({
                "sale_amount_usd": round(random.uniform(50, 5000), 2),
                "product_category": random.choice(["Software", "Hardware", "Service"])
            })
        elif table_name == "renewals":
            entry.update({
                "renewal_term_months": random.choice([6, 12, 24]),
                "renewal_value_usd": round(random.uniform(200, 5000), 2)
            })
        elif table_name == "invoices":
            entry.update({
                "invoice_amount_usd": round(random.uniform(100, 8000), 2),
                "status": random.choice(["Paid", "Unpaid", "Overdue"])
            })
        elif table_name == "leads":
            entry.update({
                "lead_source": random.choice(["Web", "Referral", "Event", "Outbound"]),
                "lead_value_usd": round(random.uniform(50, 3000), 2)
            })

        data.append(entry)

    return pd.DataFrame(data)

# Generate and save sales-focused tables
generate_sales_data("contracts", user_ids).to_csv(os.path.join(output_dir, "contracts.csv"), index=False)
generate_sales_data("sales", user_ids).to_csv(os.path.join(output_dir, "sales.csv"), index=False)
generate_sales_data("renewals", user_ids).to_csv(os.path.join(output_dir, "renewals.csv"), index=False)
generate_sales_data("invoices", user_ids).to_csv(os.path.join(output_dir, "invoices.csv"), index=False)
generate_sales_data("leads", user_ids).to_csv(os.path.join(output_dir, "leads.csv"), index=False)

print("✅ CSV files saved to ~/Athena-agent/dummy_data/")
