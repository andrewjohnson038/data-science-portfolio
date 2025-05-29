import pandas as pd

# Team members data
team_members = {
    "name": [
        "John Doe"
    ],
    "email": [
        "JohnDoe@gmail.com"
    ]
}

# Create DataFrame
df = pd.DataFrame(team_members)

# Save to CSV
df.to_csv('team_members.csv', index=False)
print("CSV file 'team_members.csv' has been created successfully!") 