import pandas as pd
import random
from datetime import datetime, timedelta

# Simulate monitoring log data
species = ["setosa", "versicolor", "virginica"]
logs = []

# Generate 50 mock entries
start_time = datetime.now() - timedelta(days=1)
for i in range(50):
    timestamp = start_time + timedelta(minutes=i * 30)
    input_data = [round(random.uniform(4.0, 7.5), 2) for _ in range(4)]
    prediction = random.choice(species)
    logs.append({
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "sepal_length": input_data[0],
        "sepal_width": input_data[1],
        "petal_length": input_data[2],
        "petal_width": input_data[3],
        "prediction": prediction
    })

# Save to CSV
monitoring_df = pd.DataFrame(logs)
monitoring_df.to_csv("monitoring/monitoring_logs.csv", index=False)
print("Monitoring logs saved to monitoring/monitoring_logs.csv")

