# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:31:12 2024

@author: othamsuwan
"""

df = data_clean

# 2. Define the features for which you want descriptive statistics
# These should match the numeric columns from "Area" onwards in your dataset
features = [
    "Area", "Jerk X", "Jerk Y", "Jerk Z",
    "RMS Acceleration", "RMS Angular Velocity", "Ellipse Area", "Box Area",
    "Total Pathway Length", "volume", "Mean Acceleration X", "Mean Acceleration Y", "Mean Acceleration Z"
]

# 3. Compute descriptive statistics by sensor
# We group by 'Sensor' and use describe() to generate descriptive statistics for the specified features
# describe() calculates: count, mean, std, min, 25%, 50%, 75%, and max by default
grouped_stats = df.groupby('Sensor')[features].describe(percentiles=[0.25, 0.75])

# 4. The describe() method includes several statistics. We'll keep only the ones you requested:
# mean, std, min, 25% (Q1), 75% (Q3), and max. We can drop 'count' and '50%' (the median).
grouped_stats = grouped_stats.drop(columns=['count', '50%'], level=1)

# 5. Rename percentile columns to Q1 and Q3 for clarity
grouped_stats = grouped_stats.rename(columns={'25%': 'Q1', '75%': 'Q3'}, level=1)

# 6. Print the resulting descriptive statistics
print(grouped_stats)

# 7. (Optional) Save the descriptive statistics to a CSV file
grouped_stats.to_csv('descriptive_statistics_by_sensor.csv')

