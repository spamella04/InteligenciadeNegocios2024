import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("realtor-data.csv")

print(df.head())
print("\n")

print(df.info())

top_15 = df.nlargest(15, 'price')
plt.figure(figsize=(10, 6))
sns.boxplot(x='bed',y='price',data=top_15)
plt.title('Boxplot of DataFrame Columns')
plt.xticks(rotation=0)
plt.show()

mean = df['price'].mean()

# Calculate the median
median = df['price'].median()

# Calculate the mode
mode = df['price'].mode()[0]

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['price'], color='lightblue')

# Add lines for mean, median, and mode
plt.axvline(mean, color='red', linestyle='--', label='Mean')
plt.axvline(median, color='green', linestyle='--', label='Median')
plt.axvline(mode, color='orange', linestyle='--', label='Mode')

plt.title('Boxplot of House Prices with Mean, Median, and Mode')
plt.xlabel('Price ($)')
plt.legend()
plt.grid(axis='x', alpha=0.75)
plt.show()