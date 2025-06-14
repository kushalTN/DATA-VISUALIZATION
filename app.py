import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("/content/Highest Holywood Grossing Movies.csv")
df.head()
df.info()
df.describe()

# Clean currency columns
currency_cols = ['Budget (in $)', 'Domestic Opening (in $)']
for col in currency_cols:  
df[col] = df[col].replace('[^0-9]', '', regex=True).replace('', np.nan).astype(float)
 
# Fill missing values
df['Distributor'].fillna('Unknown', inplace=True)
df['License'].fillna('Unrated', inplace=True)
df[currency_cols] = df[currency_cols].fillna(df[currency_cols].median())

# Prepare feature set and target
features = df[['Budget (in $)', 'Domestic Opening (in $)', 'Domestic Sales (in $)', 'International Sales (in $)']]
target = df['World Wide Sales (in $)']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Convert Running Time to minutes
df['Running Time (min)'] = df['Running Time'].str.extract(r'(\d+)\s*hr.*?(\d+)?\s*min').fillna(0).astype(int).sum(axis=1)

# Plotting
plt.figure(figsize=(20, 25))

# 1. Bar plot - Top 10 movies by worldwide sales
plt.subplot(3, 2, 1)
top10 = df.sort_values(by='World Wide Sales (in $)', ascending=False).head(10)
sns.barplot(data=top10, x='World Wide Sales (in $)', y='Title', palette='viridis')
plt.title('Top 10 Movies by Worldwide Sales')

# 2. Pie chart - Distribution by License
plt.subplot(3, 2, 2)
license_counts = df['License'].value_counts()
plt.pie(license_counts, labels=license_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Movie Licenses')

# 3. Line plot - Movies released per year
plt.subplot(3, 2, 3)
yearly_count = df['Year'].value_counts().sort_index()
sns.lineplot(x=yearly_count.index, y=yearly_count.values)
plt.title('Movies Released per Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')

# 4. Scatter plot - Budget vs Worldwide Sales
plt.subplot(3, 2, 4)
sns.scatterplot(data=df, x='Budget (in $)', y='World Wide Sales (in $)', hue='License')
plt.title('Budget vs Worldwide Sales')


# 5. Histogram - Running time distribution
plt.subplot(3, 2, 5)
sns.histplot(df['Running Time (min)'], bins=20, kde=True)
plt.title('Distribution of Movie Running Time')

plt.tight_layout()
plt.show()
