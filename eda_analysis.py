# STEP 1: Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# STEP 2: Load the Dataset

df = pd.read_csv('your_dataset.csv')

# STEP 3: Basic Data Info

print("First 5 rows:\n", df.head())
print("Shape of dataset:", df.shape)
print("Column names:", df.columns.tolist())
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())
print("Dataset info:")
df.info()

# STEP 4: Summary Statistics

print("Summary stats (numeric):\n", df.describe())
print("Summary stats (categorical):\n", df.describe(include='object'))

# STEP 5: Histograms for Numeric Features

df.hist(bins=30, figsize=(15, 10), edgecolor='black')
plt.suptitle("Histograms of Numeric Columns", fontsize=16)
plt.tight_layout()
plt.show()

# STEP 6: Boxplots to Detect Outliers

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

# STEP 7: Pairplot and Correlation Heatmap

try:
    sns.pairplot(df.select_dtypes(include=np.number))
    plt.suptitle("Pairplot", fontsize=16)
    plt.show()
except:
    print("Pairplot too large to display.")

# Correlation Matrix
corr = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

# STEP 8: Categorical Feature Analysis

categorical_cols = df.select_dtypes(include='object').columns

for cat_col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=cat_col, data=df)
    plt.title(f"Count plot of {cat_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Boxplots for numeric vs categorical
for cat_col in categorical_cols:
    for num_col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=cat_col, y=num_col, data=df)
        plt.title(f"{num_col} by {cat_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# STEP 9: Interactive Plots (Optional with Plotly)

if len(numeric_cols) >= 2:
    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=categorical_cols[0] if len(categorical_cols) > 0 else None)
    fig.show()

if len(numeric_cols) >= 1:
    fig = px.histogram(df, x=numeric_cols[0], color=categorical_cols[0] if len(categorical_cols) > 0 else None)
    fig.show()

# STEP 10: Save Cleaned or Modified Dataset (Optional)


