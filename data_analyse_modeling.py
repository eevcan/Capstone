import os
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import matplotlib
import matplotlib.pyplot as plt

def set_pandas_display_options(max_rows=50, max_columns=50):
    """
    Sets the pandas display options for showing the number of rows and columns.
    
    Parameters:
    max_rows (int): The maximum number of rows to display in the DataFrame.
    max_columns (int): The maximum number of columns to display in the DataFrame.
    """
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)

def suppress_warnings():
    """
    Suppresses warnings to avoid clutter in the output.
    """
    warnings.filterwarnings('ignore')

def initialize_plotting():
    """
    Initializes matplotlib and seaborn settings for better visualizations.
    """
    sns.set(style="whitegrid")
    matplotlib.rcParams['figure.figsize'] = (10, 6)

# Example of calling the functions
set_pandas_display_options()
suppress_warnings()
initialize_plotting()

# Get the current working directory and build the file paths
base_path = os.path.dirname(os.path.realpath(__file__))
df1_path = os.path.join(base_path, "data", "cards.csv")
df2_path = os.path.join(base_path, "data", "cardPrices.csv")

# Load CSV files
df1 = pd.read_csv(df1_path, dtype={"uuid": str})
df2 = pd.read_csv(df2_path, dtype={"uuid": str})

# Ensure uuid has no leading/trailing spaces
df1["uuid"] = df1["uuid"].str.strip()
df2["uuid"] = df2["uuid"].str.strip()

# Left join on "uuid"
merged_df = df2.merge(df1, on="uuid", how="left")

# Check if 'price' column exists
if "price" in merged_df.columns:
    print("Column 'price' exists in merged_df.")
else:
    print("Column 'price' does not exist in merged_df.")

def get_dataframe_shape(df):
    """
    Returns the shape of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame whose shape is to be queried.
    
    Returns:
    tuple: A tuple containing the number of rows and columns of the DataFrame.
    """
    return df.shape

# Example: Calling the function to get the shape of the DataFrame
shape = get_dataframe_shape(df1)
print(f"Shape of DataFrame: {shape}")

# Display first 5 rows of df1
print(df1.head())

# Plot count of cards by color
color_counts = merged_df["colors"].value_counts()

# Plot the data
plt.figure(figsize=(10, 6))
sns.barplot(x=color_counts.index, y=color_counts.values, palette="viridis")

# Customize plot
plt.xlabel("Colors")
plt.ylabel("Count")
plt.title("Count of Cards by Color")
plt.xticks(rotation=45)
plt.show()

# Ensure price column is numeric
merged_df["price"] = pd.to_numeric(merged_df["price"], errors="coerce")

# If colors are stored as lists, explode them
merged_df_exploded = merged_df.explode("colors")

# Calculate the average price per color
avg_price_per_color = merged_df_exploded.groupby("colors")["price"].mean().dropna()

# Plot the data
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_price_per_color.index, y=avg_price_per_color.values, palette="coolwarm")

# Customize plot
plt.xlabel("Colors")
plt.ylabel("Average Price")
plt.title("Average Price of Cards by Color")
plt.xticks(rotation=45)
plt.show()

# Convert frameVersion to numeric
merged_df["frameVersion"] = pd.to_numeric(merged_df["frameVersion"], errors="coerce")

# Drop NaN values (if there are invalid years)
merged_df_cleaned = merged_df.dropna(subset=["frameVersion"])

# Convert to integer
merged_df_cleaned["frameVersion"] = merged_df_cleaned["frameVersion"].astype(int)

# Count occurrences of each frameVersion
frame_counts = merged_df_cleaned["frameVersion"].value_counts().sort_index()

# Plot the data
plt.figure(figsize=(12, 6))
sns.barplot(x=frame_counts.index, y=frame_counts.values, palette="magma")

# Customize plot
plt.xlabel("Frame Version (Year)")
plt.ylabel("Count of Cards")
plt.title("Number of Cards by Frame Version (Year)")
plt.xticks(rotation=45)
plt.show()

# Ensure "price" is numeric
merged_df["price"] = pd.to_numeric(merged_df["price"], errors="coerce")

# Remove invalid rows (NaN values)
merged_df_cleaned = merged_df.dropna(subset=["frameVersion", "price"])

# Calculate average price per year
avg_price_per_year = merged_df_cleaned.groupby("frameVersion")["price"].mean()

# Plot the data
plt.figure(figsize=(12, 6))
sns.lineplot(x=avg_price_per_year.index, y=avg_price_per_year.values, marker="o", color="b")

# Customize plot
plt.xlabel("Frame Version (Year)")
plt.ylabel("Average Price")
plt.title("Average Card Price by Frame Version (Year)")
plt.xticks(rotation=45)
plt.grid(True)

plt.show()

# Count the number of cards by rarity
rarity_counts = merged_df["rarity"].value_counts()

# Plot the data
plt.figure(figsize=(10, 6))
sns.barplot(x=rarity_counts.index, y=rarity_counts.values, palette="muted")

# Customize plot
plt.xlabel("Rarity")
plt.ylabel("Count of Cards")
plt.title("Number of Cards by Rarity")
plt.xticks(rotation=45)
plt.show()

# Remove invalid rows (NaN values)
merged_df_cleaned = merged_df.dropna(subset=["rarity", "price"])

# Calculate average price per rarity
avg_price_per_rarity = merged_df_cleaned.groupby("rarity")["price"].mean()

# Plot the data
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_price_per_rarity.index, y=avg_price_per_rarity.values, palette="Blues_d")

# Customize plot
plt.xlabel("Rarity")
plt.ylabel("Average Price")
plt.title("Average Card Price by Rarity")
plt.xticks(rotation=45)
plt.show()
