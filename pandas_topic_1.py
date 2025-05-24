# pandas_topic_1.py
# Topic: Pandas DataFrames

import pandas as pd

# -----------------------------------------------------------------------------
# What is a Pandas DataFrame?
# -----------------------------------------------------------------------------
# A Pandas DataFrame is a two-dimensional, size-mutable, and potentially heterogeneous
# tabular data structure with labeled axes (rows and columns). It is similar to a
# spreadsheet, a SQL table, or a dictionary of Series objects. It is one of the
# most commonly used Pandas objects.

# Key characteristics:
# - Two-dimensional: Data is aligned in rows and columns.
# - Heterogeneous data: Columns can have different data types (e.g., integer, float, string).
# - Labeled axes: Rows and columns have labels (index for rows, column names for columns).
# - Size-mutable: You can add or remove columns and rows.

# -----------------------------------------------------------------------------
# Creating DataFrames
# -----------------------------------------------------------------------------

# --- From a Dictionary of Lists/Series ---
# The keys of the dictionary become column names, and the values (lists or Series)
# become the data in the columns. All lists/Series must be of the same length.
print("\n--- Creating DataFrame from Dictionary ---")
data_dict = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'Paris', 'London', 'Berlin']
}
df_from_dict = pd.DataFrame(data_dict)
print(df_from_dict)

# You can also specify an index:
df_from_dict_indexed = pd.DataFrame(data_dict, index=['person1', 'person2', 'person3', 'person4'])
print("\nDataFrame from Dictionary with custom index:")
print(df_from_dict_indexed)


# --- From a List of Lists ---
# Each inner list represents a row in the DataFrame.
# You need to specify column names separately if you want them.
print("\n--- Creating DataFrame from List of Lists ---")
data_lol = [
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'Paris'],
    ['Charlie', 35, 'London'],
    ['David', 28, 'Berlin']
]
column_names = ['Name', 'Age', 'City']
df_from_lol = pd.DataFrame(data_lol, columns=column_names)
print(df_from_lol)

# --- From a List of Dictionaries ---
# Each dictionary in the list represents a row. Keys become column names.
# Pandas will infer column names from the keys and fill missing values with NaN.
print("\n--- Creating DataFrame from List of Dictionaries ---")
data_lod = [
    {'Name': 'Eve', 'Age': 22, 'Country': 'USA'},
    {'Name': 'Frank', 'Age': 29, 'City': 'Toronto'}, # Note 'City' instead of 'Country'
    {'Name': 'Grace', 'Age': 31, 'Country': 'UK', 'Occupation': 'Engineer'}
]
df_from_lod = pd.DataFrame(data_lod)
print(df_from_lod) # Notice NaN for missing values and all columns present


# --- From a CSV File ---
# This is a very common way to create DataFrames.
# We'll create a dummy CSV file first for this example.
sample_csv_data = """ID,Name,Score
1,Anna,85
2,Brian,90
3,Cindy,78
"""
with open("sample_data.csv", "w") as f:
    f.write(sample_csv_data)

print("\n--- Creating DataFrame from CSV File ---")
# Assuming 'sample_data.csv' is in the same directory
try:
    df_from_csv = pd.read_csv("sample_data.csv")
    print(df_from_csv)
except FileNotFoundError:
    print("sample_data.csv not found. Please create it to run this example.")

# Common parameters for pd.read_csv():
# - filepath_or_buffer: Path to the CSV file or URL.
# - sep: Delimiter to use (default is ',').
# - header: Row number to use as column names (default is 0, first row).
# - index_col: Column to use as the row labels (index).
# - usecols: List of columns to read.
# - dtype: Dictionary to specify data types for columns.

# -----------------------------------------------------------------------------
# Basic DataFrame Attributes and Methods
# -----------------------------------------------------------------------------
print("\n--- Basic DataFrame Attributes and Methods ---")
# Using df_from_dict for these examples
print("Original DataFrame:")
print(df_from_dict)

# .shape: Returns a tuple representing the dimensionality (rows, columns)
print(f"\nShape of the DataFrame: {df_from_dict.shape}") # (4, 3)

# .dtypes: Returns the data type of each column
print("\nData types of columns:")
print(df_from_dict.dtypes)

# .index: Returns the index (row labels) of the DataFrame
print("\nIndex of the DataFrame:")
print(df_from_dict.index)

# .columns: Returns the column labels of the DataFrame
print("\nColumns of the DataFrame:")
print(df_from_dict.columns)

# .values: Returns a NumPy representation of the DataFrame's data
print("\nValues (NumPy array) of the DataFrame:")
print(df_from_dict.values)

# .head(n): Returns the first n rows (default is 5)
print("\nFirst 2 rows (head(2)):")
print(df_from_dict.head(2))

# .tail(n): Returns the last n rows (default is 5)
print("\nLast 2 rows (tail(2)):")
print(df_from_dict.tail(2))

# .info(): Provides a concise summary of the DataFrame, including dtypes and memory usage
print("\nDataFrame info:")
df_from_dict.info()

# .describe(): Generates descriptive statistics of the DataFrame
# For numerical columns: count, mean, std, min, max, and percentiles.
# For object/string columns (if include='object' or include='all'): count, unique, top, freq.
print("\nDescriptive statistics (describe()):")
print(df_from_dict.describe()) # Describes numeric columns by default
print("\nDescriptive statistics for object columns (describe(include='object')):")
print(df_from_dict.describe(include='object'))
print("\nDescriptive statistics for all columns (describe(include='all')):")
print(df_from_dict.describe(include='all'))


# -----------------------------------------------------------------------------
# Indexing and Selecting Data
# -----------------------------------------------------------------------------
print("\n--- Indexing and Selecting Data ---")
# Using the following DataFrame for these examples:
data_select = {
    'col1': [10, 20, 30, 40, 50],
    'col2': ['A', 'B', 'C', 'D', 'E'],
    'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
}
df_select = pd.DataFrame(data_select, index=['row_a', 'row_b', 'row_c', 'row_d', 'row_e'])
print("DataFrame for selection examples:")
print(df_select)

# --- Selecting Columns ---
# Using square brackets []:
print("\nSelecting a single column ('col2') as a Series:")
print(df_select['col2'])
print(f"Type: {type(df_select['col2'])}")

print("\nSelecting multiple columns as a DataFrame:")
print(df_select[['col1', 'col3']])
print(f"Type: {type(df_select[['col1', 'col3']])}")

# Using dot notation (only for valid Python variable names, no spaces, etc.):
print("\nSelecting a single column ('col2') using dot notation:")
print(df_select.col2) # Less flexible, e.g., if column name has spaces or is a method name


# --- Selecting Rows ---

# .loc[]: Label-based selection
# Selects rows and columns by their labels (index names and column names).
print("\n--- .loc[] (Label-based selection) ---")
print("\nSelecting a single row ('row_b') by label:")
print(df_select.loc['row_b']) # Returns a Series

print("\nSelecting multiple rows by label (['row_a', 'row_c']):")
print(df_select.loc[['row_a', 'row_c']]) # Returns a DataFrame

print("\nSelecting a range of rows by label ('row_b' to 'row_d'):")
print(df_select.loc['row_b':'row_d']) # Inclusive of both start and end labels

print("\nSelecting specific rows and columns by label:")
print(df_select.loc[['row_a', 'row_e'], ['col1', 'col3']])

print("\nSelecting a single value (scalar) by row and column label:")
print(df_select.loc['row_c', 'col2'])


# .iloc[]: Integer position-based selection
# Selects rows and columns by their integer positions (0-based).
print("\n--- .iloc[] (Integer position-based selection) ---")
print("\nSelecting the first row (index 0):")
print(df_select.iloc[0]) # Returns a Series

print("\nSelecting multiple rows by integer position ([1, 3]):")
print(df_select.iloc[[1, 3]]) # Returns a DataFrame (rows at index 1 and 3)

print("\nSelecting a range of rows by integer position (1 to 3, exclusive of 3):")
print(df_select.iloc[1:3]) # Returns a DataFrame (rows at index 1 and 2)

print("\nSelecting specific rows and columns by integer position:")
print(df_select.iloc[[0, 4], [0, 2]]) # Rows 0 & 4, Columns 0 & 2

print("\nSelecting a single value (scalar) by row and column integer position:")
print(df_select.iloc[2, 1]) # Value at row index 2, column index 1


# --- Conditional Selection (Boolean Indexing) ---
# Using boolean Series to filter data.
print("\n--- Conditional Selection (Boolean Indexing) ---")
print("\nOriginal DataFrame for conditional selection:")
print(df_from_dict)

print("\nSelecting rows where Age > 28:")
print(df_from_dict[df_from_dict['Age'] > 28])

print("\nSelecting rows where City is 'New York' or 'London':")
condition = (df_from_dict['City'] == 'New York') | (df_from_dict['City'] == 'London')
print(df_from_dict[condition])

print("\nSelecting 'Name' and 'City' for people where Age < 30:")
print(df_from_dict.loc[df_from_dict['Age'] < 30, ['Name', 'City']])


# -----------------------------------------------------------------------------
# End of Pandas DataFrames Topic
# -----------------------------------------------------------------------------
print("\nEnd of pandas_topic_1.py: Pandas DataFrames")

# To clean up the dummy CSV file created earlier:
import os
if os.path.exists("sample_data.csv"):
    os.remove("sample_data.csv")
    print("\nCleaned up sample_data.csv")
