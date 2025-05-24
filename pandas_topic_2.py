# pandas_topic_2.py
# Topic: Data Cleaning and Preparation using Pandas

import pandas as pd
import numpy as np # For creating NaN values for examples

# -----------------------------------------------------------------------------
# Introduction to Data Cleaning and Preparation
# -----------------------------------------------------------------------------
# Data cleaning and preparation are crucial steps in any data analysis workflow.
# Real-world data is often messy: it can have missing values, incorrect data types,
# duplicates, or inconsistent formatting. Pandas provides a powerful suite of tools
# to handle these issues efficiently.

# -----------------------------------------------------------------------------
# Sample DataFrame for Examples
# -----------------------------------------------------------------------------
# We'll use this DataFrame for most examples in this topic.
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Alice', np.nan, 'Grace'],
    'Age': [25, 30, np.nan, 28, 22, 30, 25, 35, 40],
    'Salary': [50000, 60000, 70000, 55000, np.nan, 60000, 50000, 80000, 90000.50],
    'City': ['New York', 'Paris', 'London', 'Berlin', 'Tokyo', 'Paris', 'New York', 'Berlin', 'London'],
    'JoinDate': ['2020-01-10', '2019-05-15', '2021-02-20', '2020-08-01', '2022-01-30', '2019-05-15', '2020-01-10', '2018-11-11', '2017-07-07'],
    'Experience': ['5 years', '7 years', '3 years', '6 years', '2 years', '7 years', '5 years', '10 years', '12 years']
}
df_initial = pd.DataFrame(data)
print("--- Initial Sample DataFrame ---")
print(df_initial)
print("\n")

# It's good practice to make a copy when performing cleaning operations
df = df_initial.copy()

# -----------------------------------------------------------------------------
# 1. Identifying Missing Data
# -----------------------------------------------------------------------------
# Pandas uses NaN (Not a Number) to represent missing data by default.

print("--- 1. Identifying Missing Data ---")
# .isnull() or .isna(): Return a boolean DataFrame indicating missing values
print("\nDataFrame of boolean values indicating missing data (isnull()):")
print(df.isnull())

print("\nDataFrame of boolean values indicating missing data (isna()):")
print(df.isna()) # isna() is an alias for isnull()

# To get a count of missing values per column:
print("\nCount of missing values per column:")
print(df.isnull().sum())

# To get the total number of missing values in the DataFrame:
print(f"\nTotal missing values in DataFrame: {df.isnull().sum().sum()}")

# To check if any value is missing in a column (returns a boolean Series):
print("\nCheck if any value is missing per column (any()):")
print(df.isnull().any())

# To check if all values are missing in a column:
print("\nCheck if all values are missing per column (all()):")
print(df.isnull().all()) # Will be False for all columns in this example

# -----------------------------------------------------------------------------
# 2. Handling Missing Data
# -----------------------------------------------------------------------------
print("\n--- 2. Handling Missing Data ---")

# --- a) Removing Missing Data (dropna()) ---
# .dropna() can drop rows or columns containing missing values.

# Drop rows with any missing values
df_dropped_rows = df.dropna() # Default is axis=0 (rows), how='any'
print("\nDataFrame after dropping rows with any missing values:")
print(df_dropped_rows)

# Drop columns with any missing values
df_dropped_cols = df.dropna(axis=1) # axis=1 for columns
print("\nDataFrame after dropping columns with any missing values:")
print(df_dropped_cols) # Name, Age, Salary will be dropped

# .dropna() parameters:
# - axis: {0 or 'index', 1 or 'columns'}, default 0.
# - how: {'any', 'all'}, default 'any'. If 'any', drop if any NA values are present.
#        If 'all', drop if all values are NA.
# - thresh: int, optional. Require that many non-NA values.
# - subset: array-like, optional. Labels along other axis to consider,
#           e.g., if you want to drop rows based on NAs in specific columns.

# Example: Drop rows if 'Name' or 'Salary' is NaN
df_dropped_subset = df.dropna(subset=['Name', 'Salary'])
print("\nDataFrame after dropping rows if 'Name' or 'Salary' is NaN:")
print(df_dropped_subset)

# --- b) Filling Missing Data (fillna()) ---
# .fillna() can replace missing values with a specified value or method.

# Fill all NaN values with a scalar value (e.g., 0 or a specific string)
df_filled_scalar = df.copy()
df_filled_scalar['Age'] = df_filled_scalar['Age'].fillna(0) # Fill NaN in Age with 0
df_filled_scalar['Salary'] = df_filled_scalar['Salary'].fillna(df_filled_scalar['Salary'].mean()) # Fill NaN Salary with mean
df_filled_scalar['Name'] = df_filled_scalar['Name'].fillna('Unknown') # Fill NaN Name with 'Unknown'
print("\nDataFrame after filling NaN in 'Age' with 0, 'Salary' with mean, 'Name' with 'Unknown':")
print(df_filled_scalar)

# Fill with method: 'ffill' (forward-fill) or 'bfill' (backward-fill)
df_filled_ffill = df.copy()
# Forward-fill propagates the last valid observation forward
df_filled_ffill['Age'] = df_filled_ffill['Age'].fillna(method='ffill')
print("\nDataFrame after forward-filling NaN in 'Age':")
print(df_filled_ffill[['Name', 'Age']]) # Charlie's Age becomes Bob's Age (30)

df_filled_bfill = df.copy()
# Backward-fill propagates the next valid observation backward
df_filled_bfill['Age'] = df_filled_bfill['Age'].fillna(method='bfill')
print("\nDataFrame after backward-filling NaN in 'Age':")
print(df_filled_bfill[['Name', 'Age']]) # Charlie's Age becomes David's Age (28)

# .fillna() can also be applied to the entire DataFrame or specific columns.
# It's often good practice to fill missing values based on the column's meaning.
# E.g., mean/median for numerical, mode for categorical, or a placeholder like 'Unknown'.

# Example: Filling 'Age' with median and 'Salary' with mean from the original df
age_median = df['Age'].median()
salary_mean = df['Salary'].mean()
df_filled_stats = df.copy()
df_filled_stats['Age'].fillna(age_median, inplace=True) # inplace=True modifies the DataFrame directly
df_filled_stats['Salary'].fillna(salary_mean, inplace=True)
print(f"\nDataFrame after filling 'Age' with median ({age_median}) and 'Salary' with mean ({salary_mean:.2f}):")
print(df_filled_stats[['Name', 'Age', 'Salary']])


# -----------------------------------------------------------------------------
# 3. Converting Data Types
# -----------------------------------------------------------------------------
print("\n--- 3. Converting Data Types ---")
# It's common to find columns with incorrect data types (e.g., numbers stored as strings).
# .astype() is the primary method for type conversion.

print("\nOriginal data types:")
print(df.dtypes)

df_typed = df.copy()
# Let's first fill NaN values in 'Age' and 'Salary' as astype can error with NaNs for some conversions (e.g., float to int)
df_typed['Age'].fillna(df_typed['Age'].median(), inplace=True)
df_typed['Salary'].fillna(df_typed['Salary().mean(), inplace=True)

# Convert 'Age' from float64 (due to NaN) to int64
# Note: If 'Age' still had NaNs, converting to int would raise an error.
# Pandas 1.0+ introduced pd.Int64Dtype (nullable integer type) which can handle NaNs.
try:
    df_typed['Age'] = df_typed['Age'].astype(int)
    print("\n'Age' column converted to int:")
    print(df_typed.dtypes)
except ValueError as e:
    print(f"\nError converting 'Age' to int (likely due to NaNs if not handled): {e}")
    print("Using pd.Int64Dtype for nullable integers:")
    df_typed['Age'] = df_typed['Age'].astype(pd.Int64Dtype()) # Handles potential NaNs
    print(df_typed.dtypes)


# Convert 'Salary' to integer
df_typed['Salary'] = df_typed['Salary'].astype(int)
print("\n'Salary' column converted to int:")
print(df_typed.dtypes)

# Convert 'JoinDate' from object (string) to datetime
df_typed['JoinDate'] = pd.to_datetime(df_typed['JoinDate'])
print("\n'JoinDate' column converted to datetime:")
print(df_typed.dtypes)
print("\nDataFrame with corrected types (JoinDate as datetime):")
print(df_typed[['Name', 'Age', 'Salary', 'JoinDate']].head())

# Common type conversions:
# - int, float, str
# - 'category' for categorical data (can save memory and enable specific operations)
# - pd.to_datetime() for dates and times
# - pd.to_numeric() for converting to numbers (can handle errors by setting errors='coerce' to turn problematic values into NaN)

# Example with pd.to_numeric and errors='coerce'
temp_series = pd.Series(['10', '20', 'thirty', '40', 'NaN'])
print(f"\nOriginal series for to_numeric: {temp_series.values}")
numeric_series = pd.to_numeric(temp_series, errors='coerce')
print(f"Series after pd.to_numeric(errors='coerce'): {numeric_series.values}, dtype: {numeric_series.dtype}")


# -----------------------------------------------------------------------------
# 4. Identifying and Removing Duplicate Rows
# -----------------------------------------------------------------------------
print("\n--- 4. Identifying and Removing Duplicate Rows ---")
# Duplicate data can skew analysis results.

# .duplicated(): Returns a boolean Series indicating duplicate rows.
# By default, it marks all occurrences of a duplicate except the first one as True.
print("\nBoolean Series indicating duplicate rows (keeping first):")
print(df.duplicated()) # Alice (row 6) is a duplicate of Alice (row 0)
                     # Frank (row 5) has a duplicate Age and Salary as Bob (row 1) but other fields differ.
                     # So only (Alice, 25, 50000, New York, 2020-01-10, 5 years) is a full duplicate.

# To see the duplicate rows:
print("\nDuplicate rows (keeping first by default):")
print(df[df.duplicated()])

# Parameters for .duplicated():
# - subset: list-like, optional. Only consider certain columns for identifying duplicates.
# - keep: {'first', 'last', False}, default 'first'.
#   - 'first': Mark duplicates as True except for the first occurrence.
#   - 'last': Mark duplicates as True except for the last occurrence.
#   - False: Mark all duplicates as True.

print("\nDuplicate rows (keeping last):")
print(df[df.duplicated(keep='last')]) # Now Alice (row 0) is marked as duplicate

print("\nAll occurrences of duplicate rows (keep=False):")
print(df[df.duplicated(keep=False)]) # Both Alice rows are shown

print("\nIdentifying duplicates based on 'Name' and 'City' columns:")
print(df[df.duplicated(subset=['Name', 'City'], keep=False)])


# .drop_duplicates(): Returns a DataFrame with duplicate rows removed.
# Uses the same parameters as .duplicated() for 'subset' and 'keep'.
df_no_duplicates = df.drop_duplicates()
print("\nDataFrame after removing duplicates (keeping first):")
print(df_no_duplicates) # Alice at index 6 is removed.

df_no_duplicates_last = df.drop_duplicates(keep='last')
print("\nDataFrame after removing duplicates (keeping last):")
print(df_no_duplicates_last) # Alice at index 0 is removed.

df_no_duplicates_subset = df.drop_duplicates(subset=['Name', 'Age'], keep='first')
print("\nDataFrame after removing duplicates based on 'Name' and 'Age' (keeping first):")
print(df_no_duplicates_subset) # Frank is still there as his Name/Age combo is unique even if Salary/City is not.

# -----------------------------------------------------------------------------
# 5. String Manipulation on Text Data
# -----------------------------------------------------------------------------
print("\n--- 5. String Manipulation on Text Data ---")
# Pandas Series have a .str accessor that provides vectorized string methods.

print("\nOriginal 'Name' column:")
print(df['Name'])

# .str.lower() / .str.upper(): Convert strings to lowercase/uppercase.
df_copy_str = df.copy()
# Handle potential NaN values in string columns before applying string methods,
# otherwise some methods might error or behave unexpectedly.
df_copy_str['Name'] = df_copy_str['Name'].fillna('Unknown')
df_copy_str['Experience'] = df_copy_str['Experience'].fillna('0 years')


df_copy_str['Name_Lower'] = df_copy_str['Name'].str.lower()
print("\n'Name' column converted to lowercase:")
print(df_copy_str[['Name', 'Name_Lower']])

# .str.replace(pat, repl): Replace occurrences of pattern/text 'pat' with 'repl'.
# Example: Remove " years" from 'Experience' column and convert to number
df_copy_str['Experience_Num'] = df_copy_str['Experience'].str.replace(' years', '').astype(int)
print("\n'Experience' column, ' years' removed and converted to int:")
print(df_copy_str[['Experience', 'Experience_Num']])

# .str.split(pat, expand=False): Split strings around a delimiter.
# 'expand=True' returns a DataFrame.
df_copy_str['City_Split'] = df_copy_str['City'].str.split(' ') # Example: split cities with spaces (e.g., "New York")
print("\n'City' column split by space (results in lists):")
print(df_copy_str[['City', 'City_Split']])

# Example: Splitting 'JoinDate' (after converting to string)
df_copy_str['JoinDate_Str'] = df['JoinDate'].astype(str) # Convert datetime back to string for this example
date_parts = df_copy_str['JoinDate_Str'].str.split('-', expand=True)
date_parts.columns = ['Year', 'Month', 'Day']
print("\n'JoinDate' split into Year, Month, Day columns:")
print(date_parts.head())

# Other useful .str methods:
# - .str.contains(pat): Check for occurrence of pattern.
# - .str.startswith(pat) / .str.endswith(pat): Check for prefix/suffix.
# - .str.strip() / .str.lstrip() / .str.rstrip(): Remove leading/trailing whitespace.
# - .str.len(): Get the length of each string.
# - .str.extract(pat): Extract capture groups using a regex pattern.

print("\nChecking if 'Name' contains 'li' (case-sensitive):")
print(df_copy_str['Name'].str.contains('li'))

print("\nExtracting the first word of 'City':")
print(df_copy_str['City'].str.extract(r'(\w+)')[0]) # Access column 0 of the new DataFrame

# It's important to handle NaN values when using .str methods, as they can
# result in errors or NaN outputs. Using .fillna() before string operations
# or checking for NaNs is a good practice.
# Most .str methods will propagate NaNs, e.g. np.nan.lower() is still np.nan

# -----------------------------------------------------------------------------
# End of Data Cleaning and Preparation Topic
# -----------------------------------------------------------------------------
print("\nEnd of pandas_topic_2.py: Data Cleaning and Preparation")

# Clean up the DataFrame copy if needed for memory, though Python's garbage collector handles it.
del df, df_initial, df_copy_str
del df_dropped_rows, df_dropped_cols, df_dropped_subset
del df_filled_scalar, df_filled_ffill, df_filled_bfill, df_filled_stats
del df_typed, date_parts, numeric_series, temp_series
del df_no_duplicates, df_no_duplicates_last, df_no_duplicates_subset
print("\nCleaned up intermediate DataFrames.")
