# pandas_topic_8.py
# Topic: Advanced Indexing and Reshaping Data using Pandas

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Introduction
# -----------------------------------------------------------------------------
# Pandas provides powerful tools for advanced indexing, particularly through
# MultiIndex (hierarchical indexing), and for reshaping data between different
# formats (e.g., long to wide, wide to long). These capabilities are essential
# for preparing complex datasets for analysis.

# -----------------------------------------------------------------------------
# 1. MultiIndex (Hierarchical Indexing)
# -----------------------------------------------------------------------------
print("--- 1. MultiIndex (Hierarchical Indexing) ---")
# MultiIndex allows you to have multiple levels of indexes on an axis.
# This enables more sophisticated data analysis and manipulation.

# --- a) Creating MultiIndex DataFrames and Series ---
print("\n--- a) Creating MultiIndex DataFrames and Series ---")

# From tuples
arrays = [
    ['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
    ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']
]
index_tuples = list(zip(*arrays)) # Creates [('bar', 'one'), ('bar', 'two'), ...]
multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['first', 'second'])
s_multi = pd.Series(np.random.randn(8), index=multi_index)
print("\nSeries with MultiIndex created from tuples:")
print(s_multi)

df_multi_idx = pd.DataFrame(np.random.randn(8, 2), index=multi_index, columns=['A', 'B'])
print("\nDataFrame with MultiIndex (rows) created from tuples:")
print(df_multi_idx)

# From product of iterables
iterables = [['bar', 'baz', 'foo'], ['one', 'two']]
multi_index_prod = pd.MultiIndex.from_product(iterables, names=['first', 'second'])
df_multi_prod = pd.DataFrame(np.random.randn(6, 2), index=multi_index_prod, columns=['X', 'Y'])
print("\nDataFrame with MultiIndex created from product of iterables:")
print(df_multi_prod)

# By setting index with multiple columns
df_to_set_index = pd.DataFrame({
    'Group': ['A', 'A', 'A', 'B', 'B', 'B'],
    'Category': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
    'Value1': np.arange(6),
    'Value2': np.random.randn(6)
})
print("\nOriginal DataFrame for setting MultiIndex:")
print(df_to_set_index)
df_multi_set = df_to_set_index.set_index(['Group', 'Category'])
print("\nDataFrame with MultiIndex set from columns 'Group' and 'Category':")
print(df_multi_set)

# MultiIndex on columns
df_multi_cols = pd.DataFrame(np.random.randn(3, 4), index=['Row1', 'Row2', 'Row3'])
df_multi_cols.columns = pd.MultiIndex.from_product([['Metrics', 'Stats'], ['Mean', 'Std']], names=['Level1', 'Level2'])
print("\nDataFrame with MultiIndex on columns:")
print(df_multi_cols)


# --- b) Indexing and Slicing with MultiIndex ---
print("\n--- b) Indexing and Slicing with MultiIndex ---")
# Using df_multi_idx for these examples
print("\nSample DataFrame for MultiIndex indexing (df_multi_idx):")
print(df_multi_idx)

# Selecting from the outer level
print("\nSelecting outer level 'bar':")
print(df_multi_idx.loc['bar'])

# Selecting from both outer and inner levels (pass a tuple)
print("\nSelecting ('bar', 'one'):")
print(df_multi_idx.loc[('bar', 'one')]) # Returns a Series for a single row selection

# Partial indexing (slicing)
# Select all 'one' from the second level, across all 'first' level groups
print("\nSelecting all 'one' from the second level:")
print(df_multi_idx.loc[(slice(None), 'one'), :])
# Or more conveniently:
print(df_multi_idx.xs('one', level='second'))

# Slicing ranges
# For a sorted index, you can slice ranges. Let's ensure it's sorted.
df_multi_idx_sorted = df_multi_idx.sort_index()
print("\nSorted df_multi_idx for range slicing:")
print(df_multi_idx_sorted)
print("\nSlicing from ('bar', 'two') to ('baz', 'one'):")
print(df_multi_idx_sorted.loc[('bar', 'two'):('baz', 'one')])

# Selecting specific columns with MultiIndex rows
print("\nSelecting column 'A' for index ('foo', 'two'):")
print(df_multi_idx.loc[('foo', 'two'), 'A'])

# Indexing with MultiIndex on columns
print("\nDataFrame with MultiIndex on columns (df_multi_cols):")
print(df_multi_cols)
print("\nSelecting column ('Metrics', 'Mean'):")
print(df_multi_cols[('Metrics', 'Mean')])
print("\nSelecting column 'Stats':") # Selects all sub-columns under 'Stats'
print(df_multi_cols['Stats'])


# --- c) xs() method for cross-sectioning ---
# The .xs() method is useful for selecting data at a particular level of a MultiIndex.
print("\n--- c) xs() method for cross-sectioning ---")
print("\nUsing xs to select 'one' from level 'second':")
print(df_multi_idx.xs('one', level='second')) # Drops the selected level by default

print("\nUsing xs to select 'bar' from level 'first', keeping the level:")
print(df_multi_idx.xs('bar', level='first', drop_level=False))


# --- d) Sorting MultiIndex levels (sort_index()) ---
print("\n--- d) Sorting MultiIndex levels (sort_index()) ---")
df_unsorted = df_multi_idx.sample(frac=1) # Shuffle rows
print("\nUnsorted MultiIndex DataFrame:")
print(df_unsorted)

# Sort by index (default is to sort by all levels, starting from outermost)
df_sorted_all = df_unsorted.sort_index()
print("\nDataFrame sorted by all index levels (default):")
print(df_sorted_all)

# Sort by specific level
df_sorted_level1 = df_unsorted.sort_index(level='second') # Sorts by 'second' level primarily
print("\nDataFrame sorted by index level 'second':")
print(df_sorted_level1)

# Sort by multiple levels in a specific order
df_sorted_levels_custom = df_unsorted.sort_index(level=['second', 'first'])
print("\nDataFrame sorted by 'second' then 'first' index levels:")
print(df_sorted_levels_custom)


# --- e) Naming index levels (names) ---
print("\n--- e) Naming index levels (names) ---")
print(f"\nOriginal names of levels in df_multi_idx: {df_multi_idx.index.names}")
df_multi_idx.index.names = ['OuterGroup', 'InnerGroup'] # Rename levels
print(f"New names of levels: {df_multi_idx.index.names}")
print(df_multi_idx.head()) # Display with new names

# -----------------------------------------------------------------------------
# 2. Reshaping Data
# -----------------------------------------------------------------------------
print("\n--- 2. Reshaping Data ---")

# --- a) Stacking and Unstacking ---
print("\n--- a) Stacking and Unstacking ---")
# Stacking pivots a level of the column labels to become an index level (makes DataFrame taller/narrower).
# Unstacking pivots a level of the row index to become column labels (makes DataFrame wider/shorter).

# Using df_multi_cols (MultiIndex on columns) for unstacking example
print("\nDataFrame with MultiIndex on columns (df_multi_cols) for stack/unstack:")
print(df_multi_cols)

# Stack: Pivots the innermost column level ('Level2') to the row index
stacked_df = df_multi_cols.stack() # Default level=-1 (innermost)
print("\nStacked DataFrame (innermost column level 'Level2' moved to index):")
print(stacked_df)

stacked_df_level0 = df_multi_cols.stack(level=0) # Stack 'Level1'
print("\nStacked DataFrame (column level 'Level1' moved to index):")
print(stacked_df_level0)


# Using df_multi_idx (MultiIndex on rows) for unstacking example
print("\nDataFrame with MultiIndex on rows (df_multi_idx) for unstack:")
print(df_multi_idx.head())

# Unstack: Pivots the innermost index level ('InnerGroup') to columns
unstacked_df = df_multi_idx.unstack() # Default level=-1 (innermost: 'InnerGroup')
print("\nUnstacked DataFrame (innermost index level 'InnerGroup' moved to columns):")
print(unstacked_df)

# Unstack a specific level
unstacked_df_level0 = df_multi_idx.unstack(level=0) # Unstack 'OuterGroup'
print("\nUnstacked DataFrame (index level 'OuterGroup' moved to columns):")
print(unstacked_df_level0)

# Handling missing values during unstacking with fill_value
df_missing_unstack = df_multi_set.copy()
df_missing_unstack = df_missing_unstack.drop(('A', 'Y')) # Create a missing combination
print("\nDataFrame for unstacking with potential NaNs:")
print(df_missing_unstack)
unstacked_with_nan = df_missing_unstack.unstack()
print("\nUnstacked with NaNs:")
print(unstacked_with_nan)
unstacked_filled = df_missing_unstack.unstack(fill_value=0)
print("\nUnstacked with NaNs filled with 0:")
print(unstacked_filled)


# --- b) Pivoting DataFrames ---
print("\n--- b) Pivoting DataFrames ---")
# `pivot()`: Reshapes data from long to wide format based on unique values of one column.
# `pivot_table()`: More general, allows aggregation if there are duplicate entries for pivoting keys.

# Sample DataFrame for pivoting (long format)
df_long = pd.DataFrame({
    'Date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03']),
    'Variable': ['Temp', 'Humidity', 'Temp', 'Humidity', 'Temp', 'Humidity'],
    'Value': [25, 60, 26, 62, 24, 58]
})
print("\nSample DataFrame for pivoting (long format):")
print(df_long)

# Using pivot()
# `index`: Column to use for the new DataFrame's index.
# `columns`: Column whose unique values will become the new DataFrame's column headers.
# `values`: Column(s) to fill the new DataFrame's values.
# `pivot()` requires unique index/columns combinations. If not unique, use `pivot_table()`.
try:
    df_pivoted = df_long.pivot(index='Date', columns='Variable', values='Value')
    print("\nPivoted DataFrame (wide format):")
    print(df_pivoted)
except ValueError as e:
    print(f"Error pivoting (likely duplicate entries for index/columns): {e}")
    print("Use pivot_table for such cases.")


# Using pivot_table()
# `aggfunc` parameter handles duplicate entries by applying an aggregation function (default is mean).
df_long_duplicates = pd.DataFrame({
    'Date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-02']),
    'Sensor': ['Sensor1', 'Sensor1', 'Sensor2', 'Sensor1'],
    'Measurement': ['Temp', 'Pressure', 'Temp', 'Temp'],
    'Reading': [20, 1010, 22, 21]
})
print("\nSample DataFrame with duplicates for pivot_table():")
print(df_long_duplicates)

# Pivot table to get mean Reading for each Date/Measurement combination
pivot_table_mean = df_long_duplicates.pivot_table(
    index='Date',
    columns='Measurement',
    values='Reading',
    aggfunc='mean' # Default, can be 'sum', 'count', 'min', 'max', np.std, etc.
)
print("\nPivot table (mean Reading for Date/Measurement):")
print(pivot_table_mean)

# Pivot table with multiple index levels and multiple aggregation functions
pivot_table_complex = df_long_duplicates.pivot_table(
    index=['Date', 'Sensor'],
    columns='Measurement',
    values='Reading',
    aggfunc={'Reading': [np.mean, np.min, np.max]}, # Different functions for 'Reading'
    fill_value=0 # Fill missing combinations with 0
)
print("\nComplex pivot table (multiple index, multiple aggfuncs):")
print(pivot_table_complex)


# --- c) Melting DataFrames (Unpivoting) ---
print("\n--- c) Melting DataFrames (Unpivoting) ---")
# `melt()`: Transforms a DataFrame from wide format to long format. It's the inverse of `pivot()`.

# Using df_pivoted from the pivot() example (wide format)
print("\nSample DataFrame for melting (wide format - df_pivoted):")
print(df_pivoted.reset_index()) # Reset index to make 'Date' a column for melting

df_to_melt = df_pivoted.reset_index()

# `id_vars`: Column(s) to keep as identifier variables (remain as columns).
# `value_vars`: Column(s) to unpivot. If None, all columns not in `id_vars` are used.
# `var_name`: Name for the new column that stores the original column headers (variable names).
# `value_name`: Name for the new column that stores the values.
df_melted = df_to_melt.melt(
    id_vars=['Date'],
    value_vars=['Humidity', 'Temp'], # Optional, if not specified, uses all other columns
    var_name='Variable',
    value_name='Value'
)
print("\nMelted DataFrame (long format):")
print(df_melted)


# --- d) Cross Tabulations (pd.crosstab()) ---
print("\n--- d) Cross Tabulations (pd.crosstab()) ---")
# Computes a frequency table of two or more factors.
# By default, computes a frequency table of the factors unless an array of values and an aggregation function are passed.

# Sample data for crosstab
df_crosstab_data = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Handedness': ['Right', 'Left', 'Right', 'Right', 'Left', 'Right', 'Right', 'Left'],
    'Age': [25, 30, 35, 28, 40, 32, 45, 29]
})
print("\nSample DataFrame for crosstab:")
print(df_crosstab_data)

# Basic frequency table of Gender vs Handedness
crosstab_freq = pd.crosstab(df_crosstab_data['Gender'], df_crosstab_data['Handedness'])
print("\nCrosstab: Frequency of Gender vs Handedness:")
print(crosstab_freq)

# Crosstab with values and aggregation function (e.g., mean Age)
crosstab_mean_age = pd.crosstab(
    index=df_crosstab_data['Gender'],
    columns=df_crosstab_data['Handedness'],
    values=df_crosstab_data['Age'],
    aggfunc='mean'
)
print("\nCrosstab: Mean Age by Gender and Handedness:")
print(crosstab_mean_age)

# Crosstab with normalization (proportions)
crosstab_normalized = pd.crosstab(
    df_crosstab_data['Gender'],
    df_crosstab_data['Handedness'],
    normalize=True # Normalize by total count
    # normalize='index' # Normalize by row sum
    # normalize='columns' # Normalize by column sum
)
print("\nCrosstab: Normalized by total count (proportions):")
print(crosstab_normalized)

# Crosstab with margins (subtotals)
crosstab_margins = pd.crosstab(
    df_crosstab_data['Gender'],
    df_crosstab_data['Handedness'],
    margins=True,
    margins_name="Total"
)
print("\nCrosstab: With margins (subtotals):")
print(crosstab_margins)

# -----------------------------------------------------------------------------
# End of Advanced Indexing and Reshaping Data Topic
# -----------------------------------------------------------------------------
print("\nEnd of pandas_topic_8.py: Advanced Indexing and Reshaping Data")

# Clean up (optional, if variables are large or to avoid side effects in interactive sessions)
del arrays, index_tuples, multi_index, s_multi, df_multi_idx, iterables, multi_index_prod
del df_multi_prod, df_to_set_index, df_multi_set, df_multi_cols, df_multi_idx_sorted
del stacked_df, stacked_df_level0, unstacked_df, unstacked_df_level0, df_missing_unstack
del unstacked_with_nan, unstacked_filled, df_long, df_pivoted, df_long_duplicates
del pivot_table_mean, pivot_table_complex, df_to_melt, df_melted, df_crosstab_data
del crosstab_freq, crosstab_mean_age, crosstab_normalized, crosstab_margins
print("\nCleaned up intermediate DataFrames and objects.")
