# pandas_topic_4.py
# Topic: Merging, Joining, and Concatenating DataFrames using Pandas

import pandas as pd

# -----------------------------------------------------------------------------
# Introduction
# -----------------------------------------------------------------------------
# Pandas provides various functions for easily combining Series or DataFrame objects.
# These operations are fundamental for integrating data from different sources or
# reshaping data for analysis. The main methods are:
# - pd.concat(): For combining DataFrames along an axis (stacking).
# - pd.merge(): For SQL-like joins on columns or indexes.
# - DataFrame.join(): A convenient method for joining DataFrames, often on index.

# -----------------------------------------------------------------------------
# Sample DataFrames for Examples
# -----------------------------------------------------------------------------
# DataFrame 1: Employee Information
df_employees = pd.DataFrame({
    'EmployeeID': [101, 102, 103, 104, 105],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'DepartmentID': [1, 2, 1, 3, 2]
})
print("--- Sample DataFrame: Employees ---")
print(df_employees)

# DataFrame 2: Department Information
df_departments = pd.DataFrame({
    'DepartmentID': [1, 2, 3, 4],
    'DepartmentName': ['HR', 'Engineering', 'Marketing', 'Sales'],
    'Location': ['New York', 'London', 'Paris', 'Berlin']
})
print("\n--- Sample DataFrame: Departments ---")
print(df_departments)

# DataFrame 3: Salary Information (with EmployeeID)
df_salaries = pd.DataFrame({
    'EmployeeID': [101, 102, 103, 104, 106],
    'Salary': [70000, 80000, 75000, 90000, 65000],
    'Bonus': [5000, 6000, 5500, 7000, 4000]
})
print("\n--- Sample DataFrame: Salaries ---")
print(df_salaries)

# DataFrame 4 & 5 for concatenation examples
df_concat1 = pd.DataFrame({'A': ['A0', 'A1'], 'B': ['B0', 'B1']}, index=[0, 1])
df_concat2 = pd.DataFrame({'A': ['A2', 'A3'], 'B': ['B2', 'B3']}, index=[2, 3])
df_concat3 = pd.DataFrame({'C': ['C0', 'C1'], 'D': ['D0', 'D1']}, index=[0, 1]) # Different columns

print("\n--- Sample DataFrame: Concat1 ---")
print(df_concat1)
print("\n--- Sample DataFrame: Concat2 ---")
print(df_concat2)
print("\n--- Sample DataFrame: Concat3 ---")
print(df_concat3)
print("\n")


# -----------------------------------------------------------------------------
# 1. Concatenation using pd.concat()
# -----------------------------------------------------------------------------
print("--- 1. Concatenation using pd.concat() ---")
# pd.concat() is used to append one or more DataFrames below one another (axis=0)
# or side-by-side (axis=1).

# --- a) Concatenating along rows (axis=0) ---
# This is the default behavior.
df_row_concat = pd.concat([df_concat1, df_concat2])
print("\nConcatenating df_concat1 and df_concat2 along rows (default axis=0):")
print(df_row_concat)

# If DataFrames have different columns, pd.concat() will fill non-existing columns with NaN.
df_row_concat_diff_cols = pd.concat([df_concat1, df_concat3])
print("\nConcatenating df_concat1 and df_concat3 (different columns) along rows:")
print(df_row_concat_diff_cols) # Notice NaNs

# Handling Indexes:
# - ignore_index=False (default): Preserves original indexes. Can lead to duplicate indexes.
# - ignore_index=True: Creates a new default integer index (0, 1, 2, ...).
df_row_concat_ignore_index = pd.concat([df_concat1, df_concat1], ignore_index=True)
print("\nConcatenating with ignore_index=True:")
print(df_row_concat_ignore_index)

# Using 'keys' to create a hierarchical index:
df_row_concat_keys = pd.concat([df_concat1, df_concat2], keys=['x', 'y'])
print("\nConcatenating with keys for hierarchical index:")
print(df_row_concat_keys)
print("Accessing group 'x':")
print(df_row_concat_keys.loc['x'])

# --- b) Concatenating along columns (axis=1) ---
# This stacks DataFrames side-by-side.
df_col_concat = pd.concat([df_concat1, df_concat3], axis=1)
print("\nConcatenating df_concat1 and df_concat3 along columns (axis=1):")
print(df_col_concat)

# If indexes don't align, it will introduce NaNs (outer join behavior on index).
df_concat4 = pd.DataFrame({'E': ['E2', 'E3']}, index=[2, 3])
df_col_concat_diff_index = pd.concat([df_concat1, df_concat4], axis=1)
print("\nConcatenating df_concat1 and df_concat4 (different indexes) along columns:")
print(df_col_concat_diff_index) # NaNs due to misaligned indexes

# Using 'join' parameter with axis=1:
# - join='outer' (default): Keeps all indexes from all DataFrames.
# - join='inner': Keeps only common indexes.
df_col_concat_inner_join = pd.concat([df_concat1, df_concat4], axis=1, join='inner')
print("\nConcatenating df_concat1 and df_concat4 (axis=1, join='inner'):")
print(df_col_concat_inner_join) # Will be empty if no common indexes


# -----------------------------------------------------------------------------
# 2. Merging/Joining with pd.merge()
# -----------------------------------------------------------------------------
print("\n--- 2. Merging/Joining with pd.merge() ---")
# pd.merge() is used for database-style joins (like SQL joins). It aligns rows
# based on common column values or index values.

# --- a) Basic Merge (Inner Join on a Common Column) ---
# If 'on' is not specified, merge uses the intersection of column names as join keys.
# Here, 'EmployeeID' is common between df_employees and df_salaries.
# Default join type is 'inner'.
merged_inner_default = pd.merge(df_employees, df_salaries) # Implicitly on 'EmployeeID'
print("\nInner merge of Employees and Salaries (default, on 'EmployeeID'):")
print(merged_inner_default) # Employee 105 (Eve) and 106 are missing.

# Explicitly specifying the join key using 'on':
merged_inner_on = pd.merge(df_employees, df_salaries, on='EmployeeID')
print("\nInner merge of Employees and Salaries (explicit on='EmployeeID'):")
print(merged_inner_on)


# --- b) Different Types of Joins (how parameter) ---
# - 'inner': (Default) Intersection of keys. Only matching keys from both DataFrames.
# - 'outer': Union of keys. Includes all keys from both DataFrames, fills missing with NaN.
# - 'left': Uses keys from the left DataFrame only.
# - 'right': Uses keys from the right DataFrame only.

# Left Join: All employees, with their salaries if available.
merged_left = pd.merge(df_employees, df_salaries, on='EmployeeID', how='left')
print("\nLeft merge of Employees and Salaries (on 'EmployeeID'):")
print(merged_left) # Eve (105) has NaN for Salary/Bonus.

# Right Join: All salaries, with employee info if available.
merged_right = pd.merge(df_employees, df_salaries, on='EmployeeID', how='right')
print("\nRight merge of Employees and Salaries (on 'EmployeeID'):")
print(merged_right) # Employee 106 has NaN for Name/DepartmentID.

# Outer Join: All employees and all salaries.
merged_outer = pd.merge(df_employees, df_salaries, on='EmployeeID', how='outer')
print("\nOuter merge of Employees and Salaries (on 'EmployeeID'):")
print(merged_outer) # Both Eve (105) and Employee 106 are present.


# --- c) Joining on Different Column Names (left_on, right_on) ---
# If the join key columns have different names in the two DataFrames.
df_departments_renamed = df_departments.rename(columns={'DepartmentID': 'DeptID'})
print("\nRenamed Departments DataFrame for left_on/right_on example:")
print(df_departments_renamed)

merged_diff_names = pd.merge(df_employees, df_departments_renamed,
                             left_on='DepartmentID', right_on='DeptID', how='left')
print("\nLeft merge Employees with renamed Departments (left_on='DepartmentID', right_on='DeptID'):")
print(merged_diff_names) # Note: Both 'DepartmentID' and 'DeptID' columns are present.
                         # You might want to drop one of them after the merge.


# --- d) Joining on Index (left_index=True, right_index=True) ---
df_employees_indexed = df_employees.set_index('EmployeeID')
df_salaries_indexed = df_salaries.set_index('EmployeeID')

print("\nEmployees DataFrame with 'EmployeeID' as index:")
print(df_employees_indexed)
print("\nSalaries DataFrame with 'EmployeeID' as index:")
print(df_salaries_indexed)

merged_on_index = pd.merge(df_employees_indexed, df_salaries_indexed,
                           left_index=True, right_index=True, how='inner')
print("\nInner merge on index (EmployeeID):")
print(merged_on_index)

# You can also mix joining on index with joining on columns:
merged_index_col = pd.merge(df_employees_indexed, df_departments,
                            left_on='DepartmentID', right_index=True, # Hypothetical if DeptID was index in df_departments
                            how='left') # This example is a bit contrived, df_departments doesn't have DeptID as index.
# Let's make a more sensible example for index-column join:
df_departments_indexed_on_id = df_departments.set_index('DepartmentID')
merged_emp_dept_idx_col = pd.merge(df_employees, df_departments_indexed_on_id,
                                   left_on='DepartmentID', right_index=True, how='left')
print("\nLeft merge Employees (on 'DepartmentID' col) with Departments (on 'DepartmentID' index):")
print(merged_emp_dept_idx_col)


# --- e) Handling Overlapping Column Names (suffixes) ---
# If DataFrames being merged have columns with the same name (other than join keys),
# merge will append suffixes to them to avoid ambiguity. Default is '_x', '_y'.
df_employees_extra = df_employees.copy()
df_employees_extra['Status'] = ['Active', 'Active', 'Inactive', 'Active', 'Active']

df_salaries_extra = df_salaries.copy()
df_salaries_extra['Status'] = ['Confirmed', 'Confirmed', 'Pending', 'Confirmed', 'Probation']

print("\nEmployees DataFrame with 'Status' column:")
print(df_employees_extra[['EmployeeID', 'Status']])
print("\nSalaries DataFrame with 'Status' column:")
print(df_salaries_extra[['EmployeeID', 'Status']])

merged_suffixes = pd.merge(df_employees_extra, df_salaries_extra, on='EmployeeID', how='inner')
print("\nInner merge with overlapping 'Status' column (default suffixes):")
print(merged_suffixes[['EmployeeID', 'Status_x', 'Status_y']]) # Status_x from left, Status_y from right

# Custom suffixes:
merged_custom_suffixes = pd.merge(df_employees_extra, df_salaries_extra, on='EmployeeID', how='inner',
                                  suffixes=('_emp', '_sal'))
print("\nInner merge with custom suffixes ('_emp', '_sal'):")
print(merged_custom_suffixes[['EmployeeID', 'Status_emp', 'Status_sal']])


# -----------------------------------------------------------------------------
# 3. Joining with DataFrame.join() method
# -----------------------------------------------------------------------------
print("\n--- 3. Joining with DataFrame.join() method ---")
# DataFrame.join() is a convenience method for merging, primarily by index, but can also
# join on a key column of the left DataFrame to the index of the right DataFrame.

# Default: Joins on index (left join by default)
# Using df_employees_indexed and df_salaries_indexed from earlier
joined_on_index_df_method = df_employees_indexed.join(df_salaries_indexed, how='inner')
print("\nInner join using DataFrame.join() on index:")
print(joined_on_index_df_method)

joined_left_df_method = df_employees_indexed.join(df_salaries_indexed, how='left')
print("\nLeft join using DataFrame.join() on index (default how='left'):")
print(joined_left_df_method)

# Joining on a key column of the left DataFrame to the index of the right DataFrame
# Let's use df_employees (not indexed) and df_departments_indexed_on_id
# We want to join df_employees.DepartmentID with df_departments_indexed_on_id.index
joined_col_to_index = df_employees.join(df_departments_indexed_on_id, on='DepartmentID', how='left')
print("\nLeft join df_employees (on 'DepartmentID' col) to df_departments_indexed_on_id (index) using .join():")
print(joined_col_to_index)

# If there are overlapping column names, .join() uses 'lsuffix' and 'rsuffix'
df_emp_idx_status = df_employees_indexed.copy()
df_emp_idx_status['Status'] = ['A', 'B', 'C', 'D', 'E']

df_sal_idx_status = df_salaries_indexed.copy()
df_sal_idx_status['Status'] = ['X', 'Y', 'Z', 'W', 'V']

print("\nEmployee indexed with Status:")
print(df_emp_idx_status[['Status']])
print("\nSalary indexed with Status:")
print(df_sal_idx_status[['Status']])

joined_suffixes_df_method = df_emp_idx_status.join(df_sal_idx_status, how='inner', lsuffix='_emp', rsuffix='_sal')
print("\nInner join with .join() and custom suffixes for 'Status':")
print(joined_suffixes_df_method[['Status_emp', 'Status_sal', 'Salary', 'Bonus']]) # Salary/Bonus from right df

# Joining multiple DataFrames with .join()
# The DataFrames to be joined must be passed as a list if they are not indexed by default or if you need specific 'on' keys
# However, .join() is typically used for index-on-index joins or column-on-index.
# For multiple DataFrames, it's often clearer to use pd.merge iteratively or pd.concat if appropriate.
# If all DataFrames are indexed appropriately:
# df_main.join([df_other1, df_other2])

# -----------------------------------------------------------------------------
# When to Use Which Method?
# -----------------------------------------------------------------------------
# - pd.concat():
#   - Stacking DataFrames vertically (adding rows) or horizontally (adding columns).
#   - Simple combinations where alignment is based purely on axis position or shared index structure.
#   - Less flexible for complex, value-based joining logic.

# - pd.merge():
#   - SQL-like joins: inner, outer, left, right.
#   - Joining on specific columns (keys) or on indexes.
#   - Most flexible and powerful for relational data merging.
#   - Use when you need to combine DataFrames based on matching values in one or more columns.

# - DataFrame.join():
#   - Primarily for joining DataFrames on their indexes.
#   - Can also join a DataFrame's column to another DataFrame's index.
#   - More convenient syntax for index-based joins compared to pd.merge(... left_index=True, right_index=True ...).
#   - Less flexible than pd.merge() for column-to-column joins or complex conditions.

# -----------------------------------------------------------------------------
# End of Merging, Joining, and Concatenating Topic
# -----------------------------------------------------------------------------
print("\nEnd of pandas_topic_4.py: Merging, Joining, and Concatenating DataFrames")

# Clean up
del df_employees, df_departments, df_salaries
del df_concat1, df_concat2, df_concat3, df_concat4
del df_row_concat, df_row_concat_diff_cols, df_row_concat_ignore_index, df_row_concat_keys
del df_col_concat, df_col_concat_diff_index, df_col_concat_inner_join
del merged_inner_default, merged_inner_on, merged_left, merged_right, merged_outer
del df_departments_renamed, merged_diff_names
del df_employees_indexed, df_salaries_indexed, merged_on_index, df_departments_indexed_on_id, merged_emp_dept_idx_col
del df_employees_extra, df_salaries_extra, merged_suffixes, merged_custom_suffixes
del joined_on_index_df_method, joined_left_df_method, joined_col_to_index
del df_emp_idx_status, df_sal_idx_status, joined_suffixes_df_method
print("\nCleaned up intermediate DataFrames.")
