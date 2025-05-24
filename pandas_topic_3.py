# pandas_topic_3.py
# Topic: Grouping and Aggregation using Pandas

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Introduction to Grouping and Aggregation
# -----------------------------------------------------------------------------
# Grouping and aggregation are fundamental operations in data analysis. They allow
# you to split your data into groups based on some criteria, apply a function
# to each group independently, and then combine the results into a new data structure.
# This process is often referred to as "split-apply-combine".

# Key steps in the "split-apply-combine" strategy:
# 1. Splitting: Data is split into groups based on one or more keys.
# 2. Applying: A function is applied to each group independently. This could be:
#    - Aggregation: Computes a summary statistic (e.g., sum, mean) for each group.
#    - Transformation: Performs some group-specific computations and returns a like-indexed object.
#    - Filtration: Discards some groups, according to a group-wise computation that evaluates True or False.
# 3. Combining: The results of the function applications are combined into a result object.

# -----------------------------------------------------------------------------
# Sample DataFrame for Examples
# -----------------------------------------------------------------------------
data = {
    'Department': ['Sales', 'HR', 'IT', 'Sales', 'IT', 'HR', 'IT', 'Sales', 'HR'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Heidi', 'Ivan'],
    'Salary': [70000, 60000, 80000, 75000, 82000, 65000, 90000, 68000, 62000],
    'YearsExperience': [5, 3, 7, 6, 6, 4, 9, 5, 3],
    'ProjectsCompleted': [10, 5, 12, 11, 10, 6, 15, 9, 4]
}
df = pd.DataFrame(data)
print("--- Initial Sample DataFrame ---")
print(df)
print("\n")

# -----------------------------------------------------------------------------
# 1. Using the groupby() Method
# -----------------------------------------------------------------------------
print("--- 1. Using the groupby() Method ---")
# The groupby() method is used to split the DataFrame into groups.
# It creates a DataFrameGroupBy object. This object itself doesn't display the groups
# directly but holds the information needed to then apply calculations to each group.

# Group by a single column: 'Department'
grouped_by_dept = df.groupby('Department')

# The 'grouped_by_dept' object is a DataFrameGroupBy object
print(f"Type of grouped object: {type(grouped_by_dept)}")

# You can iterate over a GroupBy object to see the groups:
print("\nIterating through groups (Department):")
for name, group in grouped_by_dept:
    print(f"\nDepartment: {name}")
    print(group)

# Get a specific group
print("\nGetting the 'IT' group:")
print(grouped_by_dept.get_group('IT'))

# Select a column after grouping
# This results in a SeriesGroupBy object if a single column is selected
salary_by_dept_series_group = grouped_by_dept['Salary']
print(f"\nType of grouped series object (Salary by Department): {type(salary_by_dept_series_group)}")


# -----------------------------------------------------------------------------
# 2. Applying Aggregation Functions
# -----------------------------------------------------------------------------
print("\n--- 2. Applying Aggregation Functions ---")
# Once data is grouped, you can apply various aggregation functions to summarize it.

# Common aggregation functions:
# - sum(): Compute sum of group values.
# - mean(): Compute mean of group values.
# - count(): Compute count of group values (non-NA).
# - size(): Compute group sizes (including NA).
# - min(): Compute minimum of group values.
# - max(): Compute maximum of group values.
# - std(): Compute standard deviation of group values.
# - var(): Compute variance of group values.
# - median(): Compute median of group values.
# - first(): Compute first of group values.
# - last(): Compute last of group values.
# - nunique(): Compute number of unique values in group.

print("\nSum of Salaries by Department:")
print(grouped_by_dept['Salary'].sum())

print("\nMean Salary and YearsExperience by Department:")
# Applying to multiple columns (will select numeric columns by default for some functions)
print(grouped_by_dept[['Salary', 'YearsExperience']].mean())

print("\nCount of Employees by Department (using count() on a non-null column):")
print(grouped_by_dept['Employee'].count())
# Note: count() gives non-NA counts. size() gives total size including NA.
print("\nSize of each Department group (using size()):")
print(grouped_by_dept.size()) # This returns a Series with group sizes

print("\nMaximum ProjectsCompleted by Department:")
print(grouped_by_dept['ProjectsCompleted'].max())

print("\nDescriptive statistics for Salary by Department:")
print(grouped_by_dept['Salary'].describe())


# -----------------------------------------------------------------------------
# 3. Aggregating with Multiple Functions using agg() or aggregate()
# -----------------------------------------------------------------------------
print("\n--- 3. Aggregating with Multiple Functions (agg() / aggregate()) ---")
# The agg() or aggregate() method allows you to apply multiple aggregation functions
# at once, and even different functions to different columns.

# Apply multiple functions to a single column ('Salary')
print("\nMultiple aggregations (sum, mean, std) for 'Salary' by Department:")
print(grouped_by_dept['Salary'].agg(['sum', 'mean', 'std']))

# Apply different functions to different columns
# Pass a dictionary where keys are column names and values are functions or list of functions
agg_functions = {
    'Salary': ['mean', 'max'],
    'YearsExperience': 'median',
    'ProjectsCompleted': ['sum', 'count']
}
print("\nDifferent aggregations for different columns by Department:")
print(grouped_by_dept.agg(agg_functions))

# You can also rename the resulting aggregated columns
print("\nAggregating and renaming columns:")
print(grouped_by_dept['Salary'].agg(
    AvgSalary='mean',
    MaxSalary='max',
    TotalEmployees='count' # count of salaries, which is count of employees in this case
))


# -----------------------------------------------------------------------------
# 4. Grouping by Multiple Columns
# -----------------------------------------------------------------------------
print("\n--- 4. Grouping by Multiple Columns ---")
# You can group by multiple columns by passing a list of column names to groupby().
# This creates a MultiIndex in the resulting aggregated DataFrame.

# Sample DataFrame with an additional 'Location' column for multi-grouping
data_multi = {
    'Department': ['Sales', 'IT', 'Sales', 'IT', 'Sales', 'IT'],
    'Location': ['New York', 'London', 'London', 'New York', 'New York', 'London'],
    'Salary': [70000, 80000, 72000, 85000, 68000, 78000]
}
df_multi = pd.DataFrame(data_multi)
print("\nDataFrame for multi-column grouping:")
print(df_multi)

grouped_multi = df_multi.groupby(['Department', 'Location'])

print("\nMean Salary grouped by Department and Location:")
print(grouped_multi['Salary'].mean())

# The result has a MultiIndex. You can unstack it if needed.
print("\nUnstacked mean Salary (Location becomes columns):")
print(grouped_multi['Salary'].mean().unstack())


# -----------------------------------------------------------------------------
# 5. Applying Custom Functions to Groups using apply()
# -----------------------------------------------------------------------------
print("\n--- 5. Applying Custom Functions to Groups (apply()) ---")
# The apply() method is very flexible. It takes a function and applies it to each
# group as a whole (i.e., to the DataFrame chunk for that group).
# The function can return a scalar, a Series, or a DataFrame.

# Example: A function that returns the employee with the highest salary in each department
def get_top_earner(group_df):
    return group_df.loc[group_df['Salary'].idxmax()]

print("\nTop earner in each Department (using apply()):")
# Using the original df and grouped_by_dept
print(grouped_by_dept.apply(get_top_earner))

# Example: A function that normalizes salaries within each department (Salary - Mean_Salary_Dept)
def normalize_salary(group_df):
    group_df['NormalizedSalary'] = group_df['Salary'] - group_df['Salary'].mean()
    return group_df

print("\nDataFrame with Normalized Salaries within each Department (using apply()):")
# Note: apply() can sometimes be slow for simple operations that have vectorized alternatives.
# The result here concatenates the DataFrames returned by normalize_salary for each group.
# The original index is preserved.
normalized_df_apply = grouped_by_dept.apply(normalize_salary)
print(normalized_df_apply[['Department', 'Employee', 'Salary', 'NormalizedSalary']])


# -----------------------------------------------------------------------------
# 6. Transformations on Groups using transform()
# -----------------------------------------------------------------------------
print("\n--- 6. Transformations on Groups (transform()) ---")
# The transform() method applies a function to each group and returns a Series or DataFrame
# that has the same shape as the input group. The result is broadcast back to the
# original DataFrame's shape.
# Useful for feature engineering where you want to create new columns based on group properties.

# Example: Fill missing values with the mean salary of each department
df_with_nan = df.copy()
df_with_nan.loc[0, 'Salary'] = np.nan # Introduce a NaN value
df_with_nan.loc[4, 'Salary'] = np.nan
print("\nDataFrame with some NaN salaries:")
print(df_with_nan[['Department', 'Employee', 'Salary']])

# Using transform to get the mean salary for each department, aligned with original df shape
dept_mean_salary = df_with_nan.groupby('Department')['Salary'].transform('mean')
print("\nMean salary per department (aligned with original DataFrame via transform):")
print(dept_mean_salary)

df_filled_transform = df_with_nan.copy()
df_filled_transform['Salary'] = df_filled_transform['Salary'].fillna(dept_mean_salary)
# A more direct way using transform within fillna if the function is simple:
# df_filled_transform['Salary'] = df_filled_transform.groupby('Department')['Salary'].transform(lambda x: x.fillna(x.mean()))

print("\nDataFrame with NaN salaries filled by department mean (using transform):")
print(df_filled_transform[['Department', 'Employee', 'Salary']])


# Example: Z-score of salaries within each department
# Z-score = (value - mean) / std
def z_score_transform(x): # x is a Series representing the 'Salary' column for a group
    return (x - x.mean()) / x.std()

df['SalaryZScore_Dept'] = df.groupby('Department')['Salary'].transform(z_score_transform)
print("\nDataFrame with Z-scores of salaries within each department:")
print(df[['Department', 'Employee', 'Salary', 'SalaryZScore_Dept']])


# -----------------------------------------------------------------------------
# 7. Filtering Groups using filter()
# -----------------------------------------------------------------------------
print("\n--- 7. Filtering Groups (filter()) ---")
# The filter() method allows you to select or discard entire groups based on a
# function that returns True or False for each group.

# Example: Select departments where the average salary is greater than 70000
print("\nDepartments where average salary is > 70000:")
# The function passed to filter takes a group (DataFrame chunk) and returns True or False
filtered_high_avg_salary_depts = grouped_by_dept.filter(lambda group: group['Salary'].mean() > 70000)
print(filtered_high_avg_salary_depts)

# Example: Select departments that have more than 2 employees
print("\nDepartments with more than 2 employees:")
filtered_large_depts = grouped_by_dept.filter(lambda group: len(group) > 2)
# Or using size(): filtered_large_depts = grouped_by_dept.filter(lambda group: group.size > 2)
print(filtered_large_depts)

# Example: Filter out groups where any salary is below a certain threshold (e.g., 65000)
print("\nDepartments where all employees earn >= 65000:")
filtered_min_salary_depts = grouped_by_dept.filter(lambda group: group['Salary'].min() >= 65000)
print(filtered_min_salary_depts)


# -----------------------------------------------------------------------------
# End of Grouping and Aggregation Topic
# -----------------------------------------------------------------------------
print("\nEnd of pandas_topic_3.py: Grouping and Aggregation")

# Clean up
del df, grouped_by_dept, salary_by_dept_series_group
del df_multi, grouped_multi
del normalized_df_apply
del df_with_nan, dept_mean_salary, df_filled_transform
del filtered_high_avg_salary_depts, filtered_large_depts, filtered_min_salary_depts
print("\nCleaned up intermediate DataFrames and objects.")
