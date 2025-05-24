# pandas_topic_10.py
# Topic: Extending Pandas and its Ecosystem

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Introduction
# -----------------------------------------------------------------------------
# Pandas is a powerful library on its own, but its true strength is amplified
# by its ability to be extended and its central role in the Python data science ecosystem.
# This topic explores how you can add custom functionality to Pandas objects,
# how Pandas interacts with other major libraries, how to customize its behavior,
# and where to continue your learning journey.

# -----------------------------------------------------------------------------
# 1. Custom Accessors
# -----------------------------------------------------------------------------
print("--- 1. Custom Accessors ---")
# Custom accessors allow you to add your own methods and properties to Pandas
# Series, DataFrames, or Index objects under a custom namespace. This is useful
# for creating domain-specific extensions that feel like built-in Pandas features.

# You use decorators like `@pd.api.extensions.register_dataframe_accessor`,
# `@pd.api.extensions.register_series_accessor`, or
# `@pd.api.extensions.register_index_accessor`.

# --- Example: A Simple DataFrame Accessor ---
@pd.api.extensions.register_dataframe_accessor("custom_ext")
class CustomDataFrameAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # Perform any validation on the DataFrame structure if needed
        # For example, check if specific columns exist
        pass

    @property
    def row_count(self):
        """Returns the number of rows in the DataFrame."""
        return len(self._obj)

    def describe_columns_dtypes(self):
        """Returns a Series describing the data types of columns."""
        return self._obj.dtypes

    def select_numeric_cols(self):
        """Selects only the numeric columns from the DataFrame."""
        return self._obj.select_dtypes(include=np.number)

# Using the custom accessor:
df_sample = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [10.5, 20.5, 30.5],
    'C': ['x', 'y', 'z'],
    'D': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
})

print("\nSample DataFrame for custom accessor:")
print(df_sample)

# Accessing custom properties and methods
print(f"\nCustom accessor: Number of rows (df_sample.custom_ext.row_count): {df_sample.custom_ext.row_count}")

print("\nCustom accessor: Describe column data types (df_sample.custom_ext.describe_columns_dtypes()):")
print(df_sample.custom_ext.describe_columns_dtypes())

print("\nCustom accessor: Select numeric columns (df_sample.custom_ext.select_numeric_cols()):")
print(df_sample.custom_ext.select_numeric_cols())

print("\nCustom accessors provide a clean way to extend Pandas with reusable, domain-specific logic.")


# -----------------------------------------------------------------------------
# 2. Pandas Ecosystem and Interoperability
# -----------------------------------------------------------------------------
print("\n--- 2. Pandas Ecosystem and Interoperability ---")
# Pandas DataFrames and Series are fundamental data structures used by many other
# libraries in the Python scientific computing and data science stack (PyData ecosystem).

# --- a) Scikit-learn (Machine Learning) ---
print("\n--- a) Scikit-learn ---")
# Scikit-learn is a popular machine learning library. Pandas DataFrames are commonly
# used to prepare and manage data for input into Scikit-learn models.

# Typically, you'd have:
# - A DataFrame `X` containing features (independent variables).
# - A Series or DataFrame `y` containing the target variable (dependent variable).

# Conceptual example (no actual model training):
data_ml = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100) * 10,
    'feature3_categorical': np.random.choice(['A', 'B', 'C'], 100),
    'target': np.random.randint(0, 2, 100)
}
df_ml = pd.DataFrame(data_ml)

# Convert categorical features (e.g., using pd.get_dummies)
df_ml_processed = pd.get_dummies(df_ml, columns=['feature3_categorical'], drop_first=True)

X = df_ml_processed.drop('target', axis=1) # Features DataFrame
y = df_ml_processed['target']             # Target Series

print("Conceptual data preparation for Scikit-learn:")
print("Features DataFrame (X) head:")
print(X.head())
print("\nTarget Series (y) head:")
print(y.head())
print("This X and y can then be passed to Scikit-learn estimators (e.g., model.fit(X, y)).")
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")


# --- b) Statsmodels (Statistical Modeling) ---
print("\n--- b) Statsmodels ---")
# Statsmodels is a library for estimating and interpreting statistical models,
# including regression, time series analysis, etc. It integrates well with Pandas.
# Often, you can pass DataFrames directly to Statsmodels functions, and it can use
# column names in model formulas.

# Conceptual example (no actual model fitting):
data_sm = {
    'Y_dependent': np.random.rand(50) * 5 + 10,
    'X1_independent': np.random.rand(50) * 2,
    'X2_independent': np.random.rand(50) * 3,
    'Category': np.random.choice(['P', 'Q'], 50)
}
df_sm = pd.DataFrame(data_sm)

print("\nConceptual data for Statsmodels:")
print(df_sm.head())
print("DataFrames like this can be used with Statsmodels formula API, e.g.:")
print("import statsmodels.formula.api as smf")
print("model = smf.ols(formula='Y_dependent ~ X1_independent + X2_independent + C(Category)', data=df_sm)")
# results = model.fit()
# print(results.summary())
print("Statsmodels uses column names for formulas and results.")


# --- c) Matplotlib and Seaborn (Visualization) ---
print("\n--- c) Matplotlib and Seaborn ---")
# As seen in `pandas_topic_7.py`, Pandas' built-in plotting capabilities
# are based on Matplotlib.
# Seaborn, another powerful visualization library, is designed to work seamlessly
# with Pandas DataFrames, often simplifying the creation of complex statistical plots.

# Pandas plot example (recap):
df_sample[['A', 'B']].plot(kind='line', title='Pandas Plot (Matplotlib backend)')
# import matplotlib.pyplot as plt
# plt.show() # Requires matplotlib.pyplot import

# Conceptual Seaborn example:
# import seaborn as sns
# import matplotlib.pyplot as plt # For showing plot
# plt.figure() # Create a new figure
# sns.scatterplot(data=df_sample, x='A', y='B', hue='C')
# plt.title('Seaborn Scatter Plot with Pandas DataFrame')
# plt.show()
print("Pandas integrates with Matplotlib for its .plot() methods.")
print("Seaborn is designed to work directly with DataFrames for advanced statistical plots.")


# -----------------------------------------------------------------------------
# 3. Options and Settings (pd.set_option())
# -----------------------------------------------------------------------------
print("\n--- 3. Options and Settings (pd.set_option()) ---")
# Pandas has an options system that allows you to customize some aspects of its behavior,
# particularly display settings.

# --- a) Common Display Options ---
# `display.max_rows`: Max number of rows displayed (e.g., when printing a DataFrame).
# `display.max_columns`: Max number of columns displayed.
# `display.width`: Width of the display in characters.
# `display.precision`: Floating point output precision.
# `display.max_colwidth`: Max width of columns for displaying text.

print(f"\nCurrent display.max_rows: {pd.get_option('display.max_rows')}")
print(f"Current display.max_columns: {pd.get_option('display.max_columns')}")

# Example: Create a slightly larger DataFrame for display purposes
df_display_example = pd.DataFrame(np.random.rand(15, 6), columns=[f'Col_{i}' for i in range(6)])
df_display_example['TextCol'] = ['This is some long text data in a cell' * 2] * 15

print("\nDefault display of df_display_example (might be truncated):")
print(df_display_example)

# Change options
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 4)
pd.set_option('display.max_colwidth', 20)

print("\nDisplay after setting max_rows=10, max_columns=4, max_colwidth=20:")
print(df_display_example)

# Reset to default (or a preferred setting)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('display.max_colwidth')
# pd.set_option('display.max_rows', 60) # A common default

print(f"\nAfter reset, display.max_rows: {pd.get_option('display.max_rows')}")

# --- b) Other Options ---
# There are options for computation, I/O, styling, etc.
# Explore `pd.describe_option()` or `pd.Options` for more.
# Example: `compute.use_numexpr` (whether to use the numexpr library for faster computation if available)
# print(f"\ncompute.use_numexpr: {pd.get_option('compute.use_numexpr')}")


# -----------------------------------------------------------------------------
# 4. Where to Go Next / Further Learning
# -----------------------------------------------------------------------------
print("\n--- 4. Where to Go Next / Further Learning ---")

# --- a) Official Pandas Documentation ---
print("\n- **Official Pandas Documentation:** (pandas.pydata.org/docs/)")
print("  - The most comprehensive and up-to-date resource.")
print("  - Includes User Guide, API Reference, tutorials, and cookbooks.")
print("  - Check out sections on advanced topics like performance, scaling, and specific data manipulations.")

# --- b) Community Resources ---
print("\n- **Community Resources:**")
print("  - **Stack Overflow:** (stackoverflow.com/questions/tagged/pandas)")
print("    A great place to find answers to specific questions and learn from real-world problems.")
print("  - **Blogs and Tutorials:** Many data scientists and developers share Pandas tips, tricks,")
print("    and tutorials on platforms like Medium, Towards Data Science, personal blogs, etc.")
print("  - **GitHub:** (github.com/pandas-dev/pandas)")
print("    Explore the source code, report issues, or even contribute.")

# --- c) Books and Courses ---
print("\n- **Books:**")
print("  - \"Python for Data Analysis\" by Wes McKinney (creator of Pandas) is a classic.")
print("  - Many other books on data analysis with Python feature Pandas extensively.")
print("\n- **Online Courses:**")
print("  - Platforms like Coursera, Udemy, DataCamp, edX, etc., offer numerous courses on")
print("    data science with Python, where Pandas is a core component.")

# --- d) Practice ---
print("\n- **Practice, Practice, Practice:**")
print("  - The best way to learn is by doing. Work on personal projects, participate in")
print("    Kaggle competitions, or analyze datasets you find interesting.")
print("  - Try to apply the concepts from these topics to real data.")

print("\nThis series of topics has covered the fundamentals and some advanced aspects of Pandas.")
print("Continuous learning and practical application are key to mastering it.")

# -----------------------------------------------------------------------------
# End of Extending Pandas and its Ecosystem Topic
# -----------------------------------------------------------------------------
print("\nEnd of pandas_topic_10.py: Extending Pandas and its Ecosystem")

# Display the sample DataFrame with the custom accessor again to show it's still available
print("\nRe-checking custom accessor on df_sample:")
print(f"Row count: {df_sample.custom_ext.row_count}")
print("Numeric columns:")
print(df_sample.custom_ext.select_numeric_cols())
