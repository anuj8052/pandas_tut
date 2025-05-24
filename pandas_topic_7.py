# pandas_topic_7.py
# Topic: Data Visualization with Pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Essential for showing and customizing plots

# -----------------------------------------------------------------------------
# Introduction to Pandas Plotting
# -----------------------------------------------------------------------------
# Pandas has built-in plotting capabilities that are a wrapper around the
# popular Matplotlib library. This makes it easy to create various types of
# plots directly from Series and DataFrame objects.

# - The primary way to plot is using the `.plot()` accessor on a Series or DataFrame.
# - For more complex visualizations or fine-grained control, you can still use
#   Matplotlib directly, or libraries like Seaborn which also integrate well with Pandas.

# Important Note for Displaying Plots:
# - When running a Python script, you need `import matplotlib.pyplot as plt` and
#   then `plt.show()` after your plotting commands to display the plots.
# - In Jupyter Notebook or IPython, you can use the magic command `%matplotlib inline`
#   at the beginning of your notebook to have plots displayed directly below the code cell.
#   Alternatively, `%matplotlib notebook` provides interactive plots.

# -----------------------------------------------------------------------------
# Sample DataFrames for Plotting Examples
# -----------------------------------------------------------------------------
print("--- Creating Sample Data for Plotting ---")
# Time series data
date_rng = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
ts_data = pd.DataFrame(
    np.random.randn(len(date_rng), 3) * 10,
    index=date_rng,
    columns=['SeriesA', 'SeriesB', 'SeriesC']
)
ts_data['SeriesA'] = ts_data['SeriesA'].cumsum()
ts_data['SeriesB'] = ts_data['SeriesB'].cumsum()
ts_data['SeriesC'] = ts_data['SeriesC'].cumsum()
print("\nTime Series Data (ts_data):")
print(ts_data.head())

# Categorical and numerical data
cat_data = pd.DataFrame({
    'Category': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y'],
    'Value1': np.random.randint(1, 100, 8),
    'Value2': np.random.randn(8) * 50 + 100,
    'Value3': np.abs(np.random.randn(8) * 20)
})
cat_data_grouped = cat_data.groupby('Category')[['Value1', 'Value2']].mean()
print("\nCategorical Data Grouped (cat_data_grouped):")
print(cat_data_grouped)

# Data for scatter plots
scatter_df = pd.DataFrame(np.random.rand(50, 4), columns=['A', 'B', 'C', 'D'])
scatter_df['A'] = scatter_df['A'] * 100
scatter_df['B'] = scatter_df['B'] * 50
print("\nScatter Plot Data (scatter_df head):")
print(scatter_df.head())
print("\n")


# -----------------------------------------------------------------------------
# 1. Basic Plotting with .plot()
# -----------------------------------------------------------------------------
print("--- 1. Basic Plotting with .plot() ---")
# The .plot() method on a Series or DataFrame is the main entry point.
# By default, it creates a line plot.

# --- a) Line Plots (Default) ---
# For a Series:
plt.figure(figsize=(10, 4)) # Create a new figure
ts_data['SeriesA'].plot(title='Line Plot of SeriesA')
plt.xlabel("Date")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout() # Adjust plot to ensure everything fits without overlapping
print("Plotting: Line Plot of SeriesA (check plot window)")
# plt.show() # Uncomment if running as a script and not in interactive env

# For a DataFrame:
# Plots all numeric columns by default, each as a separate line.
ts_data.plot(figsize=(10, 4), title='Line Plot of All Series in ts_data')
plt.xlabel("Date")
plt.ylabel("Cumulative Value")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
print("Plotting: Line Plot of All Series in ts_data (check plot window)")
# plt.show()

# Explicitly using .plot.line()
ts_data.plot.line(y='SeriesB', figsize=(10, 4), title='Line Plot of SeriesB (explicit)')
plt.tight_layout()
print("Plotting: Line Plot of SeriesB (explicit) (check plot window)")
# plt.show()


# -----------------------------------------------------------------------------
# 2. Common Plot Types
# -----------------------------------------------------------------------------
print("\n--- 2. Common Plot Types ---")

# --- a) Bar Plots: df.plot.bar() / df.plot.barh() ---
# Good for comparing categorical data.
cat_data_grouped.plot.bar(
    y='Value1',
    figsize=(8, 5),
    title='Bar Plot: Mean Value1 by Category',
    color=['skyblue', 'lightcoral', 'lightgreen']
)
plt.ylabel("Mean Value1")
plt.xticks(rotation=0) # Rotate x-axis labels if needed
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
print("Plotting: Bar Plot - Mean Value1 by Category (check plot window)")
# plt.show()

# Stacked bar plot (for DataFrames with multiple columns)
cat_data_grouped.plot.bar(
    stacked=True,
    figsize=(8, 5),
    title='Stacked Bar Plot: Mean Values by Category'
)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
print("Plotting: Stacked Bar Plot - Mean Values by Category (check plot window)")
# plt.show()

# Horizontal bar plot
cat_data_grouped.plot.barh(
    figsize=(8, 5),
    title='Horizontal Bar Plot: Mean Values by Category',
    color={'Value1': 'orange', 'Value2': 'purple'}
)
plt.xlabel("Mean Value")
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
print("Plotting: Horizontal Bar Plot - Mean Values by Category (check plot window)")
# plt.show()


# --- b) Histograms: df.plot.hist() ---
# Good for understanding the distribution of a numerical variable.
cat_data['Value1'].plot.hist(
    bins=10,  # Number of bins
    figsize=(8, 5),
    title='Histogram of Value1',
    edgecolor='black' # Add edges to bars for clarity
)
plt.xlabel("Value1")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
print("Plotting: Histogram of Value1 (check plot window)")
# plt.show()

# Histogram for multiple columns in a DataFrame (plotted side-by-side or overlaid)
# Use alpha for transparency if overlaid
ts_data[['SeriesA', 'SeriesB']].plot.hist(
    bins=15,
    alpha=0.7, # Transparency for overlapping histograms
    figsize=(10, 5),
    title='Histograms of SeriesA and SeriesB'
)
plt.xlabel("Value")
plt.tight_layout()
print("Plotting: Histograms of SeriesA and SeriesB (check plot window)")
# plt.show()


# --- c) Box Plots: df.plot.box() ---
# Shows distribution through quartiles, median, and outliers.
ts_data.plot.box(
    figsize=(8, 6),
    title='Box Plot of SeriesA, SeriesB, SeriesC',
    grid=True
)
plt.ylabel("Value")
plt.tight_layout()
print("Plotting: Box Plot of SeriesA, SeriesB, SeriesC (check plot window)")
# plt.show()

# Box plot grouped by another column (requires direct matplotlib or seaborn for complex cases)
# Pandas .boxplot() method can do this:
# df_data.boxplot(column='Value_Column', by='Category_Column')
cat_data.boxplot(column='Value1', by='Category', figsize=(8, 5), grid=True)
plt.suptitle('') # Remove default suptitle from pandas boxplot
plt.title('Box Plot of Value1 by Category')
plt.xlabel("Category")
plt.ylabel("Value1")
plt.tight_layout()
print("Plotting: Box Plot of Value1 by Category (check plot window)")
# plt.show()


# --- d) Area Plots: df.plot.area() ---
# Good for showing trends in cumulative data.
# By default, area plots are stacked.
ts_data.plot.area(
    figsize=(10, 6),
    title='Area Plot of SeriesA, SeriesB, SeriesC (Stacked)'
)
plt.ylabel("Cumulative Value")
plt.grid(True)
plt.tight_layout()
print("Plotting: Area Plot (Stacked) (check plot window)")
# plt.show()

# Unstacked area plot
ts_data.plot.area(
    stacked=False,
    alpha=0.5,
    figsize=(10, 6),
    title='Area Plot of SeriesA, SeriesB, SeriesC (Unstacked)'
)
plt.ylabel("Cumulative Value")
plt.grid(True)
plt.tight_layout()
print("Plotting: Area Plot (Unstacked) (check plot window)")
# plt.show()


# --- e) Scatter Plots: df.plot.scatter() ---
# Shows relationship between two numerical variables.
# Requires specifying x and y columns.
scatter_df.plot.scatter(
    x='A',
    y='B',
    figsize=(8, 5),
    title='Scatter Plot: A vs B',
    color='blue',
    grid=True
)
plt.tight_layout()
print("Plotting: Scatter Plot A vs B (check plot window)")
# plt.show()

# Scatter plot with point size and color varying by other columns
scatter_df.plot.scatter(
    x='A',
    y='B',
    s=scatter_df['C'] * 200,  # 's' for size, scaled for visibility
    c='D',                    # 'c' for color, mapped to a colormap
    colormap='viridis',       # Colormap to use
    alpha=0.7,
    figsize=(10, 6),
    title='Scatter Plot: A vs B (Size by C, Color by D)',
    grid=True
)
plt.tight_layout()
print("Plotting: Scatter Plot A vs B (Size by C, Color by D) (check plot window)")
# plt.show()


# --- f) Pie Charts: df.plot.pie() ---
# Shows proportions of categories. Best for a small number of categories.
# Typically used on a Series representing counts or proportions.
pie_data = cat_data_grouped['Value1'].abs() # Pie charts need positive values
pie_data.plot.pie(
    autopct='%1.1f%%',  # Format for percentages
    figsize=(7, 7),
    title='Pie Chart: Proportion of Mean Value1 by Category',
    legend=True,
    # colors=['gold', 'lightskyblue', 'lightcoral'] # Custom colors
    startangle=90 # Start first slice at 90 degrees (top)
)
plt.ylabel('') # Remove default ylabel
plt.tight_layout()
print("Plotting: Pie Chart - Proportion of Mean Value1 (check plot window)")
# plt.show()

# For a DataFrame, specify the 'y' column or use subplots=True
# cat_data_grouped.plot.pie(y='Value1', ...)
# cat_data_grouped.plot.pie(subplots=True, figsize=(12,6)) # One pie per column


# --- g) Hexagonal Bin Plots: df.plot.hexbin() ---
# Alternative to scatter plots for large datasets.
# Divides the 2D space into hexagonal bins and colors bins by point density.
# Requires x and y columns.
scatter_df.plot.hexbin(
    x='A',
    y='B',
    gridsize=15,  # Number of hexagons in x-direction
    cmap='YlGnBu', # Colormap
    figsize=(9, 6),
    title='Hexagonal Bin Plot: A vs B'
)
plt.tight_layout()
print("Plotting: Hexagonal Bin Plot A vs B (check plot window)")
# plt.show()


# -----------------------------------------------------------------------------
# 3. Customizing Plots
# -----------------------------------------------------------------------------
print("\n--- 3. Customizing Plots ---")
# Most .plot() methods accept parameters for customization, many are passed to Matplotlib.
# Common customization parameters:
# - title (str): Title of the plot.
# - xlabel (str), ylabel (str): Labels for x and y axes.
# - legend (bool or str): Whether to display legend ('reverse' to reverse order).
# - figsize (tuple): (width, height) in inches.
# - grid (bool): Whether to display a grid.
# - color (str or list): Color(s) for the plot elements.
# - style (str or list): Matplotlib line styles (e.g., '-', '--', ':', '.-').
# - xlim (tuple), ylim (tuple): Set limits for x and y axes.
# - logx (bool), logy (bool), loglog (bool): Use logarithmic scaling.
# - subplots (bool): Create subplots for each column in a DataFrame.
# - sharex (bool), sharey (bool): Whether subplots share x/y axes.
# - rot (int): Rotation for x-axis tick labels.
# - kind (str): Can also be passed to .plot(kind='bar') etc.

ts_data['SeriesC'].plot(
    kind='line',
    figsize=(12, 6),
    title='Customized Plot of SeriesC',
    xlabel='Date Index',
    ylabel='Values of SeriesC',
    legend=True,
    grid=True,
    color='purple',
    style='--o', # Dashed line with circle markers
    ylim=(ts_data['SeriesC'].min() - 10, ts_data['SeriesC'].max() + 10),
    rot=45 # Rotate x-axis labels
)
plt.legend(loc='best') # More control over legend placement via plt
plt.tight_layout()
print("Plotting: Customized Plot of SeriesC (check plot window)")
# plt.show()


# -----------------------------------------------------------------------------
# 4. Plotting Multiple Columns
# -----------------------------------------------------------------------------
print("\n--- 4. Plotting Multiple Columns ---")
# As seen in line plots, DataFrames automatically plot multiple numeric columns.
# For other plot types, behavior varies:
# - Bar plots: Grouped by default, or stacked with `stacked=True`.
# - Histograms: Overlaid by default (use `alpha`).
# - Box plots: One box per column.
# - Area plots: Stacked by default.

# Using `subplots=True` to create separate plots for each column
ts_data.plot(
    subplots=True,
    layout=(2, 2), # (rows, cols) for subplot grid
    figsize=(12, 8),
    title='Subplots for Each Series in ts_data',
    sharex=True, # Share x-axis for better comparison
    grid=True,
    style=['-', '--', '-.'] # Different style for each subplot
)
plt.suptitle('Main Title for All Subplots Using plt.suptitle()', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
print("Plotting: Subplots for Each Series (check plot window)")
# plt.show()


# -----------------------------------------------------------------------------
# 5. Beyond Pandas: Using Matplotlib and Seaborn
# -----------------------------------------------------------------------------
print("\n--- 5. Beyond Pandas: Using Matplotlib and Seaborn ---")
# While Pandas plotting is convenient for quick visualizations, for more complex
# or publication-quality plots, you'll often turn to Matplotlib directly or Seaborn.

# - Matplotlib: Provides full control over every aspect of a plot.
#   Pandas plot objects often return Matplotlib `Axes` objects, which you can then modify.
#   Example:
ax = ts_data['SeriesA'].plot(title='Accessing Matplotlib Axes')
ax.set_xlabel("Custom X Label via Matplotlib")
ax.set_ylabel("Custom Y Label")
ax.legend(['My Series A'])
# plt.show()

# - Seaborn: Built on top of Matplotlib, provides a high-level interface for
#   drawing attractive and informative statistical graphics. Works very well with Pandas DataFrames.
#   Example (requires `pip install seaborn`):
#   import seaborn as sns
#   plt.figure(figsize=(10,5))
#   sns.lineplot(data=ts_data, x=ts_data.index, y='SeriesB')
#   plt.title('Line Plot using Seaborn')
#   plt.show()

print("\nFor more advanced plotting, explore Matplotlib and Seaborn libraries.")

# -----------------------------------------------------------------------------
# Final plt.show() to display all plots if not shown individually
# -----------------------------------------------------------------------------
# If you've commented out individual plt.show() calls during development,
# a single plt.show() at the end will display all created figures.
# However, for clarity in scripts, it's often better to show plots as they are generated
# or save them to files (e.g., using `plt.savefig('plot_name.png')`).

print("\nIf plots are not showing, uncomment individual plt.show() calls or add one at the end.")
plt.show() # This will display all figures created that haven't been shown yet.

# -----------------------------------------------------------------------------
# End of Data Visualization with Pandas Topic
# -----------------------------------------------------------------------------
print("\nEnd of pandas_topic_7.py: Data Visualization with Pandas")
