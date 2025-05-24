# pandas_topic_5.py
# Topic: Working with Time Series Data using Pandas

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Introduction to Time Series Data
# -----------------------------------------------------------------------------
# Time series data is a sequence of data points indexed in time order. Pandas has
# excellent built-in capabilities for working with time series, making it easy to
# parse dates, create date ranges, index and slice data by time, handle time zones,
# resample data to different frequencies, and perform time-based calculations.

# -----------------------------------------------------------------------------
# 1. Pandas Timestamp and DatetimeIndex Objects
# -----------------------------------------------------------------------------
print("--- 1. Pandas Timestamp and DatetimeIndex Objects ---")

# --- a) Timestamp ---
# Pandas Timestamp object is the equivalent of Python's datetime.datetime object
# but is more efficient and provides additional functionalities.
# It represents a single point in time.
ts_scalar = pd.Timestamp('2023-10-26 10:30:00')
print(f"\nPandas Timestamp object: {ts_scalar}")
print(f"Year: {ts_scalar.year}, Month: {ts_scalar.month}, Day: {ts_scalar.day}")
print(f"Hour: {ts_scalar.hour}, Minute: {ts_scalar.minute}, Second: {ts_scalar.second}")
print(f"Day of week: {ts_scalar.day_name()} (Monday=0, Sunday=6: {ts_scalar.dayofweek})")

# Timestamp with time zone
ts_with_tz = pd.Timestamp('2023-10-26 10:30:00', tz='America/New_York')
print(f"\nTimestamp with timezone: {ts_with_tz}")

# --- b) DatetimeIndex ---
# A DatetimeIndex is an Index object containing Timestamp objects.
# It's used to index Series or DataFrames with time-based information.
dates_list = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
dt_index = pd.to_datetime(dates_list)
print(f"\nDatetimeIndex from a list of date strings: {dt_index}")
print(f"Type of dt_index: {type(dt_index)}")

# Creating a Series with a DatetimeIndex
data_series = pd.Series([10, 15, 20, 25], index=dt_index)
print("\nSeries with a DatetimeIndex:")
print(data_series)


# -----------------------------------------------------------------------------
# 2. Creating Date Ranges
# -----------------------------------------------------------------------------
print("\n--- 2. Creating Date Ranges ---")

# --- a) pd.to_datetime() ---
# Converts argument to datetime. Can handle various input formats.
# Already used above, but here's another example:
date_str = "27/10/2023"
dt_obj = pd.to_datetime(date_str, format='%d/%m/%Y') # Specify format for non-standard strings
print(f"\nConverted '{date_str}' to datetime: {dt_obj}")

# Handling multiple date formats or errors
mixed_dates = ['2023-10-26', '27/10/2023', 'Oct 28, 2023', 'invalid_date']
parsed_dates = pd.to_datetime(mixed_dates, errors='coerce') # 'coerce' turns unparseable dates into NaT (Not a Time)
print("\nParsing mixed date formats (errors='coerce'):")
print(parsed_dates)


# --- b) pd.date_range() ---
# Generates a DatetimeIndex with a fixed frequency.
# Specify start, end, and/or periods, and frequency.

# Date range with start and end dates (inclusive by default)
date_idx_start_end = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D') # 'D' for daily frequency
print("\nDaily date range from start to end:")
print(date_idx_start_end)

# Date range with start date and number of periods
date_idx_start_periods = pd.date_range(start='2023-03-01', periods=5, freq='M') # 'M' for end of month frequency
print("\nMonthly date range for 5 periods from start:")
print(date_idx_start_periods) # Gives end of month dates

date_idx_start_periods_b = pd.date_range(start='2023-03-01', periods=5, freq='B') # 'B' for business day frequency
print("\nBusiness day date range for 5 periods:")
print(date_idx_start_periods_b)

# Common frequency aliases:
# 'H': Hourly, 'T' or 'min': Minutely, 'S': Secondly
# 'D': Daily, 'B': Business daily
# 'W': Weekly, 'M': End of month, 'MS': Start of month
# 'Q': End of quarter, 'QS': Start of quarter
# 'A' or 'Y': End of year, 'AS' or 'YS': Start of year

# -----------------------------------------------------------------------------
# Sample Time Series Data for Examples
# -----------------------------------------------------------------------------
print("\n--- Sample Time Series Data for Examples ---")
rng = pd.date_range('2023-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randint(0, 100, len(rng)), index=rng)
df_ts = pd.DataFrame({'Value': ts, 'Value2': np.random.randn(len(rng)) * 10}, index=rng)
print("\nSample Time Series (Series):")
print(ts.head())
print("\nSample Time Series (DataFrame):")
print(df_ts.head())


# -----------------------------------------------------------------------------
# 3. Time Series Indexing, Slicing, and Selection
# -----------------------------------------------------------------------------
print("\n--- 3. Time Series Indexing, Slicing, and Selection ---")
# DatetimeIndex allows for powerful indexing and slicing.

# Selecting a specific date
print(f"\nValue at '2023-01-05':\n{df_ts.loc['2023-01-05']}")

# Slicing with date strings (partial string indexing)
print("\nValues for January 2023:")
print(df_ts['2023-01'].head()) # Selects all days in January 2023

print("\nValues from '2023-01-15' to '2023-01-20':")
print(df_ts['2023-01-15':'2023-01-20'])

# Slicing with datetime objects
start_date = pd.to_datetime('2023-02-10')
end_date = pd.to_datetime('2023-02-15')
print(f"\nValues from {start_date.date()} to {end_date.date()}:")
print(df_ts[start_date:end_date])

# Using .truncate() for selecting a date range
print("\nValues truncated before '2023-01-05' and after '2023-01-10':")
print(df_ts.truncate(before='2023-01-05', after='2023-01-10'))


# -----------------------------------------------------------------------------
# 4. Time Zone Handling
# -----------------------------------------------------------------------------
print("\n--- 4. Time Zone Handling ---")
# Pandas provides robust support for time zones.

# --- a) Timezone Localization (tz_localize) ---
# Assigns a time zone to a naive DatetimeIndex (one without timezone info).
naive_index = pd.date_range('2023-11-01', periods=3, freq='H')
ts_naive = pd.Series([1, 2, 3], index=naive_index)
print("\nNaive time series:")
print(ts_naive)

ts_localized_ny = ts_naive.tz_localize('America/New_York')
print("\nTime series localized to 'America/New_York':")
print(ts_localized_ny)
print(f"Index type: {type(ts_localized_ny.index)}, Timezone: {ts_localized_ny.index.tz}")

# Attempting to localize an already localized series will raise an error.

# --- b) Timezone Conversion (tz_convert) ---
# Converts a timezone-aware DatetimeIndex to another timezone.
ts_converted_london = ts_localized_ny.tz_convert('Europe/London')
print("\nTime series converted to 'Europe/London':")
print(ts_converted_london)

# Timestamps can also have timezone info
ts_scalar_utc = pd.Timestamp('2023-10-26 12:00:00', tz='UTC')
print(f"\nScalar Timestamp in UTC: {ts_scalar_utc}")
print(f"Converted to US/Eastern: {ts_scalar_utc.tz_convert('US/Eastern')}")


# -----------------------------------------------------------------------------
# 5. Resampling Time Series Data (resample())
# -----------------------------------------------------------------------------
print("\n--- 5. Resampling Time Series Data (resample()) ---")
# Resampling involves changing the frequency of your time series data.
# - Downsampling: Aggregating data to a lower frequency (e.g., daily to monthly).
# - Upsampling: Converting data to a higher frequency (e.g., daily to hourly).

# Using df_ts for these examples.
print("\nOriginal DataFrame head for resampling:")
print(df_ts.head())

# --- a) Downsampling ---
# Requires an aggregation function (e.g., mean, sum, min, max, ohlc).
monthly_mean_df = df_ts.resample('M').mean() # 'M' for end of month frequency
print("\nMonthly mean (downsampled):")
print(monthly_mean_df)

weekly_sum_df = df_ts.resample('W').sum() # 'W' for weekly frequency (ends on Sunday by default)
print("\nWeekly sum (downsampled):")
print(weekly_sum_df.head())

# OHLC (Open, High, Low, Close) resampling for financial data (using 'Value' column)
monthly_ohlc_value = df_ts['Value'].resample('M').ohlc()
print("\nMonthly OHLC for 'Value' column:")
print(monthly_ohlc_value)

# --- b) Upsampling ---
# Often requires a fill method for the new, higher-frequency points.
# Example: Upsample daily data to 6-hourly data
hourly_upsampled_df = df_ts.resample('6H') # This creates a Resampler object

# Fill methods for upsampling:
# - ffill(): Forward fill (propagate last valid observation forward).
# - bfill(): Backward fill (use next valid observation to fill gap).
# - interpolate(): Fill with interpolated values.
# - asfreq(): No filling, introduces NaNs.

upsampled_ffill = hourly_upsampled_df.ffill()
print("\nUpsampled to 6-hourly frequency (forward fill):")
print(upsampled_ffill.head(10)) # Show more rows to see effect

upsampled_bfill = hourly_upsampled_df.bfill()
print("\nUpsampled to 6-hourly frequency (backward fill):")
print(upsampled_bfill.head(10))

upsampled_asfreq_nan = hourly_upsampled_df.asfreq()
print("\nUpsampled to 6-hourly frequency (asfreq, with NaNs):")
print(upsampled_asfreq_nan.head(10))

# -----------------------------------------------------------------------------
# 6. Shifting and Lagging Data (shift())
# -----------------------------------------------------------------------------
print("\n--- 6. Shifting and Lagging Data (shift()) ---")
# .shift() is used to shift the data in a Series or DataFrame by a desired number of periods.
# This is useful for computing differences over time or creating lagged variables.

print("\nOriginal 'Value' column head:")
print(df_ts['Value'].head())

# Shift data forward (positive periods)
shifted_forward = df_ts['Value'].shift(1)
print("\n'Value' column shifted forward by 1 period (lagged):")
print(shifted_forward.head()) # First value becomes NaN

# Shift data backward (negative periods)
shifted_backward = df_ts['Value'].shift(-1)
print("\n'Value' column shifted backward by 1 period (leading):")
print(shifted_backward.head()) # Last value becomes NaN

# Calculating percentage change using shift
# pct_change = (current - previous) / previous
df_ts['Value_PctChange'] = (df_ts['Value'] - df_ts['Value'].shift(1)) / df_ts['Value'].shift(1) * 100
print("\nDataFrame with 'Value_PctChange':")
print(df_ts[['Value', 'Value_PctChange']].head())

# Shifting with a time frequency (requires DatetimeIndex with frequency)
# This shifts the index, not just the data.
shifted_by_time = df_ts.shift(periods=2, freq='D') # Shifts data associated with index + 2 days
print("\nDataFrame shifted by 2 days (index also shifts, data moves with it):")
print(shifted_by_time.head()) # Values from original 2023-01-01 are now at 2023-01-03


# -----------------------------------------------------------------------------
# 7. Rolling Window Calculations (rolling())
# -----------------------------------------------------------------------------
print("\n--- 7. Rolling Window Calculations (rolling()) ---")
# Rolling window calculations apply a function to a sliding window of data.
# Useful for smoothing data, calculating moving averages, etc.

# .rolling(window) creates a Rolling object.
# 'window' is the number of observations used for each calculation.

# Rolling mean (Moving Average)
rolling_mean_3d = df_ts['Value'].rolling(window=3).mean()
print("\n3-day rolling mean for 'Value':")
print(pd.DataFrame({'Value': df_ts['Value'], 'RollingMean_3D': rolling_mean_3d}).head())
# First two values of rolling mean will be NaN as window is 3.

# Rolling sum
rolling_sum_5d = df_ts['Value'].rolling(window=5).sum()
print("\n5-day rolling sum for 'Value':")
print(pd.DataFrame({'Value': df_ts['Value'], 'RollingSum_5D': rolling_sum_5d}).head(7))

# Other functions can be applied: std, var, min, max, median, count, apply (custom function)
rolling_std_3d = df_ts['Value'].rolling(window=3).std()
print("\n3-day rolling standard deviation for 'Value':")
print(pd.DataFrame({'Value': df_ts['Value'], 'RollingStd_3D': rolling_std_3d}).head())

# Centering the window (result is labeled at the center of the window)
# Default is right-aligned (labeled at the end of the window).
rolling_mean_3d_centered = df_ts['Value'].rolling(window=3, center=True).mean()
print("\n3-day centered rolling mean for 'Value':")
print(pd.DataFrame({'Value': df_ts['Value'], 'RollingMean_3D_Centered': rolling_mean_3d_centered}).head())
# Now first and last values are NaN for a window of 3.

# Time-based rolling windows (using offset strings like '3D' for 3 days)
# The index must be a DatetimeIndex.
# This considers the time interval, not just number of observations.
df_ts_sparse = df_ts.iloc[[0, 1, 5, 6, 10]] # Create a sparse series
print("\nSparse time series for time-based rolling window example:")
print(df_ts_sparse[['Value']])
rolling_time_3d = df_ts_sparse['Value'].rolling(window='3D').sum()
print("\nRolling sum over a 3-day window (time-based):")
print(pd.DataFrame({'Value': df_ts_sparse['Value'], 'RollingSum_Time3D': rolling_time_3d}))

# -----------------------------------------------------------------------------
# 8. Expanding Window Calculations (expanding())
# -----------------------------------------------------------------------------
print("\n--- 8. Expanding Window Calculations (expanding()) ---")
# Expanding window calculations apply a function to all data points up to the current point.
# The window size increases as it moves through the data.

# .expanding(min_periods) creates an Expanding object.
# 'min_periods' is the minimum number of observations required for calculation.

# Expanding sum (cumulative sum)
expanding_sum = df_ts['Value'].expanding(min_periods=1).sum()
print("\nExpanding sum (cumulative sum) for 'Value':")
print(pd.DataFrame({'Value': df_ts['Value'], 'ExpandingSum': expanding_sum}).head())
# This is equivalent to .cumsum()

# Expanding mean
expanding_mean = df_ts['Value'].expanding(min_periods=2).mean()
print("\nExpanding mean for 'Value' (min_periods=2):")
print(pd.DataFrame({'Value': df_ts['Value'], 'ExpandingMean': expanding_mean}).head())
# First value of expanding mean is NaN as min_periods=2.

# Can be used with other functions like std, var, min, max, apply, etc.
expanding_max = df_ts['Value'].expanding(min_periods=1).max()
print("\nExpanding maximum for 'Value':")
print(pd.DataFrame({'Value': df_ts['Value'], 'ExpandingMax': expanding_max}).head())

# -----------------------------------------------------------------------------
# End of Working with Time Series Data Topic
# -----------------------------------------------------------------------------
print("\nEnd of pandas_topic_5.py: Working with Time Series Data")

# Clean up
del ts_scalar, ts_with_tz, dates_list, dt_index, data_series, date_str, dt_obj
del mixed_dates, parsed_dates, date_idx_start_end, date_idx_start_periods, date_idx_start_periods_b
del rng, ts, df_ts, start_date, end_date
del naive_index, ts_naive, ts_localized_ny, ts_converted_london, ts_scalar_utc
del monthly_mean_df, weekly_sum_df, monthly_ohlc_value, hourly_upsampled_df
del upsampled_ffill, upsampled_bfill, upsampled_asfreq_nan
del shifted_forward, shifted_backward, shifted_by_time
del rolling_mean_3d, rolling_sum_5d, rolling_std_3d, rolling_mean_3d_centered
del df_ts_sparse, rolling_time_3d
del expanding_sum, expanding_mean, expanding_max
print("\nCleaned up intermediate DataFrames and objects.")
