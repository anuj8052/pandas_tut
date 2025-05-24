# pandas_topic_9.py
# Topic: Performance Optimization and Working with Large Datasets using Pandas

import pandas as pd
import numpy as np
import os # For file operations in examples
import time # For timing operations

# -----------------------------------------------------------------------------
# Introduction
# -----------------------------------------------------------------------------
# Pandas is highly optimized for many operations, but when working with very
# large datasets or computationally intensive tasks, performance can become a concern.
# This topic covers techniques within Pandas to improve memory usage and execution speed,
# as well as strategies for handling datasets that might not fit comfortably in memory.

# -----------------------------------------------------------------------------
# 1. Efficient Data Types
# -----------------------------------------------------------------------------
print("--- 1. Efficient Data Types ---")
# Using appropriate data types can significantly reduce memory footprint and speed up computations.

# Create a sample DataFrame
data_large = {
    'int_col_64': np.random.randint(0, 1000, size=1_000_000),
    'float_col_64': np.random.rand(1_000_000) * 1000,
    'string_col_object': ['Category_' + str(i % 5) for i in range(1_000_000)],
    'bool_col': np.random.choice([True, False], size=1_000_000)
}
df_large = pd.DataFrame(data_large)
print("Original DataFrame memory usage:")
df_large.info(memory_usage='deep')

# --- a) Downcasting Numerical Types ---
print("\n--- a) Downcasting Numerical Types ---")
# For integers and floats, if the range of values is known to be small,
# you can downcast to smaller dtypes (e.g., int64 -> int32, int16, int8; float64 -> float32).

# Example: If 'int_col_64' only contains values between 0 and 1000, it can fit in int16.
df_large['int_col_16'] = df_large['int_col_64'].astype(np.int16)
# Example: 'float_col_64' can often be represented as float32 with acceptable precision loss.
df_large['float_col_32'] = df_large['float_col_64'].astype(np.float32)

print("\nMemory usage after downcasting numerical types:")
df_large[['int_col_16', 'float_col_32']].info(memory_usage='deep')
original_int_mem = df_large['int_col_64'].memory_usage(deep=True)
downcast_int_mem = df_large['int_col_16'].memory_usage(deep=True)
print(f"Memory for int_col_64: {original_int_mem} bytes")
print(f"Memory for int_col_16: {downcast_int_mem} bytes (factor: {original_int_mem/downcast_int_mem:.2f}x smaller)")


# --- b) Using 'category' dtype for Low-Cardinality Strings ---
print("\n--- b) Using 'category' dtype for Low-Cardinality Strings ---")
# If a string column has a small number of unique values (low cardinality),
# converting it to the 'category' dtype can save a lot of memory and speed up group-by operations.
# Pandas stores this as integer codes (pointers) and a dictionary of unique values.

df_large['string_col_category'] = df_large['string_col_object'].astype('category')

print("\nMemory usage after converting string column to 'category':")
df_large[['string_col_object', 'string_col_category']].info(memory_usage='deep')
original_str_mem = df_large['string_col_object'].memory_usage(deep=True)
category_str_mem = df_large['string_col_category'].memory_usage(deep=True)
print(f"Memory for string_col_object: {original_str_mem} bytes")
print(f"Memory for string_col_category: {category_str_mem} bytes (factor: {original_str_mem/category_str_mem:.2f}x smaller)")


# --- c) Using pd.StringDtype() for String Data ---
print("\n--- c) Using pd.StringDtype() for String Data ---")
# Pandas introduced `pd.StringDtype()` (often aliased as "string") which is a dedicated
# string data type, different from 'object' dtype that can hold mixed types.
# It behaves more consistently for string operations and can handle missing values (pd.NA)
# more cleanly than object dtype (which uses np.nan for missing strings).
# Memory benefits depend on the data; it might not always be smaller than 'object' but provides better type safety.

df_large['string_col_pdstring'] = df_large['string_col_object'].astype(pd.StringDtype())
print("\nMemory usage with pd.StringDtype():")
df_large[['string_col_object', 'string_col_pdstring']].info(memory_usage='deep')
pdstring_mem = df_large['string_col_pdstring'].memory_usage(deep=True)
print(f"Memory for string_col_pdstring: {pdstring_mem} bytes")
print("Note: pd.StringDtype offers type consistency and better NA handling for strings.")


# -----------------------------------------------------------------------------
# 2. Efficient Operations
# -----------------------------------------------------------------------------
print("\n--- 2. Efficient Operations ---")

# --- a) Avoiding Loops: Vectorized Operations ---
print("\n--- a) Avoiding Loops: Vectorized Operations ---")
# Pandas operations are typically implemented using NumPy arrays, which means
# they are C-compiled and operate on entire arrays (vectorized) rather than element by element.
# Always prefer vectorized operations over Python loops.

# Example: Add 10 to 'int_col_16'
# Inefficient loop-based approach (DO NOT DO THIS for large data):
# start_time = time.time()
# result_loop = []
# for val in df_large['int_col_16']:
#     result_loop.append(val + 10)
# df_large['int_col_loop_add'] = result_loop
# print(f"Loop addition time: {time.time() - start_time:.4f}s")

# Efficient vectorized approach:
start_time = time.time()
df_large['int_col_vector_add'] = df_large['int_col_16'] + 10
print(f"Vectorized addition time: {time.time() - start_time:.4f}s (Expected to be much faster)")

# Other examples of vectorized operations:
df_large['float_col_log'] = np.log(df_large['float_col_32'].abs() + 1) # Using NumPy ufunc
df_large['string_col_upper'] = df_large['string_col_pdstring'].str.upper() # Pandas string methods

print("Vectorized operations (log, upper) performed.")


# --- b) Using .apply() Judiciously ---
print("\n--- b) Using .apply() Judiciously ---")
# `DataFrame.apply()` or `Series.apply()` can be powerful for custom operations,
# but they can be slow, especially with complex functions, as they often involve
# Python-level loops internally (unless the function itself is vectorized).
# Try to use built-in vectorized methods first. If `apply` is needed,
# pass a NumPy ufunc if possible, or optimize the Python function.

# Example: Custom function applied row-wise (often slow)
def custom_logic(row):
    if row['int_col_16'] > 500:
        return row['float_col_32'] * 2
    else:
        return row['float_col_32'] / 2

# For demonstration, using on a smaller subset as apply can be very slow
df_subset = df_large.head(10000).copy()
start_time = time.time()
df_subset['apply_result'] = df_subset.apply(custom_logic, axis=1)
print(f"Time for .apply() on 10k rows: {time.time() - start_time:.4f}s")

# Vectorized alternative using np.where or boolean indexing
start_time = time.time()
df_subset['vector_result'] = np.where(
    df_subset['int_col_16'] > 500,
    df_subset['float_col_32'] * 2,
    df_subset['float_col_32'] / 2
)
print(f"Time for vectorized alternative on 10k rows: {time.time() - start_time:.4f}s")
print("Vectorized alternative is significantly faster than row-wise apply().")


# --- c) Using .itertuples() and .iterrows() (and when to avoid) ---
print("\n--- c) Using .itertuples() and .iterrows() (and when to avoid) ---")
# `iterrows()`: Iterates over DataFrame rows as (index, Series) pairs. It's convenient but very slow
# due to the overhead of creating a Series object for each row and type preservation issues.
# `itertuples()`: Iterates over DataFrame rows as namedtuples. It's generally faster than `iterrows()`
# because it's less overhead. `itertuples(index=False, name=None)` can be even faster.

# These methods should generally be AVOIDED for large datasets if a vectorized
# solution exists. They are essentially loops.

# Example (illustrative, not recommended for performance):
# total_sum_itertuples = 0
# start_time = time.time()
# for row_tuple in df_subset.itertuples(index=False): # index=False avoids index in tuple
#     total_sum_itertuples += row_tuple.int_col_16 + row_tuple.float_col_32
# print(f"Time for .itertuples() sum on 10k rows: {time.time() - start_time:.4f}s")
# print(f"Sum from itertuples: {total_sum_itertuples:.2f}")

# Vectorized sum is instant:
# start_time = time.time()
# vectorized_sum = (df_subset['int_col_16'] + df_subset['float_col_32']).sum()
# print(f"Time for vectorized sum on 10k rows: {time.time() - start_time:.4f}s")
# print(f"Sum from vectorized op: {vectorized_sum:.2f}")
print("Iterrows/itertuples are much slower than vectorized operations. Avoid if possible.")


# -----------------------------------------------------------------------------
# 3. Working with Large Datasets
# -----------------------------------------------------------------------------
print("\n--- 3. Working with Large Datasets ---")

# --- a) Reading Data in Chunks ---
print("\n--- a) Reading Data in Chunks ---")
# For datasets too large to fit in memory, `pd.read_csv` (and other readers
# like `read_json`, `read_sql`) provide a `chunksize` parameter.
# This returns an iterator that yields DataFrames of the specified chunk size.

# Create a dummy large CSV file for this example
dummy_csv_path = "large_dummy_data.csv"
num_rows_csv = 2_000_000 # Larger than df_large for more realistic chunking
if not os.path.exists(dummy_csv_path):
    print(f"Creating dummy CSV file: {dummy_csv_path} with {num_rows_csv} rows...")
    temp_df = pd.DataFrame({
        'ID': np.arange(num_rows_csv),
        'Value': np.random.rand(num_rows_csv),
        'Category': ['Type_' + str(i % 10) for i in range(num_rows_csv)]
    })
    temp_df.to_csv(dummy_csv_path, index=False)
    del temp_df # Free memory
    print("Dummy CSV created.")
else:
    print(f"Dummy CSV file {dummy_csv_path} already exists.")


# Process data in chunks
chunk_iter = pd.read_csv(dummy_csv_path, chunksize=500_000) # Read 500k rows at a time
print(f"Type of chunk_iter: {type(chunk_iter)}")

total_value_sum = 0
num_chunks_processed = 0
category_counts = pd.Series(dtype=int)

print("Processing CSV in chunks...")
for chunk_df in chunk_iter:
    num_chunks_processed += 1
    print(f"  Processing chunk {num_chunks_processed}, shape: {chunk_df.shape}")
    # Example processing: sum of 'Value' and count 'Category'
    total_value_sum += chunk_df['Value'].sum()
    current_counts = chunk_df['Category'].value_counts()
    category_counts = category_counts.add(current_counts, fill_value=0)
    # Remember to optimize dtypes within the chunk processing loop if memory is very tight
    # chunk_df['Value'] = chunk_df['Value'].astype(np.float32)
    # chunk_df['Category'] = chunk_df['Category'].astype('category')

print(f"\nFinished processing {num_chunks_processed} chunks.")
print(f"Total sum of 'Value' column: {total_value_sum:.2f}")
print("Aggregated category counts:")
print(category_counts.sort_values(ascending=False).head())


# --- b) Sampling Data for Exploration ---
print("\n--- b) Sampling Data for Exploration ---")
# When dealing with a very large dataset, loading a sample can be useful for
# initial exploration, understanding data structure, and testing processing pipelines.

# If the file is too large to even load a sample directly, you might read a few chunks,
# or use command-line tools (like `shuf` on Linux/macOS) to create a smaller sample file.

# If you can load the whole dataset (or a large chunk):
# df_sample = df_large.sample(n=1000) # Sample 1000 random rows
# df_sample_frac = df_large.sample(frac=0.01) # Sample 1% of rows

# Example: Taking a sample from the first chunk of our dummy CSV
first_chunk_for_sample = pd.read_csv(dummy_csv_path, nrows=100000) # Read first 100k rows
sample_from_chunk = first_chunk_for_sample.sample(n=100, random_state=42)
print("\nSample of 100 rows from the first 100k rows of the CSV:")
print(sample_from_chunk.head())


# -----------------------------------------------------------------------------
# 4. Efficient File Formats
# -----------------------------------------------------------------------------
print("\n--- 4. Efficient File Formats ---")
# For large datasets, CSV is often not the most efficient format for I/O or storage.
# Binary columnar formats like Parquet or Feather are generally much faster for
# reading and writing, and can result in smaller file sizes due to compression
# and efficient encoding of data types.

# - **Parquet (`.parquet`)**:
#   - Widely adopted, good for long-term storage and interoperability (e.g., with Spark, Dask).
#   - Supports various compression codecs (snappy, gzip, brotli).
#   - Columnar storage allows reading only selected columns efficiently.
#   - Requires `pyarrow` or `fastparquet` library.
#   `df.to_parquet('data.parquet')`
#   `pd.read_parquet('data.parquet')`

# - **Feather (`.feather`)**:
#   - Designed for fast, language-agnostic data frame storage (Python and R).
#   - Also columnar, very fast for I/O.
#   - Good for temporary storage or quick data sharing between Python/R.
#   - Requires `pyarrow`.
#   `df.to_feather('data.feather')`
#   `pd.read_feather('data.feather')`

# - **HDF5 (`.h5`, `.hdf5`)**:
#   - Hierarchical Data Format, good for storing multiple datasets and metadata in one file.
#   - Can be efficient but sometimes more complex to manage.
#   - Requires `tables` (PyTables) library.

# Example: Writing our processed chunk to Parquet (if library available)
parquet_output_path = "processed_chunk_sample.parquet"
try:
    first_chunk_for_sample.to_parquet(parquet_output_path, engine='pyarrow', index=False)
    print(f"\nSuccessfully wrote a sample chunk to {parquet_output_path}")
    # df_read_parquet = pd.read_parquet(parquet_output_path)
    # print("Read back from Parquet (head):")
    # print(df_read_parquet.head())
except ImportError:
    print(f"\nSkipping Parquet example: pyarrow library not found. (pip install pyarrow)")
except Exception as e:
    print(f"\nError with Parquet example: {e}")

print("Consider using Parquet or Feather for faster I/O with large datasets.")


# -----------------------------------------------------------------------------
# 5. Brief Introduction to Out-of-Core/Parallel Computing (Conceptual)
# -----------------------------------------------------------------------------
print("\n--- 5. Brief Introduction to Out-of-Core/Parallel Computing (Conceptual) ---")
# When datasets are truly massive (larger than available RAM) or computations
# are very complex and can be parallelized, Pandas alone might not be sufficient.
# Libraries like Dask and Modin extend Pandas-like functionality for these scenarios.

# --- a) Dask ---
# - **How it works**: Dask breaks large DataFrames into many smaller Pandas DataFrames (partitions)
#   and orchestrates computations on these partitions in parallel, potentially out-of-core (spilling to disk).
# - **API**: Dask DataFrame API is very similar to Pandas, making it relatively easy to switch for many operations.
#   `import dask.dataframe as dd`
# - **When to consider**:
#   - Datasets significantly larger than RAM.
#   - CPU-bound computations that can benefit from parallel processing on multi-core CPUs.
#   - Distributed computing across multiple machines.
# - **Key idea**: Dask builds a task graph of operations and executes it lazily. You call `.compute()`
#   to get the actual result.

# --- b) Modin ---
# - **How it works**: Modin aims to parallelize Pandas operations by changing the execution engine
#   (e.g., using Dask or Ray in the backend) without requiring users to change their Pandas code significantly.
# - **API**: Aims to be a drop-in replacement for Pandas.
#   `import modin.pandas as pd` (then use `pd` as you would with regular Pandas)
# - **When to consider**:
#   - You have existing Pandas code and want to speed it up on multi-core machines with minimal code changes.
#   - Datasets that fit in memory but computations are slow and parallelizable.
# - **Key idea**: Modin parallelizes many Pandas functions transparently.

# --- c) General Considerations ---
# - **Overhead**: These libraries have their own overhead. For very small datasets or operations
#   that are already fast in Pandas, they might not provide benefits or could even be slower.
# - **Not a silver bullet**: Not all Pandas operations are easily parallelizable or efficiently
#   handled by these libraries. Some custom or complex operations might still require careful thought.
# - **Learning Curve**: While the API is similar, understanding the distributed/parallel computation
#   model can be important for debugging and advanced optimization.

print("\nDask and Modin are powerful options when Pandas hits its limits on a single machine.")
print("They allow working with larger-than-memory datasets and can parallelize computations.")


# -----------------------------------------------------------------------------
# End of Performance Optimization and Large Datasets Topic
# -----------------------------------------------------------------------------
print("\nEnd of pandas_topic_9.py: Performance Optimization and Working with Large Datasets")

# Optional: Clean up dummy files
if os.path.exists(dummy_csv_path):
    # os.remove(dummy_csv_path)
    print(f"\nNote: Dummy CSV file '{dummy_csv_path}' was created and can be manually deleted.")
if os.path.exists(parquet_output_path):
    # os.remove(parquet_output_path)
    print(f"Note: Parquet file '{parquet_output_path}' was created and can be manually deleted.")
