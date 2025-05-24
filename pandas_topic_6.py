# pandas_topic_6.py
# Topic: Input and Output (I/O) Operations using Pandas

import pandas as pd
import os
import json
import sqlite3 # For SQL examples with sqlite
from sqlalchemy import create_engine # For SQL examples

# -----------------------------------------------------------------------------
# Introduction to I/O Operations
# -----------------------------------------------------------------------------
# Pandas provides a rich set of functions to read data from various file formats
# and to write data back to these formats. This is crucial for loading data for
# analysis and for saving results.

# Important Notes on Libraries:
# - Excel: You'll need to install `openpyxl` (for .xlsx files) or `xlrd` (for older .xls files).
#   Install using: `pip install openpyxl`
# - SQL: You'll need `sqlalchemy` and a database-specific driver (e.g., `psycopg2` for PostgreSQL,
#   `mysqlclient` for MySQL). For these examples, we use `sqlite3` which is built-in,
#   but `sqlalchemy` is still good practice for a consistent interface.
#   Install using: `pip install sqlalchemy psycopg2-binary` (or other drivers)
# - Other formats like Parquet might require `pyarrow` or `fastparquet`.
#   Install using: `pip install pyarrow` or `pip install fastparquet`

# -----------------------------------------------------------------------------
# Sample DataFrame for Writing Examples
# -----------------------------------------------------------------------------
data_for_writing = {
    'col1': [1, 2, 3, 4, 5],
    'col2': ['a', 'b', 'c', 'd', 'e'],
    'col3': [1.1, 2.2, 3.3, 4.4, 5.5],
    'date_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
}
df_to_write = pd.DataFrame(data_for_writing)
print("--- Sample DataFrame for Writing Examples ---")
print(df_to_write)
print("\n")

# Create a directory for output files if it doesn't exist
output_dir = "pandas_io_examples"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")


# -----------------------------------------------------------------------------
# 1. CSV Files
# -----------------------------------------------------------------------------
print("--- 1. CSV Files ---")
csv_file_path = os.path.join(output_dir, "sample_data.csv")

# --- a) Writing to CSV (DataFrame.to_csv()) ---
print(f"\nWriting DataFrame to CSV: {csv_file_path}")
df_to_write.to_csv(
    csv_file_path,
    sep=',',         # Separator (delimiter), default is ','
    index=False,     # Whether to write DataFrame index as a column, default True
    header=True,     # Whether to write column names, default True
    # columns=['col1', 'col2'], # Optional: list of columns to write
    mode='w'         # Write mode, 'w' for overwrite, 'a' for append
)
print(f"DataFrame successfully written to {csv_file_path}")

# --- b) Reading from CSV (pd.read_csv()) ---
print(f"\nReading DataFrame from CSV: {csv_file_path}")
df_from_csv = pd.read_csv(
    csv_file_path,
    sep=',',                # Separator used in the file
    header=0,               # Row number to use as column names (0 is the first row after sep)
                            # Use None if there are no column names
    index_col=None,         # Column to use as row labels (index). E.g., index_col=0 for first column.
    # usecols=['col1', 'col3', 'date_col'], # List of column names to read
    dtype={'col1': int, 'col2': str}, # Dictionary to specify data types for columns
    parse_dates=['date_col'], # List of columns to parse as dates
    # nrows=3               # Number of rows to read from the beginning of the file
)
print("DataFrame read from CSV:")
print(df_from_csv)
print("\nData types from CSV read:")
print(df_from_csv.dtypes)


# -----------------------------------------------------------------------------
# 2. Excel Files
# -----------------------------------------------------------------------------
# Requires `openpyxl` for .xlsx files. `pip install openpyxl`
print("\n--- 2. Excel Files ---")
excel_file_path = os.path.join(output_dir, "sample_data.xlsx")

# --- a) Writing to Excel (DataFrame.to_excel()) ---
print(f"\nWriting DataFrame to Excel: {excel_file_path}")
df_to_write.to_excel(
    excel_file_path,
    sheet_name='Sheet1',    # Name of the sheet to write to
    index=False,            # Whether to write DataFrame index, default True
    header=True,            # Whether to write column names, default True
    # columns=['col1', 'col3'],# Optional: list of columns to write
    # startrow=0,           # (0-indexed) row to start writing DataFrame
    # startcol=0            # (0-indexed) column to start writing DataFrame
    # engine='openpyxl'     # Explicitly specify engine if needed
)
print(f"DataFrame successfully written to {excel_file_path}, sheet 'Sheet1'")

# --- b) Reading from Excel (pd.read_excel()) ---
print(f"\nReading DataFrame from Excel: {excel_file_path}")
try:
    df_from_excel = pd.read_excel(
        excel_file_path,
        sheet_name='Sheet1',    # Specify sheet name or index (0-indexed)
                                # Can be a list of names/indexes to get a dict of DataFrames
        header=0,               # Row to use for column names
        index_col=None,         # Column to use as index
        # usecols='A:C',        # Specify columns to read (e.g., 'A,C' or 'A:C' or list of names/indexes)
        # engine='openpyxl'     # Explicitly specify engine if needed
    )
    print("DataFrame read from Excel (Sheet1):")
    print(df_from_excel)
    print("\nData types from Excel read:")
    print(df_from_excel.dtypes) # Note: dates usually come as datetime64[ns] from Excel
except ImportError:
    print("openpyxl library not found. Please install it using: pip install openpyxl")


# --- c) Writing to Multiple Sheets in an Excel File ---
excel_multiple_sheets_path = os.path.join(output_dir, "multiple_sheets.xlsx")
print(f"\nWriting multiple DataFrames to different sheets in: {excel_multiple_sheets_path}")
df_sheet2 = df_to_write.copy()
df_sheet2['col1'] = df_sheet2['col1'] * 10

try:
    with pd.ExcelWriter(excel_multiple_sheets_path, engine='openpyxl') as writer:
        df_to_write.to_excel(writer, sheet_name='OriginalData', index=False)
        df_sheet2.to_excel(writer, sheet_name='ModifiedData', index=False)
    print(f"Successfully written multiple sheets to {excel_multiple_sheets_path}")

    # Reading specific sheet from the multi-sheet file
    df_from_multi_sheet_excel = pd.read_excel(excel_multiple_sheets_path, sheet_name='ModifiedData')
    print("\nReading 'ModifiedData' sheet from the multi-sheet Excel file:")
    print(df_from_multi_sheet_excel.head())
except ImportError:
    print("openpyxl library not found. Please install it using: pip install openpyxl")


# -----------------------------------------------------------------------------
# 3. JSON Files
# -----------------------------------------------------------------------------
print("\n--- 3. JSON Files ---")
json_file_path = os.path.join(output_dir, "sample_data.json")
json_lines_file_path = os.path.join(output_dir, "sample_data_lines.json")

# --- a) Writing to JSON (DataFrame.to_json()) ---
# 'orient' parameter determines the JSON structure. Common options:
# 'records': list of dicts, like [{column -> value}, … , {column -> value}]
# 'columns': dict of dicts, like {column -> {index -> value}}
# 'index': dict of dicts, like {index -> {column -> value}}
# 'split': dict like {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
# 'table': dict like {'schema': {schema}, 'data': {data}} (good for schema preservation)
# 'values': just the values array

print(f"\nWriting DataFrame to JSON (orient='records'): {json_file_path}")
df_to_write.to_json(
    json_file_path,
    orient='records', # Structure of the JSON output
    indent=4          # Number of spaces to use for indentation
)
print(f"DataFrame successfully written to {json_file_path}")

print(f"\nWriting DataFrame to JSON (orient='records', lines=True): {json_lines_file_path}")
df_to_write.to_json(
    json_lines_file_path,
    orient='records',
    lines=True        # Each line is a separate JSON object (useful for streaming)
)
print(f"DataFrame successfully written to {json_lines_file_path}")


# --- b) Reading from JSON (pd.read_json()) ---
print(f"\nReading DataFrame from JSON (orient='records'): {json_file_path}")
df_from_json = pd.read_json(
    json_file_path,
    orient='records',  # Must match the 'orient' used for writing
    # dtype={'col1': int} # Can specify dtypes
)
# Note: pd.read_json by default converts integer-like floats to int if possible.
# Timestamps are often read as integers (milliseconds since epoch) or strings.
# Explicit conversion might be needed for dates if not in ISO format.
df_from_json['date_col'] = pd.to_datetime(df_from_json['date_col'], unit='ms') # Assuming 'date_col' was written as ms
print("DataFrame read from JSON:")
print(df_from_json)
print("\nData types from JSON read:")
print(df_from_json.dtypes)


print(f"\nReading DataFrame from JSON (orient='records', lines=True): {json_lines_file_path}")
df_from_json_lines = pd.read_json(
    json_lines_file_path,
    orient='records',
    lines=True
)
df_from_json_lines['date_col'] = pd.to_datetime(df_from_json_lines['date_col'], unit='ms')
print("DataFrame read from JSON lines:")
print(df_from_json_lines)


# -----------------------------------------------------------------------------
# 4. SQL Databases
# -----------------------------------------------------------------------------
# Requires `sqlalchemy`. For this example, we use `sqlite3` (built-in Python).
# For other DBs (PostgreSQL, MySQL, etc.), install their respective drivers
# e.g., `pip install psycopg2-binary` for PostgreSQL.
print("\n--- 4. SQL Databases ---")
db_file_path = os.path.join(output_dir, "sample_database.sqlite")
table_name = "sample_table"

# Create a SQLAlchemy engine for SQLite
# For an in-memory database: engine = create_engine('sqlite:///:memory:')
# For a file-based database:
engine = create_engine(f'sqlite:///{db_file_path}')
print(f"Using SQLite database engine: {engine}")

# --- a) Writing to SQL (DataFrame.to_sql()) ---
print(f"\nWriting DataFrame to SQL table '{table_name}' in {db_file_path}")
try:
    df_to_write.to_sql(
        name=table_name,       # Name of the SQL table
        con=engine,            # SQLAlchemy engine or DBAPI2 connection
        if_exists='replace',   # What to do if table already exists:
                               # 'fail': Raise ValueError.
                               # 'replace': Drop table before inserting new values.
                               # 'append': Insert new values to the existing table.
        index=False,           # Write DataFrame index as a column, default True
        # dtype={'col1': sqlalchemy.types.Integer} # Optional: specify SQL data types
    )
    print(f"DataFrame successfully written to SQL table '{table_name}'")
except Exception as e:
    print(f"Error writing to SQL: {e}")


# --- b) Reading from SQL (pd.read_sql_query(), pd.read_sql_table(), pd.read_sql()) ---
# pd.read_sql_query: Reads data from a SQL query into a DataFrame.
query = f"SELECT col1, col2, date_col FROM {table_name} WHERE col1 > 2;"
print(f"\nReading DataFrame from SQL query: {query}")
try:
    df_from_sql_query = pd.read_sql_query(query, con=engine)
    print("DataFrame read from SQL query:")
    print(df_from_sql_query)
    # Note: date types from SQL are often read as strings or specific DB date types.
    # Explicit conversion might be needed if not automatically parsed by pandas/sqlalchemy.
    df_from_sql_query['date_col'] = pd.to_datetime(df_from_sql_query['date_col'])
    print("\nData types from SQL query read:")
    print(df_from_sql_query.dtypes)

except Exception as e:
    print(f"Error reading from SQL query: {e}")


# pd.read_sql_table: Reads a SQL database table into a DataFrame.
# This is generally more efficient if you want the whole table.
print(f"\nReading entire SQL table '{table_name}' using pd.read_sql_table:")
try:
    df_from_sql_table = pd.read_sql_table(table_name, con=engine, parse_dates=['date_col'])
    print("DataFrame read from SQL table:")
    print(df_from_sql_table)
    print("\nData types from SQL table read:")
    print(df_from_sql_table.dtypes)
except Exception as e:
    print(f"Error reading from SQL table: {e}")


# pd.read_sql: A general function that can delegate to read_sql_query or read_sql_table.
print(f"\nReading from SQL using generic pd.read_sql (with a query):")
try:
    df_from_generic_sql = pd.read_sql(f"SELECT * FROM {table_name}", con=engine, parse_dates={'date_col': {'format': '%Y-%m-%d %H:%M:%S.%f'}})
    print("DataFrame read using pd.read_sql:")
    print(df_from_generic_sql)
    print("\nData types from generic SQL read:")
    print(df_from_generic_sql.dtypes)
except Exception as e:
    print(f"Error reading with pd.read_sql: {e}")


# -----------------------------------------------------------------------------
# 5. Other File Formats (Brief Mention)
# -----------------------------------------------------------------------------
print("\n--- 5. Other File Formats ---")
# Pandas also supports other efficient binary formats. These often require
# additional libraries like `pyarrow`, `fastparquet`, or `tables`.

# --- a) Parquet ---
# Efficient columnar storage format. Requires `pyarrow` or `fastparquet`.
# `pip install pyarrow` or `pip install fastparquet`
parquet_file_path = os.path.join(output_dir, "sample_data.parquet")
try:
    print(f"\nWriting DataFrame to Parquet: {parquet_file_path}")
    df_to_write.to_parquet(parquet_file_path, engine='pyarrow', index=False)
    df_from_parquet = pd.read_parquet(parquet_file_path, engine='pyarrow')
    print("DataFrame read from Parquet (first 5 rows):")
    print(df_from_parquet.head())
except ImportError:
    print("pyarrow not found. Install with `pip install pyarrow` to use Parquet.")
except Exception as e:
    print(f"Error with Parquet: {e}")


# --- b) HDF5 (Hierarchical Data Format) ---
# Good for storing large, heterogeneous datasets. Requires `tables`.
# `pip install tables`
hdf_file_path = os.path.join(output_dir, "sample_data.h5")
hdf_key = 'my_data'
try:
    print(f"\nWriting DataFrame to HDF5: {hdf_file_path}, key: {hdf_key}")
    df_to_write.to_hdf(hdf_file_path, key=hdf_key, mode='w', format='table')
    df_from_hdf = pd.read_hdf(hdf_file_path, key=hdf_key)
    print("DataFrame read from HDF5 (first 5 rows):")
    print(df_from_hdf.head())
except ImportError:
    print("tables (PyTables) not found. Install with `pip install tables` to use HDF5.")
except Exception as e:
    print(f"Error with HDF5: {e}")

# --- c) Pickle ---
# Python's object serialization format. Stores data in a binary format.
# Useful for saving any Python object, including DataFrames with complex types.
# Note: Pickle files are not secure against erroneous or maliciously constructed data.
# Only unpickle data you trust.
pickle_file_path = os.path.join(output_dir, "sample_data.pkl")
print(f"\nWriting DataFrame to Pickle: {pickle_file_path}")
df_to_write.to_pickle(pickle_file_path)
df_from_pickle = pd.read_pickle(pickle_file_path)
print("DataFrame read from Pickle (first 5 rows):")
print(df_from_pickle.head())


# -----------------------------------------------------------------------------
# End of Input and Output (I/O) Operations Topic
# -----------------------------------------------------------------------------
print("\nEnd of pandas_topic_6.py: Input and Output (I/O) Operations")

# Optional: Clean up created files and directory
# for file_name in os.listdir(output_dir):
#     os.remove(os.path.join(output_dir, file_name))
# os.rmdir(output_dir)
# print(f"\nCleaned up directory: {output_dir} and its contents.")
