### 1. *CSV (Comma Separated Values) File*

import pandas as pd

# Read the CSV file
df = pd.read_csv('your_file.csv')

# Display the first 5 records
print(df.head())


### 2. *Excel File (.xlsx, .xls)*

import pandas as pd

# Read the Excel file
df = pd.read_excel('your_file.xlsx')

# Display the first 5 records
print(df.head())


### 3. *JSON File*

import pandas as pd

# Read the JSON file
df = pd.read_json('your_file.json')

# Display the first 5 records
print(df.head())


### 5. *SQL Database*
# To read data from a SQL database, you can use SQLAlchemy or sqlite3. Here’s an example with sqlite3:

import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('your_database.db')

# Read data from a SQL query
df = pd.read_sql_query("SELECT * FROM your_table LIMIT 5;", conn)

# Display the records
print(df)

# Close the connection
conn.close()


### 6. *Text File (Tab or Space-separated)*

import pandas as pd

# Read a space-separated file (you can change the delimiter accordingly)
df = pd.read_csv('your_file.txt', delimiter='\t')  # for tab-separated, or ' ' for space-separated

# Display the first 5 records
print(df.head())


### 7. *HDF5 File*

import pandas as pd

# Read the HDF5 file
df = pd.read_hdf('your_file.h5')

# Display the first 5 records
print(df.head())


# Read the space-separated numerical data into a DataFrame
df = pd.read_csv('your_file.dat', delimiter=' ')

# Display the first 5 records
print(df.head())


#### Example: Reading comma-separated numerical data with `pandas`

import pandas as pd

# Read the comma-separated numerical data into a DataFrame
df = pd.read_csv('your_file.dat', delimiter=',')

# Display the first 5 records
print(df.head())