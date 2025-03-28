import pandas as pd

# Read the CSV file
df = pd.read_csv('your_file.csv')

# Read the Excel file
df = pd.read_excel('your_file.xlsx')

# Read the JSON file
df = pd.read_json('your_file.json')

# Read a space-separated file (you can change the delimiter accordingly)
df = pd.read_csv('your_file.txt', delimiter='\t')  # for tab-separated, or ' ' for space-separated

# Display the first 5 records
print(df.head())


# Read data from a SQL query

import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('your_database.db')

df = pd.read_sql_query("SELECT * FROM your_table LIMIT 5;", conn)

# Display the records
print(df)

# Close the connection
conn.close()


# Read the HDF5 file
df = pd.read_hdf('your_file.h5')


# Reading space-separated numerical data

import numpy as np

# Read the space-separated numerical data from the .dat file
data = np.loadtxt('your_file.dat')

# Display the first 5 records
print(data[:5])

# Read the comma-separated numerical data from the .dat file
data = np.loadtxt('your_file.dat', delimiter=',')

# Read the tab-separated numerical data from the .dat file
data = np.loadtxt('your_file.dat', delimiter='\t')

# Read the space-separated numerical data into a DataFrame
df = pd.read_csv('your_file.dat', delimiter=' ')

# Read the comma-separated numerical data into a DataFrame
df = pd.read_csv('your_file.dat', delimiter=',')

# Read binary numerical data from a .dat file (assumes float32)
data = np.fromfile('your_file.dat', dtype=np.float32)

# For space, comma, or tab-separated data: Use numpy.loadtxt() or pandas.read_csv().
# For binary data: Use numpy.fromfile().
# For fixed-width formatted data: Use pandas.read_fwf().
# For large files: Use chunksize in pandas.read_csv() to process data in parts.