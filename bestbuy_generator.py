import pandas as pd
import numpy as np

# Define the start and end date
start_date = np.datetime64('2017-05-28')
end_date = np.datetime64('2022-07-31')

# Create the date range
date_range = np.arange(start_date, end_date+np.timedelta64(1, 'D'), dtype='datetime64[D]')

# Create the id range
id_range = np.arange(1, 576)

# Create an empty dataframe
table = pd.DataFrame()

# Create a date column
table['date'] = np.repeat(date_range, len(id_range))

# Create an id column
table['id'] = np.tile(id_range, len(date_range))

# Export the table as a csv
table.to_csv("table.csv", index=False)
