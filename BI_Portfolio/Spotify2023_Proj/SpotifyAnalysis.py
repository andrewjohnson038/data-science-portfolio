# Link to Tableau Visual: https://public.tableau.com/app/profile/andrew.johnson1314/viz/Spotify2023_Charts_Statistics/Spotify2023StatsDashboard

# In this project, the priamry objective is to retrieve data through an API call and bring it into a BI tool (Tableau) to allow for further drill through and analysis of the data
# In this example, we are pulling a 2023 dataset from kaggle on the top Spotify songs to simulate pulling from a public repository of data, doing some simple data wrangling/formatting, and preparing it for use in tableau (link at the top)

# uncomment pip install if the libraries are not installed yet
# pip install pandas
# pip install zipfile
# pip install kaggle

# import pandas library (used to create a data frame and wrangle data)
import pandas as pd
# import zipfile library (used to extract the file from kaggle)
import zipfile
# import kaggle library (used to download the dataset programmatically from kaggle w/ kaggle API)
import kaggle
# import numbpy library (used to remove whitespace in this workflow)
import numpy as np


# Run this command if you don't want other users of your system to have read access to your credentials
# chmod 600 /Users/andrewjohnson/.kaggle/kaggle.json

# import the Kaggle api
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate the Kaggle API
api = KaggleApi()
api.authenticate()

# Download dataset with kaggle (API GitHub directions on how to use https://github.com/Kaggle/kaggle-api)
dataset = 'nelgiriyewithana/top-spotify-songs-2023'
kaggle.api.dataset_download_files(dataset, path='./Spotify2023_Proj')

# Unzip the file from the download zip file ['r' leaves the backslashes in the string]
zipfile_name = './top-spotify-songs-2023.zip'
with zipfile.ZipFile(zipfile_name, 'r') as file:
    file.extractall()

# Set path name for csvfile
path = "./spotify-2023.csv"

# Read the csv file as a pandas df
# The file being used is not encoded in UTF-8, which is needed to have the csv be encoded to UTF-8. It is encoded in ISO-8859-1, so we will need to change that
Spotify2023 = pd.read_csv(path, encoding='ISO-8859-1') #csv is downloaded in ISO-8859-1 format
Spotify2023.to_csv(path, encoding='utf-8', index=False)


# Get basic summary of the data
Spotify2023.info()

# Get a count of each unique value in the song key column
print(Spotify2023.key.value_counts())

# Change the column name of "artist(s)_name column to artist_name(s)"
Spotify2023 = Spotify2023.rename(columns={'artist(s)_name':'artist_name(s)'})
Spotify2023.info()

# Create a dictionary to convert month IDs to month name
months_dictionary = {
    '1' : 'January',
    '2' : 'February',
    '3' : 'March',
    '4' : 'April',
    '5' : 'May',
    '6' : 'June',
    '7' : 'July',
    '8' : 'August',
    '9' : 'September',
    '10': 'October',
    '11' : 'November',
    '12' : 'December'}
# Change values to a string
Spotify2023.released_month = Spotify2023.released_month.astype('str')

# Map the month IDs to dictionary containing the month names
Spotify2023.released_month = Spotify2023.released_month.map(months_dictionary)

# View final dataset to check data cleansing done
print(Spotify2023.head(50).to_string()) #limit to only viewing 50 rows

# When checking the dataset, I noticed there are some null values in the "key" column. Let's fill these as we will be doing analysis on most popular keys in our visual.
# For our visual, we will only be using songs released in 2023; let's reduce the sample size to songs released in 2023
Tracks_2023 = Spotify2023[Spotify2023['released_year'] == 2023]

# Let's now isolate the null values in the "key" column to see which songs we will need to fill nulls
Key_Nulls_2023 = Tracks_2023['key'].isnull()
print(Tracks_2023[Key_Nulls_2023].to_string())

# Print the count of rows to see count of tracks with a null key value
print("Number of Track w/ a Null Value:", len(Tracks_2023[Key_Nulls_2023])) #len function counts the values

# There are only 16 tracks released in 2023 that have a null key value. Considering the # of nulls is so low, the easiest method would be to fill the nulls manually. Let's do that in this next step:
# Fill null values manually in the 'key' column for specified rows with different values
Spotify2023.loc[12, 'key'] = 'Am'
Spotify2023.loc[17, 'key'] = 'C#'
Spotify2023.loc[35, 'key'] = 'C#'
Spotify2023.loc[44, 'key'] = 'C#'
Spotify2023.loc[58, 'key'] = 'Cm'
Spotify2023.loc[135, 'key'] = 'C#'
Spotify2023.loc[144, 'key'] = 'C#'
Spotify2023.loc[151, 'key'] = 'C#'
Spotify2023.loc[181, 'key'] = 'C#'
Spotify2023.loc[259, 'key'] = 'C#'
Spotify2023.loc[287, 'key'] = 'Cm'
Spotify2023.loc[332, 'key'] = 'C#'
Spotify2023.loc[373, 'key'] = 'C#'
Spotify2023.loc[379, 'key'] = 'C#'
Spotify2023.loc[381, 'key'] = 'C#'
Spotify2023.loc[385, 'key'] = 'C#'

# Let's check the nulls have now been filled by checking the song "flowers" was filled with 'Am' (row 12) and re-printing null count
print(Spotify2023.at[12, 'key']) # checking song key was filled

# Success: We can now write the df to excel for visualization

# Write the final df to an excel sheet for visualization
Spotify2023.to_excel('Spotify2023_TableauProj.xlsx', sheet_name='Spotify2023_stats')

# Section 1:
# For the artists tab in Tableau, I want to create a new tab in the excel sheet (new table) where songs with multiple artists are split into their own rows so that we can rank individual artists by streams
# To do this, we will need to split the delimiter (delimted by a comma) and add a new tab to the excel sheet for our new table

# Split the artists column based on comma delimiter
Spotify2023['artist_name(s)'] = Spotify2023['artist_name(s)'].str.split(', ')

# Explode the list of artists
Spotify2023_artists = Spotify2023.explode('artist_name(s)')

# lets drop the fields we don't need. Let's drop all fields except artist name and streams.
Spotify2023_artists = Spotify2023_artists.drop(columns=[
    'artist_count',
    'track_name',
    'released_year',
    'released_month',
    'released_month',
    'released_day',
    'in_spotify_playlists',
    'in_spotify_charts',
    'in_apple_playlists',
    'in_apple_charts',
    'in_deezer_playlists',
    'in_shazam_charts',
    'in_deezer_charts',
    'bpm',
    'key',
    'mode',
    'danceability_%',
    'valence_%',
    'energy_%',
    'acousticness_%',
    'instrumentalness_%',
    'liveness_%',
    'speechiness_%'])

# Section 3: Check that the new df delimited and columns dropped properly.
print(Spotify2023_artists.head(50).to_string()) # This limits to viewing only 50 rows of the df

# To view the whole output, use: print(Spotify2023_artists.to_string())
# Looks good. On to next step.

# Section 4: Deduplicate Artists & Sum the Streams

# First, let's check that 'streams' was brought in as an integer and not an object:

# Check if 'streams' is an integer:
if Spotify2023_artists['streams'].dtype == 'int64':
    print("'streams' is an integer.")
else:
    print("'streams' is not an integer.")

# 'streams' is not an integer; Let's check if it came in as a string:

# Check if 'field_name' is a string
if Spotify2023_artists['streams'].dtype == 'object':
    print("'streams' is a string.")
else:
    print("'streams' is not a string.")

# Can also check using the below
print(Spotify2023_artists['streams'].info())

# Data Type came back as an object; Will need to change the dtype to an integer to run a sum function
    # Spotify2023_artists['streams'] = Spotify2023_artists['streams'].astype(int)
# When running the above, we are getting an error that there are non-numerical values in the list; Let's remove these

# Replace non-numeric values with NaN
Spotify2023_artists['streams'] = pd.to_numeric(Spotify2023_artists['streams'], errors='coerce')

# Drop rows with NaN values
Spotify2023_artists = Spotify2023_artists.dropna()

# Try again:
Spotify2023_artists['streams'] = Spotify2023_artists['streams'].astype(int)

# No Error; Success: Let's check now if the object was converted to integer:
Spotify2023_artists.info()

# Mission Success: Let's now de-dup the artists, sum up the streams by artist, and add a rank column to the df:

# Section 5: Add a column for "Rank" to rank each artists by total streams (Note: If a artist featured on a song, those streams will be included in thier total stream count)

# Group by artists and sum up the streams
Spotify2023_artists = Spotify2023_artists.groupby('artist_name(s)')['streams'].sum().reset_index()

# Rank the artists by their total streams & add 'artist_rank' column
Spotify2023_artists['artist_rank'] = Spotify2023_artists['streams'].rank(ascending=False).astype(int)
    #Notes:
    #We use groupby() to group the DataFrame by the 'artist_names(s)' column.
    #We then use the sum() function to sum up the 'streams' for each artist.
    #Next, we use the rank() function to rank the artists based on their total streams, setting ascending=False to rank in descending order.
    #Finally, we add the resulting rank as a new column 'artist_rank' to the DataFrame.

# Let's drop any null values from artist name & review our changes:
Spotify2023_artists = Spotify2023_artists.replace(r'^\s*$', np.nan, regex=True) # Replace any whitespace with NaN
Spotify2023_artists = Spotify2023_artists.dropna(subset=['artist_name(s)'])  # Drop the Nulls
print(Spotify2023_artists.head(50).to_string())  # review changes


# Section 6: Let's add "a." to each column name to seperate naming conventions from original table
Spotify2023_artists = Spotify2023_artists.rename(columns={
    'artist_name(s)' : 'a.artist_name',
    'artist_rank' : 'a.artist_rank',
    'streams' : 'a.streams'})
print(Spotify2023_artists)  # review changes

# Let's reduce this table to the top 100 artists for our visual in Tableau
Spotify2023_artists = Spotify2023_artists.sort_values(by='a.artist_rank') # sort by artist rank
Spotify2023_artists = Spotify2023_artists.head(100) # head = Select the first 100 rows

# Section 7: Write the df as a new tab in the excel sheet

# Create a Pandas Excel writer using XlsxWriter as the engine
with pd.ExcelWriter('Spotify2023_TableauProj.xlsx', engine='openpyxl', mode='a') as writer:

# Description for the code above^: This is creating a context manager using the with statement to open an Excel file for writing or appending data using Pandas
    # mode='a' stands for "append": This indicates we are appending the excel sheet
    # engine='openpyxl': This specifies the engine to use for writing to the Excel file. openpyxl is one of the supported engines by Pandas for writing Excel files.

# Write the DataFrame with exploded artists to a new tab in the Excel file
    Spotify2023_artists.to_excel(writer, sheet_name='Spotify2023_artists', index=False)

# Note: The index=False parameter is used to indicate that you do not want to include the index column when writing the DataFrame to the output file.
