#In this project, the priamry objective is to retrieve data through an API call and bring it into a BI tool (Tableau) to allow for further drill through and analysis of the data
#In this example, we are pulling a 2023 dataset from kaggle on the top Spotify songs to simulate pulling from a public repository of data, doing some simple data wrangling/formatting, and preparing it for use in tableau (link at bottom)

# uncomment pip install if the libraries are not installed yet
# pip install pandas
# pip install zipfile
# pip install kaggle

# import pandas library
import pandas as pd
# import zipfile library (used to extract the file from kaggle)
import zipfile
# import kaggle library (used to download the dataset programmatically from kaggle w/ kaggle API)
import kaggle


# Run this command if you don't want other users of your system to have read access to your credentials
# chmod 600 /Users/andrewjohnson/.kaggle/kaggle.json

# import the Kaggle api
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate the Kaggle API
api = KaggleApi()
api.authenticate()

# download dataset with kaggle (API GitHub directions on how to use https://github.com/Kaggle/kaggle-api)
dataset = 'nelgiriyewithana/top-spotify-songs-2023'
kaggle.api.dataset_download_files(dataset, path='/Users/andrewjohnson/IdeaProjects/DataSciencePortfolio/Spotify_Proj')

# unzip the file from the download zip file ['r' leaves the backslashes in the string]
zipfile_name = '/Users/andrewjohnson/IdeaProjects/DataSciencePortfolio/Spotify_Proj/top-spotify-songs-2023.zip'
with zipfile.ZipFile(zipfile_name, 'r') as file:
    file.extractall()

# set path name for csvfile
path = "/Users/andrewjohnson/IdeaProjects/DataSciencePortfolio/Spotify_Proj/spotify-2023.csv"

# read the csv file as a pandas df
# the file being used is not encoded in UTF-8, which is needed to have the csv be encoded to UTF-8. It is encoded in ISO-8859-1, so we will need to change that
Spotify2023 = pd.read_csv(path, encoding='ISO-8859-1') #csv is downloaded in ISO-8859-1 format
Spotify2023.to_csv(path, encoding='utf-8', index=False)

# get basic summary of the data
Spotify2023.info()

# get a count of each unique value in the song key column
print(Spotify2023.key.value_counts())

# change the column name of "artist(s)_name column to artist_name(s)"
Spotify2023 = Spotify2023.rename(columns={'artist(s)_name':'artist_name(s)'})
Spotify2023.info()

# create a dictionary to convert month IDs to month name
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
# change values to a string
Spotify2023.released_month = Spotify2023.released_month.astype('str')
# map the month IDs to dictionary containing the month names
Spotify2023.released_month = Spotify2023.released_month.map(months_dictionary)

# view final dataset to check data cleansing done
#print(Spotify2023.to_string()) #view whole output
print(Spotify2023.head(50).to_string()) #limit to viewing 50 rows
# write the final df to an excel sheet for visualization
Spotify2023.to_excel('Spotify2023_TableauProj.xlsx', sheet_name='Spotify2023_stats')

# Link to Visual in Tableau:
# https://public.tableau.com/app/profile/andrew.johnson1314/viz/Spotify2023_Charts_Statistics/Spotify2023StatsDashboard

# for the artists tab in Tableau, I want to create a new tab in the excel sheet (new table) where songs with multiple artists are split into their own rows so that we can run rank individual artists by streams
# to do this, we will need to spliat the delimiter (delimted by a comma) and add a new tab to the excel sheet for our new table

# Step 1: Split the artists column based on comma delimiter
Spotify2023['artist_name(s)'] = Spotify2023['artist_name(s)'].str.split(', ')

# Step 2: Explode the list of artists
Spotify2023_artists = Spotify2023.explode('artist_name(s)')

#Step 2.1: lets drop the "artist_count" column from this df since it is not needed
Spotify2023_artists = Spotify2023_artists.drop(columns=['artist_count'])

# Step 3: add "a." to each column name to seperate naming conventions from original table
Spotify2023_artists = Spotify2023_artists.rename(columns={
    'artist_name(s)' : 'a.artist_name(s)',
    'track_name' : 'a.track_name',
    'released_year' : 'a.released_year',
    'released_month' : 'a.released_month',
    'released_day' : 'a.released_day',
    'in_spotify_playlists' : 'a.in_spotify_playlists',
    'in_spotify_charts' : 'a.in_spotify_charts',
    'streams' : 'a.streams',
    'in_apple_playlists' : 'a.in_apple_playlists',
    'in_apple_charts' : 'a.in_apple_charts',
    'in_deezer_playlists' : 'a.in_deezer_playlists',
    'in_deezer_charts' : 'a.in_deezer_charts',
    'in_shazam_charts' : 'a.in_shazam_charts',
    'bpm' : 'a.bpm',
    'key' : 'a.key',
    'mode' : 'a.mode',
    'danceability_%' : 'a.danceability_%',
    'valence_% ' : 'a.valence_% ',
    'energy_%' : 'a.energy_%',
    'acousticness_%' : 'a.acousticness_%',
    'instrumentalness_%' : 'a.instrumentalness_%',
    'liveness_%' : 'a.liveness_%',
    'speechiness_%' : 'a.speechiness_%'})

# Step 4: Check that the new df delimited and appended properly
#print(Spotify2023_artists.to_string()) #view whole output
print(Spotify2023_artists.head(50).to_string()) #limit to viewing 50 rows
#looks good. on to next step

#now lets sum streams by each artist

# Step 5: Write the df as a new tab in the excel sheet
# Create a Pandas Excel writer using XlsxWriter as the engine
with pd.ExcelWriter('Spotify2023_TableauProj.xlsx', engine='openpyxl', mode='a') as writer:

# Description for the code above^: This is creating a context manager using the with statement to open an Excel file for writing or appending data using Pandas
    # mode='a' stands for "append": This indicates we are appending the excel sheet
    # engine='openpyxl': This specifies the engine to use for writing to the Excel file. openpyxl is one of the supported engines by Pandas for writing Excel files.

# Write the DataFrame with exploded artists to a new tab in the Excel file
    Spotify2023_artists.to_excel(writer, sheet_name='Spotify2023_artists', index=False)
# Note: The index=False parameter is used to indicate that you do not want to include the index column when writing the DataFrame to the output file.
