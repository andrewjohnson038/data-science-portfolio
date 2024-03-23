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
print(Spotify2023.to_string())
# write the final df to an excel sheet for visualization
Spotify2023.to_excel('Spotify2023_TableauProj.xlsx', sheet_name='Spotify2023_stats')

# Link to Visual in Tableau:
# https://public.tableau.com/app/profile/andrew.johnson1314/viz/Spotify2023_Charts_Statistics/Spotify2023StatsDashboard
