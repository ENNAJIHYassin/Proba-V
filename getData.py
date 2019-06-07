from urllib.request import urlretrieve
from zipfile import ZipFile
import os

print("downloading data... please wait")
urlretrieve('https://kelvins.esa.int/media/competitions/proba-v-super-resolution/probav_data.zip',
            filename='probav_data.zip')

print("extracting data... please wait")
ZipFile('probav_data.zip').extractall('probav_data/')

# Create directory
dirName = 'output'
try:
    os.mkdir(dirName)
    print(dirName," Directory Created!") 
except FileExistsError:
    print(dirName," Directory already exists!")
