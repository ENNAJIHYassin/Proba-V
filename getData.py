from urllib.request import urlretrieve
from zipfile import ZipFile

print("downloading data... please wait")
urlretrieve('https://kelvins.esa.int/media/competitions/proba-v-super-resolution/probav_data.zip',
            filename='probav_data.zip')

print("extracting data... please wait")
ZipFile('probav_data.zip').extractall('probav_data/')
