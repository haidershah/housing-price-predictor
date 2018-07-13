from datetime import date
from dateutil.parser import parse
from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults
from pathlib import Path
from geopy import distance

import pandas as pd
import numpy as np
import math

def does_unit_exist(unit):
	return isinstance(unit, str) and unit

def get_full_address(number, street, unit, city, state):
	if does_unit_exist(unit):
		return number + ' ' + street + ' ' + unit + ', ' + city + ', ' + state
	else:
		return number + ' ' + street + ', ' + city + ', ' + state

def get_house_info(address, unit, zipcode, latitude, longitude):
	house_info = []
	try:
		# if unit value exists, do not make zillow api call
		if does_unit_exist(unit):
			house_info.extend(('', address, '', '', '', '', '', '', '', '', '', '', ''))
		elif math.isnan(zipcode): # if zipcode info isn't available, do not make zillow api call
			house_info.extend(('', address, '', '', '', '', '', '', '', '', '', '', ''))
		else: # make zillow api call
			zipcode = str(int(zipcode))
			deep_search_response = zillow_data.get_deep_search_results(address, zipcode)
			result = GetDeepSearchResults(deep_search_response)

			zillow_id = result.zillow_id
			home_type = result.home_type
			home_detail_link = result.home_detail_link
			graph_data_link = result.graph_data_link
			year_built = result.year_built
			sqft = result.home_size
			lot = result.property_size
			bed = result.bedrooms
			bath = result.bathrooms
			last_sold_date = result.last_sold_date
			last_sold_price = result.last_sold_price

			house_info.extend((zillow_id, address, zipcode, latitude, longitude, sqft, lot, 
								bed, bath, home_type, year_built, last_sold_date, last_sold_price))
	except:
		print(address + ': exception')
		return []

	return house_info

def fetch_bart_data(bart_data_file_name):
	# bart data already downloaded
	if Path(bart_data_file_name).is_file():
		return

	from pybart.api import BART

	bart = BART(json_format = True)
	stations = bart.stn.stns()['stations']['station']

	# this station had to be added manually as it wasn't part of the api
	east_dublin_station = {}
	east_dublin_station['name'] = 'East Dublin'
	east_dublin_station['gtfs_latitude'] = '37.703178'
	east_dublin_station['gtfs_longitude'] = '-121.899423'
	east_dublin_station['city'] = 'Dublin'
	east_dublin_station['county'] = 'alameda'
	east_dublin_station['state'] = 'CA'
	east_dublin_station['zipcode'] = '94568'

	# this station had to be added manually as it wasn't part of the api
	west_pleasanton_station = {}
	west_pleasanton_station['name'] = 'West Pleasanton'
	west_pleasanton_station['gtfs_latitude'] = '37.698888'
	west_pleasanton_station['gtfs_longitude'] = '-121.928104'
	west_pleasanton_station['city'] = 'Pleasanton'
	west_pleasanton_station['county'] = 'alameda'
	west_pleasanton_station['state'] = 'CA'
	west_pleasanton_station['zipcode'] = '94588'

	stations.append(east_dublin_station)
	stations.append(west_pleasanton_station)

	columns = ['name', 'abbr', 'gtfs_latitude', 'gtfs_longitude', 'address', 'city', 'county', 'state', 'zipcode']
	df = pd.DataFrame(data = stations, columns = columns)
	df.to_csv(bart_data_file_name)

# Get Bart stations for zipcode
def get_bart_stations(bart_data_array, zipcode):
	return bart_data_array[bart_data_array[:, 0] == zipcode]

# Get distance of house to closest public transportation
def get_distance_to_public_trans(bart_data_array, zipcode, house_latitude, house_longitude):
	dist_to_public_trans = ''
	bart_stations = get_bart_stations(bart_data_array, zipcode)

	dist_to_public_trans_list = []
	for station in bart_stations:

		house_coordinates = (house_latitude, house_longitude)
		station_coordinates = (station[1], station[2])

		dist_to_public_trans_list.append(distance.distance(house_coordinates, station_coordinates).miles)

		if dist_to_public_trans_list:
			dist_to_public_trans = np.min(np.array(dist_to_public_trans_list))

	return dist_to_public_trans

bart_data_file_name = 'data/bart.csv'
fetch_bart_data(bart_data_file_name)

bart_data_df = pd.read_csv(bart_data_file_name)
bart_data_array = np.array(bart_data_df[['zipcode', 'gtfs_latitude', 'gtfs_longitude']])

# get zillow api key
with open("./bin/config/zillow_key_3.conf", 'r') as f:
    key = f.readline().replace("\n", "")

zillow_data = ZillowWrapper(key)

addresses_df = pd.read_csv('data/dublin_addresses.csv')
addresses_array = np.array(addresses_df[['NUMBER', 'STREET', 'UNIT', 'CITY', 'POSTCODE', 'LAT', 'LON']])

housing_data_file_name = 'data/dublin_housing_data.csv'
housing_data_df_columns = ['zillow_id', 'address', 'zipcode', 'latitude', 'longitude', 
							'sqft', 'lot', 'bed', 'bath', 'home_type', 'year_built', 
							'last_sold_date', 'last_sold_price', 'dist_to_public_trans']
house_address_set = set()
housing_data_list = []

if Path(housing_data_file_name).is_file():
	print('file exists')
	housing_data_df = pd.read_csv(housing_data_file_name)
	house_address_array = np.array(housing_data_df['address'])
	house_address_set = set(house_address_array.flat)

	house_data_array = np.array(housing_data_df[housing_data_df_columns])
	housing_data_list = house_data_array.tolist()

num_properties_downloaded = 0

deep_search_response = zillow_data.get_deep_search_results('4855 Swinford ct, dublin, ca', '94568')
result = GetDeepSearchResults(deep_search_response)
print (result.home_size)

for address_array in addresses_array:

	if num_properties_downloaded == 1000:
		break

	number = address_array[0]
	street = address_array[1]
	unit = address_array[2]
	city = address_array[3]
	state = 'CA'
	zipcode = address_array[4]
	latitude = address_array[5]
	longitude = address_array[6]

	address = get_full_address(number, street, unit, city, state)

	# if info for this house has already been downloaded, skip it
	if address in house_address_set:
		print(address + ': already downloaded')
		continue

	house_info = get_house_info(address, unit, zipcode, latitude, longitude)

	# if info is available for this house
	if house_info:
		dist_to_public_trans = get_distance_to_public_trans(bart_data_array, zipcode, latitude, longitude)
		house_info.append(dist_to_public_trans)

		num_properties_downloaded = num_properties_downloaded + 1
		print (str(num_properties_downloaded) + ' properties downloaded.')
		housing_data_list.append(house_info)

housing_data_df = pd.DataFrame(data = housing_data_list, columns = housing_data_df_columns)
housing_data_df.to_csv(housing_data_file_name)