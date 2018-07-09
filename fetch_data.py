from datetime import date
from dateutil.parser import parse
from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults
from pathlib import Path

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

def get_house_info(address, unit, zipcode):
	house_info = []
	try:
		# if unit value exists, don't make zillow api call
		if does_unit_exist(unit):
			house_info.extend(('', address, '', '', '', '', '', '', '', '', ''))
		elif math.isnan(zipcode): # zipcode info not available
			house_info.extend(('', address, '', '', '', '', '', '', '', '', ''))
		else:
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

			house_info.extend((zillow_id, address, zipcode, sqft, lot, bed, bath,
								home_type, year_built, last_sold_date, last_sold_price))
	except:
		print(address + ': exception')
		return []

	return house_info

zillow_data = ZillowWrapper('X1-ZWz18haqkyt2q3_ac8os')

addresses_df = pd.read_csv('data/dublin_addresses.csv')
addresses_array = np.array(addresses_df[['NUMBER', 'STREET', 'UNIT', 'CITY', 'POSTCODE']])

housing_data_file_name = 'data/dublin_housing_data.csv'
house_address_set = set()
housing_data_list = []

if Path(housing_data_file_name).is_file():
	print('file exists')
	housing_data_df = pd.read_csv(housing_data_file_name)
	house_address_array = np.array(housing_data_df['address'])
	house_address_set = set(house_address_array.flat)

	house_data_array = np.array(housing_data_df[['zillow_id', 'address', 'zipcode', 'sqft', 
												'lot', 'bed', 'bath', 'home_type', 'year_built',
												'last_sold_date', 'last_sold_price']])
	housing_data_list = house_data_array.tolist()

num_properties_downloaded = 0

deep_search_response = zillow_data.get_deep_search_results('4855 Swinford ct, dublin, ca', '94568')
result = GetDeepSearchResults(deep_search_response)
print (result.home_size)

for address_array in addresses_array:

	if num_properties_downloaded == 500:
		break

	number = address_array[0]
	street = address_array[1]
	unit = address_array[2]
	city = address_array[3]
	state = 'CA'
	zipcode = address_array[4]

	address = get_full_address(number, street, unit, city, state)

	# if this property has already been downloaded, skip it
	if address in house_address_set:
		print(address + ': already downloaded')
		continue

	house_info = get_house_info(address, unit, zipcode)

	if house_info: # if we have info for this house
		num_properties_downloaded = num_properties_downloaded + 1
		print (str(num_properties_downloaded) + ' properties downloaded.')
		housing_data_list.append(house_info)

columns = ['zillow_id', 'address', 'zipcode', 'sqft', 'lot', 'bed', 'bath', 
			'home_type', 'year_built', 'last_sold_date', 'last_sold_price']
housing_data_df = pd.DataFrame(data = housing_data_list, columns = columns)
housing_data_df.to_csv('data/dublin_housing_data.csv')