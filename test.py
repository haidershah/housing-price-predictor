from datetime import date
from dateutil.parser import parse
from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults

import pandas as pd
import numpy as np
import math

def get_full_address(number, street, city, state):
	return number + ' ' + street + ', ' + city + ', ' + state

def get_house_info(address, zipcode):
	try:
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

		# for properties with missing info, don't add them to training data
		if (sqft is None) or (int(bed) == 0) or (lot is None) or (year_built is None) or (last_sold_date is None) or (last_sold_price is None):
			return []

		house_info = []
		house_info.extend((zillow_id, address, zipcode, sqft, lot, bed, bath,
							home_type, year_built, last_sold_date, last_sold_price))
		housing_data_list.append(house_info)
	except:
		return []

zillow_data = ZillowWrapper('X1-ZWz18hg7w22k97_6frtk')

addresses_df = pd.read_csv('data/dublin_addresses.csv')
addresses_array = np.array(addresses_df[['NUMBER', 'STREET', 'CITY', 'POSTCODE']])

housing_data_df = pd.read_csv('data/dublin_housing_data.csv')
house_address_array = np.array(housing_data_df['address'])
house_address_set = set(house_address_array.flat)

house_data_array = np.array(housing_data_df[['zillow_id', 'address', 'zipcode', 'sqft', 
												'lot', 'bed', 'bath', 'home_type', 'year_built',
												'last_sold_date', 'last_sold_price']])

index = 0
housing_data_list = house_data_array.tolist()

for address_array in addresses_array:

	index = index + 1
	print (str(index) + ' properties downloaded.')

	if index == 500:
		break

	number = address_array[0]
	street = address_array[1]
	city = address_array[2]
	state = 'CA'
	zipcode = address_array[3]

	address = get_full_address(number, street, city, state)
	zipcode = str(int(zipcode))

	#if this property has already been downloaded, skip it
	if address in house_address_set:
		continue

	house_info = get_house_info(address, zipcode)

	if house_info: # if we have info for this house
		housing_data_list.append(house_info)

columns = ['zillow_id', 'address', 'zipcode', 'sqft', 'lot', 'bed', 'bath', 
			'home_type', 'year_built', 'last_sold_date', 'last_sold_price']
housing_data_df = pd.DataFrame(data = housing_data_list, columns = columns)
housing_data_df.to_csv('data/dublin_housing_data.csv')