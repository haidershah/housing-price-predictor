from pybart.api import BART

import pandas as pd

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
df.to_csv('data/bart.csv')
