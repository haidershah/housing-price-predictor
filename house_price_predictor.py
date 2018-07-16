import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

np.set_printoptions(suppress = True)

def format(number):
	return "{:,}".format(number)

# Get number of days between 2 dates
def diff_days(start_date, end_date):
    delta = end_date - start_date
    return delta.days

def clean_data(df):
	df.drop(df[should_remove_data(df)].index, inplace = True)

def should_remove_data(df):
	return (df.sqft < 500) | \
		(df.sqft > 10000) | \
		(df.lot < 500) | \
		(df.lot > 1000000) | \
		(df.bed == 0.0) | \
		(df.bed > 10) | \
		(df.bath == 0.0) | \
		(df.bath > 10) | \
		(df.last_sold_price < 100000) | \
		(df.last_sold_price > 10000000) | \
		(df.home_type == 'Miscellaneous') | \
		(df.home_type == 'Cooperative') | \
		(df.home_type == 'Duplex') | \
		(df.home_type == 'Mobile')

def add_feature_sold_days_ago(df):
	sold_days_ago = []

	for ind in df.index:
		last_sold_date = df['last_sold_date'][ind]
		sold_days_ago.append(diff_days(parse(last_sold_date), datetime.now()))

	df['sold_days_ago'] = sold_days_ago

def add_feature_age(df):
	age = []

	for year_built in np.array(df['year_built']):
		age.append(datetime.now().year - year_built)

	df['age'] = age

def remove_outliers(features, labels, percentage):
	lr = linear_model.LinearRegression()
	lr.fit(features, labels)
	pred = lr.predict(features)

	errors = []

	# calculate errors
	for index in range(0, labels.size):
		error = labels[index] - pred[index]
		error = error ** 2
		errors.append(error)
	errors_array = np.array(errors)

	num_outliers_to_remove = int(percentage * labels.size)

	# remove outliers
	for index in range(0, num_outliers_to_remove):
		arg_max_error = errors_array.argmax()
		features = np.delete(features, arg_max_error, axis = 0) # delete row
		labels = np.delete(labels, arg_max_error)
		errors_array = np.delete(errors_array, arg_max_error)

	return features, labels

def add_one_hot_encoding(df):
	housing_features_home_type = np.array(df['home_type'])

	# perform integer encoding
	label_encoder = LabelEncoder()
	home_type_integer_encoded = label_encoder.fit_transform(housing_features_home_type)

	# perform binary encoding
	onehot_encoder = OneHotEncoder(sparse = False)
	home_type_integer_encoded = home_type_integer_encoded.reshape(len(home_type_integer_encoded), 1)
	home_type_onehot_encoded = onehot_encoder.fit_transform(home_type_integer_encoded)

	# add features
	df['is_condominium'] = home_type_onehot_encoded[:, 0]
	df['is_multi_family'] = home_type_onehot_encoded[:, 1]
	df['is_single_family'] = home_type_onehot_encoded[:, 2]
	df['is_townhouse'] = home_type_onehot_encoded[:, 3]

# Load training data
df = pd.read_csv('data/dublin_housing_data.csv')

# preprocess data
df.dropna(inplace = True)
clean_data(df)
add_feature_sold_days_ago(df)
add_feature_age(df)
add_one_hot_encoding(df)

housing_feature_names = ['sqft', 'lot', 'bed', 'bath', 'age', 'dist_to_public_trans', 'sold_days_ago',
				 		'is_condominium', 'is_multi_family', 'is_single_family', 'is_townhouse']
housing_features = np.array(df[housing_feature_names])
housing_labels = np.array(df['last_sold_price'])

# Scale features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
housing_features = scaler.fit_transform(housing_features)

# Remove outliers
housing_features, housing_labels = remove_outliers(housing_features, housing_labels, percentage = 0.20)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
housing_features_train, housing_features_test, housing_labels_train, housing_labels_test = train_test_split(
	housing_features, housing_labels, test_size=0.10, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

print ('Total training data:', format(housing_labels_train.size))
print ('Total testing data:', format(housing_labels_test.size))

# Train the model using the training sets
regr.fit(housing_features_train, housing_labels_train)

# Make predictions using the testing set
housing_labels_pred = regr.predict(housing_features_test)

# Features by importance
coef = np.abs(np.array(regr.coef_))
coef_index_sort = np.argsort(coef)
feature_importance = []
for x in coef_index_sort:
	feature_importance.append(housing_feature_names[x])
feature_importance = feature_importance[::-1]
index = 1
print('\nFeatures by importance(most important first):\n')
for x in feature_importance:
	if x == 'sold_days_ago':
		continue

	print(str(index) + ". " + x)
	index = index + 1

# Calculate accuracy
from sklearn.metrics import r2_score
score = r2_score(housing_labels_test, housing_labels_pred)
accuracy = int(round(score * 100))
print ('\nAccuracy:', str(accuracy) + '%')

# Predict Rabbani Mansion's value
rabbani_mansion_sqft = 1920
rabbani_mansion_lot = 1920
rabbani_mansion_bed = 4
rabbani_mansion_bath = 3
rabbani_mansion_age = 58
rabbani_mansion_dist_to_public_trans = 0.729592997803104
rabbani_mansion_sold_days_ago = 0 # zero to get current value
rabbani_mansion_is_condominium = 0
rabbani_mansion_is_multi_family = 0
rabbani_mansion_is_single_family = 1
rabbani_mansion_is_townhouse = 0
rabbani_mansion_features = np.array([np.array([rabbani_mansion_sqft, rabbani_mansion_lot, 
												rabbani_mansion_bed, rabbani_mansion_bath,
												rabbani_mansion_age, 
												rabbani_mansion_dist_to_public_trans, 
												rabbani_mansion_sold_days_ago,
												rabbani_mansion_is_condominium, 
												rabbani_mansion_is_multi_family, 
												rabbani_mansion_is_single_family,
												rabbani_mansion_is_townhouse])])
rabbani_mansion_features_scaled = scaler.transform(rabbani_mansion_features)
rabbani_mansion_pred = regr.predict(rabbani_mansion_features_scaled).item(0)
rabbani_mansion_pred_round_int = int(round(rabbani_mansion_pred))
rabbani_mansion_formatted = format(rabbani_mansion_pred_round_int)
print('\n7545 Honey Ct, Dublin, CA')
print ("Sold in May 2017 for: $806,000")
print ("Current value: $" + str(rabbani_mansion_formatted))

# Plot outputs
housing_features_sqft = housing_features[:, 1]
plt.scatter(housing_features_sqft, housing_labels, color="blue", label="training data")
plt.scatter(housing_features_test[:, 0], housing_labels_test,  color='red', label="testing data")
plt.legend(loc=2)
plt.xlabel("Square Feet")
plt.ylabel("Price")
# plt.show()