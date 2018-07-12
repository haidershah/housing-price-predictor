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

def diff_month(start_date, end_date):
    return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month

def clean_data(df):
	houses_to_remove = []

	for ind in df.index:

    	# for properties with missing info, don't add them to training data
		if is_data_missing(ind, df):
			houses_to_remove.append(ind)

	df.drop(df.index[houses_to_remove], inplace = True)

def is_data_missing(index, df):
	sqft = df['sqft'][index]
	lot = df['lot'][index]
	bed = df['bed'][index]
	bath = df['bath'][index]
	home_type = df['home_type'][index]
	year_built = df['year_built'][index]
	last_sold_date = df['last_sold_date'][index]
	last_sold_price = df['last_sold_price'][index]

	return math.isnan(sqft) or \
			math.isnan(lot) or \
			math.isnan(bed) or \
			(bed == 0.0) or \
			math.isnan(bath) or \
			(bath == 0.0) or \
			pd.isnull(home_type) or \
			math.isnan(year_built) or \
			(not last_sold_date) or \
			math.isnan(last_sold_price) or \
			home_type == 'Miscellaneous' or \
			home_type == 'Cooperative'

def add_sold_months_ago_column(df):
	sold_months_ago = []

	for ind in df.index:
		last_sold_date = df['last_sold_date'][ind]
		sold_months_ago.append(diff_month(parse(last_sold_date), datetime.now()))

	df['sold_months_ago'] = sold_months_ago

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
		features = np.delete(features, arg_max_error, axis=0) # delete row
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
clean_data(df)
add_sold_months_ago_column(df)
add_one_hot_encoding(df)

housing_feature_names = ['sqft', 'lot', 'bed', 'bath', 'year_built', 'sold_months_ago',
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
coef = np.array(regr.coef_)
coef_index_sort = np.argsort(coef)
feature_importance = []
for x in coef_index_sort:
	feature_importance.append(housing_feature_names[x])
feature_importance = feature_importance[::-1]
print('\nFeatures by importance(most important first):')
for x in feature_importance:
	print(x)

# Calculate accuracy
from sklearn.metrics import r2_score
score = r2_score(housing_labels_test, housing_labels_pred)
accuracy = int(round(score * 100))
print ('\nAccuracy:', str(accuracy) + '%')

# Predict Rabbani Mansion's value
rabbani_mansion_features = np.array([np.array([1920, 7405, 4, 3, 1960, 0, 0, 0, 1, 0])])
rabbani_mansion_features_scaled = scaler.transform(rabbani_mansion_features)
rabbani_mansion_pred = regr.predict(rabbani_mansion_features_scaled).item(0)
rabbani_mansion_pred_round_int = int(round(rabbani_mansion_pred))
rabbani_mansion_formatted = format(rabbani_mansion_pred_round_int)
print ("\nRabbani Mansion's purchase price: $806,000")
print ("Rabbani Mansion's current value: $" + str(rabbani_mansion_formatted))

# Plot outputs
housing_features_sqft = housing_features[:, 1]
plt.scatter(housing_features_sqft, housing_labels, color="blue", label="training data")
plt.scatter(housing_features_test[:, 0], housing_labels_test,  color='red', label="testing data")
plt.legend(loc=2)
plt.xlabel("Square Feet")
plt.ylabel("Price")
# plt.show()