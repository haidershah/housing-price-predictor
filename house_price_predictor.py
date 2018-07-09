import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from dateutil.parser import parse

def diff_month(start_date, end_date):
    return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month

def clean_data(df):
	houses_to_remove = []

	for ind in df.index:
		sqft = df['sqft'][ind]
		bed = df['bed'][ind]
		bath = df['bath'][ind]
		lot = df['lot'][ind]
		year_built = df['year_built'][ind]
		last_sold_date = df['last_sold_date'][ind]
		last_sold_price = df['last_sold_price'][ind]

    	# for properties with missing info, don't add them to training data
		if math.isnan(sqft) or math.isnan(bed) or (bed == 0.0) or math.isnan(bath) or (bath == 0.0) or math.isnan(lot) or math.isnan(year_built) or (not last_sold_date) or math.isnan(last_sold_price):
			houses_to_remove.append(ind)

	return df.drop(df.index[houses_to_remove])

def add_sold_months_ago_column(df):
	df['sold_months_ago'] = 0
	for ind in df.index:
		last_sold_date = df['last_sold_date'][ind]
		df['sold_months_ago'][ind] = diff_month(parse(last_sold_date), datetime.now())

# Load training data
df = pd.read_csv('data/dublin_housing_data.csv')
df = clean_data(df)
add_sold_months_ago_column(df)

housing_feature_names = ['sqft', 'lot', 'bed', 'bath', 'year_built', 'sold_months_ago']
housing_features = np.array(df[housing_feature_names])
housing_labels = np.array(df['last_sold_price'])

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
housing_features_train, housing_features_test, housing_labels_train, housing_labels_test = train_test_split(housing_features, housing_labels, test_size=0.10, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

print ('Total training data:', housing_labels_train.size)
print ('Total testing data:', housing_labels_test.size)

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
# from sklearn.metrics import r2_score
# score = r2_score(housing_labels_test, housing_labels_pred)
# accuracy = int(round(score * 100))
# print ('\nAccuracy:', str(accuracy) + '%')

# Predict Rabbani Mansion's value
rabbani_mansion_features = np.array([np.array([1920, 7405, 4, 3, 1960, 0])])
rabbani_mansion_pred = regr.predict(rabbani_mansion_features).item(0)
rabbani_mansion_pred_round_int = int(round(rabbani_mansion_pred))
rabbani_mansion_formatted = "{:,}".format(rabbani_mansion_pred_round_int)
print ("\nRabbani Mansion's purchase price: $806,000")
print ("Rabbani Mansion's current value: $" + str(rabbani_mansion_formatted))
