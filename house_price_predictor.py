import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load training data
import pandas as pd
df = pd.read_csv('housing_data.csv')
housing_features = np.array(df[['sqft', 'lot', 'bed', 'bath', 'sold_months_ago']])
housing_labels = np.array(df['price'])

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
housing_features_train, housing_features_test, housing_labels_train, housing_labels_test = train_test_split(housing_features, housing_labels, test_size=0.30, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(housing_features_train, housing_labels_train)

# Make predictions using the testing set
housing_labels_pred = regr.predict(housing_features_test)

# Calculate accuracy
from sklearn.metrics import r2_score
score = r2_score(housing_labels_test, housing_labels_pred)
accuracy = int(round(score * 100))
print ('Accuracy:', str(accuracy) + '%')

# Predict Rabbani Mansion's value
rabbani_mansion_features = np.array([np.array([1920, 7405, 4, 3, 0])])
rabbani_mansion_pred = regr.predict(rabbani_mansion_features).item(0)
rabbani_mansion_pred_round_int = int(round(rabbani_mansion_pred))
rabbani_mansion_formatted = "{:,}".format(rabbani_mansion_pred_round_int)
print ("Rabbani Mansion's purchase price: $806,000")
print ("Rabbani Mansion's current value: $" + str(rabbani_mansion_formatted))

# Plot outputs
# plt.scatter(housing_features_train, housing_labels_train, color="blue", label="train data")
# plt.scatter(housing_features_test, housing_labels_test,  color='red', label="test data")
# plt.plot(housing_features_test, housing_labels_pred, color='black')
# plt.legend(loc=2)
# plt.xlabel("Square Feet")
# plt.ylabel("Price")
# plt.show()
