# Dublin Housing Price Predictor

Predicts housing prices for Dublin, CA

## Deployment

$ python house_price_predictor.py

## Data

Here's how I gathered the data:

1) From OpenAddresses I retrieved addresses of houses in Dublin, CA.
2) I made an Api call to Zillow to get sqft, lot, bed, bath, last sold date, and last sold price info.
3) I made an Api call to Bart to get a list of public transportation stations. Then for each house I calculated the distance between it and the closets public transportation station.

## Preprocessing

Here are the steps I took to preprocess my data:

### Data Cleaning

For houses that don't have information available on Zillow's Api, I didn't add them to the training set. For example if there's a house with missing sqft or bed etc. I ignore it.

### One Hot Encoding

Home Type is a Categorical Feature in the dataset that comprises of values: Condominium, MultiFamily, SingleFamily, and Townhouse. I've used One Hot Encoding to create 4 new features: is_condominium, is_multi_family, is_single_family, and is_townhouse.

### Outlier Removal

I made a scatter to visualize the dataset. It turned out that I had a few outliers. When running the model with those outliers, I had a negative value for my Accuracy. After removing 10% of outliers from the dataset though, I had a marked improvement in my Accuracy.

### Feature Scaling

Number of bedrooms typically range from 1-5, while lot size are in the thousands so Feature Scaling was an obvious choice.

## Features

I'm using the following features for my model:

1. sqft - Square footage of constructed area
2. lot - Square footage of land
3. bed - Number of total bedrooms
4. bath - Number of total bathrooms
5. year_built - Year the house was built
6. sold_months_ago - Number of months from today when the house was last sold
7. dist_to_public_trans - Distance to the closest public transportation station
8. is_single_family - Whether the house is of type: SingleFamily
9. is_condominium - Whether the house is of type Condominium
10. is_townhouse - Whether the house is of type Townhouse
11. is_multi_family - Whether the house is of type Multi Family

## Label

last_sold_price - Last time the house was sold

## Features by importance

A list of all features by their importance (most importance first). This gives us a sense of which features play an important role in defining the price of a house.

1. sqft
2. is_multi_family
3. lot
4. dist_to_public_trans
5. is_single_family
6. bath
7. age
8. bed
9. is_condominium
10. is_townhouse

## Accuracy

91%

## Next Steps

Here are some ways to improve Accuracy of the model:

1. Add feature accessibility_to_freeway
2. Add feature elementary_school_rating
3. Add feature middle_school_rating
4. Add feature high_school_rating

## Tools

Machine Learning Library - scikit-learn <br />
Machine Learning Algorithm - Linear Regression

## Licenses

Address Data - https://openaddresses.io
<br>
Dublin Address Data - http://www.acgov.org/acdata/terms.htm
<br>
Zillow Api - https://www.zillow.com/howto/api/APIOverview.htm
<br>
Python Zillow Api Wrapper - https://pypi.org/project/pyzillow/
<br>
Python Bart Api Wrapper - https://pypi.org/project/pybart/
<br>
Python client for geocoding web services - https://pypi.org/project/geopy/