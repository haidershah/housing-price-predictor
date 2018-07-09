# Dublin Housing Price Predictor

Predicts housing prices for Dublin, CA

## Deployment

$ python house_price_predictor.py

## Data

I first retrieved addresses of houses in Dublin, CA from OpenAddresses. Then I made Zillow Api calls to gather the missing info: sqft, bed, bath etc.

## Data Cleaning

For houses that don't have information available on Zillow's Api, I didn't add them to the training set. For example if there's a house with missing sqft or bed etc. I ignore it.

## Features

I'm using the following features for my model:

1. sqft - Square footage of constructed area
2. lot - Square footage of land
3. bed - Number of total bedrooms
4. bath - Number of total bathrooms
5. year_built - Year the house was built
6. sold_months_ago - Number of months from today when the house was last sold

## Label

last_sold_price - Last time the house was sold

## Features by importance

A list of all features by their importance (most importance first). This gives us a sense of which features play an important role in defining the price of a house.

1. bath
2. bed
3. year_built
4. sqft
5. lot
6. sold_months_ago

## Tools

Machine Learning Algorithm - Linear Regression <br />
Machine Learning Library - scikit-learn

## Licenses

Address Data - https://openaddresses.io
<br>
Dublin Address Data - http://www.acgov.org/acdata/terms.htm
<br>
Zillow Api - https://www.zillow.com/howto/api/APIOverview.htm
<br>
Python Zillow Api Wrapper - https://pypi.org/project/pyzillow/

### Note

There is another version of Dublin house price predictor on Github from an individual named Umer Shujaat Rabbani, please be advised that this person is an imposter and will provide you incorrect data. This is the real Dublin house price predictor.