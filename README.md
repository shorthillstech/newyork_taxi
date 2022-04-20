# newyork_taxi

# **NYC Taxi Trip Duration Prediction using Machine Learning - Step by Step Programming**

## Introduction

Did you know that in City Island and Pelham Bay Park in the Bronx, and Great Kills and Great Kills Park in Staten Island green cab taxis are more popular than yellow or for-hire cab taxis?

Did you know that you end up spending more time traveling (~12 minutes) on average in for-hire cab taxis as compared to yellow cab taxis?

You must know that Manhattan (Upper East Side North being the most prominent) and JFK Airport are the busiest areas for yellow cab taxis but did you know that Jackson Heights and Astoria in Queens and Stapleton in Staten Island are the most sort after by for-hire cabs?

Welcome to the blog post, today we explore the data provided by New York City Taxi and Limousine Commission(TLC) on their **[website](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)** using Pandas, Numpy, and Sklearn in Python.

![Screenshot 2022-02-28 at 10.00.17 AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6b50d4bc-73ca-44e1-8840-0ac32d71215d/Screenshot_2022-02-28_at_10.00.17_AM.png)

## **Step 1:** Download the data

You can also download the data using the AWS CLI using (Access No AWS account required):

```bash
aws s3 sync s3://nyc-tlc/ . --no-sign-request
```

The main purpose of this post is to develop a basic machine learning model, to predict the average travel time and fare for a given Pick up location, Drop location, Date, and Time. Every organization nowadays has to utilize its data properly to get an edge over its competitors and provide more value to customers. Machine learning has become a very important tool in making important business decisions and even people with no coding knowledge or domain experience can develop models with libraries such as **[data prep](https://github.com/sfu-db/dataprep)**, and **[sklearn](https://github.com/scikit-learn/scikit-learn)**. Scikit learn is one of the most powerful machine learning libraries out there. It is used by major corporations around the work such as **[J.P.Morgan](https://www.jpmorgan.com/)**, **[Spotify](https://www.spotify.com/)**, **[Evernote](https://evernote.com/)**, and many more.

## **Step 2:** Install and import sklearn

The package is named scikit-learn therefore you can do

```bash
pip install scikit-learn
```

however, inside your python file, you’d have to do

```python
import sklearn
```

To keep the post under 1000 words, we will look at only one month of data as the entire data would be 100 GB+ and more than what a single machine can handle (probably we will have a post in the future about distributed machine learning techniques).

[https://media-exp1.licdn.com/dms/image/C4E12AQGwPgok__Narg/article-inline_image-shrink_1500_2232/0/1635738229598?e=1651708800&v=beta&t=9evPKozLhLG8uaMtwJe2fEJ4r-xb-LjYkgBe2zaRB-Q](https://media-exp1.licdn.com/dms/image/C4E12AQGwPgok__Narg/article-inline_image-shrink_1500_2232/0/1635738229598?e=1651708800&v=beta&t=9evPKozLhLG8uaMtwJe2fEJ4r-xb-LjYkgBe2zaRB-Q)

## **Step 3:** Import the other libraries

```python
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
```

We have three different data sets, namely, green taxi, yellow taxi, and for-hire which would include Uber and Lyft. You can also look at the taxi zone lookup table to understand the data pickup and drop locations.

## **Step 4:** Load

We will import the libraries and download the data from the source mentioned above and load the data as a pandas dataframe:

```python
green_taxi_data = pd.read_csv('green_tripdata_2020-12.csv')
yellow_taxi_data = pd.read_csv('yellow_tripdata_2020-12.csv')
fhv_taxi_data = pd.read_csv('fhv_tripdata_2020-12.csv')

```

[https://media-exp1.licdn.com/dms/image/C4E12AQG5NSNYUcdqbw/article-inline_image-shrink_1500_2232/0/1635738317292?e=1651708800&v=beta&t=WABf_adbIkcgm5xVyQkTaNvpmiYAtXVEZtEbudNuv78](https://media-exp1.licdn.com/dms/image/C4E12AQG5NSNYUcdqbw/article-inline_image-shrink_1500_2232/0/1635738317292?e=1651708800&v=beta&t=WABf_adbIkcgm5xVyQkTaNvpmiYAtXVEZtEbudNuv78)

## **Step 5:** Explore

Next, we explore the data set and the fields available. In summary, we have fare and distance fields available for the green and yellow cabs but not for for-hire cabs. So as a fun exercise we would try to compute the total fares for-hire cabs (assuming they are similar to yellow cabs; which is not the case).

```python
green_taxi_data.info()
yellow_taxi_data.info()
fhv_taxi_data.info()
```

The output of the above is which you can skip as well, the main columns of interest are the PULocationID and DOLocationID.

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 83130 entries, 0 to 83129
Data columns (total 20 columns):
 #   Column                 Non-Null Count  Dtype
---  ------                 --------------  -----
 0   VendorID               46292 non-null  float64
 1   lpep_pickup_datetime   83130 non-null  object
 2   lpep_dropoff_datetime  83130 non-null  object
 3   store_and_fwd_flag     46292 non-null  object
 4   RatecodeID             46292 non-null  float64
 5   PULocationID           83130 non-null  int64
 6   DOLocationID           83130 non-null  int64
 7   passenger_count        46292 non-null  float64
 8   trip_distance          83130 non-null  float64
 9   fare_amount            83130 non-null  float64
 10  extra                  83130 non-null  float64
 11  mta_tax                83130 non-null  float64
 12  tip_amount             83130 non-null  float64
 13  tolls_amount           83130 non-null  float64
 14  ehail_fee              0 non-null      float64
 15  improvement_surcharge  83130 non-null  float64
 16  total_amount           83130 non-null  float64
 17  payment_type           46292 non-null  float64
 18  trip_type              46292 non-null  float64
 19  congestion_surcharge   46292 non-null  float64
dtypes: float64(15), int64(2), object(3)
memory usage: 12.7+ MB
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1461897 entries, 0 to 1461896
Data columns (total 18 columns):
 #   Column                 Non-Null Count    Dtype
---  ------                 --------------    -----
 0   VendorID               1362441 non-null  float64
 1   tpep_pickup_datetime   1461897 non-null  object
 2   tpep_dropoff_datetime  1461897 non-null  object
 3   passenger_count        1362441 non-null  float64
 4   trip_distance          1461897 non-null  float64
 5   RatecodeID             1362441 non-null  float64
 6   store_and_fwd_flag     1362441 non-null  object
 7   PULocationID           1461897 non-null  int64
 8   DOLocationID           1461897 non-null  int64
 9   payment_type           1362441 non-null  float64
 10  fare_amount            1461897 non-null  float64
 11  extra                  1461897 non-null  float64
 12  mta_tax                1461897 non-null  float64
 13  tip_amount             1461897 non-null  float64
 14  tolls_amount           1461897 non-null  float64
 15  improvement_surcharge  1461897 non-null  float64
 16  total_amount           1461897 non-null  float64
 17  congestion_surcharge   1461897 non-null  float64
dtypes: float64(13), int64(2), object(3)
memory usage: 200.8+ MB
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1151404 entries, 0 to 1151403
Data columns (total 7 columns):
 #   Column                  Non-Null Count    Dtype
---  ------                  --------------    -----
 0   dispatching_base_num    1151404 non-null  object
 1   pickup_datetime         1151404 non-null  object
 2   dropoff_datetime        1151404 non-null  object
 3   PULocationID            190903 non-null   float64
 4   DOLocationID            981028 non-null   float64
 5   SR_Flag                 0 non-null        float64
 6   Affiliated_base_number  1141618 non-null  object
dtypes: float64(3), object(4)
memory usage: 61.5+ MB
```

## **Step 6:** Plot

Next, we convert the data and develop basic plots:

```python
green_taxi_data['lpep_pickup_datetime'] =  pd.to_datetime(green_taxi_data['lpep_pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
green_taxi_data['lpep_dropoff_datetime'] =  pd.to_datetime(green_taxi_data['lpep_dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')
green_taxi_data['trip_duration'] = (green_taxi_data['lpep_dropoff_datetime'] - green_taxi_data['lpep_pickup_datetime']).dt.seconds
green_taxi_data['PULocationID'].fillna(-1, inplace = True)
green_taxi_data['DOLocationID'].fillna(-1, inplace = True)
yellow_taxi_data['tpep_pickup_datetime'] =  pd.to_datetime(yellow_taxi_data['tpep_pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
yellow_taxi_data['tpep_dropoff_datetime'] =  pd.to_datetime(yellow_taxi_data['tpep_dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')
yellow_taxi_data['trip_duration'] = (yellow_taxi_data['tpep_dropoff_datetime'] - yellow_taxi_data['tpep_pickup_datetime']).dt.seconds
yellow_taxi_data['PULocationID'].fillna(-1, inplace = True)
yellow_taxi_data['DOLocationID'].fillna(-1, inplace = True)
fhv_taxi_data['pickup_datetime'] =  pd.to_datetime(fhv_taxi_data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
fhv_taxi_data['dropoff_datetime'] =  pd.to_datetime(fhv_taxi_data['dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')
fhv_taxi_data['trip_duration'] = (fhv_taxi_data['dropoff_datetime'] - fhv_taxi_data['pickup_datetime']).dt.seconds
fhv_taxi_data['PULocationID'].fillna(-1, inplace = True)
fhv_taxi_data['DOLocationID'].fillna(-1, inplace = True)
```

We group the data at a daily level and plot the total duration:

```python
green_date_wise_sum = green_taxi_data.groupby(green_taxi_data['lpep_pickup_datetime'].dt.date).sum()[2:-1]
yellow_date_wise_sum = yellow_taxi_data.groupby(yellow_taxi_data['tpep_pickup_datetime'].dt.date).sum()[4:-8]
fhv_date_wise_sum = fhv_taxi_data.groupby(fhv_taxi_data['pickup_datetime'].dt.date).sum()
```

[https://media-exp1.licdn.com/dms/image/C4E12AQGIzSGSKMfQkw/article-inline_image-shrink_1500_2232/0/1635738502031?e=1651708800&v=beta&t=8vxv5bZitQemph_3Bx0bvysPFyogwbpv_oVDltc4KH0](https://media-exp1.licdn.com/dms/image/C4E12AQGIzSGSKMfQkw/article-inline_image-shrink_1500_2232/0/1635738502031?e=1651708800&v=beta&t=8vxv5bZitQemph_3Bx0bvysPFyogwbpv_oVDltc4KH0)

```
plt.plot(green_date_wise_sum.index,green_date_wise_sum['trip_duration'])
plt.plot(yellow_date_wise_sum.index,yellow_date_wise_sum['trip_duration'])
plt.plot(fhv_date_wise_sum.index,fhv_date_wise_sum['trip_duration'])
plt.legend(['Green','Yellow','For-Hire'])
plt.gcf().autofmt_xdate()
plt.title('Daily Duration in Seconds')
plt.xlabel('Date')
plt.ylabel('Travel Duration (in secs)')
```

Next, we group the data according to the PickUp Location to see, if some PickUp Locations have more demand or are it a horizontal line.

```
green_PU_wise_sum = green_taxi_data[green_taxi_data['PULocationID'].notna()].groupby(green_taxi_data['PULocationID']).sum()
yellow_PU_wise_sum = yellow_taxi_data[yellow_taxi_data['PULocationID'].notna()].groupby(yellow_taxi_data['PULocationID']).sum()
fhv_PU_wise_sum = fhv_taxi_data[fhv_taxi_data['PULocationID'].notna()].groupby(fhv_taxi_data['PULocationID']).sum()
```

[https://media-exp1.licdn.com/dms/image/C4E12AQEBraittm6JTA/article-inline_image-shrink_1500_2232/0/1635738556515?e=1651708800&v=beta&t=FUy0fn4aOWfZaMf1y2Af_VFbDmVYpEbHktwXP0lr-H8](https://media-exp1.licdn.com/dms/image/C4E12AQEBraittm6JTA/article-inline_image-shrink_1500_2232/0/1635738556515?e=1651708800&v=beta&t=FUy0fn4aOWfZaMf1y2Af_VFbDmVYpEbHktwXP0lr-H8)

```
plt.plot(green_PU_wise_sum.index[:-1],green_PU_wise_sum[:-1]['trip_duration'])
plt.plot(yellow_PU_wise_sum.index[:-1],yellow_PU_wise_sum[:-1]['trip_duration'])
plt.plot(fhv_PU_wise_sum.index[1:],fhv_PU_wise_sum[1:]['trip_duration'])
plt.legend(['Green','Yellow','For-Hire'])
plt.title('Trip Duration by PickUp Location')
plt.xlabel('PickUp Location')
plt.ylabel('Travel Duration (in secs)')
```

As you can see, in some locations yellow cabs are very prominent whereas in others for-hire cabs dominate. It is interesting to note that in very few locations green cabs are also the front runner. Since yellow dominate we will see a correlation between the yellow cab variables:

[https://media-exp1.licdn.com/dms/image/C4E12AQHBnY2905vRWA/article-inline_image-shrink_1500_2232/0/1635738596705?e=1651708800&v=beta&t=OrQtfLYXCAY2LEln2Uy1k0A0y9LROgnTQsxjBTiqWPM](https://media-exp1.licdn.com/dms/image/C4E12AQHBnY2905vRWA/article-inline_image-shrink_1500_2232/0/1635738596705?e=1651708800&v=beta&t=OrQtfLYXCAY2LEln2Uy1k0A0y9LROgnTQsxjBTiqWPM)

We can see that the total amount (or fare amount) has almost a zero correlation to trip distance (~0.0004) and trip duration (~0.004).

## **Step 7:** ML Model

Let us develop a machine learning model (linear regression) to predict the time for-hire cabs based on Pick Up and Drop Location IDs.

```python
train_X = yellow_taxi_data[['PULocationID','DOLocationID']]
train_y = yellow_taxi_data[['trip_duration']]
model_y = yellow_taxi_data[['total_amount']]
test_X = fhv_taxi_data[['PULocationID','DOLocationID']]
test_y = fhv_taxi_data[['trip_duration']]
```

Here, we have used the yellow taxi data to train and the for-hire taxi data to predict:

```python
reg = LinearRegression().fit(train_X, train_y)
print(reg.score(train_X, train_y))
print("coeff -" + str(reg.coef_))
print("intercept-"+str(reg.intercept_))
pred_y = reg.predict(test_X)
reg.score(test_X,test_y)
np.mean(pred_y-test_y)
```

You will get a mean absolute error of 1238.71 (secs), coefficients as [[ 6.82346983 -4.06077363]] and intercept as [754.75303061] for the train data. For the test data, the mean absolute error is 932.06 (secs). The mean difference between predicted and actual duration is -739.25 i.e. a model based on yellow taxies predicts almost a ~12 minute lesser travel duration.

## Conclusion

One reason for the lower travel time in yellow cabs could be the pricing model, $0.35 per minute + $1.75 per mile for-hire and $0.50 per 1/5 mile or $0.50 per 60 seconds in slow traffic or when the vehicle is stopped for yellow cabs. Since you are charged throughout for the time in for-hire the charges are lower and therefore people may prefer them to wait or delay them, whereas in the case of yellow cabs you are only charged for the time when the vehicle is in slow traffic or stopped (and the price is higher because it also includes a vehicle cost).

Of course, the pricing model is only one of the reasons, the driver's efficiency, behavioral patterns, and other factors can also have a major impact. Thank you for reading this article. If you want to predict the prices or any other value for the for-hire cabs change ‘train_y’ to your preferred value (such as ‘model_y’) and you are good to go. **[ShortHills Tech](https://clutch.co/profile/shorthills-tech#summary)** can help you train your Machine Learning Models and arrive at important business decisions (they are one of the best).

***[ShortHills Tech](http://ShortHills Tech (Gold Microsoft Partner))** is contributor to R**[egistry of Open Data on AWS](https://github.com/awslabs/open-data-registry).** The company can be found on LinkedIn **[ShortHills Tech (Gold Microsoft Partner)](https://www.linkedin.com/company/shorthills-tech/?originalSubdomain=in)**.*
