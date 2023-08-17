<div align="center">!<img src=https://github.com/RunningWomann/Project_4/assets/126130038/8eab9c6b-fa75-4252-b435-3882f18d7f08 ></div>


# Project 4: Airline Delays in United States

## Contributors 
Andy Nguyen, Cassie Contreras, Chun Zhao, Jing Xu, Rihanna Afkami

## Background
Flight delays have been gradually increasing and bringing more financial difficulties to airline companies.  Despite the increase in travel delays, more and more people are choosing to travel.   

Using data collected from the Bureau of Transportation Statistics, Govt. of the USA, we analyzed a dataset that contained all flights in the month of January 2020 for 5 major airlines in the United States, a total of over 300,000 flights. We applied several machine learning models to track the on-time performance of domestic flights.  


## Key Things to Note
What is considered “delayed”?
A flight is considered delayed when it arrived 15 or more minutes later than the schedule

How many months/years are we analyzing?
We are analyzing fight data for January 2020.

How many airline carriers are we comparing in our Dashboard?
5 major airline carriers
(American Airlines Inc., Delta Air Lines Inc., Spirit Airlines, United Air Lines Inc., Southwest Airlines Co.)

## Questions
1) What ML model is recommended to predict delayed or ontime flights accurately?
2) Is it possible to predict which airports will have delayed or cancelled flights?
3) How can we solve over-fitted/imbalanced data?
   

## Coding Approach
### 1) Run Python script for our machine learning model

```
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Series
import re
from pathlib import Path
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
%matplotlib inline
from matplotlib import pyplot as plt

#Bring in the CSV
file = Path('Resources/Jan_2020_ontime.csv')
df = pd.read_csv(file) 

#Display DF
df.head()

# Review columns in data
df.columns

# Check data types
df.dtypes


# Review unique carrier names to identify what will be excluded
df.OP_UNIQUE_CARRIER.unique()

#Drop columns not needed
columns_to_drop = ['CANCELLED','DIVERTED','OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'ORIGIN_AIRPORT_ID','DEST_AIRPORT_ID','DEP_TIME_BLK', 'ARR_TIME','TAIL_NUM','DEP_DEL15','ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID','DISTANCE','Unnamed: 21']
df_clean = df.drop(columns=columns_to_drop)
df_clean = df_clean.set_index('OP_CARRIER_FL_NUM')
print(df_clean.head())
                  
#Find null values
for column in df_clean.columns:
    print(f'Column {column} has {df_clean[column].isnull().sum()}null values')
Column DAY_OF_MONTH has 0null values
Column DAY_OF_WEEK has 0null values
Column OP_CARRIER has 0null values
Column ORIGIN has 0null values
Column DEST has 0null values
Column DEP_TIME has 6664null values
Column ARR_DEL15 has 8078null values
#drop null values
df_cleaned = df_clean.dropna()
print(df_cleaned)
                   

# List of airline codes to keep
OP_CARRIER = ['AA', 'NK', 'DL', 'UA', 'WN']
filter_df = df_cleaned[df_cleaned['OP_CARRIER'].isin(OP_CARRIER)]
filter_df = filter_df.dropna()
print(filter_df)
    
corr_matrix = filter_df.corr()

# Plot the correlation matrix heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Flight Delays by Airline 
# Calculate the count of flights for each airline
carrier_counts = filter_df['OP_CARRIER'].value_counts()

# Create the bar chart
plt.figure(figsize=(6, 4))
sns.barplot(x=carrier_counts.index, y=carrier_counts.values)
plt.title('Flight Count by Airline (Bar Chart)')
plt.xlabel('Airline')
plt.ylabel('Flight Count')
plt.xticks(rotation=45)  
plt.show()

# Flight Delays by Day of Week
# Calculate the mean arrival delay for each day of the week
mean_delay_by_day = filter_df.groupby('DAY_OF_WEEK')['ARR_DEL15'].mean()

# Create the line chart
plt.figure(figsize=(6, 4))
sns.lineplot(x=mean_delay_by_day.index, y=mean_delay_by_day.values)
plt.title('Average Arrival Delay by Day of Week (Line Chart)')
plt.xlabel('Day of Week')
plt.ylabel('Average Arrival Delay')
plt.xticks(mean_delay_by_day.index, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])  
plt.show()

# Flight delays during the time of the day
# Categorize time intervals
def categorize_time(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

# create a new column
filter_df['TimeCategory'] = filter_df['DEP_TIME'].apply(lambda x: categorize_time(x // 100))

# Calculate the mean arrival delay 
mean_delay_by_time = filter_df.groupby('TimeCategory')['ARR_DEL15'].mean()

# Create the line chart
plt.figure(figsize=(6, 4))
sns.lineplot(x=mean_delay_by_time.index, y=mean_delay_by_time.values)
plt.title('Average Arrival Delay by Time Interval (Line Chart)')
plt.xlabel('Time Interval')
plt.ylabel('Average Arrival Delay')
plt.show()

#Find the total number of delayed and on time flights
#0 is On Time
#1 is Delayed
filter_df["ARR_DEL15"].value_counts()
0.0    289763
1.0     38697
Name: ARR_DEL15, dtype: int64
#Huge imbalance in dataset
#we must sample the minority by Randomoversampling to balance data
from sklearn.utils import resample
import seaborn as sns
# majority and minority class
df_majority = filter_df[(filter_df["ARR_DEL15"]==0)]
df_minority = filter_df[(filter_df["ARR_DEL15"]==1)]
#upsample minority class 
df_minority_upsampled = resample(df_minority,
                                replace=True,
                                n_samples= 289763,
                                random_state=42)
# combine
df_upsampled = pd.concat([df_minority_upsampled,df_majority])
#Display the new oversampled dataframe
df_upsampled


#Confirm that value counts are the same
df_upsampled["ARR_DEL15"].value_counts()

#Display plot
sns.countplot(df_upsampled["ARR_DEL15"])

#display imbalanced data 
sns.countplot(filter_df["ARR_DEL15"])


#Display airlines
df_upsampled["OP_CARRIER"].value_counts()

#Change airline codes to numeric data
# WN = Southwest (1)
# AA = Alaskan Airlines (2)
# DL = Delta Airlines (3)
# UA = United Airlines (4)
# NK = Spirit Airlines (5)
number_carrier = {"WN": 1,"AA": 2, "DL": 3,"UA": 4, "NK":5}
df_upsampled["CARRIER_NUM"] = df_upsampled["OP_CARRIER"].map(number_carrier)
#drop OP carrier for now
df_upsampled.drop(columns = "OP_CARRIER", axis=1)


# Liner regression 
# R-squared is used to measure the variance in the dependent variable that's predictable from the independent variables. It didn't provide a good evaluation for binary classification tasks.
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

y = df_upsampled['ARR_DEL15']
data = df_upsampled.drop(['ARR_DEL15'], axis=1)
X = data.select_dtypes(include=['int', 'float']).values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model 
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2*100)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy * 100)

# Create a Logistic Regression Model to predict Delay
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs',max_iter=1000,
                            random_state=78)
classifier

#train
classifier.fit(X_train, y_train)
LogisticRegression(max_iter=1000, random_state=78)
#see results
print(f"Training Data Score: {classifier.score(X_train, y_train)*100}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)*100}")
Training Data Score: 59.03757387515638
Testing Data Score: 59.2894241885666
#Display the predictions
predictions = classifier.predict(X_test)
predictions_df = pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)
predictions_df.head(50)

from sklearn.metrics import accuracy_score
# Display the accuracy score for the test dataset.
AS = accuracy_score(y_test, predictions) * 100
AS

 
# Random Forest
# Convert categorical data to numeric with `pd.get_dummies`
dummies = pd.get_dummies(df_upsampled)
dummies


#drop null values
dummies.dropna()


# Define features set
X = dummies.copy()
X.drop('ARR_DEL15', axis=1, inplace=True)
X.dropna()


# Define target vector
y = dummies['ARR_DEL15'].ravel()
y[:5]
array([1., 1., 1., 1., 1.])
dummies['ARR_DEL15'].value_counts()

# Splitting into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

# Creating StandardScaler instance
scaler = StandardScaler()

# Fitting Standard Scaller

X_scaler = scaler.fit(X_train)
# Scaling data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
# Create a random forest classifier
rf_model = RandomForestClassifier(n_estimators=500, random_state=78)
# Fitting the model
rf_model = rf_model.fit(X_train_scaled, y_train)
# Making predictions using the testing data
predictions = rf_model.predict(X_test_scaled)
# Calculating the confusion matrix
cm = confusion_matrix(y_test, predictions)
class_names = ['on_time', 'delayed']
cm_df= pd.DataFrame(cm, index=class_names, columns=class_names)
# Calculating the accuracy score
acc_score = accuracy_score(y_test, predictions)
print('Confusion Matrix')
display(cm_df)
print(f'Accuracy Score : {acc_score}')
print('Classification Report')
print(classification_report(y_test, predictions))

# Random Forests in sklearn will automatically calculate feature importance
importances = rf_model.feature_importances_
# We can sort the features by their importance
sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)


# Visualize the features by importance
importances_df = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns), reverse=True))
importances_df.set_index(importances_df[1], inplace=True)
importances_df.drop(columns=1, inplace=True)
importances_df.rename(columns={0: 'Feature Importances'}, inplace=True)
importances_sorted = importances_df.sort_values(by='Feature Importances')
top_features = importances_sorted.head(20)
top_features.plot(kind='barh', color='lightgreen', title= 'Top 20 Features Importances', legend=False)
<AxesSubplot:title={'center':'Top 20 Features Importances'}, ylabel='1'>

#BONUS MACHINE LEARNING ATTEMPT
# Import the KNeighborsClassifier module from sklearn
from sklearn.neighbors import KNeighborsClassifier
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# Instantiate the KNeighborsClassifier model with n_neighbors = 3 
knn = KNeighborsClassifier(n_neighbors=3)
# Train the model using the training data
knn.fit(X_train_scaled, y_train)
# Create predictions using the testing data
y_pred = knn.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
             
```



### 2) Flask Dashboard
A dashboard that combines a Tableau Dashboard and a Javascript chart.

Install requirements and build flask dev server
```
from flask import Flask, jsonify, render_template

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

#################################################
# Flask Routes
#################################################

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug = True)
```


### 3) Open the localhost (http://127.0.0.1:5000/) or view our Dashboard Link

![Screenshot_1](https://github.com/RunningWomann/Project_4-Airline-Delays-in-United-States/assets/126130038/162f6b30-876e-404e-bff6-51cb13ac395e)
![Screenshot_2](https://github.com/RunningWomann/Project_4-Airline-Delays-in-United-States/assets/126130038/bfb2d2e4-d155-4492-b753-384df813dd8c)







## Slide Deck URL Link
https://docs.google.com/presentation/d/1GMbDEG-OmT-2sZ0gFJJDa3f32OXHFAf0Fo3mm4_SQyo/edit?usp=sharing

## Dashboard URL Link
http://project4bootcamp.pythonanywhere.com/






## Data
Source: https://www.kaggle.com/datasets/divyansh22/flight-delay-prediction

Collection Methodology: Download data from website
