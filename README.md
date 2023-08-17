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
### 1) Run Python script for our machine learning model.

```
Enter Code here.
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

![Screenshot_1](https://github.com/RunningWomann/Project_4-Airline-Delays-in-United-States/assets/126130038/3aad3aef-62cd-4a05-9c5d-7834e0515018)
![Screenshot_2](https://github.com/RunningWomann/Project_4-Airline-Delays-in-United-States/assets/126130038/bfb2d2e4-d155-4492-b753-384df813dd8c)







## Slide Deck URL Link
https://docs.google.com/presentation/d/1GMbDEG-OmT-2sZ0gFJJDa3f32OXHFAf0Fo3mm4_SQyo/edit?usp=sharing

## Dashboard Link
http://project4bootcamp.pythonanywhere.com/






## Data
Source: https://www.kaggle.com/datasets/divyansh22/flight-delay-prediction

Collection Methodology: Download data from website
