<div align="center">!<img src=https://github.com/RunningWomann/Project_4/assets/126130038/8eab9c6b-fa75-4252-b435-3882f18d7f08 ></div>


# Project 4: Airline Delays in United States

## Contributors 
Andy Nguyen, Cassie Contreras, Chun Zhao, Jing Xu, Rihanna Afkami

## Background




## Key Things to Note
What is considered “delayed”?
A flight is considered delayed when it arrived 15 or more minutes later than the schedule

How many years are we analyzing?
We are analyzing 5 years worth of historical data. We will be reviewing data from April 2018 to April 2023.

How many airline carriers are we comparing in our Dashboard?
5 major airline carriers
(American Airlines Inc., Delta Air Lines Inc., Spirit Airlines, United Air Lines Inc., Southwest Airlines Co.)



## Slide Deck URL Link
https://docs.google.com/presentation/d/1GMbDEG-OmT-2sZ0gFJJDa3f32OXHFAf0Fo3mm4_SQyo/edit?usp=sharing




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

Below is our index.html file.
```

```

### 3) Open the localhost (http://127.0.0.1:5000/)



## Data
Source: https://www.kaggle.com/datasets/divyansh22/flight-delay-prediction

Collection Methodology: Download data from website
