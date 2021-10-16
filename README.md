# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - The application run.py is hard coded to use the data/dr.db file
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - Training the classiffier takes about 5 minutes on an AMD Ryzen 7 3700X system
    - The application run.py is hard coded to use the models/model.pickle file

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Motivation for this project

This project is part of an assignment of the [Udacity Data Science nano degree program](https://www.udacity.com/course/data-scientist-nanodegree--nd025)


## Files
### Data files

data/disaster_messages.csv - messages which need to be classified
data/disaster_categories.csv - curated labels for the messages above

### Modules required to run the code

nltk
flask
plotly
joblib
sqlalchemy
pickle
json
numpy
pandas
sys
os
sklearn

The code was tested in a conda created environment using conda 4.10.3 with Python 3.7.7 running on Ubuntu 20.04
