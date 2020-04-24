# Disaster-Management
Disaster Management done by classifying thousands of real messages, sent during natural disasters either via social media or directly to disaster response organizations into categories of interest.

## Project Overview

This data has been collected and lebelled by Figure Eight. Following a disaster we may get millions of messages via tweets and social media right at the time when the disaster response agencies have least amount of capacities to filter the most important ones. So typically they way disaster response organisations respond to these messages is that different organisations take up different parts of the problem. So we use machine learning techniques  to classify these messages into different categories in order to facilitate these organisations and find trends in the data.

## Data files

We have two data files. One of them contains all the messages. The second data file contains all the categories associated with those messages.

## Project Components
There are two components in this project. 

The data files can be found in the data folder. The step by step creation of the ETL pipeline and the Machine Learning pipeline can be seen in the jupyter notebook /data/ETL Pipeline Preparation.ipynb and /models/ML Pipeline Preparation.ipynb. These jupyter notebooks reflect the step by step process but please execute python scripts in the step below for classification results

### 1. ETL Pipeline

In a Python script, process_data.py, write a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database


### 2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

## Instructions

Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
