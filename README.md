# Disaster Response Pipeline Project
https://github.com/lstodd/disaster-response-pipeline.git

When a disaster strikes we want to be able to meet the needs of the people quickly and efficiently. In this project we 
aim to use historic data to be able to categorise incoming comments and requests so that we can deploy 
the appropriate expertise quickly. 

### Data
The original data consists of messages from multiple sources and the annotated categories. Messages can be labelled as 
more than one category.  

### Model 
We create a multi class multi label model using RandomForestClassifier and natural language processing (NLP) features. 

### Instructions:

### Requirements:
* pandas
* numpy
* scikit-learn
* nltk
* sqlalchemy
* joblib
* flask
* plotly

#### Installing
```pip install .```


1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Web App

To classify a new message using the trained model you can enter a message on the web app:

![Enter message](/app/screenshots/enter_message.PNG)

After clicking the 'classify' button, you will see the selected categories for your message highlighted in green:

![Categorised message](/app/screenshots/categorised_message.PNG)

Now we can deploy the appropriate response depending on the selected categories. In the above image we can see that this message relates to:
* related
* aid related
* medical help
* direct report
