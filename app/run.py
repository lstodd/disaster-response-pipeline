import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

from models.train_classifier import tokenize, StartingNounExtractor

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('RawData', engine)

# load model
# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    request_counts = df.request.value_counts()
    request_names = list(request_counts.index)

    offer_counts = df.offer.value_counts()
    offer_names = list(offer_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=request_names,
                    y=request_counts
                )
            ],

            'layout': {
                'title': 'Volume of requests',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Request"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=offer_names,
                    y=offer_counts
                )
            ],

            'layout': {
                'title': 'Volume of offers',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Offer"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
