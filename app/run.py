import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine
import pickle

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/dr.db')
df = pd.read_sql_table('dr', engine)

# load model
model = joblib.load("../models/model.pickle")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # graph 2; display the label distribution 0 or 1 in the category fields
    categories = list(set(df.columns).difference({'id', 'message', 'original', 'genre'}))
    classes_counts = []
    for c in categories:
        class_counts = df.value_counts(subset=c, sort=True)
        try:
            zeros = class_counts[0]
        except:
            zeros = 0
        try:
            ones = class_counts[1]
        except:
            ones = 0
        classes_counts.append((c, zeros, ones))
    
    # sort the counts ascending by the number of zeroes; all classes have at most two values: zero and one 
    sorted_class_counts = sorted(classes_counts, key=lambda x:x[1])
    categories = [ x[0] for x in sorted_class_counts ]
    zeros      = [ x[1] for x in sorted_class_counts ]
    ones       = [ x[2] for x in sorted_class_counts ]
    
    # for the next graph, Distribution of Prediction Accuracy per Category
    # keep the sorted categories above and create y values with the accuracy of the model for each category:
    cr = pickle.load(open('../models/classification_report_rf_tuned.pickle', 'rb'))
    accuracies = [ cr[c]['accuracy'] for c in categories ]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories,
                    y=zeros,
                    name='zeros'
                ),
                Bar(
                    x=categories,
                    y=ones,
                    name='ones'
                )
            ],

            'layout': {
                'title': 'Distribution of Category Classes',
                'barmode': 'stack',
                'yaxis': {
                    'title': "Classes Counts"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },                
        {
            'data': [
                Bar(
                    x=categories,
                    y=accuracies
                )
            ],

            'layout': {
                'title': 'Distribution of Prediction Accuracy per Category',
                'yaxis': {
                    'title': "Accuracy"
                },
                'xaxis': {
                    'title': "Categories"
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
