import sys
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    engine = create_engine('sqlite:///../data/dr.db')
    cnx = engine.connect()
    df = pd.read_sql_table('dr', cnx)
    X = df.message
    y = df.drop(columns=['id', 'message', 'original', 'genre'], axis=1)
    return X, y, y.columns


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100,
                                                             min_samples_split=2)))
        ])
    return pipeline


def search_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
             'clf__estimator__n_estimators': [50, 100, 200],
             'clf__estimator__min_samples_split': [2, 3, 4],
             }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    c_rep = {}
    y_pred = model.predict(X_test)
    for i in range(36):
        c_rep[Y_test.columns[i]] = classification_report(y_pred[:,i], Y_test.iloc[:,i], output_dict=True)
        print(Y_test.columns[i])
        print(classification_report(y_pred[:,i], Y_test.iloc[:,i]))
    # pickle.dump(c_rep, open('classification_report_rf_default.pickle', 'wb'))
    pickle.dump(c_rep, open('classification_report_rf_tuned.pickle', 'wb'))
    return


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()