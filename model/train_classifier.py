# import libraries
import sys
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
import re
from sklearn.svm import LinearSVC
import pickle
import os.path


def load_data(database_filepath):
    """
    parameters:
    - database_filepath database where is data

    return:
    - X
    - Y
    - category_names

    """

    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    # run a query
    df=pd.read_sql_table('data',engine)

    category_names=['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    X = df['message']
    Y = df[category_names]
    return(X,Y,category_names)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens







def build_model():
    """
    parameters:
    none

    return:
    - model
    """

    pipeline=Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',RandomForestClassifier())
    ])
    return(pipeline)

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred=model.predict(X_test)
    Y_pred=pd.DataFrame(Y_pred,columns=category_names)
    for i in category_names:
        print(i)
        print(classification_report(Y_test[i], Y_pred[i]))



def save_model(model, model_filepath):
    """
    parameters:
    - model to save
    -model_filepath path where the model will be saved

    action:
    - save pkl file
    """
    #https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    file_path = model_filepath
    n_bytes = 2**31
    max_bytes = 2**31 - 1
    data = bytearray(n_bytes)
    ## write
    bytes_out = pickle.dumps(model)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

def load_model(model_filepath):
    """
    parameters:
    -model_filepath path where is model

    return:
    - model from that pkl path
    """
    file_path = model_filepath
    max_bytes = 2**31 - 1
    ## read
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return(pickle.loads(bytes_in))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        print('load data')
        X, Y, category_names = load_data(database_filepath)
        print('split data')
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
