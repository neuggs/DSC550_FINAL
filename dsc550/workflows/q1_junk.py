import gc
import pandas as pd
import numpy as np
import re
import warnings
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,confusion_matrix
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

warnings.filterwarnings('ignore')

stopWords = set(stopwords.words('english'))

def get_data(option, file):
    PATH = '../../data/source/reddit/'
    CAT_FILE_FULL = PATH + 'categorized-comments.jsonl'
    CAT_FILE_SMALL = PATH + 'small_cat_comments.csv'
    if option == 'small':
        cat_comments = pd.read_csv(CAT_FILE_SMALL)
        return cat_comments
    elif option == 'sample':
        cat_comments_full = pd.read_json(CAT_FILE_FULL, lines=True)
        cat_comments = cat_comments_full.sample(n=200)
        return cat_comments
    elif option == 'full':
        cat_comments = pd.read_json(CAT_FILE_FULL, lines=True)
        return cat_comments
    else:
        return None

def processing(df):
    #lowering and removing punctuation
    df['processed'] = df['txt'].apply(lambda x: re.sub(r'[^\w\s]','', x.lower()))
    cat_categories = {'cat': {'news': 1.0, 'science_and_technology': 2.0, 'sports': 3.0, 'video_games': 4.0}}
    df.replace(cat_categories, inplace=True)

    return(df)

df = get_data('sample', 'cat')
df.dropna(axis=0)
df = processing(df)

df.to_pickle('junk.pkl')

un_df = pd.read_pickle('junk.pkl')
print(un_df)

"""
features= [c for c in df.columns.values if c not in ['cat','txt']]
target = [c for c in df.columns.values if c not in ['processed','txt']]

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33, random_state=42)
tfidf = TfidfVectorizer(stop_words='english')
features_train = tfidf.fit_transform(X_train['processed'].tolist())
print('Train:', features_train.shape[1])
features_test = tfidf.transform(X_test['processed'].tolist())
print('Test', features_test.nnz)
"""
