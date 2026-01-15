import pandas as pd
import json
import re
import requests
import zipfile
import io

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from joblib import dump

# Needs to be global because of the poor design of pandas.DataFrame
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))

def fetch_and_extract(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        print("Download successful")
    else:
        print("Failed to download the dataset")

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(".")
        print("Extraction successful")

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def preproc_inner(text):
    # Forbidden: HTML
    text = re.sub(r'<[^>]+>', '', text)

    # Forbidden: anything that isn't a lowercase Latin letter
    text = re.sub(r'[^a-z\s]', '', text.lower())

    # Tokenization
    tokens = word_tokenize(text)

    # Stopwords, stemming
    tokens = [STEMMER.stem(word) for word in tokens if word not in STOP_WORDS]

    # Return string containing tokens
    return " ".join(tokens)

def preproc(filename):
    df_train = load_json(filename)
    df_train['preprocessed'] = df_train['text'].apply(preproc_inner)

    return df_train

def build_model(df_train):
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])

    param_grid = {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'vectorizer__max_features': [5000, 10000, 20000],
        'vectorizer__min_df': [1, 2, 5],
        'classifier__alpha': [0.5, 1.0, 1.5]
    }

    print('GridSearchCV: tuning hyperparameters...')
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(df_train['preprocessed'], df_train['label'])

    print('Parameter tuning complete. Best parameters:', grid_search.best_params_)

    best_model = grid_search.best_estimator_
    return best_model

def save_model(model, filename):
    dump(model, filename)
    print(f'Final model saved as {filename}')
