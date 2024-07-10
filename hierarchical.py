import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from textblob import TextBlob

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocess(text) for text in X]

    def preprocess(self, text):
        tokens = nltk.word_tokenize(str(text).lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

class HierarchicalMultiOutputClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        self.label_encoders = {}
        self.models = {'Type 2': None, 'Type 3': {}, 'Type 4': {}}
        self.text_preprocessor = TextPreprocessor()
def preprocess_data(classifier, df):
    df['combined_text'] = df['Ticket Summary'].fillna('') + ' ' + df['Interaction content'].fillna('')
    df['combined_text'] = classifier.text_preprocessor.transform(df['combined_text'])

    # Add new features
    df['text_length'] = df['combined_text'].apply(len)
    df['word_count'] = df['combined_text'].apply(lambda x: len(x.split()))
    df['sentiment'] = df['combined_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    for col in ['Type 2', 'Type 3', 'Type 4']:
        classifier.label_encoders[col] = LabelEncoder()
        df[col] = classifier.label_encoders[col].fit_transform(df[col].fillna('Unknown'))

    return df

def prepare_data(classifier, df):
    X_text = classifier.vectorizer.fit_transform(df['combined_text'])
    X_features = df[['text_length', 'word_count', 'sentiment']].values
    X = np.hstack((X_text.toarray(), X_features))
    return X, df[['Type 2', 'Type 3', 'Type 4']]

def safe_smote(X, y):
    if len(np.unique(y)) > 1:
        try:
            return SMOTE(random_state=42).fit_resample(X, y)
        except ValueError:
            print(f"SMOTE failed. Using original data. Unique classes: {np.unique(y)}")
    return X, y
def train(classifier, X, y):
    # Train Type 2 model
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    classifier.models['Type 2'] = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
    classifier.models['Type 2'].fit(X, y['Type 2'])

    # Train Type 3 models
    for class_2 in np.unique(y['Type 2']):
        mask = y['Type 2'] == class_2
        X_filtered, y_filtered = safe_smote(X[mask], y.loc[mask, 'Type 3'])

        # Ensure y_filtered has consecutive integer labels starting from 0
        y_filtered = pd.factorize(y_filtered)[0]

        classifier.models['Type 3'][class_2] = xgb.XGBClassifier(random_state=42)
        classifier.models['Type 3'][class_2].fit(X_filtered, y_filtered)

        # Train Type 4 models
        for class_3 in np.unique(y_filtered):
            mask = (y['Type 2'] == class_2) & (y['Type 3'] == class_3)
            X_filtered, y_filtered = safe_smote(X[mask], y.loc[mask, 'Type 4'])

            # Ensure y_filtered has consecutive integer labels starting from 0
            y_filtered = pd.factorize(y_filtered)[0]

            classifier.models['Type 4'][(class_2, class_3)] = xgb.XGBClassifier(random_state=42)
            classifier.models['Type 4'][(class_2, class_3)].fit(X_filtered, y_filtered)