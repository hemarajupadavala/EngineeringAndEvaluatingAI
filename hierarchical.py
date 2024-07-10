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
def predict(classifier, X):
    index = range(X.shape[0])
    predictions = pd.DataFrame(index=index, columns=['Type 2', 'Type 3', 'Type 4'])

    predictions['Type 2'] = classifier.models['Type 2'].predict(X)

    for i in range(X.shape[0]):
        class_2 = predictions.iloc[i]['Type 2']
        if class_2 in classifier.models['Type 3']:
            pred_3 = classifier.models['Type 3'][class_2].predict(X[i:i+1])[0]
            predictions.iloc[i, predictions.columns.get_loc('Type 3')] = pred_3

            if (class_2, pred_3) in classifier.models['Type 4']:
                pred_4 = classifier.models['Type 4'][(class_2, pred_3)].predict(X[i:i+1])[0]
                predictions.iloc[i, predictions.columns.get_loc('Type 4')] = pred_4

    return predictions
def hierarchical_accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    for i in range(total):
        if y_true.iloc[i]['Type 2'] == y_pred.iloc[i]['Type 2']:
            if y_true.iloc[i]['Type 3'] == y_pred.iloc[i]['Type 3']:
                if y_true.iloc[i]['Type 4'] == y_pred.iloc[i]['Type 4']:
                    correct += 1
                else:
                    correct += 2/3
            else:
                correct += 1/3
    return correct / total

def custom_classification_report(y_true, y_pred, labels, target_names):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    report = "              precision    recall  f1-score   support\n\n"
    for i, label in enumerate(labels):
        report += f"{target_names[i]:>15} {precision[i]:>9.2f} {recall[i]:>8.2f} {f1[i]:>8.2f} {support[i]:>8d}\n"

    report += "\n"
    accuracy = accuracy_score(y_true, y_pred)
    report += f"    accuracy                         {accuracy:>8.2f} {np.sum(support):>8d}\n"

    avg_precision = np.average(precision, weights=support)
    avg_recall = np.average(recall, weights=support)
    avg_f1 = np.average(f1, weights=support)
    report += f"   macro avg {avg_precision:>9.2f} {avg_recall:>8.2f} {avg_f1:>8.2f} {np.sum(support):>8d}\n"

    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    report += f"weighted avg {weighted_precision:>9.2f} {weighted_recall:>8.2f} {weighted_f1:>8.2f} {np.sum(support):>8d}\n"

    return report

def evaluate(classifier, y_true, y_pred):
    # Ensure y_true and y_pred have the same index
    common_index = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[common_index]
    y_pred = y_pred.loc[common_index]

    print(f"Hierarchical Accuracy: {hierarchical_accuracy(y_true, y_pred):.4f}")

    for col in ['Type 2', 'Type 3', 'Type 4']:
        print(f"\nEvaluating {col}")
        y_true_filtered = y_true[col]
        y_pred_filtered = y_pred[col]

        print(f"Number of predictions: {len(y_true_filtered)}")

        true_labels = set(y_true_filtered)
        pred_labels = set(y_pred_filtered)
        all_labels = sorted(true_labels.union(pred_labels))

        print(f"Unique values in y_true: {true_labels}")
        print(f"Unique values in y_pred: {pred_labels}")
        print(f"All unique labels: {all_labels}")

        try:
            acc = accuracy_score(y_true_filtered, y_pred_filtered)
            print(f"Accuracy for {col}: {acc:.4f}")

            target_names = [classifier.label_encoders[col].classes_[label] if label < len(classifier.label_encoders[col].classes_) else f'Unknown_{label}'
                            for label in all_labels]

            report = custom_classification_report(y_true_filtered, y_pred_filtered,
                                                  labels=all_labels,
                                                  target_names=target_names)
            print(report)
        except Exception as e:
            print(f"Error in evaluation for {col}: {str(e)}")
        print("-" * 50)

def load_and_combine_data(file_paths):
    return pd.concat([pd.read_csv(file_path) for file_path in file_paths], ignore_index=True)
def main():
    file_paths = ['Purchasing.csv', 'AppGallery.csv']
    df = load_and_combine_data(file_paths)

    print("Data shape:", df.shape)
    print("Column names:", df.columns)
    print("Type 2 value counts:", df['Type 2'].value_counts())
    print("Type 3 value counts:", df['Type 3'].value_counts())
    print("Type 4 value counts:", df['Type 4'].value_counts())

    classifier = HierarchicalMultiOutputClassifier()
    df_processed = preprocess_data(classifier, df)
    X, y = prepare_data(classifier, df_processed)

    print("Processed data shape:", X.shape)
    print("Target data shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)

    train(classifier, X_train, y_train)
    y_pred = predict(classifier, X_test)
    evaluate(classifier, y_test, y_pred)
if __name__ == "__main__":
    main()