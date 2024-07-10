import pandas as pd
import numpy as np
read= pd.read_csv('AppGallery.csv')
read.info()
print("---------------------")
import pandas as pd
import numpy as np
read= pd.read_csv("Purchasing.csv")
read.info()
print("___________________")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
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
        # Tokenize
        tokens = nltk.word_tokenize(text.lower())
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

class LengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([len(text.split()) for text in X]).reshape(-1, 1)

class ImprovedChainedMultiOutputClassifier:
    """

    Attributes:
      label_encoders:
      classifier:
      preprocessor:
    """
    def __init__(self):
        self.label_encoders = {}
        self.classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

        self.preprocessor = Pipeline([
            ('text_prep', TextPreprocessor()),
            ('features', FeatureUnion([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
                ('length', LengthExtractor())
            ]))
        ])

    def preprocess_data(self, df):
        # Combine text features
        df['combined_text'] = df['Ticket Summary'].fillna('') + ' ' + df['Interaction content'].fillna('')

        # Create chained labels
        df['Type_2'] = df['Type 2']
        df['Type_2_3'] = df['Type 2'] + ' + ' + df['Type 3']
        df['Type_2_3_4'] = df['Type 2'] + ' + ' + df['Type 3'] + ' + ' + df['Type 4']

        # Encode labels
        for col in ['Type_2', 'Type_2_3', 'Type_2_3_4']:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].fillna('Unknown'))

        return df

    def prepare_data(self, df):
        X = self.preprocessor.fit_transform(df['combined_text'])
        y = df[['Type_2', 'Type_2_3', 'Type_2_3_4']].values
        return X, y

    def train(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    def evaluate(self, y_true, y_pred):
        type_names = ['Type 2', 'Type 2 + Type 3', 'Type 2 + Type 3 + Type 4']
        instance_accuracies = []

        for i in range(y_true.shape[0]):
            print(f"Email Instance {i+1}:")

            # True Labels
            true_labels = []
            for j in range(min(3, y_true.shape[1])):
                encoder_key = f'Type_{j+2}'
                if encoder_key in self.label_encoders:
                    true_labels.append(self.label_encoders[encoder_key].inverse_transform([y_true[i, j]])[0])
                else:
                    true_labels.append("Unknown")
            print("• True Labels:", ", ".join([f"Type{j+2}: {label}" for j, label in enumerate(true_labels)]))

            # Predicted Labels
            pred_labels = []
            for j in range(min(3, y_pred.shape[1])):
                encoder_key = f'Type_{j+2}'
                if encoder_key in self.label_encoders:
                    pred_labels.append(self.label_encoders[encoder_key].inverse_transform([y_pred[i, j]])[0])
                else:
                    pred_labels.append("Unknown")
            print("• Predicted Labels:", ", ".join([f"Type{j+2}: {label}" for j, label in enumerate(pred_labels)]))

            print("• Evaluation:")
            instance_accuracy = 0
            correct_so_far = True

            for j in range(min(len(true_labels), len(pred_labels))):
                if correct_so_far:
                    if y_true[i, j] == y_pred[i, j]:
                        print(f"  • {type_names[j]}: Correct")
                        instance_accuracy += 1
                    else:
                        print(f"  • {type_names[j]}: Incorrect")
                        correct_so_far = False
                else:
                    print(f"  • {type_names[j]}: Not evaluated (previous prediction was incorrect)")

            instance_accuracy /= len(type_names)
            instance_accuracies.append(instance_accuracy)

            print(f"• Final Accuracy for Instance {i+1}: {instance_accuracy:.0%}")
            print("-" * 50)

        total_accuracy = sum(instance_accuracies) / len(instance_accuracies)
        print(f"Total Accuracy: {total_accuracy:.4f}")

def load_and_combine_data(file_paths):
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    return pd.concat(dfs, ignore_index=True)

def main():
    # Load and combine data
    file_paths = ['Purchasing.csv', 'AppGallery.csv']#FIle paths
    df = load_and_combine_data(file_paths)

    # Initialize and use the classifier
    classifier = ImprovedChainedMultiOutputClassifier()
    df_processed = classifier.preprocess_data(df)
    X, y = classifier.prepare_data(df_processed)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    classifier.train(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    classifier.evaluate(y_test, y_pred)

if __name__ == "__main__":
    main()