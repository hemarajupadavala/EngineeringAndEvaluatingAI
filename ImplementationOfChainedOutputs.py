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




print("Main Chained Outputs")

#Importing Libraries
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


# Pre Proessing the Textual Data
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
    
'''

Output
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 122 entries, 0 to 121
Data columns (total 11 columns):
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   Ticket id               122 non-null    int64
 1   Interaction id          122 non-null    int64
 2   Interaction date        96 non-null     float64
 3   Mailbox                 122 non-null    object
 4   Ticket Summary          121 non-null    object
 5   Interaction content     120 non-null    object
 6   Innso TYPOLOGY_TICKET   122 non-null    object
 7   Type 1                  122 non-null    object
 8   Type 2                  122 non-null    object
 9   Type 3                  89 non-null     object
 10  Type 4                  87 non-null     object
dtypes: float64(1), int64(2), object(8)
memory usage: 10.6+ KB
---------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 84 entries, 0 to 83
Data columns (total 11 columns):
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   Ticket id               84 non-null     int64
 1   Interaction id          84 non-null     int64
 2   Interaction date        73 non-null     float64
 3   Mailbox                 84 non-null     object
 4   Ticket Summary          76 non-null     object
 5   Interaction content     84 non-null     object
 6   Innso TYPOLOGY_TICKET   84 non-null     object
 7   Type 1                  84 non-null     object
 8   Type 2                  84 non-null     object
 9   Type 3                  82 non-null     object
 10  Type 4                  76 non-null     object
dtypes: float64(1), int64(2), object(8)
memory usage: 7.3+ KB
___________________
Main Chained Outputs
Email Instance 1:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 1: 100%
--------------------------------------------------
Email Instance 2:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Incorrect
  • Type 2 + Type 3: Not evaluated (previous prediction was incorrect)
  • Type 2 + Type 3 + Type 4: Not evaluated (previous prediction was incorrect)
• Final Accuracy for Instance 2: 0%
--------------------------------------------------
Email Instance 3:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 3: 100%
--------------------------------------------------
Email Instance 4:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 4: 100%
--------------------------------------------------
Email Instance 5:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 5: 100%
--------------------------------------------------
Email Instance 6:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 6: 100%
--------------------------------------------------
Email Instance 7:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 7: 100%
--------------------------------------------------
Email Instance 8:
• True Labels: Type2: Others, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Others, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 8: 100%
--------------------------------------------------
Email Instance 9:
• True Labels: Type2: Others, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Others, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 9: 100%
--------------------------------------------------
Email Instance 10:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 10: 100%
--------------------------------------------------
Email Instance 11:
• True Labels: Type2: Others, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Others, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 11: 100%
--------------------------------------------------
Email Instance 12:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Incorrect
  • Type 2 + Type 3 + Type 4: Not evaluated (previous prediction was incorrect)
• Final Accuracy for Instance 12: 33%
--------------------------------------------------
Email Instance 13:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 13: 100%
--------------------------------------------------
Email Instance 14:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 14: 100%
--------------------------------------------------
Email Instance 15:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 15: 100%
--------------------------------------------------
Email Instance 16:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Incorrect
  • Type 2 + Type 3: Not evaluated (previous prediction was incorrect)
  • Type 2 + Type 3 + Type 4: Not evaluated (previous prediction was incorrect)
• Final Accuracy for Instance 16: 0%
--------------------------------------------------
Email Instance 17:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Incorrect
  • Type 2 + Type 3 + Type 4: Not evaluated (previous prediction was incorrect)
• Final Accuracy for Instance 17: 33%
--------------------------------------------------
Email Instance 18:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 18: 100%
--------------------------------------------------
Email Instance 19:
• True Labels: Type2: Others, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Incorrect
  • Type 2 + Type 3: Not evaluated (previous prediction was incorrect)
  • Type 2 + Type 3 + Type 4: Not evaluated (previous prediction was incorrect)
• Final Accuracy for Instance 19: 0%
--------------------------------------------------
Email Instance 20:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Incorrect
  • Type 2 + Type 3: Not evaluated (previous prediction was incorrect)
  • Type 2 + Type 3 + Type 4: Not evaluated (previous prediction was incorrect)
• Final Accuracy for Instance 20: 0%
--------------------------------------------------
Email Instance 21:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 21: 100%
--------------------------------------------------
Email Instance 22:
• True Labels: Type2: Others, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Others, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 22: 100%
--------------------------------------------------
Email Instance 23:
• True Labels: Type2: Others, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Others, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 23: 100%
--------------------------------------------------
Email Instance 24:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 24: 100%
--------------------------------------------------
Email Instance 25:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 25: 100%
--------------------------------------------------
Email Instance 26:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Incorrect
  • Type 2 + Type 3 + Type 4: Not evaluated (previous prediction was incorrect)
• Final Accuracy for Instance 26: 33%
--------------------------------------------------
Email Instance 27:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 27: 100%
--------------------------------------------------
Email Instance 28:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 28: 100%
--------------------------------------------------
Email Instance 29:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 29: 100%
--------------------------------------------------
Email Instance 30:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Incorrect
• Final Accuracy for Instance 30: 67%
--------------------------------------------------
Email Instance 31:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 31: 100%
--------------------------------------------------
Email Instance 32:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 32: 100%
--------------------------------------------------
Email Instance 33:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Others, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Incorrect
  • Type 2 + Type 3: Not evaluated (previous prediction was incorrect)
  • Type 2 + Type 3 + Type 4: Not evaluated (previous prediction was incorrect)
• Final Accuracy for Instance 33: 0%
--------------------------------------------------
Email Instance 34:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 34: 100%
--------------------------------------------------
Email Instance 35:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 35: 100%
--------------------------------------------------
Email Instance 36:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 36: 100%
--------------------------------------------------
Email Instance 37:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Incorrect
• Final Accuracy for Instance 37: 67%
--------------------------------------------------
Email Instance 38:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 38: 100%
--------------------------------------------------
Email Instance 39:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 39: 100%
--------------------------------------------------
Email Instance 40:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 40: 100%
--------------------------------------------------
Email Instance 41:
• True Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Problem/Fault, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 41: 100%
--------------------------------------------------
Email Instance 42:
• True Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Predicted Labels: Type2: Suggestion, Type3: Unknown, Type4: Unknown
• Evaluation:
  • Type 2: Correct
  • Type 2 + Type 3: Correct
  • Type 2 + Type 3 + Type 4: Correct
• Final Accuracy for Instance 42: 100%
--------------------------------------------------
Total Accuracy: 0.8175
'''