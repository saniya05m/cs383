import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import re
from symspellpy import SymSpell, Verbosity


def clean_text(text):
    # Remove URLs, mentions, hashtags, and special characters
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)    # Remove mentions
    text = re.sub(r"#\w+", "", text)    # Remove hashtags
    text = re.sub(r"[^a-z\s]", "", text)  # Remove special characters
    return text
# Load Dataset
def load_trac_data(train_filepath, dev_filepath):
    # Load train and dev datasets without headers
    column_names = ['id', 'text', 'label']  # Specify column names manually
    train_data = pd.read_csv(train_filepath, header=None, names=column_names)
    dev_data = pd.read_csv(dev_filepath, header=None, names=column_names)

    # Combine datasets
    combined_data = dev_data

    # Extract 'text' and 'label' columns
    return combined_data['text'], combined_data['label']


# Map Labels
def encode_labels(labels):
    # Map textual labels to numeric values
    label_mapping = {'NAG': 0, 'CAG': 1, 'OAG': 2}
    return labels.map(label_mapping)


# Preprocessing and Splitting Data
def preprocess_and_split_data(text, labels, test_size=0.2):
    # Convert text to TF-IDF features
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(text)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=42,
                                                        stratify=labels)
    return X_train, X_test, y_train, y_test, tfidf


# Train Logistic Regression
def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    return lr


# Train SVM
def train_svm(X_train, y_train, kernel='linear'):
    svm = SVC(kernel=kernel, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    return svm


# Evaluate Model
def evaluate_model(model, X_test, y_test, label_names):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# Main Function
if __name__ == "__main__":
    # Filepaths to TRAC train and dev files
    train_filepath = 'agr_en_train.csv'
    dev_filepath = 'agr_en_dev_cleaned.csv'

    # Load and preprocess data
    print("Loading and preprocessing data...")
    text, labels = load_trac_data(train_filepath, dev_filepath)
    text = text.apply(clean_text)
    print(text)
    labels = encode_labels(labels)  # Convert labels to numeric values
    X_train, X_test, y_train, y_test, tfidf = preprocess_and_split_data(text, labels)

    # Label names for output
    label_names = ['Non-Aggressive', 'Covertly Aggressive', 'Overtly Aggressive']

    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    print("\nEvaluating Logistic Regression...")
    evaluate_model(lr_model, X_test, y_test, label_names)

    # Train SVM
    print("\nTraining SVM...")
    svm_model = train_svm(X_train, y_train, kernel='linear')
    print("\nEvaluating SVM...")
    evaluate_model(svm_model, X_test, y_test, label_names)
