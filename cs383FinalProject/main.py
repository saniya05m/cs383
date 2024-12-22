import optparse
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import re
from symspellpy import SymSpell, Verbosity

def parse_args():
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='run decision tree method')

    parser.add_option('-d', '--database', type='string', default = "combined", help='choose ' +\
        ' dev_cleaned, dev_only, or combined')
    parser.add_option('-f', '--feature_extraction', type='string', default = "tdif", help='feature extraction method' +\
        ' tdif or bow')
    parser.add_option('-c', '--cleanup', type='string', default="no", help='clean up type: no, manual, spell, simple')

    (opts, args) = parser.parse_args()

    mandatories = ['cleanup', ]
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)    # Remove mentions
    text = re.sub(r"#\w+", "", text)    # Remove hashtags
    text = re.sub(r"[^a-z\s]", "", text) # Remove special characters
    return text

def clean_spelling(text):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

    # Load a prebuilt dictionary
    sym_spell.load_dictionary("en-80k.txt", term_index=0, count_index=1)
    corrected_words = []
    for word in text.split():
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_words.append(suggestions[0].term if suggestions else word)
    return " ".join(corrected_words)
# Load Dataset
def load_trac_data(train_filepath, dev_filepath):
    # Load train and dev datasets without headers
    column_names = ['id', 'text', 'label']  # Specify column names manually
    train_data = pd.read_csv(train_filepath, header=None, names=column_names)
    dev_data = pd.read_csv(dev_filepath, header=None, names=column_names)

    # Combine datasets
    combined_data = pd.concat([train_data, dev_data], ignore_index=True)


    # Extract 'text' and 'label' columns
    return combined_data['text'], combined_data['label']

def load_trac_data_cleaned_dev(dev_filepath):
    # Load train and dev datasets without headers
    column_names = ['id', 'text', 'label']  # Specify column names manually
    dev_data = pd.read_csv(dev_filepath, header=None, names=column_names)

    # Extract 'text' and 'label' columns
    return dev_data['text'], dev_data['label']
# Map Labels
def encode_labels(labels):
    # Map textual labels to numeric values
    label_mapping = {'NAG': 0, 'CAG': 1, 'OAG': 2}
    return labels.map(label_mapping)


# Preprocessing and Splitting Data
def preprocess_and_split_data_tdif(text, labels, test_size=0.2):
    # Convert text to TF-IDF features
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(text)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=42,
                                                        stratify=labels)
    return X_train, X_test, y_train, y_test, tfidf

def preprocess_and_split_data_bow(text, labels, test_size=0.2):
    # Convert text to Bag-of-Words features
    bow_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X = bow_vectorizer.fit_transform(text)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=42,
                                                        stratify=labels)
    return X_train, X_test, y_train, y_test, bow_vectorizer
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



def main():
    opts = parse_args()
    # Filepaths to TRAC train and dev files
    train_filepath = 'agr_en_train.csv'
    dev_filepath = 'agr_en_dev.csv'
    dev_clean_filepath = 'agr_en_dev_cleaned.csv'

    # Load and preprocess data
    print("Loading and preprocessing data...")
    if opts.cleanup == "manual":
        text, labels = load_trac_data_cleaned_dev(dev_clean_filepath)
        labels = encode_labels(labels)  # Convert labels to numeric values

    if opts.cleanup == "no":
        if opts.database == "dev_cleaned":
            text, labels = load_trac_data_cleaned_dev(dev_clean_filepath)
        elif opts.database == "dev_only":
            text, labels = load_trac_data_cleaned_dev(dev_filepath)
        else:
            text, labels = load_trac_data(train_filepath, dev_filepath)
        labels = encode_labels(labels)

    if opts.cleanup == "spell":
        text, labels = load_trac_data_cleaned_dev(dev_filepath)
        labels = encode_labels(labels)
        text = text.apply(clean_spelling)
    if opts.feature_extraction == "tdif":
        X_train, X_test, y_train, y_test, tfidf = preprocess_and_split_data_tdif(text, labels)
    else:
        X_train, X_test, y_train, y_test, tfidf = preprocess_and_split_data_bow(text, labels)


    # Label names for output
    label_names = ['NAG', 'CAG', 'OAG']

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

main()