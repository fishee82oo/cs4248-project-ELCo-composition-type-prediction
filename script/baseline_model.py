import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def preprocess_text(text):
    return text.str.lower().replace(r'[^a-zA-Z0-9\s]', ' ', regex=True).str.strip().str.split().apply(lambda tokens: ' '.join(tokens))

def main():
    # load the dataset
    ELCo_df = pd.read_csv('data/ELCo.csv')
    ELCo_df = ELCo_df.drop(columns=['EM'])

    # preprocess the dataset
    ELCo_df['Description'] = preprocess_text(ELCo_df['Description'])
    ELCo_df['EN'] = preprocess_text(ELCo_df['EN'])

    # map the 'Composition strategy' column to numerical values
    composition_strategy_mapping = {name: idx for idx, name in enumerate(ELCo_df['Composition strategy'].unique())}
    ELCo_df['Composition strategy'] = ELCo_df['Composition strategy'].map(composition_strategy_mapping)

    # split the dataset into train, validate and test sets
    train_df, test_df = train_test_split(ELCo_df, test_size=0.2, random_state=42, stratify=ELCo_df['Composition strategy'])

    # further split the test set into validate and test sets
    train_df, validate_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Composition strategy'])
    X_train, y_train = train_df.drop(columns=['Composition strategy']), train_df['Composition strategy']
    X_validate, y_validate = validate_df.drop(columns=['Composition strategy']), validate_df['Composition strategy']
    X_test, y_test = test_df.drop(columns=['Composition strategy']), test_df['Composition strategy']

    # feature extraction
    vectorizer_en = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    vectorizer_description = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train_en = vectorizer_en.fit_transform(X_train['EN'])
    X_train_description = vectorizer_description.fit_transform(X_train['Description'])
    X_train_vectorized = hstack([X_train_en, X_train_description])

    # define the model
    model = LogisticRegression(max_iter=1000, verbose=1)

    # train the model
    model.fit(X_train_vectorized, y_train)
    
    # validate the model
    X_validate_en = vectorizer_en.transform(X_validate['EN'])
    X_validate_description = vectorizer_description.transform(X_validate['Description'])
    X_validate_vectorized = hstack([X_validate_en, X_validate_description])
    y_validate_pred = model.predict(X_validate_vectorized)
    validate_accuracy = np.mean(y_validate_pred == y_validate)
    print(f"Validation Accuracy: {validate_accuracy:.4f}")

    # test the model
    X_test_en = vectorizer_en.transform(X_test['EN'])
    X_test_description = vectorizer_description.transform(X_test['Description'])
    X_test_vectorized = hstack([X_test_en, X_test_description])
    y_test_pred = model.predict(X_test_vectorized)
    test_accuracy = np.mean(y_test_pred == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()

