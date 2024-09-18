import joblib
import pandas as pd
import numpy as np
import os
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# use in word_tokenize

def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt_tab/english/')
    except LookupError:
        nltk.download('punkt_tab')

def load_csv(file_path):
    return pd.read_csv(file_path)


def combine_feature(df):
    df['combined_text'] = df['AUTHOR'] + ' ' + df['VIDEO_NAME'] + ' ' + df['CONTENT']
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    text = ' '.join(tokens)
    return text

def create_embedding(df, method='tfidf', embedding_config=None):
    """
    Create emebedding from text data
    :param df: the dataframe containing the text data
    :param method: 'tfidf' or 'transformer'
    :param kwargs: The arguments for the method
    :return: embedding and model
    """
    df['cleaned_text'] = df['combined_text'].apply(clean_text)

    if method == 'tfidf':
        max_features = embedding_config.get('max_features', 1000)
        vectorizer = TfidfVectorizer(max_features=max_features)
        embedding = vectorizer.fit_transform(df['cleaned_text'])
        return embedding, vectorizer

    elif method == 'transformer':
        model_name = embedding_config.get('model_name', 'distilbert-base-nli-mean-tokens')
        max_seq_length = embedding_config.get('max_seq_length', 128)
        batch_size = embedding_config.get('batch_size', 32)

        model = SentenceTransformer(model_name)
        model.max_seq_length = max_seq_length

        embeddings = model.encode(df['cleaned_text'], batch_size=batch_size, show_progress_bar=True)
        return embeddings, model

    else:
        raise ValueError('Invalid method. Choose from tfidf or transformer')

def preprocess_data(data, embedding_method, embedding_config):
    download_nltk_resources()
    data = combine_feature(data)
    X, embedder = create_embedding(data, embedding_method, embedding_config)
    y = data['CLASS'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
    return X_train, X_test, y_train, y_test, embedder

def save_embedder(embedder, file_path):
    if isinstance(embedder, SentenceTransformer):
        embedder.save(file_path)
    else:
        joblib.dump(embedder, file_path)

def load_embedder(file_path, embedder_type):
    if embedder_type == 'transformer':
        return SentenceTransformer(file_path)
    else:
        return joblib.load(file_path)

def save_process_data(X_train, X_test, y_train, y_test, embedder, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if sparse.issparse(X_train):
        sparse.save_npz(os.path.join(output_dir, 'X_train.npz'), X_train)
        sparse.save_npz(os.path.join(output_dir, 'X_test.npz'), X_test)
    else:
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)

    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    save_embedder(embedder, os.path.join(output_dir, 'embedder.joblib'))

def load_process_data(output_dir):
    try:
        X_train = sparse.load_npz(os.path.join(output_dir, 'X_train.npz'))
        X_test = sparse.load_npz(os.path.join(output_dir, 'X_test.npz'))
    except:
        X_train = np.load(os.path.join(output_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(output_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(output_dir, 'y_test.npy'))
    return X_train, X_test, y_train, y_test
