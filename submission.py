import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
import sys
import scipy.sparse as sp
from langchain_huggingface import HuggingFaceEmbeddings
import pickle

# Read base data
df = pd.read_parquet('artifacts/Processed_dataset.parquet')

# Load TF-IDF vectorizer
with open('artifacts/tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Load tfidf matrix
tfidf_matrix = sp.load_npz("artifacts/sparse_matrix.npz")

# Load embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load embedding matrix
embedding_matrix = np.vstack(df["embedding"].values)


if __name__ =="__main__":
    if len(sys.argv)<2:
        query = "I like action movies set in space"
    else:
        query = sys.argv[1]

    print(f"Generating recommendations for the query {query}")
    print("Recommendations using TF-IDF and cosine similarity:")
    print(get_recommendations(query, df, tfidf_vectorizer, tfidf_matrix, top_n=5))

    print("\nRecommendations using HuggingFace embeddings and cosine similarity:")
    print(find_similar_movies(df,query,embedding_model, embedding_matrix, top_n=5))