from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Function to compute cosine similarity and return top recommendations
def get_recommendations(query, _df, tfidf_vectorizer, tfidf_matrix, top_n=5):
    query_tfidf = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    _df["similarity"] = cosine_similarities  # Store similarity scores
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return _df.iloc[top_indices][['title','similarity', 'overview', 'popularity']]

# Function to find similar movies
def find_similar_movies(_df,query,embedding_model, embedding_matrix, top_n=5):
    query_embedding = np.array(embedding_model.embed_query(query)).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embedding_matrix)[0]
    _df["similarity"] = similarities  # Store similarity scores
    top_movies = _df.sort_values(by="similarity", ascending=False).head(top_n)
    
    return top_movies[["title", "similarity",'overview', "popularity"]]