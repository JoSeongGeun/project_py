from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_tfidf_vector(user_keywords, corpus):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(corpus + [user_keywords])
    return tfidf_matrix[-1], tfidf_matrix[:-1]
