from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_word2vec_model(path="model/word2vec.model"):
    return Word2Vec.load(path)

def get_avg_word2vec_vector(keywords, model):
    vectors = []
    for word in keywords:
        if word in model.wv:
            vectors.append(model.wv[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def get_corpus_avg_vectors(corpus, model):
    return np.array([
        get_avg_word2vec_vector(doc, model) for doc in corpus
    ])