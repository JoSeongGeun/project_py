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
        else:
            # 모델에 없으면 가장 유사한 단어 하나 추천해서 사용
            alt = suggest_similar(word, model)
            if alt and alt in model.wv:
                print(f"💡 '{word}' 대신 '{alt}' 사용됨")
                vectors.append(model.wv[alt])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def get_corpus_avg_vectors(corpus, model):
    return np.array([
        get_avg_word2vec_vector(doc, model) for doc in corpus
    ])

def suggest_similar(word, model):
    try:
        similar = model.most_similar(positive=[word], topn=1)
        return similar[0][0]  # 가장 유사한 단어 하나 반환
    except:
        return None