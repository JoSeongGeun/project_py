from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fasttext
import fasttext.util
import os

# FastText 모델 로딩
FASTTEXT_MODEL_PATH = "cc.ko.300.bin"

# 로딩된 모델을 전역으로 저장
_fasttext_model = None

def load_fasttext_model():
    global _fasttext_model
    if _fasttext_model is None:
        if not os.path.exists(FASTTEXT_MODEL_PATH):
            raise FileNotFoundError(f"FastText 모델 파일이 '{FASTTEXT_MODEL_PATH}'에 존재하지 않습니다.")
        _fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    return _fasttext_model

def get_fasttext_vector(user_keywords, corpus):
    """
    사용자 키워드와 예식장 텍스트 데이터를 FastText로 벡터화 후 유사도 계산.
    """
    model = load_fasttext_model()

    # 사용자 벡터
    user_vec = np.mean([model.get_word_vector(word) for word in user_keywords if word.strip()], axis=0).reshape(1, -1)

    # 데이터 벡터 (문장별 평균 벡터)
    corpus_vectors = []
    for text in corpus:
        words = text.split()
        vecs = [model.get_word_vector(w) for w in words if w.strip()]
        avg_vec = np.mean(vecs, axis=0) if vecs else np.zeros((300,))
        corpus_vectors.append(avg_vec)
    
    corpus_vectors = np.vstack(corpus_vectors)

    return user_vec, corpus_vectors

def calculate_similarity(df, col, target_val):
    """
    수치형 데이터 유사도 계산 (차이값의 정규화 후 역변환).
    """
    diff = np.abs(df[col].values - target_val).reshape(-1, 1)
    sim = 1 - MinMaxScaler().fit_transform(diff).flatten()
    return sim