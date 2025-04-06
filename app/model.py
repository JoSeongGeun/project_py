import json
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

class WeddingHallRecommender:
    def __init__(self, model_path="doc2vec.model", data_path="wedding_halls_with_vectors.csv"):
        self.model_path = model_path
        self.data_path = data_path

        # 🔹 모델과 데이터프레임 로딩
        self.model = Doc2Vec.load(self.model_path)
        print("✅ Doc2Vec 모델 로딩 완료")

        self.df = pd.read_csv(self.data_path)
        self.df["doc2vec_vector"] = self.df["doc2vec_vector"].apply(lambda x: np.array(json.loads(x)))
        print("✅ 예식장 데이터 로딩 완료")

    def recommend(self, user_input, top_n=5):
        """사용자 입력과 가장 유사한 예식장 추천"""
        # 🔹 사용자 리뷰 벡터화
        user_vector = self.model.infer_vector(user_input.split()).reshape(1, -1)

        # 🔹 유사도 계산
        all_vectors = np.stack(self.df["doc2vec_vector"].values)
        similarities = cosine_similarity(user_vector, all_vectors).flatten()

        # 🔹 유사도 상위 예식장 반환
        top_indices = similarities.argsort()[::-1][:top_n]
        return self.df.iloc[top_indices][["예식장", "대관료", "식대", "최소수용인원", "최대수용인원", "주차장(대)"]]
