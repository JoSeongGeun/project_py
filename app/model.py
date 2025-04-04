import json
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

class WeddingHallRecommender:
    def __init__(self, model_path="doc2vec.model"):
        self.model_path = model_path
        self.model = None
        self.df = None  # 예식장 데이터

    def train_model(self, data_path="wedding_halls.csv", vector_size=300, epochs=20):
        """Doc2Vec 모델을 학습하고 저장"""
        self.df = pd.read_csv(data_path)

        # 텍스트를 토큰화하여 TaggedDocument로 변환
        documents = [TaggedDocument(words=review.split(), tags=[i]) for i, review in enumerate(self.df["리뷰"])]

        # Doc2Vec 모델 생성 및 학습
        model = Doc2Vec(vector_size=vector_size, window=5, min_count=1, workers=4, epochs=epochs)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

        # 모델 저장
        model.save(self.model_path)
        print(f"✅ 모델이 '{self.model_path}'에 저장되었습니다.")

        # 데이터프레임에 벡터 추가
        self.df["doc2vec_vector"] = self.df["리뷰"].apply(lambda x: model.infer_vector(x.split()))
        self.df.to_csv("wedding_halls_with_vectors.csv", index=False)
        print("✅ 예식장 데이터에 벡터를 추가하여 저장했습니다.")

    def load_model(self):
        """저장된 Doc2Vec 모델 로드"""
        self.model = Doc2Vec.load(self.model_path)
        print("✅ 모델이 로드되었습니다.")

    def recommend(self, user_input, top_n=5):
        """사용자 입력과 가장 유사한 예식장 추천"""
        if self.model is None:
            self.load_model()

        # 사용자 입력을 벡터화
        user_vector = self.model.infer_vector(user_input.split()).reshape(1, -1)

        # 데이터 로드
        self.df = pd.read_csv("wedding_halls_with_vectors.csv")
        self.df["doc2vec_vector"] = self.df["doc2vec_vector"].apply(lambda x: np.array(json.loads(x)))

        # 유사도 계산
        all_vectors = np.stack(self.df["doc2vec_vector"].values)
        similarities = cosine_similarity(user_vector, all_vectors).flatten()

        # 유사도가 높은 상위 예식장 반환
        top_indices = similarities.argsort()[::-1][:top_n]
        return self.df.iloc[top_indices][["예식장", "대관료", "식대", "최소수용인원", "최대수용인원", "주차장(대)"]]

