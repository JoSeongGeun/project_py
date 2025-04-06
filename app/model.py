import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from .utils import get_tfidf_vector

class WeddingRecommender:
    def __init__(self, csv_path="data/data.csv"):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.df["cleaned_doctagged_doc"] = self.df["cleaned_doc"].apply(eval)

    def recommend(self, survey_data):
        user_keywords = sum(survey_data["리뷰"], [])  # [['좋다'], ['예쁘다']] -> ['좋다', '예쁘다']
        user_text = " ".join(user_keywords)
        corpus = [" ".join(words) for words in self.df["cleaned_doc"]]

        user_vec, data_vecs = get_tfidf_vector(user_keywords=user_keywords, corpus=corpus)
        tfidf_sim = cosine_similarity(user_vec, data_vecs).flatten()

        df = self.df.copy()
        df["tfidf_sim"] = tfidf_sim

        # 각 수치적 유사도 계산
        def calculate_similarity(col, target_val):
            diff = np.abs(df[col].values - target_val).reshape(-1, 1)
            sim = 1 - MinMaxScaler().fit_transform(diff).flatten()
            return sim

        df["rental_fee_sim"] = calculate_similarity("대관료", survey_data["대관료"])
        df["food_price_sim"] = calculate_similarity("식대", survey_data["식대"])
        df["mini_hc_sim"] = calculate_similarity("최소수용인원", survey_data["최소수용인원"])
        df["limit_hc_sim"] = calculate_similarity("최대수용인원", survey_data["최대수용인원"])
        df["car_park_sim"] = calculate_similarity("주차장(대)", survey_data["주차장"])

        # 가중치 기반 유사도 합산
        df["total_sim"] = (
            df["tfidf_sim"] +
            df["rental_fee_sim"] * survey_data["rental_fee_weight"] +
            df["food_price_sim"] * survey_data["food_price_weight"] +
            df["mini_hc_sim"] * survey_data["mini_hc_weight"] +
            df["limit_hc_sim"] * survey_data["limit_hc_weight"] +
            df["car_park_sim"] * survey_data["car_park_weight"]
        )

        top5 = df.sort_values("total_sim", ascending=False).head(5)

        return top5[[
            "예식장", "대관료", "식대", "최소수용인원", "최대수용인원", "주차장(대)", "total_sim"
        ]].to_dict(orient="records")
