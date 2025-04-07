import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from .utils import load_word2vec_model, get_avg_word2vec_vector, get_corpus_avg_vectors

class WeddingRecommender:
    def __init__(self):
        self.df = pd.read_csv("data/data.csv")

        if isinstance(self.df["cleaned_doc"].iloc[0], str) and self.df["cleaned_doc"].iloc[0].startswith("["):
            self.df["cleaned_doc"] = self.df["cleaned_doc"].apply(eval)

        self.model = load_word2vec_model()
        self.df["vector"] = get_corpus_avg_vectors(self.df["cleaned_doc"], self.model).tolist()

    def recommend(self, survey_data):
        user_keywords = sum(survey_data["리뷰"], [])
        user_vector = get_avg_word2vec_vector(user_keywords, self.model).reshape(1, -1)

        vectors = np.array(self.df["vector"].tolist())
        text_sim = cosine_similarity(user_vector, vectors).flatten()

        df = self.df.copy()
        df["text_sim"] = text_sim

        def calculate_similarity(col, target_val):
            diff = np.abs(df[col].values - target_val).reshape(-1, 1)
            sim = 1 - MinMaxScaler().fit_transform(diff).flatten()
            return sim

        df["rental_fee_sim"] = calculate_similarity("대관료", survey_data["대관료"])
        df["food_price_sim"] = calculate_similarity("식대", survey_data["식대"])
        df["mini_hc_sim"] = calculate_similarity("최소수용인원", survey_data["최소수용인원"])
        df["limit_hc_sim"] = calculate_similarity("최대수용인원", survey_data["최대수용인원"])
        df["car_park_sim"] = calculate_similarity("주차장(대)", survey_data["주차장"])

        df["total_sim"] = (
            df["text_sim"] +
            df["rental_fee_sim"] * survey_data["rental_fee_weight"] +
            df["food_price_sim"] * survey_data["food_price_weight"] +
            df["mini_hc_sim"] * survey_data["mini_hc_weight"] +
            df["limit_hc_sim"] * survey_data["limit_hc_weight"] +
            df["car_park_sim"] * survey_data["car_park_weight"]
        )

        df = df.drop_duplicates(subset="예식장", keep="first")

        top5 = df.sort_values("total_sim", ascending=False).head(5)

        return top5[[
            "예식장", "대관료", "식대", "최소수용인원", "최대수용인원", "주차장(대)", "total_sim"
        ]].to_dict(orient="records")