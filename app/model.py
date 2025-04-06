import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from gensim.models.fasttext import load_facebook_model

class WeddingRecommender:
    def __init__(self):
        self.df = pd.read_csv("data/data.csv")

        if isinstance(self.df["cleaned_doc"].iloc[0], str) and self.df["cleaned_doc"].iloc[0].startswith("["):
            self.df["cleaned_doctagged_doc"] = self.df["cleaned_doc"].apply(eval)
        else:
            self.df["cleaned_doctagged_doc"] = self.df["cleaned_doc"]

        # FastText 모델 로드 (cc.ko.300.bin 파일 필요)
        self.ft_model = load_facebook_model("data/cc.ko.300.bin")

    def get_fasttext_vector(self, words):
        vectors = [self.ft_model.wv[word] for word in words if word in self.ft_model.wv]
        if not vectors:
            return np.zeros(self.ft_model.vector_size)
        return np.mean(vectors, axis=0)

    def recommend(self, survey_data):
        user_keywords = sum(survey_data["리뷰"], [])
        user_vec = self.get_fasttext_vector(user_keywords)

        doc_vectors = [self.get_fasttext_vector(words) for words in self.df["cleaned_doc"]]
        ft_sim = cosine_similarity([user_vec], doc_vectors).flatten()

        df = self.df.copy()
        df["ft_sim"] = ft_sim

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
            df["ft_sim"] +
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