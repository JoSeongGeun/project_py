import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

class WeddingRecommender:
    def __init__(self, csv_path="data/data.csv"):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # 👉 안전하게 doc2vec_vector 컬럼 파싱하는 함수
        def parse_vector(x):
            try:
                if isinstance(x, str):
                    return np.array(x.strip("[]").split(), dtype=float)
                elif isinstance(x, list) or isinstance(x, np.ndarray):
                    return np.array(x)
                else:
                    return np.zeros(300)
            except Exception as e:
                print("❌ doc2vec_vector 파싱 에러:", e)
                return np.zeros(300)

        # 👉 파싱 함수 적용
        self.df["doc2vec_vector"] = self.df["doc2vec_vector"].apply(parse_vector)

    def recommend(self, survey):
        # 입력 데이터 프레임 변환
        review = [survey["리뷰"]]
        doc = [TaggedDocument(words=review[0], tags=["user"])]
        model = Doc2Vec(vector_size=300, window=5, min_count=1, epochs=20)
        model.build_vocab(doc)
        model.train(doc, total_examples=model.corpus_count, epochs=model.epochs)

        survey_vector = model.infer_vector(review[0])
        survey_df = pd.DataFrame([{
            **survey,
            "doc2vec_vector": survey_vector
        }])

        return self._recommend_core(survey_df)

    def _recommend_core(self, survey_df, top_n=5,
                        review_weight=1, rental_fee_weight=0.7,
                        food_price_weight=0.5, mini_hc_weight=0.7,
                        limit_hc_weight=0.1, car_park_weight=0.5):
        
        df = self.df.copy()

        # 1. 리뷰 유사도 계산
        target_vector = survey_df.loc[0, "doc2vec_vector"].reshape(1, -1)
        all_vector = np.stack(df["doc2vec_vector"].values, axis=0)
        review_sim = cosine_similarity(target_vector, all_vector).flatten()

        # 2. 대관료 유사도 계산
        target_rental_fee = survey_df.loc[0, "대관료"]
        all_rental_fee = df["대관료"].values.reshape(1, -1)
        rental_fee_diff = np.abs(all_rental_fee - target_rental_fee)
        rental_fee_sim = 1 - MinMaxScaler().fit_transform(rental_fee_diff).flatten()

        # 3. 식대 유사도 계산
        target_food_price = survey_df.loc[0, "식대"]
        all_food_price = df["식대"].values.reshape(1, -1)
        food_price_diff = np.abs(all_food_price - target_food_price)
        food_price_sim = 1 - MinMaxScaler().fit_transform(food_price_diff).flatten()

        # 4. 최소수용인원 유사도 계산 및 기준 이하 예식장 제거
        target_mini_hc = survey_df.loc[0, "최소수용인원"]
        all_mini_hc = df["최소수용인원"].values

        # 조건에 맞는 예식장만 남기기
        valid_indices = np.where(all_mini_hc >= target_mini_hc)[0]
        df_filtered = df.iloc[valid_indices].reset_index(drop=True)

        review_sim = review_sim[valid_indices]
        rental_fee_sim = rental_fee_sim[valid_indices]
        food_price_sim = food_price_sim[valid_indices]
        all_mini_hc = all_mini_hc[valid_indices]

        # 최소수용인원 차이 계산
        mini_hc_diff = np.abs(all_mini_hc - target_mini_hc).reshape(-1, 1)
        mini_hc_sim = 1 - MinMaxScaler().fit_transform(mini_hc_diff).flatten()

        # 5. 최대수용인원 유사도
        target_limit_hc = survey_df.loc[0, "최대수용인원"]
        all_limit_hc = df_filtered["최대수용인원"].values
        limit_hc_diff = np.abs(all_limit_hc - target_limit_hc)
        limit_hc_sim = 1 - MinMaxScaler().fit_transform(limit_hc_diff.reshape(-1, 1)).flatten()

        # 6. 주차장(대) 유사도
        target_car_park = survey_df.loc[0, "주차장"]
        all_car_park = df_filtered["주차장(대)"].values
        car_park_diff = np.abs(all_car_park - target_car_park)
        car_park_sim = 1 - MinMaxScaler().fit_transform(car_park_diff.reshape(-1, 1)).flatten()

        # 7. 최종 유사도 계산
        final_sim = (
            (review_sim * review_weight) +
            (rental_fee_sim * rental_fee_weight) +
            (food_price_sim * food_price_weight) +
            (mini_hc_sim * mini_hc_weight) +
            (limit_hc_sim * limit_hc_weight) +
            (car_park_sim * car_park_weight)
        )

        # 정규화 및 자기 자신 제외
        final_sim = (final_sim - final_sim.min()) / (final_sim.max() - final_sim.min())
        final_sim[0] = -np.inf

        # Top-N 추천 추출
        selected_halls = []
        checked_names = set()

        for idx in np.argsort(final_sim)[::-1]:
            hall_name = df_filtered.loc[idx, "예식장"]
            if hall_name not in checked_names:
                selected_halls.append(idx)
                checked_names.add(hall_name)
            if len(selected_halls) == top_n:
                break

        top_indices = selected_halls

        result_df = df_filtered.loc[top_indices, [
            "예식장", "대관료", "식대", "최소수용인원", "최대수용인원", "주차장(대)"
        ]].copy()
        result_df["final_similarity"] = final_sim[top_indices]

        return result_df.to_dict(orient="records")