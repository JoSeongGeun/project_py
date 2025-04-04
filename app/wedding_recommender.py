# wedding_recommender.py
import pandas as pd
import numpy as np
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ===================== CSV 저장 함수 =====================
def save_dataframe(df, filename="data.csv"):
    df_copy = df.copy()
    df_copy["tagged_doc"] = df_copy["tagged_doc"].apply(json.dumps)
    df_copy["doc2vec_vector"] = df_copy["doc2vec_vector"].apply(lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else json.dumps(x))
    df_copy.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"✅ DataFrame이 '{filename}' 파일로 저장됨!")

# ===================== CSV 불러오기 함수 =====================
def load_dataframe(filename="data.csv"):
    df = pd.read_csv(filename, encoding="utf-8-sig")
    df["tagged_doc"] = df["tagged_doc"].apply(json.loads)
    df["doc2vec_vector"] = df["doc2vec_vector"].apply(json.loads)
    df["doc2vec_vector"] = df["doc2vec_vector"].apply(lambda x: np.array(x))
    print(f"📥 '{filename}' 파일에서 DataFrame 불러오기 완료!")
    return df

# ===================== 추천 함수 =====================
def recommend_wedding_hall(survey_df, df, top_n=10, review_weight=1,
                           rental_fee_weight=0.7, food_price_weight=0.5,
                           mini_hc_weight=0.7, limit_hc_weight=0.1, car_park_weight=0.5):

    target_vector = survey_df.loc[0, "doc2vec_vector"].reshape(1, -1)
    all_vector = np.stack(df["doc2vec_vector"].values, axis=0)
    review_sim = cosine_similarity(target_vector, all_vector).flatten()

    target_rental_fee = survey_df.loc[0, "대관료"]
    rental_fee_diff = np.abs(df["대관료"].values.reshape(1, -1) - target_rental_fee)
    rental_fee_sim = 1 - MinMaxScaler().fit_transform(rental_fee_diff).flatten()

    target_food_price = survey_df.loc[0, "식대"]
    food_price_diff = np.abs(df["식대"].values.reshape(1, -1) - target_food_price)
    food_price_sim = 1 - MinMaxScaler().fit_transform(food_price_diff).flatten()

    target_mini_hc = survey_df.loc[0, "최소수용인원"]
    all_mini_hc = df["최소수용인원"].values
    valid_indices = np.where(all_mini_hc >= target_mini_hc)[0]

    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    review_sim = review_sim[valid_indices]
    rental_fee_sim = rental_fee_sim[valid_indices]
    food_price_sim = food_price_sim[valid_indices]
    all_mini_hc = all_mini_hc[valid_indices]

    mini_hc_diff = np.abs(all_mini_hc - target_mini_hc).reshape(-1, 1)
    mini_hc_sim = 1 - MinMaxScaler().fit_transform(mini_hc_diff).flatten()

    target_limit_hc = survey_df.loc[0, "최대수용인원"]
    all_limit_hc = df_filtered["최대수용인원"].values
    limit_hc_diff = np.abs(all_limit_hc - target_limit_hc)
    limit_hc_sim = 1 - MinMaxScaler().fit_transform(limit_hc_diff.reshape(-1, 1)).flatten()

    target_car_park = survey_df.loc[0, "주차장(대)"]
    all_car_park = df_filtered["주차장(대)"].values
    car_park_diff = np.abs(all_car_park - target_car_park)
    car_park_sim = 1 - MinMaxScaler().fit_transform(car_park_diff.reshape(-1, 1)).flatten()

    final_sim = (
        (review_sim * review_weight) +
        (rental_fee_sim * rental_fee_weight) +
        (food_price_sim * food_price_weight) +
        (mini_hc_sim * mini_hc_weight) +
        (limit_hc_sim * limit_hc_weight) +
        (car_park_sim * car_park_weight)
    )

    final_sim = (final_sim - final_sim.min()) / (final_sim.max() - final_sim.min())
    final_sim[0] = -np.inf

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

    target_df = survey_df.loc[[0], ["대관료", "식대", "최소수용인원", "최대수용인원", "주차장(대)"]].copy()
    result_df = df_filtered.loc[top_indices, ["예식장", "대관료", "식대", "최소수용인원", "최대수용인원", "주차장(대)"]].copy()
    result_df["final_similarity"] = final_sim[top_indices]

    return target_df, result_df

# ===================== 실행부 =====================
if __name__ == "__main__":
    # 데이터 불러오기
    df = load_dataframe()

    # 설문 데이터 예시
    survey = {
        "리뷰": [['좋다', '멋지다', '예쁘다', '축복', '광주', '결혼식']],
        "대관료": [2000000],
        "식대": [60000],
        "최소수용인원": [150],
        "최대수용인원": [1000],
        "주차장(대)": [1000]
    }
    survey_df = pd.DataFrame(survey)

    # Doc2Vec 모델 학습
    documents = [TaggedDocument(words=review, tags=[i]) for i, review in enumerate(survey_df["리뷰"])]
    model = Doc2Vec(vector_size=300, window=5, min_count=1, workers=4, epochs=20)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    survey_df["doc2vec_vector"] = survey_df["리뷰"].apply(lambda x: model.infer_vector(x))

    # 추천 함수 호출
    target_info, recommendation = recommend_wedding_hall(
        survey_df, df,
        top_n=5,
        review_weight=1,
        rental_fee_weight=0.7,
        food_price_weight=0.5,
        mini_hc_weight=0.7,
        limit_hc_weight=0.1,
        car_park_weight=0.5
    )

    # 결과 출력
    print("\n🎯 사용자 입력 기준 정보:")
    print(target_info.to_string(index=False))

    print("\n🏆 추천 예식장 리스트:")
    print(recommendation.to_string(index=False))


