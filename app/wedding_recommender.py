import os
import json
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# 📌 현재 스크립트가 있는 디렉토리 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.csv")


# 📌 CSV 저장 함수
def save_dataframe(df, filename=DATA_FILE):
    df_copy = df.copy()
    df_copy["tagged_doc"] = df_copy["tagged_doc"].apply(json.dumps)
    df_copy["doc2vec_vector"] = df_copy["doc2vec_vector"].apply(lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else json.dumps(x))
    df_copy.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"✅ DataFrame이 '{filename}' 파일로 저장됨!")


# 📌 CSV 불러오기 함수
def load_dataframe(filename=DATA_FILE):
    if not os.path.exists(filename):
        print(f"⚠️ '{filename}' 파일이 존재하지 않습니다. 기본 데이터를 생성합니다.")
        return create_sample_data()
    
    df = pd.read_csv(filename, encoding="utf-8-sig")
    df["tagged_doc"] = df["tagged_doc"].apply(json.loads)
    df["doc2vec_vector"] = df["doc2vec_vector"].apply(json.loads)
    df["doc2vec_vector"] = df["doc2vec_vector"].apply(lambda x: np.array(x))
    
    print(f"✅ '{filename}' 파일에서 DataFrame 불러오기 완료!")
    return df


# 📌 샘플 데이터 생성 함수
def create_sample_data():
    sample_data = {
        "예식장": ["웨딩홀A", "웨딩홀B", "웨딩홀C"],
        "대관료": [1500000, 2000000, 2500000],
        "식대": [50000, 60000, 70000],
        "최소수용인원": [100, 150, 200],
        "최대수용인원": [500, 1000, 1500],
        "주차장(대)": [100, 200, 300],
        "tagged_doc": [["좋다", "멋지다"], ["예쁘다", "화려하다"], ["고급스럽다", "아름답다"]],
        "doc2vec_vector": [np.random.rand(300).tolist(), np.random.rand(300).tolist(), np.random.rand(300).tolist()]
    }
    df_sample = pd.DataFrame(sample_data)
    save_dataframe(df_sample, DATA_FILE)
    return df_sample


# 📌 웨딩홀 추천 함수
def recommend_wedding_hall(survey_df, df, top_n=10, review_weight=1, rental_fee_weight=0.7, food_price_weight=0.5, mini_hc_weight=0.7, limit_hc_weight=0.1, car_park_weight=0.5):

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

    # 4. 최소수용인원 필터링 및 유사도 계산
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

    # 5. 최대수용인원 유사도 계산
    target_limit_hc = survey_df.loc[0, "최대수용인원"]
    all_limit_hc = df_filtered["최대수용인원"].values
    limit_hc_diff = np.abs(all_limit_hc - target_limit_hc)
    limit_hc_sim = 1 - MinMaxScaler().fit_transform(limit_hc_diff.reshape(-1, 1)).flatten()

    # 6. 주차장(대) 유사도 계산
    target_car_park = survey_df.loc[0, "주차장(대)"]
    all_car_park = df_filtered["주차장(대)"].values
    car_park_diff = np.abs(all_car_park - target_car_park)
    car_park_sim = 1 - MinMaxScaler().fit_transform(car_park_diff.reshape(-1, 1)).flatten()

    # 7. 최종 유사도 계산 (가중치 적용)
    final_sim = (
        (review_sim * review_weight) +
        (rental_fee_sim * rental_fee_weight) +
        (food_price_sim * food_price_weight) +
        (mini_hc_sim * mini_hc_weight) +
        (limit_hc_sim * limit_hc_weight) +
        (car_park_sim * car_park_weight)
    )

    final_sim[0] = -np.inf  # 자기 자신 제외
    top_indices = np.argsort(final_sim)[-top_n:][::-1]

    # 8. 결과 반환
    result_df = df_filtered.iloc[top_indices].copy()
    result_df["final_similarity"] = final_sim[top_indices]

    return result_df


# 📌 실행 코드
if __name__ == "__main__":
    df = load_dataframe()

    survey = {
        "리뷰": [["좋다", "멋지다", "예쁘다", "축복", "광주", "결혼식"]],
        "대관료": [2000000],
        "식대": [60000],
        "최소수용인원": [150],
        "최대수용인원": [1000],
        "주차장(대)": [1000]
    }
    survey_df = pd.DataFrame(survey)
    survey_df["doc2vec_vector"] = survey_df["리뷰"].apply(lambda x: np.random.rand(300))

    recommendations = recommend_wedding_hall(survey_df, df, top_n=5)

    print("\n🏆 추천 예식장 리스트:")
    print(recommendations.to_string(index=False))
