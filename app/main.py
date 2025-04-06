from fastapi import FastAPI
from app.model import WeddingHallRecommender
from app.database import fetch_all_halls
from app.schemas import RecommendRequest  # ✅ NEW: 요청 모델 import

app = FastAPI()
recommender = WeddingHallRecommender()

@app.get("/")
def home():
    return {"message": "예식장 추천 시스템 API"}

@app.get("/wedding-halls")
def get_wedding_halls():
    return fetch_all_halls().to_dict(orient="records")

# ✅ POST 방식으로 추천 API 추가
@app.post("/recommend")
def recommend_wedding_hall(request: RecommendRequest):
    """사용자 입력 기반 예식장 추천"""

    # 예시로 첫 번째 리뷰 리스트만 사용해 추천
    user_review_text = " ".join(request.리뷰[0])
    df_result = recommender.recommend(user_review_text)

    return df_result.to_dict(orient="records")