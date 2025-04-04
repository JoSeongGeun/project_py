from fastapi import FastAPI
from app.model import WeddingHallRecommender
from app.database import fetch_all_halls

app = FastAPI()
recommender = WeddingHallRecommender()

@app.get("/")
def home():
    return {"message": "예식장 추천 시스템 API"}

@app.get("/wedding-halls")
def get_wedding_halls():
    """저장된 모든 예식장 데이터 조회"""
    return fetch_all_halls().to_dict(orient="records")

@app.get("/recommend")
def recommend_wedding_hall(user_review: str):
    """사용자 입력 기반 예식장 추천"""
    recommendations = recommender.recommend(user_review)
    return recommendations.to_dict(orient="records")