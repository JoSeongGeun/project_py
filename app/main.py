from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ✅ FastAPI용 CORS 미들웨어
from app.model import WeddingRecommender
from app.schema import SurveyRequest

app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 로컬 개발 환경 React 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = WeddingRecommender()

@app.post("/recommend")
def recommend(survey_data: SurveyRequest):
    result = recommender.recommend(survey_data.dict())
    return {"recommendations": result}
