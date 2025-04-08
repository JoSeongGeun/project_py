from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ✅ FastAPI용 CORS 미들웨어
from app.model import WeddingRecommender
from app.schema import SurveyRequest

app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # 자격 정보 안 쓸 거면 이렇게
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = WeddingRecommender()

@app.post("/recommend")
def recommend(survey_data: SurveyRequest):
    result = recommender.recommend(survey_data.dict())
    return {"recommendations": result}
