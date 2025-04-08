from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.model import WeddingRecommender
from app.schema import SurveyRequest
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = WeddingRecommender()

@app.options("/recommend")  # ✅ OPTIONS 메서드 허용
async def options_handler():
    return JSONResponse(content={"message": "CORS preflight accepted"}, status_code=200)

@app.post("/recommend")
def recommend(survey_data: SurveyRequest):
    result = recommender.recommend(survey_data.dict())
    return {"recommendations": result}
