from fastapi import FastAPI
from app.model import WeddingRecommender
from app.schema import SurveyRequest

app = FastAPI()
recommender = WeddingRecommender()

@app.post("/recommend")
def recommend(survey_data: SurveyRequest):
    result = recommender.recommend(survey_data.dict())
    return {"recommendations": result}