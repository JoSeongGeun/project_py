from fastapi import FastAPI
from app.schema import SurveyInput
from app.model import WeddingRecommender

app = FastAPI()
recommender = WeddingRecommender()

@app.get("/")
def root():
    return {"message": "Hello, Wedding Recommender!"}

@app.post("/recommend")
def recommend(survey: SurveyInput):
    result = recommender.recommend(survey.dict())
    return {"recommendations": result}