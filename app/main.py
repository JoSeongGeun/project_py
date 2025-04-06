from fastapi import FastAPI
from app.schema import SurveyInput
from app.model import WeddingRecommender
import os

if __name__ != "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
    
app = FastAPI()
recommender = WeddingRecommender()

@app.get("/")
def root():
    return {"message": "Hello, Wedding Recommender!"}

@app.post("/recommend")
def recommend(survey: SurveyInput):
    result = recommender.recommend(survey.dict())
    return {"recommendations": result}