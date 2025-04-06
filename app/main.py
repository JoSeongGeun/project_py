from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schema import SurveyRequest
from .model import WeddingRecommender

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트와 연동 시 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = WeddingRecommender()

@app.post("/recommend")
def recommend_wedding_hall(request: SurveyRequest):
    result = recommender.recommend(request.dict())
    return {"recommendations": result}
