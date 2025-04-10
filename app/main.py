from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.model import WeddingRecommender
from app.schema import SurveyRequest

app = FastAPI()


@app.middleware("http")
async def log_middleware(request: Request, call_next):
    print(f"🔥 들어온 요청: {request.method} {request.url}")
    response = await call_next(request)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)

recommender = WeddingRecommender()

# ✅ OPTIONS 핸들러 명시 (꼭 필요하진 않지만, 안전하게)
@app.options("/recommend")
async def preflight_handler():
    return JSONResponse(status_code=200, content={"message": "Preflight OK"})

@app.post("/recommend")
def recommend(survey_data: SurveyRequest):
    result = recommender.recommend(survey_data.dict())
    return {"recommendations": result}

@app.get("/ping")
def ping():
    return {"message": "pong"}
