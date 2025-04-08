from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.model import WeddingRecommender
from app.schema import SurveyRequest

app = FastAPI()

# ✅ CORS 미들웨어 - 명시적으로 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 💡 로컬 개발 React 주소만 허용
    allow_credentials=True,  # 💡 브라우저 요청에 권한 정보 포함 허용
    allow_methods=["GET", "POST", "OPTIONS"],  # 💡 OPTIONS 명시
    allow_headers=["*"],  # 💡 헤더도 허용
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
