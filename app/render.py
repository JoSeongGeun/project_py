from contextlib import asynccontextmanager
import threading
import schedule
import time
import requests
import datetime


def send_ping():
    now = datetime.datetime.now().strftime("%H:%M:%S")
    url = "https://project-py-5lx0.onrender.com/ping"
    try:
        print(f"[{now}] Sending ping to {url}")
        res = requests.get(url)
        print(f"[{now}] Ping OK: {res.status_code}")
    except Exception as e:
        print(f"[{now}] Ping failed: {e}")

def run_scheduler():
    schedule.every(14).minutes.do(send_ping)
    while True:
        schedule.run_pending()
        time.sleep(1)

# ✅ lifespan 이벤트로 scheduler 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=run_scheduler, daemon=True).start()
    print("✅ keep-alive scheduler started.")
    yield
    print("🛑 App shutting down...")

# ✅ FastAPI 객체에 lifespan 연결
app = FastAPI(lifespan=lifespan)

# ✅ 기존 라우터들 유지
class QueryInput(BaseModel):
    query: Union[str, list[str]]

@app.post("/recommend")
def recommend_weddingHall(input: QueryInput):
    try:
        query = input.query if isinstance(input.query, list) else [k.strip() for k in input.query.split(",")]
        result = get_weddingHall_recommendations(query)
        return result["recommendations"]
    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

@app.get("/")
async def index():
    return {
        "message": "Plan My Wedding 추천 API가 정상 작동 중입니다.",
        "example_query": "드메르, 까사디루체, 위더스 입니다."
    }

@app.get("/ping")
def ping():
    return {"message": "pong"}