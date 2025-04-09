import schedule
import time
import datetime
import requests

PING_URL = "https://project-py-5lx0.onrender.com/ping"

def send_ping():
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] Sending ping to {PING_URL}")
    try:
        res = requests.get(PING_URL)
        print(f"일어났습니다")
    except Exception as e:
        print(f"안일어났습니다")

# 14분 간격으로 등록 (업무 시간 내에서만 실행되도록)
schedule.every(14).minutes.do(lambda: (
    send_ping() if 9 <= datetime.datetime.now().hour < 18 else print("퇴근시간입니다.")
))

print("출근했습니다")

while True:
    schedule.run_pending()
    time.sleep(1)