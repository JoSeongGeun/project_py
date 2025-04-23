# 💍 Plan My Wedding - 예식장 추천 시스템

> 자연어 리뷰 + 수치 기반 조건을 통해 나에게 딱 맞는 예식장을 추천해주는 AI 기반 시스템입니다.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.0+-green)](https://docs.pydantic.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Gensim](https://img.shields.io/badge/gensim-4.3.0+-purple)](https://radimrehurek.com/gensim/)
[![Deployed on Render](https://img.shields.io/badge/Hosted%20on-Render-430098?logo=render)](https://render.com)

---

## 🚀 프로젝트 소개

자연어 리뷰 데이터를 Word2Vec 기반 임베딩으로 처리하고, 사용자의 예산·수용인원·주차장 수 등의 조건에 따라 최적의 예식장을 추천해주는 FastAPI 기반의 웹 서비스입니다.

---

## 🛠️ 사용 기술 스택

### 🚀 Backend
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-121212?style=for-the-badge&logo=uvicorn&logoColor=white)

### 🤖 ML / NLP
![Gensim](https://img.shields.io/badge/Gensim-FFD700?style=for-the-badge&logo=gensim&logoColor=black)
![Word2Vec](https://img.shields.io/badge/Word2Vec-339933?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### 📊 Data
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### ☁️ Infra / DevOps
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![Schedule](https://img.shields.io/badge/Schedule-FF6F00?style=for-the-badge)
![Threading](https://img.shields.io/badge/Threading-4CAF50?style=for-the-badge)

### 🧩 ETC
![Pydantic](https://img.shields.io/badge/Pydantic-0865A6?style=for-the-badge)
![Requests](https://img.shields.io/badge/Requests-2A6EBB?style=for-the-badge&logo=python&logoColor=white)


---

## 📁 프로젝트 구조

```
project_py/
├── app/
│   ├── main.py           # FastAPI 진입점 및 라우팅
│   ├── model.py          # 추천 알고리즘 로직 (Word2Vec, 유사도 계산)
│   ├── render.py         # Render용 keep-alive 스케줄러
│   ├── schema.py         # Pydantic 기반 요청 데이터 모델 정의
│   └── utils.py          # Word2Vec 벡터 관련 유틸 함수
├── data/
│   └── data.csv          # 예식장 데이터셋
├── model/
│   └── word2vec.model    # 학습된 Word2Vec 모델 파일
├── .gitignore            # Git에서 무시할 파일 목록
├── Procfile              # Render 서버 실행 명령
├── render.yaml           # Render 배포 설정 파일
├── requirements.txt      # Python 패키지 의존성 목록
```


---


## 🌐 API 엔드포인트

| Method | Endpoint        | Description                     |
|--------|------------------|---------------------------------|
| GET    | `/ping`          | 헬스 체크 (pong 반환)           |
| GET    | `/`              | API 소개 메시지 반환            |
| POST   | `/recommend`     | 사용자 입력 기반 예식장 추천    |


---


## 🧪 추천 방식 요약

1. 사용자 리뷰 키워드 → Word2Vec 평균 벡터화
2. 모든 예식장 문서 → Word2Vec 벡터화
3. Cosine Similarity 계산 (리뷰 유사도)
4. 대관료, 식대, 수용인원 등 수치 정보 유사도 계산
5. 가중치 적용 후 `total_sim`로 정렬 → Top 5 추천


---


## 📦 설치 및 실행 방법


```
git clone https://github.com/yourusername/plan-my-wedding.git
cd plan-my-wedding
pip install -r requirements.txt
uvicorn app.main:app --reload
🔁 Keep-Alive 기능 (Render 호스팅용)
Render 무료 플랜에서 앱이 슬립되는 문제를 방지하기 위해 14분마다 /ping 엔드포인트로 자동 요청을 보내는 스케줄러 내장.

```

---

## PlanMyWedding 메인 github

https://github.com/yugwangmyeong/PlanMyWedding/tree/main

---
