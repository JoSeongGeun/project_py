from pydantic import BaseModel
from typing import List

class SurveyRequest(BaseModel):
    리뷰: List[List[str]]
    대관료: int
    식대: int
    최소수용인원: int
    최대수용인원: int
    주차장: int
    rental_fee_weight: float
    food_price_weight: float
    mini_hc_weight: float
    limit_hc_weight: float
    car_park_weight: float