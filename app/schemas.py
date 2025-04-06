from typing import List
from pydantic import BaseModel

class RecommendRequest(BaseModel):
    리뷰: List[List[str]]
    대관료: int
    식대: int
    최소수용인원: int
    최대수용인원: int
    주차장: int