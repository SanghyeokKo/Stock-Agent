"""구조화 출력(OutputParser)용 Pydantic 스키마.

OpenAI Strict JSON Mode(method="json_schema", strict=True)와 함께 사용되어
LLM 출력이 스키마와 100% 일치함을 보장한다.
"""
from typing import List

from pydantic import BaseModel, Field


class StockRecommendation(BaseModel):
    """과제 요구 스키마: {"종목명": str, "추천비중": int, "투자포인트": str}"""

    종목명: str = Field(description="추천 종목의 정식 명칭 (예: 삼성전자)")
    추천비중: int = Field(ge=0, le=100, description="포트폴리오 내 추천 비중(%)")
    투자포인트: str = Field(description="해당 종목을 추천하는 핵심 근거 1~2문장")


class PortfolioProposal(BaseModel):
    """리밸런싱/추천 요청에 대한 최종 구조화 응답."""

    추천목록: List[StockRecommendation] = Field(description="추천 종목 목록 (비중 합계 100 이하)")
    총평: str = Field(description="포트폴리오 전체에 대한 종합 코멘트")
