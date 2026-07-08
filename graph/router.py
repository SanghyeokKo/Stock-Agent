"""의도 분류 라우터 — LangGraph 조건부 분기(Conditional Edge)의 기준.

gpt-4o-mini(경량 모델)로 비용을 절약하면서, Strict 구조화 출력으로
분류 결과가 항상 Literal 집합 안에 들어오도록 강제한다.
"""
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import settings
from state import AgentState


class RouteDecision(BaseModel):
    intent: Literal["tool_use", "knowledge", "portfolio", "chat"] = Field(
        description=(
            "tool_use: 실시간 시세/뉴스/계좌잔고 조회가 필요한 질문 | "
            "knowledge: 투자 용어·지표·제도 등 지식 설명 질문(RAG) | "
            "portfolio: 포트폴리오 추천·리밸런싱 요청(구조화 JSON 응답) | "
            "chat: 그 외 일반 대화"
        )
    )


_router_llm = ChatOpenAI(model=settings.ROUTER_MODEL, temperature=0)
_classifier = _router_llm.with_structured_output(RouteDecision)

_ROUTER_PROMPT = (
    "너는 주식 투자 에이전트의 라우터다. 대화 맥락을 보고 마지막 사용자 "
    "질문의 의도를 하나로 분류하라.\n\n"
    "분류 예시:\n"
    "- '삼성전자 지금 얼마야?', 'AAPL 현재가', '내 계좌 잔고 보여줘', "
    "'테슬라 관련 최근 뉴스' → tool_use (실시간 데이터 필요)\n"
    "- 'PER이 뭐야?', '분산투자란?', 'ROE 계산법 알려줘' → knowledge (개념/용어 설명)\n"
    "- '내 자산을 리밸런싱해줘', '어떤 종목에 얼마씩 넣을까?' → portfolio (구조화 추천)\n"
    "- '안녕', '고마워' → chat\n\n"
    "종목명이 나왔다고 무조건 knowledge가 아니다. '현재가/가격/얼마/시세/잔고/뉴스' "
    "같은 실시간성 키워드가 있으면 tool_use다."
)


def route_node(state: AgentState) -> dict:
    """상태에 intent를 기록하는 노드 (분기 자체는 selector가 수행)."""
    decision: RouteDecision = _classifier.invoke(
        [("system", _ROUTER_PROMPT)] + state["messages"][-6:]  # 최근 맥락만
    )
    return {"intent": decision.intent}


def route_selector(state: AgentState) -> str:
    """조건부 엣지에서 다음 노드 이름을 반환하는 함수."""
    return state.get("intent") or "chat"
