"""LangGraph 노드 구현 (Phase 1)."""
import json

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from state import AgentState
from schemas import PortfolioProposal
from tools.stock_tools import get_stock_quote, get_account_balance
from tools.news_tools import search_finance_news
from rag.pipeline import get_retriever

TOOLS = [get_stock_quote, get_account_balance, search_finance_news]

_llm = ChatOpenAI(model=settings.MAIN_MODEL, temperature=0)
_llm_with_tools = _llm.bind_tools(TOOLS)

# OpenAI Strict JSON Mode — 스키마 100% 준수 보장
_structured_llm = _llm.with_structured_output(
    PortfolioProposal, method="json_schema", strict=True
)

_AGENT_SYSTEM = SystemMessage(content=(
    "너는 국내/해외 주식 통합 투자 가이드 에이전트다. 실시간 시세·잔고는 "
    "get_stock_quote/get_account_balance, 최신 뉴스는 search_finance_news "
    "도구를 자율적으로 선택해 사용하라. 도구 결과에 [API_ERROR]나 "
    "[TOOL_ERROR]가 포함되면 오류 상황을 사용자에게 정중히 설명하고 "
    "가능한 대안을 안내하라. 답변은 한국어로, 수치의 출처(provider)를 명시하라."
))


def agent_node(state: AgentState) -> dict:
    """자율적 도구 선택(Tool Calling) 노드 — ReAct 루프의 두뇌."""
    response = _llm_with_tools.invoke([_AGENT_SYSTEM] + state["messages"])
    return {"messages": [response]}


def rag_node(state: AgentState) -> dict:
    """지식 질문(RAG) 노드 — 투자 가이드북 기반 근거 답변."""
    question = state["messages"][-1].content
    docs = get_retriever().invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    response = _llm.invoke([
        ("system", "아래 [투자 가이드 자료]에 근거해서만 답하고, 자료에 없으면 "
                   "모른다고 말하라. 답변 끝에 참고한 자료 요지를 한 줄로 요약하라."),
        ("user", f"[투자 가이드 자료]\n{context}\n\n[질문]\n{question}"),
    ])
    return {"messages": [response]}


def portfolio_node(state: AgentState) -> dict:
    """구조화 출력(OutputParser) 노드 — Pydantic + Strict JSON Mode."""
    proposal: PortfolioProposal = _structured_llm.invoke([
        ("system", "너는 보수적인 포트폴리오 어드바이저다. 대화 맥락을 바탕으로 "
                   "종목별 추천 비중(합계 100 이하)과 투자포인트를 제시하라."),
        *state["messages"],
    ])
    payload = proposal.model_dump()
    pretty = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "structured_output": payload,
        "messages": [AIMessage(content=f"요청하신 포트폴리오 제안입니다.\n```json\n{pretty}\n```")],
    }


def chat_node(state: AgentState) -> dict:
    """일반 대화 노드."""
    response = _llm.invoke(
        [("system", "너는 친절한 주식 투자 가이드 에이전트다.")] + state["messages"]
    )
    return {"messages": [response]}
