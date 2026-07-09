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
    question = state["messages"][-1].content
    docs = get_retriever().invoke(question)
    # ── 검색 결과 로깅 (RAG 동작 확인용) ─────────────
    print("\n" + "=" * 60)
    print(f"[RAG DEBUG] 질문: {question}")
    print(f"[RAG DEBUG] 검색된 문서 {len(docs)}개:")
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "?")
        row_id = d.metadata.get("row_id", "?")
        preview = d.page_content[:150].replace("\n", " ")
        print(f"  [{i}] source={source} row_id={row_id}")
        print(f"      {preview}...")
    print("=" * 60 + "\n")

    context = "\n\n---\n\n".join(d.page_content for d in docs)
    response = _llm.invoke([
        ("system",
         "너는 투자 지식 어시스턴트다. 아래 [참고 자료]에 근거해서만 답하라. "
         "각 문장 끝에 어떤 자료 번호에서 왔는지 [자료 1], [자료 2] 형식으로 "
         "반드시 표기하라. 참고 자료에 없는 내용은 절대 추가하지 말고, "
         "관련 내용이 전혀 없으면 '자료에서 찾지 못했다'고만 답하라. "
         "답변 맨 끝에 '📚 출처: 한국은행 경제금융용어 700선 (row_id: X, Y, Z)' 형식으로 "
         "참고한 자료의 row_id를 명시하라."),
        ("user",
         f"[참고 자료]\n"
         + "\n\n---\n\n".join(
             f"[자료 {i+1}] (row_id={d.metadata.get('row_id')})\n{d.page_content}"
             for i, d in enumerate(docs)
         )
         + f"\n\n[질문]\n{question}"),
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
