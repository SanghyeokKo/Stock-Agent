"""Tool 2: Tavily 기반 최신 금융/종목 뉴스 검색 도구."""
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from middleware.guards import broker_tool_guard

_tavily = TavilySearch(max_results=5, topic="news")


@tool
@broker_tool_guard
def search_finance_news(query: str) -> str:
    """특정 종목/시장/경제 이슈에 대한 최신 뉴스를 검색한다.

    Args:
        query: 검색어 (예: "삼성전자 실적", "미국 금리 전망")
    """
    result = _tavily.invoke({"query": f"{query} 주식 뉴스"})
    items = result.get("results", []) if isinstance(result, dict) else result
    lines = [
        f"- {it.get('title')}: {it.get('content', '')[:150]} ({it.get('url')})"
        for it in items
    ]
    return "\n".join(lines) or "관련 뉴스를 찾지 못했습니다."
