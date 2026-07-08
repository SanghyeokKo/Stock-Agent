"""StateGraph 조립 — 노드/엣지 배선과 checkpointer(Memory) 설정."""
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from graph.router import route_node, route_selector
from graph.nodes import agent_node, rag_node, portfolio_node, chat_node, TOOLS
from middleware.guards import disclaimer_node


def build_graph():
    builder = StateGraph(AgentState)

    # 노드 등록
    builder.add_node("router", route_node)
    builder.add_node("agent", agent_node)
    # handle_tool_errors=True: 도구 내부 예외도 ToolMessage로 변환(2차 방어선)
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    builder.add_node("rag", rag_node)
    builder.add_node("portfolio", portfolio_node)
    builder.add_node("chat", chat_node)
    builder.add_node("disclaimer", disclaimer_node)

    # 엣지 배선
    builder.add_edge(START, "router")

    # [조건부 분기 1] 사용자 의도 기반 라우팅
    builder.add_conditional_edges(
        "router", route_selector,
        {"tool_use": "agent", "knowledge": "rag",
         "portfolio": "portfolio", "chat": "chat"},
    )

    # [조건부 분기 2] ReAct 루프: tool_call 있으면 tools, 없으면 종료 경로
    builder.add_conditional_edges(
        "agent", tools_condition,
        {"tools": "tools", "__end__": "disclaimer"},
    )
    builder.add_edge("tools", "agent")  # 도구 결과를 들고 다시 사고

    # 모든 종단 경로는 Disclaimer 미들웨어를 강제 통과
    builder.add_edge("rag", "disclaimer")
    builder.add_edge("portfolio", "disclaimer")
    builder.add_edge("chat", "disclaimer")
    builder.add_edge("disclaimer", END)

    # 멀티턴 대화 이력 유지 — thread_id 단위 체크포인트
    return builder.compile(checkpointer=MemorySaver())
