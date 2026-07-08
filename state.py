"""LangGraph에서 사용할 전역 상태(State) 정의."""
from typing import Annotated, Optional

from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """그래프 전체를 관통하는 상태 객체.

    - messages: add_messages 리듀서 덕분에 노드가 새 메시지를 '반환'하면
      기존 이력에 append(같은 id면 replace)된다. checkpointer와 결합해
      thread_id 단위의 멀티턴 대화 이력을 유지한다.
    - intent: 라우터 노드가 분류한 사용자 의도. 조건부 분기의 기준값.
    - tool_error: 미들웨어가 감지한 마지막 도구 오류(로깅/재시도 판단용).
    - structured_output: 포트폴리오 추천 노드가 생성한 Pydantic 기반 JSON.
    """

    messages: Annotated[list, add_messages]
    intent: Optional[str]            # "tool_use" | "knowledge" | "portfolio" | "chat"
    tool_error: Optional[str]
    structured_output: Optional[dict]
