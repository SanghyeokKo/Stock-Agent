"""미들웨어 계층.

1) broker_tool_guard  : 도구 실행을 감싸 API 실패를 '예외'가 아닌
   '구조화된 오류 문자열'로 변환 → 그래프가 죽지 않고 LLM이
   사용자에게 상황을 설명하거나 다른 도구로 우회하게 만든다.
2) disclaimer_node    : 최종 답변 직전에 반드시 통과하는 노드로,
   금융 투자 위험 안내 문구를 강제 삽입한다. (add_messages의
   same-id replace 특성을 이용해 마지막 AI 메시지를 교체)
"""
import functools

import requests
from langchain_core.messages import AIMessage

from state import AgentState

DISCLAIMER = (
    "⚠️ 투자 유의사항: 본 답변은 정보 제공 목적이며 특정 종목의 매수·매도 "
    "권유가 아닙니다. 모든 투자의 책임은 투자자 본인에게 있으며, 원금 손실이 "
    "발생할 수 있습니다."
)


def broker_tool_guard(func):
    """API 호출 실패 시 작동하는 에러 핸들링 미들웨어(데코레이터)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.Timeout:
            return "[API_ERROR] 증권사 API 응답 시간 초과. 잠시 후 다시 시도해 주세요."
        except requests.exceptions.RequestException as e:
            return (f"[API_ERROR] 증권사 API 호출 실패({e}). "
                    "네트워크 상태 확인 또는 BROKER_PROVIDER=mock 전환을 안내하세요.")
        except Exception as e:  # noqa: BLE001 — 그래프 생존이 최우선
            return f"[TOOL_ERROR] {type(e).__name__}: {e}"

    return wrapper


def disclaimer_node(state: AgentState) -> dict:
    """모든 종단 경로가 END 직전에 통과하는 가드레일 노드."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and DISCLAIMER not in (last.content or ""):
        patched = AIMessage(
            content=f"{last.content}\n\n---\n{DISCLAIMER}",
            id=last.id,  # 같은 id → add_messages가 append가 아닌 replace 수행
        )
        return {"messages": [patched]}
    return {}
