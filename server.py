"""FastAPI 웹 서버 — LangGraph 그래프를 HTTP 엔드포인트로 노출.

- POST /api/chat : 사용자 메시지를 그래프에 전달하고 응답 반환
- GET  /        : web/index.html 정적 프론트 서빙

기존 main.py(CLI)는 개발/디버깅용으로 그대로 유지된다.
"""
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graph.builder import build_graph

app = FastAPI(title="Stock Agent API")

# 그래프는 서버 프로세스 생애 동안 하나만 유지 (MemorySaver도 함께 살아있음)
_graph = build_graph()


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None  # 없으면 서버가 새로 발급


class ChatResponse(BaseModel):
    thread_id: str
    intent: Optional[str]
    reply: str
    structured_output: Optional[dict] = None


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="빈 메시지는 보낼 수 없습니다.")

    thread_id = req.thread_id or f"web-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = _graph.invoke({"messages": [("user", req.message)]}, config)
    except Exception as e:  # noqa: BLE001 — 어떤 예외든 사용자에게 JSON으로 전달
        raise HTTPException(status_code=500, detail=f"에이전트 실행 실패: {e}")

    return ChatResponse(
        thread_id=thread_id,
        intent=result.get("intent"),
        reply=result["messages"][-1].content,
        # 이 턴이 portfolio 노드를 실제로 지났을 때만 노출 (State 누출 방지)
        structured_output=(
            result.get("structured_output")
            if result.get("intent") == "portfolio" else None
        ),
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


# 정적 프론트 서빙 — /web/* 는 파일, / 는 index.html
_WEB_DIR = Path(__file__).parent / "web"
app.mount("/web", StaticFiles(directory=str(_WEB_DIR)), name="web")


@app.get("/")
def index():
    return FileResponse(_WEB_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
