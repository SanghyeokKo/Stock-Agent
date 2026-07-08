"""CLI 데모 — thread_id 기반 멀티턴 대화."""
import uuid

from graph.builder import build_graph


def main():
    app = build_graph()
    config = {"configurable": {"thread_id": f"user-{uuid.uuid4().hex[:8]}"}}

    print("📈 주식 투자 가이드 에이전트 (종료: exit)")
    print(f"   thread_id = {config['configurable']['thread_id']}\n")

    while True:
        try:
            query = input("👤 You  : ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"exit", "quit"}:
            break

        result = app.invoke({"messages": [("user", query)]}, config)
        print(f"\n🤖 Agent: {result['messages'][-1].content}")
        if result.get("structured_output"):
            print(f"\n🧾 structured_output(state): {result['structured_output']}")
        print()


if __name__ == "__main__":
    main()
