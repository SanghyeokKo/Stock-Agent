"""전역 설정 — 모든 시크릿은 .env 파일에서 로드한다."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "gpt-4o-mini")  # 의도 분류(경량)
    MAIN_MODEL: str = os.getenv("MAIN_MODEL", "gpt-4o")           # 본 답변/구조화 출력

    # News
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Broker — "mock" | "kis" | (Phase 2) "toss"
    BROKER_PROVIDER: str = os.getenv("BROKER_PROVIDER", "mock").lower()

    # 한국투자증권 (모의투자 도메인)
    KIS_BASE_URL: str = os.getenv(
        "KIS_BASE_URL", "https://openapivts.koreainvestment.com:29443"
    )
    KIS_APP_KEY: str = os.getenv("KIS_APP_KEY", "")
    KIS_APP_SECRET: str = os.getenv("KIS_APP_SECRET", "")
    KIS_ACCOUNT_NO: str = os.getenv("KIS_ACCOUNT_NO", "")
    KIS_ACCOUNT_PRDT: str = os.getenv("KIS_ACCOUNT_PRDT", "01")

    # RAG
    RAG_PDF_PATH: str = os.getenv("RAG_PDF_PATH", "data/financial_guide.pdf")
    VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", ".chroma")


settings = Settings()
