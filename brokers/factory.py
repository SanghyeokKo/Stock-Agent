"""증권사 어댑터 팩토리 — '단 한 줄'로 증권사를 갈아끼우는 지점.

Phase 2에서 토스증권 API 키가 발급되면:
  1) brokers/toss.py 에 BrokerClient를 구현한 TossBrokerClient 작성
  2) 아래 _REGISTRY에 "toss": TossBrokerClient  ← 이 한 줄 추가
  3) .env 의 BROKER_PROVIDER=toss 로 변경
상위의 @tool 함수, LangGraph 노드, 프롬프트는 전혀 수정하지 않는다.
"""
from functools import lru_cache
from typing import Dict, Type

from config import settings
from brokers.base import BrokerClient
from brokers.mock import MockBrokerClient
from brokers.kis import KISBrokerClient

_REGISTRY: Dict[str, Type[BrokerClient]] = {
    "mock": MockBrokerClient,
    "kis": KISBrokerClient,
    # "toss": TossBrokerClient,   # ← Phase 2: 이 한 줄로 교체 완료
}


@lru_cache(maxsize=1)
def get_broker() -> BrokerClient:
    provider = settings.BROKER_PROVIDER
    if provider not in _REGISTRY:
        raise ValueError(
            f"알 수 없는 BROKER_PROVIDER='{provider}'. "
            f"사용 가능: {list(_REGISTRY)}"
        )
    return _REGISTRY[provider]()
