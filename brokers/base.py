"""증권사 클라이언트 추상 인터페이스 (Adapter Pattern의 Target).

Phase 2에서 토스증권 API로 전환할 때, 이 인터페이스를 구현한
TossBrokerClient만 추가하면 상위 레이어(@tool, LangGraph 노드)는
단 한 줄도 수정할 필요가 없다.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Quote:
    """증권사에 무관한 표준 시세 DTO."""
    symbol: str
    name: str
    price: float
    change_rate: float   # 전일 대비 등락률(%)
    currency: str        # "KRW" | "USD"
    market: str          # "KOSPI" | "NASDAQ" | ...
    provider: str        # 데이터 출처 (kis | mock | toss)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Position:
    """증권사에 무관한 표준 보유종목 DTO."""
    symbol: str
    name: str
    quantity: float
    avg_price: float
    current_price: float
    pnl_rate: float      # 평가손익률(%)

    def to_dict(self) -> dict:
        return asdict(self)


class BrokerClient(ABC):
    """모든 증권사 어댑터가 반드시 구현해야 하는 계약(Contract)."""

    provider_name: str = "base"

    @abstractmethod
    def get_domestic_quote(self, symbol: str) -> Quote:
        """국내 주식 현재가 조회. symbol: 6자리 종목코드 (예: '005930')"""

    @abstractmethod
    def get_overseas_quote(self, symbol: str, exchange: str = "NAS") -> Quote:
        """해외 주식 현재가 조회. symbol: 티커 (예: 'AAPL'), exchange: NAS/NYS/AMS"""

    @abstractmethod
    def get_balance(self) -> List[Position]:
        """계좌 보유종목/잔고 조회."""
