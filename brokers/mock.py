"""Mock/yfinance 어댑터 — KIS API 키 발급 대기 중에도 전체 그래프를
끊김 없이 디버깅할 수 있도록 하는 개발용 구현체.

yfinance 호출이 실패하면 정적 Mock 데이터로 한 번 더 폴백하므로
네트워크가 막힌 환경(CI 등)에서도 그래프 흐름 테스트가 가능하다.
"""
from typing import List

from brokers.base import BrokerClient, Quote, Position

_STATIC = {
    "005930": ("삼성전자", 78400.0, 1.2, "KRW", "KOSPI"),
    "000660": ("SK하이닉스", 245000.0, -0.8, "KRW", "KOSPI"),
    "AAPL": ("Apple Inc.", 231.5, 0.6, "USD", "NASDAQ"),
    "TSLA": ("Tesla Inc.", 318.2, -1.4, "USD", "NASDAQ"),
}


class MockBrokerClient(BrokerClient):
    provider_name = "mock"

    def _yf_quote(self, ticker: str) -> tuple:
        import yfinance as yf  # 지연 임포트 — mock 전용 의존성
        info = yf.Ticker(ticker).fast_info
        price = float(info["last_price"])
        prev = float(info.get("previous_close") or price)
        rate = round((price - prev) / prev * 100, 2) if prev else 0.0
        return price, rate

    def get_domestic_quote(self, symbol: str) -> Quote:
        name, price, rate, cur, mkt = _STATIC.get(
            symbol, (symbol, 10000.0, 0.0, "KRW", "KOSPI"))
        try:
            price, rate = self._yf_quote(f"{symbol}.KS")
        except Exception:
            pass  # 정적 데이터 폴백
        return Quote(symbol, name, price, rate, cur, mkt, self.provider_name)

    def get_overseas_quote(self, symbol: str, exchange: str = "NAS") -> Quote:
        name, price, rate = _STATIC.get(symbol, (symbol, 100.0, 0.0, "", ""))[:3]
        try:
            price, rate = self._yf_quote(symbol)
        except Exception:
            pass  # 정적 데이터 폴백
        return Quote(symbol, name, price, rate, "USD", exchange,
                     self.provider_name)

    def get_balance(self) -> List[Position]:
        return [
            Position("005930", "삼성전자", 10, 72000, 78400, 8.89),
            Position("AAPL", "Apple Inc.", 5, 210.0, 231.5, 10.24),
        ]
