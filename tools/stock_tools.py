"""Tool 1: 증권사 추상 인터페이스 기반 시세/잔고 조회 도구.

@tool 함수는 get_broker()가 돌려주는 '어떤 증권사인지 모르는'
BrokerClient에만 의존한다. → KIS ↔ Toss 교체 시 이 파일 무수정.
"""
import json

from langchain_core.tools import tool

from brokers.factory import get_broker
from middleware.guards import broker_tool_guard


@tool
@broker_tool_guard
def get_stock_quote(symbol: str, market: str = "KR", exchange: str = "NAS") -> str:
    """국내/해외 주식의 실시간 현재가와 등락률을 조회한다.

    Args:
        symbol: 국내는 6자리 종목코드(예: 005930), 해외는 티커(예: AAPL)
        market: "KR"(국내) 또는 "US"(해외)
        exchange: 해외일 때 거래소 코드 (NAS=나스닥, NYS=뉴욕, AMS=아멕스)
    """
    broker = get_broker()
    quote = (broker.get_domestic_quote(symbol) if market.upper() == "KR"
             else broker.get_overseas_quote(symbol, exchange))
    return json.dumps(quote.to_dict(), ensure_ascii=False)


@tool
@broker_tool_guard
def get_account_balance() -> str:
    """사용자의 증권 계좌 보유종목, 수량, 평단가, 평가손익률을 조회한다."""
    positions = get_broker().get_balance()
    if not positions:
        return "보유 중인 종목이 없습니다."
    return json.dumps([p.to_dict() for p in positions], ensure_ascii=False)
