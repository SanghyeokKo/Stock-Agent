"""한국투자증권 Open API 어댑터 (모의투자 도메인 기준).

- 도메인: https://openapivts.koreainvestment.com:29443
- 접근토큰은 24시간 유효 + 발급 자체에 분당 호출 제한이 있으므로
  로컬 파일(.kis_token.json)에 캐싱한다.
"""
import json
import time
from pathlib import Path
from typing import List

import requests

from config import settings
from brokers.base import BrokerClient, Quote, Position

_TOKEN_CACHE = Path(".kis_token.json")


class KISBrokerClient(BrokerClient):
    provider_name = "kis"

    def __init__(self):
        self.base = settings.KIS_BASE_URL.rstrip("/")
        self.app_key = settings.KIS_APP_KEY
        self.app_secret = settings.KIS_APP_SECRET
        if not (self.app_key and self.app_secret):
            raise RuntimeError(
                "KIS_APP_KEY / KIS_APP_SECRET 이 .env에 없습니다. "
                "키 발급 전이라면 BROKER_PROVIDER=mock 으로 실행하세요."
            )

    # ── 인증 ────────────────────────────────────────────────
    def _get_token(self) -> str:
        if _TOKEN_CACHE.exists():
            cached = json.loads(_TOKEN_CACHE.read_text())
            if time.time() < cached.get("expires_at", 0) - 600:  # 10분 여유
                return cached["access_token"]

        resp = requests.post(
            f"{self.base}/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data["access_token"]
        _TOKEN_CACHE.write_text(json.dumps({
            "access_token": token,
            "expires_at": time.time() + int(data.get("expires_in", 86400)),
        }))
        return token

    def _headers(self, tr_id: str) -> dict:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._get_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def _get(self, path: str, tr_id: str, params: dict) -> dict:
        resp = requests.get(
            f"{self.base}{path}", headers=self._headers(tr_id),
            params=params, timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("rt_cd") != "0":
            raise RuntimeError(f"KIS API 오류: {body.get('msg1', body)}")
        return body

    # ── 시세 ────────────────────────────────────────────────
    def get_domestic_quote(self, symbol: str) -> Quote:
        body = self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            tr_id="FHKST01010100",
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
        )
        out = body["output"]
        return Quote(
            symbol=symbol,
            name=out.get("bstp_kor_isnm", symbol),
            price=float(out["stck_prpr"]),
            change_rate=float(out["prdy_ctrt"]),
            currency="KRW",
            market="KOSPI/KOSDAQ",
            provider=self.provider_name,
        )

    def get_overseas_quote(self, symbol: str, exchange: str = "NAS") -> Quote:
        body = self._get(
            "/uapi/overseas-price/v1/quotations/price",
            tr_id="HHDFS00000300",
            params={"AUTH": "", "EXCD": exchange, "SYMB": symbol},
        )
        out = body["output"]
        return Quote(
            symbol=symbol,
            name=symbol,
            price=float(out["last"]),
            change_rate=float(out.get("rate", 0.0)),
            currency="USD",
            market=exchange,
            provider=self.provider_name,
        )

    # ── 잔고 ────────────────────────────────────────────────
    def get_balance(self) -> List[Position]:
        body = self._get(
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id="VTTC8434R",  # 모의투자용 TR_ID (실전: TTTC8434R)
            params={
                "CANO": settings.KIS_ACCOUNT_NO,
                "ACNT_PRDT_CD": settings.KIS_ACCOUNT_PRDT,
                "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02",
                "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "01",
                "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
            },
        )
        return [
            Position(
                symbol=row["pdno"],
                name=row["prdt_name"],
                quantity=float(row["hldg_qty"]),
                avg_price=float(row["pchs_avg_pric"]),
                current_price=float(row["prpr"]),
                pnl_rate=float(row["evlu_pfls_rt"]),
            )
            for row in body.get("output1", [])
            if float(row.get("hldg_qty", 0)) > 0
        ]
