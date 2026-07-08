"""RAG 파이프라인 — 금융소비자 투자 가이드북(PDF) 기반 지식 베이스.

data/financial_guide.pdf 가 있으면
그것을 인덱싱하고, 없으면 내장 투자 용어 사전으로 폴백하여
과제 데모가 항상 재현 가능하도록 설계했다.
"""
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

_FALLBACK_DOCS = [
    ("PER(주가수익비율)", "PER은 주가를 주당순이익(EPS)으로 나눈 값으로, 이익 대비 "
     "주가가 몇 배에 거래되는지를 나타낸다. 일반적으로 낮을수록 저평가로 해석하지만 "
     "업종 평균과 성장성을 함께 비교해야 한다."),
    ("PBR(주가순자산비율)", "PBR은 주가를 주당순자산(BPS)으로 나눈 값이다. 1배 미만이면 "
     "장부가치보다 낮게 거래됨을 의미하나, 자산의 질을 함께 평가해야 한다."),
    ("분산투자", "분산투자는 자산군·지역·업종을 나누어 투자함으로써 개별 종목 위험을 "
     "줄이는 전략이다. 상관관계가 낮은 자산을 섞을수록 효과가 크다."),
    ("리밸런싱", "리밸런싱은 시장 변동으로 달라진 자산 비중을 목표 비중으로 되돌리는 "
     "작업이다. 정기적(예: 분기) 또는 임계치 초과 시 수행한다."),
    ("ROE(자기자본이익률)", "ROE는 순이익을 자기자본으로 나눈 지표로, 주주 자본을 "
     "얼마나 효율적으로 활용해 이익을 내는지를 보여준다."),
]


@lru_cache(maxsize=1)
def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    pdf = Path(settings.RAG_PDF_PATH)

    docs = []
    if pdf.exists():
        from langchain_community.document_loaders import PyPDFLoader
        raw = PyPDFLoader(str(pdf)).load()
        # 텍스트가 실제로 있는 페이지만 채택 (스캔/이미지 PDF 방어)
        docs = [d for d in raw if d.page_content and d.page_content.strip()]
        if not docs:
            print(f"[RAG] '{pdf}'에서 텍스트를 추출하지 못해 내장 사전으로 폴백합니다.")

    if not docs:
        docs = [
            Document(page_content=f"{t}\n{c}",
                     metadata={"source": "builtin", "term": t})
            for t, c in _FALLBACK_DOCS
        ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    if not chunks:
        raise RuntimeError("RAG 인덱싱할 문서가 비어 있습니다.")

    vectordb = Chroma.from_documents(
        chunks, embeddings,
        collection_name="finance_guide",
        persist_directory=settings.VECTOR_DB_DIR,
    )
    return vectordb.as_retriever(search_kwargs={"k": 4})
