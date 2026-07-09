"""RAG 파이프라인 — FinShibainu QA 데이터셋 기반.

데이터셋 aiqwe/FinShibainu (Apache-2.0)의 qa subset을 지식 베이스로 활용한다.
원본 데이터셋은 DPO 학습용 선호도 데이터로, preference 컬럼이 B인 경우
answer_B가 더 우수한 답변으로 검증된 상태이므로 이를 RAG의 정답 텍스트로 사용한다.
"""
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import settings

# 사용할 원본 자료 필터 (여러 개 지정 가능)
TARGET_REFERENCES = [
    "한국은행_경제금융_용어_700선",
    # 필요하면 다른 소스도 추가:
    # "시사경제용어사전",
    # "금융감독_용어사전",
]

# 데모용 최대 문서 수 (None이면 전체 사용)
MAX_ROWS = None


def _load_finshibainu_docs() -> list[Document]:
    """FinShibainu qa subset에서 우수 답변을 지식 문서로 변환."""
    print("[RAG] FinShibainu qa 데이터셋 로드 중 (첫 실행은 다운로드 몇 분 소요)...")
    ds = load_dataset("aiqwe/FinShibainu", "qa", split="train")
    print(f"[RAG] 전체 {len(ds)}행 로드됨")

    docs = []
    for i, row in enumerate(ds):
        # 1) 원본 자료 필터
        if TARGET_REFERENCES and row["reference"] not in TARGET_REFERENCES:
            continue

        # 2) 선호 답변만 채택 (실질적 "정답")
        preferred = row["preference"]           # "A" or "B"
        answer_col = f"answer_{preferred}"
        answer = row.get(answer_col, "").strip()
        question = row["question"].strip()
        if not (question and answer):
            continue

        # 3) 검색 대상 문서 형태로 조립
        content = f"[질문] {question}\n[해설] {answer}"
        docs.append(Document(
            page_content=content,
            metadata={
                "source": row["reference"],
                "row_id": i,
                "quality": row.get("value", 0),
            },
        ))

        if MAX_ROWS and len(docs) >= MAX_ROWS:
            break

    print(f"[RAG] 필터링 후 {len(docs)}개 QA 문서 확보")
    return docs


@lru_cache(maxsize=1)
def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_dir = Path(settings.VECTOR_DB_DIR)
    already_indexed = persist_dir.exists() and any(persist_dir.iterdir())

    docs = _load_finshibainu_docs()  # BM25는 항상 필요하므로 로드

    if already_indexed:
        print(f"[RAG] 기존 Chroma 재사용: {persist_dir}")
        vectordb = Chroma(
            collection_name="finshibainu_qa",
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
    else:
        vectordb = Chroma.from_documents(
            docs, embeddings,
            collection_name="finshibainu_qa",
            persist_directory=str(persist_dir),
        )
        print(f"[RAG] Chroma에 {len(docs)}개 문서 최초 인덱싱 완료")

    # 하이브리드 검색 (영어 약어 매칭 + 의미 검색)
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 4
    return EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.5, 0.5],
    )