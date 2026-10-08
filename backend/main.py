import gzip
import json
import os
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

INDEX_NAME = "cliffe-bot-v3"
NAMESPACE = "cliffe-v3"
EMBED_MODEL = "llama-text-embed-v2"

DENSE_TOP_K = 20
BM25_TOP_K = 20
FINAL_TOP_K = 7
RRF_K = 60

BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "data" / "cliffe_v2_corpus.json.gz"

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "of", "on", "or",
    "that", "the", "there", "to", "what", "when", "where", "which", "who",
    "why", "with", "you", "your",
}

pc = None
index = None
gemini_client = None
bm25 = None
corpus_rows = None
row_by_id = None
initialize_lock = threading.Lock()


def normalize_token(token: str) -> str:
    token = token.lower().strip()

    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"

    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token


def tokenize(text: str) -> list[str]:
    return [
        normalize_token(token)
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if token
    ]


def significant_query_tokens(question: str) -> list[str]:
    return [
        token
        for token in tokenize(question)
        if token not in STOP_WORDS
    ]


def embedding_values(item):
    if hasattr(item, "values"):
        return list(item.values)

    if isinstance(item, dict):
        return list(item["values"])

    raise TypeError(
        f"Unsupported embedding response item: {type(item)}"
    )


def initialize_rag():
    global pc, index, gemini_client, bm25, corpus_rows, row_by_id

    if index is not None and gemini_client is not None and bm25 is not None:
        return

    with initialize_lock:
        if index is not None and gemini_client is not None and bm25 is not None:
            return

        print("🧠 Initializing Cliffe AI V3 free hybrid RAG...")
        start = time.time()

        pinecone_key = os.getenv("PINECONE_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")

        if not pinecone_key:
            raise RuntimeError("PINECONE_API_KEY is missing.")

        if not google_key:
            raise RuntimeError("GOOGLE_API_KEY is missing.")

        if not CORPUS_PATH.exists():
            raise FileNotFoundError(
                f"Missing corpus: {CORPUS_PATH}. "
                "Commit backend/data/cliffe_v2_corpus.json.gz."
            )

        from google import genai
        from pinecone import Pinecone
        from rank_bm25 import BM25Okapi

        with gzip.open(
            CORPUS_PATH,
            "rt",
            encoding="utf-8",
        ) as handle:
            corpus_rows = json.load(handle)

        tokenized_corpus = [
            tokenize(row["text"])
            for row in corpus_rows
        ]

        bm25 = BM25Okapi(tokenized_corpus)
        row_by_id = {
            row["id"]: row
            for row in corpus_rows
        }

        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(INDEX_NAME)

        gemini_client = genai.Client(api_key=google_key)

        print(
            f"   Loaded {len(corpus_rows)} chunks for BM25."
        )
        print(
            f"✅ Cliffe AI V3 ready in "
            f"{time.time() - start:.2f}s"
        )


def dense_search(question: str) -> list[dict]:
    embedding_result = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[question],
        parameters={
            "input_type": "query",
            "truncate": "END",
        },
    )

    data = getattr(embedding_result, "data", embedding_result)
    query_vector = embedding_values(data[0])

    response = index.query(
        vector=query_vector,
        top_k=DENSE_TOP_K,
        namespace=NAMESPACE,
        include_metadata=True,
    )

    matches = getattr(response, "matches", None)

    if matches is None and isinstance(response, dict):
        matches = response.get("matches", [])

    results = []

    for rank, match in enumerate(matches or [], start=1):
        metadata = getattr(match, "metadata", None)

        if metadata is None and isinstance(match, dict):
            metadata = match.get("metadata", {})

        score = getattr(match, "score", None)

        if score is None and isinstance(match, dict):
            score = match.get("score")

        metadata = dict(metadata or {})

        results.append(
            {
                "id": metadata.get("chunk_id"),
                "text": metadata.get("text", ""),
                "source": metadata.get("source", ""),
                "title": metadata.get("title", ""),
                "section": metadata.get("section", ""),
                "unit": metadata.get("unit", ""),
                "dense_rank": rank,
                "dense_score": float(score or 0),
            }
        )

    return results


def bm25_search(question: str) -> list[dict]:
    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:BM25_TOP_K]

    results = []

    for rank, row_index in enumerate(ranked_indexes, start=1):
        if scores[row_index] <= 0:
            continue

        row = corpus_rows[row_index]

        results.append(
            {
                **row,
                "bm25_rank": rank,
                "bm25_score": float(scores[row_index]),
            }
        )

    return results


def hybrid_search(question: str):
    dense = dense_search(question)
    lexical = bm25_search(question)

    fused = {}

    for item in dense:
        key = item.get("id") or (
            f"{item.get('source', '')}|{item.get('text', '')[:250]}"
        )

        entry = fused.setdefault(
            key,
            {
                "doc": item,
                "score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
            },
        )

        entry["score"] += 1.0 / (
            RRF_K + item["dense_rank"]
        )
        entry["dense_rank"] = item["dense_rank"]

    for item in lexical:
        key = item["id"]

        entry = fused.setdefault(
            key,
            {
                "doc": item,
                "score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
            },
        )

        entry["score"] += 1.20 / (
            RRF_K + item["bm25_rank"]
        )
        entry["bm25_rank"] = item["bm25_rank"]

        if entry["dense_rank"] is not None:
            entry["doc"] = {
                **entry["doc"],
                **item,
            }

    important_tokens = significant_query_tokens(question)
    normalized_question = " ".join(important_tokens)

    for entry in fused.values():
        doc = entry["doc"]
        text = doc.get("text", "")
        text_tokens = set(tokenize(text))
        normalized_text = " ".join(tokenize(text))

        if important_tokens:
            overlap = sum(
                1
                for token in important_tokens
                if token in text_tokens
            )

            coverage = overlap / len(important_tokens)

            entry["score"] += 0.020 * coverage

            if coverage == 1.0:
                entry["score"] += 0.025

            if (
                len(important_tokens) >= 2
                and normalized_question
                and normalized_question in normalized_text
            ):
                entry["score"] += 0.035

    ranked = sorted(
        fused.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    selected = []
    per_source = {}

    for entry in ranked:
        doc = dict(entry["doc"])
        source = doc.get("source", "unknown")

        if per_source.get(source, 0) >= 2:
            continue

        doc["dense_rank"] = entry["dense_rank"]
        doc["bm25_rank"] = entry["bm25_rank"]
        doc["hybrid_score"] = round(
            entry["score"],
            6,
        )

        selected.append(doc)
        per_source[source] = (
            per_source.get(source, 0) + 1
        )

        if len(selected) >= FINAL_TOP_K:
            break

    return selected, dense, lexical


def build_context(docs: list[dict]) -> str:
    blocks = []

    for number, doc in enumerate(docs, start=1):
        blocks.append(
            f"[SOURCE {number}]\n"
            f"Title: {doc.get('title', '')}\n"
            f"Unit: {doc.get('unit', '')}\n"
            f"Section: {doc.get('section', '')}\n"
            f"URL: {doc.get('source', '')}\n\n"
            f"{doc.get('text', '')}"
        )

    return "\n\n---\n\n".join(blocks)


def answer_with_gemini(
    question: str,
    docs: list[dict],
) -> str:
    from google.genai import types

    context = build_context(docs)

    prompt = f"""You are Cliffe AI, a student assistant for Youngstown State University's Cliffe College of Creative Arts.

Answer ONLY from the provided Cliffe website context.

Rules:
- Give the direct answer first.
- Be accurate and concise.
- When asked about a person, include their title or role when available.
- For broad questions such as scholarships, summarize relevant options across the retrieved Cliffe units instead of mentioning only one page.
- Never invent facts.
- Do not use outside knowledge.
- If the context is insufficient, say exactly: "I cannot find that info on the Cliffe website."
- Do not mention embeddings, Pinecone, BM25, chunks, RRF, or internal retrieval systems.

Question:
{question}

Cliffe website context:
{context}
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
        ),
    )

    return (
        response.text
        or "I cannot find that info on the Cliffe website."
    )


def unique_sources(
    docs: list[dict],
    limit: int = 5,
) -> list[str]:
    sources = []
    seen = set()

    for doc in docs:
        source = doc.get("source")

        if source and source not in seen:
            sources.append(source)
            seen.add(source)

        if len(sources) >= limit:
            break

    return sources


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "Cliffe AI",
        "rag_version": "v3-free-hybrid",
        "index": INDEX_NAME,
        "namespace": NAMESPACE,
        "embedding_model": EMBED_MODEL,
        "rag_ready": (
            index is not None
            and bm25 is not None
        ),
    }


@app.post("/ask")
def ask(q: Query):
    request_start = time.time()
    question = q.question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "response_time": 0,
        }

    print("\n" + "-" * 72)
    print(f"📝 Question: {question}")

    try:
        init_start = time.time()
        initialize_rag()
        init_elapsed = time.time() - init_start

        retrieval_start = time.time()
        docs, dense, lexical = hybrid_search(question)
        retrieval_elapsed = time.time() - retrieval_start

        print(
            f"🔎 Dense: {len(dense)} | "
            f"BM25: {len(lexical)} | "
            f"Final: {len(docs)}"
        )

        for number, doc in enumerate(docs, start=1):
            print(f"\n📄 RESULT {number}")
            print(f"🔗 {doc.get('source', 'Unknown')}")
            print(
                f"   dense_rank={doc.get('dense_rank')} "
                f"bm25_rank={doc.get('bm25_rank')} "
                f"hybrid={doc.get('hybrid_score')}"
            )
            print(
                doc.get("text", "")[:500]
                .replace("\n", " ")
            )

        generation_start = time.time()
        answer = answer_with_gemini(
            question,
            docs,
        )
        generation_elapsed = (
            time.time() - generation_start
        )

        total_elapsed = (
            time.time() - request_start
        )

        print(
            f"\n⚙️ Initialization: "
            f"{init_elapsed:.2f}s"
        )
        print(
            f"🔎 Retrieval: "
            f"{retrieval_elapsed:.2f}s"
        )
        print(
            f"🤖 Generation: "
            f"{generation_elapsed:.2f}s"
        )
        print(
            f"⏱️ TOTAL: "
            f"{total_elapsed:.2f}s"
        )
        print("-" * 72)

        return {
            "answer": answer,
            "sources": unique_sources(docs),
            "response_time": round(
                total_elapsed,
                2,
            ),
            "timing": {
                "initialization": round(
                    init_elapsed,
                    2,
                ),
                "retrieval": round(
                    retrieval_elapsed,
                    2,
                ),
                "generation": round(
                    generation_elapsed,
                    2,
                ),
            },
        }

    except Exception as exc:
        total_elapsed = (
            time.time() - request_start
        )

        print(
            f"❌ API ERROR after "
            f"{total_elapsed:.2f}s: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "answer": "System Error. Please try again.",
            "sources": [],
            "response_time": round(
                total_elapsed,
                2,
            ),
        }
