import gzip
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException, PineconeApiException

load_dotenv()

INDEX_NAME = "cliffe-bot-v3"
NAMESPACE = "cliffe-v3"
EMBED_MODEL = "llama-text-embed-v2"
EMBED_DIMENSION = 1024
EMBED_BATCH_SIZE = 32

# Keeps us comfortably under Pinecone Starter's tokens-per-minute limit.
SECONDS_BETWEEN_EMBED_CALLS = 4
RATE_LIMIT_WAIT_SECONDS = 65

BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "data" / "cliffe_v2_corpus.json.gz"


def embedding_values(item):
    if hasattr(item, "values"):
        return list(item.values)
    if isinstance(item, dict):
        return list(item["values"])
    raise TypeError(f"Unsupported embedding item type: {type(item)}")


def is_index_ready(description) -> bool:
    status = getattr(description, "status", None)

    if isinstance(status, dict):
        return bool(status.get("ready"))

    if status is not None and hasattr(status, "ready"):
        return bool(status.ready)

    if isinstance(description, dict):
        return bool(description.get("status", {}).get("ready"))

    return False


def existing_ids_for_batch(index, ids):
    """Return IDs already uploaded so reruns resume instead of restarting."""
    try:
        response = index.fetch(ids=ids, namespace=NAMESPACE)
    except NotFoundException:
        return set()
    except Exception as exc:
        if "Namespace not found" in str(exc) or "404" in str(exc):
            return set()
        raise

    vectors = getattr(response, "vectors", None)

    if vectors is None and isinstance(response, dict):
        vectors = response.get("vectors", {})

    return set((vectors or {}).keys())


def embed_with_retry(pc, texts):
    while True:
        try:
            return pc.inference.embed(
                model=EMBED_MODEL,
                inputs=texts,
                parameters={
                    "input_type": "passage",
                    "truncate": "END",
                },
            )

        except PineconeApiException as exc:
            if getattr(exc, "status", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc):
                print(
                    f"   ⏳ Pinecone free-tier rate limit reached. "
                    f"Waiting {RATE_LIMIT_WAIT_SECONDS}s, then continuing..."
                )
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
                continue

            raise


def main():
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing.")

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Missing corpus: {CORPUS_PATH}\n"
            "Run the V2 ingestion first."
        )

    with gzip.open(CORPUS_PATH, "rt", encoding="utf-8") as handle:
        rows = json.load(handle)

    print(f"📚 Loaded {len(rows)} chunks from {CORPUS_PATH.name}")

    pc = Pinecone(api_key=api_key)
    existing_indexes = set(pc.list_indexes().names())

    if INDEX_NAME not in existing_indexes:
        print(
            f"🆕 Creating Pinecone index '{INDEX_NAME}' "
            f"({EMBED_DIMENSION} dimensions, cosine)..."
        )

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        while True:
            description = pc.describe_index(INDEX_NAME)

            if is_index_ready(description):
                break

            print("   Waiting for index to become ready...")
            time.sleep(2)

        print("✅ Index is ready.")
    else:
        description = pc.describe_index(INDEX_NAME)
        dimension = getattr(description, "dimension", None)

        if dimension is None and isinstance(description, dict):
            dimension = description.get("dimension")

        if dimension and int(dimension) != EMBED_DIMENSION:
            raise RuntimeError(
                f"Existing index '{INDEX_NAME}' has dimension {dimension}; "
                f"expected {EMBED_DIMENSION}."
            )

        print(f"✅ Using existing index '{INDEX_NAME}'.")

    index = pc.Index(INDEX_NAME)

    print(
        f"🧠 Resumable embedding with Pinecone '{EMBED_MODEL}' "
        "(passage mode)..."
    )
    print("   Existing vectors will be skipped; nothing is deleted.")

    total = len(rows)
    already_done = 0
    newly_uploaded = 0

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = rows[start:start + EMBED_BATCH_SIZE]
        batch_ids = [row["id"] for row in batch]

        existing = existing_ids_for_batch(index, batch_ids)
        missing = [row for row in batch if row["id"] not in existing]

        already_done += len(existing)

        if not missing:
            print(
                f"   Skipped existing {min(start + len(batch), total)}/{total}"
            )
            continue

        texts = [row["text"] for row in missing]

        result = embed_with_retry(pc, texts)
        data = getattr(result, "data", result)

        if len(data) != len(missing):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(data)} "
                f"for {len(missing)} inputs."
            )

        vectors = []

        for row, item in zip(missing, data):
            values = embedding_values(item)

            if len(values) != EMBED_DIMENSION:
                raise RuntimeError(
                    f"Unexpected embedding dimension {len(values)}; "
                    f"expected {EMBED_DIMENSION}."
                )

            vectors.append(
                {
                    "id": row["id"],
                    "values": values,
                    "metadata": {
                        "chunk_id": row["id"],
                        "text": row["text"],
                        "source": row.get("source", ""),
                        "title": row.get("title", ""),
                        "section": row.get("section", ""),
                        "unit": row.get("unit", ""),
                        "rag_version": "v3-free",
                        "embedding_model": EMBED_MODEL,
                    },
                }
            )

        index.upsert(
            vectors=vectors,
            namespace=NAMESPACE,
        )

        newly_uploaded += len(missing)
        progress = min(start + len(batch), total)

        print(
            f"   Progress {progress}/{total} "
            f"(new this run: {newly_uploaded})"
        )

        # Proactive throttle for free-tier tokens/minute.
        time.sleep(SECONDS_BETWEEN_EMBED_CALLS)

    print()
    print(f"✅ Existing vectors skipped: {already_done}")
    print(f"✅ New vectors uploaded this run: {newly_uploaded}")
    print(
        f"✅ FREE V3 indexing complete: {total} chunks in "
        f"index '{INDEX_NAME}', namespace '{NAMESPACE}'."
    )
    print("✅ Hugging Face hosted inference is no longer required.")


if __name__ == "__main__":
    main()
