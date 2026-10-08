import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import re
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pinecone import Pinecone

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_URL = "https://academics.ysu.edu"
INDEX_NAME = "cliffe-bot"
DEFAULT_NAMESPACE = "cliffe-v2"

# These are section roots, not one-off fixes for individual people/pages.
# The crawler recursively discovers relevant pages underneath them.
SEED_URLS = [
    f"{BASE_URL}/cliffe-college-of-creative-arts",
    f"{BASE_URL}/dana-school-of-music",
    f"{BASE_URL}/art",
    f"{BASE_URL}/university-theatre",
    f"{BASE_URL}/mcdonough-museum-of-art",
    f"{BASE_URL}/percussion",
]

ALLOWED_PREFIXES = (
    "/cliffe-college-of-creative-arts",
    "/cliffe-college-creative-arts",
    "/dana-school-of-music",
    "/art",
    "/university-theatre",
    "/mcdonough-museum-of-art",
    "/percussion",
)

# JSON:API is used as a second discovery channel for pages that might not
# be reachable from normal navigation. We discover all Drupal node content
# types automatically from /jsonapi. The list below is only a fallback if the
# discovery document is temporarily unavailable.
JSONAPI_ROOT = f"{BASE_URL}/jsonapi"

FALLBACK_API_ENDPOINTS = [
    f"{BASE_URL}/jsonapi/node/office",
    f"{BASE_URL}/jsonapi/node/program",
    f"{BASE_URL}/jsonapi/node/generic_page",
    f"{BASE_URL}/jsonapi/node/page",
    f"{BASE_URL}/jsonapi/node/article",
    f"{BASE_URL}/jsonapi/node/events",
    f"{BASE_URL}/jsonapi/node/studio_recitals",
    f"{BASE_URL}/jsonapi/node/percussion_student_recitals",
    f"{BASE_URL}/jsonapi/node/percussion_ensemble",
]

HEADERS = {
    "User-Agent": (
        "CliffeCollegeStudentRAG/2.0 "
        "(educational project; contact via Youngstown State University)"
    )
}

REQUEST_TIMEOUT = 15
DEFAULT_WORKERS = 6
DEFAULT_MAX_PAGES = 1200

# Keep chunks focused enough for retrieval while preserving local context.
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 180
MIN_CHUNK_CHARS = 40

DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = DATA_DIR / "cliffe_v2_corpus.json.gz"
REPORT_PATH = DATA_DIR / "cliffe_v2_crawl_report.json"

SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp3", ".wav", ".mp4", ".mov", ".avi",
    ".zip", ".rar", ".7z",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pdf",
)

DROP_SELECTORS = [
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "noscript",
    "iframe",
    "form",
    ".breadcrumb",
    ".breadcrumbs",
    ".block-system-breadcrumb-block",
    ".menu",
    ".pager",
    ".tabs",
    ".sidebar",
    ".region-sidebar-first",
    ".region-sidebar-second",
    ".region-header",
    ".region-footer",
    ".visually-hidden",
    ".skip-link",
    "[aria-hidden='true']",
]

HEADING_MARKERS = {
    "h1": "#",
    "h2": "##",
    "h3": "###",
    "h4": "####",
    "h5": "#####",
    "h6": "######",
}


# ---------------------------------------------------------------------------
# TYPES
# ---------------------------------------------------------------------------

@dataclass
class ScrapedPage:
    source: str
    title: str
    unit: str
    text: str


@dataclass
class CrawlFailure:
    url: str
    reason: str


# ---------------------------------------------------------------------------
# URL HELPERS
# ---------------------------------------------------------------------------

def canonicalize_url(raw_url: str) -> str | None:
    """Return a stable, crawlable academics.ysu.edu URL or None."""
    if not raw_url:
        return None

    raw_url = raw_url.strip()

    if raw_url.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None

    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        return None

    if parsed.scheme not in ("http", "https"):
        return None

    if parsed.hostname != "academics.ysu.edu":
        return None

    path = re.sub(r"/{2,}", "/", parsed.path or "/")

    if path != "/":
        path = path.rstrip("/")

    lower_path = path.lower()

    if lower_path.startswith("/jsonapi/"):
        return None

    if lower_path.endswith(SKIP_EXTENSIONS):
        return None

    # Query strings on this site are largely filters/tracking state and can
    # create duplicate crawl targets. The canonical content URL is enough for
    # the RAG corpus.
    return urlunsplit(("https", "academics.ysu.edu", path, "", ""))


def is_allowed_url(url: str) -> bool:
    try:
        path = urlsplit(url).path
    except ValueError:
        return False

    return any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in ALLOWED_PREFIXES
    )


def unit_from_url(url: str) -> str:
    path = urlsplit(url).path.lower()

    if path.startswith("/dana-school-of-music") or path.startswith("/percussion"):
        return "Dana School of Music"

    if path.startswith("/art"):
        return "Department of Art"

    if path.startswith("/university-theatre"):
        return "University Theatre"

    if path.startswith("/mcdonough-museum-of-art"):
        return "McDonough Museum of Art"

    return "Cliffe College of Creative Arts"


# ---------------------------------------------------------------------------
# DISCOVERY
# ---------------------------------------------------------------------------

def _collect_jsonapi_node_hrefs(value, found: set[str]) -> None:
    """Recursively collect /jsonapi/node/* hrefs from Drupal's discovery JSON."""
    if isinstance(value, dict):
        for child in value.values():
            _collect_jsonapi_node_hrefs(child, found)
        return

    if isinstance(value, list):
        for child in value:
            _collect_jsonapi_node_hrefs(child, found)
        return

    if not isinstance(value, str):
        return

    if "/jsonapi/node/" not in value:
        return

    candidate = urljoin(BASE_URL, value)
    parsed = urlsplit(candidate)
    path = parsed.path.rstrip("/")
    parts = path.strip("/").split("/")

    # Keep collection endpoints only: /jsonapi/node/<content_type>
    if len(parts) == 3 and parts[0] == "jsonapi" and parts[1] == "node":
        found.add(
            urlunsplit(("https", "academics.ysu.edu", path, "", ""))
        )


def discover_api_endpoints() -> list[str]:
    """Discover all Drupal node content types, with a safe fallback list."""
    found: set[str] = set(FALLBACK_API_ENDPOINTS)

    try:
        response = requests.get(
            JSONAPI_ROOT,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            print(
                f"⚠️ JSON:API discovery returned HTTP "
                f"{response.status_code}; using fallback list."
            )
            return sorted(found)

        data = response.json()
        _collect_jsonapi_node_hrefs(data, found)

    except Exception as exc:
        print(
            f"⚠️ JSON:API discovery failed: {exc}; "
            "using fallback list."
        )

    return sorted(found)


def collect_urls_from_api(endpoint: str) -> set[str]:
    """Walk a Drupal JSON:API endpoint and keep only relevant Cliffe URLs."""
    found: set[str] = set()
    current_url: str | None = endpoint

    while current_url:
        try:
            response = requests.get(
                current_url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                print(f"   ⚠️ JSON:API {response.status_code}: {current_url}")
                break

            data = response.json()

            for item in data.get("data", []):
                alias = (
                    item.get("attributes", {})
                    .get("path", {})
                    .get("alias", "")
                )

                if not alias:
                    continue

                candidate = canonicalize_url(urljoin(BASE_URL, alias))

                if candidate and is_allowed_url(candidate):
                    found.add(candidate)

            next_link = data.get("links", {}).get("next")
            if isinstance(next_link, dict):
                current_url = next_link.get("href")
            elif isinstance(next_link, str):
                current_url = next_link
            else:
                current_url = None

        except Exception as exc:
            print(f"   ⚠️ JSON:API error: {endpoint}: {exc}")
            break

    return found


# ---------------------------------------------------------------------------
# CONTENT EXTRACTION
# ---------------------------------------------------------------------------

def clean_visible_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()

    if text in {"", "/", "Home", "Skip to main content"}:
        return ""

    return text


def page_title_from_soup(soup: BeautifulSoup, container) -> str:
    h1 = container.find("h1") if container else None

    if h1:
        title = clean_visible_text(h1.get_text(" ", strip=True))
        if title:
            return title

    if soup.title:
        title = clean_visible_text(soup.title.get_text(" ", strip=True))
        title = re.sub(r"\s*\|\s*academics\.ysu\.edu\s*$", "", title, flags=re.I)
        if title:
            return title

    return "Cliffe College Page"


def extract_structured_text(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Extract the page's main content while preserving useful headings.

    In addition to normal paragraphs/lists/tables, leaf <div> blocks are
    included so Drupal staff cards and other card-based content are not lost.
    """
    for selector in DROP_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    container = (
        soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("article")
        or soup.body
        or soup
    )

    title = page_title_from_soup(soup, container)

    structured_tags = {
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "li", "dt", "dd", "td", "th", "address",
    }

    lines: list[str] = []
    seen_lines: set[str] = set()

    for tag in container.find_all(
        [
            "h1", "h2", "h3", "h4", "h5", "h6",
            "p", "li", "dt", "dd", "td", "th", "address", "div",
        ]
    ):
        # A div is useful only when it is a leaf-ish content block. This helps
        # capture Drupal cards without duplicating whole sections repeatedly.
        if tag.name == "div":
            if tag.find(list(structured_tags)):
                continue

            text = clean_visible_text(tag.get_text(" ", strip=True))

            if not (2 <= len(text) <= 600):
                continue

            rendered = text

        else:
            text = clean_visible_text(tag.get_text(" ", strip=True))

            if not text:
                continue

            if tag.name in HEADING_MARKERS:
                rendered = f"{HEADING_MARKERS[tag.name]} {text}"
            elif tag.name == "li":
                rendered = f"- {text}"
            else:
                rendered = text

        normalized = " ".join(rendered.lower().split())

        if normalized in seen_lines:
            continue

        seen_lines.add(normalized)
        lines.append(rendered)

    text = "\n".join(lines).strip()

    return title, text


def discover_links(soup: BeautifulSoup, source_url: str) -> set[str]:
    links: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(source_url, anchor["href"])
        candidate = canonicalize_url(absolute)

        if candidate and is_allowed_url(candidate):
            links.add(candidate)

    return links


def scrape_page(url: str) -> tuple[ScrapedPage | None, set[str], CrawlFailure | None]:
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        requested_url = canonicalize_url(url)
        redirected_url = canonicalize_url(response.url)

        # Keep a clean Cliffe URL as the source. Drupal can redirect some
        # aliases to raw /node/<id> URLs; do not let those pollute metadata.
        if redirected_url and is_allowed_url(redirected_url):
            final_url = redirected_url
        elif requested_url and is_allowed_url(requested_url):
            final_url = requested_url
        else:
            return None, set(), CrawlFailure(
                url=url,
                reason="Redirected outside allowed Cliffe scope",
            )

        if response.status_code != 200:
            return None, set(), CrawlFailure(
                url=url,
                reason=f"HTTP {response.status_code}",
            )

        content_type = response.headers.get("content-type", "").lower()

        if "text/html" not in content_type:
            return None, set(), CrawlFailure(
                url=url,
                reason=f"Skipped content-type: {content_type or 'unknown'}",
            )

        soup = BeautifulSoup(response.text, "html.parser")

        # Discover links before extraction removes menus/containers.
        links = discover_links(soup, final_url)

        title, text = extract_structured_text(soup)

        if len(text) < 80:
            return None, links, CrawlFailure(
                url=final_url,
                reason="Too little useful main content",
            )

        page = ScrapedPage(
            source=final_url,
            title=title,
            unit=unit_from_url(final_url),
            text=text,
        )

        return page, links, None

    except Exception as exc:
        return None, set(), CrawlFailure(
            url=url,
            reason=f"{type(exc).__name__}: {exc}",
        )


def crawl_site(
    initial_urls: Iterable[str],
    max_pages: int,
    workers: int,
) -> tuple[list[ScrapedPage], list[CrawlFailure], set[str]]:
    """
    Recursively crawl the allowed Cliffe ecosystem until the queue is empty
    or max_pages is reached.
    """
    pending = deque()
    queued: set[str] = set()
    visited: set[str] = set()
    pages_by_url: dict[str, ScrapedPage] = {}
    failures: list[CrawlFailure] = []

    for raw_url in initial_urls:
        candidate = canonicalize_url(raw_url)

        if candidate and is_allowed_url(candidate) and candidate not in queued:
            pending.append(candidate)
            queued.add(candidate)

    print(f"🌐 Starting focused crawl with {len(pending)} discovered URLs...")

    while pending and len(visited) < max_pages:
        batch: list[str] = []

        while pending and len(batch) < workers * 2 and len(visited) + len(batch) < max_pages:
            url = pending.popleft()

            if url in visited:
                continue

            batch.append(url)

        if not batch:
            continue

        for url in batch:
            visited.add(url)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_url = {
                executor.submit(scrape_page, url): url
                for url in batch
            }

            for future in concurrent.futures.as_completed(future_to_url):
                page, links, failure = future.result()

                if page:
                    pages_by_url[page.source] = page

                if failure:
                    failures.append(failure)

                for link in links:
                    if link not in visited and link not in queued:
                        pending.append(link)
                        queued.add(link)

        print(
            f"   Crawled {len(visited)} | "
            f"Useful pages {len(pages_by_url)} | "
            f"Queue {len(pending)}"
        )

    return list(pages_by_url.values()), failures, visited


# ---------------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------------

def normalize_for_hash(text: str) -> str:
    return " ".join(text.lower().split())


def stable_chunk_id(source: str, section: str, text: str) -> str:
    payload = (
        source.strip()
        + "\n"
        + section.strip()
        + "\n"
        + normalize_for_hash(text)
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def split_pages(pages: list[ScrapedPage]) -> list[dict]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ],
        strip_headers=False,
    )

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    seen_content: set[str] = set()

    for page in pages:
        try:
            sections = header_splitter.split_text(page.text)
        except Exception:
            sections = []

        if not sections:
            sections = [Document(page_content=page.text, metadata={})]

        for section_doc in sections:
            section = (
                section_doc.metadata.get("h4")
                or section_doc.metadata.get("h3")
                or section_doc.metadata.get("h2")
                or section_doc.metadata.get("h1")
                or page.title
            )

            pieces = size_splitter.split_text(section_doc.page_content)

            for piece in pieces:
                body = piece.strip()

                if len(body) < MIN_CHUNK_CHARS:
                    continue

                # Rich context is deliberately included in the embedded text.
                # This makes partial names, page titles, and section names much
                # easier to retrieve.
                retrieval_text = (
                    f"Page: {page.title}\n"
                    f"Unit: {page.unit}\n"
                    f"Section: {section}\n"
                    f"Source: {page.source}\n\n"
                    f"{body}"
                )

                normalized = normalize_for_hash(retrieval_text)

                # Global text dedupe prevents repeated template/card content
                # from dominating retrieval.
                if normalized in seen_content:
                    continue

                seen_content.add(normalized)

                chunk_id = stable_chunk_id(
                    page.source,
                    section,
                    body,
                )

                chunks.append(
                    {
                        "id": chunk_id,
                        "text": retrieval_text,
                        "content": body,
                        "source": page.source,
                        "title": page.title,
                        "section": section,
                        "unit": page.unit,
                    }
                )

    return chunks


# ---------------------------------------------------------------------------
# OUTPUTS FOR FUTURE HYBRID RETRIEVAL
# ---------------------------------------------------------------------------

def write_corpus(chunks: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with gzip.open(CORPUS_PATH, "wt", encoding="utf-8") as handle:
        json.dump(chunks, handle, ensure_ascii=False)

    print(f"💾 Hybrid-search corpus: {CORPUS_PATH}")
    print(f"   Size: {CORPUS_PATH.stat().st_size / (1024 * 1024):.2f} MB")


def write_report(
    pages: list[ScrapedPage],
    chunks: list[dict],
    failures: list[CrawlFailure],
    visited: set[str],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    unit_counts = Counter(page.unit for page in pages)
    failure_counts = Counter(failure.reason for failure in failures)

    report = {
        "generated_at_epoch": int(time.time()),
        "visited_url_count": len(visited),
        "useful_page_count": len(pages),
        "chunk_count": len(chunks),
        "unit_counts": dict(unit_counts),
        "failure_reason_counts": dict(failure_counts),
        "pages": [
            {
                "source": page.source,
                "title": page.title,
                "unit": page.unit,
                "text_characters": len(page.text),
            }
            for page in sorted(pages, key=lambda item: item.source)
        ],
        "failures": [asdict(item) for item in failures],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"🧾 Crawl report: {REPORT_PATH}")


# ---------------------------------------------------------------------------
# PINECONE V2 NAMESPACE
# ---------------------------------------------------------------------------

def index_chunks(chunks: list[dict], namespace: str) -> None:
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing from the environment.")

    print("\n🧠 Loading local all-MiniLM-L6-v2 embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Validate the existing Pinecone index dimension before clearing the V2
    # namespace. This protects the current/default namespace from mistakes.
    sample_vector = embeddings.embed_query("Cliffe College dimension check")

    pc = Pinecone(api_key=api_key)
    description = pc.describe_index(INDEX_NAME)

    index_dimension = getattr(description, "dimension", None)

    if index_dimension is None and isinstance(description, dict):
        index_dimension = description.get("dimension")

    if index_dimension and len(sample_vector) != int(index_dimension):
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"MiniLM produced {len(sample_vector)}, "
            f"but Pinecone index '{INDEX_NAME}' expects {index_dimension}. "
            "Nothing was deleted."
        )

    index = pc.Index(INDEX_NAME)

    print(
        f"\n🧹 Rebuilding ONLY namespace '{namespace}'. "
        "The current/default namespace is untouched."
    )

    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"   Existing namespace '{namespace}' cleared.")
    except Exception as exc:
        if "Namespace not found" in str(exc) or "404" in str(exc):
            print(
                f"   Namespace '{namespace}' does not exist yet. "
                "Creating it with the first upload."
            )
        else:
            raise

    embedding_batch_size = 64
    upsert_batch_size = 100

    total = len(chunks)
    uploaded = 0

    for start in range(0, total, embedding_batch_size):
        batch = chunks[start:start + embedding_batch_size]
        texts = [item["text"] for item in batch]

        vectors = embeddings.embed_documents(texts)

        records = []

        for item, vector in zip(batch, vectors):
            # Pinecone metadata stays compact and contains the exact text that
            # the retrieval layer will later hand to Gemini.
            metadata = {
                "text": item["text"],
                "source": item["source"],
                "title": item["title"],
                "section": item["section"],
                "unit": item["unit"],
                "chunk_id": item["id"],
                "rag_version": "v2",
            }

            records.append(
                {
                    "id": item["id"],
                    "values": vector,
                    "metadata": metadata,
                }
            )

        for record_start in range(0, len(records), upsert_batch_size):
            record_batch = records[
                record_start:record_start + upsert_batch_size
            ]

            index.upsert(
                vectors=record_batch,
                namespace=namespace,
            )

        uploaded += len(batch)
        print(f"   Uploaded {uploaded}/{total} chunks")

    print(
        f"\n✅ V2 indexing complete: {total} chunks "
        f"in namespace '{namespace}'."
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Cliffe AI V2 knowledge base. "
            "Running without --index is a safe crawl/chunk audit only."
        )
    )

    parser.add_argument(
        "--index",
        action="store_true",
        help=(
            "After crawling, rebuild the dedicated Pinecone V2 namespace. "
            "The default namespace is never deleted."
        ),
    )

    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help=f"Pinecone namespace to rebuild (default: {DEFAULT_NAMESPACE})",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=f"Maximum pages to crawl (default: {DEFAULT_MAX_PAGES})",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent page fetches (default: {DEFAULT_WORKERS})",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("🚀 Cliffe AI V2.1 ingestion")
    print("=" * 72)

    # 1. Root section seeds.
    discovered_urls: set[str] = set(SEED_URLS)

    # 2. JSON:API discovers relevant orphaned pages, but only under the
    # allowed Cliffe/Dana/Art/Theatre/Museum path prefixes.
    print("\n🗺️ Discovering Drupal node content types...")

    api_endpoints = discover_api_endpoints()

    print(
        f"✅ Drupal node content types discovered: "
        f"{len(api_endpoints)}"
    )

    print("\n🗺️ Discovering relevant Cliffe pages from Drupal JSON:API...")

    endpoint_hits = []

    for endpoint in api_endpoints:
        urls = collect_urls_from_api(endpoint)

        if urls:
            endpoint_hits.append(
                (endpoint.rsplit("/", 1)[-1], len(urls))
            )
            discovered_urls.update(urls)
            print(
                f"   {endpoint.rsplit('/', 1)[-1]}: "
                f"+{len(urls)}"
            )

    print(
        f"✅ Content types containing Cliffe-scope URLs: "
        f"{len(endpoint_hits)}"
    )
    print(f"✅ Initial relevant URL set: {len(discovered_urls)}")

    # 3. Recursive focused crawl.
    pages, failures, visited = crawl_site(
        initial_urls=discovered_urls,
        max_pages=max(1, args.max_pages),
        workers=max(1, args.workers),
    )

    print("\n📚 Crawl summary")
    print(f"   Visited URLs: {len(visited)}")
    print(f"   Useful pages: {len(pages)}")
    print(f"   Skipped/failed: {len(failures)}")

    # 4. Structured section-aware chunks.
    print("\n✂️ Building retrieval chunks...")

    chunks = split_pages(pages)

    print(f"✅ High-quality chunks: {len(chunks)}")

    # 5. Permanent corpus/report files. The corpus is what we will use for
    # BM25 lexical retrieval in the next stage.
    write_corpus(chunks)
    write_report(
        pages=pages,
        chunks=chunks,
        failures=failures,
        visited=visited,
    )

    # Useful checks for the two types of questions that exposed weaknesses in
    # the old system. These are diagnostics only; they do not influence what
    # gets crawled or indexed.
    corpus_lower = "\n".join(item["text"].lower() for item in chunks)

    print("\n🔎 Corpus sanity checks")
    print(
        "   Contains 'samantha':",
        "YES" if "samantha" in corpus_lower else "NO",
    )
    print(
        "   Contains 'scholarship':",
        "YES" if "scholarship" in corpus_lower else "NO",
    )

    if not args.index:
        print(
            "\n✅ Audit complete. Pinecone was NOT changed.\n"
            "If the counts/report look reasonable, run:\n"
            "   python ingest.py --index"
        )
        return

    # 6. Rebuild the isolated V2 namespace. Existing/default namespace stays
    # available to the current production backend throughout this process.
    index_chunks(
        chunks=chunks,
        namespace=args.namespace,
    )


if __name__ == "__main__":
    main()
