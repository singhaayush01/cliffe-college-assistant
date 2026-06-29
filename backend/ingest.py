import os
import concurrent.futures
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "cliffe-bot"

HEADERS = {"User-Agent": "YSU-Student-Project/1.0"}

# Pages we always want, even if the API doesn't surface them directly
priority_urls = [
    "https://academics.ysu.edu/cliffe-college-of-creative-arts/leadership-administration/phyllis-m-paul",
    "https://academics.ysu.edu/cliffe-college-of-creative-arts/leadership-administration",
    "https://academics.ysu.edu/cliffe-college-of-creative-arts/about-us",
]

# Every Drupal content type worth indexing (merged from both old scripts)
api_endpoints = [
    "https://academics.ysu.edu/jsonapi/node/office",                      # Office (Deans etc.)
    "https://academics.ysu.edu/jsonapi/node/program",                     # Programs (Majors)
    "https://academics.ysu.edu/jsonapi/node/generic_page",                # General info
    "https://academics.ysu.edu/jsonapi/node/page",                        # Basic page
    "https://academics.ysu.edu/jsonapi/node/article",                     # News
    "https://academics.ysu.edu/jsonapi/node/events",                      # Events
    "https://academics.ysu.edu/jsonapi/node/stem_blog",                   # STEM Blog
    "https://academics.ysu.edu/jsonapi/node/etc_blog",                    # ETC Blog
    "https://academics.ysu.edu/jsonapi/node/studio_recitals",             # Studio Recitals
    "https://academics.ysu.edu/jsonapi/node/percussion_student_recitals", # Percussion Recitals
    "https://academics.ysu.edu/jsonapi/node/percussion_ensemble",         # Percussion Ensemble
]


def collect_urls_from_api(endpoint):
    """Walk one JSON:API endpoint's pagination, return every page URL it lists."""
    found = set()
    current_url = endpoint
    while current_url:
        try:
            resp = requests.get(current_url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json()
            items = data.get("data", [])
            if not items:
                break
            for item in items:
                path = item.get("attributes", {}).get("path", {}).get("alias", "")
                if path:
                    found.add(f"https://academics.ysu.edu{path}")
            current_url = data.get("links", {}).get("next", {}).get("href")
        except Exception as e:
            print(f"      ⚠️ API error on {endpoint}: {e}")
            break
    return found


HEADING_MARKERS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}


def extract_structured_text(soup):
    """Walk the page in document order and mark headings/list items with
    lightweight markdown (#, ##, -) instead of flattening everything into
    one blob. This is what lets the chunker later split on real section
    boundaries (a bio, a listing, an event) instead of blind character counts."""
    lines = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th"]):
        text = tag.get_text(separator=" ", strip=True)
        if not text:
            continue
        if tag.name in HEADING_MARKERS:
            lines.append(f"\n{HEADING_MARKERS[tag.name]} {text}\n")
        elif tag.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def scrape_page(url):
    """Fetch one page; return its structured text plus any cliffe-related links it contains."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        if resp.status_code != 200:
            return None, []

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = extract_structured_text(soup)
        if len(text) < 50:
            return None, []

        links = []
        for a in soup.find_all("a", href=True):
            full = urljoin(url, a["href"]).split("#")[0]
            if "academics.ysu.edu" in full and "cliffe" in full:
                links.append(full)

        return {"url": url, "text": text}, links
    except Exception:
        return None, []


def scrape_many(urls):
    """Scrape a batch of URLs in parallel, return (documents, newly-found links)."""
    docs, new_links = [], set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(scrape_page, u): u for u in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            result, links = future.result()
            if result:
                docs.append(Document(
                    page_content=result["text"],
                    metadata={"source": result["url"]},
                ))
            new_links.update(links)
    return docs, new_links


def chunk_documents(documents):
    """Split on document structure first (headings = natural section
    boundaries, e.g. one bio or one listing), then fall back to size-based
    splitting only for sections that are still too long."""
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    final_chunks = []
    for doc in documents:
        try:
            sections = md_splitter.split_text(doc.page_content) or [doc]
        except Exception:
            sections = [doc]

        for section in sections:
            section.metadata["source"] = doc.metadata.get("source")
            if len(section.page_content) > 1200:
                final_chunks.extend(size_splitter.split_documents([section]))
            else:
                final_chunks.append(section)

    return final_chunks


def main():
    print("🚀 Phase 1: Collecting URLs from the JSON:API (finding the map)...")
    unique_urls = set(priority_urls)
    for endpoint in api_endpoints:
        print(f"   📂 Scanning: {endpoint}")
        unique_urls |= collect_urls_from_api(endpoint)
    print(f"   ✅ Found {len(unique_urls)} candidate pages.")

    print("🚀 Phase 2: Scraping pages in parallel (100% free, 0 API calls)...")
    documents, discovered_links = scrape_many(unique_urls)
    visited = set(unique_urls)

    # One follow-up pass over any new links found while scraping (e.g. a bio
    # page linked from the leadership listing that the API didn't surface).
    new_links = discovered_links - visited
    if new_links:
        print(f"🚀 Phase 3: Following {len(new_links)} newly discovered links...")
        more_docs, _ = scrape_many(new_links)
        documents.extend(more_docs)

    print(f"\n📦 Final document count: {len(documents)}")
    if not documents:
        print("❌ No documents found. Something went wrong — check your network/endpoints.")
        return

    print("✂️  Splitting by document structure (headings first, size as fallback)...")
    splits = chunk_documents(documents)

    print("🧠 Loading free local embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"💾 Uploading {len(splits)} chunks to Pinecone index '{index_name}'...")
    batch_size = 100
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        print(f"      Uploading batch {i}-{i + len(batch)}...")
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=index_name,
            pinecone_api_key=pinecone_api_key,
        )

    print("🎉 SUCCESS! The Brain is up to date — 0 OpenAI calls, 0 cost.")


if __name__ == "__main__":
    main()