import os
import requests
import concurrent.futures
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "cliffe-bot"

# 1. THE FINAL API LIST (Every possible drawer)
# 1. THE COMPLETE API LIST (Based on your Drupal Admin Panel)
base_endpoints = [
    # CORE INFO (Deans, Majors, Depts)
    "https://academics.ysu.edu/jsonapi/node/office",                     # Office 
    "https://academics.ysu.edu/jsonapi/node/program",                    # Program Page (Majors)
    "https://academics.ysu.edu/jsonapi/node/generic_page",               # Generic Page (General Info)
    "https://academics.ysu.edu/jsonapi/node/page",                       # Basic Page

    # NEWS & UPDATES
    "https://academics.ysu.edu/jsonapi/node/article",                    # Article (News)
    "https://academics.ysu.edu/jsonapi/node/events",                     # Events
    "https://academics.ysu.edu/jsonapi/node/stem_blog",                  # STEM Blog
    "https://academics.ysu.edu/jsonapi/node/etc_blog",                   # ETC Blog

    # MUSIC SPECIFIC (Critical for Arts College)
    "https://academics.ysu.edu/jsonapi/node/studio_recitals",            # Studio Recitals
    "https://academics.ysu.edu/jsonapi/node/percussion_student_recitals",# Percussion Recitals
    "https://academics.ysu.edu/jsonapi/node/percussion_ensemble"         # Percussion Ensemble
]

# Set to keep unique URLs (Drupal sometimes lists pages twice)
unique_urls = set()
documents = []

def get_visual_text(full_url):
    """
    Worker function: Visits the page like a browser to see what the student sees.
    """
    try:
        headers = {"User-Agent": "YSU-Student-Project/1.0"}
        # 5 second timeout to keep it moving
        response = requests.get(full_url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove invisible junk
            for script in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                script.decompose()
                
            # Get text with spacing
            text = soup.get_text(separator=" ", strip=True)
            return {"url": full_url, "text": text}
    except:
        return None
    return None

print("🚀 Phase 1: Scouting URLs via API (Finding the map)...")

# --- PHASE 1: COLLECT URLS ---
for start_url in base_endpoints:
    current_url = start_url
    print(f"   📂 Scanning Category: {start_url}...")
    
    while current_url:
        try:
            headers = {"User-Agent": "YSU-Student-Project/1.0"}
            response = requests.get(current_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('data', [])
                
                if not items: break

                for item in items:
                    attrs = item.get('attributes', {})
                    path = attrs.get('path', {}).get('alias', '')
                    if path:
                        full_url = f"https://academics.ysu.edu{path}"
                        unique_urls.add(full_url)

                # Pagination: Go to next page
                next_link = data.get('links', {}).get('next', {}).get('href')
                current_url = next_link if next_link else None
            else:
                break
        except Exception as e:
            print(f"      ⚠️ API Error: {e}")
            break

print(f"   ✅ Map Complete. Found {len(unique_urls)} unique pages.")
print("🚀 Phase 2: The Swarm (Scraping text in parallel)...")

# --- PHASE 2: PARALLEL SCRAPING ---
# We use 10 workers to scrape faster
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    future_to_url = {executor.submit(get_visual_text, url): url for url in unique_urls}
    
    counter = 0
    for future in concurrent.futures.as_completed(future_to_url):
        result = future.result()
        counter += 1
        
        if counter % 50 == 0:
            print(f"      ⚡️ Scraped {counter}/{len(unique_urls)} pages...")

        # Filter: Keep valid pages with more than 50 characters of text
        if result and len(result['text']) > 50:
            doc = Document(
                page_content=f"Source: {result['url']}\nContent: {result['text']}",
                metadata={"source": result['url']}
            )
            documents.append(doc)

print(f"\n📦 Final Document Count: {len(documents)}")

# --- PHASE 3: UPLOAD TO BRAIN ---
if documents:
    print(f"✂️  Splitting and Uploading to Pinecone...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Upload in batches of 100 to prevent errors
    batch_size = 100
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        print(f"      💾 Uploading batch {i} - {i+len(batch)}...")
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=index_name,
            pinecone_api_key=pinecone_api_key
        )
    print("🎉 SUCCESS! The Brain contains ALL data (Deans, Majors, Events).")
else:
    print("❌ No documents found. Something went wrong.")