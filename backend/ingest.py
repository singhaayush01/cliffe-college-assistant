import os
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load Keys
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "cliffe-bot"

# 2. The Full List of Content Types
# We start with the base URL for each type
base_endpoints = [
    "https://academics.ysu.edu/jsonapi/node/program",        # Majors
    "https://academics.ysu.edu/jsonapi/node/generic_page",   # Dept Info
    "https://academics.ysu.edu/jsonapi/node/events",         # Events
    "https://academics.ysu.edu/jsonapi/node/article",        # News
    "https://academics.ysu.edu/jsonapi/node/studio_recitals" # Music
]

documents = []

def clean_html(html_content):
    if not html_content: return ""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ", strip=True)

print("🚀 Starting Complete Data Ingestion (This may take a minute)...")

# 3. Fetch Data with Pagination (The "Next Page" Loop)
for start_url in base_endpoints:
    current_url = start_url
    print(f"   📂 Processing Category: {start_url}...")
    
    while current_url:
        try:
            # Be polite to the server
            headers = {"User-Agent": "YSU-Student-Project/1.0"}
            response = requests.get(current_url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('data', [])
                
                # If page is empty, stop this category
                if not items:
                    break
                
                print(f"      📄 fetching batch of {len(items)} items...")

                for item in items:
                    attrs = item.get('attributes', {})
                    title = attrs.get('title', 'No Title')
                    
                    # Get the text content
                    body = attrs.get('body')
                    raw_text = body['value'] if body and 'value' in body else ""
                    clean_text = clean_html(raw_text)

                    # Only save if it has real content
                    if len(clean_text) > 100:
                        path = attrs.get('path', {}).get('alias', '')
                        full_url = f"https://academics.ysu.edu{path}"
                        
                        doc = Document(
                            page_content=f"Title: {title}\nSource: {full_url}\nContent: {clean_text}",
                            metadata={"source": full_url, "title": title}
                        )
                        documents.append(doc)
                
                # CHECK FOR NEXT PAGE LINK
                # Drupal tells us the link for the next 50 items here:
                next_link = data.get('links', {}).get('next', {}).get('href')
                
                if next_link:
                    current_url = next_link
                    time.sleep(0.5) # Wait half a second before next request
                else:
                    current_url = None # No more pages, stop loop
                    
            else:
                print(f"      ⚠️ Error {response.status_code} at {current_url}")
                current_url = None
                
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            current_url = None

print(f"\n📦 Total Documents Found: {len(documents)}")

# 4. Save to Pinecone
if documents:
    print(f"✂️  Splitting {len(documents)} documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(documents)

    print(f"💾 Saving {len(splits)} chunks to Pinecone...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Batch the upload so it doesn't timeout
    vectorstore = PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=pinecone_api_key
    )
    print("🎉 SUCCESS! Full dataset ingested.")
else:
    print("❌ No documents found.")