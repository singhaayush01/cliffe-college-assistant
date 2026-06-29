import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. Load Brain
#    IMPORTANT: this embedding model must be IDENTICAL to whatever
#    ingest.py used to build the index, or search results will be
#    meaningless (or Pinecone will reject the query outright).
load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="cliffe-bot",
    embedding=embeddings
)

# 2. Ask the Database directly
query = "Who is the Dean of Cliffe College?"
print(f"🔎 Searching for: '{query}'")

# Get top 5 matches
results = vectorstore.similarity_search(query, k=5)

print("\n--- WHAT THE BRAIN SEES ---")
for i, doc in enumerate(results):
    print(f"\n📄 MATCH #{i+1}")
    print(f"SOURCE: {doc.metadata.get('source', 'Unknown')}")
    print(f"CONTENT SNAPSHOT: {doc.page_content[:200]}...")  # First 200 chars
    print("-" * 50)