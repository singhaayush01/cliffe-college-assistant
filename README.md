# Cliffe AI 🤖

A RAG (Retrieval-Augmented Generation) chatbot for **Cliffe College of Creative Arts** at Youngstown State University. Ask it about programs, faculty, leadership, events, or anything else published on the Cliffe College site, and it answers using content scraped directly from `academics.ysu.edu`.

> **Status:** Running locally for development. Live deployment is coming soon — this README will be updated with a public URL once it's hosted.


[Cliffe AI demo](./live-demo.png)


## Why this exists

A focused, always-available assistant for prospective and current Cliffe College students who don't want to dig through the university website to find a dean's name, a program's requirements, or an upcoming recital.

## Architecture

```
┌─────────────┐      POST /ask      ┌──────────────┐      similarity search     ┌───────────┐
│  Next.js UI │ ──────────────────► │   FastAPI    │ ─────────────────────────► │ Pinecone  │
│ (port 3000) │ ◄────────────────── │ (port 8000)  │ ◄───────────────────────── │  (vectors)│
└─────────────┘      answer         └──────┬───────┘                            └───────────┘
                                            │ generation
                                            ▼
                                     ┌─────────────┐
                                     │ Gemini 2.5  │
                                     │   Flash     │
                                     └─────────────┘
```

**Ingestion** (`ingest.py`, run offline/periodically):
1. Crawls Cliffe College pages via YSU's Drupal JSON:API + link-following
2. Scrapes each page in parallel, preserving headings/lists as lightweight markdown so chunking respects document structure (a bio or listing stays together, instead of being cut mid-sentence)
3. Splits each page on its headings first, falling back to size-based splitting only for oversized sections
4. Embeds chunks locally and uploads them to Pinecone

## Tech stack — 100% free tier, $0 to run

| Layer | Choice | Why |
|---|---|---|
| Frontend | Next.js (React) | Simple chat UI |
| Backend | FastAPI | Lightweight Python API |
| Orchestration | LangChain | Glue between retriever + LLM |
| Vector DB | Pinecone (free tier) | Hosted, fast similarity search |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Runs **locally**, no API key, no cost, 384-dim |
| Generation | Google Gemini `2.5-flash` | Free tier, no card required |

No OpenAI dependency anywhere in this stack — embeddings run on your own machine, and Gemini's free tier covers the answering step.

## Project structure

```
cliffe-rag/
├── backend/
│   ├── main.py                 # FastAPI server — answers questions via RAG
│   ├── ingest.py               # Scrapes YSU site, chunks, embeds, uploads to Pinecone
│   ├── debug.py                # Inspect what the vector store actually retrieves
│   ├── check_google_api.py     # Verify your Gemini API key works
│   ├── requirements.txt
│   └── .env                    # GOOGLE_API_KEY, PINECONE_API_KEY (not committed)
└── frontend/
    ├── app/
    │   ├── page.tsx             # Chat UI
    │   └── layout.tsx
    └── package.json
```

## Setup

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env`:
```
GOOGLE_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```
Get a free Gemini key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), and a free Pinecone key at [app.pinecone.io](https://app.pinecone.io). Create a Pinecone index named `cliffe-bot` with **dimension 384** (matches the embedding model above) and metric `cosine`.

Verify your key, then build the index:
```bash
python check_google_api.py
python ingest.py
```

Start the API:
```bash
python -m uvicorn main:app --reload
```
Runs on `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```
Runs on `http://localhost:3000`.

With both running, open `localhost:3000` and ask a question.

## Re-ingesting content

If you change the chunking/scraping logic or want to refresh the data, clear the index first so you don't end up with duplicate chunks:

```python
# clear_index.py (one-off script, delete after use)
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc.Index("cliffe-bot").delete(delete_all=True)
print("Index cleared.")
```

Then `python ingest.py` again.

## Roadmap

- [ ] Deploy backend + frontend publicly
- [ ] Hierarchical / parent-document retrieval for longer pages
- [ ] Expand crawl coverage beyond Cliffe College

## Author

Built by Aayush K. Singh, Class of 2026.