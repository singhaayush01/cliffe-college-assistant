from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🧠 Loading Gemini 2.5 Flash (Free Tier)...")

# 1. FREE EMBEDDINGS — runs locally on your machine, $0, no API key needed.
#    Must always match whatever model ingest.py used to build the index.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. DATABASE CONNECTION
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="cliffe-bot",
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 3. THE MODEL
#    gemini-2.0-flash was shut down by Google on June 1, 2026 — calls to it
#    now fail. gemini-2.5-flash is the current stable, free-tier model.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=2
)

# 4. PROMPT
system_prompt = (
    "You are a helpful assistant for Cliffe College at YSU. "
    "Use the context below to answer the student's question accurately. "
    "If the answer includes a person's name, include their title. "
    "If you cannot find the answer, say 'I cannot find that info on the Cliffe website'. "
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    print(f"📝 Question: {q.question}")
    try:
        response = chain.invoke({"input": q.question})
        return {"answer": response["answer"]}

    except Exception as e:
        print(f"❌ API ERROR: {e}")
        # Fallback if Gemini fails
        docs = retriever.invoke(q.question)
        if docs:
            fallback = "⚠️ **AI is busy, but here are the relevant pages:**\n\n"
            seen = set()
            for doc in docs[:3]:
                src = doc.metadata.get('source', 'Unknown')
                if src not in seen:
                    fallback += f"🔗 {src}\n"
                    seen.add(src)
            return {"answer": fallback}
        return {"answer": "System Error. Please try again."}