import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Global state - initialized on startup AFTER port is bound
chain = None
retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs after uvicorn binds the port — Render can detect it immediately."""
    global chain, retriever

    print("🧠 Initializing Cliffe AI...")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="cliffe-bot",
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2
    )

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

    chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt)
    )

    print("✅ Cliffe AI ready!")
    yield
    print("🛑 Shutting down.")


app = FastAPI(lifespan=lifespan)

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
    return {"status": "ok", "service": "Cliffe AI"}


@app.post("/ask")
def ask(q: Query):
    print(f"📝 Question: {q.question}")
    try:
        response = chain.invoke({"input": q.question})
        return {"answer": response["answer"]}
    except Exception as e:
        print(f"❌ API ERROR: {e}")
        docs = retriever.invoke(q.question)
        if docs:
            fallback = "⚠️ AI is busy, here are relevant pages:\n\n"
            seen = set()
            for doc in docs[:3]:
                src = doc.metadata.get('source', 'Unknown')
                if src not in seen:
                    fallback += f"🔗 {src}\n"
                    seen.add(src)
            return {"answer": fallback}
        return {"answer": "System Error. Please try again."}