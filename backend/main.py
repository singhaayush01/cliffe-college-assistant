import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# RAG objects are created only when first needed.
chain = None
retriever = None


def initialize_rag():
    global chain, retriever

    # Don't initialize twice.
    if chain is not None and retriever is not None:
        return

    print("🧠 Initializing Cliffe AI...")
    start = time.time()

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

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 20}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2
    )

    system_prompt = (
        "You are a helpful assistant for Cliffe College at YSU. "
        "Use the context below to answer the student's question accurately. "
        "If the answer includes a person's name, include their title. "
        "If you cannot find the answer, say "
        "'I cannot find that info on the Cliffe website'. "
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

    elapsed = time.time() - start
    print(f"✅ Cliffe AI ready in {elapsed:.2f} seconds!")


# FastAPI starts without waiting for RAG initialization.
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
        "rag_ready": chain is not None
    }


@app.post("/ask")
def ask(q: Query):
    global chain, retriever

    request_start = time.time()

    print(f"📝 Question: {q.question}")

    try:
        # Initialize RAG only when it is actually needed.
        if chain is None:
            print("⚙️ RAG not initialized. Initializing now...")
            initialize_rag()

        rag_start = time.time()

        print("⏳ Starting RAG chain...")

        response = chain.invoke({
            "input": q.question
        })

        rag_elapsed = time.time() - rag_start
        total_elapsed = time.time() - request_start

        print(f"✅ RAG completed in {rag_elapsed:.2f} seconds")
        print(f"⏱️ Total request time: {total_elapsed:.2f} seconds")

        return {
            "answer": response["answer"],
            "response_time": round(total_elapsed, 2)
        }

    except Exception as e:
        chain_elapsed = time.time() - request_start

        print(f"❌ RAG failed after {chain_elapsed:.2f} seconds")
        print(f"❌ API ERROR: {type(e).__name__}: {e}")

        # Only attempt fallback if the retriever was initialized.
        if retriever is not None:
            fallback_start = time.time()

            try:
                print("⏳ Starting fallback retrieval...")

                docs = retriever.invoke(q.question)

                fallback_elapsed = time.time() - fallback_start
                total_elapsed = time.time() - request_start

                print(
                    f"⚠️ Fallback retrieval completed in "
                    f"{fallback_elapsed:.2f} seconds"
                )

                if docs:
                    fallback = (
                        "⚠️ AI is busy, here are relevant pages:\n\n"
                    )

                    seen = set()

                    for doc in docs[:3]:
                        src = doc.metadata.get("source", "Unknown")

                        if src not in seen:
                            fallback += f"🔗 {src}\n"
                            seen.add(src)

                    return {
                        "answer": fallback,
                        "response_time": round(total_elapsed, 2)
                    }

            except Exception as fallback_error:
                print(
                    f"❌ FALLBACK ERROR: "
                    f"{type(fallback_error).__name__}: "
                    f"{fallback_error}"
                )

        total_elapsed = time.time() - request_start

        return {
            "answer": "System Error. Please try again.",
            "response_time": round(total_elapsed, 2)
        }