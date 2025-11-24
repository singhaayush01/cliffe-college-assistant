from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load Keys
load_dotenv()

# 2. Setup the App
app = FastAPI()

# Allow your React Frontend to talk to this Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Connect to the Brain (Pinecone + OpenAI)
print("🧠 Loading the Brain...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="cliffe-bot",
    embedding=embeddings
)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-4o-mini")

# 4. Instructions for the AI
system_prompt = (
    "You are a helpful assistant for Cliffe College at YSU. "
    "Use the context below to answer the student's question. "
    "If you don't know, say 'I cannot find that info on the Cliffe website'. "
    "Keep your answer to 3 sentences max."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Build the Thinking Chain
chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    print(f"📝 Question received: {q.question}")
    response = chain.invoke({"input": q.question})
    print(f"✅ Answer sent.")
    return {"answer": response["answer"]}