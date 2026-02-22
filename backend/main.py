import os
import uuid
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain & Vector Store Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# --- LangSmith Tracing Configuration ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

app = FastAPI(
    title="PageWise Pro API (Open Source Stack)",
    description="Advanced RAG application using HuggingFace & Groq.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class QueryRequest(BaseModel):
    question: str
    namespace: Optional[str] = "default"

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# --- Core RAG Logic ---

def get_vectorstore(namespace: str):
    """Initializes connection to Pinecone Vector Store using HuggingFace Embeddings."""
    # Using a high-performance open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    index_name = os.getenv("PINECONE_INDEX", "pagewise-index")
    
    return PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings, 
        namespace=namespace
    )

def create_rag_chain(namespace: str):
    """Constructs the RAG chain with Multi-Query Retrieval using Groq."""
    # Using Llama 3 for inference and query expansion
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    vectorstore = get_vectorstore(namespace)
    
    # Multi-Query Retriever: Uses Llama 3 to generate variations of the user query
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
    
    prompt_template = """You are a helpful assistant for PageWise. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer based on the context, say that you don't know. Do not make up facts.

    Context:
    {context}

    Question: {question}

    Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

# --- API Endpoints ---

@app.post("/upload-pdf", status_code=201)
async def upload_pdf(file: UploadFile = File(...), namespace: str = "default"):
    """Uploads a PDF, chunks it, and indexes it into Pinecone."""
    temp_file_path = f"temp_{uuid.uuid4()}.pdf"
    
    try:
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        loader = PyMuPDFLoader(temp_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        for chunk in chunks:
            chunk.metadata["filename"] = file.filename
            
        vectorstore = get_vectorstore(namespace)
        vectorstore.add_documents(chunks)
        
        return {
            "status": "success", 
            "filename": file.filename, 
            "chunks_indexed": len(chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Processes a question by retrieving context and generating an answer."""
    try:
        rag_chain = create_rag_chain(request.namespace)
        response = rag_chain.invoke(request.question)
        
        sources = list(set([
            doc.metadata.get("filename", "Unknown Source") 
            for doc in response["source_documents"]
        ]))
        
        return {
            "answer": response["result"],
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "online", 
        "tracing_enabled": os.getenv("LANGCHAIN_TRACING_V2") == "true",
        "embedding_model": "all-MiniLM-L6-v2"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)