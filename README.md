# PageWise  -RAG Application

PageWise Pro is a robust Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with their content through an intelligent chat interface. Built with a modern open-source stack, it leverages high-performance embeddings and large language models to provide accurate, context-aware answers.

##  Features

* **PDF Indexing**: Automatically parses, chunks, and indexes uploaded PDF files into a vector database.
* **Intelligent Chat**: Uses Groq-powered LLMs to answer questions based strictly on the uploaded context.
* **Source Citation**: Every response includes the specific source documents used to generate the answer.
* **Hybrid Search**: Implements Maximum Marginal Relevance (MMR) for diverse and relevant context retrieval.
* **Real-time Status**: Provides visual feedback during document processing and indexing.

## Tech Stack

* **Backend**: FastAPI
* **LLM**: Groq (Llama-3.3-70b-versatile)
* **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Store**: Pinecone
* **Orchestration**: LangChain
* **PDF Processing**: PyMuPDF

##  Prerequisites

* Python 3.8+
* Pinecone API Key and Index
* Groq API Key
* LangChain API Key (optional, for tracing)

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd RAG/backend
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables**
    Create a `.env` file in the `backend/` directory with the following keys:
    ```env
    GROQ_API_KEY=your_groq_key
    PINECONE_API_KEY=your_pinecone_key
    PINECONE_INDEX=pagewise-index
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=your_langchain_smith_key
    ```

4.  **Run the Application**
    ```bash
    python main.py
    ```
    The server will start at `http://0.0.0.0:8000`.

## Usage

1.  Open your browser and navigate to `http://localhost:8000`.
2.  Use the **"Click to Upload PDF"** button in the sidebar to index a document.
3.  Wait for the **"Document indexed successfully"** message.
4.  Type your question in the chat input and receive answers with source references.

##  Project Structure

* `main.py`: FastAPI backend containing RAG logic, document processing, and API endpoints.
* `index.html`: Clean, responsive frontend interface for document management and chat.
* `requirements.txt`: List of necessary Python libraries.
* `.gitignore`: Pre-configured to ignore virtual environments, environment variables, and temporary data.

## Safety & Reliability

The system is configured with a strict prompt template that instructs the assistant to:
* Use only the retrieved context to answer.
* Explicitly state when an answer is unknown rather than fabricating facts.
* Maintain a helpful and professional persona.
