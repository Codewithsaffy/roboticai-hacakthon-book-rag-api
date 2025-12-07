"""
Vercel Serverless Function Entry Point for RAG Chat API (Agentic Version).
"""

import os
import uuid
import json
import re # Added for regex splitting
from contextlib import asynccontextmanager
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Import from openai-agents package
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, function_tool

# Load environment variables
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768
CHAT_MODEL_NAME = "gemini-2.0-flash"

# --- Validation ---
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")
if not QDRANT_URL:
    raise ValueError("QDRANT_URL environment variable is required")
if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY environment variable is required")

# --- Initialization ---
genai.configure(api_key=GEMINI_API_KEY)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

set_tracing_disabled(True)

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

# --- Helper Functions (Core Logic) ---

def recursive_split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Splits text recursively by separators to ensure chunks are semantically meaningful.
    Separators: Paragraphs (\n\n), Lines (\n), Sentences (.), Words ( )
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def _split(text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[0]
        new_separators = separators[1:]
        
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text) # Split by character if no separator left
            
        final_split = []
        good_splits = []
        
        for s in splits:
            if len(s) < chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = separator.join(good_splits)
                    final_split.append(merged)
                    good_splits = []
                if new_separators:
                    final_split.extend(_split(s, new_separators))
                else:
                    final_split.append(s)
                    
        if good_splits:
            merged = separator.join(good_splits)
            final_split.append(merged)
            
        # Merge small chunks
        current_chunk = ""
        for s in final_split:
            # If s is huge (failed to split), just take it
            if len(s) > chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk)
                    current_chunk = ""
                final_chunks.append(s)
                continue
                
            if len(current_chunk) + len(s) + len(separator) <= chunk_size:
                current_chunk += (separator if current_chunk else "") + s
            else:
                final_chunks.append(current_chunk)
                current_chunk = s
                
        if current_chunk:
            final_chunks.append(current_chunk)
            
        return final_chunks

    # Basic cleaning
    text = text.replace('\r', '')
    return _split(text, separators)

def get_embedding(text: str, task_type: str = "retrieval_document") -> List[float]:
    """Generate embedding using Gemini embedding model"""
    try:
        # Retry logic could be added here
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise e

def ensure_collection_exists():
    """Create collection if it doesn't exist"""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if COLLECTION_NAME not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Collection '{COLLECTION_NAME}' created.")
    except Exception as e:
        print(f"⚠️ Error checking/creating collection: {e}")

# --- Agent Tools ---
@function_tool
async def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for relevant documents.
    Use this tool whenever you need to answer a question based on stored information.
    
    Args:
        query: The search query string.
    """
    try:
        # Generate query embedding
        query_embedding = get_embedding(query, task_type="retrieval_query")
        
        # Search Qdrant (Higher limit because chunks are smaller now)
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=7 
        ).points
        
        if not search_results:
            return "No relevant documents found."
            
        # Format results for the LLM
        results_text = []
        for i, result in enumerate(search_results, 1):
            payload = result.payload or {}
            text = payload.get('text', 'No text content')
            # Only show relevant metadata to save tokens
            meta = {k: v for k, v in payload.items() if k in ['filename', 'page', 'chunk_index']}
            results_text.append(f"Source {i} (Score: {result.score:.2f}, ID: {result.id}):\nContent: {text}\nMetadata: {meta}")
            
        return "\n---\n".join(results_text)
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

# --- Agent Definition ---
agent = Agent(
    name="Expert RAG Assistant",
    instructions=(
        "You are an expert retrieval-augmented generation (RAG) agent. "
        "Your goal is to answer user questions comprehensively using the provided knowledge base. "
        "1.  **Analyze** the user's request. "
        "2.  **Retrieve:** ALWAYS use `search_knowledge_base` first. If the first search is insufficient, rephrase the query and search again (up to 2 times). "
        "3.  **Synthesize:** Formulate a clear, well-structured answer. Cite sources by referring to the context (e.g., 'According to the document...'). "
        "4.  **Honesty:** If you cannot find the answer in the retrieved context, say 'I couldn't find specific information about that in the documents.' Do not hallucinate."
    ),
    tools=[search_knowledge_base],
    model=model
)

async def get_agent_response(user_query: str) -> str:
    """Run the agent loop"""
    result = await Runner.run(agent, user_query)
    return result.final_output

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Optimized API...")
    ensure_collection_exists()
    yield
    print("🛑 Shutting down...")

# --- FastAPI App ---
app = FastAPI(
    title="Optimized Agentic RAG API",
    description="Autonomous RAG Agent with Chunking & Smart Retrieval",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class Document(BaseModel):
    text: str
    metadata: Optional[dict] = {}

class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class DocumentResponse(BaseModel):
    id: str
    message: str
    chunks_count: int

class ChatResponse(BaseModel):
    answer: str

# --- Routes ---
@app.get("/")
async def root():
    """Health check"""
    status = {"status": "healthy", "service": "Agentic RAG API"}
    try:
        qdrant_client.get_collections()
        status["qdrant"] = "connected"
    except Exception as e:
        status["qdrant"] = f"error: {str(e)}"
    return status

@app.post("/documents", response_model=DocumentResponse)
async def add_document(document: Document):
    """Add document (with Chunking) to Qdrant"""
    try:
        # OPTIMIZATION: Chunk text before embedding
        chunks = recursive_split_text(document.text)
        
        points = []
        doc_group_id = str(uuid.uuid4())
        
        for i, chunk_text in enumerate(chunks):
            embedding = get_embedding(chunk_text, task_type="retrieval_document")
            chunk_id = str(uuid.uuid4())
            
            payload = {
                "text": chunk_text,
                "doc_group_id": doc_group_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(document.metadata or {})
            }
            
            points.append(PointStruct(
                id=chunk_id,
                vector=embedding,
                payload=payload
            ))
            
        # Batch upsert
        if points:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            
        return DocumentResponse(
            id=doc_group_id, 
            message="Document processed and stored",
            chunks_count=len(points)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/batch")
async def add_documents_batch(documents: List[Document]):
    """Add multiple documents at once (with Chunking)"""
    try:
        all_points = []
        
        for doc in documents:
            if not doc.text or not doc.text.strip():
                continue
                
            chunks = recursive_split_text(doc.text)
            doc_group_id = str(uuid.uuid4())
            
            for i, chunk_text in enumerate(chunks):
                # Validation: Ensure chunk is not empty
                if not chunk_text or not chunk_text.strip():
                    continue

                # NOTE: In production, consider queuing these embedding calls 
                # to avoid Rate Limits. For now, we do sequential.
                embedding = get_embedding(chunk_text, task_type="retrieval_document")
                chunk_id = str(uuid.uuid4())
                
                payload = {
                    "text": chunk_text,
                    "doc_group_id": doc_group_id,
                    "chunk_index": i,
                    **(doc.metadata or {})
                }
                
                all_points.append(PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload
                ))
        
        if all_points:
            # Batch upsert to Qdrant (chunks of chunks if necessary, but list is okay here)
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=all_points
            )
        
        return {
            "message": f"Processed {len(documents)} files into {len(all_points)} chunks."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Agentic Chat Endpoint"""
    try:
        answer = await get_agent_response(request.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
