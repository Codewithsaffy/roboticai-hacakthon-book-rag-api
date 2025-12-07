"""
Vercel Serverless Function Entry Point for RAG Chat API (Agentic Version).
"""

import os
import uuid
import json
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
def get_embedding(text: str, task_type: str = "retrieval_document") -> List[float]:
    """Generate embedding using Gemini embedding model"""
    try:
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
        
        # Search Qdrant
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=5 
        ).points
        
        if not search_results:
            return "No relevant documents found."
            
        # Format results for the LLM
        results_text = []
        for i, result in enumerate(search_results, 1):
            payload = result.payload or {}
            text = payload.get('text', 'No text content')
            meta = {k: v for k, v in payload.items() if k != 'text'}
            results_text.append(f"Result {i} (Score: {result.score:.2f}):\nContent: {text}\nMetadata: {meta}")
            
        return "\n\n".join(results_text)
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

# --- Agent Definition ---
agent = Agent(
    name="RAG Assistant",
    instructions=(
        "You are an intelligent RAG assistant. "
        "Your goal is to answer user questions accurately using the knowledge base. "
        "1. receive the user query. "
        "2. ALWAYS use the `search_knowledge_base` tool to find relevant information first. "
        "3. Synthesize the answer from the tool output. "
        "4. If no information is found, admit it honestly."
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
    print("🚀 Starting Agentic RAG API...")
    ensure_collection_exists()
    yield
    print("🛑 Shutting down...")

# --- FastAPI App ---
app = FastAPI(
    title="Agentic RAG API",
    description="Autonomous RAG Agent using OpenAI Agents SDK + Qdrant",
    version="2.0.0",
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
    """Add document to Qdrant"""
    try:
        embedding = get_embedding(document.text, task_type="retrieval_document")
        doc_id = str(uuid.uuid4())
        
        point = PointStruct(
            id=doc_id,
            vector=embedding,
            payload={
                "text": document.text,
                **(document.metadata or {})
            }
        )
        
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        return DocumentResponse(id=doc_id, message="Document added")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Agentic Chat Endpoint"""
    try:
        # The Agent handles the entire retrieval logic autonomously
        answer = await get_agent_response(request.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
