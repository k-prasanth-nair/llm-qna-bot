from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from backend.embedding_service import FAQEmbeddingService
from sse_starlette.sse import EventSourceResponse
import logging
import asyncio
import json
import os
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv()

print(f"API Key: ************** {os.getenv('OPENAI_API_KEY')}")
# API Models
class Question(BaseModel):
    text: str = Field(..., description="The question text to search for answers")

class Answer(BaseModel):
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The answer to the question")
    similarity_score: float = Field(..., description="Confidence score of the match")

# Configuration
class Settings:
    CORS_ORIGINS: List[str] = ["http://localhost:8001"]
    CHROMA_PERSIST_DIR: str = "chroma_db"
    COLLECTION_NAME: str = "product_faq_collection"
    MAX_RESULTS: int = 3

# Application state
class AppState:
    def __init__(self):
        self.embedding_service: Optional[FAQEmbeddingService] = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - setup and cleanup."""
    try:
        # Initialize services
        app_state.embedding_service = FAQEmbeddingService(
            persist_directory=Settings.CHROMA_PERSIST_DIR,
            collection_name=Settings.COLLECTION_NAME
        )
        logger.info("Embedding service initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")
        raise
    finally:
        # Cleanup (if needed)
        logger.info("Shutting down application")

# Initialize FastAPI app
app = FastAPI(
    title="FAQ Search API",
    description="API for searching relevant FAQ answers using embeddings",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only. In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
async def get_embedding_service() -> FAQEmbeddingService:
    """Dependency injection for embedding service."""
    if app_state.embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not initialized"
        )
    return app_state.embedding_service

# Additional API Models
class IngestionResponse(BaseModel):
    status: str
    message: str
    file_name: str

# ... (keep existing initialization code until routes) ...

@app.post("/api/ingest_csv", response_model=IngestionResponse)
async def ingest_csv(
    embedding_service: FAQEmbeddingService = Depends(get_embedding_service)
) -> IngestionResponse:
    """
    Ingest FAQ data from a CSV file.
    
    Args:
        file: Uploaded CSV file
        embedding_service: Injected embedding service
    
    Returns:
        IngestionResponse with status and message
    
    Raises:
        HTTPException: If ingestion fails or file format is invalid
    """
    file_path = "/Users/prasanth/sandbox/llm-qna-bot/data/product_faq.csv"
    
    try:
        # Debug: Check if source file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found: {file_path}")
            
        print(f"Source file exists at: {file_path}")
        
        # Create UploadFile
        file = UploadFile(
            filename="product_faq.csv",
            file=open(file_path, "rb")
        )

        # Read content
        content = await file.read()
        
        # Save to temp file with proper cleanup
        temp_file_path = "temp_product_faq.csv"
        try:
            with open(temp_file_path, "wb") as buffer:
                buffer.write(content)
            
            print(f"Temp file created at: {temp_file_path}")
            print(f"Temp file size: {os.path.getsize(temp_file_path)} bytes")
            
            # Process the CSV file
            embedding_service.ingest_faq_csv(
                csv_file_path=temp_file_path,
                question_column="Question",
                answer_column="Answer"
            )
            
            return IngestionResponse(
            status="success",
            message="FAQ data ingested successfully",
            file_name="product_faq.csv"
        )
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"Cleaned up temp file: {temp_file_path}")
                
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Close the original file
        file.file.close()

@app.get("/api/search")
async def search_answers_sse(
    query: str,
    embedding_service: FAQEmbeddingService = Depends(get_embedding_service)
):
    """Search for relevant FAQ answers using Server-Sent Events."""
    async def event_generator():
        try:
            logger.info(f"Starting search for query: {query}")
            
            yield {
                "event": "start",
                "data": json.dumps({"status": "searching"})
            }
            
            # Get relevant documents
            relevant_docs = embedding_service.get_relevant_chunks(
                query,
                k=Settings.MAX_RESULTS
            )
            
            logger.info(f"Found {len(relevant_docs)} documents")

            # Lower the threshold and check the actual score value
            SIMILARITY_THRESHOLD = 0.3  # Lowered threshold
            filtered_docs = []
            
            for doc in relevant_docs:
                score = doc.metadata.get('score', 0)
                logger.info(f"Document score: {score}")  # Debug log
                if score > SIMILARITY_THRESHOLD:
                    filtered_docs.append(doc)
            
            logger.info(f"Filtered to {len(filtered_docs)} relevant documents")

            if filtered_docs:
                best_doc = filtered_docs[0]
                content = best_doc.page_content
                logger.info(f"Best match content: {content}")
                
                try:
                    _, answer = content.split("\nA: ")
                    result = {
                        "answer": answer.strip(),
                        "similarity_score": best_doc.metadata.get('score', 0.0)
                    }
                    
                    logger.info(f"Sending result: {result}")
                    yield {
                        "event": "result",
                        "data": json.dumps(result)
                    }
                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": str(e)})
                    }
            else:
                logger.info("No relevant documents found")
                yield {
                    "event": "result",
                    "data": json.dumps({
                        "answer": "I'm sorry, I couldn't find a relevant answer to your question.",
                        "similarity_score": 0.0
                    })
                }

            yield {
                "event": "end",
                "data": json.dumps({"status": "completed"})
            }

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())

