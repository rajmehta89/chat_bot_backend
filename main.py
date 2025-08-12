from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import uuid
import json
import shutil
from pathlib import Path
from datetime import datetime
import cohere
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="PDF Q&A Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://your-frontend-domain.com",
        "https://chat-bot-frontend-wbpt.onrender.com",
        "https://chat-bot-backend-mhmj.onrender.com",
        "https://chat-bot-frontend-awui.onrender.com",
        "https://*.onrender.com",
        "https://*.vercel.app"  # Allow Vercel deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Cohere client
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    print("Warning: COHERE_API_KEY not found. PDF processing will be limited.")
    co = None
else:
    co = cohere.Client(cohere_api_key)

# Pydantic models
class ChatCreate(BaseModel):
    chat_name: str

class ChatUpdate(BaseModel):
    chat_name: str

class MessageRequest(BaseModel):
    message: str
    is_greeting: Optional[bool] = False

class Message(BaseModel):
    role: str
    content: str
    timestamp: str

class Chat(BaseModel):
    chat_id: str
    chat_name: str
    file_name: str
    created_at: str
    last_updated: str

# Helper classes
class CohereEmbeddings:
    """Custom Cohere embeddings wrapper compatible with LangChain-like interface."""

    def embed_documents(self, texts):
        if not co:
            return [[] for _ in texts]
        try:
            response = co.embed(texts=texts, model="embed-english-v2.0")
            return response.embeddings
        except Exception as e:
            print(f"Error during embedding: {e}")
            return [[] for _ in texts]

    def embed_query(self, text):
        if not co:
            return []
        try:
            res = co.embed(texts=[text], model="embed-english-v2.0")
            return res.embeddings[0]
        except Exception as e:
            print(f"Error during query embedding: {e}")
            return []

# Utility functions
def get_base_user_dir(email: str) -> Path:
    """Return base dir path for storing user data."""
    safe_email = email.replace("@", "_at_").replace(".", "_dot_")
    base_dir = Path("user_data") / safe_email
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def get_chats_metadata_path(user_dir: Path) -> Path:
    return user_dir / "chats.json"

def load_chats_metadata(user_dir: Path) -> List[Dict]:
    """Load chats metadata JSON or return empty list if none."""
    meta_path = get_chats_metadata_path(user_dir)
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except:
            return []
    return []

def save_chats_metadata(user_dir: Path, chats: List[Dict]):
    """Save chats metadata list as JSON."""
    meta_path = get_chats_metadata_path(user_dir)
    meta_path.write_text(json.dumps(chats, indent=2), encoding="utf-8")

def get_chat_history_path(user_dir: Path, chat_id: str) -> Path:
    """Return path for chat history JSON file."""
    return user_dir / f"{chat_id}_history.json"

def load_chat_history(user_dir: Path, chat_id: str) -> List[Dict]:
    """Load chat history for a specific chat."""
    history_path = get_chat_history_path(user_dir, chat_id)
    if history_path.exists():
        try:
            return json.loads(history_path.read_text(encoding="utf-8"))
        except:
            return []
    return []

def save_chat_history(user_dir: Path, chat_id: str, history: List[Dict]):
    """Save chat history for a specific chat."""
    history_path = get_chat_history_path(user_dir, chat_id)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

def add_message_to_history(user_dir: Path, chat_id: str, role: str, content: str) -> List[Dict]:
    """Add a message to chat history."""
    history = load_chat_history(user_dir, chat_id)
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    history.append(message)

    # Keep only last 20 messages
    if len(history) > 20:
        history = history[-20:]

    save_chat_history(user_dir, chat_id, history)
    return history

def get_chat_dir(user_dir: Path, chat_id: str) -> Path:
    """Return folder path for a specific chat."""
    chat_dir = user_dir / chat_id
    chat_dir.mkdir(parents=True, exist_ok=True)
    return chat_dir

def load_pdf_chunks(pdf_path: str):
    """Load PDF and split into chunks."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)
        return chunks
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

def get_vectorstore(chunks, collection_name):
    """Create a new Chroma vector store from document chunks."""
    embeddings = CohereEmbeddings()
    return Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=collection_name,
        persist_directory=None  # In-memory for simplicity
    )

def answer_query_with_context(question, relevant_docs, chat_history, is_greeting=False):
    """Use retrieved chunks as context and get answer from Cohere chat model."""
    if not co:
        return "Sorry, the AI service is currently unavailable. Please check your API configuration."

    if is_greeting:
        return "Hello! I'm your PDF assistant. Upload a PDF document and I'll help you ask questions about its content. I can remember our conversation and provide context-aware responses."

    context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""

    # Build conversation context
    conversation_context = ""
    if chat_history:
        recent_messages = chat_history[-6:]  # Last 6 messages for context
        for msg in recent_messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful PDF assistant. Answer the question based on the provided context from the uploaded PDF and previous conversation.

Previous conversation:
{conversation_context}

Context from PDF:
{context}

Current question: {question}

Instructions:
- If the context contains relevant information, provide a helpful answer
- If the context doesn't contain the specific information asked about, explain what the PDF actually contains instead
- Reference previous conversation when relevant
- Be conversational and helpful, not overly rigid
- If no PDF context is available, let the user know they need to upload a PDF first

Answer:"""

    try:
        response = co.chat(
            model="command-xlarge-nightly",
            message=prompt,
            max_tokens=500,
            temperature=0.3
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer: {e}"

# API Routes
@app.get("/")
async def root():
    return {
        "message": "PDF Q&A Chatbot API is running",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Test Cohere connection
        co.chat(model="command-xlarge-nightly", message="test", max_tokens=1)
        cohere_status = "connected"
    except Exception as e:
        cohere_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "cohere": cohere_status,
            "filesystem": "accessible" if Path("user_data").exists() or Path("user_data").mkdir(parents=True, exist_ok=True) else "error"
        }
    }

@app.get("/chats/{user_email}", response_model=List[Chat])
async def get_user_chats(user_email: str):
    """Get all chats for a user."""
    user_dir = get_base_user_dir(user_email)
    chats = load_chats_metadata(user_dir)
    return chats

@app.post("/chats/{user_email}", response_model=Chat)
async def create_chat(user_email: str, chat_data: ChatCreate):
    """Create a new chat for a user."""
    user_dir = get_base_user_dir(user_email)
    chats = load_chats_metadata(user_dir)

    new_chat_id = str(uuid.uuid4())
    new_chat = {
        "chat_id": new_chat_id,
        "chat_name": chat_data.chat_name,
        "file_name": "",
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }

    chats.insert(0, new_chat)

    # Limit to max 10 chats
    if len(chats) > 10:
        oldest = chats.pop(-1)
        # Clean up old chat files
        old_chat_dir = get_chat_dir(user_dir, oldest["chat_id"])
        if old_chat_dir.exists():
            shutil.rmtree(old_chat_dir, ignore_errors=True)
        old_history_path = get_chat_history_path(user_dir, oldest["chat_id"])
        if old_history_path.exists():
            old_history_path.unlink()

    save_chats_metadata(user_dir, chats)
    return new_chat

@app.put("/chats/{user_email}/{chat_id}")
async def update_chat(user_email: str, chat_id: str, chat_data: ChatUpdate):
    """Update chat name."""
    user_dir = get_base_user_dir(user_email)
    chats = load_chats_metadata(user_dir)

    for chat in chats:
        if chat["chat_id"] == chat_id:
            chat["chat_name"] = chat_data.chat_name
            chat["last_updated"] = datetime.now().isoformat()
            save_chats_metadata(user_dir, chats)
            return {"message": "Chat updated successfully"}

    raise HTTPException(status_code=404, detail="Chat not found")

@app.get("/chats/{user_email}/{chat_id}/messages", response_model=List[Message])
async def get_chat_messages(user_email: str, chat_id: str):
    """Get messages for a specific chat."""
    user_dir = get_base_user_dir(user_email)
    messages = load_chat_history(user_dir, chat_id)
    return messages

@app.post("/chats/{user_email}/{chat_id}/message")
async def send_message(user_email: str, chat_id: str, message_data: MessageRequest):
    """Send a message and get AI response."""
    user_dir = get_base_user_dir(user_email)
    chats = load_chats_metadata(user_dir)

    # Find the chat
    chat = next((c for c in chats if c["chat_id"] == chat_id), None)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Add user message to history (unless it's a greeting)
    if not message_data.is_greeting:
        add_message_to_history(user_dir, chat_id, "user", message_data.message)

    # Load vectorstore if available
    relevant_docs = []
    chat_dir = get_chat_dir(user_dir, chat_id)
    if chat["file_name"] and chat_dir.exists():
        try:
            # Placeholder for loading vectorstore, as it's in-memory
            pass
        except Exception as e:
            print(f"Error loading vectorstore: {e}")

    # Get chat history (excluding current message if not greeting)
    chat_history = load_chat_history(user_dir, chat_id)
    if not message_data.is_greeting and chat_history:
        chat_history = chat_history[:-1]  # Exclude the just-added user message

    # Generate AI response
    ai_response = answer_query_with_context(
        message_data.message,
        relevant_docs,
        chat_history,
        message_data.is_greeting
    )

    # Add AI response to history
    add_message_to_history(user_dir, chat_id, "assistant", ai_response)

    # Update chat timestamp
    for c in chats:
        if c["chat_id"] == chat_id:
            c["last_updated"] = datetime.now().isoformat()
            break
    save_chats_metadata(user_dir, chats)

    return {
        "assistant_message": {
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        },
        "updated_chat": chat
    }

@app.post("/chats/{user_email}/{chat_id}/upload")
async def upload_pdf(user_email: str, chat_id: str, file: UploadFile = File(...)):
    """Upload and process a PDF file for a chat."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    user_dir = get_base_user_dir(user_email)
    chats = load_chats_metadata(user_dir)

    # Find the chat
    chat = next((c for c in chats if c["chat_id"] == chat_id), None)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    try:
        # Save uploaded file
        chat_dir = get_chat_dir(user_dir, chat_id)
        pdf_path = chat_dir / file.filename

        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF
        chunks = load_pdf_chunks(str(pdf_path))
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to process PDF")

        # Create vectorstore
        collection_name = f"chat_{chat_id}_{uuid.uuid4().hex[:8]}"
        vectorstore = get_vectorstore(chunks, collection_name)

        # Update chat metadata
        for c in chats:
            if c["chat_id"] == chat_id:
                c["chat_name"] = f"Chat about {file.filename}"
                c["file_name"] = file.filename
                c["last_updated"] = datetime.now().isoformat()
                break

        save_chats_metadata(user_dir, chats)

        return {
            "message": "PDF uploaded and processed successfully",
            "updated_chat": chat
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/login")
async def login_user(user_data: dict):
    """Login or register a user."""
    email = user_data.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    # Create user directory if it doesn't exist
    user_dir = get_base_user_dir(email)

    return {
        "message": "Login successful",
        "user_email": email,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user_email: str = Form(...)):
    """Upload a PDF file and create a new chat."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    user_dir = get_base_user_dir(user_email)

    # Create new chat
    new_chat_id = str(uuid.uuid4())
    chat_name = f"Chat about {file.filename}"

    try:
        # Save uploaded file
        chat_dir = get_chat_dir(user_dir, new_chat_id)
        pdf_path = chat_dir / file.filename

        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF
        chunks = load_pdf_chunks(str(pdf_path))
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to process PDF")

        # Create vectorstore
        collection_name = f"chat_{new_chat_id}_{uuid.uuid4().hex[:8]}"
        vectorstore = get_vectorstore(chunks, collection_name)

        # Save vectorstore (in a real app, you'd persist this properly)
        # For now, we'll recreate it when needed

        # Update chats metadata
        chats = load_chats_metadata(user_dir)
        new_chat = {
            "chat_id": new_chat_id,
            "chat_name": chat_name,
            "file_name": file.filename,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }

        chats.insert(0, new_chat)
        save_chats_metadata(user_dir, chats)

        return {
            "message": "File uploaded and processed successfully",
            "chat": new_chat,
            "chunks_processed": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Railway sets PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
