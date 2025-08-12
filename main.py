from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
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
        "https://*.vercel.app"  # Allow Vercel deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Cohere client
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable not found")

co = cohere.Client(COHERE_API_KEY)

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
    """Wrap Cohere embedding calls for LangChain compatibility."""
    
    def embed_documents(self, texts):
        try:
            response = co.embed(texts=texts, model="embed-english-v2.0")
            return response.embeddings
        except Exception as e:
            print(f"Embedding error: {e}")
            return [[] for _ in texts]
    
    def embed_query(self, text):
        try:
            response = co.embed(texts=[text], model="embed-english-v2.0")
            return response.embeddings[0]
        except Exception as e:
            print(f"Query embedding error: {e}")
            return []

class SimpleVectorStore:
    def __init__(self, persist_dir: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.documents = []
        self.vectors = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.load_from_disk()
    
    def add_documents(self, docs):
        """Add documents to the vector store."""
        self.documents = [doc.page_content for doc in docs]
        if self.documents:
            self.vectors = self.vectorizer.fit_transform(self.documents)
            self.save_to_disk()
    
    def similarity_search(self, query: str, k: int = 3):
        """Search for similar documents."""
        if not self.documents or self.vectors is None:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top k most similar documents
        top_indices = similarities.argsort()[-k:][::-1]
        
        # Return documents with similarity scores
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                # Create a simple document-like object
                doc = type('Document', (), {
                    'page_content': self.documents[idx],
                    'metadata': {'similarity': similarities[idx]}
                })()
                results.append(doc)
        
        return results
    
    def save_to_disk(self):
        """Save vector store to disk."""
        data = {
            'documents': self.documents,
            'vectorizer_vocab': self.vectorizer.vocabulary_ if hasattr(self.vectorizer, 'vocabulary_') else None
        }
        with open(self.persist_dir / 'vectorstore.json', 'w') as f:
            json.dump(data, f)
    
    def load_from_disk(self):
        """Load vector store from disk."""
        store_path = self.persist_dir / 'vectorstore.json'
        if store_path.exists():
            try:
                with open(store_path, 'r') as f:
                    data = json.load(f)
                self.documents = data.get('documents', [])
                if self.documents:
                    self.vectors = self.vectorizer.fit_transform(self.documents)
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.documents = []
                self.vectors = None

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
    """Load PDF and split into text chunks."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)
        return chunks
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

def create_vectorstore(chunks, persist_dir: str):
    vectorstore = SimpleVectorStore(persist_dir)
    vectorstore.add_documents(chunks)
    return vectorstore

def load_vectorstore(persist_dir: str):
    return SimpleVectorStore(persist_dir)

def generate_chat_name_from_pdf(chunks):
    """Generate a descriptive chat name based on PDF content."""
    if not chunks:
        return "New Chat"
    
    sample_text = " ".join([chunk.page_content[:200] for chunk in chunks[:3]])
    prompt = f"""Based on this PDF content, suggest a short, descriptive name (max 4 words) for this document:

Content: {sample_text[:500]}

Instructions:
- If it's a resume, use: "Resume - [Name]"
- If it's a technical document, use the main topic
- If it's Lorem Ipsum or placeholder text, use: "Sample Document"
- Keep it concise and professional
- Only return the name, nothing else

Name:"""
    
    try:
        response = co.chat(
            model="command-xlarge-nightly",
            message=prompt,
            max_tokens=50,
            temperature=0.3
        )
        name = response.text.strip().replace('"', '').replace("'", "")
        if len(name) > 30:
            name = name[:30] + "..."
        return name if name else "New Chat"
    except Exception:
        return "New Chat"

def format_chat_history_for_context(history: List[Dict]) -> str:
    """Format chat history into a readable context string."""
    if not history:
        return ""
    
    recent_history = history[-10:] if len(history) > 10 else history
    context_parts = []
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        context_parts.append(f"{role}: {msg['content']}")
    
    return "\n".join(context_parts)

def answer_query_with_context(question: str, relevant_docs, chat_history: List[Dict], is_greeting: bool = False):
    """Generate an answer based on question, relevant PDF documents, and chat history."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M")
    
    if is_greeting:
        return f"Hello! I'm your AI assistant. Today is {current_date} and it's currently {current_time}. I can help you with questions about PDFs you upload, provide information based on current context, or just have a general conversation. What would you like to talk about today?"
    
    if not relevant_docs:
        if chat_history:
            history_context = format_chat_history_for_context(chat_history)
            prompt = f"""You are a helpful AI assistant. Today's date is {current_date} and current time is {current_time}.

Previous conversation:
{history_context}

Current question: {question}

Instructions:
- Provide helpful responses based on our conversation history
- You can discuss future events but acknowledge uncertainty when appropriate
- If asked about current events or recent information, mention that your knowledge has a cutoff date
- Be conversational and remember what we've talked about
- If the question is about a PDF and no PDF content is available, let the user know they can upload a PDF

Please provide a helpful response:"""
        else:
            prompt = f"""i am helpful AI assistant.

Question: {question}

Instructions:
- Provide a helpful and informative response
- You can discuss future events but acknowledge uncertainty when appropriate
- Be friendly and conversational
- If asked about uploading PDFs, explain that they can upload PDF files to ask specific questions about them

Please provide a helpful response:"""
    else:
        pdf_context = "\n\n".join([doc.page_content for doc in relevant_docs])
        history_context = format_chat_history_for_context(chat_history)
        
        prompt = f"""i am helpful PDF assistant.

PDF Content:
{pdf_context}

Previous conversation:
{history_context}

Current question: {question}

Instructions:
- Use both the PDF content and conversation history to provide a comprehensive answer
- If the question refers to something we discussed before, acknowledge that context
- If the PDF content is relevant, prioritize it in your answer
- Be conversational and remember what we've talked about
- You can reference future dates but acknowledge uncertainty when appropriate
- If the PDF contains Lorem Ipsum or placeholder text, mention that and suggest uploading a real document

Answer:"""
    
    try:
        response = co.chat(
            model="command-xlarge-nightly",
            message=prompt,
            max_tokens=800,
            temperature=0.3
        )
        return response.text.strip()
    except Exception as e:
        return f"I apologize, but I encountered an error while generating a response. Please try again. Error: {str(e)}"

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
            vectorstore = load_vectorstore(str(chat_dir))
            relevant_docs = vectorstore.similarity_search(message_data.message, k=3)
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
        
        # Generate smart chat name
        smart_name = generate_chat_name_from_pdf(chunks)
        
        # Create vectorstore
        vectorstore = create_vectorstore(chunks, persist_dir=str(chat_dir))
        
        # Update chat metadata
        for c in chats:
            if c["chat_id"] == chat_id:
                c["chat_name"] = smart_name
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Railway sets PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
