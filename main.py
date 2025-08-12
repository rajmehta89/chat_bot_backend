from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="PDF Q&A Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://your-frontend-domain.com",
        "https://chat-bot-frontend-0s4i.onrender.com",
        "https://*.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable not found")
co = cohere.Client(COHERE_API_KEY)


# Models
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


# Cohere-based Vector Store
class CohereVectorStore:
    def __init__(self, persist_dir: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.documents = []  # List[str]
        self.embeddings = []  # List[List[float]]
        self.load_from_disk()

    def add_documents(self, docs):
        self.documents = [doc.page_content for doc in docs]
        if self.documents:
            response = co.embed(texts=self.documents, model="embed-english-v2.0")
            self.embeddings = response.embeddings
            self.save_to_disk()

    def similarity_search(self, query: str, k: int = 3):
        if not self.documents or not self.embeddings:
            return []

        query_embedding = co.embed(texts=[query], model="embed-english-v2.0").embeddings[0]

        sims = cosine_similarity([query_embedding], self.embeddings)[0]

        top_indices = sims.argsort()[-k:][::-1]
        results = []
        for idx in top_indices:
            if sims[idx] > 0.2:  # similarity threshold
                doc = type('Document', (), {
                    'page_content': self.documents[idx],
                    'metadata': {'similarity': sims[idx]}
                })()
                results.append(doc)
        return results

    def save_to_disk(self):
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings
        }
        with open(self.persist_dir / 'vectorstore.json', 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load_from_disk(self):
        store_path = self.persist_dir / 'vectorstore.json'
        if store_path.exists():
            try:
                with open(store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.documents = data.get('documents', [])
                self.embeddings = data.get('embeddings', [])
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.documents = []
                self.embeddings = []


# Utility functions
def get_base_user_dir(email: str) -> Path:
    safe_email = email.replace("@", "_at_").replace(".", "_dot_")
    base_dir = Path("user_data") / safe_email
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def get_chats_metadata_path(user_dir: Path) -> Path:
    return user_dir / "chats.json"


def load_chats_metadata(user_dir: Path) -> List[Dict]:
    meta_path = get_chats_metadata_path(user_dir)
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except:
            return []
    return []


def save_chats_metadata(user_dir: Path, chats: List[Dict]):
    meta_path = get_chats_metadata_path(user_dir)
    meta_path.write_text(json.dumps(chats, indent=2), encoding="utf-8")


def get_chat_history_path(user_dir: Path, chat_id: str) -> Path:
    return user_dir / f"{chat_id}_history.json"


def load_chat_history(user_dir: Path, chat_id: str) -> List[Dict]:
    history_path = get_chat_history_path(user_dir, chat_id)
    if history_path.exists():
        try:
            return json.loads(history_path.read_text(encoding="utf-8"))
        except:
            return []
    return []


def save_chat_history(user_dir: Path, chat_id: str, history: List[Dict]):
    history_path = get_chat_history_path(user_dir, chat_id)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def add_message_to_history(user_dir: Path, chat_id: str, role: str, content: str) -> List[Dict]:
    history = load_chat_history(user_dir, chat_id)
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    history.append(message)
    if len(history) > 20:
        history = history[-20:]
    save_chat_history(user_dir, chat_id, history)
    return history


def get_chat_dir(user_dir: Path, chat_id: str) -> Path:
    chat_dir = user_dir / chat_id
    chat_dir.mkdir(parents=True, exist_ok=True)
    return chat_dir


def load_pdf_chunks(pdf_path: str):
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
    vectorstore = CohereVectorStore(persist_dir)
    vectorstore.add_documents(chunks)
    return vectorstore


def load_vectorstore(persist_dir: str):
    return CohereVectorStore(persist_dir)


def generate_chat_name_from_pdf(chunks):
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
    if not history:
        return ""
    recent_history = history[-10:] if len(history) > 10 else history
    context_parts = []
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        context_parts.append(f"{role}: {msg['content']}")
    return "\n".join(context_parts)


def answer_query_with_context(question: str, relevant_docs, chat_history: List[Dict], is_greeting: bool = False):
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
            prompt = f"""I am a helpful AI assistant.

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

        prompt = f"""You are a helpful PDF assistant.

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
    try:
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
    user_dir = get_base_user_dir(user_email)
    chats = load_chats_metadata(user_dir)
    return chats


@app.post("/chats/{user_email}", response_model=Chat)
async def create_chat(user_email: str, chat_data: ChatCreate):
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

    if len(chats) > 10:
        oldest = chats.pop(-1)
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
    user_dir = get_base_user_dir(user_email)
    messages = load_chat_history(user_dir, chat_id)
    return messages


@app.post("/chats/{user_email}/{chat_id}/message")
async def send_message(user_email: str, chat_id: str, message_data: MessageRequest):
    user_dir = get_base_user_dir(user_email)
    chats = load_chats_metadata(user_dir)

    chat = next((c for c in chats if c["chat_id"] == chat_id), None)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if not message_data.is_greeting:
        add_message_to_history(user_dir, chat_id, "user", message_data.message)

    relevant_docs = []
    chat_dir = get_chat_dir(user_dir, chat_id)
    if chat["file_name"] and chat_dir.exists():
        try:
            vectorstore = load_vectorstore(str(chat_dir))
            relevant_docs = vectorstore.similarity_search(message_data.message, k=3)
        except Exception as e:
            print(f"Error loading vectorstore: {e}")

    chat_history = load_chat_history(user_dir, chat_id)
    if not message_data.is_greeting and chat_history:
        chat_history = chat_history[:-1]

    ai_response = answer_query_with_context(
        message_data.message,
        relevant_docs,
        chat_history,
        message_data.is_greeting
    )

    add_message_to_history(user_dir, chat_id, "assistant", ai_response)

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
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    user_dir = get_base_user_dir(user_email)
    chats = load_chats_metadata(user_dir)

    chat = next((c for c in chats if c["chat_id"] == chat_id), None)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    try:
        chat_dir = get_chat_dir(user_dir, chat_id)
        pdf_path = chat_dir / file.filename

        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = load_pdf_chunks(str(pdf_path))
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to process PDF")

        smart_name = generate_chat_name_from_pdf(chunks)

        vectorstore = create_vectorstore(chunks, persist_dir=str(chat_dir))

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
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
