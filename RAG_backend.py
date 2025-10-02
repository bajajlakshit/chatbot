from typing import TypedDict, Annotated, List, Dict, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
import os
import requests
import sqlite3
import json
from datetime import datetime
from dotenv import load_dotenv
import uuid
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
import fitz  # PyMuPDF for PDF processing
from docx import Document  # python-docx for DOCX processing
import re
from PIL import Image
import pytesseract  # OCR for images
import traceback

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class DocumentProcessor:
    """Handles document processing for various file types"""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.txt': self._process_text,
            '.md': self._process_text,
            '.jpg': self._process_image,
            '.jpeg': self._process_image,
            '.png': self._process_image,
            '.bmp': self._process_image,
            '.tiff': self._process_image
        }
    
    def process_document(self, file_path: str, file_content: bytes) -> Dict:
        """Process document and extract text"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext not in self.supported_formats:
                return {
                    'success': False,
                    'error': f'Unsupported file type: {file_ext}'
                }
            
            # Save content temporarily for processing
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Process based on file type
            text = self.supported_formats[file_ext](file_path)
            
            if not text or not text.strip():
                return {
                    'success': False,
                    'error': 'No text could be extracted from the document'
                }
            
            return {
                'success': True,
                'text': text,
                'file_type': file_ext,
                'text_length': len(text)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Document processing error: {str(e)}'
            }
    
    def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"PDF processing error: {e}")
            return ""
    
    def _process_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"DOCX processing error: {e}")
            return ""
    
    def _process_text(self, file_path: str) -> str:
        """Process text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                print(f"Text processing error: {e}")
                return ""
    
    def _process_image(self, file_path: str) -> str:
        """Extract text from images using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"Image OCR error: {e}")
            return ""

class VectorStore:
    """Simple vector store using TF-IDF and cosine similarity"""
    
    def __init__(self, storage_path: str = "vector_store.pkl"):
        self.storage_path = storage_path
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.vectors = None
        self.load_store()
    
    def add_document(self, doc_id: str, text: str, metadata: Dict):
        """Add document to vector store"""
        document = {
            'id': doc_id,
            'text': text,
            'metadata': metadata
        }
        
        self.documents.append(document)
        self._rebuild_vectors()
        self.save_store()
        
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if not self.documents or self.vectors is None:
            return []
        
        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only return relevant results
                    results.append({
                        'document': self.documents[idx],
                        'similarity': float(similarities[idx])
                    })
            
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _rebuild_vectors(self):
        """Rebuild vector representations"""
        if not self.documents:
            self.vectors = None
            return
        
        texts = [doc['text'] for doc in self.documents]
        self.vectors = self.vectorizer.fit_transform(texts)
    
    def save_store(self):
        """Save vector store to disk"""
        try:
            data = {
                'documents': self.documents,
                'vectorizer': self.vectorizer,
                'vectors': self.vectors
            }
            with open(self.storage_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving vector store: {e}")
    
    def load_store(self):
        """Load vector store from disk"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.vectorizer = data.get('vectorizer', TfidfVectorizer(stop_words='english', max_features=1000))
                    self.vectors = data.get('vectors', None)
                    print(f"Loaded {len(self.documents)} documents from vector store")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.documents = []
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            self.vectors = None

class RAGDatabase:
    """Enhanced database with document and thread management"""
    
    def __init__(self, db_path="rag_database.db"):
        self.db_path = db_path
        self._init_db()
        print(f"📁 RAG Database initialized: {db_path}")
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Conversations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    text_content TEXT NOT NULL,
                    text_length INTEGER NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Thread-Document associations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS thread_documents (
                    thread_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, doc_id)
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON conversations(thread_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_thread ON thread_documents(thread_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_doc ON thread_documents(doc_id)")
            
            conn.commit()
    
    def save_message(self, thread_id: str, message_type: str, content: str):
        """Save message to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO conversations (thread_id, message_type, content) VALUES (?, ?, ?)",
                (thread_id, message_type, content)
            )
            conn.commit()
    
    def get_conversation_history(self, thread_id: str) -> List[BaseMessage]:
        """Get conversation history for thread"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT message_type, content FROM conversations 
                WHERE thread_id = ? 
                ORDER BY timestamp ASC
            """, (thread_id,))
            
            messages = []
            for msg_type, content in cursor.fetchall():
                if msg_type == "human":
                    messages.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    messages.append(AIMessage(content=content))
            return messages
    
    def save_document(self, doc_id: str, filename: str, file_type: str, 
                     text_content: str, metadata: Dict) -> bool:
        """Save document to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, filename, file_type, text_content, text_length, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    doc_id, filename, file_type, text_content, 
                    len(text_content), json.dumps(metadata)
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving document: {e}")
            return False
    
    def associate_document_with_thread(self, thread_id: str, doc_id: str):
        """Associate document with thread"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO thread_documents (thread_id, doc_id)
                VALUES (?, ?)
            """, (thread_id, doc_id))
            conn.commit()
    
    def get_thread_documents(self, thread_id: str) -> List[Dict]:
        """Get documents associated with thread"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT d.id, d.filename, d.file_type, d.text_length, d.metadata, d.timestamp
                FROM documents d
                JOIN thread_documents td ON d.id = td.doc_id
                WHERE td.thread_id = ?
                ORDER BY d.timestamp DESC
            """, (thread_id,))
            
            documents = []
            for row in cursor.fetchall():
                doc = {
                    'id': row[0],
                    'filename': row[1],
                    'file_type': row[2],
                    'text_length': row[3],
                    'metadata': json.loads(row[4]) if row[4] else {},
                    'timestamp': row[5]
                }
                documents.append(doc)
            return documents
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, filename, file_type, text_length, metadata, timestamp
                FROM documents
                ORDER BY timestamp DESC
            """)
            
            documents = []
            for row in cursor.fetchall():
                doc = {
                    'id': row[0],
                    'filename': row[1],
                    'file_type': row[2],
                    'text_length': row[3],
                    'metadata': json.loads(row[4]) if row[4] else {},
                    'timestamp': row[5]
                }
                documents.append(doc)
            return documents
    
    def get_document_content(self, doc_id: str) -> Optional[str]:
        """Get document content by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT text_content FROM documents WHERE id = ?",
                (doc_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else None

class GeminiLLM:
    """Gemini API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"

    def invoke(self, messages):
        headers = {"Content-Type": "application/json"}
        
        # Better message handling
        prompt_parts = []
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                role_prefix = "Human: " if isinstance(msg, HumanMessage) else "Assistant: "
                prompt_parts.append(f"{role_prefix}{msg.content}")
        
        prompt_text = "\n".join(prompt_parts)
        if not prompt_text.strip():
            return AIMessage(content="I didn't receive any message to respond to.")
        
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000,
                "topP": 0.8,
                "topK": 10
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'candidates' in data and len(data['candidates']) > 0:
                content = data['candidates'][0]['content']['parts'][0]['text']
                return AIMessage(content=content)
            else:
                return AIMessage(content="No response from Gemini API")
        except requests.RequestException as e:
            print(f"Gemini API Error: {e}")
            return AIMessage(content=f"Error communicating with AI service: {e}")

class RAGChatbot:
    """Main RAG Chatbot with document context"""
    
    def __init__(self, llm, db, vector_store, doc_processor):
        self.llm = llm
        self.db = db
        self.vector_store = vector_store
        self.doc_processor = doc_processor
    
    def upload_document(self, file_path: str, file_content: bytes, thread_id: str = None):
        """Upload and process document"""
        try:
            # Process document
            result = self.doc_processor.process_document(file_path, file_content)
            
            if not result['success']:
                return result
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            filename = os.path.basename(file_path)
            
            # Prepare metadata
            metadata = {
                'file_name': filename,
                'file_type': result['file_type'],
                'upload_time': datetime.now().isoformat(),
                'text_length': result['text_length']
            }
            
            # Save to database
            if not self.db.save_document(
                doc_id, filename, result['file_type'], 
                result['text'], metadata
            ):
                return {'success': False, 'error': 'Failed to save document to database'}
            
            # Add to vector store
            self.vector_store.add_document(doc_id, result['text'], metadata)
            
            # Associate with thread if provided
            if thread_id:
                self.db.associate_document_with_thread(thread_id, doc_id)
            
            return {
                'success': True,
                'doc_id': doc_id,
                'filename': filename,
                'file_type': result['file_type'],
                'text_length': result['text_length'],
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Document upload error: {e}")
            return {'success': False, 'error': str(e)}
    
    def invoke(self, state):
        """Main RAG invoke method"""
        try:
            messages = state.get('messages', [])
            thread_id = state.get('thread_id', 'default')
            
            print(f"🔄 Processing RAG chat for thread: {thread_id}")
            
            # Find the user message
            user_message = None
            for msg in messages:
                if isinstance(msg, HumanMessage) and hasattr(msg, 'content'):
                    user_message = msg
                    break
            
            if not user_message:
                return {"messages": [AIMessage(content="No user message found.")]}
            
            print(f"📝 User query: {user_message.content}")
            
            # Get relevant documents using vector search
            search_results = self.vector_store.search(user_message.content, top_k=3)
            
            # Get conversation history
            history = self.db.get_conversation_history(thread_id)
            
            # Build context with retrieved documents
            context_parts = []
            
            if search_results:
                context_parts.append("=== RELEVANT DOCUMENTS ===")
                for i, result in enumerate(search_results, 1):
                    doc_text = result['document']['text']
                    # Limit document text to avoid token limits
                    doc_snippet = doc_text[:500] + "..." if len(doc_text) > 500 else doc_text
                    context_parts.append(f"Document {i} ({result['document']['metadata'].get('file_name', 'Unknown')}):")
                    context_parts.append(doc_snippet)
                    context_parts.append("")
                
                context_parts.append("=== END DOCUMENTS ===")
                context_parts.append("")
            
            # Add instruction for using documents
            if search_results:
                context_parts.append("Please answer the user's question based on the relevant documents provided above. If the documents don't contain relevant information, please say so and provide a general response.")
                context_parts.append("")
            
            # Prepare full context
            context = "\n".join(context_parts)
            
            # Prepare messages for LLM
            recent_history = history[-6:] if len(history) > 6 else history
            
            # Create enhanced user message with context
            if context.strip():
                enhanced_content = f"{context}\nUser Question: {user_message.content}"
                enhanced_message = HumanMessage(content=enhanced_content)
            else:
                enhanced_message = user_message
            
            context_messages = recent_history + [enhanced_message]
            
            print(f"🤖 Sending {len(context_messages)} messages to LLM with {len(search_results)} documents")
            
            # Get AI response
            ai_response = self.llm.invoke(context_messages)
            
            # Save messages to database (save original user message, not enhanced)
            self.db.save_message(thread_id, "human", user_message.content)
            print(f"💾 Saved user message to DB")
            
            if hasattr(ai_response, 'content') and ai_response.content:
                self.db.save_message(thread_id, "ai", ai_response.content)
                print(f"💾 Saved AI response to DB")
            
            return {"messages": [ai_response]}
            
        except Exception as e:
            print(f"❌ Error in RAG chatbot invoke: {e}")
            print(f"📍 Full traceback: {traceback.format_exc()}")
            
            error_response = AIMessage(content=f"I encountered an error: {str(e)}")
            return {"messages": [error_response]}

# Initialize all components
print("🔧 Initializing RAG components...")

# Initialize document processor
doc_processor = DocumentProcessor()
print("✅ Document processor initialized")

# Initialize vector store
vector_store = VectorStore()
print(f"✅ Vector store initialized with {len(vector_store.documents)} documents")

# Initialize database
rag_db = RAGDatabase()
print("✅ RAG database initialized")

# Initialize LLM
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found in environment variables")
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

llm = GeminiLLM(api_key=GEMINI_API_KEY)
print("✅ Gemini LLM initialized")

# Initialize RAG chatbot
rag_chatbot = RAGChatbot(llm, rag_db, vector_store, doc_processor)
print("✅ RAG chatbot initialized")

print("🚀 All RAG components initialized successfully!")