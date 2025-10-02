import streamlit as st
import sqlite3
import uuid
import json
import os
import requests
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pickle
import tempfile
import base64

# Document processing imports
try:
    import fitz  # PyMuPDF for PDF processing
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document  # python-docx for DOCX processing
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract  # OCR for images
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Vector store imports
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
load_dotenv()

# Page config
st.set_page_config(
    page_title="RAG ChatBot Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196F3;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        border-left-color: #667eea;
    }
    
    .document-card {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state first"""
    if 'current_thread_id' not in st.session_state:
        st.session_state.current_thread_id = str(uuid.uuid4())
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    # Initialize database states
    if 'db_conversations' not in st.session_state:
        st.session_state.db_conversations = {}
    if 'db_documents' not in st.session_state:
        st.session_state.db_documents = {}
    if 'db_thread_docs' not in st.session_state:
        st.session_state.db_thread_docs = {}

class DocumentProcessor:
    """Handles document processing for various file types"""
    
    def __init__(self):
        self.supported_formats = {
            '.txt': self._process_text,
            '.md': self._process_text,
        }
        
        if PDF_AVAILABLE:
            self.supported_formats['.pdf'] = self._process_pdf
        if DOCX_AVAILABLE:
            self.supported_formats['.docx'] = self._process_docx
        if OCR_AVAILABLE:
            self.supported_formats.update({
                '.jpg': self._process_image,
                '.jpeg': self._process_image,
                '.png': self._process_image,
                '.bmp': self._process_image,
                '.tiff': self._process_image
            })
    
    def process_document(self, file_content: bytes, filename: str) -> Dict:
        """Process document and extract text"""
        try:
            file_ext = Path(filename).suffix.lower()
            
            if file_ext not in self.supported_formats:
                return {
                    'success': False,
                    'error': f'Unsupported file type: {file_ext}. Supported: {", ".join(self.supported_formats.keys())}'
                }
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Process based on file type
                text = self.supported_formats[file_ext](tmp_file_path)
                
                if not text or not text.strip():
                    return {
                        'success': False,
                        'error': 'No text could be extracted from the document'
                    }
                
                return {
                    'success': True,
                    'text': text,
                    'file_type': file_ext,
                    'text_length': len(text),
                    'filename': filename
                }
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
            
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

class SimpleVectorStore:
    """In-memory vector store using TF-IDF"""
    
    def __init__(self):
        self.documents = {}
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.vectors = None
        self.doc_ids = []
    
    def add_document(self, doc_id: str, text: str, metadata: Dict):
        """Add document to vector store"""
        self.documents[doc_id] = {
            'id': doc_id,
            'text': text,
            'metadata': metadata
        }
        self._rebuild_vectors()
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
                if similarities[idx] > 0.1:  # Only return relevant results
                    doc_id = self.doc_ids[idx]
                    results.append({
                        'document': self.documents[doc_id],
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
            self.doc_ids = []
            return
        
        texts = []
        self.doc_ids = []
        for doc_id, doc in self.documents.items():
            texts.append(doc['text'])
            self.doc_ids.append(doc_id)
        
        self.vectors = self.vectorizer.fit_transform(texts)

class SimpleDatabase:
    """In-memory database with session state persistence"""
    
    def __init__(self):
        # Session state is already initialized in init_session_state()
        pass
    
    def save_message(self, thread_id: str, message_type: str, content: str):
        """Save message to session state"""
        if thread_id not in st.session_state.db_conversations:
            st.session_state.db_conversations[thread_id] = []
        
        st.session_state.db_conversations[thread_id].append({
            'type': message_type,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_conversation_history(self, thread_id: str) -> List[Dict]:
        """Get conversation history for thread"""
        return st.session_state.db_conversations.get(thread_id, [])
    
    def save_document(self, doc_id: str, filename: str, file_type: str, 
                     text_content: str, metadata: Dict) -> bool:
        """Save document to session state"""
        try:
            st.session_state.db_documents[doc_id] = {
                'id': doc_id,
                'filename': filename,
                'file_type': file_type,
                'text_content': text_content,
                'text_length': len(text_content),
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }
            return True
        except Exception as e:
            print(f"Error saving document: {e}")
            return False
    
    def associate_document_with_thread(self, thread_id: str, doc_id: str):
        """Associate document with thread"""
        if thread_id not in st.session_state.db_thread_docs:
            st.session_state.db_thread_docs[thread_id] = []
        
        if doc_id not in st.session_state.db_thread_docs[thread_id]:
            st.session_state.db_thread_docs[thread_id].append(doc_id)
    
    def get_thread_documents(self, thread_id: str) -> List[Dict]:
        """Get documents associated with thread"""
        doc_ids = st.session_state.db_thread_docs.get(thread_id, [])
        documents = []
        
        for doc_id in doc_ids:
            if doc_id in st.session_state.db_documents:
                documents.append(st.session_state.db_documents[doc_id])
        
        return documents
    
    def get_all_threads(self) -> List[Dict]:
        """Get all conversation threads"""
        threads = []
        for thread_id, messages in st.session_state.db_conversations.items():
            docs = self.get_thread_documents(thread_id)
            threads.append({
                "thread_id": thread_id,
                "message_count": len(messages),
                "document_count": len(docs),
                "last_activity": messages[-1]['timestamp'] if messages else None
            })
        
        # Sort by last activity
        threads.sort(key=lambda x: x['last_activity'] or '', reverse=True)
        return threads
    
    def delete_thread(self, thread_id: str):
        """Delete a conversation thread"""
        if thread_id in st.session_state.db_conversations:
            del st.session_state.db_conversations[thread_id]
        if thread_id in st.session_state.db_thread_docs:
            del st.session_state.db_thread_docs[thread_id]

class SimpleLLM:
    """Simple LLM interface using Gemini API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    def generate_response(self, messages: List[Dict]) -> str:
        """Generate response from messages"""
        headers = {
            "Content-Type": "application/json"
        }
        
        # Format messages for Gemini
        prompt_parts = []
        for msg in messages:
            role_prefix = "Human: " if msg['type'] == 'human' else "Assistant: "
            prompt_parts.append(f"{role_prefix}{msg['content']}")
        
        prompt_text = "\n".join(prompt_parts)
        if not prompt_text.strip():
            return "I didn't receive any message to respond to."
        
        # Add system prompt for better responses
        system_prompt = "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."
        full_prompt = f"{system_prompt}\n\n{prompt_text}\n\nAssistant:"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": full_prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
                "topP": 0.8,
                "topK": 40
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        try:
            # Make request with API key as parameter
            params = {"key": self.api_key}
            response = requests.post(
                self.base_url, 
                json=payload, 
                headers=headers, 
                params=params,
                timeout=30
            )
            
            # Debug information
            if response.status_code != 200:
                print(f"API Response Status: {response.status_code}")
                print(f"API Response: {response.text}")
            
            response.raise_for_status()
            data = response.json()
            
            if 'candidates' in data and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    return candidate['content']['parts'][0]['text']
                else:
                    return "No content in AI response"
            else:
                return "No candidates in AI response"
                
        except requests.RequestException as e:
            error_msg = f"API Error: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\nStatus Code: {e.response.status_code}"
                error_msg += f"\nResponse: {e.response.text}"
            print(error_msg)
            return f"Error communicating with AI service. Please check your API key and try again."
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(error_msg)
            return error_msg

class RAGSystem:
    """Complete RAG system"""
    
    def __init__(self, llm, db, vector_store, doc_processor):
        self.llm = llm
        self.db = db
        self.vector_store = vector_store
        self.doc_processor = doc_processor
    
    def upload_document(self, file_content: bytes, filename: str, thread_id: str):
        """Upload and process document"""
        try:
            result = self.doc_processor.process_document(file_content, filename)
            
            if not result['success']:
                return result
            
            doc_id = str(uuid.uuid4())
            
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
                return {'success': False, 'error': 'Failed to save document'}
            
            # Add to vector store
            self.vector_store.add_document(doc_id, result['text'], metadata)
            
            # Associate with thread
            self.db.associate_document_with_thread(thread_id, doc_id)
            
            return {
                'success': True,
                'doc_id': doc_id,
                'filename': filename,
                'file_type': result['file_type'],
                'text_length': result['text_length']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def chat(self, user_message: str, thread_id: str):
        """Process chat message with RAG"""
        try:
            # Get relevant documents using vector search
            search_results = self.vector_store.search(user_message, top_k=3)
            
            # Get conversation history
            history = self.db.get_conversation_history(thread_id)
            
            # Build context with retrieved documents
            context_parts = []
            
            if search_results:
                context_parts.append("=== RELEVANT DOCUMENTS ===")
                for i, result in enumerate(search_results, 1):
                    doc_text = result['document']['text']
                    doc_snippet = doc_text[:500] + "..." if len(doc_text) > 500 else doc_text
                    context_parts.append(f"Document {i} ({result['document']['metadata'].get('file_name', 'Unknown')}):")
                    context_parts.append(doc_snippet)
                    context_parts.append("")
                
                context_parts.append("=== END DOCUMENTS ===")
                context_parts.append("Please answer based on the relevant documents above. If documents don't contain relevant info, provide a helpful general response.")
                context_parts.append("")
            
            # Prepare messages for LLM
            messages = []
            
            # Add recent history (last 6 messages)
            recent_history = history[-6:] if len(history) > 6 else history
            messages.extend(recent_history)
            
            # Add current message with context
            if context_parts:
                enhanced_content = "\n".join(context_parts) + f"\nUser Question: {user_message}"
                messages.append({'type': 'human', 'content': enhanced_content})
            else:
                messages.append({'type': 'human', 'content': user_message})
            
            # Get AI response
            ai_response = self.llm.generate_response(messages)
            
            # Save messages to database
            self.db.save_message(thread_id, "human", user_message)
            self.db.save_message(thread_id, "ai", ai_response)
            
            return {
                'success': True,
                'response': ai_response,
                'documents_used': len(search_results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'response': f"Error: {str(e)}",
                'documents_used': 0
            }

# Initialize components
@st.cache_resource
def get_components():
    """Initialize all components"""
    # Get API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("⚠️ Please set GEMINI_API_KEY in your environment or .env file")
        st.stop()
    
    # Initialize components
    doc_processor = DocumentProcessor()
    vector_store = SimpleVectorStore()
    db = SimpleDatabase()
    llm = SimpleLLM(api_key=gemini_api_key)
    rag_system = RAGSystem(llm, db, vector_store, doc_processor)
    
    return doc_processor, vector_store, db, llm, rag_system

def load_conversation_history(db):
    """Load conversation history"""
    history = db.get_conversation_history(st.session_state.current_thread_id)
    st.session_state.messages = []
    
    for msg in history:
        st.session_state.messages.append({
            'type': 'user' if msg['type'] == 'human' else 'assistant',
            'content': msg['content']
        })
    
    # Load documents
    docs = db.get_thread_documents(st.session_state.current_thread_id)
    st.session_state.documents = docs

def main():
    """Main application"""
    # Initialize session state FIRST
    init_session_state()
    
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG ChatBot Assistant</h1>
        <p>AI-powered assistant with document analysis capabilities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components after session state
    doc_processor, vector_store, db, llm, rag_system = get_components()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Thread Management
        st.subheader("💬 Conversations")
        
        if st.button("🆕 New Conversation"):
            st.session_state.current_thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.documents = []
            st.success("New conversation started!")
            st.rerun()
        
        # Load existing threads
        threads = db.get_all_threads()
        if threads:
            thread_options = [(f"Chat ({t['message_count']} msgs, {t['document_count']} docs)", t['thread_id']) for t in threads]
            
            selected = st.selectbox("Select Conversation:", options=thread_options, format_func=lambda x: x[0])
            
            if selected and selected[1] != st.session_state.current_thread_id:
                if st.button("📂 Load Conversation"):
                    st.session_state.current_thread_id = selected[1]
                    load_conversation_history(db)
                    st.success("Conversation loaded!")
                    st.rerun()
        
        st.divider()
        
        # Document Upload
        st.subheader("📚 Upload Documents")
        
        # Show supported formats
        supported_formats = list(doc_processor.supported_formats.keys())
        st.info(f"Supported: {', '.join(supported_formats)}")
        
        uploaded_file = st.file_uploader("Choose file", type=[ext[1:] for ext in supported_formats])
        
        if uploaded_file:
            if st.button("📤 Upload & Process"):
                with st.spinner("Processing document..."):
                    result = rag_system.upload_document(
                        uploaded_file.read(),
                        uploaded_file.name,
                        st.session_state.current_thread_id
                    )
                    
                    if result.get('success'):
                        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                        st.info(f"📄 {result.get('text_length', 0)} characters extracted")
                        st.session_state.documents = db.get_thread_documents(st.session_state.current_thread_id)
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        # Current documents
        st.subheader("📋 Documents in Chat")
        current_docs = db.get_thread_documents(st.session_state.current_thread_id)
        st.session_state.documents = current_docs
        
        if current_docs:
            for doc in current_docs:
                st.markdown(f"""
                <div class="document-card">
                    <strong>📄 {doc['filename']}</strong><br>
                    <small>{doc['file_type']} • {doc['text_length']} chars</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No documents uploaded")
        
        st.divider()
        
        # Search Documents
        st.subheader("🔍 Search Documents")
        search_query = st.text_input("Search in uploaded documents:")
        
        if search_query and st.button("Search"):
            results = vector_store.search(search_query, top_k=3)
            if results:
                for result in results:
                    st.markdown(f"""
                    **{result['document']['metadata'].get('file_name', 'Unknown')}** 
                    (Match: {result['similarity']:.2f})
                    
                    {result['document']['text'][:200]}...
                    """)
            else:
                st.info("No results found")
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stats-card"><h4>Session</h4><p>{st.session_state.current_thread_id[:8]}...</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stats-card"><h4>Messages</h4><p>{len(st.session_state.messages)}</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stats-card"><h4>Documents</h4><p>{len(st.session_state.documents)}</p></div>', unsafe_allow_html=True)
    
    # Chat messages
    for message in st.session_state.messages:
        if message['type'] == 'user':
            st.markdown(f'<div class="chat-message user-message"><strong>👤 You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>🤖 Assistant:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input("Your message:", placeholder="Ask me anything or ask about your documents...")
    with col2:
        send_button = st.button("Send 📤")
    
    # Handle send
    if (send_button or user_input) and user_input.strip():
        st.session_state.messages.append({'type': 'user', 'content': user_input})
        
        with st.spinner("🤖 Thinking..."):
            response = rag_system.chat(user_input, st.session_state.current_thread_id)
            
            if response.get('success'):
                st.session_state.messages.append({
                    'type': 'assistant',
                    'content': response['response']
                })
                
                if response.get('documents_used', 0) > 0:
                    st.info(f"📚 Referenced {response['documents_used']} document(s)")
            else:
                st.session_state.messages.append({
                    'type': 'assistant',
                    'content': f"❌ {response.get('response', 'Error occurred')}"
                })
        
        st.rerun()
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.success("Chat cleared!")
            st.rerun()
    
    with col2:
        if st.button("📥 Reload History"):
            load_conversation_history(db)
            st.success("History reloaded!")
            st.rerun()
    
    with col3:
        if st.button("🔄 Refresh"):
            st.session_state.documents = db.get_thread_documents(st.session_state.current_thread_id)
            st.success("Refreshed!")
            st.rerun()
    
    with col4:
        if st.button("❌ Delete Conversation"):
            if st.session_state.messages:  # Only show confirmation if there are messages
                if st.button("⚠️ Confirm Delete"):
                    db.delete_thread(st.session_state.current_thread_id)
                    st.session_state.current_thread_id = str(uuid.uuid4())
                    st.session_state.messages = []
                    st.session_state.documents = []
                    st.success("Conversation deleted!")
                    st.rerun()
            else:
                st.info("No conversation to delete")

if __name__ == "__main__":
    main()