# chatbot_local_mistral.py
import streamlit as st
import numpy as np
from chromadb import PersistentClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
import ollama
import logging
import re
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Memory components from LlamaIndex
from llama_index.core.memory import VectorMemory, ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings

# ======================
# CONFIGURATION SECTION
# ======================
DB_PATH = "db_chroma"
OLLAMA_MODEL = "mistral:latest"
MAX_CONTEXT_LENGTH = 4096
MEMORY_TOKEN_LIMIT = 3000
DEBUG_MODE = True

MOCK_DATA = {
    "Login Page URL": "https://portal.company.internal",
    "Support Portal": "https://support.company.internal",
    "Support Email": "help@company.internal",
    "_DEFAULT": "our IT portal"
}

# ======================
# CORE SYSTEM CLASSES
# ======================
class ChatSystem:
    def __init__(self):
        self._init_embeddings()
        self._init_chroma()
        self._init_memory()
        self._init_ollama()
        
    def _init_embeddings(self):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                device=self.device
            )
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise

    def _init_chroma(self):
        try:
            self.client = PersistentClient(path=DB_PATH)
            self.collection = self.client.get_collection("rag_documents")
        except Exception as e:
            logger.error(f"ChromaDB error: {str(e)}")
            raise

    def _init_memory(self):
        try:
            self.vector_memory = VectorMemory.from_defaults(
                retriever_kwargs={"similarity_top_k": 2},
                embed_model=Settings.embed_model
            )
            self.chat_memory = ChatMemoryBuffer.from_defaults(
                token_limit=MEMORY_TOKEN_LIMIT
            )
            self.vector_memory.set([
                ChatMessage(role="system", content="IT Support Agent"),
                ChatMessage(role="system", content=f"Login URL: {MOCK_DATA['Login Page URL']}"),
                ChatMessage(role="system", content=f"Support Portal: {MOCK_DATA['Support Portal']}")
            ])
        except Exception as e:
            logger.error(f"Memory error: {str(e)}")
            raise

    def _init_ollama(self):
        try:
            models = ollama.list()
            if not any(m['model'] == OLLAMA_MODEL for m in models.get('models', [])):
                raise ValueError(f"Model {OLLAMA_MODEL} not found")
            ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "test"}])
        except Exception as e:
            logger.error(f"Ollama error: {str(e)}")
            st.error("Ollama service unavailable")
            st.stop()

class ChatProcessor(ChatSystem):
    def retrieve_documents(self, query: str, top_k: int = 3) -> Tuple[list, dict]:
        try:
            embedding = Settings.embed_model.get_text_embedding(query)
            results = self.collection.query(
                query_embeddings=[np.array(embedding, dtype=np.float32).tolist()],
                n_results=top_k
            )
            return [doc for doc_list in results['documents'] for doc in doc_list if doc], {}
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return [], {}

    def _build_context(self, query: str) -> Tuple[str, dict]:
        try:
            docs, _ = self.retrieve_documents(query)
            history = self.chat_memory.get()
            
            # Pre-formatted components
            docs_str = '\n'.join(docs)
            history_entries = [f"{msg.role}: {msg.content}" for msg in history]
            history_str = '\n'.join(history_entries)
            
            context = f"""IT Knowledge Base:
{docs_str}

Conversation History:
{history_str}

Current Issue: {query}"""
            
            return context, {}
        except Exception as e:
            logger.error(f"Context error: {str(e)}")
            raise

    def _replace_templates(self, text: str) -> str:
        return re.sub(
            r"\{\{(.+?)\}\}",
            lambda m: MOCK_DATA.get(m.group(1).strip(), MOCK_DATA["_DEFAULT"]),
            text
        )

    def generate_response(self, query: str) -> Tuple[str, dict]:
        try:
            query_lower = query.lower().strip()
            
            # Handle greetings
            if any(greet in query_lower for greet in ["hi", "hello", "hey"]):
                return "Hello!. How can I help?", {}
            
            # Handle gratitude
            if any(kw in query_lower for kw in ["thank", "thanks"]):
                return "You're welcome!" + \
                       MOCK_DATA["Support Portal"], {}
            
            # Handle non-IT queries
            support_keywords = ["login", "password", "error", "system", "network"]
            if not any(kw in query_lower for kw in support_keywords):
                return f"Contact general support: {MOCK_DATA['Support Email']}", {}
            
            # Process IT request
            self.chat_memory.put(ChatMessage(role="user", content=query))
            context, _ = self._build_context(query)
            
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": context}],
                options={"temperature": 0.5}
            )
            
            processed_response = self._replace_templates(response['message']['content'])
            self.chat_memory.put(ChatMessage(role="assistant", content=processed_response))
            
            return processed_response, {}
            
        except Exception as e:
            logger.error(f"Response error: {str(e)}")
            return f"IT Support Error: {str(e)}", {}

# ======================
# STREAMLIT UI
# ======================
def main():
    st.title("IT Support Chatbot")
    
    if 'processor' not in st.session_state:
        st.session_state.processor = ChatProcessor()
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Describe your IT issue:"):
        st.session_state.history.append({"role": "user", "content": prompt})
        
        with st.spinner("Analyzing..."):
            response, _ = st.session_state.processor.generate_response(prompt)
            st.session_state.history.append({"role": "assistant", "content": response})
        
        st.rerun()

if __name__ == "__main__":
    main()