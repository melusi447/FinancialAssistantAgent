"""
Core RAG Service
Robust document retrieval with semantic search using sentence transformers and FAISS
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import threading
import time

from config import config

logger = logging.getLogger(__name__)

class RAGService:
    """Robust RAG service with semantic search capabilities"""
    
    def __init__(self, docs_folder: Optional[str] = None, embedding_model: str = None):
        self.docs_folder = docs_folder or config.DOCS_FOLDER
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL
        self.model = None
        self._query_cache: dict = {}   # cache: query_text -> doc results
        self._cache_max = 50           # max cached queries
        self.index = None
        self.documents = []
        self.document_paths = []
        self.document_metadata = []
        self.is_initialized = False
        self._lock = threading.Lock()
        self._last_build = 0
        
        # Ensure docs folder exists
        os.makedirs(self.docs_folder, exist_ok=True)
        
        # Initialize in background
        self._initialize_async()
    
    def _initialize_async(self):
        """Initialize RAG service asynchronously"""
        def init_worker():
            try:
                self._load_embedding_model()
                self._load_documents()
                if self.documents:
                    self._build_faiss_index()
                else:
                    self._create_fallback_knowledge()
                    self._build_faiss_index()
                self.is_initialized = True
                logger.info("RAG service initialized successfully")
            except Exception as e:
                logger.error(f"RAG service initialization failed: {e}")
                # Create fallback anyway
                self._create_fallback_knowledge()
                self._build_faiss_index()
                self.is_initialized = True
        
        # Start initialization in background thread
        thread = threading.Thread(target=init_worker, daemon=True)
        thread.start()
    
    def _load_embedding_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded")
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise RuntimeError("sentence-transformers is required but not installed")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _load_documents(self):
        """Load documents from the docs folder"""
        try:
            if not os.path.exists(self.docs_folder):
                logger.warning(f"Docs folder does not exist: {self.docs_folder}")
                return
            
            supported_extensions = {'.txt', '.pdf', '.doc', '.docx', '.md'}
            loaded_files = []
            
            for file_path in Path(self.docs_folder).rglob('*'):
                if file_path.suffix.lower() in supported_extensions and file_path.is_file():
                    try:
                        content = self._read_document(str(file_path))
                        if content and content.strip():
                            chunks = self._chunk_document(content, file_path.name)
                            for i, chunk in enumerate(chunks):
                                self.documents.append(chunk)
                                self.document_paths.append(str(file_path))
                                self.document_metadata.append({
                                    'filename': file_path.name,
                                    'chunk_id': i,
                                    'total_chunks': len(chunks)
                                })
                            loaded_files.append(file_path.name)
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path.name}: {e}")
            
            if loaded_files:
                logger.info(f"Loaded {len(self.documents)} chunks from {len(loaded_files)} files")
            else:
                logger.warning("No documents found in docs folder")
                
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
    
    def _read_document(self, file_path: str) -> str:
        """Read document content based on file extension"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif file_ext == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                except ImportError:
                    logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
                    return f"[PDF file: {Path(file_path).name} - PyPDF2 not installed]"
            
            elif file_ext in ['.doc', '.docx']:
                try:
                    import docx
                    doc = docx.Document(file_path)
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    return text
                except ImportError:
                    logger.warning("python-docx not installed. Install with: pip install python-docx")
                    return f"[Word document: {Path(file_path).name} - python-docx not installed]"
            
            else:
                return f"[Unsupported file type: {file_ext}]"
                
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return f"[Error reading file: {str(e)}]"
    
    def _chunk_document(self, content: str, filename: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split large documents into overlapping chunks"""
        chunk_size = chunk_size or config.RAG_CHUNK_SIZE
        overlap = overlap or config.RAG_CHUNK_OVERLAP
        
        if len(content) <= chunk_size:
            return [f"[Source: {filename}]\n{content}"]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                search_start = max(start + chunk_size - 100, start)
                for i in range(end - 1, search_start, -1):
                    if content[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(f"[Source: {filename}]\n{chunk}")
            
            start = end - overlap
        
        return chunks
    
    def _create_fallback_knowledge(self):
        """Create fallback financial knowledge when no documents are available"""
        fallback_docs = [
            "Financial markets involve trading assets like stocks, bonds, and currencies. Market participants include individual investors, institutional investors, and market makers.",
            "Investing means allocating money with expectation of profit through stocks, bonds, mutual funds, or ETFs. Diversification reduces risk by spreading investments across different assets and sectors.",
            "Inflation is the rate at which prices for goods and services increase over time. Central banks use interest rates to control inflation and stimulate economic growth.",
            "Risk management is crucial in investing. Common strategies include diversification, asset allocation, and regular portfolio rebalancing based on investment goals and risk tolerance.",
            "Financial planning involves setting goals, creating budgets, managing debt, and building emergency funds. It's important to start early and maintain disciplined saving habits.",
            "Stock market analysis includes fundamental analysis (examining company financials) and technical analysis (studying price patterns and market trends).",
            "Bonds are debt securities that pay fixed interest over time. They are generally considered lower risk than stocks but offer lower potential returns.",
            "Mutual funds and ETFs pool money from multiple investors to buy diversified portfolios of stocks, bonds, or other assets. They provide instant diversification for individual investors."
        ]
        
        self.documents = fallback_docs
        self.document_paths = ["fallback_knowledge"] * len(fallback_docs)
        self.document_metadata = [{"filename": "fallback", "chunk_id": i, "total_chunks": len(fallback_docs)} 
                                 for i in range(len(fallback_docs))]
        logger.info("Created fallback financial knowledge base")
    
    def _build_faiss_index(self):
        """Build FAISS index from documents"""
        try:
            if not self.documents:
                logger.warning("No documents to build index from")
                return
            
            logger.info("Building FAISS index...")
            
            # Generate embeddings
            embeddings = self.model.encode(self.documents, convert_to_numpy=True, show_progress_bar=True)
            
            # Create FAISS index
            import faiss
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            self._last_build = time.time()
            logger.info(f"Built FAISS index with {len(self.documents)} documents, dimension {dimension}")
            
        except ImportError:
            logger.error("FAISS not installed")
            raise RuntimeError("faiss-cpu is required but not installed")
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            # Create a dummy index as fallback
            import faiss
            dummy_embeddings = np.random.random((len(self.documents), 384)).astype('float32')
            self.index = faiss.IndexFlatIP(384)
            faiss.normalize_L2(dummy_embeddings)
            self.index.add(dummy_embeddings)
    
    def retrieve_docs(self, query: str, k: int = None, min_score: float = None) -> List[str]:
        """Retrieve top-k most relevant documents with caching and timeout protection."""
        k = k or config.RAG_TOP_K
        min_score = min_score or config.RAG_MIN_SCORE

        if not self.is_initialized or not self.documents or self.index is None:
            logger.warning("RAG service not initialized or no documents available")
            return []

        # Return cached result if we've seen this query before
        cache_key = f"{query.strip().lower()[:100]}:{k}"
        if cache_key in self._query_cache:
            logger.debug(f"RAG cache hit for: '{query[:40]}'")
            return self._query_cache[cache_key]

        # Run the encode+search in a thread with a hard 8-second timeout
        # If the CPU is too slow, we return [] and the LLM answers without context
        result_holder = []
        error_holder = []

        def _do_retrieve():
            try:
                query_embedding = self.model.encode([query], convert_to_numpy=True)
                import faiss as _faiss
                _faiss.normalize_L2(query_embedding)
                scores, indices = self.index.search(query_embedding.astype("float32"), k)
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if score >= min_score and idx < len(self.documents):
                        doc_content = self.documents[idx]
                        if idx < len(self.document_metadata):
                            meta = self.document_metadata[idx]
                            doc_content = f"[Source: {meta['filename']}]\n{doc_content}"
                        results.append(doc_content)
                result_holder.extend(results)
            except Exception as e:
                error_holder.append(str(e))

        t = threading.Thread(target=_do_retrieve, daemon=True)
        t.start()
        t.join(timeout=8)   # hard 8-second cap

        if t.is_alive():
            logger.warning("RAG encode timed out after 8s — skipping context for this query")
            return []

        if error_holder:
            logger.error(f"RAG retrieval error: {error_holder[0]}")
            return []

        results = result_holder
        if not results:
            logger.info("No relevant documents found above threshold")
            return []

        logger.info(f"Retrieved {len(results)} doc(s) for: '{query[:50]}...'")

        # Cache result, evict oldest if over limit
        if len(self._query_cache) >= self._cache_max:
            oldest = next(iter(self._query_cache))
            del self._query_cache[oldest]
        self._query_cache[cache_key] = results
        return results
    
    def add_document(self, file_path: str) -> bool:
        """Add a new document to the search index"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            content = self._read_document(file_path)
            if not content or not content.strip():
                logger.warning(f"Empty or invalid content in: {file_path}")
                return False
            
            # Chunk the document
            chunks = self._chunk_document(content, Path(file_path).name)
            
            # Add chunks to documents
            for i, chunk in enumerate(chunks):
                self.documents.append(chunk)
                self.document_paths.append(file_path)
                self.document_metadata.append({
                    'filename': Path(file_path).name,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                })
            
            # Rebuild index
            self._build_faiss_index()
            
            logger.info(f"Added document to index: {Path(file_path).name} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return False
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about loaded documents"""
        return {
            "total_documents": len(self.documents),
            "document_names": list(set([meta['filename'] for meta in self.document_metadata])),
            "is_index_built": self.is_initialized and self.index is not None,
            "docs_folder": self.docs_folder,
            "embedding_model": self.embedding_model_name,
            "last_build": self._last_build
        }
    
    def is_ready(self) -> bool:
        """Check if RAG service is ready"""
        return self.is_initialized and self.index is not None

# Global RAG service instance
rag_service = RAGService()

# Convenience function for backward compatibility
def get_retrieval_service():
    return rag_service



