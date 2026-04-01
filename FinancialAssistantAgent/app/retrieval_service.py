"""
Modern RAG Service with Sentence Transformers and FAISS
Provides semantic document retrieval for financial knowledge base
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Modern RAG service using sentence transformers and FAISS for semantic search
    """
    
    def __init__(self, docs_folder: Optional[str] = None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.docs_folder = docs_folder or config.DOCS_FOLDER
        self.embedding_model_name = embedding_model
        self.model = None
        self.index = None
        self.documents = []
        self.document_paths = []
        self.document_metadata = []
        self.is_initialized = False
        
        logger.info(f"📁 Initializing RetrievalService with docs folder: {self.docs_folder}")
        self._initialize()
    
    def _initialize(self):
        """Initialize the retrieval service"""
        try:
            # Initialize embedding model
            logger.info(f"🔄 Loading embedding model: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name)
            
            # Load documents and build index
            self._load_documents()
            if self.documents:
                self._build_faiss_index()
                self.is_initialized = True
                logger.info(f"✅ RetrievalService initialized with {len(self.documents)} documents")
            else:
                logger.warning("⚠️ No documents loaded, using fallback knowledge")
                self._create_fallback_knowledge()
                self._build_faiss_index()
                self.is_initialized = True
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize RetrievalService: {e}")
            self._create_fallback_knowledge()
            self._build_faiss_index()
            self.is_initialized = True
    
    def _load_documents(self):
        """Load documents from the docs folder"""
        try:
            if not os.path.exists(self.docs_folder):
                logger.warning(f"📁 Docs folder does not exist: {self.docs_folder}")
                os.makedirs(self.docs_folder, exist_ok=True)
                return
            
            supported_extensions = {'.txt', '.pdf', '.doc', '.docx', '.md'}
            loaded_files = []
            
            for file_path in Path(self.docs_folder).rglob('*'):
                if file_path.suffix.lower() in supported_extensions and file_path.is_file():
                    try:
                        content = self._read_document(str(file_path))
                        if content and content.strip():
                            # Chunk large documents
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
                            logger.debug(f"✅ Loaded document: {file_path.name} ({len(chunks)} chunks)")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load {file_path.name}: {e}")
            
            if loaded_files:
                logger.info(f"📚 Loaded {len(self.documents)} document chunks from {len(loaded_files)} files")
            else:
                logger.warning("📚 No documents found in docs folder")
                
        except Exception as e:
            logger.error(f"❌ Error loading documents: {e}")
    
    def _read_document(self, file_path: str) -> str:
        """Read document content based on file extension"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.txt' or file_ext == '.md':
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
                    logger.warning("📄 PyPDF2 not installed. Install with: pip install PyPDF2")
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
                    logger.warning("📄 python-docx not installed. Install with: pip install python-docx")
                    return f"[Word document: {Path(file_path).name} - python-docx not installed]"
            
            else:
                return f"[Unsupported file type: {file_ext}]"
                
        except Exception as e:
            logger.error(f"❌ Error reading {file_path}: {e}")
            return f"[Error reading file: {str(e)}]"
    
    def _chunk_document(self, content: str, filename: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split large documents into overlapping chunks"""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings within the last 100 characters
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
        logger.info("📚 Created fallback financial knowledge base")
    
    def _build_faiss_index(self):
        """Build FAISS index from documents"""
        try:
            if not self.documents:
                logger.warning("No documents to build index from")
                return
            
            logger.info("🔄 Building FAISS index...")
            
            # Generate embeddings
            embeddings = self.model.encode(self.documents, convert_to_numpy=True, show_progress_bar=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            logger.info(f"✅ Built FAISS index with {len(self.documents)} documents, dimension {dimension}")
            
        except Exception as e:
            logger.error(f"❌ Error building FAISS index: {e}")
            # Create a dummy index as fallback
            dummy_embeddings = np.random.random((len(self.documents), 384)).astype('float32')
            self.index = faiss.IndexFlatIP(384)
            faiss.normalize_L2(dummy_embeddings)
            self.index.add(dummy_embeddings)
    
    def retrieve_docs(self, query: str, k: int = 3, min_score: float = 0.1) -> List[str]:
        """
        Retrieve top-k most relevant documents for the query
        
        Args:
            query: User query string
            k: Number of top documents to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of relevant document contents
        """
        if not self.is_initialized or not self.documents or self.index is None:
            logger.warning("Retrieval service not initialized or no documents available")
            return ["No documents available for retrieval."]
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Filter by minimum score and return results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= min_score and idx < len(self.documents):
                    doc_content = self.documents[idx]
                    # Add metadata if available
                    if idx < len(self.document_metadata):
                        metadata = self.document_metadata[idx]
                        doc_content = f"[Source: {metadata['filename']}]\n{doc_content}"
                    results.append(doc_content)
            
            if not results:
                logger.info("No sufficiently relevant documents found")
                return ["No highly relevant documents found. Using general knowledge."]
            
            logger.info(f"🔍 Retrieved {len(results)} relevant documents for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during document retrieval: {e}")
            return ["Error retrieving documents. Using general knowledge."]
    
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
            
            logger.info(f"✅ Added document to index: {Path(file_path).name} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding document {file_path}: {e}")
            return False
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about loaded documents"""
        return {
            "total_documents": len(self.documents),
            "document_names": list(set([meta['filename'] for meta in self.document_metadata])),
            "is_index_built": self.is_initialized and self.index is not None,
            "docs_folder": self.docs_folder,
            "embedding_model": self.embedding_model_name
        }
    
    def save_index(self, file_path: str) -> bool:
        """Save the search index to disk"""
        try:
            if not self.is_initialized or self.index is None:
                logger.warning("No index to save")
                return False
            
            index_data = {
                'documents': self.documents,
                'document_paths': self.document_paths,
                'document_metadata': self.document_metadata,
                'embedding_model': self.embedding_model_name
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(index_data, f)
            
            # Save FAISS index separately
            faiss.write_index(self.index, file_path + '.faiss')
            
            logger.info(f"💾 Saved search index to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving index: {e}")
            return False
    
    def load_index(self, file_path: str) -> bool:
        """Load the search index from disk"""
        try:
            with open(file_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data['documents']
            self.document_paths = index_data['document_paths']
            self.document_metadata = index_data['document_metadata']
            self.embedding_model_name = index_data.get('embedding_model', 'all-MiniLM-L6-v2')
            
            # Reinitialize model and load FAISS index
            self.model = SentenceTransformer(self.embedding_model_name)
            self.index = faiss.read_index(file_path + '.faiss')
            self.is_initialized = True
            
            logger.info(f"📂 Loaded search index from: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading index: {e}")
            return False


# Create global instance
retrieval_service = RetrievalService()

# Helper function for backward compatibility
def get_retrieval_service():
    return retrieval_service