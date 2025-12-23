#!/usr/bin/env python3
"""
EVA RAG - Query Pipeline
========================
End-to-end RAG: ingest documents → query → get answers with sources.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

# Add parent directory to path for direct script execution
_this_dir = Path(__file__).parent
if str(_this_dir.parent) not in sys.path:
    sys.path.insert(0, str(_this_dir.parent))
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from langchain_core.documents import Document


@dataclass
class RAGResponse:
    """Container for RAG query response."""
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    context_chunks: List[str] = field(default_factory=list)
    num_chunks_retrieved: int = 0


class RAGPipeline:
    """
    End-to-end RAG pipeline: ingest → query → answer.
    
    Features:
        - Document ingestion with processing
        - Hybrid retrieval (dense + sparse)
        - LLM-powered answer generation
        - Source citations
        - Persistent storage
    """
    
    def __init__(
        self,
        persist_dir: str = "./rag_data",
        collection_name: str = "eva_rag",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        use_reranker: bool = True,
        chunk_size: int = 500,
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name for the vector collection
            embedding_model: Embedding model name
            use_reranker: Whether to use cross-encoder reranking
            chunk_size: Target chunk size for splitting
        """
        self.persist_dir = os.path.expanduser(persist_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.use_reranker = use_reranker
        self.chunk_size = chunk_size
        
        # Lazy-load components
        self._store = None
        self._retriever = None
        self._processor = None
    
    def _get_store(self):
        """Get or create vector store."""
        if self._store is None:
            try:
                from rag.vector_store import VectorStore
            except ImportError:
                from vector_store import VectorStore
            
            self._store = VectorStore(
                collection_name=self.collection_name,
                embedding_model=self.embedding_model,
                persist_directory=self.persist_dir,
            )
        return self._store
    
    def _get_retriever(self):
        """Get or create retriever."""
        if self._retriever is None:
            try:
                from rag.retriever import RAGRetriever
            except ImportError:
                from retriever import RAGRetriever
            
            self._retriever = RAGRetriever(
                vector_store=self._get_store(),
                use_reranker=self.use_reranker,
                use_math_rewriter=True,
            )
        return self._retriever
    
    def _get_processor(self):
        """Get or create chunk processor."""
        if self._processor is None:
            try:
                from rag.chunk_processor import ChunkProcessor
            except ImportError:
                from chunk_processor import ChunkProcessor
            self._processor = ChunkProcessor()
        return self._processor
    
    def ingest(
        self,
        path: str,
        verbose: bool = True,
    ) -> int:
        """
        Ingest a document or directory into the RAG system.
        
        Args:
            path: Path to document or directory
            verbose: Whether to print progress
            
        Returns:
            Number of chunks indexed
        """
        try:
            from rag.document_loader import load_and_split
        except ImportError:
            from document_loader import load_and_split
        
        path = os.path.expanduser(path)
        
        if verbose:
            print(f"📄 Loading: {path}")
        
        # Load and split
        chunks = load_and_split(
            path,
            chunk_size=self.chunk_size,
            semantic=False,  # Use character splitting for speed
            verbose=verbose,
        )
        
        if not chunks:
            if verbose:
                print("⚠ No content extracted")
            return 0
        
        if verbose:
            print(f"   Loaded {len(chunks)} chunks")
        
        # Process chunks
        processor = self._get_processor()
        processed = processor.process(chunks)
        
        if verbose:
            print(f"   Processed: {len(processed)} chunks (after filtering)")
        
        # Index in store
        store = self._get_store()
        store.add_documents(processed)
        
        if verbose:
            print(f"✓ Indexed {len(processed)} chunks")
        
        return len(processed)
    
    def ingest_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
        verbose: bool = True,
    ) -> int:
        """
        Ingest all documents from a directory.
        
        Args:
            directory: Path to directory
            extensions: File extensions to include
            recursive: Whether to search subdirectories
            verbose: Whether to print progress
            
        Returns:
            Total number of chunks indexed
        """
        try:
            from rag.document_loader import load_documents_smart
        except ImportError:
            from document_loader import load_documents_smart
        
        directory = os.path.expanduser(directory)
        
        if verbose:
            print(f"📁 Ingesting directory: {directory}")
        
        # Load all documents
        docs = load_documents_smart(
            directory,
            recursive=recursive,
            extensions=extensions,
        )
        
        if not docs:
            if verbose:
                print("⚠ No documents found")
            return 0
        
        if verbose:
            print(f"   Found {len(docs)} documents")
        
        # Split documents
        try:
            from rag.semantic_splitter import SemanticSplitter
        except ImportError:
            from semantic_splitter import SemanticSplitter
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size // 10,
        )
        chunks = splitter.split_documents(docs)
        
        if verbose:
            print(f"   Split into {len(chunks)} chunks")
        
        # Process and index
        processor = self._get_processor()
        processed = processor.process(chunks)
        
        store = self._get_store()
        store.add_documents(processed)
        
        if verbose:
            print(f"✓ Indexed {len(processed)} chunks")
        
        return len(processed)
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        include_context: bool = False,
    ) -> RAGResponse:
        """
        Query the RAG system and get an answer.
        
        Args:
            question: The question to ask
            top_k: Number of context chunks to retrieve
            include_context: Whether to include context chunks in response
            
        Returns:
            RAGResponse with answer and sources
        """
        # Retrieve relevant context
        retriever = self._get_retriever()
        result = retriever.retrieve(question, top_k=top_k)
        
        if not result.documents:
            return RAGResponse(
                question=question,
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                num_chunks_retrieved=0,
            )
        
        # Build context from retrieved documents
        context_parts = []
        sources = set()
        context_chunks = []
        
        for doc in result.documents:
            context_parts.append(doc.page_content)
            context_chunks.append(doc.page_content)
            
            # Extract source info
            source = doc.metadata.get('source_file') or doc.metadata.get('source', 'Unknown')
            sources.add(source)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build prompt
        prompt = self._build_prompt(question, context)
        
        # Get LLM response
        answer = self._get_llm_response(prompt)
        
        return RAGResponse(
            question=question,
            answer=answer,
            sources=list(sources),
            context_chunks=context_chunks if include_context else [],
            num_chunks_retrieved=len(result.documents),
        )
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for the LLM."""
        return f"""You are a helpful AI assistant. Answer the question based on the provided context.
If the context doesn't contain enough information to answer the question fully, say so.
Be concise but complete.

Context:
{context}

Question: {question}

Answer:"""
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from the LLM."""
        try:
            # Import from models.py
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from models import get_chat_response
            return get_chat_response(prompt)
        except ImportError:
            # Fallback if models.py not available
            return "[LLM not configured - install and configure models.py]"
        except Exception as e:
            return f"[Error getting LLM response: {e}]"
    
    def save(self) -> None:
        """Save the pipeline state to disk."""
        if self._store:
            self._store.save(self.persist_dir)
            print(f"✓ Saved to {self.persist_dir}")
    
    def load(self) -> bool:
        """
        Load the pipeline state from disk.
        
        Returns:
            True if loaded successfully
        """
        store = self._get_store()
        docs_path = Path(self.persist_dir) / "documents.pkl"
        
        if docs_path.exists():
            store.load(self.persist_dir)
            return True
        return False
    
    @property
    def document_count(self) -> int:
        """Return number of indexed documents."""
        return self._get_store().document_count


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_rag_pipeline(
    persist_dir: str = "./rag_data",
    use_reranker: bool = True,
) -> RAGPipeline:
    """Create a RAG pipeline with default settings."""
    return RAGPipeline(
        persist_dir=persist_dir,
        use_reranker=use_reranker,
    )


def ask(
    question: str,
    persist_dir: str = "./rag_data",
) -> str:
    """
    Quick function to ask a question.
    
    Args:
        question: The question to ask
        persist_dir: Path to the RAG data directory
        
    Returns:
        The answer string
    """
    pipeline = RAGPipeline(persist_dir=persist_dir, use_reranker=False)
    pipeline.load()
    response = pipeline.query(question)
    return response.answer


# ============================================================
# CLI INTERFACE
# ============================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="EVA RAG Pipeline - Ingest documents and ask questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a document
  python3 rag/rag_pipeline.py ingest document.pdf
  
  # Ingest a directory
  python3 rag/rag_pipeline.py ingest ./documents/
  
  # Ask a question
  python3 rag/rag_pipeline.py ask "What is machine learning?"
  
  # Interactive chat mode
  python3 rag/rag_pipeline.py chat
        """
    )
    
    parser.add_argument(
        "--data-dir",
        default="./rag_data",
        help="Directory for RAG data (default: ./rag_data)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document or directory")
    ingest_parser.add_argument("path", help="Path to document or directory")
    ingest_parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="The question to ask")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of context chunks")
    ask_parser.add_argument("--show-sources", action="store_true", help="Show source files")
    ask_parser.add_argument("--show-context", action="store_true", help="Show retrieved context")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    chat_parser.add_argument("--top-k", type=int, default=5, help="Number of context chunks")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show pipeline status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create pipeline
    pipeline = RAGPipeline(persist_dir=args.data_dir)
    
    if args.command == "ingest":
        path = os.path.expanduser(args.path)
        
        if os.path.isdir(path):
            count = pipeline.ingest_directory(path)
        else:
            count = pipeline.ingest(path)
        
        pipeline.save()
        print(f"\n✓ Total: {count} chunks indexed")
    
    elif args.command == "ask":
        # Load existing data
        if not pipeline.load():
            print("⚠ No data found. Run 'ingest' first.")
            return
        
        response = pipeline.query(
            args.question,
            top_k=args.top_k,
            include_context=args.show_context,
        )
        
        print(f"\n📝 Answer:\n{response.answer}")
        
        if args.show_sources and response.sources:
            print(f"\n📚 Sources: {', '.join(response.sources)}")
        
        if args.show_context and response.context_chunks:
            print("\n📄 Context chunks:")
            for i, chunk in enumerate(response.context_chunks):
                print(f"\n--- Chunk {i+1} ---")
                print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    
    elif args.command == "chat":
        # Load existing data
        if not pipeline.load():
            print("⚠ No data found. Run 'ingest' first.")
            return
        
        print("🤖 EVA RAG Chat (type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            try:
                question = input("\n❓ You: ").strip()
                
                if not question:
                    continue
                if question.lower() in ('quit', 'exit', 'q'):
                    print("👋 Goodbye!")
                    break
                
                response = pipeline.query(question, top_k=args.top_k)
                print(f"\n🤖 EVA: {response.answer}")
                
                if response.sources:
                    print(f"\n   📚 Sources: {', '.join(response.sources)}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    
    elif args.command == "status":
        if pipeline.load():
            print(f"📊 RAG Pipeline Status")
            print(f"   Data directory: {pipeline.persist_dir}")
            print(f"   Documents indexed: {pipeline.document_count}")
        else:
            print("⚠ No data found. Run 'ingest' first.")


if __name__ == "__main__":
    main()
