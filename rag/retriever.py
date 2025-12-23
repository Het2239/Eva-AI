#!/usr/bin/env python3
"""
EVA RAG - Retriever
===================
Full retrieval pipeline with hybrid search, math-aware query rewriting, and reranking.
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    documents: List[Document]
    query_original: str
    query_rewritten: str
    num_candidates: int
    num_reranked: int


class MathQueryRewriter:
    """
    Math-aware query rewriting using SymPy for normalization.
    
    Features:
        - Detect mathematical expressions
        - Normalize LaTeX notation
        - Expand mathematical terms
        - Add equivalent representations
    """
    
    def __init__(self, use_sympy: bool = True):
        """
        Initialize the math query rewriter.
        
        Args:
            use_sympy: Whether to use SymPy for expression parsing
        """
        self.use_sympy = use_sympy
        self._sympy_available = None
    
    def _check_sympy(self) -> bool:
        """Check if SymPy is available."""
        if self._sympy_available is None:
            try:
                import sympy
                self._sympy_available = True
            except ImportError:
                self._sympy_available = False
        return self._sympy_available
    
    def _normalize_latex(self, text: str) -> str:
        """Normalize LaTeX expressions to plain text."""
        # Common LaTeX to text conversions
        replacements = [
            (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)'),
            (r'\\sqrt\{([^}]+)\}', r'sqrt(\1)'),
            (r'\\int', 'integral'),
            (r'\\sum', 'sum'),
            (r'\\prod', 'product'),
            (r'\\partial', 'partial'),
            (r'\\nabla', 'gradient'),
            (r'\\infty', 'infinity'),
            (r'\^2', ' squared'),
            (r'\^3', ' cubed'),
            (r'\^n', ' to the n'),
            (r'\\alpha', 'alpha'),
            (r'\\beta', 'beta'),
            (r'\\gamma', 'gamma'),
            (r'\\theta', 'theta'),
            (r'\\pi', 'pi'),
            (r'\\sigma', 'sigma'),
            (r'\\lambda', 'lambda'),
            (r'\$', ''),
        ]
        
        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common math abbreviations."""
        expansions = {
            r'\bderiv\b': 'derivative',
            r'\bdiff\b': 'differential',
            r'\beqn\b': 'equation',
            r'\bfn\b': 'function',
            r'\bvar\b': 'variable',
            r'\bconst\b': 'constant',
            r'\bcoeff\b': 'coefficient',
            r'\bpoly\b': 'polynomial',
            r'\bexp\b': 'exponential',
            r'\blog\b': 'logarithm',
            r'\bln\b': 'natural logarithm',
            r'\btrig\b': 'trigonometric',
        }
        
        result = text
        for pattern, replacement in expansions.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def _sympy_expand(self, text: str) -> str:
        """Use SymPy to parse and expand expressions."""
        if not self._check_sympy():
            return text
        
        import sympy
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
        
        # Try to find and expand mathematical expressions
        math_pattern = r'([a-z]\s*\^\s*\d+|[a-z]\s*\*\*\s*\d+|\d+\s*\*\s*[a-z])'
        
        def expand_match(match):
            expr_str = match.group(0)
            try:
                # Parse expression
                transformations = standard_transformations + (implicit_multiplication,)
                expr = parse_expr(expr_str.replace('^', '**'), transformations=transformations)
                
                # Get derivative form for common patterns
                if '**' in expr_str or '^' in expr_str:
                    # This is a power expression
                    return f"{expr_str} (power function)"
                
                return expr_str
            except:
                return expr_str
        
        return re.sub(math_pattern, expand_match, text, flags=re.IGNORECASE)
    
    def _detect_math_query(self, query: str) -> bool:
        """Detect if query contains math-related content."""
        math_indicators = [
            r'[∑∏∫∂∇√∞≈≠≤≥±×÷]',
            r'\$.*?\$',
            r'\\[a-z]+',
            r'\b(derivative|integral|equation|function|variable|theorem)\b',
            r'\b(sin|cos|tan|log|ln|exp|sqrt)\b',
            r'[a-z]\s*[\^=]\s*\d',
            r'\d\s*[+\-*/]\s*\d',
        ]
        
        for pattern in math_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def rewrite(self, query: str) -> str:
        """
        Rewrite query for better math retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Rewritten query
        """
        if not self._detect_math_query(query):
            return query
        
        # Apply transformations
        result = self._normalize_latex(query)
        result = self._expand_abbreviations(result)
        
        if self.use_sympy:
            result = self._sympy_expand(result)
        
        return result


class Reranker:
    """
    Cross-encoder reranker for precision.
    
    Uses a cross-encoder model to rerank retrieved documents
    based on query-document relevance.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: Cross-encoder model from sentence-transformers
        """
        self.model_name = model_name
        self._model = None
    
    def _get_model(self):
        """Get or create the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents
        """
        if not documents:
            return []
        
        if len(documents) <= top_k:
            # No need to rerank if we have fewer docs than requested
            return documents
        
        model = self._get_model()
        
        # Create query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get scores
        scores = model.predict(pairs)
        
        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: -x[1])
        
        # Add score to metadata and return top_k
        results = []
        for doc, score in scored_docs[:top_k]:
            doc.metadata['rerank_score'] = round(float(score), 4)
            results.append(doc)
        
        return results


class RAGRetriever:
    """
    Full RAG retrieval pipeline.
    
    Combines:
        - Math-aware query rewriting
        - Hybrid retrieval (dense + sparse)
        - Cross-encoder reranking
    """
    
    def __init__(
        self,
        vector_store,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        dense_weight: float = 0.5,
        use_reranker: bool = True,
        use_math_rewriter: bool = True,
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance
            rerank_model: Cross-encoder model for reranking
            dense_weight: Weight for dense vs sparse (0-1)
            use_reranker: Whether to use reranking
            use_math_rewriter: Whether to use math query rewriting
        """
        self.vector_store = vector_store
        self.dense_weight = dense_weight
        self.use_reranker = use_reranker
        self.use_math_rewriter = use_math_rewriter
        
        # Initialize components
        self.math_rewriter = MathQueryRewriter() if use_math_rewriter else None
        self.reranker = Reranker(rerank_model) if use_reranker else None
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank_candidates: int = 20,
    ) -> RetrievalResult:
        """
        Full retrieval with rewriting, hybrid search, and reranking.
        
        Args:
            query: Search query
            top_k: Number of final results
            rerank_candidates: Number of candidates for reranking
            
        Returns:
            RetrievalResult with documents and metadata
        """
        original_query = query
        
        # Math-aware query rewriting
        if self.math_rewriter:
            query = self.math_rewriter.rewrite(query)
        
        # Hybrid search
        candidates = self.vector_store.hybrid_search(
            query,
            k=rerank_candidates if self.use_reranker else top_k,
            dense_weight=self.dense_weight,
        )
        
        num_candidates = len(candidates)
        
        # Reranking
        if self.use_reranker and self.reranker and len(candidates) > top_k:
            documents = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            documents = candidates[:top_k]
        
        return RetrievalResult(
            documents=documents,
            query_original=original_query,
            query_rewritten=query,
            num_candidates=num_candidates,
            num_reranked=len(documents),
        )
    
    def retrieve_simple(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Document]:
        """
        Simple retrieval returning just documents.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of Documents
        """
        result = self.retrieve(query, top_k=top_k)
        return result.documents


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_retriever(
    vector_store,
    use_reranker: bool = True,
    use_math_rewriter: bool = True,
) -> RAGRetriever:
    """
    Create a RAG retriever.
    
    Args:
        vector_store: VectorStore instance
        use_reranker: Whether to use reranking
        use_math_rewriter: Whether to use math query rewriting
        
    Returns:
        RAGRetriever instance
    """
    return RAGRetriever(
        vector_store=vector_store,
        use_reranker=use_reranker,
        use_math_rewriter=use_math_rewriter,
    )


# ============================================================
# CLI INTERFACE
# ============================================================

if __name__ == "__main__":
    # Test math query rewriter
    print("MathQueryRewriter Test")
    print("=" * 50)
    
    rewriter = MathQueryRewriter()
    
    test_queries = [
        "What is the derivative of x^2?",
        "Calculate \\frac{1}{2} + \\frac{1}{3}",
        "Explain the \\int of sin(x)",
        "What is machine learning?",  # Non-math query
    ]
    
    for query in test_queries:
        rewritten = rewriter.rewrite(query)
        if query != rewritten:
            print(f"'{query}' → '{rewritten}'")
        else:
            print(f"'{query}' (unchanged)")
    
    print("\n" + "=" * 50)
    print("Reranker Test")
    print("=" * 50)
    
    # Test reranker with sample docs
    reranker = Reranker()
    test_docs = [
        Document(page_content="The derivative of x squared is 2x.", metadata={}),
        Document(page_content="Machine learning is a type of AI.", metadata={}),
        Document(page_content="Calculus deals with derivatives and integrals.", metadata={}),
    ]
    
    print("Reranking for query: 'derivative of x^2'")
    reranked = reranker.rerank("derivative of x^2", test_docs, top_k=2)
    
    for doc in reranked:
        print(f"  [{doc.metadata.get('rerank_score', 0):.3f}] {doc.page_content[:50]}...")
