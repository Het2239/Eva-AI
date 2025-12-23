#!/usr/bin/env python3
"""
EVA RAG - Document Loader
=========================
Document parsing using Docling (IBM Research) for high-quality extraction.
Supports PDFs, DOCX, PPTX, XLSX, HTML, images, and more.
"""

import os
from pathlib import Path
from typing import List, Optional, Literal, Union

from langchain_core.documents import Document


# Export format types
ExportFormat = Literal["markdown", "text", "json"]

# Supported file extensions and their categories
DOCLING_SUPPORTED = {
    # Documents
    ".pdf": "PDF Document",
    ".docx": "Word Document",
    ".doc": "Word Document (Legacy)",
    ".pptx": "PowerPoint Presentation",
    ".ppt": "PowerPoint (Legacy)",
    ".xlsx": "Excel Spreadsheet",
    ".xls": "Excel (Legacy)",
    ".html": "HTML Document",
    ".htm": "HTML Document",
    ".md": "Markdown Document",
    ".asciidoc": "AsciiDoc Document",
    ".adoc": "AsciiDoc Document",
    # Open Document Formats
    ".odt": "OpenDocument Text",
    ".odp": "OpenDocument Presentation",
    ".ods": "OpenDocument Spreadsheet",
    # Images (OCR)
    ".png": "PNG Image",
    ".jpg": "JPEG Image",
    ".jpeg": "JPEG Image",
    ".tiff": "TIFF Image",
    ".tif": "TIFF Image",
    ".bmp": "BMP Image",
    ".webp": "WebP Image",
}

# Plain text files (handled separately)
TEXT_EXTENSIONS = {
    ".txt": "Plain Text",
    ".csv": "CSV File",
    ".tsv": "TSV File",
    ".log": "Log File",
    ".json": "JSON File",
    ".xml": "XML File",
    ".yaml": "YAML File",
    ".yml": "YAML File",
    ".ini": "INI Config",
    ".cfg": "Config File",
    ".conf": "Config File",
    ".rst": "reStructuredText",
    ".tex": "LaTeX Document",
    ".rtf": "Rich Text Format",
}

# All supported extensions
ALL_SUPPORTED = {**DOCLING_SUPPORTED, **TEXT_EXTENSIONS}


class DoclingLoader:
    """
    Document Loader using IBM's Docling library.
    
    Features:
        - Advanced PDF understanding (layout, tables, formulas, images)
        - Multiple format support (PDF, DOCX, PPTX, XLSX, HTML, images, ODT, ODP)
        - High-quality OCR for scanned documents
        - Exports to Markdown, text, or structured JSON
    
    Supported Formats:
        - Documents: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, HTML, MD
        - Open Document: ODT, ODP, ODS
        - Images: PNG, JPG, JPEG, TIFF, BMP, WebP
        - Text: TXT, CSV, JSON, XML, YAML, LOG, RST, TEX
    """
    
    def __init__(
        self,
        export_format: ExportFormat = "markdown",
        ocr_enabled: bool = True,
        extract_tables: bool = True,
        extract_images: bool = False,
    ):
        """
        Initialize the Docling loader.
        
        Args:
            export_format: Output format ('markdown', 'text', 'json')
            ocr_enabled: Whether to enable OCR for scanned content
            extract_tables: Whether to extract and format tables
            extract_images: Whether to extract images (stores paths in metadata)
        """
        self.export_format = export_format
        self.ocr_enabled = ocr_enabled
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        
        # Lazy-load the converter
        self._converter = None
    
    @staticmethod
    def supported_formats() -> dict:
        """Return dictionary of all supported file formats."""
        return ALL_SUPPORTED
    
    @staticmethod
    def is_supported(file_path: str) -> bool:
        """Check if a file format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in ALL_SUPPORTED
    
    def _get_converter(self):
        """Get or create the DocumentConverter with configured options."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TableFormerMode,
            )
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import PdfFormatOption
            
            # Configure PDF pipeline options
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.ocr_enabled
            pipeline_options.do_table_structure = self.extract_tables
            
            if self.extract_tables:
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            
            # Create converter with options
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        
        return self._converter
    
    def _load_text_file(self, file_path: str) -> List[Document]:
        """Load a plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        source_file = os.path.basename(file_path)
        ext = Path(file_path).suffix.lower()
        
        metadata = {
            "source": file_path,
            "source_file": source_file,
            "format": "text",
            "file_type": TEXT_EXTENSIONS.get(ext, "Text File"),
            "num_pages": 1,
            "ocr_enabled": False,
        }
        
        return [Document(page_content=content, metadata=metadata)]
    
    def load(self, file_path: str) -> List[Document]:
        """
        Load a single document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of LangChain Document objects (one per document)
        """
        file_path = os.path.expanduser(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = Path(file_path).suffix.lower()
        
        # Check if supported
        if ext not in ALL_SUPPORTED:
            raise ValueError(f"Unsupported file format: {ext}. Supported: {list(ALL_SUPPORTED.keys())}")
        
        # Handle plain text files directly
        if ext in TEXT_EXTENSIONS:
            return self._load_text_file(file_path)
        
        # Use Docling for document parsing
        converter = self._get_converter()
        
        # Convert the document
        result = converter.convert(file_path)
        
        # Export based on format preference
        if self.export_format == "markdown":
            content = result.document.export_to_markdown()
        elif self.export_format == "text":
            content = result.document.export_to_text()
        else:  # json
            content = result.document.export_to_dict()
            import json
            content = json.dumps(content, indent=2)
        
        # Build metadata
        source_file = os.path.basename(file_path)
        metadata = {
            "source": file_path,
            "source_file": source_file,
            "format": self.export_format,
            "file_type": DOCLING_SUPPORTED.get(ext, "Document"),
            "num_pages": len(result.document.pages) if hasattr(result.document, 'pages') else 1,
            "ocr_enabled": self.ocr_enabled,
        }
        
        # Create single document with full content
        documents = [Document(
            page_content=content,
            metadata=metadata
        )]
        
        return documents
    
    def load_directory(
        self, 
        directory: str, 
        recursive: bool = True,
        show_progress: bool = True,
        extensions: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to the directory
            recursive: Whether to search subdirectories
            show_progress: Whether to print progress
            extensions: List of extensions to include (default: all supported)
            
        Returns:
            List of all Document objects
        """
        directory = os.path.expanduser(directory)
        
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        # Default to all supported formats
        if extensions is None:
            extensions = list(ALL_SUPPORTED.keys())
        else:
            # Normalize extensions
            extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
        
        # Find all matching files
        all_files = []
        for ext in extensions:
            ext_clean = ext.lstrip('.')
            pattern = f"**/*.{ext_clean}" if recursive else f"*.{ext_clean}"
            all_files.extend(Path(directory).glob(pattern))
        
        # Sort for deterministic order
        all_files = sorted(set(all_files))
        
        if show_progress:
            print(f"Found {len(all_files)} document(s) in {directory}")
        
        all_documents = []
        
        for i, file_path in enumerate(all_files):
            if show_progress:
                print(f"  [{i+1}/{len(all_files)}] Loading: {file_path.name}")
            
            try:
                docs = self.load(str(file_path))
                all_documents.extend(docs)
                if show_progress:
                    print(f"      → Extracted successfully")
            except Exception as e:
                print(f"  ⚠ Error loading {file_path.name}: {e}")
        
        if show_progress:
            print(f"✓ Loaded {len(all_documents)} document(s)")
        
        return all_documents


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def load_document(
    file_path: str,
    export_format: ExportFormat = "markdown",
    ocr_enabled: bool = True,
) -> List[Document]:
    """
    Load a single document file (any supported format).
    
    Args:
        file_path: Path to document file
        export_format: 'markdown' (default), 'text', or 'json'
        ocr_enabled: Whether to enable OCR
        
    Returns:
        List of Document objects
    """
    loader = DoclingLoader(
        export_format=export_format,
        ocr_enabled=ocr_enabled,
    )
    return loader.load(file_path)


def load_pdf(
    file_path: str,
    export_format: ExportFormat = "markdown",
    ocr_enabled: bool = True,
) -> List[Document]:
    """Alias for load_document() - kept for backward compatibility."""
    return load_document(file_path, export_format, ocr_enabled)


def load_documents_from_directory(
    directory: str,
    export_format: ExportFormat = "markdown",
    ocr_enabled: bool = True,
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
) -> List[Document]:
    """
    Load all documents from a directory.
    
    Args:
        directory: Path to directory
        export_format: 'markdown' (default), 'text', or 'json'
        ocr_enabled: Whether to enable OCR
        recursive: Whether to search subdirectories
        extensions: List of extensions to include (default: all supported)
        
    Returns:
        List of Document objects
    """
    loader = DoclingLoader(
        export_format=export_format,
        ocr_enabled=ocr_enabled,
    )
    return loader.load_directory(directory, recursive=recursive, extensions=extensions)


# ============================================================
# HYBRID LOADER - CONFIDENCE-BASED ROUTING
# ============================================================

class HybridDocumentLoader:
    """
    Smart document loader with confidence-based routing.
    
    Strategy:
        1. Try Unstructured first (fast)
        2. Assess extraction quality
        3. If quality < threshold, fallback to Docling (slower but accurate)
    
    This provides the best of both worlds:
        - Fast extraction for clean digital documents
        - High-quality OCR fallback for scanned/complex documents
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.5,
        unstructured_strategy: str = "fast",
        docling_ocr: bool = True,
        docling_format: ExportFormat = "markdown",
    ):
        """
        Initialize the hybrid loader.
        
        Args:
            quality_threshold: Minimum quality score to accept Unstructured result (0.0-1.0)
            unstructured_strategy: Strategy for Unstructured ("fast", "hi_res", "auto")
            docling_ocr: Whether to enable OCR in Docling fallback
            docling_format: Export format for Docling ("markdown", "text", "json")
        """
        self.quality_threshold = quality_threshold
        self.unstructured_strategy = unstructured_strategy
        self.docling_ocr = docling_ocr
        self.docling_format = docling_format
        
        # Lazy-load components
        self._unstructured_loader = None
        self._docling_loader = None
        self._quality_assessor = None
    
    def _get_unstructured_loader(self):
        """Get or create UnstructuredLoader."""
        if self._unstructured_loader is None:
            from .unstructured_loader import UnstructuredLoader
            self._unstructured_loader = UnstructuredLoader(
                strategy=self.unstructured_strategy
            )
        return self._unstructured_loader
    
    def _get_docling_loader(self):
        """Get or create DoclingLoader."""
        if self._docling_loader is None:
            self._docling_loader = DoclingLoader(
                export_format=self.docling_format,
                ocr_enabled=self.docling_ocr,
            )
        return self._docling_loader
    
    def _get_quality_assessor(self):
        """Get or create ExtractionQuality assessor."""
        if self._quality_assessor is None:
            from .extraction_quality import ExtractionQuality
            self._quality_assessor = ExtractionQuality()
        return self._quality_assessor
    
    def load(
        self,
        file_path: str,
        verbose: bool = False,
    ) -> List[Document]:
        """
        Load a document with automatic parser selection based on quality.
        
        Args:
            file_path: Path to the document file
            verbose: Whether to print routing decisions
            
        Returns:
            List of LangChain Document objects
        """
        file_path = os.path.expanduser(file_path)
        ext = Path(file_path).suffix.lower()
        
        # For plain text files, use direct loading (no routing needed)
        if ext in TEXT_EXTENSIONS:
            docling = self._get_docling_loader()
            return docling._load_text_file(file_path)
        
        # Try Unstructured first
        unstructured = self._get_unstructured_loader()
        quality_assessor = self._get_quality_assessor()
        
        try:
            if verbose:
                print(f"⚡ Trying Unstructured (fast)...")
            
            docs = unstructured.load(file_path)
            
            if not docs or not docs[0].page_content.strip():
                raise ValueError("Empty extraction result")
            
            # Assess quality
            metrics = quality_assessor.assess(docs[0].page_content)
            
            if verbose:
                print(f"   Quality score: {metrics.confidence_score:.3f} (threshold: {self.quality_threshold})")
            
            if metrics.confidence_score >= self.quality_threshold:
                # Good quality - use Unstructured result
                if verbose:
                    print(f"   ✓ Accepted (fast path)")
                
                # Add quality metrics to metadata
                docs[0].metadata["quality_score"] = round(metrics.confidence_score, 3)
                docs[0].metadata["routing"] = "fast"
                return docs
            
            if verbose:
                print(f"   ✗ Quality too low, falling back to Docling...")
        
        except Exception as e:
            if verbose:
                print(f"   ✗ Unstructured failed: {e}")
                print(f"   Falling back to Docling...")
        
        # Fallback to Docling
        try:
            docling = self._get_docling_loader()
            docs = docling.load(file_path)
            
            # Mark as fallback in metadata
            if docs:
                docs[0].metadata["parser"] = "docling"
                docs[0].metadata["routing"] = "fallback"
                
                # Assess quality of Docling result
                metrics = quality_assessor.assess(docs[0].page_content)
                docs[0].metadata["quality_score"] = round(metrics.confidence_score, 3)
            
            if verbose:
                print(f"   ✓ Docling extraction complete")
            
            return docs
        
        except Exception as e:
            raise RuntimeError(f"Both parsers failed for {file_path}: {e}")
    
    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
        show_progress: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load all documents from a directory with hybrid routing.
        
        Args:
            directory: Path to the directory
            recursive: Whether to search subdirectories
            show_progress: Whether to print progress
            extensions: List of extensions to include
            
        Returns:
            List of all Document objects
        """
        directory = os.path.expanduser(directory)
        
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        # Default to all supported formats
        if extensions is None:
            extensions = list(ALL_SUPPORTED.keys())
        else:
            extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
        
        # Find all matching files
        all_files = []
        for ext in extensions:
            ext_clean = ext.lstrip('.')
            pattern = f"**/*.{ext_clean}" if recursive else f"*.{ext_clean}"
            all_files.extend(Path(directory).glob(pattern))
        
        all_files = sorted(set(all_files))
        
        if show_progress:
            print(f"Found {len(all_files)} document(s) in {directory}")
        
        all_documents = []
        fast_count = 0
        fallback_count = 0
        
        for i, file_path in enumerate(all_files):
            if show_progress:
                print(f"  [{i+1}/{len(all_files)}] Loading: {file_path.name}")
            
            try:
                docs = self.load(str(file_path), verbose=show_progress)
                all_documents.extend(docs)
                
                # Track routing stats
                if docs and docs[0].metadata.get("routing") == "fast":
                    fast_count += 1
                else:
                    fallback_count += 1
                    
            except Exception as e:
                print(f"  ⚠ Error loading {file_path.name}: {e}")
        
        if show_progress:
            print(f"✓ Loaded {len(all_documents)} document(s)")
            print(f"  Fast path: {fast_count}, Fallback: {fallback_count}")
        
        return all_documents


def load_document_smart(
    file_path: str,
    quality_threshold: float = 0.5,
    verbose: bool = False,
) -> List[Document]:
    """
    Load a document with automatic parser selection.
    
    Uses Unstructured for fast extraction, falls back to Docling
    if extraction quality is below threshold.
    
    Args:
        file_path: Path to document file
        quality_threshold: Minimum quality score (0.0-1.0)
        verbose: Whether to print routing decisions
        
    Returns:
        List of Document objects
    """
    loader = HybridDocumentLoader(quality_threshold=quality_threshold)
    return loader.load(file_path, verbose=verbose)


def load_documents_smart(
    directory: str,
    quality_threshold: float = 0.5,
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
) -> List[Document]:
    """
    Load all documents from a directory with smart routing.
    
    Args:
        directory: Path to directory
        quality_threshold: Minimum quality score (0.0-1.0)
        recursive: Whether to search subdirectories
        extensions: List of extensions to include
        
    Returns:
        List of Document objects
    """
    loader = HybridDocumentLoader(quality_threshold=quality_threshold)
    return loader.load_directory(directory, recursive=recursive, extensions=extensions)


def load_and_split(
    file_path: str,
    chunk_size: int = 1000,
    semantic: bool = True,
    quality_threshold: float = 0.5,
    verbose: bool = False,
) -> List[Document]:
    """
    Load a document and split it into chunks.
    
    Combines smart loading (Unstructured + Docling fallback) with
    semantic or recursive character splitting.
    
    Args:
        file_path: Path to document file
        chunk_size: Target chunk size (used as min_chunk_size for semantic)
        semantic: Whether to use semantic splitting (True) or character splitting
        quality_threshold: Quality threshold for hybrid loader
        verbose: Whether to print progress
        
    Returns:
        List of Document chunks
    """
    # Load the document
    docs = load_document_smart(file_path, quality_threshold=quality_threshold, verbose=verbose)
    
    if not docs:
        return []
    
    if semantic:
        # Use semantic splitter
        from .semantic_splitter import SemanticSplitter
        splitter = SemanticSplitter(min_chunk_size=chunk_size)
        return splitter.split_documents(docs)
    else:
        # Use recursive character splitter
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 10,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(docs)


# ============================================================
# CLI INTERFACE
# ============================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load documents using Docling (IBM Research)",
        epilog=f"Supported formats: {', '.join(sorted(ALL_SUPPORTED.keys()))}"
    )
    parser.add_argument("path", help="Document file or directory path")
    parser.add_argument(
        "--format", 
        choices=["markdown", "text", "json"], 
        default="markdown",
        help="Export format (default: markdown)"
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR for scanned content"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=1000,
        help="Number of characters to preview (default: 1000)"
    )
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="List all supported file formats and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_formats:
        print("Supported file formats:\n")
        print("=== Documents (Docling) ===")
        for ext, desc in sorted(DOCLING_SUPPORTED.items()):
            print(f"  {ext:10} - {desc}")
        print("\n=== Text Files (Direct) ===")
        for ext, desc in sorted(TEXT_EXTENSIONS.items()):
            print(f"  {ext:10} - {desc}")
        sys.exit(0)
    
    path = os.path.expanduser(args.path)
    
    print(f"📄 Format: {args.format}")
    print(f"🔍 OCR: {'enabled' if not args.no_ocr else 'disabled'}")
    print()
    
    if os.path.isfile(path):
        docs = load_document(
            path,
            export_format=args.format,
            ocr_enabled=not args.no_ocr,
        )
        print(f"\n✓ Loaded {len(docs)} document(s) from {path}")
        
        if docs:
            print(f"\n--- Preview (first {args.preview} chars) ---")
            print(docs[0].page_content[:args.preview])
            print("\n--- Metadata ---")
            for key, value in docs[0].metadata.items():
                print(f"  {key}: {value}")
    
    elif os.path.isdir(path):
        docs = load_documents_from_directory(
            path,
            export_format=args.format,
            ocr_enabled=not args.no_ocr,
        )
        print(f"\n✓ Total: {len(docs)} document(s) loaded")
    
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)
