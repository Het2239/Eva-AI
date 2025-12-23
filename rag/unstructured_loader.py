#!/usr/bin/env python3
"""
EVA RAG - Unstructured Loader
=============================
Fast document loader using the Unstructured library.
"""

import os
from pathlib import Path
from typing import List, Optional, Literal

from langchain_core.documents import Document


# Strategy types for Unstructured
UnstructuredStrategy = Literal["fast", "hi_res", "auto"]

# Supported formats for Unstructured
UNSTRUCTURED_SUPPORTED = {
    ".pdf": "PDF Document",
    ".docx": "Word Document",
    ".doc": "Word Document (Legacy)",
    ".pptx": "PowerPoint Presentation",
    ".xlsx": "Excel Spreadsheet",
    ".html": "HTML Document",
    ".htm": "HTML Document",
    ".txt": "Plain Text",
    ".md": "Markdown Document",
    ".rst": "reStructuredText",
    ".xml": "XML Document",
    ".csv": "CSV File",
    ".eml": "Email Message",
    ".msg": "Outlook Message",
    ".rtf": "Rich Text Format",
    ".epub": "EPUB Document",
    ".odt": "OpenDocument Text",
    ".png": "PNG Image",
    ".jpg": "JPEG Image",
    ".jpeg": "JPEG Image",
    ".tiff": "TIFF Image",
    ".bmp": "BMP Image",
}


class UnstructuredLoader:
    """
    Fast document loader using the Unstructured library.
    
    Features:
        - Fast extraction for digital documents
        - Multiple format support
        - Configurable strategy (fast, hi_res, auto)
    """
    
    def __init__(
        self,
        strategy: UnstructuredStrategy = "hi_res",
        ocr_languages: Optional[str] = "eng",
        include_page_breaks: bool = True,
    ):
        """
        Initialize the Unstructured loader.
        
        Args:
            strategy: Extraction strategy ("fast", "hi_res", "auto")
            ocr_languages: OCR languages (e.g., "eng", "eng+fra")
            include_page_breaks: Whether to include page break markers
        """
        self.strategy = strategy
        self.ocr_languages = ocr_languages
        self.include_page_breaks = include_page_breaks
    
    @staticmethod
    def supported_formats() -> dict:
        """Return dictionary of supported file formats."""
        return UNSTRUCTURED_SUPPORTED
    
    @staticmethod
    def is_supported(file_path: str) -> bool:
        """Check if a file format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in UNSTRUCTURED_SUPPORTED
    
    def load(self, file_path: str) -> List[Document]:
        """
        Load a single document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of LangChain Document objects
        """
        file_path = os.path.expanduser(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = Path(file_path).suffix.lower()
        
        if ext not in UNSTRUCTURED_SUPPORTED:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Import unstructured partition function
        from unstructured.partition.auto import partition
        
        # Build kwargs based on file type
        kwargs = {
            "filename": file_path,
            "include_page_breaks": self.include_page_breaks,
        }
        
        # Add strategy for PDFs and images
        if ext in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            kwargs["strategy"] = self.strategy
            if self.ocr_languages:
                kwargs["ocr_languages"] = self.ocr_languages
        
        # Partition the document
        elements = partition(**kwargs)
        
        # Combine elements into text content
        content_parts = []
        for element in elements:
            text = str(element)
            if text.strip():
                content_parts.append(text)
        
        content = "\n\n".join(content_parts)
        
        # Build metadata
        source_file = os.path.basename(file_path)
        metadata = {
            "source": file_path,
            "source_file": source_file,
            "parser": "unstructured",
            "strategy": self.strategy,
            "file_type": UNSTRUCTURED_SUPPORTED.get(ext, "Document"),
            "num_elements": len(elements),
        }
        
        return [Document(page_content=content, metadata=metadata)]
    
    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
        show_progress: bool = True,
        extensions: Optional[List[str]] = None,
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
            extensions = list(UNSTRUCTURED_SUPPORTED.keys())
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
# CLI INTERFACE
# ============================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load documents using Unstructured (fast extraction)",
    )
    parser.add_argument("path", help="Document file or directory path")
    parser.add_argument(
        "--strategy",
        choices=["fast", "hi_res", "auto"],
        default="hi_res",
        help="Extraction strategy (default: hi_res)"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=1000,
        help="Number of characters to preview (default: 1000)"
    )
    
    args = parser.parse_args()
    
    path = os.path.expanduser(args.path)
    
    print(f"📄 Strategy: {args.strategy}")
    print()
    
    loader = UnstructuredLoader(strategy=args.strategy)
    
    if os.path.isfile(path):
        docs = loader.load(path)
        print(f"\n✓ Loaded {len(docs)} document(s) from {path}")
        
        if docs:
            print(f"\n--- Preview (first {args.preview} chars) ---")
            print(docs[0].page_content[:args.preview])
            print("\n--- Metadata ---")
            for key, value in docs[0].metadata.items():
                print(f"  {key}: {value}")
    
    elif os.path.isdir(path):
        docs = loader.load_directory(path)
        print(f"\n✓ Total: {len(docs)} document(s) loaded")
    
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)
