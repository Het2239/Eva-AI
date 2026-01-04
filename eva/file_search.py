#!/usr/bin/env python3
"""
EVA Intelligent File Search
============================
AI-powered file search with:
- LLM query understanding (extracts drive, folder, file hints)
- Cross-volume search with auto-mount
- Semantic + fuzzy matching
- File caching with content summaries
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

# Local imports
try:
    from eva.file_cache import FileCache, CachedFile
    from eva.os_tools import OSTools, FileMatch
except ImportError:
    from file_cache import FileCache, CachedFile
    from os_tools import OSTools, FileMatch


@dataclass
class ParsedQuery:
    """Structured query extracted from natural language."""
    original: str
    file_hint: str  # What to search for
    drive_hint: Optional[str] = None  # D drive, USB, external
    folder_hint: Optional[str] = None  # downloads, documents, projects
    file_type: Optional[str] = None  # .pdf, .docx, image


@dataclass  
class FileSearchResult:
    """Search result with metadata."""
    path: str
    name: str
    score: float
    source: str  # "cache", "search", "fuzzy"
    summary: Optional[str] = None
    drive: Optional[str] = None


class FileSearchEngine:
    """
    Intelligent file search with LLM query parsing.
    
    Flow:
        User Query -> LLM Parse -> Cache Check -> Volume Search -> Results
    """
    
    def __init__(
        self,
        cache_path: str = "file_cache.db",
        auto_mount: bool = False,
    ):
        """
        Initialize file search engine.
        
        Args:
            cache_path: Path to SQLite cache database
            auto_mount: Whether to auto-mount volumes (requires confirmation)
        """
        self.cache = FileCache(cache_path)
        self.os_tools = OSTools()
        self.auto_mount = auto_mount
        self._mount_callback = None  # For UI confirmation
    
    def set_mount_callback(self, callback):
        """Set callback for mount confirmation: callback(device, mount_point) -> bool"""
        self._mount_callback = callback
    
    # ========================================
    # QUERY PARSING
    # ========================================
    
    def parse_query(self, query: str) -> ParsedQuery:
        """
        Parse natural language query to extract location hints.
        
        Uses pattern matching first, falls back to LLM if available.
        
        Examples:
            "find physics quiz in downloads" -> file_hint="physics quiz", folder_hint="downloads"
            "open resume.pdf on D drive" -> file_hint="resume.pdf", drive_hint="D"
        """
        query_lower = query.lower()
        
        # Extract file type hints
        file_type = None
        type_patterns = {
            r'\bpdf\b': '.pdf',
            r'\bdocx?\b': '.docx',
            r'\bimage\b|\bpicture\b|\bphoto\b': '.jpg',
            r'\bvideo\b': '.mp4',
            r'\bexcel\b|\bxlsx?\b': '.xlsx',
            r'\bpowerpoint\b|\bpptx?\b': '.pptx',
            r'\btext\b|\btxt\b': '.txt',
        }
        for pattern, ext in type_patterns.items():
            if re.search(pattern, query_lower):
                file_type = ext
                break
        
        # Extract drive hints
        drive_hint = None
        drive_patterns = [
            (r'\b([a-z])\s*drive\b', lambda m: m.group(1).upper()),
            (r'\busb\b|\bpendrive\b|\bflash\b', lambda m: "USB"),
            (r'\bexternal\b', lambda m: "EXTERNAL"),
            (r'\bsd\s*card\b', lambda m: "SD"),
        ]
        for pattern, extractor in drive_patterns:
            match = re.search(pattern, query_lower)
            if match:
                drive_hint = extractor(match)
                break
        
        # Extract folder hints
        folder_hint = None
        folder_patterns = {
            r'\bdownloads?\b': 'downloads',
            r'\bdocuments?\b': 'documents',
            r'\bdesktop\b': 'desktop',
            r'\bpictures?\b': 'pictures',
            r'\bvideos?\b': 'videos',
            r'\bmusic\b': 'music',
            r'\bprojects?\b': 'projects',
            r'\bhome\b': 'home',
        }
        for pattern, folder in folder_patterns.items():
            if re.search(pattern, query_lower):
                folder_hint = folder
                break
        
        # Extract file hint (the actual search term)
        file_hint = self._extract_file_hint(query_lower, drive_hint, folder_hint, file_type)
        
        return ParsedQuery(
            original=query,
            file_hint=file_hint,
            drive_hint=drive_hint,
            folder_hint=folder_hint,
            file_type=file_type,
        )
    
    def _extract_file_hint(
        self,
        query: str,
        drive_hint: Optional[str],
        folder_hint: Optional[str],
        file_type: Optional[str],
    ) -> str:
        """Extract the actual file search term from query."""
        # Remove common action words
        removals = [
            r'^(find|search|look for|locate|open|show|get)\s+',
            r'\s+(file|document|image|video|photo)\s*$',
            r'\s+in\s+(the\s+)?(\w+\s+)?(folder|directory|drive)\s*$',
            r'\s+on\s+(the\s+)?(\w+\s+)?(drive|disk|usb)\s*$',
            r'\s+somewhere\b.*$',
            r'\s+which\s+would\s+be\b.*$',
            r'\bthe\b',
            r'\bthat\b',
            r'\bmy\b',
            r'\ba\b',
        ]
        
        result = query
        for pattern in removals:
            result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)
        
        # Remove extracted hints from search term
        if drive_hint:
            result = re.sub(rf'\b{drive_hint}\b', '', result, flags=re.IGNORECASE)
            result = re.sub(r'\bdrive\b', '', result, flags=re.IGNORECASE)
        if folder_hint:
            result = re.sub(rf'\b{folder_hint}\b', '', result, flags=re.IGNORECASE)
        if file_type:
            ext_word = file_type.replace('.', '')
            result = re.sub(rf'\b{ext_word}\b', '', result, flags=re.IGNORECASE)
        
        # Clean up
        result = ' '.join(result.split()).strip()
        return result if result else query
    
    # ========================================
    # VOLUME DISCOVERY
    # ========================================
    
    def get_all_volumes(self) -> List[Dict[str, Any]]:
        """
        Get all volumes/partitions on the system.
        
        Returns list of:
            {name, path, mounted, device, size, fstype}
        """
        volumes = []
        
        try:
            # Use lsblk for comprehensive info
            result = subprocess.run(
                ['lsblk', '-J', '-o', 'NAME,MOUNTPOINT,SIZE,TYPE,FSTYPE'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                for device in data.get('blockdevices', []):
                    self._parse_device(device, volumes)
        except Exception as e:
            print(f"Volume discovery error: {e}")
        
        # Add standard user directories
        home = os.path.expanduser("~")
        for folder in ["Downloads", "Documents", "Desktop", "Pictures", "Videos", "Music"]:
            path = os.path.join(home, folder)
            if os.path.isdir(path):
                volumes.append({
                    "name": folder,
                    "path": path,
                    "mounted": True,
                    "device": None,
                    "type": "user_folder",
                })
        
        return volumes
    
    def _parse_device(self, device: Dict, volumes: List, parent_name: str = ""):
        """Recursively parse lsblk device tree."""
        dev_type = device.get('type', '')
        mount = device.get('mountpoint')
        name = device.get('name', '')
        
        if dev_type == 'part' and device.get('fstype'):
            volumes.append({
                "name": name,
                "path": mount,
                "mounted": mount is not None,
                "device": f"/dev/{name}",
                "size": device.get('size'),
                "fstype": device.get('fstype'),
                "type": "partition",
            })
        
        # Recurse into children
        for child in device.get('children', []):
            self._parse_device(child, volumes, name)
    
    def get_search_paths(self, parsed: ParsedQuery) -> List[str]:
        """
        Get paths to search based on query hints.
        
        If no hints, returns all searchable paths.
        """
        volumes = self.get_all_volumes()
        paths = []
        
        # If drive hint, filter by drive
        if parsed.drive_hint:
            for vol in volumes:
                if not vol.get('mounted'):
                    # Offer to mount
                    if self._should_mount(vol):
                        mount_path = self._mount_volume(vol)
                        if mount_path:
                            paths.append(mount_path)
                elif parsed.drive_hint in vol['name'].upper():
                    paths.append(vol['path'])
        
        # If folder hint, use mapped folder
        if parsed.folder_hint:
            folder_map = {
                'downloads': '~/Downloads',
                'documents': '~/Documents',
                'desktop': '~/Desktop',
                'pictures': '~/Pictures',
                'videos': '~/Videos',
                'music': '~/Music',
                'home': '~',
            }
            if parsed.folder_hint in folder_map:
                paths.append(os.path.expanduser(folder_map[parsed.folder_hint]))
        
        # No hints -> search everywhere
        if not paths:
            for vol in volumes:
                if vol.get('mounted') and vol.get('path'):
                    paths.append(vol['path'])
                elif not vol.get('mounted') and vol.get('type') == 'partition':
                    if self._should_mount(vol):
                        mount_path = self._mount_volume(vol)
                        if mount_path:
                            paths.append(mount_path)
        
        return list(set(paths))  # Dedupe
    
    def _should_mount(self, volume: Dict) -> bool:
        """Check if we should mount this volume."""
        if not self.auto_mount:
            return False
        if self._mount_callback:
            return self._mount_callback(volume['device'], f"/mnt/{volume['name']}")
        return False
    
    def _mount_volume(self, volume: Dict) -> Optional[str]:
        """Attempt to mount a volume."""
        device = volume.get('device')
        if not device:
            return None
        
        mount_point = f"/mnt/{volume['name']}"
        try:
            # Create mount point
            subprocess.run(['sudo', 'mkdir', '-p', mount_point], check=True)
            # Mount
            subprocess.run(['sudo', 'mount', device, mount_point], check=True)
            print(f"✓ Mounted {device} to {mount_point}")
            return mount_point
        except subprocess.CalledProcessError as e:
            print(f"Mount failed: {e}")
            return None
    
    # ========================================
    # SEARCH
    # ========================================
    
    def search(
        self,
        query: str,
        limit: int = 10,
        use_cache: bool = True,
        generate_summaries: bool = True,
    ) -> List[FileSearchResult]:
        """
        Main search method.
        
        Flow:
            1. Parse query
            2. Check cache
            3. Scan paths
            4. Rank results
            5. Cache new files
            6. Generate summaries (optional)
        """
        parsed = self.parse_query(query)
        results = []
        
        # 1. Check cache first (fast path)
        if use_cache:
            cached = self.cache.search(
                parsed.file_hint,
                file_type=parsed.file_type,
                limit=limit,
            )
            for c in cached:
                # Verify file still exists
                if os.path.exists(c.path):
                    results.append(FileSearchResult(
                        path=c.path,
                        name=c.name,
                        score=1.0,
                        source="cache",
                        summary=c.summary,
                    ))
        
        # 2. If not enough results, do full search
        if len(results) < limit:
            paths = self.get_search_paths(parsed)
            fresh_results = self._scan_paths(parsed, paths, limit - len(results))
            
            # Cache new results
            for r in fresh_results:
                self.cache.add(r.path)
                results.append(r)
        
        # 3. Generate summaries for top results
        if generate_summaries:
            for r in results[:3]:  # Top 3 only (to save tokens)
                if not r.summary and self._is_summarizable(r.path):
                    summary = self._generate_summary(r.path)
                    if summary:
                        r.summary = summary
                        self.cache.update_summary(r.path, summary)
        
        # Sort by score
        results.sort(key=lambda x: -x.score)
        return results[:limit]
    
    def _scan_paths(
        self,
        parsed: ParsedQuery,
        paths: List[str],
        limit: int,
    ) -> List[FileSearchResult]:
        """Scan paths with fuzzy matching."""
        results = []
        
        for path in paths:
            if not os.path.isdir(path):
                continue
            
            matches = self.os_tools.resolve_file(
                parsed.file_hint,
                search_path=path,
                file_types=[parsed.file_type] if parsed.file_type else None,
                recursive=True,
                limit=limit,
            )
            
            for m in matches:
                results.append(FileSearchResult(
                    path=m.path,
                    name=m.name,
                    score=m.score / 100.0,
                    source="search",
                ))
        
        return results
    
    def _is_summarizable(self, path: str) -> bool:
        """Check if file can be summarized."""
        ext = os.path.splitext(path)[1].lower()
        return ext in ['.pdf', '.docx', '.doc', '.txt', '.md', '.pptx']
    
    def _generate_summary(self, path: str) -> Optional[str]:
        """Generate content summary using LLM."""
        try:
            # Import models for LLM
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from models import get_chat_response
            
            # Read file content (first 2000 chars)
            content = self._read_file_content(path, max_chars=2000)
            if not content:
                return None
            
            prompt = f"""Summarize this document in 1-2 sentences. Focus on what it's about:

{content}

Summary:"""
            
            summary = get_chat_response(prompt)
            return summary.strip()[:500]  # Cap at 500 chars
        except Exception as e:
            print(f"Summary generation failed: {e}")
            return None
    
    def _read_file_content(self, path: str, max_chars: int = 2000) -> Optional[str]:
        """Read file content for summarization."""
        ext = os.path.splitext(path)[1].lower()
        
        try:
            if ext == '.txt' or ext == '.md':
                with open(path, 'r', errors='ignore') as f:
                    return f.read(max_chars)
            
            elif ext == '.pdf':
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(path)
                    text = ""
                    for page in doc[:3]:  # First 3 pages
                        text += page.get_text()
                        if len(text) > max_chars:
                            break
                    return text[:max_chars]
                except:
                    pass
            
            elif ext in ['.docx', '.doc']:
                try:
                    from docx import Document
                    doc = Document(path)
                    text = "\n".join([p.text for p in doc.paragraphs[:20]])
                    return text[:max_chars]
                except:
                    pass
        except Exception as e:
            print(f"Read error: {e}")
        
        return None


# ============================================================
# CLI for testing
# ============================================================

if __name__ == "__main__":
    import sys
    
    engine = FileSearchEngine()
    
    if len(sys.argv) < 2:
        print("Usage: python file_search.py <query>")
        print("Example: python file_search.py 'find physics quiz pdf in downloads'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    print(f"Query: {query}")
    print("-" * 40)
    
    # Parse
    parsed = engine.parse_query(query)
    print(f"Parsed:")
    print(f"  File hint: {parsed.file_hint}")
    print(f"  Drive: {parsed.drive_hint}")
    print(f"  Folder: {parsed.folder_hint}")
    print(f"  Type: {parsed.file_type}")
    print("-" * 40)
    
    # Search
    results = engine.search(query, generate_summaries=False)
    print(f"Results ({len(results)}):")
    for r in results:
        print(f"  [{r.score:.2f}] {r.name}")
        print(f"       {r.path}")
        if r.summary:
            print(f"       Summary: {r.summary[:80]}...")
