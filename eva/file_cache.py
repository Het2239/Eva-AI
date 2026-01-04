#!/usr/bin/env python3
"""
EVA File Cache - SQLite-based file indexing with content summaries
===================================================================
Caches file paths and LLM-generated summaries for fast semantic search.
"""

import os
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CachedFile:
    """Cached file entry."""
    path: str
    name: str
    summary: Optional[str]
    keywords: List[str]
    file_type: str
    size: int
    last_accessed: datetime
    content_hash: Optional[str] = None


class FileCache:
    """
    SQLite-based file cache with content summaries.
    
    Stores:
        - File paths and metadata
        - LLM-generated content summaries
        - Keywords for search
    """
    
    def __init__(self, db_path: str = "file_cache.db"):
        """Initialize cache database."""
        self.db_path = os.path.expanduser(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if not exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                summary TEXT,
                keywords TEXT,
                file_type TEXT,
                size INTEGER,
                content_hash TEXT,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_path ON files(path);
            CREATE INDEX IF NOT EXISTS idx_name ON files(name);
            CREATE INDEX IF NOT EXISTS idx_keywords ON files(keywords);
            
            CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                name, summary, keywords,
                content='files',
                content_rowid='id'
            );
            
            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
                INSERT INTO files_fts(rowid, name, summary, keywords)
                VALUES (new.id, new.name, new.summary, new.keywords);
            END;
            
            CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
                INSERT INTO files_fts(files_fts, rowid, name, summary, keywords)
                VALUES ('delete', old.id, old.name, old.summary, old.keywords);
            END;
            
            CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE ON files BEGIN
                INSERT INTO files_fts(files_fts, rowid, name, summary, keywords)
                VALUES ('delete', old.id, old.name, old.summary, old.keywords);
                INSERT INTO files_fts(rowid, name, summary, keywords)
                VALUES (new.id, new.name, new.summary, new.keywords);
            END;
        """)
        self.conn.commit()
    
    def add(
        self,
        path: str,
        summary: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> bool:
        """
        Add or update a file in the cache.
        
        Args:
            path: Full file path
            summary: Content summary (optional)
            keywords: Search keywords (optional)
            
        Returns:
            True if added/updated successfully
        """
        try:
            path = os.path.abspath(os.path.expanduser(path))
            if not os.path.exists(path):
                return False
            
            stat = os.stat(path)
            name = os.path.basename(path)
            file_type = os.path.splitext(name)[1].lower()
            keywords_str = ",".join(keywords) if keywords else ""
            
            # Compute content hash for change detection
            content_hash = None
            if os.path.isfile(path) and stat.st_size < 10_000_000:  # < 10MB
                try:
                    with open(path, "rb") as f:
                        content_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    pass
            
            self.conn.execute("""
                INSERT INTO files (path, name, summary, keywords, file_type, size, content_hash, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(path) DO UPDATE SET
                    summary = COALESCE(excluded.summary, summary),
                    keywords = COALESCE(NULLIF(excluded.keywords, ''), keywords),
                    size = excluded.size,
                    content_hash = excluded.content_hash,
                    last_accessed = CURRENT_TIMESTAMP
            """, (path, name, summary, keywords_str, file_type, stat.st_size, content_hash))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Cache add error: {e}")
            return False
    
    def search(
        self,
        query: str,
        file_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[CachedFile]:
        """
        Search cached files using FTS.
        
        Args:
            query: Search query
            file_type: Filter by extension (e.g., ".pdf")
            limit: Max results
            
        Returns:
            List of matching cached files
        """
        try:
            # Build FTS query
            fts_query = query.replace("'", "''")
            
            sql = """
                SELECT f.*, rank
                FROM files f
                JOIN files_fts fts ON f.id = fts.rowid
                WHERE files_fts MATCH ?
            """
            params = [fts_query]
            
            if file_type:
                sql += " AND f.file_type = ?"
                params.append(file_type.lower())
            
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
            
            rows = self.conn.execute(sql, params).fetchall()
            return [self._row_to_cached_file(row) for row in rows]
        except Exception as e:
            # FTS might fail on complex queries, fall back to LIKE
            return self._search_fallback(query, file_type, limit)
    
    def _search_fallback(
        self,
        query: str,
        file_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[CachedFile]:
        """Fallback search using LIKE."""
        sql = """
            SELECT * FROM files
            WHERE (name LIKE ? OR summary LIKE ? OR keywords LIKE ?)
        """
        like_query = f"%{query}%"
        params = [like_query, like_query, like_query]
        
        if file_type:
            sql += " AND file_type = ?"
            params.append(file_type.lower())
        
        sql += " ORDER BY last_accessed DESC LIMIT ?"
        params.append(limit)
        
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_cached_file(row) for row in rows]
    
    def get(self, path: str) -> Optional[CachedFile]:
        """Get cached file by path."""
        row = self.conn.execute(
            "SELECT * FROM files WHERE path = ?",
            (os.path.abspath(path),)
        ).fetchone()
        return self._row_to_cached_file(row) if row else None
    
    def update_summary(self, path: str, summary: str) -> bool:
        """Update file summary."""
        try:
            self.conn.execute(
                "UPDATE files SET summary = ? WHERE path = ?",
                (summary, os.path.abspath(path))
            )
            self.conn.commit()
            return True
        except:
            return False
    
    def remove(self, path: str) -> bool:
        """Remove file from cache."""
        try:
            self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
            self.conn.commit()
            return True
        except:
            return False
    
    def get_recent(self, limit: int = 20) -> List[CachedFile]:
        """Get recently accessed files."""
        rows = self.conn.execute(
            "SELECT * FROM files ORDER BY last_accessed DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [self._row_to_cached_file(row) for row in rows]
    
    def _row_to_cached_file(self, row) -> CachedFile:
        """Convert database row to CachedFile."""
        keywords = row["keywords"].split(",") if row["keywords"] else []
        return CachedFile(
            path=row["path"],
            name=row["name"],
            summary=row["summary"],
            keywords=keywords,
            file_type=row["file_type"] or "",
            size=row["size"] or 0,
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else datetime.now(),
            content_hash=row["content_hash"],
        )
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        row = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(summary) as with_summary,
                SUM(size) as total_size
            FROM files
        """).fetchone()
        return {
            "total_files": row["total"],
            "with_summary": row["with_summary"],
            "total_size_bytes": row["total_size"] or 0,
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()


# ============================================================
# CLI for testing
# ============================================================

if __name__ == "__main__":
    import sys
    
    cache = FileCache("file_cache.db")
    
    if len(sys.argv) < 2:
        print("Usage: python file_cache.py [add|search|stats] ...")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "add" and len(sys.argv) >= 3:
        path = sys.argv[2]
        summary = sys.argv[3] if len(sys.argv) > 3 else None
        if cache.add(path, summary):
            print(f"✓ Added: {path}")
        else:
            print(f"✗ Failed: {path}")
    
    elif cmd == "search" and len(sys.argv) >= 3:
        query = " ".join(sys.argv[2:])
        results = cache.search(query)
        print(f"Found {len(results)} results for '{query}':")
        for r in results:
            print(f"  {r.name} ({r.path})")
            if r.summary:
                print(f"    Summary: {r.summary[:100]}...")
    
    elif cmd == "stats":
        stats = cache.stats()
        print(f"Cache stats: {stats}")
    
    else:
        print("Unknown command")
    
    cache.close()
