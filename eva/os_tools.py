#!/usr/bin/env python3
"""
EVA OS Tools - Safe OS Operations
==================================
Whitelisted, cross-platform OS operations. NO raw shell access.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Fuzzy matching
try:
    from thefuzz import fuzz, process
except ImportError:
    fuzz = None
    process = None


@dataclass
class FileMatch:
    """Result of fuzzy file matching."""
    path: str
    name: str
    score: int
    file_type: str


class OSTools:
    """
    Safe OS operations for EVA.
    
    NO raw shell access - only whitelisted operations.
    """
    
    def __init__(self, cwd: Optional[str] = None, custom_paths: Optional[Dict[str, str]] = None):
        """
        Initialize with current working directory.
        
        Args:
            cwd: Current working directory
            custom_paths: Custom folder mappings e.g. {"data": "/mnt/data", "projects": "/home/user/Projects"}
        """
        self._cwd = cwd or os.path.expanduser("~")
        self._system = platform.system()
        self._custom_paths = custom_paths or {}
    
    # ========================================
    # PROPERTIES
    # ========================================
    
    @property
    def cwd(self) -> str:
        """Get current working directory."""
        return self._cwd
    
    @cwd.setter
    def cwd(self, path: str) -> None:
        """Set current working directory."""
        path = self._resolve_path(path)
        if os.path.isdir(path):
            self._cwd = path
    
    # ========================================
    # PATH RESOLUTION
    # ========================================
    
    def _resolve_path(self, path: str) -> str:
        """Resolve path (expand ~, make absolute)."""
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.join(self._cwd, path)
        return os.path.normpath(path)
    
    # ========================================
    # FILE/FOLDER OPERATIONS
    # ========================================
    
    def open_path(self, path: str) -> str:
        """
        Open file or folder with default application.
        
        Cross-platform: Linux (xdg-open), Mac (open), Windows (startfile)
        """
        path = self._resolve_path(path)
        
        if not os.path.exists(path):
            return f"Error: Path does not exist: {path}"
        
        try:
            if self._system == "Windows":
                os.startfile(path)
            elif self._system == "Darwin":
                subprocess.run(["open", path], check=True)
            else:
                subprocess.run(["xdg-open", path], check=True)
            
            name = os.path.basename(path)
            return f"Opened: {name}"
        except Exception as e:
            return f"Error opening path: {e}"
    
    def open_file(self, path: str) -> str:
        """Open a file with default application."""
        path = self._resolve_path(path)
        if not os.path.isfile(path):
            return f"Error: Not a file: {path}"
        return self.open_path(path)
    
    def open_folder(self, path: str) -> str:
        """Open a folder in file manager."""
        path = self._resolve_path(path)
        if not os.path.isdir(path):
            return f"Error: Not a folder: {path}"
        return self.open_path(path)
    
    def list_directory(self, path: Optional[str] = None) -> List[Dict]:
        """
        List directory contents safely.
        
        Returns list of {name, path, type, size, modified}
        """
        path = self._resolve_path(path) if path else self._cwd
        
        if not os.path.isdir(path):
            return []
        
        items = []
        try:
            for name in os.listdir(path):
                full_path = os.path.join(path, name)
                try:
                    stat = os.stat(full_path)
                    items.append({
                        "name": name,
                        "path": full_path,
                        "type": "dir" if os.path.isdir(full_path) else "file",
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    })
                except:
                    pass
        except PermissionError:
            pass
        
        return items
    
    def get_cwd(self) -> str:
        """Get current working directory."""
        return self._cwd
    
    def set_cwd(self, path: str) -> str:
        """Set current working directory."""
        path = self._resolve_path(path)
        if os.path.isdir(path):
            self._cwd = path
            return f"Changed to: {path}"
        return f"Error: Not a directory: {path}"
    
    # ========================================
    # FUZZY FILE RESOLUTION
    # ========================================
    
    def resolve_file(
        self,
        query: str,
        search_path: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        limit: int = 5,
        recursive: bool = False,
    ) -> List[FileMatch]:
        """
        Fuzzy match files by name.
        
        Args:
            query: Search query (e.g., "physics notes")
            search_path: Directory to search (default: cwd)
            file_types: Filter by extensions (e.g., [".pdf", ".docx"])
            limit: Max results
            recursive: Search subdirectories
            
        Returns:
            List of FileMatch sorted by score
        """
        search_path = self._resolve_path(search_path) if search_path else self._cwd
        
        if not os.path.isdir(search_path):
            return []
        
        matches = []
        
        def search_dir(path: str, depth: int = 0):
            if depth > 5:  # Max recursion depth
                return
            
            for item in self.list_directory(path):
                name = item["name"]
                
                # Recurse into directories
                if recursive and item["type"] == "dir" and not name.startswith("."):
                    search_dir(item["path"], depth + 1)
                
                # Skip directories for matching
                if item["type"] == "dir":
                    continue
                
                # Filter by type
                if file_types:
                    ext = os.path.splitext(name)[1].lower()
                    if ext not in file_types:
                        continue
                
                # Score using fuzzy matching
                if fuzz:
                    score = fuzz.partial_ratio(query.lower(), name.lower())
                else:
                    score = 100 if query.lower() in name.lower() else 0
                
                if score > 30:
                    matches.append(FileMatch(
                        path=item["path"],
                        name=name,
                        score=score,
                        file_type=item["type"],
                    ))
        
        search_dir(search_path)
        
        # Sort by score
        matches.sort(key=lambda x: -x.score)
        return matches[:limit]
    
    def get_mounted_drives(self) -> List[Dict[str, str]]:
        """
        Get list of mounted drives/partitions.
        
        Returns:
            List of {name: str, path: str, type: str}
        """
        drives = []
        
        if self._system == "Linux":
            # Check /media/<user>/ for removable drives
            media_path = f"/media/{os.getenv('USER', 'user')}"
            if os.path.isdir(media_path):
                for name in os.listdir(media_path):
                    full_path = os.path.join(media_path, name)
                    if os.path.isdir(full_path):
                        drives.append({
                            "name": name,
                            "path": full_path,
                            "type": "removable"
                        })
            
            # Check /mnt/ for manual mounts
            mnt_path = "/mnt"
            if os.path.isdir(mnt_path):
                for name in os.listdir(mnt_path):
                    full_path = os.path.join(mnt_path, name)
                    if os.path.isdir(full_path) and os.listdir(full_path):
                        drives.append({
                            "name": name,
                            "path": full_path,
                            "type": "mount"
                        })
            
            # Check /run/media/<user>/ (some distros)
            run_media = f"/run/media/{os.getenv('USER', 'user')}"
            if os.path.isdir(run_media):
                for name in os.listdir(run_media):
                    full_path = os.path.join(run_media, name)
                    if os.path.isdir(full_path):
                        drives.append({
                            "name": name,
                            "path": full_path,
                            "type": "removable"
                        })
        
        elif self._system == "Windows":
            # Get drive letters
            import string
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append({
                        "name": f"{letter} Drive",
                        "path": drive,
                        "type": "local"
                    })
        
        return drives
    
    def smart_find(self, query: str, limit: int = 10) -> List[FileMatch]:
        """
        Smart file search with folder hints and recursive search.
        
        Parses queries like "mahabharatta in downloads or documents folder"
        Also searches mounted drives if mentioned.
        """
        query_lower = query.lower()
        
        # Common folder mappings
        folder_map = {
            "downloads": "~/Downloads",
            "download": "~/Downloads",
            "documents": "~/Documents",
            "document": "~/Documents",
            "desktop": "~/Desktop",
            "pictures": "~/Pictures",
            "videos": "~/Videos",
            "music": "~/Music",
            "home": "~",
        }
        
        # Add mounted drives to folder map
        for drive in self.get_mounted_drives():
            drive_name = drive["name"].lower()
            folder_map[drive_name] = drive["path"]
            # Also add without spaces for easier matching
            folder_map[drive_name.replace(" ", "")] = drive["path"]
        
        # Add custom paths
        for name, path in self._custom_paths.items():
            folder_map[name.lower()] = path
        
        # Parse folder hints and clean query
        search_folders = []
        search_query = query_lower
        
        for folder, path in folder_map.items():
            if folder in query_lower:
                search_folders.append(os.path.expanduser(path))
                # Remove folder mention from search query
                search_query = search_query.replace(f" in {folder}", "")
                search_query = search_query.replace(f" in the {folder}", "")
                search_query = search_query.replace(f" or {folder}", "")
                search_query = search_query.replace(f"{folder} folder", "")
                search_query = search_query.replace(folder, "")
        
        # Clean up query - remove common filler words but keep key terms
        filler_words = [" folder", " file", " named", " called", " with name", " in the"]
        for filler in filler_words:
            search_query = search_query.replace(filler, " ")
        search_query = " ".join(search_query.split()).strip()
        
        # Default to common folders if no hints, always include Downloads and Documents
        if not search_folders:
            search_folders = [
                os.path.expanduser("~/Downloads"),
                os.path.expanduser("~/Documents"),
                os.path.expanduser("~"),
            ]
        else:
            # Always add Downloads and Documents to search
            for path in ["~/Downloads", "~/Documents"]:
                expanded = os.path.expanduser(path)
                if expanded not in search_folders:
                    search_folders.append(expanded)
        
        # Search all folders
        all_matches = []
        for folder in search_folders:
            if os.path.isdir(folder):
                matches = self.resolve_file(
                    search_query,
                    search_path=folder,
                    recursive=True,
                    limit=limit,
                )
                all_matches.extend(matches)
        
        # Deduplicate and sort
        seen = set()
        unique = []
        for m in sorted(all_matches, key=lambda x: -x.score):
            if m.path not in seen:
                seen.add(m.path)
                unique.append(m)
        
        return unique[:limit]
    
    def find_and_open(self, query: str, confirm: bool = True) -> str:
        """
        Find best matching file and open it.
        
        Args:
            query: Search query
            confirm: Whether to ask for confirmation
            
        Returns:
            Status message
        """
        matches = self.resolve_file(query)
        
        if not matches:
            return f"No files found matching: {query}"
        
        best = matches[0]
        
        if best.score < 70 and confirm:
            return f"Did you mean: {best.name}? (score: {best.score})"
        
        return self.open_path(best.path)
    
    # ========================================
    # APPLICATION CONTROL
    # ========================================
    
    def get_installed_apps(self) -> Dict[str, str]:
        """
        Auto-detect installed applications.
        
        Returns: {app_name: path_or_command}
        """
        apps = {}
        
        if self._system == "Linux":
            # Check common locations
            app_dirs = [
                "/usr/share/applications",
                os.path.expanduser("~/.local/share/applications"),
            ]
            
            for app_dir in app_dirs:
                if not os.path.isdir(app_dir):
                    continue
                    
                for f in os.listdir(app_dir):
                    if f.endswith(".desktop"):
                        name = f.replace(".desktop", "").replace("-", " ").title()
                        # Simple name extraction
                        apps[name.lower()] = f.replace(".desktop", "")
            
            # Common commands
            common = [
                "firefox", "chrome", "chromium", "code", "nautilus",
                "terminal", "gnome-terminal", "konsole", "spotify",
                "vlc", "gimp", "inkscape", "libreoffice", "thunderbird",
            ]
            for cmd in common:
                if self._command_exists(cmd):
                    apps[cmd] = cmd
        
        elif self._system == "Darwin":
            # Mac - check /Applications
            app_dir = "/Applications"
            if os.path.isdir(app_dir):
                for f in os.listdir(app_dir):
                    if f.endswith(".app"):
                        name = f.replace(".app", "").lower()
                        apps[name] = os.path.join(app_dir, f)
        
        elif self._system == "Windows":
            # Windows - common apps
            common = [
                "notepad", "calc", "mspaint", "explorer",
                "chrome", "firefox", "code",
            ]
            for cmd in common:
                apps[cmd] = cmd
        
        return apps
    
    def _command_exists(self, cmd: str) -> bool:
        """Check if command exists in PATH."""
        try:
            subprocess.run(
                ["which", cmd],
                capture_output=True,
                check=True,
            )
            return True
        except:
            return False
    
    def open_application(self, app_name: str) -> str:
        """
        Open an application by name.
        
        Uses fuzzy matching to find best match.
        """
        apps = self.get_installed_apps()
        
        if not apps:
            return "Error: No applications detected"
        
        # Fuzzy match
        app_name_lower = app_name.lower()
        
        # Exact match
        if app_name_lower in apps:
            return self._launch_app(apps[app_name_lower])
        
        # Fuzzy match
        if fuzz:
            matches = process.extractBests(
                app_name_lower,
                list(apps.keys()),
                score_cutoff=60,
                limit=1,
            )
            if matches:
                matched_name = matches[0][0]
                return self._launch_app(apps[matched_name])
        
        return f"Application not found: {app_name}"
    
    def _launch_app(self, app: str) -> str:
        """Launch an application."""
        try:
            if self._system == "Darwin" and app.endswith(".app"):
                subprocess.Popen(["open", app])
            elif self._system == "Windows":
                subprocess.Popen(["start", app], shell=True)
            else:
                subprocess.Popen([app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            return f"Opened: {app}"
        except Exception as e:
            return f"Error launching {app}: {e}"
    
    def close_application(self, app_name: str) -> str:
        """
        Close an application (best effort).
        
        Note: This uses pkill on Linux/Mac, taskkill on Windows.
        """
        try:
            if self._system == "Windows":
                subprocess.run(["taskkill", "/IM", f"{app_name}.exe"], capture_output=True)
            else:
                subprocess.run(["pkill", "-f", app_name], capture_output=True)
            return f"Closed: {app_name}"
        except Exception as e:
            return f"Error closing {app_name}: {e}"
    
    # ========================================
    # BROWSER / WEB TOOLS
    # ========================================
    
    def open_url(self, url: str, browser: str = "google-chrome") -> str:
        """
        Open a URL in browser.
        
        Args:
            url: URL to open (with or without https://)
            browser: Browser to use (default: google-chrome)
        """
        # Add https if missing
        if not url.startswith("http"):
            url = f"https://{url}"
        
        try:
            if self._system == "Windows":
                subprocess.Popen(["start", browser, url], shell=True)
            elif self._system == "Darwin":
                subprocess.Popen(["open", "-a", browser, url])
            else:
                subprocess.Popen([browser, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            return f"Opened: {url}"
        except Exception as e:
            # Fallback to default browser
            try:
                subprocess.Popen(["xdg-open", url])
                return f"Opened: {url}"
            except:
                return f"Error opening URL: {e}"
    
    def open_website(self, site_name: str) -> str:
        """
        Open a common website by name.
        """
        # Common site mappings
        sites = {
            "youtube": "youtube.com",
            "youtube music": "music.youtube.com",
            "google": "google.com",
            "gmail": "mail.google.com",
            "github": "github.com",
            "twitter": "twitter.com",
            "x": "x.com",
            "facebook": "facebook.com",
            "instagram": "instagram.com",
            "linkedin": "linkedin.com",
            "reddit": "reddit.com",
            "netflix": "netflix.com",
            "spotify": "open.spotify.com",
            "amazon": "amazon.com",
            "whatsapp": "web.whatsapp.com",
            "chatgpt": "chat.openai.com",
            "google drive": "drive.google.com",
            "google docs": "docs.google.com",
            "google maps": "maps.google.com",
        }
        
        site_lower = site_name.lower().strip()
        
        # Exact match
        if site_lower in sites:
            return self.open_url(sites[site_lower])
        
        # Fuzzy match
        if fuzz:
            matches = process.extractBests(site_lower, list(sites.keys()), score_cutoff=70, limit=1)
            if matches:
                matched = matches[0][0]
                return self.open_url(sites[matched])
        
        # If looks like a URL, open directly
        if "." in site_name:
            return self.open_url(site_name)
        
        # Search Google
        return self.open_url(f"google.com/search?q={site_name.replace(' ', '+')}")
    
    def play_on_youtube_music(self, query: str) -> str:
        """
        Play music on YouTube Music.
        
        Opens YouTube Music with a search query.
        """
        search_query = query.replace(" ", "+")
        url = f"https://music.youtube.com/search?q={search_query}"
        return self.open_url(url)
    
    def search_youtube(self, query: str) -> str:
        """
        Search on YouTube.
        """
        search_query = query.replace(" ", "+")
        url = f"https://youtube.com/results?search_query={search_query}"
        return self.open_url(url)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

_tools = None

def get_os_tools() -> OSTools:
    """Get singleton OSTools instance."""
    global _tools
    if _tools is None:
        _tools = OSTools()
    return _tools


def open_file(path: str) -> str:
    return get_os_tools().open_file(path)


def open_folder(path: str) -> str:
    return get_os_tools().open_folder(path)


def open_app(app_name: str) -> str:
    return get_os_tools().open_application(app_name)


def list_dir(path: str = None) -> List[Dict]:
    return get_os_tools().list_directory(path)


def find_file(query: str) -> List[FileMatch]:
    return get_os_tools().resolve_file(query)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EVA OS Tools")
    subparsers = parser.add_subparsers(dest="command")
    
    # open
    open_p = subparsers.add_parser("open")
    open_p.add_argument("path")
    
    # list
    list_p = subparsers.add_parser("list")
    list_p.add_argument("path", nargs="?", default=".")
    
    # find
    find_p = subparsers.add_parser("find")
    find_p.add_argument("query")
    
    # apps
    apps_p = subparsers.add_parser("apps")
    
    # launch
    launch_p = subparsers.add_parser("launch")
    launch_p.add_argument("app")
    
    args = parser.parse_args()
    tools = OSTools()
    
    if args.command == "open":
        print(tools.open_path(args.path))
    elif args.command == "list":
        for item in tools.list_directory(args.path):
            icon = "📁" if item["type"] == "dir" else "📄"
            print(f"{icon} {item['name']}")
    elif args.command == "find":
        matches = tools.smart_find(args.query)
        if not matches:
            print(f"No files found matching: {args.query}")
        else:
            print(f"Found {len(matches)} matches:")
            for m in matches:
                print(f"  [{m.score}] {m.path}")
    elif args.command == "apps":
        for name, path in sorted(tools.get_installed_apps().items()):
            print(f"  {name}: {path}")
    elif args.command == "launch":
        print(tools.open_application(args.app))
    else:
        parser.print_help()
