import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re
from tqdm import tqdm
import pymupdf4llm
import datetime
import time
import signal
from contextlib import contextmanager
import multiprocessing
from functools import partial
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_modules.document_summary import (  # noqa: E402
    extract_summary_from_markdown,
    extract_summary_from_text,
    resolve_document_summary,
)
import fitz

# Found challenges:
# 1. The two chunks are created if Main section (1) and subsection (1.1) are both present.
# 2. Script has issues handling double columns, sometimes created chunk when it should not.
# 3. If section contains symbols it does not handle it well.
# 4. PDF's can have many spaces between words, this is sometimes created as a new chunk.
# 5. If a section is empty it will be created as a chunk.
# 6. References ruins everything, references may contain different bolding, spacing and so on, a lot of chunks created there.
# 7. Page breaks are not handled well, creates new chunk on new page? Because of double /n/n ?

# Timeout exception class, used to avoid infinite loops.
class TimeoutException(Exception):
    pass


# For the @data and @context parts, following the pymupdf4llm approach found, not sure if they are necessary, but since the implementation had them, I added them.
# Some of them are not used, but nice to have as a reference. 
# Context manager for timeout
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

@dataclass
class TextProperties:
    """Store text properties from PDF"""
    text: str
    font_size: float
    is_bold: bool
    bbox: tuple  # Bounding box coordinates (x0, y0, x1, y1, is_italic)
    page_num: int  # Page number

@dataclass
class Section:
    """Represent a document section"""
    title: str
    content: str
    level: int  # Hierarchy level (1 for main sections, 2 for subsections, etc.)
    page_number: int


# A lot of functions in this class are no longer used, they were implemented when working with chunking, as I forgot to include download_url, some meta data and so on
# so instead of running the code from the beginning each time which took hours, I implemented additional functions to save time.
# these functions are currently kept for reference what has been done to not forget mentioning them in the thesis. 
class SectionChunker:
    """Extract sections from PDFs using pymupdf4llm for accurate header detection"""
    
    def __init__(self, 
                 input_dir: str = None,
                 output_file: str = None,
                 markdown_dir: str = None,
                 timeout_seconds: int = 60,
                 num_processes: int = None,
                 log_file: str = None): 
        _c8_root = Path(__file__).resolve().parents[2]
        _nasa_data = (_c8_root / "data" / "nasa").resolve()
        self.input_dir = input_dir or str(_nasa_data / "pdfs" / "conference")
        self.output_file = output_file or str(_nasa_data / "chunks" / "section_chunks.json")
        self.markdown_dir = markdown_dir or str(_nasa_data / "markdown" / "conference")
        self.chunk_cache_dir = str(_nasa_data / "chunks" / "per_pdf")
        self.timeout_seconds = timeout_seconds
        self.num_processes = num_processes if num_processes else max(1, multiprocessing.cpu_count() - 1)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("section_chunking.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create a temporary directory for intermediate results if it doesn't exist
        self.temp_dir = "temp_chunks"
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.markdown_dir, exist_ok=True)
        os.makedirs(self.chunk_cache_dir, exist_ok=True)
        
        # Load download URLs from log file (optional)
        self.download_urls = {}
        if log_file and os.path.exists(log_file):
            self.download_urls = self.extract_download_urls_from_log(log_file)
            self.logger.info(f"Loaded {len(self.download_urls)} download URLs from log file")
    
    @classmethod
    def _looks_like_section_heading(cls, line: str) -> bool:
        if not line:
            return False
        if re.match(r"^(?:\d+(?:\.\d+)*|[IVX]+\.)\s+[A-Z]", line):
            return True
        if line.isupper() and len(line.split()) <= 8 and len(line) > 8:
            return True
        return False

    def _extract_abstract_from_pdf(self, pdf_path: str) -> Tuple[str, str]:
        """从前几页 PDF 原文中抽取概要。"""
        try:
            with fitz.open(pdf_path) as pdf_doc:
                page_texts = [
                    pdf_doc[page_index].get_text("text")
                    for page_index in range(min(6, len(pdf_doc)))
                ]
                combined = "\n\n".join(page_texts)
                summary, kind = extract_summary_from_text(combined)
                if summary:
                    return summary, kind
                for text in page_texts:
                    summary, kind = extract_summary_from_text(text)
                    if summary:
                        return summary, kind
        except Exception as e:
            self.logger.warning(f"Failed to extract summary from {pdf_path}: {e}")
        return "", ""

    def _resolve_document_abstract(
        self,
        pdf_path: str,
        markdown_text: str,
        sections: List[Section],
    ) -> Tuple[str, str]:
        """按优先级抽取整篇文档概要：abstract → conclusions → introduction。"""
        strategies = (
            ("sections", lambda: resolve_document_summary(sections)),
            ("markdown", lambda: extract_summary_from_markdown(markdown_text)),
            ("pdf", lambda: self._extract_abstract_from_pdf(pdf_path)),
        )
        for source, extractor in strategies:
            summary, kind = extractor()
            if summary:
                self.logger.debug(
                    f"Extracted document summary via {source}/{kind} for "
                    f"{os.path.basename(pdf_path)} ({len(summary.split())} words)"
                )
                return summary, kind

        self.logger.warning(
            f"No document summary found for {os.path.basename(pdf_path)}; "
            "retrieval will use adjacent sections or full document as parent context."
        )
        return "", ""

    def _markdown_path_for_pdf(self, pdf_file: str) -> str:
        stem = Path(pdf_file).stem
        return os.path.join(self.markdown_dir, f"{stem}.md")

    def _load_or_create_markdown(self, pdf_path: str, pdf_file: str) -> Tuple[str, str]:
        """Load cached markdown for a PDF, converting and saving on first run."""
        md_path = self._markdown_path_for_pdf(pdf_file)
        pdf_mtime = os.path.getmtime(pdf_path)

        if os.path.exists(md_path):
            md_mtime = os.path.getmtime(md_path)
            if md_mtime >= pdf_mtime:
                self.logger.debug(f"Using cached markdown: {md_path}")
                with open(md_path, "r", encoding="utf-8") as f:
                    return f.read(), md_path
            self.logger.debug(f"PDF newer than markdown, regenerating: {pdf_file}")

        self.logger.debug(f"Converting PDF to markdown: {pdf_file}")
        markdown_text = pymupdf4llm.to_markdown(pdf_path)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        self.logger.debug(f"Saved markdown: {md_path}")
        return markdown_text, md_path

    def _chunk_cache_path(self, pdf_file: str) -> str:
        stem = Path(pdf_file).stem
        return os.path.join(self.chunk_cache_dir, f"{stem}_chunks.json")

    def _source_mtime(self, pdf_path: str, pdf_file: str) -> float:
        mtimes = [os.path.getmtime(pdf_path)]
        md_path = self._markdown_path_for_pdf(pdf_file)
        if os.path.exists(md_path):
            mtimes.append(os.path.getmtime(md_path))
        return max(mtimes)

    def _stats_from_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        stats = {
            "words": 0,
            "chars": 0,
            "min_words": float("inf"),
            "max_words": 0,
            "min_chars": float("inf"),
            "max_chars": 0,
            "chunks": 0,
            "status": "success",
        }
        for chunk in chunks:
            content = chunk.get("content", "")
            word_count = len(content.split())
            char_count = len(content)
            stats["words"] += word_count
            stats["chars"] += char_count
            stats["min_words"] = min(stats["min_words"], word_count)
            stats["max_words"] = max(stats["max_words"], word_count)
            stats["min_chars"] = min(stats["min_chars"], char_count)
            stats["max_chars"] = max(stats["max_chars"], char_count)
            stats["chunks"] += 1
        return stats

    def _load_cached_chunks_if_valid(
        self, pdf_path: str, pdf_file: str
    ) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        cache_path = self._chunk_cache_path(pdf_file)
        if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
            return None, None

        if os.path.getmtime(cache_path) < self._source_mtime(pdf_path, pdf_file):
            self.logger.debug(
                f"Chunk cache stale for {pdf_file}, will regenerate: {cache_path}"
            )
            return None, None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not isinstance(chunks, list) or not chunks:
                return None, None
            stats = self._stats_from_chunks(chunks)
            stats["status"] = "cached"
            self.logger.debug(f"Using cached chunks: {cache_path}")
            return chunks, stats
        except Exception as e:
            self.logger.warning(f"Invalid chunk cache {cache_path}: {e}")
            return None, None

    def _save_chunk_cache(self, pdf_file: str, file_chunks: List[Dict]) -> None:
        cache_path = self._chunk_cache_path(pdf_file)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(file_chunks, f, indent=2)

    def extract_sections_from_markdown(self, markdown_text: str, pdf_path: str) -> List[Section]:
        sections = []
        current_section = None
        current_content = []
        current_level = 0
        current_page = 0  # Will be updated when page markers are found
        
        # Track the main section that subsections belong to
        current_main_section = None
        current_main_level = 0
        
        # Split the markdown text into lines
        lines = markdown_text.split('\n')
        
        # Flag to track if we're processing a page break
        in_page_break = False
        # Flag to track if we're in a figure/table section
        in_figure_or_table = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for page markers (if any)
            page_match = re.search(r'<!-- Page (\d+) -->', line)
            if page_match:
                current_page = int(page_match.group(1)) - 1  # Convert to 0-based
                i += 1
                continue
            
            # Check for page break indicators (horizontal rules often indicate page breaks)
            if line.strip() == '-----':
                in_page_break = True
                i += 1
                continue
            
            # If we just saw a page break and this line is empty, skip it
            if in_page_break and not line.strip():
                in_page_break = False
                i += 1
                continue
            
            # Reset page break flag if we have content
            if line.strip():
                in_page_break = False
            
            # Enhanced table detection - check for table indicators
            if (re.search(r'(table|tbl\.)\s+\d', line.lower(), re.IGNORECASE) or 
                line.strip().startswith('|') or  # Markdown table row
                re.match(r'^\s*\+[-+]+\+\s*$', line) or  # ASCII table border
                re.match(r'^\s*\|[-|]+\|\s*$', line) or  # Markdown table separator
                (line.strip().count('|') >= 2 and len(line.strip()) - len(line.strip().replace('|', '')) >= 3) or  # Multiple pipe characters
                'Table' in line or 'TABLE' in line):
                
                in_figure_or_table = True
                i += 1
                continue
            
            # Check for figure/image indicators
            if (re.search(r'(figure|fig\.|image|chart|graph)', line.lower()) or 
                '**Figure' in line or 
                'Figure' in line or 
                'FIGURE' in line):
                
                in_figure_or_table = True
                i += 1
                continue
            
            # If we're in a figure/table section, check if we've exited it
            # We exit when we hit an empty line followed by text that doesn't look like a caption
            if in_figure_or_table:
                # If this is an empty line, check the next line to see if we're exiting the figure/table
                if not line.strip():
                    if i + 1 < len(lines) and lines[i + 1].strip():
                        next_line = lines[i + 1].strip()
                        # If next line doesn't look like a caption or continuation, exit figure/table mode
                        if (not re.search(r'(figure|fig\.|table|image|chart)', next_line.lower(), re.IGNORECASE) and
                            not next_line.startswith('|') and
                            not re.match(r'^\d+\.', next_line) and
                            not next_line.startswith('(')):
                            in_figure_or_table = False
                
                i += 1
                continue
            
            # Check if line is a header (# style)
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            # Check for Roman numeral headers (e.g., "I. INTRODUCTION", "II. Methods", "A. Subsection")
            roman_header_match = re.match(r'^([IVX]+\.)\s+([A-Z][A-Za-z\s\-]+)$', line.strip())
            
            # Check for letter-based subsection headers (e.g., "A. Overview", "B. Methods")
            letter_header_match = re.match(r'^([A-Z]\.)\s+([A-Z][A-Za-z\s\-]+)$', line.strip())
            
            # Check for numbered headers (e.g., "1. Introduction", "2.1 Background")
            numbered_header_match = re.match(r'^(\d+(?:\.\d+)?\.?)\s+([A-Z][A-Za-z\s\-]+)$', line.strip())
            
            # Check for bold headers at the start of a line (e.g., "**Introduction:**")
            bold_header_match = re.match(r'^\*\*([^*]+)\*\*:?\s*(.*)$', line)
            
            # Check for headers that end with a colon (e.g., "Introduction:")
            colon_header_match = re.match(r'^([A-Z][a-zA-Z\s]+):\s+(.+)$', line)
            
            # Handle Roman numeral headers (I., II., III., etc.)
            if roman_header_match or letter_header_match or numbered_header_match:
                # If we have a current section, save it
                if current_section is not None and current_content:
                    content = '\n'.join(current_content).strip()
                    if len(content.split()) >= 7:  # Skip sections with less than 7 words
                        sections.append(Section(
                            title=current_section,
                            content=content,
                            level=current_level,
                            page_number=current_page
                        ))
                
                # Determine which match we have and extract header info
                if roman_header_match:
                    header_marker = roman_header_match.group(1)
                    header_text = roman_header_match.group(2).strip()
                    # Roman numerals I-III are level 1, IV+ could be deeper
                    level = 1
                elif letter_header_match:
                    header_marker = letter_header_match.group(1)
                    header_text = letter_header_match.group(2).strip()
                    level = 2  # Letter subsections are level 2
                else:  # numbered_header_match
                    header_marker = numbered_header_match.group(1)
                    header_text = numbered_header_match.group(2).strip()
                    # Count dots to determine level (1. = level 1, 1.1 = level 2, etc.)
                    level = header_marker.count('.') if '.' in header_marker else 1
                
                # Include the marker in the title for clarity
                full_title = f"{header_marker} {header_text}"
                
                # Skip headers that appear to be figure/table captions
                if re.search(r'(figure|fig\.|table|image|chart)', header_text.lower()):
                    if current_section is not None:
                        current_content.append(line)
                    i += 1
                    continue
                
                # Update the current section
                current_section = full_title
                current_level = level
                current_content = []
                
                # If this is a main section (Roman numeral), update the main section tracker
                if roman_header_match:
                    current_main_section = full_title
                    current_main_level = level
                
                i += 1
                continue
            
            if header_match:
                # If we have a current section, save it
                if current_section is not None and current_content:
                    content = '\n'.join(current_content).strip()
                    if len(content.split()) >= 7:  # Skip sections with less than 7 words
                        sections.append(Section(
                            title=current_section,
                            content=content,
                            level=current_level,
                            page_number=current_page
                        ))
                
                # Start a new section
                header_markers = header_match.group(1)
                header_text = header_match.group(2).strip()
                level = len(header_markers)  # Number of # symbols
                
                # Skip headers that are just numbers (likely page numbers or artifacts)
                if re.match(r'^\d+$', header_text):
                    # Add this line to current content instead
                    if current_section is not None:
                        current_content.append(line)
                    i += 1
                    continue
                
                # Skip headers that appear to be figure/table captions
                if re.search(r'(figure|fig\.|table|image|chart)', header_text.lower()):
                    i += 1
                    continue
                
                # Update the current section
                current_section = header_text
                current_level = level
                current_content = []
                
                # If this is a main section, update the main section tracker
                if level <= 2:  # Assuming level 1-2 are main sections
                    current_main_section = header_text
                    current_main_level = level
            elif bold_header_match or colon_header_match:
                # If we have a current section, save it
                if current_section is not None and current_content:
                    content = '\n'.join(current_content).strip()
                    if len(content.split()) >= 7:  # Skip sections with less than 7 words
                        sections.append(Section(
                            title=current_section,
                            content=content,
                            level=current_level,
                            page_number=current_page
                        ))
                
                # Extract the header text
                if bold_header_match:
                    header_text = bold_header_match.group(1).strip()
                    remaining_text = bold_header_match.group(2).strip()
                else:  # colon_header_match
                    header_text = colon_header_match.group(1).strip()
                    remaining_text = colon_header_match.group(2).strip()
                
                # Skip headers that are just numbers or appear to be figure/table captions
                if re.match(r'^\d+$', header_text) or re.search(r'(figure|fig\.|table|image|chart)', header_text.lower()):
                    if current_section is not None:
                        current_content.append(line)
                    i += 1
                    continue
                
                # Update the current section
                current_section = header_text
                current_level = 2  # Treat these as level 2 headers (subsections)
                current_content = []
                
                # If there's remaining text on the same line, add it to the content
                if remaining_text:
                    current_content.append(remaining_text)
                
                # If this is a main section, update the main section tracker
                if not re.match(r'^[a-z]', header_text):  # If it starts with uppercase, treat as main section
                    current_main_section = header_text
                    current_main_level = current_level
            else:
                # Check for subsection markers in the content (italic or bold text at start of line)
                subsection_match = re.match(r'^_([^_]+)_\s*(.*)$', line) or re.match(r'^\*\*([^*]+)\*\*\s*(.*)$', line)
                
                if subsection_match and current_section is not None:
                    # Extract subsection title
                    subsection_title = subsection_match.group(1).strip()
                    
                    # Skip subsections that are just numbers or figure/table related
                    if re.match(r'^\d+$', subsection_title) or re.search(r'(figure|fig\.|table|image|chart)', subsection_title.lower()):
                        if current_section is not None:
                            current_content.append(line)
                        i += 1
                        continue
                    
                    # If we have content for the current section, save it before starting the subsection
                    if current_content:
                        content = '\n'.join(current_content).strip()
                        if len(content.split()) >= 7:  # Skip sections with less than 7 words
                            # Check if this is a continuation of a section across a page break
                            is_continuation = False
                            for s in sections:
                                if s.title == current_section and s.level == current_level:
                                    # This is a continuation - append to existing section
                                    s.content += "\n\n" + content
                                    is_continuation = True
                                    break
                            
                            if not is_continuation:
                                sections.append(Section(
                                    title=current_section,
                                    content=content,
                                    level=current_level,
                                    page_number=current_page
                                ))
                    
                    # Start collecting content for this subsection
                    subsection_content = []
                    i += 1  # Move past the subsection title
                    
                    # Collect content until we hit another subsection or main section
                    while i < len(lines):
                        next_line = lines[i]
                        
                        # Stop if we hit a main section header
                        if re.match(r'^(#{1,6})\s+(.+)$', next_line):
                            break
                        
                        # Stop if we hit another subsection
                        if re.match(r'^_([^_]+)_\s*$', next_line) or re.match(r'^\*\*([^*]+)\*\*\s*$', next_line):
                            break
                        
                        # Skip figure/table content
                        if re.search(r'(figure|fig\.|table|image|chart)', next_line.lower()) or '**Figure' in next_line or '**Table' in next_line:
                            in_figure_or_table = True
                            i += 1
                            continue
                        
                        if in_figure_or_table and next_line.strip() and not next_line.startswith('**'):
                            in_figure_or_table = False
                        
                        if not in_figure_or_table:
                            subsection_content.append(next_line)
                        
                        i += 1
                    
                    # Add the subsection as its own section with increased level
                    if subsection_content:
                        content = '\n'.join(subsection_content).strip()
                        if len(content.split()) >= 7:  # Skip sections with less than 7 words
                            full_title = f"{current_main_section} - {subsection_title}"
                            
                            # Check if this is a continuation of a subsection across a page break
                            is_continuation = False
                            for s in sections:
                                if s.title == full_title:
                                    # This is a continuation - append to existing section
                                    s.content += "\n\n" + content
                                    is_continuation = True
                                    break
                            
                            if not is_continuation:
                                sections.append(Section(
                                    title=full_title,
                                    content=content,
                                    level=current_main_level + 1,  # Increase level for subsection
                                    page_number=current_page
                                ))
                    
                    # Reset content collection for the main section
                    current_content = []
                    continue  # Skip the i += 1 at the end of the loop
                
                # Add line to current section content
                if current_section is not None and not in_figure_or_table:
                    current_content.append(line)
            
            i += 1
        
        # Add the last section
        if current_section is not None and current_content:
            content = '\n'.join(current_content).strip()
            if len(content.split()) >= 7:  # Skip sections with less than 7 words
                # Check if this is a continuation of a section across a page break
                is_continuation = False
                for s in sections:
                    if s.title == current_section and s.level == current_level:
                        # This is a continuation - append to existing section
                        s.content += "\n\n" + content
                        is_continuation = True
                        break
                
                if not is_continuation:
                    sections.append(Section(
                        title=current_section,
                        content=content,
                        level=current_level,
                        page_number=current_page
                    ))
        
        # If no sections were found, create a single section with the document title
        if not sections:
            doc_title = os.path.basename(pdf_path).replace('.pdf', '')
            # Filter out figure/table content from the full text
            filtered_lines = []
            in_figure_or_table = False
            for line in markdown_text.split('\n'):
                if re.search(r'(figure|fig\.|table|image|chart)', line.lower()) or '**Figure' in line or '**Table' in line:
                    in_figure_or_table = True
                    continue
                if in_figure_or_table and line.strip() and not line.startswith('**'):
                    in_figure_or_table = False
                if not in_figure_or_table:
                    filtered_lines.append(line)
            
            filtered_text = '\n'.join(filtered_lines)
            sections.append(Section(
                title="Document Title",
                content=filtered_text,
                level=0,
                page_number=0
            ))
        
        return sections
    
    def post_process_sections(self, sections: List[Section]) -> List[Section]:

        if not sections:
            return []
        
        # First pass: identify and merge sections that are continuations across page breaks
        processed_sections = []
        skip_indices = set()
        
        for i, section in enumerate(sections):
            if i in skip_indices:
                continue
            
            current_section = section
            
            # Check if there are any continuations of this section
            for j in range(i + 1, len(sections)):
                if j in skip_indices:
                    continue
                    
                next_section = sections[j]
                
                # Check if this is a continuation of the same section or subsection
                if current_section.title == next_section.title:
                    # Merge the content
                    current_section.content += "\n\n" + next_section.content
                    skip_indices.add(j)
                # Special case: if main section content appears between subsections
                elif " - " in current_section.title and " - " not in next_section.title:
                    parts = current_section.title.split(" - ")
                    if len(parts) >= 2 and parts[0] == next_section.title:
                        # This is main section content that should be merged with the next subsection
                        # We'll mark it to skip and handle it in the next pass
                        skip_indices.add(j)
            
            processed_sections.append(current_section)
        
        # Second pass: handle main section content that appears between subsections
        final_sections = []
        main_section_content = {}
        
        for section in processed_sections:
            if " - " in section.title:  # This is a subsection (note: space-hyphen-space)
                parts = section.title.split(" - ")
                if len(parts) >= 2:
                    main_title = parts[0]
                    subsection_title = parts[1]
                else:
                    # No proper separator found, treat as non-subsection
                    final_sections.append(section)
                    continue
                
                # If we have accumulated content for this main section, add it to the subsection
                if main_title in main_section_content:
                    # Add the main section content to the beginning of this subsection
                    section.content = main_section_content[main_title] + "\n\n" + section.content
                    del main_section_content[main_title]
                
                final_sections.append(section)
            else:  # This is a main section
                # Check if the next section is a subsection of this main section
                is_followed_by_subsection = False
                for next_section in processed_sections:
                    if next_section.title.startswith(section.title + " - "):
                        # Store this content to be merged with the subsection
                        main_section_content[section.title] = section.content
                        is_followed_by_subsection = True
                        break
                
                if not is_followed_by_subsection:
                    final_sections.append(section)
        
        # Third pass: handle references section specially
        reference_sections = []
        non_reference_sections = []
        
        for section in final_sections:
            if any(ref in section.title.lower() for ref in ['reference', 'bibliography', 'works cited']):
                reference_sections.append(section)
            else:
                non_reference_sections.append(section)
        
        if reference_sections:
            # Combine all reference sections into one
            reference_content = ""
            reference_title = reference_sections[0].title
            reference_page = reference_sections[0].page_number
            reference_level = reference_sections[0].level
            
            for section in reference_sections:
                reference_content += section.content + "\n\n"
                
            combined_reference = Section(
                title=reference_title,
                content=reference_content.strip(),
                level=reference_level,
                page_number=reference_page
            )
            
            return non_reference_sections + [combined_reference]
        else:
            return final_sections
    
    def verify_sections(self, sections: List[Section], pdf_path: str) -> None:
        self.logger.debug(
            f"{os.path.basename(pdf_path)}: {len(sections)} sections extracted"
        )
    
    def process_single_pdf(self, pdf_file: str) -> Tuple[List[Dict], List[str], Dict]:
        """Process a single PDF file and return its chunks and statistics"""
        pdf_path = os.path.join(self.input_dir, pdf_file)
        file_chunks = []
        stats = {
            "words": 0,
            "chars": 0,
            "min_words": float('inf'),
            "max_words": 0,
            "min_chars": float('inf'),
            "max_chars": 0,
            "chunks": 0,
            "status": "success"
        }
        error_info = None

        cached_chunks, cached_stats = self._load_cached_chunks_if_valid(pdf_path, pdf_file)
        if cached_chunks is not None and cached_stats is not None:
            return cached_chunks, [], cached_stats
        
        try:
            # Use timeout context manager
            with time_limit(self.timeout_seconds):
                start_time = time.time()
                markdown_text, markdown_path = self._load_or_create_markdown(
                    pdf_path, pdf_file
                )
                
                # Extract sections from markdown
                sections = self.extract_sections_from_markdown(markdown_text, pdf_path)
                
                # Post-process sections to merge small adjacent ones of the same level
                sections = self.post_process_sections(sections)
                abstract_text, summary_kind = self._resolve_document_abstract(
                    pdf_path, markdown_text, sections
                )
                
                # Verify sections
                self.verify_sections(sections, pdf_path)
                
                # Create chunks from sections
                for i, section in enumerate(sections):
                    # Skip empty sections
                    if not section.content.strip():
                        continue
                    
                    # Skip sections with very little content (likely just subsection headers)
                    if len(section.content.split()) < 7:
                        continue
                    
                    # Calculate statistics for this chunk
                    content = section.content
                    word_count = len(content.split())
                    char_count = len(content)
                    
                    # Update statistics
                    stats["words"] += word_count
                    stats["chars"] += char_count
                    stats["min_words"] = min(stats["min_words"], word_count)
                    stats["max_words"] = max(stats["max_words"], word_count)
                    stats["min_chars"] = min(stats["min_chars"], char_count)
                    stats["max_chars"] = max(stats["max_chars"], char_count)
                    stats["chunks"] += 1
                    
                    chunk = {
                        'chunk_id': f"{pdf_file}_{i}",
                        'title': section.title,
                        'content': section.content,
                        'page_number': section.page_number,
                        'section_level': section.level,
                        'section_number': i + 1,
                        'total_sections': len(sections),
                        'abstract': abstract_text,
                        'summary_kind': summary_kind,
                        'metadata': {
                            'file_name': pdf_file,
                            'file_path': pdf_path,
                            'markdown_path': markdown_path,
                            'download_url': self.download_urls.get(pdf_file),
                            'abstract': abstract_text,
                            'summary_kind': summary_kind,
                            'statistics': {
                                'word_count': word_count,
                                'character_count': char_count,
                                'average_word_length': char_count / word_count if word_count > 0 else 0,
                                'processing_time_seconds': time.time() - start_time
                            }
                        }
                    }
                    file_chunks.append(chunk)
                
                self._save_chunk_cache(pdf_file, file_chunks)
                # Save individual file results
                with open(os.path.join(self.temp_dir, f"{pdf_file.replace('.pdf', '')}_chunks.json"), 'w') as f:
                    json.dump(file_chunks, f, indent=2)
        
        except TimeoutException:
            error_info = f"Timeout: exceeded {self.timeout_seconds} seconds"
            stats["status"] = "timeout"
            with open(os.path.join(self.temp_dir, f"{pdf_file.replace('.pdf', '')}_timeout.txt"), 'w') as f:
                f.write(f"Skipped due to timeout (exceeded {self.timeout_seconds} seconds)")
        
        except Exception as e:
            error_info = f"Error: {str(e)}"
            stats["status"] = "error"
            with open(os.path.join(self.temp_dir, f"{pdf_file.replace('.pdf', '')}_error.txt"), 'w') as f:
                f.write(f"Error: {str(e)}")
        
        return file_chunks, [error_info] if error_info else [], stats
    
    def process_pdfs(self) -> None:
        """
        Process all PDFs in the input directory using multiprocessing.
        """
        pdf_files = [f for f in os.listdir(self.input_dir) if f.endswith('.pdf')]
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        self.logger.info(f"Using {self.num_processes} processes for parallel processing")
        
        # Statistics tracking
        all_chunks = []
        skipped_files = []
        cached_files = 0
        total_stats = {
            "total_words": 0,
            "total_chars": 0,
            "min_words": float('inf'),
            "max_words": 0,
            "min_chars": float('inf'),
            "max_chars": 0,
            "total_chunks": 0
        }
        
        # Create a pool of workers
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            # Process files in parallel and show progress with tqdm
            results = list(tqdm(
                pool.imap(self.process_single_pdf, pdf_files),
                total=len(pdf_files),
                desc="Processing PDFs"
            ))
        
        # Collect results
        for i, (chunks, errors, stats) in enumerate(results):
            # Add chunks to the overall list
            all_chunks.extend(chunks)
            
            # Update statistics
            if stats["status"] in ("success", "cached"):
                if stats["status"] == "cached":
                    cached_files += 1
                total_stats["total_words"] += stats["words"]
                total_stats["total_chars"] += stats["chars"]
                total_stats["min_words"] = min(total_stats["min_words"], stats["min_words"]) if stats["min_words"] != float('inf') else total_stats["min_words"]
                total_stats["max_words"] = max(total_stats["max_words"], stats["max_words"])
                total_stats["min_chars"] = min(total_stats["min_chars"], stats["min_chars"]) if stats["min_chars"] != float('inf') else total_stats["min_chars"]
                total_stats["max_chars"] = max(total_stats["max_chars"], stats["max_chars"])
                total_stats["total_chunks"] += stats["chunks"]
            else:
                skipped_files.append(pdf_files[i])
            
            # Save intermediate results periodically
            if (i + 1) % 10 == 0 or i == len(results) - 1:
                self.logger.info(f"Saving intermediate results after processing {i + 1}/{len(pdf_files)} files")
                with open(f"{self.output_file}.partial", 'w') as f:
                    json.dump(all_chunks, f, indent=2)
                
                # Also save current statistics
                current_stats = {
                    "total_pdfs_processed": i + 1,
                    "total_chunks": total_stats["total_chunks"],
                    "skipped_files": len(skipped_files),
                    "skipped_file_list": skipped_files,
                    "average_chunks_per_pdf": total_stats["total_chunks"]/(i + 1 - len(skipped_files)) if (i + 1 - len(skipped_files)) > 0 else 0,
                    "average_words_per_chunk": total_stats["total_words"] / total_stats["total_chunks"] if total_stats["total_chunks"] > 0 else 0,
                    "average_chars_per_chunk": total_stats["total_chars"] / total_stats["total_chunks"] if total_stats["total_chunks"] > 0 else 0,
                    "min_words": total_stats["min_words"] if total_stats["min_words"] != float('inf') else 0,
                    "max_words": total_stats["max_words"],
                    "min_chars": total_stats["min_chars"] if total_stats["min_chars"] != float('inf') else 0,
                    "max_chars": total_stats["max_chars"],
                    "timestamp": datetime.datetime.now().isoformat(),
                    "progress_percentage": ((i + 1) / len(pdf_files)) * 100
                }
                with open("chunk_statistics.partial.json", 'w') as f:
                    json.dump(current_stats, f, indent=2)
        
        # Save final results
        with open(self.output_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        
        # Calculate averages
        avg_words = total_stats["total_words"] / total_stats["total_chunks"] if total_stats["total_chunks"] > 0 else 0
        avg_chars = total_stats["total_chars"] / total_stats["total_chunks"] if total_stats["total_chunks"] > 0 else 0
        
        # Log overall statistics
        self.logger.info(f"\nProcessing complete:")
        self.logger.info(f"Total PDFs processed: {len(pdf_files) - len(skipped_files)}")
        self.logger.info(f"Total PDFs loaded from cache: {cached_files}")
        self.logger.info(f"Total PDFs skipped (errors/timeouts): {len(skipped_files)}")
        self.logger.info(f"Total chunks created: {total_stats['total_chunks']}")
        if pdf_files and len(pdf_files) > len(skipped_files):
            self.logger.info(f"Average chunks per PDF: {total_stats['total_chunks']/(len(pdf_files) - len(skipped_files)):.1f}")
        
        self.logger.info(f"\nChunk Statistics:")
        self.logger.info(f"Average words per chunk: {avg_words:.1f}")
        self.logger.info(f"Average characters per chunk: {avg_chars:.1f}")
        self.logger.info(f"Word count range: {total_stats['min_words']} to {total_stats['max_words']}")
        self.logger.info(f"Character count range: {total_stats['min_chars']} to {total_stats['max_chars']}")
        
        # Save overall statistics to a separate JSON file
        statistics = {
            "total_pdfs": len(pdf_files),
            "total_pdfs_processed": len(pdf_files) - len(skipped_files),
            "total_pdfs_skipped": len(skipped_files),
            "skipped_file_list": skipped_files,
            "total_chunks": total_stats["total_chunks"],
            "average_chunks_per_pdf": total_stats["total_chunks"]/(len(pdf_files) - len(skipped_files)) if (len(pdf_files) - len(skipped_files)) > 0 else 0,
            "average_words_per_chunk": avg_words,
            "average_chars_per_chunk": avg_chars,
            "min_words": total_stats["min_words"] if total_stats["min_words"] != float('inf') else 0,
            "max_words": total_stats["max_words"],
            "min_chars": total_stats["min_chars"] if total_stats["min_chars"] != float('inf') else 0,
            "max_chars": total_stats["max_chars"],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open("chunk_statistics.json", 'w') as f:
            json.dump(statistics, f, indent=2)

    def generate_download_url(self, filename: str) -> str:
        """Generate a direct download URL for the NASA Technical Reports Server based on filename."""
        # Base domain for NASA Technical Reports Server
        base_domain = "https://ntrs.nasa.gov"
        
        # Extract document ID from filename if possible
        doc_id = None
        
        # Try to extract numeric ID from filename
        numeric_match = re.search(r'(\d{8,})', filename)
        if numeric_match:
            doc_id = numeric_match.group(1)
        
        # If we found an ID, construct the direct download URL
        if doc_id:
            # Format matches the download URL pattern in PDFDownloader class
            return f"{base_domain}/api/citations/{doc_id}/downloads/{filename}"
        
        # If filename contains NASA identifiers like NASA-TM, try to use that
        nasa_id_match = re.search(r'(NASA-[A-Z]+-\d+)', filename, re.IGNORECASE)
        if nasa_id_match:
            nasa_id = nasa_id_match.group(1)
            return f"{base_domain}/api/citations/{nasa_id}/downloads/{filename}"
        
        # For other formats, try to construct a direct download URL based on the filename
        # This matches the format seen in the PDFDownloader where URLs contain /downloads/
        return f"{base_domain}/api/citations/downloads/{filename}"

    def extract_download_urls_from_log(self, log_file):
        """Extract download URLs from the log file."""
        download_urls = {}
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if "Downloading from:" in line:
                        url = line.split("Downloading from: ")[1].strip()
                        # Get the next line which should contain the filename
                        next_line = next(f, "")
                        if "Downloaded" in next_line and ":" in next_line:
                            # Extract filename from the log line
                            filename_part = next_line.split(":")[-1].strip()
                            download_urls[filename_part] = url
        except Exception as e:
            self.logger.error(f"Error extracting download URLs from log file: {e}")
            return {}
        
        self.logger.info(f"Extracted {len(download_urls)} download URLs from log file")
        return download_urls

if __name__ == "__main__":
    chunker = SectionChunker()
    chunker.process_pdfs()
