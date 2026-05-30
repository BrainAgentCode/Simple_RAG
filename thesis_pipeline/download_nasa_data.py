#!/usr/bin/env python3
"""
NASA 数据下载与 PDF 分块
运行: python thesis_pipeline/download_nasa_data.py
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("nasa_download.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
NASA_DATA = (PROJECT_ROOT / "data" / "nasa").resolve()

DEFAULT_LESSONS_OUTPUT = NASA_DATA / "lessons_learned"
DEFAULT_PDFS_OUTPUT = NASA_DATA / "pdfs" / "conference"
DEFAULT_MARKDOWN_OUTPUT = NASA_DATA / "markdown" / "conference"
DEFAULT_CHUNKS_OUTPUT = NASA_DATA / "chunks" / "section_chunks.json"
WEB_SCRAPING_DIR = SCRIPT_DIR / "nasa_web_scraping"
CHUNKING_DIR = SCRIPT_DIR / "section_chunking"


def print_banner():
    print("\n" + "=" * 60)
    print("       NASA Data Downloader & Processor")
    print("=" * 60 + "\n")
    print(f"数据根目录: {NASA_DATA}")
    print("浏览器: 统一使用 Chrome/Chromium (headless)\n")


def print_menu():
    print("What would you like to do?\n")
    print("  [1] NASA Lessons Learned (CSV)")
    print("  [2] NASA Conference Papers (PDFs)")
    print("  [3] Both")
    print("  [4] Process existing PDFs (chunking only)")
    print("  [5] Exit\n")


def get_user_choice(prompt: str, valid_choices: list) -> str:
    while True:
        choice = input(prompt).strip()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Please enter one of: {', '.join(valid_choices)}")


def get_directory_input(prompt: str, default: Path) -> Path:
    print(f"\n{prompt}")
    print(f"  Default: {default}")
    user_input = input("  Press Enter for default, or enter custom path: ").strip()
    return default if not user_input else Path(user_input)


def get_numeric_input(
    prompt: str, default: int, min_val: int = 1, max_val: int = 1_000_000
) -> int:
    while True:
        user_input = input(f"{prompt} (default: {default}): ").strip()
        if not user_input:
            return default
        try:
            value = int(user_input)
            if min_val <= value <= max_val:
                return value
            print(f"Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid number")


def get_download_scope(resource_name: str, default_count: int = 100) -> Optional[int]:
    """
    返回最大下载数量；None 表示下载全部可用记录。
    """
    print(f"\n{resource_name} 下载范围:")
    print("  [1] 部分下载（指定数量）")
    print("  [2] 全部下载（尽可能多，直到网站无更多结果）")
    choice = get_user_choice("请选择 [1-2]: ", ["1", "2"])
    if choice == "2":
        print(f"  -> 将下载全部可用的 {resource_name}")
        return None
    count = get_numeric_input("最大数量", default_count, 1, 1_000_000)
    print(f"  -> 最多下载 {count} 条")
    return count


def download_lessons_learned(
    output_dir: Path,
    start_year: int = 2000,
    end_year: int = None,
    max_workers: int = 2,
    max_lessons: Optional[int] = None,
    headless: bool = True,
):
    print("\nStarting NASA Lessons Learned download...")
    print("Resume: existing URLs in CSV will be skipped automatically.")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sys.path.insert(0, str(WEB_SCRAPING_DIR))
        from lessons_learned import NASALessonsLearned

        csv_path = output_dir / f"nasa_lessons_learned_{start_year}_{end_year if end_year else 'present'}.csv"
        scraper = NASALessonsLearned(
            max_workers=max_workers,
            start_year=start_year,
            end_year=end_year,
            csv_path=str(csv_path),
            max_lessons=max_lessons,
            headless=headless,
        )

        df = scraper.collect_all_lessons()
        print(f"\nSaved to: {csv_path}")
        print(f"Total lessons in CSV: {len(df)}")
        return True
    except Exception as e:
        logger.error(f"Lessons learned download failed: {e}")
        print(f"\nError: {e}")
        print(
            "\n若提示 Chrome 无法启动，请在服务器安装:\n"
            "  sudo apt update && sudo apt install -y chromium-browser chromium-driver\n"
            "或设置 CHROME_BIN=/usr/bin/chromium"
        )
        return False


def download_technical_documents(
    output_dir: Path,
    max_docs: Optional[int] = 100,
    headless: bool = True,
    max_workers: int = 4,
):
    print("\nStarting NASA Conference Papers download...")
    print("Resume: existing PDF files in output dir will be skipped automatically.")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sys.path.insert(0, str(WEB_SCRAPING_DIR))
        from technical_documents import PDFDownloader

        nasa_url = "https://ntrs.nasa.gov/search?stiTypeDetails=Conference%20Paper"
        downloader = PDFDownloader(
            base_url=nasa_url,
            output_dir=str(output_dir),
            max_docs=max_docs,
            headless=headless,
            max_workers=max_workers,
        )
        num_downloaded = downloader.download_pdfs()
        print(f"\nSaved to: {output_dir}")
        print(f"Total PDFs in folder: {num_downloaded}")
        return True
    except Exception as e:
        logger.error(f"PDF download failed: {e}")
        print(f"\nError: {e}")
        print(
            "\n若提示 Chrome 无法启动，请在服务器安装:\n"
            "  sudo apt update && sudo apt install -y chromium-browser chromium-driver\n"
            "或设置 CHROME_BIN=/usr/bin/chromium"
        )
        return False


def process_pdfs_to_chunks(input_dir: Path, output_file: Path, timeout: int = 60):
    print("\nStarting PDF chunking...")
    print("Pipeline: PDF -> cached Markdown -> section chunks JSON")
    print("Resume: existing Markdown / per-PDF chunk cache will be reused when up to date.")
    if not input_dir.exists():
        print(f"\nInput directory does not exist: {input_dir}")
        return False

    output_file = Path(str(output_file))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    markdown_dir = DEFAULT_MARKDOWN_OUTPUT
    markdown_dir.mkdir(parents=True, exist_ok=True)

    try:
        sys.path.insert(0, str(CHUNKING_DIR))
        from section_based_chunking import SectionChunker

        log_candidates = [
            PROJECT_ROOT / "nasa_download.log",
            SCRIPT_DIR / "nasa_download.log",
        ]
        log_file = next((p for p in log_candidates if p.exists()), None)

        chunker = SectionChunker(
            input_dir=str(input_dir),
            output_file=str(output_file),
            markdown_dir=str(markdown_dir),
            timeout_seconds=timeout,
            log_file=str(log_file) if log_file else None,
        )
        chunker.process_pdfs()
        print(f"\nMarkdown cache: {markdown_dir}")
        print(f"Chunks saved to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        print(f"\nError: {e}")
        return False


def run_lessons_learned_flow():
    from datetime import datetime

    output_dir = get_directory_input("CSV output directory:", DEFAULT_LESSONS_OUTPUT)
    start_year = get_numeric_input("Start year", 2000, 1990, datetime.now().year)
    end_year = get_numeric_input("End year", datetime.now().year, start_year, datetime.now().year)
    max_lessons = get_download_scope("Lessons Learned", default_count=50)
    max_workers = get_numeric_input(
        "Parallel Chrome workers (建议 1-2，避免占满内存)", 2, 1, 64
    )

    confirm = input("Continue? [Y/n]: ").strip().lower()
    if confirm in ("", "y", "yes"):
        return download_lessons_learned(
            output_dir,
            start_year,
            end_year,
            max_workers=max_workers,
            max_lessons=max_lessons,
        )
    return False


def run_technical_documents_flow():
    output_dir = get_directory_input("PDF output directory:", DEFAULT_PDFS_OUTPUT)
    max_docs = get_download_scope("Conference Papers (PDF)", default_count=100)
    max_workers = get_numeric_input(
        "Parallel download workers (建议 3-6，过高可能触发站点限流)", 4, 1, 64
    )

    confirm = input("Continue? [Y/n]: ").strip().lower()
    if confirm in ("", "y", "yes"):
        if download_technical_documents(
            output_dir, max_docs=max_docs, max_workers=max_workers
        ):
            return output_dir
    return None


def run_chunking_flow(suggested_input_dir: Path = None):
    default_input = suggested_input_dir or DEFAULT_PDFS_OUTPUT
    input_dir = get_directory_input("PDF input directory:", default_input)

    if input_dir.exists():
        pdf_count = len([f for f in os.listdir(input_dir) if f.endswith(".pdf")])
        print(f"\nFound {pdf_count} PDFs in {input_dir}")
        if pdf_count == 0:
            return False
    else:
        print(f"\nDirectory does not exist: {input_dir}")
        return False

    output_file = get_directory_input("Chunks JSON output:", DEFAULT_CHUNKS_OUTPUT)
    timeout = get_numeric_input("Timeout per PDF (seconds)", 60, 10, 300)

    confirm = input("\nStart chunking? [Y/n]: ").strip().lower()
    if confirm in ("", "y", "yes"):
        return process_pdfs_to_chunks(input_dir, Path(str(output_file)), timeout=timeout)
    return False


def main():
    NASA_DATA.mkdir(parents=True, exist_ok=True)
    print_banner()

    while True:
        print_menu()
        choice = get_user_choice("Enter choice [1-5]: ", ["1", "2", "3", "4", "5"])

        if choice == "1":
            run_lessons_learned_flow()
        elif choice == "2":
            pdf_dir = run_technical_documents_flow()
            if pdf_dir:
                if input("\nChunk these PDFs now? [Y/n]: ").strip().lower() in ("", "y", "yes"):
                    run_chunking_flow(suggested_input_dir=pdf_dir)
        elif choice == "3":
            run_lessons_learned_flow()
            pdf_dir = run_technical_documents_flow()
            if pdf_dir:
                if input("\nChunk PDFs now? [Y/n]: ").strip().lower() in ("", "y", "yes"):
                    run_chunking_flow(suggested_input_dir=pdf_dir)
        elif choice == "4":
            run_chunking_flow()
        elif choice == "5":
            print("\nBye.\n")
            sys.exit(0)

        if input("\nReturn to menu? [Y/n]: ").strip().lower() not in ("", "y", "yes"):
            break


if __name__ == "__main__":
    main()
