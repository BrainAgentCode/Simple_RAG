import os
import time
import logging
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from browser_utils import create_chrome_driver


class NASALessonsLearned:
    def __init__(
        self,
        max_workers: int = 2,
        start_year: int = 2000,
        end_year: int = None,
        csv_path: str = None,
        max_lessons: Optional[int] = None,
        headless: bool = True,
    ):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

        self.max_workers = max(1, min(max_workers, 4))
        self.start_year = start_year
        self.end_year = end_year if end_year else datetime.now().year
        self.max_lessons = max_lessons
        self.headless = headless
        self.base_url = "https://llis.nasa.gov"
        self.date_ranges = self._build_date_ranges()

        self.logger.info(f"Will scrape lessons from {self.start_year} to {self.end_year}")
        if self.max_lessons:
            self.logger.info(f"Max lessons to download: {self.max_lessons}")
        else:
            self.logger.info("Downloading all lessons found in date range")

        if csv_path:
            self.csv_path = csv_path
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.csv_path = os.path.join(
                script_dir, f"nasa_lessons_learned_{self.start_year}_{self.end_year}.csv"
            )

        if not os.path.exists(self.csv_path):
            pd.DataFrame(
                columns=[
                    "url",
                    "subject",
                    "abstract",
                    "driving_event",
                    "lessons_learned",
                    "recommendations",
                    "evidence",
                    "program_relation",
                    "program_phase",
                    "mission_directorate",
                    "topics",
                    "date_range",
                ]
            ).to_csv(self.csv_path, index=False)

        self.driver = None
        self._init_list_driver()

    def _init_list_driver(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
        self.driver = create_chrome_driver(headless=self.headless)

    def _build_date_ranges(self) -> List[str]:
        date_ranges = []
        current_year = self.start_year

        if current_year <= 2003 and self.end_year >= 2000:
            if self.start_year <= 2003:
                date_ranges.append("2000-2003")
                current_year = 2004

        while current_year <= self.end_year:
            date_ranges.append(str(current_year))
            current_year += 1

        return date_ranges

    def get_lessons_urls(self, max_pages_per_range: int = 50) -> List[tuple]:
        all_lesson_urls = []
        seen = set()

        for date_range in self.date_ranges:
            self.logger.info(f"Processing date range: {date_range}")
            page = 1

            while page <= max_pages_per_range:
                if self.max_lessons and len(all_lesson_urls) >= self.max_lessons:
                    break

                try:
                    search_url = f"{self.base_url}/search?lesson_date={date_range}&page={page}"
                    self.logger.info(f"Collecting URLs from: {search_url}")
                    self.driver.get(search_url)
                    time.sleep(3)

                    lesson_elements = self.driver.find_elements(
                        By.CSS_SELECTOR, "a[href*='/lesson/']"
                    )
                    if not lesson_elements:
                        break

                    for elem in lesson_elements:
                        href = elem.get_attribute("href")
                        if href and href not in seen:
                            seen.add(href)
                            all_lesson_urls.append((href, date_range))
                            if self.max_lessons and len(all_lesson_urls) >= self.max_lessons:
                                break

                    self.logger.info(
                        f"Page {page} ({date_range}): total unique URLs {len(all_lesson_urls)}"
                    )

                    try:
                        pagination = self.driver.find_element(By.CSS_SELECTOR, ".pagination")
                        active_page = pagination.find_element(By.CSS_SELECTOR, ".active").text
                        page_numbers = [
                            el.text
                            for el in pagination.find_elements(
                                By.CSS_SELECTOR, "li:not(.prev):not(.next) a"
                            )
                        ]
                        if page_numbers and active_page == page_numbers[-1]:
                            break
                    except Exception:
                        pass

                    page += 1
                    time.sleep(1)
                except Exception as e:
                    self.logger.error(f"Error on page {page} for {date_range}: {e}")
                    page += 1

            if self.max_lessons and len(all_lesson_urls) >= self.max_lessons:
                break

        self.logger.info(f"Total lesson URLs collected: {len(all_lesson_urls)}")
        return all_lesson_urls

    def _get_text(self, soup: BeautifulSoup, field_name: str) -> str:
        try:
            for div in soup.find_all("div", class_="ember-view"):
                h3 = div.find("h3", string=lambda x: x and field_name in x)
                if h3:
                    content = []
                    for sibling in h3.next_siblings:
                        if sibling.name == "h3":
                            break
                        if hasattr(sibling, "stripped_strings"):
                            content.extend(sibling.stripped_strings)
                        elif hasattr(sibling, "string") and sibling.string:
                            content.append(sibling.string.strip())
                    return " ".join(filter(None, content))
            return "None"
        except Exception as e:
            self.logger.error(f"Error extracting {field_name}: {e}")
            return "None"

    def _get_subject(self, soup: BeautifulSoup) -> str:
        try:
            for div in soup.find_all("div", class_="ember-view"):
                h3 = div.find("h3", string="Subject")
                if h3:
                    em = div.find("em")
                    if em:
                        strong = em.find("strong")
                        if strong:
                            return strong.get_text(strip=True)
                    content = []
                    for sibling in h3.next_siblings:
                        if hasattr(sibling, "stripped_strings"):
                            content.extend(sibling.stripped_strings)
                    if content:
                        return " ".join(content)
            return "None"
        except Exception:
            return "None"

    def extract_lesson_data(self, url_tuple: tuple) -> Dict:
        url, date_range = url_tuple
        driver = None
        try:
            driver = create_chrome_driver(headless=self.headless)
            driver.get(url)
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "ember-view"))
                )
            except Exception:
                pass
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            return {
                "url": url,
                "subject": self._get_subject(soup),
                "abstract": self._get_text(soup, "Abstract"),
                "driving_event": self._get_text(soup, "Driving Event"),
                "lessons_learned": self._get_text(soup, "Lesson(s) Learned"),
                "recommendations": self._get_text(soup, "Recommendation(s)"),
                "evidence": self._get_text(
                    soup, "Evidence of Recurrence Control Effectiveness"
                ),
                "program_relation": self._get_text(soup, "Program Relation"),
                "program_phase": self._get_text(soup, "Program/Project Phase"),
                "mission_directorate": self._get_text(soup, "Mission Directorate(s)"),
                "topics": self._get_text(soup, "Topic(s)"),
                "date_range": date_range,
            }
        except Exception as e:
            self.logger.error(f"Error processing {url}: {e}")
            return {
                "url": url,
                "subject": "Error",
                "abstract": "Error",
                "driving_event": "Error",
                "lessons_learned": "Error",
                "recommendations": "Error",
                "evidence": "Error",
                "program_relation": "Error",
                "program_phase": "Error",
                "mission_directorate": "Error",
                "topics": "Error",
                "date_range": date_range,
            }
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    def save_to_csv(self, data: Dict):
        pd.DataFrame([data]).to_csv(
            self.csv_path, mode="a", header=False, index=False, encoding="utf-8"
        )

    def collect_all_lessons(self) -> pd.DataFrame:
        try:
            lesson_url_tuples = self.get_lessons_urls()
            existing_urls: set = set()
            try:
                existing_df = pd.read_csv(self.csv_path)
                existing_urls = {
                    str(url)
                    for url in existing_df["url"].tolist()
                    if pd.notna(url) and str(url) not in {"url", "Error"}
                }
                before = len(lesson_url_tuples)
                lesson_url_tuples = [
                    t for t in lesson_url_tuples if t[0] not in existing_urls
                ]
                skipped = before - len(lesson_url_tuples)
                self.logger.info(
                    f"Resume: {len(existing_urls)} lessons already in CSV, "
                    f"skipped {skipped}, {len(lesson_url_tuples)} remaining"
                )
            except Exception:
                self.logger.info(
                    f"No existing CSV entries to resume from, "
                    f"{len(lesson_url_tuples)} lessons to fetch"
                )

            if not lesson_url_tuples:
                self.logger.info("No new lessons to process")
                return pd.read_csv(self.csv_path)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = {
                    executor.submit(self.extract_lesson_data, t): t
                    for t in lesson_url_tuples
                }
                for i, future in enumerate(
                    concurrent.futures.as_completed(futures), 1
                ):
                    url_tuple = futures[future]
                    try:
                        data = future.result()
                        self.save_to_csv(data)
                        self.logger.info(
                            f"Processed {i}/{len(lesson_url_tuples)}: {url_tuple[0]}"
                        )
                    except Exception as exc:
                        self.logger.error(f"Error processing {url_tuple[0]}: {exc}")

            return pd.read_csv(self.csv_path)
        finally:
            self.close()

    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def __del__(self):
        self.close()
