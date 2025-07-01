import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import csv
import json
import os
from collections import deque


class RSQKitScraper:
    def __init__(self, base_url="https://everse.software/RSQKit/", delay=1):
        self.base_url = base_url
        self.delay = delay
        self.visited_urls = set()
        self.scraped_data = []
        self.session = requests.Session()

        # Headers to avoid being blocked
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def is_valid_url(self, url):
        """Check if URL belongs to the RSQKit domain"""
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc == "everse.software" and "/RSQKit/" in parsed.path

    def get_page_content(self, url):
        """Fetch and parse a single page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, "html.parser")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def extract_page_data(self, soup, url):
        """Extract relevant data from a page"""
        data = {
            "url": url,
            "title": "",
            "content": "",
            "links": [],
            "headings": [],
            "meta_description": "",
            "categories": [],
        }

        # Title
        title_tag = soup.find("title")
        if title_tag:
            data["title"] = title_tag.get_text().strip()

        # Meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            data["meta_description"] = meta_desc.get("content", "")

        # Main content - adjust selectors based on site structure
        content_selectors = [
            "main",
            "article",
            ".content",
            "#content",
            ".main-content",
            ".page-content",
        ]

        content_element = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break

        if not content_element:
            content_element = soup.find("body")

        if content_element:
            # Remove script and style elements
            for script in content_element(["script", "style"]):
                script.decompose()

            data["content"] = content_element.get_text(separator=" ", strip=True)

        # Headings
        for i in range(1, 7):
            headings = soup.find_all(f"h{i}")
            for heading in headings:
                data["headings"].append(
                    {"level": i, "text": heading.get_text().strip()}
                )

        # Links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urllib.parse.urljoin(url, href)
            if self.is_valid_url(full_url):
                data["links"].append({"text": link.get_text().strip(), "url": full_url})

        # Categories or tags (adjust based on site structure)
        category_selectors = [
            ".category",
            ".tag",
            ".breadcrumb",
            '[class*="category"]',
            '[class*="tag"]',
            ".taxonomy",
        ]

        for selector in category_selectors:
            categories = soup.select(selector)
            for cat in categories:
                text = cat.get_text().strip()
                if text and text not in data["categories"]:
                    data["categories"].append(text)

        return data

    def scrape_site(self, max_pages=None):
        """Scrape the entire site using BFS"""
        queue = deque([self.base_url])
        pages_scraped = 0

        while queue and (max_pages is None or pages_scraped < max_pages):
            current_url = queue.popleft()

            if current_url in self.visited_urls:
                continue

            print(f"Scraping: {current_url}")
            self.visited_urls.add(current_url)

            soup = self.get_page_content(current_url)
            if not soup:
                continue

            # Extract data
            page_data = self.extract_page_data(soup, current_url)
            self.scraped_data.append(page_data)

            # Add new URLs to queue
            for link_data in page_data["links"]:
                new_url = link_data["url"]
                if new_url not in self.visited_urls:
                    queue.append(new_url)

            pages_scraped += 1
            print(f"Scraped {pages_scraped} pages so far...")

            # Rate limiting
            time.sleep(self.delay)

        print(f"Scraping completed! Total pages: {len(self.scraped_data)}")
        return self.scraped_data

    def save_to_json(self, filename="rsqkit_data.json"):
        """Save scraped data to JSON file"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filename}")

    def save_to_csv(self, filename="rsqkit_data.csv"):
        """Save scraped data to CSV file"""
        if not self.scraped_data:
            print("No data to save")
            return

        fieldnames = [
            "url",
            "title",
            "meta_description",
            "content",
            "headings",
            "categories",
        ]

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for item in self.scraped_data:
                row = {
                    "url": item["url"],
                    "title": item["title"],
                    "meta_description": item["meta_description"],
                    "content": (
                        item["content"][:1000] + "..."
                        if len(item["content"]) > 1000
                        else item["content"]
                    ),
                    "headings": " | ".join([h["text"] for h in item["headings"]]),
                    "categories": " | ".join(item["categories"]),
                }
                writer.writerow(row)

        print(f"Data saved to {filename}")

    def get_site_structure(self):
        """Get a summary of the site structure"""
        structure = {
            "total_pages": len(self.scraped_data),
            "unique_categories": set(),
            "page_types": {},
            "url_patterns": {},
        }

        for page in self.scraped_data:
            # Collect categories
            structure["unique_categories"].update(page["categories"])

            # Analyze URL patterns
            path = urllib.parse.urlparse(page["url"]).path
            path_parts = [p for p in path.split("/") if p]
            if len(path_parts) > 1:
                pattern = "/".join(path_parts[:-1])
                structure["url_patterns"][pattern] = (
                    structure["url_patterns"].get(pattern, 0) + 1
                )

        structure["unique_categories"] = list(structure["unique_categories"])
        return structure


def main():
    # Initialize scraper
    scraper = RSQKitScraper(delay=2)  # 2 second delay between requests

    # Scrape the site (set max_pages for testing)
    data = scraper.scrape_site(max_pages=50)  # Remove max_pages to scrape everything

    # Save data
    scraper.save_to_json()
    scraper.save_to_csv()

    # Print site structure
    structure = scraper.get_site_structure()
    print("\nSite Structure:")
    print(f"Total pages scraped: {structure['total_pages']}")
    print(f"Categories found: {structure['unique_categories']}")
    print(f"URL patterns: {structure['url_patterns']}")


if __name__ == "__main__":
    main()
