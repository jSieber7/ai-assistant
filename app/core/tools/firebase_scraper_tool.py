"""
Firebase web scraping tool for the AI Assistant tool system.

This module provides a tool for scraping web content using Firebase Admin SDK
and storing results in Firebase Firestore with optional Selenium rendering.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
import firebase_admin
from firebase_admin import credentials, firestore, storage
from firebase_admin.exceptions import FirebaseError
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import httpx

from .base import BaseTool, ToolResult, ToolExecutionError

logger = logging.getLogger(__name__)


class FirebaseScraperTool(BaseTool):
    """Scrape web content using Firebase with advanced rendering and storage"""

    def __init__(self):
        super().__init__()
        self._firebase_initialized = False
        self._db = None
        self._storage_bucket = None
        self._selenium_driver = None

    @property
    def name(self) -> str:
        return "firebase_scrape"

    @property
    def description(self) -> str:
        return "Scrape web content using Firebase with advanced rendering and Firestore storage"

    @property
    def keywords(self) -> List[str]:
        return [
            "scrape",
            "web",
            "firebase",
            "crawl",
            "extract",
            "content",
            "selenium",
            "firestore",
        ]

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "url": {
                "type": str,
                "description": "URL to scrape",
                "required": True,
            },
            "use_selenium": {
                "type": bool,
                "description": "Use Selenium for JavaScript rendering",
                "required": False,
                "default": True,
            },
            "store_in_firestore": {
                "type": bool,
                "description": "Store results in Firestore",
                "required": False,
                "default": True,
            },
            "extract_images": {
                "type": bool,
                "description": "Extract and store images",
                "required": False,
                "default": False,
            },
            "extract_links": {
                "type": bool,
                "description": "Extract all links from page",
                "required": False,
                "default": True,
            },
            "timeout": {
                "type": int,
                "description": "Timeout in seconds",
                "required": False,
                "default": 30,
            },
            "collection_name": {
                "type": str,
                "description": "Firestore collection name",
                "required": False,
                "default": "scraped_data",
            },
        }

    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        from ..config import settings

        if self._firebase_initialized:
            return

        if not settings.firebase_settings.enabled:
            raise ToolExecutionError("Firebase is not enabled in settings")

        # Check if Firebase credentials are available
        if not all([
            settings.firebase_settings.project_id,
            settings.firebase_settings.private_key,
            settings.firebase_settings.client_email,
        ]):
            raise ToolExecutionError("Firebase credentials are not configured")

        try:
            # Create credentials from settings
            cred_dict = {
                "type": "service_account",
                "project_id": settings.firebase_settings.project_id,
                "private_key_id": settings.firebase_settings.private_key_id,
                "private_key": settings.firebase_settings.private_key.replace('\\n', '\n'),
                "client_email": settings.firebase_settings.client_email,
                "client_id": settings.firebase_settings.client_id,
                "auth_uri": settings.firebase_settings.auth_uri,
                "token_uri": settings.firebase_settings.token_uri,
                "auth_provider_x509_cert_url": settings.firebase_settings.auth_provider_x509_cert_url,
                "client_x509_cert_url": settings.firebase_settings.client_x509_cert_url,
            }

            cred = credentials.Certificate(cred_dict)

            # Initialize Firebase app
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {
                    'databaseURL': settings.firebase_settings.database_url,
                    'storageBucket': settings.firebase_settings.storage_bucket,
                })

            # Initialize Firestore and Storage
            self._db = firestore.client()
            if settings.firebase_settings.storage_bucket:
                self._storage_bucket = storage.bucket()

            self._firebase_initialized = True
            logger.info("Firebase initialized successfully")

        except Exception as e:
            raise ToolExecutionError(f"Failed to initialize Firebase: {str(e)}")

    def _initialize_selenium(self):
        """Initialize Selenium WebDriver"""
        from ..config import settings

        if self._selenium_driver:
            return

        try:
            options = Options()
            if settings.firebase_settings.headless_browser:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")

            if settings.firebase_settings.selenium_driver_type == "chrome":
                self._selenium_driver = webdriver.Chrome(options=options)
            elif settings.firebase_settings.selenium_driver_type == "firefox":
                self._selenium_driver = webdriver.Firefox(options=options)
            else:
                raise ToolExecutionError(f"Unsupported driver type: {settings.firebase_settings.selenium_driver_type}")

            self._selenium_driver.implicitly_wait(10)
            logger.info("Selenium WebDriver initialized")

        except Exception as e:
            raise ToolExecutionError(f"Failed to initialize Selenium: {str(e)}")

    async def _scrape_with_selenium(self, url: str, timeout: int) -> Dict[str, Any]:
        """Scrape URL using Selenium for JavaScript rendering"""
        self._initialize_selenium()

        try:
            self._selenium_driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self._selenium_driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Get page source and process
            page_source = self._selenium_driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')

            return self._extract_content(soup, url)

        except Exception as e:
            raise ToolExecutionError(f"Selenium scraping failed: {str(e)}")

    async def _scrape_with_httpx(self, url: str, timeout: int) -> Dict[str, Any]:
        """Scrape URL using HTTPX for simple content extraction"""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                return self._extract_content(soup, url)

        except Exception as e:
            raise ToolExecutionError(f"HTTP scraping failed: {str(e)}")

    def _extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract structured content from BeautifulSoup object"""
        from ..config import settings

        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""

        # Extract main content
        main_content = ""
        content_selectors = ['article', 'main', '.content', '#content', '.main-content']
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element.get_text(strip=True)
                if len(main_content) > 100:  # Ensure meaningful content
                    break

        # Fallback to body if no specific content found
        if not main_content:
            body = soup.find('body')
            main_content = body.get_text(strip=True) if body else ""

        # Extract metadata
        meta_description = ""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            meta_description = meta_desc['content']

        # Extract links if enabled
        links = []
        if settings.firebase_settings.extract_links:
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('http') or href.startswith('/'):
                    links.append({
                        'text': link.get_text(strip=True),
                        'url': href,
                        'title': link.get('title', '')
                    })

        # Extract images if enabled
        images = []
        if settings.firebase_settings.extract_images:
            for img in soup.find_all('img', src=True):
                images.append({
                    'src': img['src'],
                    'alt': img.get('alt', ''),
                    'title': img.get('title', '')
                })

        return {
            'url': url,
            'title': title_text,
            'content': main_content,
            'description': meta_description,
            'links': links,
            'images': images,
            'content_length': len(main_content),
            'link_count': len(links),
            'image_count': len(images),
        }

    async def _store_in_firestore(self, data: Dict[str, Any], collection_name: str) -> str:
        """Store scraped data in Firestore"""
        self._initialize_firebase()

        try:
            # Create document ID from URL
            doc_id = data['url'].replace('/', '_').replace(':', '_').replace('.', '_')
            
            # Add timestamp
            data['scraped_at'] = firestore.SERVER_TIMESTAMP
            
            # Store in Firestore
            doc_ref = self._db.collection(collection_name).document(doc_id)
            doc_ref.set(data)
            
            logger.info(f"Stored scraped data in Firestore: {doc_id}")
            return doc_id

        except FirebaseError as e:
            raise ToolExecutionError(f"Firestore storage failed: {str(e)}")

    async def execute(
        self,
        url: str,
        use_selenium: bool = True,
        store_in_firestore: bool = True,
        extract_images: bool = False,
        extract_links: bool = True,
        timeout: int = 30,
        collection_name: str = "scraped_data",
    ) -> Dict[str, Any]:
        """Execute web scraping using Firebase infrastructure"""
        from ..config import settings

        if not settings.firebase_settings.scraping_enabled:
            raise ToolExecutionError("Firebase scraping is not enabled")

        # Validate URL
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"

        try:
            # Scrape content
            if use_selenium and settings.firebase_settings.use_selenium:
                scraped_data = await self._scrape_with_selenium(url, timeout)
            else:
                scraped_data = await self._scrape_with_httpx(url, timeout)

            # Store in Firestore if enabled
            if store_in_firestore and settings.firebase_settings.enabled:
                doc_id = await self._store_in_firestore(scraped_data, collection_name)
                scraped_data['firestore_doc_id'] = doc_id

            return scraped_data

        except Exception as e:
            logger.error(f"Firebase scraping failed for {url}: {str(e)}")
            raise ToolExecutionError(f"Scraping failed: {str(e)}")

    async def batch_scrape(
        self,
        urls: List[str],
        use_selenium: bool = True,
        store_in_firestore: bool = True,
        timeout: int = 30,
    ) -> List[Dict[str, Any]]:
        """Scrape multiple URLs concurrently"""
        tasks = [
            self.execute(
                url=url,
                use_selenium=use_selenium,
                store_in_firestore=store_in_firestore,
                timeout=timeout,
            )
            for url in urls
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)

    def cleanup(self):
        """Clean up resources"""
        if self._selenium_driver:
            self._selenium_driver.quit()
            self._selenium_driver = None
            logger.info("Selenium WebDriver cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()