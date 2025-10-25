"""
ScraperAgent - A specialized agent for web scraping with Firebase integration using LangChain.

This agent can scrape web content, extract structured data, and sync results
to Firebase for real-time collaboration and persistence.
"""

import logging
import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Any, AsyncGenerator
from urllib.parse import urljoin, urlparse
from datetime import datetime

from pydantic import BaseModel, Field
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.schema.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

try:
    import firebase_admin
    from firebase_admin import firestore, storage
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    firestore = None
    storage = None

try:
    from bs4 import BeautifulSoup
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    BeautifulSoup = None
    requests = None

from ...config import settings
from ...llm_providers import get_llm

logger = logging.getLogger(__name__)


class ScrapingState(BaseModel):
    """State for web scraping workflow"""
    urls: List[str] = Field(description="URLs to scrape")
    scraping_config: Dict[str, Any] = Field(default_factory=dict, description="Scraping configuration")
    extracted_data: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted data")
    current_url: Optional[str] = Field(default=None, description="Currently processing URL")
    messages: List[BaseMessage] = Field(default_factory=list, description="Conversation messages")
    firebase_sync_enabled: bool = Field(default=False, description="Whether to sync to Firebase")
    firebase_collection: str = Field(default="scraped_data", description="Firebase collection name")
    error: Optional[str] = Field(default=None, description="Error message if any")
    success: bool = Field(default=False, description="Whether scraping was successful")
    total_processed: int = Field(default=0, description="Total URLs processed")
    total_failed: int = Field(default=0, description="Total URLs failed")


class ScraperAgent:
    """
    A specialized agent for web scraping with Firebase integration using LangChain.
    
    This agent can scrape web content, extract structured data, and sync
    results to Firebase for real-time collaboration.
    """
    
    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ScraperAgent.
        
        Args:
            llm: LangChain LLM instance (if None, will get from settings)
            config: Additional configuration for agent
        """
        self.llm = llm
        self.config = config or {}
        self.name = "scraper_agent"
        self.description = "Specialized agent for web scraping with Firebase integration"
        
        # Session for HTTP requests
        self.session = None
        
        # Firebase clients
        self.firestore_client = None
        self.storage_client = None
        
        # LangGraph workflow
        self.workflow = None
        self.checkpointer = MemorySaver()
        
        # Initialize components
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize components asynchronously"""
        try:
            if not self.llm:
                self.llm = await get_llm()
            
            # Initialize HTTP session
            self._initialize_session()
            
            # Initialize Firebase if enabled
            if settings.firebase_settings.enabled:
                self._initialize_firebase()
            
            # Create LangGraph workflow
            self.workflow = self._create_workflow()
            
            logger.info("ScraperAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ScraperAgent: {str(e)}")
    
    def _initialize_session(self):
        """Initialize HTTP session with retry strategy"""
        if not SCRAPING_AVAILABLE:
            logger.warning("Requests/BeautifulSoup not available - scraping will be limited")
            return
        
        try:
            self.session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            
            # Set headers
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            logger.info("HTTP session initialized with retry strategy")
            
        except Exception as e:
            logger.error(f"Failed to initialize HTTP session: {str(e)}")
    
    def _initialize_firebase(self):
        """Initialize Firebase clients"""
        if not FIREBASE_AVAILABLE:
            logger.warning("Firebase not available - sync will be disabled")
            return
        
        try:
            # Get Firebase app from settings initialization
            from ...config import initialize_firebase_integration
            firebase_components = initialize_firebase_integration()
            
            if firebase_components and "firestore" in firebase_components:
                self.firestore_client = firebase_components["firestore"]
                logger.info("Firebase Firestore client initialized")
            
            if firebase_components and "storage" in firebase_components:
                self.storage_client = firebase_components["storage"]
                logger.info("Firebase Storage client initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {str(e)}")
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for web scraping"""
        workflow = StateGraph(ScrapingState)
        
        # Define nodes
        def validate_urls(state: ScrapingState) -> ScrapingState:
            """Validate and normalize URLs"""
            valid_urls = []
            
            for url in state.urls:
                # Normalize URL
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                
                # Basic validation
                parsed = urlparse(url)
                if parsed.netloc and parsed.scheme:
                    valid_urls.append(url)
                else:
                    state.messages.append(SystemMessage(content=f"Invalid URL: {url}"))
                    state.total_failed += 1
            
            state.urls = valid_urls
            state.messages.append(SystemMessage(content=f"Validated {len(valid_urls)} URLs"))
            return state
        
        def scrape_url(state: ScrapingState) -> ScrapingState:
            """Scrape a single URL"""
            if not state.urls or not SCRAPING_AVAILABLE:
                state.error = "No URLs to scrape or scraping not available"
                state.success = False
                return state
            
            # Get next URL
            url = state.urls[0]
            state.current_url = url
            state.urls = state.urls[1:]  # Remove processed URL
            
            try:
                # Make request
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract basic data
                extracted_data = {
                    'url': url,
                    'title': soup.title.string if soup.title else '',
                    'content': soup.get_text(strip=True),
                    'meta_description': '',
                    'meta_keywords': '',
                    'links': [a.get('href') for a in soup.find_all('a', href=True)],
                    'images': [img.get('src') for img in soup.find_all('img', src=True)],
                    'scraped_at': datetime.utcnow().isoformat(),
                }
                
                # Extract meta tags
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc:
                    extracted_data['meta_description'] = meta_desc.get('content', '')
                
                meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
                if meta_keywords:
                    extracted_data['meta_keywords'] = meta_keywords.get('content', '')
                
                # Generate content hash
                content_hash = hashlib.md5(
                    f"{url}{extracted_data['content']}".encode()
                ).hexdigest()
                extracted_data['content_hash'] = content_hash
                
                state.extracted_data.append(extracted_data)
                state.total_processed += 1
                state.messages.append(SystemMessage(content=f"Successfully scraped: {url}"))
                
            except Exception as e:
                state.error = f"Error scraping {url}: {str(e)}"
                state.total_failed += 1
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def process_with_llm(state: ScrapingState) -> ScrapingState:
            """Process extracted data with LLM for better structure"""
            if not state.extracted_data or not self.llm:
                return state
            
            try:
                # Process each extracted item
                for i, item in enumerate(state.extracted_data):
                    # Create prompt for LLM processing
                    prompt = f"""
                    Analyze and structure the following web content:
                    
                    URL: {item['url']}
                    Title: {item['title']}
                    Content: {item['content'][:2000]}...
                    
                    Please extract and structure the following information:
                    1. Main topic/category
                    2. Key entities (people, organizations, places)
                    3. Summary (2-3 sentences)
                    4. Important dates or numbers
                    5. Sentiment (positive, negative, neutral)
                    
                    Respond with JSON format.
                    """
                    
                    messages = [HumanMessage(content=prompt)]
                    response = self.llm.invoke(messages)
                    
                    try:
                        # Try to parse LLM response as JSON
                        llm_analysis = json.loads(response.content)
                        item['llm_analysis'] = llm_analysis
                    except json.JSONDecodeError:
                        item['llm_analysis'] = {'raw_response': response.content}
                
                state.messages.append(SystemMessage(content="LLM processing completed"))
                
            except Exception as e:
                state.messages.append(SystemMessage(content=f"LLM processing failed: {str(e)}"))
            
            return state
        
        def sync_to_firebase(state: ScrapingState) -> ScrapingState:
            """Sync extracted data to Firebase"""
            if not state.extracted_data or not self.firestore_client:
                state.messages.append(SystemMessage(content="Firebase sync skipped - no data or client"))
                return state
            
            try:
                collection = self.firestore_client.collection(state.firebase_collection)
                
                # Batch write for efficiency
                batch = self.firestore_client.batch()
                
                for item in state.extracted_data:
                    doc_ref = collection.document(item['content_hash'])
                    batch.set(doc_ref, item)
                
                # Commit batch
                batch.commit()
                
                state.firebase_sync_enabled = True
                state.messages.append(SystemMessage(content=f"Synced {len(state.extracted_data)} items to Firebase"))
                
            except Exception as e:
                state.error = f"Firebase sync failed: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def finalize(state: ScrapingState) -> ScrapingState:
            """Finalize scraping process"""
            state.success = state.total_processed > 0
            
            if state.success:
                state.messages.append(SystemMessage(content=f"Scraping completed: {state.total_processed} successful, {state.total_failed} failed"))
            else:
                state.messages.append(SystemMessage(content="Scraping completed with no successful extractions"))
            
            return state
        
        # Add nodes to workflow
        workflow.add_node("validate_urls", validate_urls)
        workflow.add_node("scrape_url", scrape_url)
        workflow.add_node("process_with_llm", process_with_llm)
        workflow.add_node("sync_to_firebase", sync_to_firebase)
        workflow.add_node("finalize", finalize)
        
        # Define conditional routing
        def route_scraping(state: ScrapingState) -> str:
            """Route based on remaining URLs"""
            if state.urls:
                return "scrape_url"
            else:
                return "process_with_llm"
        
        # Set up workflow
        workflow.set_entry_point("validate_urls")
        workflow.add_conditional_edges(
            "validate_urls",
            lambda state: "scrape_url" if state.urls else "process_with_llm",
            {
                "scrape_url": "scrape_url",
                "process_with_llm": "process_with_llm",
            }
        )
        workflow.add_conditional_edges(
            "scrape_url",
            route_scraping,
            {
                "scrape_url": "scrape_url",
                "process_with_llm": "process_with_llm",
            }
        )
        workflow.add_edge("process_with_llm", "sync_to_firebase")
        workflow.add_edge("sync_to_firebase", "finalize")
        workflow.add_edge("finalize", END)
        
        # Compile workflow with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def scrape_urls(
        self,
        urls: List[str],
        config: Optional[Dict[str, Any]] = None,
        sync_to_firebase: bool = None,
        firebase_collection: str = "scraped_data",
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scrape multiple URLs and optionally sync to Firebase.
        
        Args:
            urls: List of URLs to scrape
            config: Scraping configuration options
            sync_to_firebase: Whether to sync results to Firebase
            firebase_collection: Firebase collection name
            thread_id: Thread ID for conversation tracking
            
        Returns:
            Dictionary containing scraping results and metadata
        """
        if not SCRAPING_AVAILABLE:
            return {
                "success": False,
                "error": "Web scraping dependencies not available - install requests and beautifulsoup4",
            }
        
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                raise RuntimeError("Failed to initialize ScraperAgent workflow")
        
        try:
            # Create initial state
            state = ScrapingState(
                urls=urls,
                scraping_config=config or {},
                firebase_sync_enabled=sync_to_firebase if sync_to_firebase is not None else settings.firebase_settings.enabled,
                firebase_collection=firebase_collection,
            )
            
            # Run workflow
            config_dict = {"thread_id": thread_id or "default"} if thread_id else {}
            result = await self.workflow.ainvoke(state, config=config_dict)
            
            return {
                "success": result.success,
                "extracted_data": result.extracted_data,
                "total_processed": result.total_processed,
                "total_failed": result.total_failed,
                "firebase_synced": result.firebase_sync_enabled,
                "error": result.error,
                "messages": [msg.content for msg in result.messages],
            }
            
        except Exception as e:
            logger.error(f"Error in web scraping: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "extracted_data": [],
            }
    
    async def stream_scrape(
        self,
        urls: List[str],
        config: Optional[Dict[str, Any]] = None,
        sync_to_firebase: bool = None,
        firebase_collection: str = "scraped_data",
        thread_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream scraping process for real-time updates.
        
        Args:
            urls: List of URLs to scrape
            config: Scraping configuration options
            sync_to_firebase: Whether to sync results to Firebase
            firebase_collection: Firebase collection name
            thread_id: Thread ID for conversation tracking
            
        Yields:
            Dictionary containing intermediate results and updates
        """
        if not SCRAPING_AVAILABLE:
            yield {
                "type": "error",
                "error": "Web scraping dependencies not available - install requests and beautifulsoup4",
                "success": False,
            }
            return
        
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                yield {
                    "type": "error",
                    "error": "Failed to initialize ScraperAgent workflow",
                    "success": False,
                }
                return
        
        try:
            # Create initial state
            state = ScrapingState(
                urls=urls,
                scraping_config=config or {},
                firebase_sync_enabled=sync_to_firebase if sync_to_firebase is not None else settings.firebase_settings.enabled,
                firebase_collection=firebase_collection,
            )
            
            # Stream workflow
            config_dict = {"thread_id": thread_id or "default"} if thread_id else {}
            
            async for event in self.workflow.astream(state, config=config_dict):
                # Yield URL processing updates
                if "scrape_url" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'current_url') and node_state.current_url:
                        yield {
                            "type": "url_processed",
                            "url": node_state.current_url,
                            "processed": node_state.total_processed,
                            "failed": node_state.total_failed,
                        }
                
                # Yield final result
                if "__end__" in event:
                    final_state = list(event.values())[0]
                    yield {
                        "type": "scraping_complete",
                        "success": final_state.success,
                        "total_processed": final_state.total_processed,
                        "total_failed": final_state.total_failed,
                        "firebase_synced": final_state.firebase_sync_enabled,
                        "extracted_data": final_state.extracted_data,
                    }
                    break
                    
        except Exception as e:
            logger.error(f"Error in streaming web scraping: {str(e)}")
            yield {
                "type": "error",
                "error": str(e),
                "success": False,
            }
    
    def __del__(self):
        """Cleanup when agent is destroyed"""
        if self.session:
            self.session.close()