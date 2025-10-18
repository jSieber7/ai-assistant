"""
Jina Reranker Service
A proxy service for Jina AI reranker API with caching and monitoring
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import redis
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus Metrics
REQUEST_COUNT = Counter('jina_reranker_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('jina_reranker_request_duration_seconds', 'Request duration')
RERANK_REQUESTS = Counter('jina_reranker_api_requests_total', 'Jina API rerank requests', ['status'])
CACHE_HITS = Counter('jina_reranker_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('jina_reranker_cache_misses_total', 'Cache misses')


class Settings(BaseSettings):
    """Application settings"""
    jina_api_key: str = Field(..., env="JINA_API_KEY")
    redis_url: str = Field("redis://redis:6379/1", env="REDIS_URL")
    config_path: str = Field("/app/config.yml", env="CONFIG_PATH")


class RerankRequest(BaseModel):
    """Rerank request model"""
    model: str = Field("jina-reranker-v2-base-multilingual", description="Model name")
    query: str = Field(..., description="Query to rank against")
    documents: List[str] = Field(..., description="Documents to rerank")
    top_n: Optional[int] = Field(None, description="Number of top results to return")


class RerankResponse(BaseModel):
    """Rerank response model"""
    results: List[Dict[str, Any]]
    model: str
    query: str
    total_documents: int
    cached: bool = False


class JinaRerankerService:
    """Main service class for Jina reranker"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.config = self._load_config()
        self.redis_client = None
        self.http_client = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.settings.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Failed to load config", error=str(e))
            return {}
    
    async def initialize(self):
        """Initialize service components"""
        # Initialize Redis client
        if self.config.get('cache', {}).get('enabled', True):
            try:
                import redis.asyncio as redis_async
                self.redis_client = redis_async.from_url(self.settings.redis_url)
                await self.redis_client.ping()
                logger.info("Redis client initialized", url=self.settings.redis_url)
            except Exception as e:
                logger.error("Failed to initialize Redis", error=str(e))
                self.redis_client = None
        
        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=self.config.get('jina_api', {}).get('timeout', 30),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        logger.info("Jina Reranker service initialized")
    
    async def shutdown(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.aclose()
        if self.redis_client:
            await self.redis_client.aclose()
    
    def _generate_cache_key(self, request: RerankRequest) -> str:
        """Generate cache key for request"""
        content = f"{request.model}:{request.query}:{request.top_n}:{'|'.join(request.documents)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache"""
        if not self.redis_client:
            return None
        
        try:
            cached = await self.redis_client.get(cache_key)
            if cached:
                CACHE_HITS.inc()
                return json.loads(cached)
        except Exception as e:
            logger.warning("Cache get failed", key=cache_key, error=str(e))
        
        CACHE_MISSES.inc()
        return None
    
    async def _store_in_cache(self, cache_key: str, result: Dict[str, Any]):
        """Store result in cache"""
        if not self.redis_client:
            return
        
        try:
            ttl = self.config.get('cache', {}).get('ttl', 3600)
            await self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(result)
            )
        except Exception as e:
            logger.warning("Cache store failed", key=cache_key, error=str(e))
    
    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """Rerank documents using Jina API"""
        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_result = await self._get_from_cache(cache_key)
        
        if cached_result:
            logger.info("Cache hit for rerank request", query=request.query[:50])
            return RerankResponse(**cached_result, cached=True)
        
        # Call Jina API
        try:
            api_url = self.config.get('jina_api', {}).get('base_url', 'https://api.jina.ai/v1/rerank')
            headers = {
                'Authorization': f'Bearer {self.settings.jina_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': request.model,
                'query': request.query,
                'documents': request.documents
            }
            
            if request.top_n:
                payload['top_n'] = request.top_n
            
            max_retries = self.config.get('jina_api', {}).get('max_retries', 3)
            retry_delay = self.config.get('jina_api', {}).get('retry_delay', 1)
            
            for attempt in range(max_retries):
                try:
                    response = await self.http_client.post(
                        api_url,
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Create response
                        rerank_response = RerankResponse(
                            results=result,
                            model=request.model,
                            query=request.query,
                            total_documents=len(request.documents),
                            cached=False
                        )
                        
                        # Store in cache
                        await self._store_in_cache(cache_key, rerank_response.model_dump())
                        
                        RERANK_REQUESTS.labels(status='success').inc()
                        logger.info("Rerank completed", 
                                  query=request.query[:50], 
                                  documents_count=len(request.documents),
                                  cached=False)
                        
                        return rerank_response
                    
                    else:
                        error_msg = f"Jina API error: {response.status_code} - {response.text}"
                        logger.error("Jina API request failed", 
                                   status=response.status_code, 
                                   error=response.text)
                        
                        if attempt == max_retries - 1:
                            RERANK_REQUESTS.labels(status='error').inc()
                            raise HTTPException(status_code=502, detail=error_msg)
                        
                except httpx.RequestError as e:
                    logger.warning("Jina API request failed, retrying", 
                                 attempt=attempt + 1, 
                                 error=str(e))
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
        
        except Exception as e:
            RERANK_REQUESTS.labels(status='error').inc()
            logger.error("Rerank failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


# Initialize FastAPI app
app = FastAPI(
    title="Jina Reranker Service",
    description="A proxy service for Jina AI reranker API with caching and monitoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
service: Optional[JinaRerankerService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global service
    settings = Settings()
    service = JinaRerankerService(settings)
    await service.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global service
    if service:
        await service.shutdown()


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware for collecting metrics"""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "jina-reranker"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if not service.config.get('monitoring', {}).get('enabled', True):
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/rerank", response_model=RerankResponse)
async def rerank_endpoint(request: RerankRequest):
    """Main rerank endpoint"""
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await service.rerank(request)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Jina Reranker Service",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Load configuration
    settings = Settings()
    
    # Initialize service
    service = JinaRerankerService(settings)
    
    # Run server
    uvicorn.run(
        app,
        host=service.config.get('service', {}).get('host', '0.0.0.0'),
        port=service.config.get('service', {}).get('port', 8080),
        log_level=service.config.get('logging', {}).get('level', 'info').lower()
    )