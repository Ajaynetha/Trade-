from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Header
import 'google'.generativeai as genai
from duckduckgo_search import DDGS
import os
import time
import hashlib
import logging
from datetime import datetime
from typing import Optional
import secrets
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trade Opportunities API",
    description="API for market analysis and trade opportunities in Indian sectors",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
rate_limits = {}


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')

class SecurityManager:
    def __init__(self):
        self.api_keys = {
            "admin": os.getenv("API_SECRET_KEY", "trade_secret_2024"),
            "guest": "guest123"
        }
        self.max_requests = int(os.getenv("RATE_LIMIT", "10"))
    
    def authenticate(self, api_key: str):
        if api_key not in self.api_keys.values():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        return True
    
    def rate_limit_check(self, client_id: str):
        current_time = time.time()
        if client_id not in rate_limits:
            rate_limits[client_id] = []
        
        
        rate_limits[client_id] = [
            req_time for req_time in rate_limits[client_id] 
            if current_time - req_time < 60
        ]
        
        if len(rate_limits[client_id]) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Maximum 10 requests per minute."
            )
        
        rate_limits[client_id].append(current_time)
        return True
    
    def get_client_id(self, request: Request) -> str:
        client_ip = request.client.host or "unknown"
        user_agent = request.headers.get("user-agent", "")
        return hashlib.md5(f"{client_ip}-{user_agent}".encode()).hexdigest()

security_manager = SecurityManager()

class DataCollector:
    def __init__(self):
        self.ddgs = DDGS()
    
    async def search_market_data(self, sector: str) -> dict:
        """Collect current market data and news"""
        try:
            logger.info(f"Collecting market data for sector: {sector}")
            
            # Search for recent news
            news_results = self.ddgs.news(
                keywords=f"{sector} sector market news India 2024",
                region="in",
                max_results=8
            )
            
            
            financial_results = self.ddgs.text(
                keywords=f"{sector} industry analysis trends India",
                max_results=5
            )
            
            return {
                "sector": sector,
                "news": news_results[:5],  # Limit to 5 news items
                "financial_data": financial_results[:3],
                "collected_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Data collection error: {str(e)}")
            return {"error": f"Data collection failed: {str(e)}"}

class AIAnalyzer:
    @staticmethod
    async def generate_markdown_report(sector: str, market_data: dict) -> str:
        """Generate structured markdown report using Gemini AI"""
        try:
            if not GEMINI_API_KEY:
                return await AIAnalyzer._generate_fallback_report(sector, market_data)
            
            prompt = f"""
            Create a comprehensive market analysis report for the {sector} sector in India.
            
            MARKET DATA: {market_data}
            
            Please generate a structured markdown report with the following sections:
            
            # {sector.title()} Sector Market Analysis - India
            
            
            """
            
            response = gemini_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"AI analysis error: {str(e)}")
            return await AIAnalyzer._generate_fallback_report(sector, market_data)
    
    @staticmethod
    async def _generate_fallback_report(sector: str, market_data: dict) -> str:
        """Fallback report when Gemini is not available"""
        return f"""
# {sector.title()} Sector Market Analysis - India



*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*Data source: Market analysis and news aggregation*
"""


async def verify_access(request: Request, api_key: str = Header(..., alias="X-API-Key")):
    """Verify API key and apply rate limiting"""
    client_id = security_manager.get_client_id(request)
    security_manager.authenticate(api_key)
    security_manager.rate_limit_check(client_id)
    return {"client_id": client_id, "api_key": api_key}

@app.get("/", summary="API Root", response_description="Welcome message")
async def root():
    """API root endpoint with basic information"""
    return {
        "message": "Trade Opportunities API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "analyze": "/analyze/{sector}",
            "docs": "/docs"
        }
    }

@app.get(
    "/analyze/{sector}",
    summary="Analyze Sector",
    response_description="Markdown market analysis report",
    response_class=PlainTextResponse
)
async def analyze_sector(
    sector: str,
    request: Request,
    auth: dict = Depends(verify_access)
):
    """
    Analyze a specific sector and generate markdown market analysis report
    
    - **sector**: Name of the sector to analyze (e.g., pharmaceuticals, technology, agriculture)
    - **X-API-Key**: API key for authentication (use 'guest123' for testing)
    
    Returns a structured markdown report that can be saved as .md file
    """
    try:
        logger.info(f"Analysis request for sector: {sector} from client: {auth['client_id']}")
        
        # Input validation
        if not sector or len(sector.strip()) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sector name must be at least 2 characters long"
            )
        
        sector = sector.strip().lower()
        
        
        data_collector = DataCollector()
        market_data = await data_collector.search_market_data(sector)
        
        
        ai_analyzer = AIAnalyzer()
        markdown_report = await ai_analyzer.generate_markdown_report(sector, market_data)
        
        
        final_report = f"""# {sector.title()} Sector Analysis Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Sector**: {sector}  
**API Version**: 1.0.0  

---

{markdown_report}

---
*This report was generated automatically by Trade Opportunities API*
"""
        
        logger.info(f"Successfully generated report for sector: {sector}")
        return final_report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for sector {sector}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/health", summary="Health Check")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini_available": bool(GEMINI_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")