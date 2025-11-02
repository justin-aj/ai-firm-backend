from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Dict, List


class Settings(BaseSettings):
    """Application settings"""
    debug: bool = False  # Default to False for production safety
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # LM Studio settings
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_model: str = "local-model"
    
    # Google Custom Search API settings
    google_api_key: str = ""
    google_cx: str = ""
    
    # Dask distributed scraping settings
    use_dask: bool = False  # Disable by default (AsyncIO works great, Dask can have pickling issues)
    dask_scheduler: str = ""  # Dask scheduler address (empty = local cluster)
    dask_workers: int = 4  # Number of Dask workers (for local cluster)
    
    # Rate limiting (future use)
    rate_limit_per_minute: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def validate_credentials(self) -> Dict[str, bool | List[str]]:
        """Validate that required credentials are set"""
        issues: List[str] = []
        if not self.google_api_key:
            issues.append("GOOGLE_API_KEY not set")
        if not self.google_cx:
            issues.append("GOOGLE_CX not set")
        return {"valid": len(issues) == 0, "issues": issues}


@lru_cache()
def get_settings():
    """Get cached settings instance"""
    return Settings()
