from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field

from typing import Optional
from datetime import datetime, timedelta
from functools import lru_cache

import os, sys, traceback
from loguru import logger 
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger.remove()
def set_logging():
    logger.add(sys.stdout, colorize=True, format="[{time} - {file} {level}] {message}" ,level="DEBUG")
    logger.add("logs/pvvector.log", retention="1 days")
    return logger


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class OpenAISettings(LLMSettings):
    """OpenAI-specific settings extending LLMSettings."""

    credential: str = Field(default_factory=lambda: os.getenv("AZURE_API_KEY"))
    api_version: str = Field(default_factory=lambda: os.getenv("AZURE_API_VERSION"))
    endpoint: str = Field(default_factory=lambda: os.getenv("AZURE_ENDPOINT"))
    default_model: str = Field(default="gpt-4.1")
    embedding_model: str = Field(default="text-embedding-3-large")


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = "embeddings"
    embedding_dimensions: int = 1536
    time_partition_interval: timedelta = timedelta(days=7)


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    set_logging()
    return settings
