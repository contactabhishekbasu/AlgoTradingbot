"""Configuration management for the trading system."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # Database Configuration
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="trading", description="PostgreSQL database name")
    postgres_user: str = Field(default="trading_user", description="PostgreSQL user")
    postgres_password: str = Field(
        default="changeme", description="PostgreSQL password"
    )
    database_url: Optional[str] = Field(
        default=None, description="Complete database URL"
    )

    # Redis Configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_url: Optional[str] = Field(default=None, description="Complete Redis URL")

    # API Keys
    alpaca_api_key: Optional[str] = Field(default=None, description="Alpaca API key")
    alpaca_secret_key: Optional[str] = Field(
        default=None, description="Alpaca secret key"
    )
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets", description="Alpaca API base URL"
    )

    # Application Settings
    environment: str = Field(default="development", description="Environment name")
    log_level: str = Field(default="INFO", description="Logging level")
    log_path: str = Field(default="./logs", description="Log file directory")
    debug: bool = Field(default=False, description="Debug mode")

    # Model Configuration
    model_path: str = Field(default="./models", description="Model storage path")
    model_device: str = Field(
        default="cpu", description="Device for model inference (cpu/mps/cuda)"
    )

    # Performance Settings
    cache_ttl: int = Field(default=300, description="Cache TTL in seconds")
    max_workers: int = Field(default=4, description="Max worker threads")
    batch_size: int = Field(default=32, description="Batch size for processing")
    rate_limit_requests: int = Field(
        default=2000, description="Rate limit requests per period"
    )
    rate_limit_period: int = Field(
        default=3600, description="Rate limit period in seconds"
    )

    # Trading Settings
    paper_trading: bool = Field(default=True, description="Use paper trading")
    initial_capital: float = Field(default=100000.0, description="Initial capital")
    max_position_size: float = Field(
        default=10000.0, description="Maximum position size"
    )
    max_drawdown: float = Field(default=0.15, description="Maximum drawdown threshold")
    stop_loss: float = Field(default=0.05, description="Stop loss percentage")

    @field_validator("database_url", mode="before")
    @classmethod
    def assemble_db_url(cls, v: Optional[str], info) -> str:
        """Construct database URL if not provided."""
        if v is not None:
            return v
        values = info.data
        return (
            f"postgresql://{values.get('postgres_user')}:"
            f"{values.get('postgres_password')}@"
            f"{values.get('postgres_host')}:"
            f"{values.get('postgres_port')}/"
            f"{values.get('postgres_db')}"
        )

    @field_validator("redis_url", mode="before")
    @classmethod
    def assemble_redis_url(cls, v: Optional[str], info) -> str:
        """Construct Redis URL if not provided."""
        if v is not None:
            return v
        values = info.data
        return (
            f"redis://{values.get('redis_host')}:"
            f"{values.get('redis_port')}/"
            f"{values.get('redis_db')}"
        )

    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.log_path,
            self.model_path,
            "./data/raw",
            "./data/processed",
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Create directories on import
settings.create_directories()
