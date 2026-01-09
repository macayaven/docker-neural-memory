"""
Configuration for Docker Neural Memory.

Uses Pydantic Settings for environment variable management.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class MemoryConfig(BaseSettings):
    """Configuration for NeuralMemory module."""

    dim: int = Field(
        default=512,
        description="Embedding dimension for memory",
        ge=64,
        le=4096,
    )
    memory_depth: int = Field(
        default=2,
        description="Number of layers in memory network",
        ge=1,
        le=8,
    )
    learning_rate: float = Field(
        default=0.01,
        description="Learning rate for test-time training",
        gt=0,
        le=1.0,
    )
    momentum: float = Field(
        default=0.9,
        description="Momentum for weight updates",
        ge=0,
        le=1.0,
    )
    device: str = Field(
        default="cpu",
        description="Device to run on (cpu, cuda, mps)",
    )

    class Config:
        env_prefix = "MEMORY_"


class TTTConfig(BaseSettings):
    """Configuration for TTT layer."""

    variant: str = Field(
        default="mlp",
        description="TTT variant: 'linear' or 'mlp'",
    )
    hidden_dim: Optional[int] = Field(
        default=None,
        description="Hidden dimension (default: dim * 2)",
    )
    num_steps: int = Field(
        default=1,
        description="Number of gradient steps per token",
        ge=1,
        le=10,
    )

    class Config:
        env_prefix = "TTT_"


class MCPConfig(BaseSettings):
    """Configuration for MCP server."""

    host: str = Field(
        default="0.0.0.0",
        description="Host to bind to",
    )
    port: int = Field(
        default=8765,
        description="Port to listen on",
        ge=1,
        le=65535,
    )
    mode: str = Field(
        default="stdio",
        description="MCP mode: 'stdio' or 'http'",
    )

    class Config:
        env_prefix = "MCP_"


class AppConfig(BaseSettings):
    """Main application configuration."""

    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    ttt: TTTConfig = Field(default_factory=TTTConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    class Config:
        env_prefix = "APP_"


def get_config() -> AppConfig:
    """Get application configuration from environment."""
    return AppConfig()
